#include <metal_stdlib>
using namespace metal;

// ============================================================================
// attention — split-KV tiled softmax attention for single-token decode.
//
// Decode path only: M_q = 1 (one query token), arbitrary seq_len history.
// The MPP prefill path handles M_q > 1 separately.
//
// Layout (caller-side contract):
//   Q   : [num_q_heads,  head_dim]                      FP16, contiguous.
//   K   : [seq_len, num_kv_heads, head_dim]             FP16, contiguous.
//   V   : [seq_len, num_kv_heads, head_dim]             FP16, same shape as K.
//         Full attention reuses the raw K projection for V, but its separate
//         normalization and RoPE paths make these buffers distinct here.
//   out : [num_q_heads,  head_dim]                      FP16.
//
// GQA: q_head -> kv_head = q_head / (num_q_heads / num_kv_heads).
//      Multiple Q heads share one KV head; the dispatch indexes Q heads.
//
// Online softmax recurrence (FP32 accumulators) — Milakov & Gimelshein 2018,
// also FlashAttention:
//   m_new   = max(m, s)
//   alpha   = exp(m - m_new)                 // rescale factor for past state
//   d       = d * alpha + exp(s - m_new)
//   o[i]    = o[i] * alpha + exp(s - m_new) * V[p, i]
//   m       = m_new
// Final normalization: out[i] = o[i] / d.
//
// ============================================================================

constant constexpr uint kAttnThreads      = 256;
// kAttnMaxSimdGroups must cover kAttnThreads / 32 = 8.
constant constexpr uint kAttnMaxSimdGroups = 8;
constant constexpr uint kAttnMaxQPerKV     = 2;
constant constexpr uint kAttnMaxFullQPerKV = 8;
constant constexpr uint kAttnFullQPerThreadgroup = 2;
// Largest head_dim we run with (full-attention layers). SWA uses 256 — the
// kernel still allocates the 512-slot scratch but only touches the live half.
constant constexpr uint kAttnMaxHeadDim   = 512;
constant uint FC_ATTN_HEAD_DIM [[function_constant(60)]];
constant uint FC_ATTN_NUM_Q_HEADS [[function_constant(61)]];
constant uint FC_ATTN_NUM_KV_HEADS [[function_constant(62)]];
constant bool FC_ATTN_USE_FC [[function_constant(63)]];
constant float FC_ATTN_SCALE [[function_constant(64)]];
constant uint FC_ATTN_NUM_CHUNKS [[function_constant(65)]];
constant uint FC_ATTN_RING_CAP [[function_constant(69)]];

static inline uint attn_fc_head_dim(constant uint& head_dim) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_HEAD_DIM))
        ? FC_ATTN_HEAD_DIM
        : head_dim;
}
static inline uint attn_fc_num_q_heads(constant uint& num_q_heads) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_NUM_Q_HEADS))
        ? FC_ATTN_NUM_Q_HEADS
        : num_q_heads;
}

static inline uint attn_fc_num_kv_heads(constant uint& num_kv_heads) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_NUM_KV_HEADS))
        ? FC_ATTN_NUM_KV_HEADS
        : num_kv_heads;
}

static inline float attn_fc_scale(float scale) {
    return is_function_constant_defined(FC_ATTN_SCALE) ? FC_ATTN_SCALE : scale;
}

static inline uint attn_fc_num_chunks(constant uint& num_chunks) {
    return is_function_constant_defined(FC_ATTN_NUM_CHUNKS) ? FC_ATTN_NUM_CHUNKS : num_chunks;
}

static inline uint attn_ring_slot(uint p) {
    return (is_function_constant_defined(FC_ATTN_RING_CAP) &&
            FC_ATTN_RING_CAP != 0u)
        ? (p % FC_ATTN_RING_CAP)
        : p;
}

static inline float attn_softmax_exp(float x) {
    return fast::exp(x);
}

// Block reduce: per-SIMD-group simd_sum, write partial to scratch, lane 0 of
// SIMD-group 0 finishes the merge with a second simd_sum and broadcasts.
// `scratch` must hold at least `simdgroups` floats; `bcast` is one float used
// to publish the final reduced value to all threads.
inline float block_reduce_sum(float v,
                              uint simd_lane_id,
                              uint simd_group_id,
                              uint simdgroups,
                              threadgroup float* scratch,
                              threadgroup float* bcast) {
    float s = simd_sum(v);
    if (simd_lane_id == 0) { scratch[simd_group_id] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float t = (simd_lane_id < simdgroups) ? scratch[simd_lane_id] : 0.0f;
        t = simd_sum(t);
        if (simd_lane_id == 0) { *bcast = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *bcast;
}


// ============================================================================
// Split-KV (Flash-Decoding) decode attention — the default path.
//

// Pass 1 (attention_decode_partial): grid = num_q_heads * num_chunks. Each TG
// KEEP RECURRENCE IN SYNC with attention_decode_single below (same math).
//   runs the same online-softmax recurrence over its chunk [p_start, p_end) and
//   writes the UN-normalized partial state (m_chunk, d_chunk, o_chunk[head_dim])
//   to scratch — no division yet.
// Pass 2 (attention_decode_combine): grid = num_q_heads. Each TG merges its
//   head's num_chunks partials with the standard online-softmax rescale
//   (m_glob = max_c m_c; D = Σ d_c·e^{m_c−m_glob}; O = Σ o_c·e^{m_c−m_glob}) and
//   writes out[i] = O[i] / D in FP16.
//
// At num_chunks == 1 the chunk spans the whole [kv_start, seq_len) range and
// the partial is the exact single-pass accumulation; the combine's only chunk
// has m_glob == m_chunk so e^0 == 1 and out == o/d — byte-identical to the
// single-pass kernels above. num_chunks > 1 changes the FP rounding of the
// partial sums only (same position summation order), not the algorithm.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_partial(
    device const half*  Q             [[buffer(0)]],
    device const half*  K             [[buffer(1)]],
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);

    const uint q_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    const uint kv_head = q_head / (NQ / NKV);

    device const half* Q_row = Q + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        q_smem[i] = float(Q_row[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kPerThread = (kAttnMaxHeadDim + kAttnThreads - 1) / kAttnThreads;
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    // p_start can land past the end when num_chunks > range length (the tail
    // chunks are empty); the loop simply does not execute and the partial is
    // (-inf, 0, 0), which the combine weights to zero via e^{-inf}.
    for (uint p = p_start; p < p_end; ++p) {
        const uint phys_p = attn_ring_slot(p);
        device const half* K_row = K + (phys_p * NKV + kv_head) * HD;
        device const half* V_row = V + (phys_p * NKV + kv_head) * HD;

        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            partial = fma(q_smem[i], float(K_row[i]), partial);
        }
        float s = block_reduce_sum(partial,
                                   simd_lane_id, simd_group_id, simdgroups,
                                   reduce_scratch, &bcast);
        s *= attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s     - m_new);
        d_run = d_run * alpha + p_exp;

        uint slot = 0;
        for (uint i = lid; i < HD; i += lsize) {
            o_local[slot] = o_local[slot] * alpha + p_exp * float(V_row[i]);
            slot += 1;
        }
        m_run = m_new;
    }

    const uint base = uint(q_head) * NC + chunk;
    if (lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = lid; i < HD; i += lsize) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

// ============================================================================
// Plan B (2026-08-08): single-pass decode attention — one Q head per TG, scans
// KEEP RECURRENCE IN SYNC with attention_decode_partial above (same math).
// the whole [kv_start, seq_len) K/V range with the online-softmax recurrence
// and writes `out` directly (no partial scratch, no combine pass). This is the
// num_chunks==1 collapse done as a first-class kernel so the grid is one TG per
// Q head (16 for the SWA shape) with all 256 threads active on one head, unlike
// the chunked path whose per-TG parallelism is split across Q heads. Env-gated
// TURBO_FIELDFARE_ATTN_SINGLE=1 (SWA 256/16/8 only, ring-free short context).
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_single(
    device const half*  Q             [[buffer(0)]],
    device const half*  K             [[buffer(1)]],
    device const half*  V             [[buffer(2)]],
    device       half*  out           [[buffer(3)]],
    constant     uint&  head_dim      [[buffer(4)]],
    constant     uint&  num_q_heads   [[buffer(5)]],
    constant     uint&  num_kv_heads  [[buffer(6)]],
    constant     uint&  seq_len       [[buffer(7)]],
    constant     uint&  kv_start      [[buffer(8)]],
    constant     float& scale         [[buffer(9)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);

    const uint q_head = tg_id;
    const uint kv_head = q_head / (NQ / NKV);

    device const half* Q_row = Q + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        q_smem[i] = float(Q_row[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kPerThread =
        (kAttnMaxHeadDim + kAttnThreads - 1) / kAttnThreads;
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (uint p = kv_start; p < seq_len; ++p) {
        const uint phys_p = attn_ring_slot(p);
        device const half* K_row = K + (phys_p * NKV + kv_head) * HD;
        device const half* V_row = V + (phys_p * NKV + kv_head) * HD;

        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            partial = fma(q_smem[i], float(K_row[i]), partial);
        }
        float s = block_reduce_sum(partial,
                                   simd_lane_id, simd_group_id, simdgroups,
                                   reduce_scratch, &bcast);
        s *= attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s     - m_new);
        d_run = d_run * alpha + p_exp;

        uint slot = 0;
        for (uint i = lid; i < HD; i += lsize) {
            o_local[slot] = o_local[slot] * alpha + p_exp * float(V_row[i]);
            slot += 1;
        }
        m_run = m_new;
    }

    device half* out_row = out + uint(q_head) * HD;
    uint slot = 0;
    for (uint i = lid; i < HD; i += lsize) {
        out_row[i] = d_run > 0.0f ? half(o_local[slot] / d_run) : half(0.0f);
        slot += 1;
    }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_gqa_swa_partial(
    device const half*  Q             [[buffer(0)]],
    device const half*  K             [[buffer(1)]],
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxQPerKV * kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxQPerKV * kAttnMaxSimdGroups];
    threadgroup float bcast[kAttnMaxQPerKV];
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);

    const uint q_per_kv = NQ / NKV;
    if (q_per_kv > kAttnMaxQPerKV) { return; }

    const uint kv_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    const uint q_base = kv_head * q_per_kv;
    for (uint qg = 0; qg < q_per_kv; ++qg) {
        device const half* Q_row = Q + (q_base + qg) * HD;
        threadgroup float* Q_s = q_smem + qg * kAttnMaxHeadDim;
        for (uint i = lid; i < HD; i += lsize) {
            Q_s[i] = float(Q_row[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint groups_per_q = max(1u, simdgroups / q_per_kv);
    const uint active_q = min(q_per_kv - 1u, simd_group_id / groups_per_q);
    const uint local_group = simd_group_id - active_q * groups_per_q;
    const uint threads_per_q = groups_per_q * 32u;
    const uint local_lid = local_group * 32u + simd_lane_id;

    constexpr uint kGQAPerThread =
        (kAttnMaxHeadDim + (kAttnThreads / kAttnMaxQPerKV) - 1) /
        (kAttnThreads / kAttnMaxQPerKV);
    float o_local[kGQAPerThread];
    for (uint k = 0; k < kGQAPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (uint p = p_start; p < p_end; ++p) {
        const uint phys_p = attn_ring_slot(p);
        device const half* K_row = K + (phys_p * NKV + kv_head) * HD;
        device const half* V_row = V + (phys_p * NKV + kv_head) * HD;

        float partial = 0.0f;
        for (uint i = local_lid; i < HD; i += threads_per_q) {
            const float k_val = float(K_row[i]);
            partial = fma(q_smem[active_q * kAttnMaxHeadDim + i], k_val, partial);
        }

        float s = simd_sum(partial);
        if (simd_lane_id == 0) {
            reduce_scratch[active_q * kAttnMaxSimdGroups + local_group] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_group == 0) {
            float t = (simd_lane_id < groups_per_q)
                ? reduce_scratch[active_q * kAttnMaxSimdGroups + simd_lane_id]
                : 0.0f;
            t = simd_sum(t);
            if (simd_lane_id == 0) { bcast[active_q] = t; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s = bcast[active_q] * attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s - m_new);
        d_run = d_run * alpha + p_exp;
        for (uint slot = 0; slot < kGQAPerThread; ++slot) { o_local[slot] *= alpha; }
        m_run = m_new;

        uint slot = 0;
        for (uint i = local_lid; i < HD; i += threads_per_q) {
            o_local[slot] += p_exp * float(V_row[i]);
            slot += 1;
        }
    }

    const uint q_head = q_base + active_q;
    const uint base = uint(q_head) * NC + chunk;
    if (local_lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = local_lid; i < HD; i += threads_per_q) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

// Full-attention GQA kernel, 1024 threads per threadgroup.
//
// Why 1024 threads: the generic full kernel runs 256 threads each scanning 2
// head_dim elements per position. Splitting 8 Q heads across 8x32-thread
// simdgroups (256 total) makes each thread scan 16 elements serially — long
// FMA chains, low occupancy, measured 1.5-2x SLOWER than generic. Instead we
// give every Q head 128 threads (4 simdgroups, 32 lanes each) and scan 4
// CONTIGUOUS head_dim elements per thread. Each K/V row is still read from
// device memory exactly once per chunk (shared across all 8 Q heads), which
// is the point of the GQA kernel.
//
// Layout per threadgroup:
//   q_smem   [8][512]  FP32  — Q slice of each Q head (16 KiB)
//   kv_smem  [512]     FP32  — current position's K row (2 KiB)
//   vv_smem  [512]     FP32  — current position's V row (2 KiB)
//   score_sg [8][8]    FP32  — per-(q_head, simdgroup) partial scores
//   m_head   [8], d_head [8] — per-q_head softmax state (atomics)
// Total ≈ 22.5 KiB threadgroup memory; 1024 threads = 32 simdgroups.
//
// Barriers: ONE threadgroup barrier per position (after K/V load). The score
// reduction uses simd_sum + a threadgroup-scratch broadcast instead of a
// second barrier, so per-position cost is one barrier and one simd_sum.
[[kernel, max_total_threads_per_threadgroup(1024)]]
void attention_decode_gqa_full_partial(
    device const half*  Q             [[buffer(0)]],
    device const half*  K             [[buffer(1)]],
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxFullQPerKV * kAttnMaxHeadDim];
    threadgroup float kv_smem[kAttnMaxHeadDim];
    threadgroup float vv_smem[kAttnMaxHeadDim];
    threadgroup float score_sg[kAttnMaxFullQPerKV * kAttnMaxSimdGroups];
    threadgroup float m_head[kAttnMaxFullQPerKV];
    threadgroup float d_head[kAttnMaxFullQPerKV];
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);

    const uint kv_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    const uint q_per_kv = NQ / NKV;
    if (q_per_kv > kAttnMaxFullQPerKV) { return; }
    // This kernel requires 128 threads per Q head (4 simdgroups). With
    // simdgroups=32 and q_per_kv=8 that is exactly the dispatch we issue.
    const uint groups_per_q = max(1u, simdgroups / q_per_kv);
    if (groups_per_q != 4) { return; }   // safety: geometry assumes 4
    const uint threads_per_q = groups_per_q * 32u;

    const uint active_q = simd_group_id / groups_per_q;
    const uint local_group = simd_group_id - active_q * groups_per_q;
    const uint local_lid = local_group * 32u + simd_lane_id;
    const uint q_head = kv_head * q_per_kv + active_q;
    const uint q_per_thread = (HD + threads_per_q - 1u) / threads_per_q;   // 4

    // Load this thread's Q slice into shared memory. Each Q head has 128
    // threads (4 simdgroups x 32 lanes); thread t covers columns
    // [t*4, t*4+4) of its head — together all 512 columns. (A global
    // `i += lsize` stride would step 1024 > 512 and cover only 128 columns.)
    device const half* Q_row = Q + uint(q_head) * HD;
    {
        const uint col = local_lid * q_per_thread;
        for (uint k = 0; k < q_per_thread; ++k) {
            const uint i = col + k;
            q_smem[active_q * kAttnMaxHeadDim + i] =
                (i < HD) ? float(Q_row[i]) : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float o_local[kAttnMaxHeadDim / 32];   // 16 slots, q_per_thread(4) used
    for (uint k = 0; k < q_per_thread; ++k) { o_local[k] = 0.0f; }
    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (uint p = p_start; p < p_end; ++p) {
        const uint phys_p = attn_ring_slot(p);
        device const half* K_row = K + (phys_p * NKV + kv_head) * HD;
        device const half* V_row = V + (phys_p * NKV + kv_head) * HD;
        for (uint i = lid; i < HD; i += lsize) { kv_smem[i] = float(K_row[i]); }
        for (uint i = lid; i < HD; i += lsize) { vv_smem[i] = float(V_row[i]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Dot product over this thread's 4 contiguous columns.
        const uint col = local_lid * q_per_thread;
        float partial = 0.0f;
        for (uint k = 0; k < q_per_thread; ++k) {
            partial = fma(q_smem[active_q * kAttnMaxHeadDim + col + k],
                          kv_smem[col + k], partial);
        }
        // Reduce within the simdgroup; the group's lane 0 publishes to
        // score_sg, and every thread reads its own head's 4 partials after
        // the single barrier (the K/V load barrier of the NEXT iteration is
        // not relied on — this barrier below orders the publish+read).
        float s = simd_sum(partial);
        if (simd_lane_id == 0) {
            score_sg[active_q * kAttnMaxSimdGroups + local_group] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s = score_sg[active_q * kAttnMaxSimdGroups + 0]
          + score_sg[active_q * kAttnMaxSimdGroups + 1]
          + score_sg[active_q * kAttnMaxSimdGroups + 2]
          + score_sg[active_q * kAttnMaxSimdGroups + 3];
        s *= attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s - m_new);
        d_run = d_run * alpha + p_exp;
        for (uint k = 0; k < q_per_thread; ++k) { o_local[k] *= alpha; }
        m_run = m_new;

        // Accumulate V over this thread's 4 contiguous columns.
        for (uint k = 0; k < q_per_thread; ++k) {
            o_local[k] += p_exp * vv_smem[col + k];
        }
    }

    const uint base = uint(q_head) * NC + chunk;
    if (local_lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    const uint col = local_lid * q_per_thread;
    for (uint k = 0; k < q_per_thread; ++k) {
        const uint i = col + k;
        if (i < HD) { o_row[i] = o_local[k]; }
    }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_combine(
    device const float* m_in         [[buffer(0)]],    // [num_q_heads * num_chunks]
    device const float* d_in         [[buffer(1)]],
    device const float* o_in         [[buffer(2)]],    // [num_q_heads * num_chunks * head_dim]
    device       half*  out          [[buffer(3)]],    // [num_q_heads * head_dim]
    constant     uint&  head_dim     [[buffer(4)]],
    constant     uint&  num_chunks   [[buffer(5)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]]
) {
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NC = attn_fc_num_chunks(num_chunks);
    const uint q_head = tg_id;
    device const float* m_row  = m_in + uint(q_head) * NC;
    device const float* d_row  = d_in + uint(q_head) * NC;
    device const float* o_base = o_in + uint(q_head) * NC * HD;

    // num_chunks is small (<= a few dozen); each thread recomputes the global
    // max and denominator rather than pay a threadgroup reduction + barriers.
    float m_glob = -INFINITY;
    for (uint c = 0; c < NC; ++c) { m_glob = max(m_glob, m_row[c]); }
    float D = 0.0f;
    for (uint c = 0; c < NC; ++c) { D += d_row[c] * attn_softmax_exp(m_row[c] - m_glob); }
    const float inv_d = (D > 0.0f) ? (1.0f / D) : 0.0f;

    device half* out_row = out + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        float acc = 0.0f;
        for (uint c = 0; c < NC; ++c) {
            acc += o_base[c * HD + i] * attn_softmax_exp(m_row[c] - m_glob);
        }
        out_row[i] = half(acc * inv_d);
    }
}
