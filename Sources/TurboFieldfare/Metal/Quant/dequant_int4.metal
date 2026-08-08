#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_int4 — MLX `affine` 4-bit dequant.
//
// Layout (per row of length N):
//   W       : N/2 bytes. Low nibble of byte k = component 2k (unsigned 0..15),
//             high nibble = component 2k+1.
//   scales  : N/64 BF16, one per group of 64.
//   biases  : N/64 BF16, one per group of 64.
//   value   : w[i] = float(nibble[i]) * scale[i/64] + bias[i/64].
//
// Affine factoring for GEMV (sum over a group of 64):
//   sum_k (q_k * s + b) * x_k = s * sum_k(q_k * x_k) + b * sum_k x_k
// so scale and bias each cost one mul + one FMA per group instead of per
// element; the per-element inner loop keeps the scalar path's FMA count.
// ============================================================================

constant constexpr uint kGroupSize = 64;
constant constexpr uint kMaxGemvN = 8192;   // largest decode matmul width (half)
constant constexpr uint kQKVXMax = 4096;   // smem x cap (halfs) for the QKV smem-x kernel; gemma4 N=2816 fits (8KB)
constant uint FC_INT4_M [[function_constant(20)]];
constant uint FC_INT4_N [[function_constant(21)]];
constant bool FC_INT4_USE_FC [[function_constant(22)]];
constant uint FC_INT4_QKV_MQ [[function_constant(23)]];
constant uint FC_INT4_QKV_MKV [[function_constant(24)]];
constant uint FC_INT4_QKV_N [[function_constant(25)]];
constant bool FC_INT4_QKV_USE_FC [[function_constant(26)]];

static inline uint int4_fc_m(constant uint& M) {
    return (is_function_constant_defined(FC_INT4_USE_FC) &&
            FC_INT4_USE_FC &&
            is_function_constant_defined(FC_INT4_M)) ? FC_INT4_M : M;
}

static inline uint int4_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_INT4_USE_FC) &&
            FC_INT4_USE_FC &&
            is_function_constant_defined(FC_INT4_N)) ? FC_INT4_N : N;
}

static inline uint int4_qkv_fc_mq(constant uint& Mq) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_MQ)) ? FC_INT4_QKV_MQ : Mq;
}

static inline uint int4_qkv_fc_mkv(constant uint& Mkv) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_MKV)) ? FC_INT4_QKV_MKV : Mkv;
}

static inline uint int4_qkv_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_N)) ? FC_INT4_QKV_N : N;
}

inline uint nib_lo(uint8_t b) { return uint(b & 0x0F); }
inline uint nib_hi(uint8_t b) { return uint(b >> 4); }


kernel void embed_lookup_int4(
    device const uint8_t* table     [[buffer(0)]],   // [V, D/2] nibbles
    device const bfloat*  scales    [[buffer(1)]],   // [V, D/64] BF16
    device const bfloat*  biases    [[buffer(2)]],   // [V, D/64] BF16
    device half*          out       [[buffer(3)]],   // [D] FP16
    constant uint&        token_id  [[buffer(4)]],
    constant uint&        D         [[buffer(5)]],
    constant float&       out_scale [[buffer(6)]],   // pass 1.0 to disable
    uint                  gid       [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    const uint groups_per_row = D / kGroupSize;
    device const uint8_t* row_q = table  + uint(token_id) * (D / 2u);
    device const bfloat*  row_s = scales + uint(token_id) * groups_per_row;
    device const bfloat*  row_b = biases + uint(token_id) * groups_per_row;
    uint8_t byte = row_q[gid >> 1];
    uint    q    = (gid & 1u) ? uint(byte >> 4) : uint(byte & 0xFu);
    float   s    = float(row_s[gid / kGroupSize]);
    float   b    = float(row_b[gid / kGroupSize]);
    out[gid] = half((float(q) * s + b) * out_scale);
}

// y[m] = sum_{n} W[m, n] * x[n]. One-SIMD-per-row variant: 32 threads
// cooperate on a single output row, each handling 2 elements per group of 64
// (one byte → two nibbles). simd_sum reduces across the group; lane 0 writes.
//
// Requires N % 64 == 0 (per group of 64). Validated at the wrapper.
// Each threadgroup handles eight consecutive rows, one SIMD per row. The
// larger work unit gives the scheduler enough independent rows while sharing
// the L1-cached input-vector reads.
static inline void dequant_int4_gemv_simd_body(
    device const uint8_t* W,
    device const bfloat*  scales,
    device const bfloat*  biases,
    device const half*    x,
    device half*          y,
    uint                  M,
    uint                  N,
    uint                  rows_per_tg,
    uint                  tg_idx,
    uint                  sg_idx,
    uint                  lane
) {
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint n_groups  = N / kGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    // The vectorized row path reads
    // weights a uint (4 bytes = 8 nibbles) and x as half4 in 4-group (128-byte)
    // blocks, with a scalar byte-per-lane remainder. Within a block the 32
    // lanes split 8-per-group, each handling 8 contiguous elements of one
    // 64-element group, so the affine factoring s·Σqx + b·Σx is preserved
    // (simd_sum aggregates; s/b are constant within a group). Aligned: row
    // stride N/2 and weightsOffset are multiples of 4; x is
    // half4-aligned (lane*8 elements). N=2816/4096/8192 → 44/64/128 groups, all
    // exact 4-blocks; the remainder covers any non-multiple-of-4 group count.
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        // Read the 4-byte weight chunk as two ushorts. The resident weight
        // tensors are 2-byte aligned but NOT 4-byte aligned (BF16 scale/bias
        // regions leave a 2-aligned weightsOffset), so a `uint*` load would be
        // misaligned (undefined → garbage); a `ushort*` load is safe (row stride
        // N/2, weightsOffset, and byte_base are all even) and halves the loads
        // vs byte-by-byte.
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kGroupSize / 2) + lane];
        const float x0 = float(x[g * kGroupSize + lane * 2u]);
        const float x1 = float(x[g * kGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        const float sum = x0 + x1;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

// Threadgroup-x variant of the body: x lives in threadgroup memory (loaded
// once per TG by the QKV smem-x kernel). Same affine factoring and fma order
// as the device version -> byte-identical output.
static inline void dequant_int4_gemv_simd_body_smem(
    device const uint8_t* W,
    device const bfloat*  scales,
    device const bfloat*  biases,
    const threadgroup half* x,
    device half*          y,
    uint                  M,
    uint                  N,
    uint                  rows_per_tg,
    uint                  tg_idx,
    uint                  sg_idx,
    uint                  lane
) {
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint n_groups  = N / kGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const float e0 = float(x[elem]);
        const float e1 = float(x[elem + 1u]);
        const float e2 = float(x[elem + 2u]);
        const float e3 = float(x[elem + 3u]);
        const float e4 = float(x[elem + 4u]);
        const float e5 = float(x[elem + 5u]);
        const float e6 = float(x[elem + 6u]);
        const float e7 = float(x[elem + 7u]);
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kGroupSize / 2) + lane];
        const float x0 = float(x[g * kGroupSize + lane * 2u]);
        const float x1 = float(x[g * kGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        const float sum = x0 + x1;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

kernel void dequant_int4_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint MM = int4_fc_m(M);
    const uint NN = int4_fc_n(N);
    dequant_int4_gemv_simd_body(W, scales, biases, x, y, MM, NN,
                                rows_per_tg, tg_idx, sg_idx, lane);
}

// ============================================================================
// dequant_int4_gemv_tgx — threadgroup-cached x GEMV.
//
// Every row of the matvec multiplies the SAME input vector x. The classic
// one-SIMD-per-row kernel re-reads x from device memory once per row: for
// M=4096 rows that is 4096 × N × 2B of L1/L2 traffic, dwarfing the weight
// bytes. This variant copies x into threadgroup memory once per threadgroup
// (8 rows share it), so the hot loop reads x from fast shared memory.
//
// Layout (unchanged weight layout, group=64, BF16 scale/bias):
//   threadgroup x_smem[N]  (half)
//   256 threads = 8 SIMD × 32 lanes; each SIMD owns one row.
// ============================================================================
kernel void dequant_int4_gemv_tgx(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]],
    uint                  lid    [[thread_position_in_threadgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint MM = int4_fc_m(M);
    const uint NN = int4_fc_n(N);
    if (NN > kMaxGemvN) { return; }   // x_smem sized for decode shapes

    threadgroup half x_smem[kMaxGemvN];
    // Cooperative load of x into threadgroup memory (256 threads).
    for (uint i = lid; i < NN; i += 256) { x_smem[i] = x[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= MM) return;
    const uint n_groups  = NN / kGroupSize;
    const uint row_bytes = NN / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((threadgroup const half4*)(x_smem + elem));
        const half4 xb = *((threadgroup const half4*)(x_smem + elem + 4u));
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kGroupSize / 2) + lane];
        const float x0 = float(x_smem[g * kGroupSize + lane * 2u]);
        const float x1 = float(x_smem[g * kGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        const float sum = x0 + x1;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}


kernel void dequant_int4_qkv_gemv_simd(
    device const uint8_t* qW      [[buffer(0)]],
    device const bfloat*  qScales [[buffer(1)]],
    device const bfloat*  qBiases [[buffer(2)]],
    device const uint8_t* kW      [[buffer(3)]],
    device const bfloat*  kScales [[buffer(4)]],
    device const bfloat*  kBiases [[buffer(5)]],
    device const uint8_t* vW      [[buffer(6)]],
    device const bfloat*  vScales [[buffer(7)]],
    device const bfloat*  vBiases [[buffer(8)]],
    device const half*    x       [[buffer(9)]],
    device half*          qY      [[buffer(10)]],
    device half*          kY      [[buffer(11)]],
    device half*          vY      [[buffer(12)]],
    constant uint&        Mq      [[buffer(13)]],
    constant uint&        Mkv     [[buffer(14)]],
    constant uint&        N       [[buffer(15)]],
    uint                  tg_idx  [[threadgroup_position_in_grid]],
    uint                  sg_idx  [[simdgroup_index_in_threadgroup]],
    uint                  lane    [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint QQ = int4_qkv_fc_mq(Mq);
    const uint KK = int4_qkv_fc_mkv(Mkv);
    const uint NN = int4_qkv_fc_n(N);
    const uint global_row = tg_idx * rows_per_tg + sg_idx;
    const uint total_rows = QQ + 2u * KK;
    if (global_row >= total_rows) { return; }

    device const uint8_t* W;
    device const bfloat* scales;
    device const bfloat* biases;
    device half* y;
    uint local_row;
    uint M;
    if (global_row < QQ) {
        W = qW; scales = qScales; biases = qBiases; y = qY;
        local_row = global_row;
        M = QQ;
    } else if (global_row < QQ + KK) {
        W = kW; scales = kScales; biases = kBiases; y = kY;
        local_row = global_row - QQ;
        M = KK;
    } else {
        W = vW; scales = vScales; biases = vBiases; y = vY;
        local_row = global_row - QQ - KK;
        M = KK;
    }
    dequant_int4_gemv_simd_body(W, scales, biases, x, y, M, NN,
                                1u, local_row, 0u, lane);
}

// QKV GEMV with threadgroup-x smem: loads the shared input x (hidden state,
// N halfs) into threadgroup memory once per TG, then runs the 8 row workunits
// against smem instead of re-reading x from DRAM per row. Byte-identical
// (exact copy, same fma order). Env: TURBO_FIELDFARE_QKV_SMEM_X=1 (Swift side).
kernel void dequant_int4_qkv_gemv_simd_smemx(
    device const uint8_t* qW      [[buffer(0)]],
    device const bfloat*  qScales [[buffer(1)]],
    device const bfloat*  qBiases [[buffer(2)]],
    device const uint8_t* kW      [[buffer(3)]],
    device const bfloat*  kScales [[buffer(4)]],
    device const bfloat*  kBiases [[buffer(5)]],
    device const uint8_t* vW      [[buffer(6)]],
    device const bfloat*  vScales [[buffer(7)]],
    device const bfloat*  vBiases [[buffer(8)]],
    device const half*    x       [[buffer(9)]],
    device half*          qY      [[buffer(10)]],
    device half*          kY      [[buffer(11)]],
    device half*          vY      [[buffer(12)]],
    constant uint&        Mq      [[buffer(13)]],
    constant uint&        Mkv     [[buffer(14)]],
    constant uint&        N       [[buffer(15)]],
    uint                  tg_idx  [[threadgroup_position_in_grid]],
    uint                  sg_idx  [[simdgroup_index_in_threadgroup]],
    uint                  lane    [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint QQ = int4_qkv_fc_mq(Mq);
    const uint KK = int4_qkv_fc_mkv(Mkv);
    const uint NN = int4_qkv_fc_n(N);
    // Cooperative load of x (all 256 threads) BEFORE the barrier; the later
    // early-return on global_row is after the barrier so no lane can skip it.
    threadgroup half xSmem[kQKVXMax];
    if (NN <= kQKVXMax) {
        const uint tid = sg_idx * 32u + lane;
        for (uint i = tid; i < NN; i += 256u) {
            xSmem[i] = x[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint global_row = tg_idx * rows_per_tg + sg_idx;
    const uint total_rows = QQ + 2u * KK;
    if (global_row >= total_rows) { return; }

    device const uint8_t* W;
    device const bfloat* scales;
    device const bfloat* biases;
    device half* y;
    uint local_row;
    uint M;
    if (global_row < QQ) {
        W = qW; scales = qScales; biases = qBiases; y = qY;
        local_row = global_row;
        M = QQ;
    } else if (global_row < QQ + KK) {
        W = kW; scales = kScales; biases = kBiases; y = kY;
        local_row = global_row - QQ;
        M = KK;
    } else {
        W = vW; scales = vScales; biases = vBiases; y = vY;
        local_row = global_row - QQ - KK;
        M = KK;
    }
    if (NN <= kQKVXMax) {
        dequant_int4_gemv_simd_body_smem(W, scales, biases, xSmem, y, M, NN,
                                         1u, local_row, 0u, lane);
    } else {
        // Width beyond the smem cap: device-x fallback (uniform branch, safe).
        dequant_int4_gemv_simd_body(W, scales, biases, x, y, M, NN,
                                    1u, local_row, 0u, lane);
    }
}

// ============================================================================
// bw_ceiling_read — pure DRAM read benchmark. Sums every uint, writes one
// value. Never used in inference; only for measuring the hardware ceiling.
// ============================================================================
kernel void bw_ceiling_read(
    device const uint* buf  [[buffer(0)]],
    device uint*       out  [[buffer(1)]],
    constant uint&     n    [[buffer(2)]],
    uint gid               [[thread_position_in_grid]]
) {
    // Each thread reads 256 contiguous uints (1 KiB) to amortize launch and
    // instruction overhead; the number of threads is n/256.
    constexpr uint per_thread = 256;
    const uint base = gid * per_thread;
    uint sum = 0;
    for (uint k = 0; k < per_thread; ++k) {
        sum += buf[base + k];
    }
    if (gid == 0) { out[0] = sum; }
}

// ---------------------------------------------------------------------------
// Attention 3-bit variants (2026-08-08, TURBO_FIELDFARE_ATTN_BITS=3):
// identical structure to the 4-bit kernels, 3-bit LSB-first unpack per
// moe_int3: row_bytes = N*3/8; block = 3 bytes (24 bits) = 8 codes, 8 x
// elements per lane at elem = blk*256 + lane*8; tail = 2 elements/lane at
// bit (g*64 + lane*2)*3 spanning two bytes.
// ---------------------------------------------------------------------------

static inline void dequant_int4_gemv_simd_body_b3(
    device const uint8_t* W,
    device const bfloat*  scales,
    device const bfloat*  biases,
    device const half*    x,
    device half*          y,
    uint                  M,
    uint                  N,
    uint                  rows_per_tg,
    uint                  tg_idx,
    uint                  sg_idx,
    uint                  lane
) {
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint n_groups  = N / kGroupSize;
    const uint row_bytes = N * 3u / 8u;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 96u + lane * 3u;
        const uint gv = uint(W_row[byte_base])
                      | (uint(W_row[byte_base + 1u]) << 8)
                      | (uint(W_row[byte_base + 2u]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = blk * 256u + lane * 8u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float((gv      ) & 0x7u), e0, dot);
        dot = fma(float((gv >> 3 ) & 0x7u), e1, dot);
        dot = fma(float((gv >> 6 ) & 0x7u), e2, dot);
        dot = fma(float((gv >> 9 ) & 0x7u), e3, dot);
        dot = fma(float((gv >> 12) & 0x7u), e4, dot);
        dot = fma(float((gv >> 15) & 0x7u), e5, dot);
        dot = fma(float((gv >> 18) & 0x7u), e6, dot);
        dot = fma(float((gv >> 21) & 0x7u), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint c0 = g * kGroupSize + lane * 2u;
        const uint bbit = c0 * 3u;
        const uint b0i = bbit / 8u;
        const uint bsh = bbit % 8u;
        const uint gv = uint(W_row[b0i]) | (uint(W_row[b0i + 1u]) << 8);
        const uint q0 = (gv >> bsh) & 0x7u;
        const uint q1 = (gv >> (bsh + 3u)) & 0x7u;
        const float x0 = float(x[c0]);
        const float x1 = float(x[c0 + 1u]);
        float dot = fma(float(q0), x0, 0.0f);
        dot = fma(float(q1), x1, dot);
        const float sum = x0 + x1;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

kernel void dequant_int4_qkv_gemv_simd_b3(
    device const uint8_t* qW      [[buffer(0)]],
    device const bfloat*  qScales [[buffer(1)]],
    device const bfloat*  qBiases [[buffer(2)]],
    device const uint8_t* kW      [[buffer(3)]],
    device const bfloat*  kScales [[buffer(4)]],
    device const bfloat*  kBiases [[buffer(5)]],
    device const uint8_t* vW      [[buffer(6)]],
    device const bfloat*  vScales [[buffer(7)]],
    device const bfloat*  vBiases [[buffer(8)]],
    device const half*    x       [[buffer(9)]],
    device half*          qY      [[buffer(10)]],
    device half*          kY      [[buffer(11)]],
    device half*          vY      [[buffer(12)]],
    constant uint&        Mq      [[buffer(13)]],
    constant uint&        Mkv     [[buffer(14)]],
    constant uint&        N       [[buffer(15)]],
    uint                  tg_idx  [[threadgroup_position_in_grid]],
    uint                  sg_idx  [[simdgroup_index_in_threadgroup]],
    uint                  lane    [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint QQ = int4_qkv_fc_mq(Mq);
    const uint KK = int4_qkv_fc_mkv(Mkv);
    const uint NN = int4_qkv_fc_n(N);
    const uint global_row = tg_idx * rows_per_tg + sg_idx;
    const uint total_rows = QQ + 2u * KK;
    if (global_row >= total_rows) { return; }

    device const uint8_t* W;
    device const bfloat* scales;
    device const bfloat* biases;
    device half* y;
    uint local_row;
    uint M;
    if (global_row < QQ) {
        W = qW; scales = qScales; biases = qBiases; y = qY;
        local_row = global_row;
        M = QQ;
    } else if (global_row < QQ + KK) {
        W = kW; scales = kScales; biases = kBiases; y = kY;
        local_row = global_row - QQ;
        M = KK;
    } else {
        W = vW; scales = vScales; biases = vBiases; y = vY;
        local_row = global_row - QQ - KK;
        M = KK;
    }
    dequant_int4_gemv_simd_body_b3(W, scales, biases, x, y, M, NN,
                                   1u, local_row, 0u, lane);
}

kernel void dequant_int4_gemv_tgx_b3(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]],
    uint                  lid    [[thread_position_in_threadgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint MM = int4_fc_m(M);
    const uint NN = int4_fc_n(N);
    if (NN > kMaxGemvN) { return; }

    threadgroup half x_smem[kMaxGemvN];
    for (uint i = lid; i < NN; i += 256) { x_smem[i] = x[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= MM) return;
    const uint n_groups  = NN / kGroupSize;
    const uint row_bytes = NN * 3u / 8u;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 96u + lane * 3u;
        const uint gv = uint(W_row[byte_base])
                      | (uint(W_row[byte_base + 1u]) << 8)
                      | (uint(W_row[byte_base + 2u]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = blk * 256u + lane * 8u;
        const half4 xa = *((threadgroup const half4*)(x_smem + elem));
        const half4 xb = *((threadgroup const half4*)(x_smem + elem + 4u));
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float((gv      ) & 0x7u), e0, dot);
        dot = fma(float((gv >> 3 ) & 0x7u), e1, dot);
        dot = fma(float((gv >> 6 ) & 0x7u), e2, dot);
        dot = fma(float((gv >> 9 ) & 0x7u), e3, dot);
        dot = fma(float((gv >> 12) & 0x7u), e4, dot);
        dot = fma(float((gv >> 15) & 0x7u), e5, dot);
        dot = fma(float((gv >> 18) & 0x7u), e6, dot);
        dot = fma(float((gv >> 21) & 0x7u), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint c0 = g * kGroupSize + lane * 2u;
        const uint bbit = c0 * 3u;
        const uint b0i = bbit / 8u;
        const uint bsh = bbit % 8u;
        const uint gv = uint(W_row[b0i]) | (uint(W_row[b0i + 1u]) << 8);
        const uint q0 = (gv >> bsh) & 0x7u;
        const uint q1 = (gv >> (bsh + 3u)) & 0x7u;
        const float x0 = float(x_smem[c0]);
        const float x1 = float(x_smem[c0 + 1u]);
        float dot = fma(float(q0), x0, 0.0f);
        dot = fma(float(q1), x1, dot);
        const float sum = x0 + x1;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}
