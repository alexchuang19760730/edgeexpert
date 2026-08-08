// ============================================================================
// deltanet.metal — Gated Delta Rule 线性注意力 (Qwen3.6-35B-A3B, 30/40 层)
//
// 数学 (权威: transformers qwen3_5_moe, torch_recurrent_gated_delta_rule):
//   1. 投影:  qkv = x·W_qkv (2048→8192), z = x·W_z (→4096),
//             b = x·W_b (→32), a = x·W_a (→32)
//   2. conv:  qkv = silu(conv1d(qkv))  (groups=8192, kernel=4, 因果)
//   3. 切分:  [key_dim=2048, key_dim=2048, value_dim=4096] → q[16,128] k[16,128] v[32,128]
//   4. 归一:  q = l2norm(q,eps=1e-6)·(1/√128);  k = l2norm(k,eps=1e-6)
//   5. GQA:   q/k repeat_interleave(2) → [32,128]
//   6. 门控:  g = -exp(A_log)·softplus(a+dt_bias)  (每 V 头, 负数)
//             beta = sigmoid(b)                     (每 V 头)
//   7. Delta Rule (每 V 头独立状态 h ∈ R^{128×128}, FP32):
//       h ← h·exp(g);  kv_mem = hᵀ·k;
//       delta = (v - kv_mem)·beta;  h ← h + k⊗delta;  o = hᵀ·q
//   8. 输出:  o = silu(z)·rmsnorm_gated(o);  y = o·W_out (4096→2048)
//
// Golden 对照: golden_layer0.json (token 级 q/k/v/g/beta/h/o)
// ============================================================================
#include <metal_stdlib>
using namespace metal;

constant constexpr uint kDeltaThreads      = 256;   // 每 V 头一个 threadgroup
constant constexpr uint kDeltaHeadDim      = 128;   // k/v head_dim
constant constexpr uint kDeltaNumVHeads    = 32;    // 32 V 头
constant constexpr uint kDeltaNumKHeads    = 16;    // 16 K 头
constant constexpr uint kDeltaKeyDim       = 2048;  // 16×128
constant constexpr uint kDeltaValueDim     = 4096;  // 32×128
// conv_dim = key_dim*2 + value_dim = 2048*2 + 4096 = 8192 (官方: key_dim*2 = 4096)
constant constexpr uint kDeltaConvDim      = 8192;  // key_dim*2 + value_dim
constant constexpr uint kDeltaConvKernel   = 4;     // 因果卷积窗口
constant constexpr float kDeltaScale       = 0.088388; // 1/√128
constant constexpr float kDeltaNormEps     = 1e-6;

// ============================================================================
// Kernel 1: 投影 GEMV — x [2048] → qkv[8192], z[4096], b[32], a[32]
// dispatch: 4 组 GEMV (qkv/z/b/a), 每组合并 256 线程
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kDeltaThreads)]]
void deltanet_project(
    device const half*  x          [[buffer(0)]],  // [2048]
    device const half*  W_qkv      [[buffer(1)]],  // [8192, 2048] (行主序, 每行内积)
    device const half*  W_z        [[buffer(2)]],  // [4096, 2048]
    device const half*  W_b        [[buffer(3)]],  // [32, 2048]
    device const half*  W_a        [[buffer(4)]],  // [32, 2048]
    device       half*  qkv_out    [[buffer(5)]],  // [8192]
    device       half*  z_out      [[buffer(6)]],  // [4096]
    device       half*  b_out      [[buffer(7)]],  // [32]
    device       half*  a_out      [[buffer(8)]],  // [32]
    constant     uint&  mode       [[buffer(9)]],  // 0=qkv 1=z 2=b 3=a
    uint tid [[thread_position_in_grid]],
    uint tnum [[threads_per_grid]]
) {
    // 统一 GEMV: out[r] = Σ_c x[c]·W[r,c]
    uint total_out = (mode == 0) ? kDeltaConvDim : (mode == 1) ? kDeltaValueDim : 32;
    const device half* W = (mode == 0) ? W_qkv : (mode == 1) ? W_z : (mode == 2) ? W_b : W_a;
    device half* out = (mode == 0) ? qkv_out : (mode == 1) ? z_out : (mode == 2) ? b_out : a_out;
    for (uint r = tid; r < total_out; r += tnum) {
        const device half* wrow = W + (size_t)r * 2048;
        float acc = 0.0f;
        for (uint c = 0; c < 2048; c++) {
            acc += float(x[c]) * float(wrow[c]);
        }
        out[r] = half(acc);
    }
}

// ============================================================================
// Kernel 2: 因果深度卷积 + silu — qkv [8192] (单 token), 用 conv_state 历史
// dispatch: 8192 通道并行 (每线程 1 通道 × kernel 4)
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kDeltaThreads)]]
void deltanet_conv1d(
    device const half*  qkv_in      [[buffer(0)]],  // [8192]  (投影后)
    device const half*  conv_weight [[buffer(1)]],  // [8192, 1, 4] (groups 深度卷积)
    device const half*  conv_state  [[buffer(2)]],  // [8192, 4] (前 3 token 历史 + 1 空槽, 滚动)
    device       half*  qkv_out     [[buffer(3)]],  // [8192]   (conv+silu 后)
    device       half*  conv_state_new [[buffer(4)]], // [8192, 4] (新历史)
    uint tid [[thread_position_in_grid]],
    uint tnum [[threads_per_grid]]
) {
    // 每通道独立 depthwise conv (kernel=4 因果):
    //   y[c] = silu( Σ_{k=0..3} conv_w[c,0,k] · seq[c, k])   (conv1d.weight 布局 [C,1,K])
    // conv1d padding = kernel-1 = 3 (左侧填充), 输出同长
    // 单 token decode: seq 是 [conv_state[c,0..2], qkv_in[c]] (3 填充历史 + 1 当前)
    //   y[c] = w[c,0]·conv_state[c,2] + w[c,1]·conv_state[c,1]
    //        + w[c,2]·conv_state[c,0] + w[c,3]·qkv_in[c]
    for (uint c = tid; c < kDeltaConvDim; c += tnum) {
        float acc = 0.0f;
        acc += float(conv_weight[c*4 + 0]) * float(conv_state[c*4 + 2]);
        acc += float(conv_weight[c*4 + 1]) * float(conv_state[c*4 + 1]);
        acc += float(conv_weight[c*4 + 2]) * float(conv_state[c*4 + 0]);
        acc += float(conv_weight[c*4 + 3]) * float(qkv_in[c]);
        // 精确 silu (golden 对照)
        float y = acc * (1.0f / (1.0f + exp(-acc)));
        qkv_out[c] = half(y);
        // 新历史 = [conv_state[c,1], conv_state[c,0], qkv_in[c], 0] (滚动窗口)
        conv_state_new[c*4 + 0] = conv_state[c*4 + 1];
        conv_state_new[c*4 + 1] = conv_state[c*4 + 0];
        conv_state_new[c*4 + 2] = qkv_in[c];
        conv_state_new[c*4 + 3] = half(0.0f);
    }
}

// ============================================================================
// Kernel 2.5: 预处理 — qkv_post [8192] → q/k/v [32,128] (L2norm+GQA) + g/beta [32]
// 通道布局 (官方): mixed_qkv = [key_dim=2048, key_dim=2048, value_dim=4096]
//   q = qkv_post[0:2048]    → [16,128]  L2norm ×1/√128, GQA repeat_interleave(2) → [32,128]
//   k = qkv_post[2048:4096] → [16,128]  L2norm, GQA repeat_interleave(2) → [32,128]
//   v = qkv_post[4096:8192] → [32,128]
//   g = -exp(A_log)·softplus(a+dt_bias)  (per V 头, FP32)
//   beta = sigmoid(b)                     (per V 头, FP32)
// dispatch: 64 线程 (32 V 头 × 每头 2 相位) — 每线程处理一头的 L2norm 归约 + g/beta
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kDeltaNumVHeads)]]
void deltanet_prepare(
    device const half*  qkv_post [[buffer(0)]],  // [8192] conv+silu 后
    device const half*  a_in     [[buffer(1)]],  // [32]  x·W_a (FP16, Kernel 1 输出)
    device const half*  b_in     [[buffer(2)]],  // [32]  x·W_b (FP16)
    device const half*  A_log    [[buffer(3)]],  // [32] (bf16 权重)
    device const half*  dt_bias  [[buffer(4)]],  // [32] (bf16 权重)
    device       half*  q_k      [[buffer(5)]],  // [32,128] 输出
    device       half*  k_k      [[buffer(6)]],  // [32,128]
    device       half*  v_k      [[buffer(7)]],  // [32,128]
    device       float* g_k      [[buffer(8)]],  // [32]
    device       float* beta_k   [[buffer(9)]],  // [32]
    uint v_head [[thread_position_in_grid]],
    uint tnum   [[threads_per_grid]]
) {
    // 每线程处理一个 V 头 (32 线程)
    // ---- g / beta (per V 头) ----
    float av = float(a_in[v_head]);
    float bv = float(b_in[v_head]);
    float A = exp(float(A_log[v_head]));         // A_log 直接存 log(A)
    float sp = max(av + float(dt_bias[v_head]), 0.0f) + log(1.0f + exp(-fabs(av + float(dt_bias[v_head]))));  // softplus 稳定式
    g_k[v_head] = -A * sp;                   // g < 0
    beta_k[v_head] = 1.0f / (1.0f + exp(-bv));

    // ---- v: qkv_post[4096:8192] → [32,128] (直接拷贝) ----
    // ---- q/k: L2norm + GQA (V 头 h 对应 K 头 h/2, GQA 2:1) ----
    uint k_src = v_head / 2;                 // K 头索引 (16 个 K 头, 每 K 头服务 2 个 V 头)
    // q: qkv_post[0:2048] 中第 k_src 个 K 头 → q_k[v_head]
    // 注意: GQA 展开 q/k 时, V 头 h 用 K 头 (h/2) 的 q/k, 但 q 是 16 头全量 — Q 头数=16=K 头数*8
    // 核对 golden: q/k 是 [16,128] (16 K/Q 头), v 是 [32,128]; GQA 展开指 q/k repeat_interleave(2)
    // 即 q_k[v_head] = q_k_16[v_head/2], k_k[v_head] = k_k_16[v_head/2]
    uint qsrc = k_src * kDeltaHeadDim;       // q 头基址 (qkv_post 0..2048)
    uint ksrc = kDeltaKeyDim + k_src * kDeltaHeadDim;  // k 头基址 (qkv_post 2048..4096)
    uint vsrc = kDeltaKeyDim * 2 + v_head * kDeltaHeadDim;  // v 头基址 (4096..8192)

    // L2norm (per K 头, 128 维) — 单线程归约 (32 线程并行, 每线程 128 维串行, 可接受)
    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint i = 0; i < kDeltaHeadDim; i++) {
        float qv = float(qkv_post[qsrc + i]);
        float kv = float(qkv_post[ksrc + i]);
        q_sum += qv * qv;
        k_sum += kv * kv;
    }
    float q_norm = rsqrt(max(q_sum, kDeltaNormEps)) * kDeltaScale;  // L2norm × 1/√128
    float k_norm = rsqrt(max(k_sum, kDeltaNormEps));
    for (uint i = 0; i < kDeltaHeadDim; i++) {
        uint qout = v_head * kDeltaHeadDim + i;
        q_k[qout] = half(float(qkv_post[qsrc + i]) * q_norm);
        k_k[qout] = half(float(qkv_post[ksrc + i]) * k_norm);
        v_k[qout] = qkv_post[vsrc + i];
    }
}

// ============================================================================
// Kernel 3: Gated Delta Rule — 状态更新 + 输出 (核心)
// dispatch: 每 V 头 1 个 threadgroup (32 TGs), 每 TG 128 线程 (k_dim 并行)
// 状态 h: [32, 128, 128] FP32, 跨 token 持久 (buffer 5)
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kDeltaHeadDim)]]
void deltanet_delta_rule(
    device const half*  q_k    [[buffer(0)]],  // [32, 128]   (L2norm+scale 后, GQA 展开)
    device const half*  k_k    [[buffer(1)]],  // [32, 128]   (L2norm 后, GQA 展开)
    device const half*  v_k    [[buffer(2)]],  // [32, 128]
    device const float* g_k    [[buffer(3)]],  // [32]  (g = -exp(A)·softplus(a+dt), 每 V 头)
    device const float* beta_k [[buffer(4)]],  // [32]  (sigmoid(b))
    device       float* h      [[buffer(5)]],  // [32, 128, 128] 状态 (FP32, 跨 token)
    device       float* o_out  [[buffer(6)]],  // [32, 128]
    uint v_head [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]]   // 0..127 (k_dim 索引)
) {
    threadgroup float k_smem[kDeltaHeadDim];   // k 广播
    threadgroup float q_smem[kDeltaHeadDim];   // q 广播
    threadgroup float kv_mem[kDeltaHeadDim];   // hᵀk 结果
    threadgroup float delta_smem[kDeltaHeadDim];

    const uint hoff = v_head * (kDeltaHeadDim * kDeltaHeadDim);  // h 状态基址

    // 载入 q/k/v (FP16→FP32)
    float qv = float(q_k[v_head * kDeltaHeadDim + lid]);
    float kv = float(k_k[v_head * kDeltaHeadDim + lid]);
    float vv = float(v_k[v_head * kDeltaHeadDim + lid]);
    k_smem[lid] = kv;
    q_smem[lid] = qv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gv = exp(g_k[v_head]);       // g < 0 → exp(g) < 1 (衰减)
    float bv = beta_k[v_head];

    // ① 状态衰减 + 计算 kv_mem = hᵀ·k (每线程负责一列 j=lid):
    //    kv_mem[j] = Σ_i h[i,j]·k[i]  —— 用【衰减后】h (对齐 golden:
    //    h = h*g_t; kv_mem = (h * k).sum(-2))
    float acc_kv = 0.0f;
    for (uint i = 0; i < kDeltaHeadDim; i++) {
        uint idx = hoff + i * kDeltaHeadDim + lid;   // h[i, j=lid]
        float hval = h[idx] * gv;                    // ① 衰减
        h[idx] = hval;                               // 写回衰减后
        acc_kv += hval * k_smem[i];                  // ② 读: 用衰减后 h (对齐 golden)
    }
    kv_mem[lid] = acc_kv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ③ 误差: delta[j] = (v[j] - kv_mem[j])·beta
    float delta = (vv - kv_mem[lid]) * bv;
    delta_smem[lid] = delta;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ④ 写: h[i,j] += k[i]·delta[j]  (每线程一列 j=lid, 遍历 i)
    for (uint i = 0; i < kDeltaHeadDim; i++) {
        uint idx = hoff + i * kDeltaHeadDim + lid;
        h[idx] += k_smem[i] * delta_smem[lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ⑤ 输出: o[j] = Σ_i h[i,j]·q[i]  (用更新后 h)
    float acc_o = 0.0f;
    for (uint i = 0; i < kDeltaHeadDim; i++) {
        uint idx = hoff + i * kDeltaHeadDim + lid;
        acc_o += h[idx] * q_smem[i];
    }
    o_out[v_head * kDeltaHeadDim + lid] = acc_o;
}

// ============================================================================
// Kernel 4: Gated RMSNorm + 输出 GEMV
// o [32,128] → silu(z)·rmsnorm → out_proj [4096→2048]
// mode 0 dispatch: 32 threadgroups × 128 线程 (每头 1 TG, 4 simdgroup)
// mode 1 dispatch: 2048 线程 (out_proj 行并行)
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kDeltaThreads)]]
void deltanet_output(
    device const half*  o_in     [[buffer(0)]],  // [32, 128]  (Delta Rule 输出, FP16)
    device const half*  z_in     [[buffer(1)]],  // [32, 128]  (z 投影, 门控)
    device const half*  norm_w   [[buffer(2)]],  // [128] per-head 共享
    device const half*  W_out    [[buffer(3)]],  // [2048, 4096]
    device       half*  y_out    [[buffer(4)]],  // [2048]
    device       half*  o_normed [[buffer(5)]],  // [32, 128] norm 输出 (mode 0 写, mode 1 读)
    constant     uint&  mode     [[buffer(6)]],  // 0=norm 1=out_proj
    uint tid [[thread_position_in_grid]],
    uint tnum [[threads_per_grid]],
    uint tg   [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    threadgroup float tg_sq_sum[4];   // 每 simdgroup 的平方和 (4 个 simdgroup)
    if (mode == 0) {
        // ---- Gated RMSNorm (per V 头 128 独立, 官方 core_attn_out.reshape(-1,128)) ----
        // 每 TG = 1 个头 (dispatch 32 TG × 128 线程)
        float v = float(o_in[tg * kDeltaHeadDim + lane]);
        float sq = v * v;
        float s = simd_sum(sq);                       // simdgroup 内归约
        if (simd_is_first()) tg_sq_sum[lane / 32] = s;  // 每 simdgroup first 线程写槽 (lane 0/32/64/96)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            float total = tg_sq_sum[0] + tg_sq_sum[1] + tg_sq_sum[2] + tg_sq_sum[3];
            float mean_sq = total / float(kDeltaHeadDim);
            tg_sq_sum[0] = mean_sq;                   // 复用槽广播均值
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mean_sq = tg_sq_sum[0];
        float rms = v * rsqrt(max(mean_sq, kDeltaNormEps));
        float z = float(z_in[tg * kDeltaHeadDim + lane]);
        float gated = rms * float(norm_w[lane]) * (z / (1.0f + exp(-z)));  // silu(z)
        o_normed[tg * kDeltaHeadDim + lane] = half(gated);
    } else {
        // ---- out_proj: y[r] = Σ_c o[c]·W_out[r,c], o = [4096] (32×128) ----
        for (uint r = tid; r < 2048; r += tnum) {
            const device half* wrow = W_out + (size_t)r * 4096;
            float acc = 0.0f;
            for (uint c = 0; c < 4096; c++) {
                acc += float(o_normed[c]) * float(wrow[c]);
            }
            y_out[r] = half(acc);
        }
    }
}
