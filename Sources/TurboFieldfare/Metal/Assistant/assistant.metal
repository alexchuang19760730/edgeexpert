// Assistant MTP Draft Model Metal Kernels
// All weight buffers are BF16 (upper 16 bits of FP32, stored as uint16).
// Intermediate activation buffers are F16 (half).

#include <metal_stdlib>
using namespace metal;

// Convert BF16 (uint16) to float32
static inline float bf16_to_float(uint16_t v) {
    return as_type<float>(uint32_t(v) << 16);
}

// Convert F16 (half) to float32
static inline float f16_to_float(half h) {
    return float(h);
}

// MARK: - Embedding Lookup (BF16 weights -> F16 output)

kernel void assistant_embed_lookup_bf16(
    device const uint16_t* weights [[buffer(0)]],  // [vocab, hidden] BF16
    device const uint32_t* token   [[buffer(1)]],   // single token ID
    device half* output            [[buffer(2)]],   // [hidden] F16
    constant uint32_t& rowWidth    [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= rowWidth) return;
    uint32_t tok = token[0];
    output[tid] = half(bf16_to_float(weights[tok * rowWidth + tid]));
}

// MARK: - Concat two halves

kernel void assistant_concat_halves(
    device const half* lhs [[buffer(0)]],
    device const half* rhs [[buffer(1)]],
    device half* output     [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    output[tid] = lhs[tid];
    output[count + tid] = rhs[tid];
}

// MARK: - Residual add then scale: output = (base + delta) * scale
//
// Matches `Gemma4TextDecoderLayer.forward`, where the residual is added
// unscaled and `layer_scalar` multiplies the *whole* layer output exactly
// once at the very end:
//
//     hidden_states = residual + hidden_states
//     ...
//     hidden_states *= self.layer_scalar
//
// Pass scale = 1.0 for the intermediate (post-attention) residual add.

kernel void assistant_add_then_scale(
    device const half* base   [[buffer(0)]],
    device const half* delta  [[buffer(1)]],
    device half* output        [[buffer(2)]],
    constant uint32_t& count   [[buffer(3)]],
    constant float& scale      [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    output[tid] = half((float(base[tid]) + float(delta[tid])) * scale);
}

// MARK: - GELU (tanh approximation) multiply: output = gelu_tanh(gate) * up
//
// Gemma 4 sets `hidden_activation = "gelu_pytorch_tanh"`, i.e.
// `F.gelu(x, approximate="tanh")`, NOT SiLU:
//
//     0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

kernel void assistant_gelu_mul(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half* output      [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    float g = float(gate[tid]);
    float u = float(up[tid]);
    const float kSqrt2OverPi = 0.7978845608028654f;
    float inner = kSqrt2OverPi * (g + 0.044715f * g * g * g);
    float gelu = 0.5f * g * (1.0f + precise::tanh(inner));
    output[tid] = half(gelu * u);
}

// MARK: - BF16 GEMV (matrix-vector multiply)
// weights: [rows, cols] BF16, row-major
// input:   [cols] F16
// output:  [rows] F16
// One threadgroup per row, threads cooperate via threadgroup reduction

kernel void assistant_bf16_gemv(
    device const uint16_t* weights [[buffer(0)]],  // [rows, cols] BF16
    device const half* input        [[buffer(1)]],  // [cols] F16
    device half* output             [[buffer(2)]],  // [rows] F16
    constant uint32_t& rows         [[buffer(3)]],
    constant uint32_t& cols         [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Each threadgroup handles one row
    if (tg_id >= rows) return;

    device const uint16_t* row = weights + (uint64_t)tg_id * cols;

    // Each thread accumulates a partial dot product
    float partial = 0.0f;
    for (uint i = tid; i < cols; i += tg_size) {
        partial += bf16_to_float(row[i]) * f16_to_float(input[i]);
    }

    // Threadgroup reduction
    if (tg_size >= 256) {
        threadgroup float shared[256];
        shared[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = 128; s > 0; s >>= 1) {
            if (tid < s && tid + s < tg_size) {
                shared[tid] += shared[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            output[tg_id] = half(shared[0]);
        }
    } else {
        // Small threadgroup: single thread does it all
        if (tid == 0) {
            float sum = 0.0f;
            for (uint i = 0; i < cols; i++) {
                sum += bf16_to_float(row[i]) * f16_to_float(input[i]);
            }
            output[tg_id] = half(sum);
        }
    }
}

// MARK: - Argmax over F16 values -> UInt32 token

kernel void assistant_argmax_half(
    device const half* values     [[buffer(0)]],
    device uint32_t* outputToken  [[buffer(1)]],
    constant uint32_t& count      [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Single-threadgroup reduction for argmax
    threadgroup float shared_vals[256];
    threadgroup uint32_t shared_idx[256];

    float my_val = -INFINITY;
    uint32_t my_idx = 0;

    for (uint i = tid; i < count; i += tg_size) {
        float v = float(values[i]);
        if (v > my_val) {
            my_val = v;
            my_idx = i;
        }
    }

    shared_vals[tid] = my_val;
    shared_idx[tid] = my_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_vals[tid + s] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        outputToken[0] = shared_idx[0];
    }
}
