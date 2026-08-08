import Foundation
import Metal


struct AttentionSplitGeometry: Sendable, Equatable {
    let effectiveLength: Int
    let numChunks: Int
    let chunkLength: Int
    let partialThreadgroups: Int
    let useSWAGroupedPartial: Bool
}


/// Swift wrapper for sliding-window and full-causal decode attention.
///
/// The kernels assume a single decoded query token (`M_q = 1`) and a
/// contiguous KV cache of length `seqLen`. The MPP-tensor-core prefill path
/// (`M_q > 1`) is separate.
///
/// Buffer contracts (FP16 throughout):
///   - `q`   : `[numQHeads, headDim]`
///   - `k`   : `[seqLen, numKVHeads, headDim]`
///   - `v`   : same shape as `k`. Full-layer K and V must remain distinct after
///             their separate per-head normalization and RoPE paths.
///   - `out` : `[numQHeads, headDim]`
final class Attention {
    private let ctx: MetalContext
    private let psoPartial: MTLComputePipelineState
    private let psoGQAPartial: MTLComputePipelineState
    private let psoGQAFullPartial: MTLComputePipelineState
    private let psoCombine: MTLComputePipelineState
    private let psoPartialSWA: MTLComputePipelineState
    private let psoPartialFull: MTLComputePipelineState
    private let psoGQAPartialSWA: MTLComputePipelineState
    private let psoGQAPartialSWAChunks16: MTLComputePipelineState
    private let psoPartialFullChunks16: MTLComputePipelineState
    private let psoGQAFullPartialFull: MTLComputePipelineState
    private let psoGQAFullPartialFullChunks16: MTLComputePipelineState
    private let psoCombineSWA: MTLComputePipelineState
    private let psoCombineFull: MTLComputePipelineState
    private let psoCombineSWAChunks16: MTLComputePipelineState
    private let psoCombineFullChunks16: MTLComputePipelineState
    /// B3: single-pass MPP tensor-core decode attention for the full-attention
    /// shape (512/16/2). Reuses the prefill tensor-ops kernel with queryCount=1
    /// (the same flash attention math as prefill, one query token). Env-gated
    /// via TURBO_FIELDFARE_ATTN_TENSOROPS=1, off by default.
    private let psoDecodeTensorOps: MTLComputePipelineState?
    /// Plan B (2026-08-08): single-pass SWA decode kernel — one Q head per TG,
    /// full KV scan, direct write (no partial scratch / no combine). Env-gated
    /// TURBO_FIELDFARE_ATTN_SINGLE=1, SWA 256/16/8, ring-free only.
    private let psoSingle: MTLComputePipelineState?

    /// Mirrors `kAttnThreads` in `attention.metal`. The kernel was authored
    /// with a hardcoded 256-thread group so its threadgroup-memory scratch
    /// (q_smem[512] + reduce[8] + bcast) sizes are correct.
    static let threadsPerGroup: Int = 256

    /// Project ceilings for the split-KV partial scratch. `kAttnMaxHeadDim` in
    /// attention.metal is 512; the model has 16 Q heads; `maxChunks` bounds the
    /// split factor (and therefore the scratch size: 16·64·512 FP32 ≈ 2 MB).
    static let maxQHeads = 16
    static let maxHeadDim = 512
    static let maxChunks = 64
    /// Full attention uses 16 base chunks by default.
    private static let defaultFullChunks = 16
    private static let defaultGQASWAChunks = 8

    // Partial state written by pass 1, read by pass 2. One shared allocation:
    // attention runs once per layer, serially, and pass 2 hazard-tracks pass 1
    // within the same command buffer — no race (mirrors MoE.routerLogits).
    private let mPartial: MTLBuffer
    private let dPartial: MTLBuffer
    private let oPartial: MTLBuffer

    init(context: MetalContext) throws {
        self.ctx = context
        self.psoPartial = try context.pipeline("attention_decode_partial")
        self.psoGQAPartial = try context.pipeline("attention_decode_gqa_swa_partial")
        self.psoGQAFullPartial = try context.pipeline("attention_decode_gqa_full_partial")
        self.psoCombine = try context.pipeline("attention_decode_combine")
        self.psoPartialSWA = try Self.specializedPipeline(context,
                                                          "attention_decode_partial",
                                                          headDim: 256,
                                                          numQHeads: 16,
                                                          numKVHeads: 8)
        self.psoPartialFull = try Self.specializedPipeline(context,
                                                           "attention_decode_partial",
                                                           headDim: 512,
                                                           numQHeads: 16,
                                                           numKVHeads: 2)
        self.psoGQAPartialSWA = try Self.specializedPipeline(context,
                                                             "attention_decode_gqa_swa_partial",
                                                             headDim: 256,
                                                             numQHeads: 16,
                                                             numKVHeads: 8)
        self.psoGQAFullPartialFull = try Self.specializedPipeline(context,
                                                                  "attention_decode_gqa_full_partial",
                                                                  headDim: 512,
                                                                  numQHeads: 16,
                                                                  numKVHeads: 2)
        self.psoGQAFullPartialFullChunks16 = try Self.specializedPipeline(context,
                                                                          "attention_decode_gqa_full_partial",
                                                                          headDim: 512,
                                                                          numQHeads: 16,
                                                                          numKVHeads: 2,
                                                                          numChunks: 16)
        self.psoGQAPartialSWAChunks16 = try Self.specializedPipeline(context,
                                                                     "attention_decode_gqa_swa_partial",
                                                                     headDim: 256,
                                                                     numQHeads: 16,
                                                                     numKVHeads: 8,
                                                                     numChunks: 16)
        self.psoPartialFullChunks16 = try Self.specializedPipeline(context,
                                                                   "attention_decode_partial",
                                                                   headDim: 512,
                                                                   numQHeads: 16,
                                                                   numKVHeads: 2,
                                                                   numChunks: 16)
        self.psoCombineSWA = try Self.specializedPipeline(context,
                                                          "attention_decode_combine",
                                                          headDim: 256,
                                                          numQHeads: 16,
                                                          numKVHeads: 8)
        self.psoCombineFull = try Self.specializedPipeline(context,
                                                           "attention_decode_combine",
                                                           headDim: 512,
                                                           numQHeads: 16,
                                                           numKVHeads: 2)
        self.psoCombineSWAChunks16 = try Self.specializedPipeline(context,
                                                                  "attention_decode_combine",
                                                                  headDim: 256,
                                                                  numQHeads: 16,
                                                                  numKVHeads: 8,
                                                                  numChunks: 16)
        self.psoCombineFullChunks16 = try Self.specializedPipeline(context,
                                                                   "attention_decode_combine",
                                                                   headDim: 512,
                                                                   numQHeads: 16,
                                                                   numKVHeads: 2,
                                                                   numChunks: 16)
        // MPP tensor cores (Apple9+) only; nil on older GPUs keeps the tiled
        // path. Mirrors PrefillAttention's gating.
        self.psoDecodeTensorOps = context.device.supportsFamily(.apple9)
            ? try? context.pipeline("attention_prefill_full_tensorops_2d_validity_v2")
            : nil
        self.psoSingle = try? Self.specializedPipeline(context,
                                                       "attention_decode_single",
                                                       headDim: 256,
                                                       numQHeads: 16,
                                                       numKVHeads: 8)
        let md = Self.maxQHeads * Self.maxChunks
        guard let m = context.device.makeBuffer(length: md * MemoryLayout<Float>.size,
                                                options: .storageModeShared),
	              let d = context.device.makeBuffer(length: md * MemoryLayout<Float>.size,
	                                                options: .storageModeShared),
	              let o = context.device.makeBuffer(length: md * Self.maxHeadDim * MemoryLayout<Float>.size,
	                                                options: .storageModeShared) else {
            throw MetalError.missingFunction("attention split-KV scratch")
        }
        self.mPartial = m; self.dPartial = d; self.oPartial = o
    }

    /// Number of K/V chunks for a range of `effLen` positions — the split
    /// factor used by the production split path.
    /// A/B knob: TURBO_FIELDFARE_ATTN_CHUNKS=<n> overrides the default chunk
    /// count (both SWA and full) to sweep the per-TG-work vs TG-count tradeoff
    /// (SWA phase-2 probe). Absent -> production defaults (SWA 8, full 16).
    private static let chunkOverride: Int? = {
        guard let v = ProcessInfo.processInfo.environment["TURBO_FIELDFARE_ATTN_CHUNKS"],
              let n = Int(v), n > 0 else { return nil }
        return n
    }()

    /// Plan A (2026-08-08): dynamic chunk count targeting ~64 K/V positions per
    /// chunk. Short contexts (the decode-heavy case, e.g. 128 tok) naturally
    /// collapse to fewer chunks (128 -> 2), which the chunk sweep showed is
    /// the 128-token sweet spot (+7% decode, -8% attn GPU vs fixed 8); long
    /// contexts stay bounded by the per-path default. `chunkOverride` still
    /// wins for A/B probes.
    static func chunkCount(effLen: Int, preferGQASWA: Bool = false) -> Int {
        let eff = max(1, effLen)
        if let c = Self.chunkOverride {
            return max(1, min(c, min(maxChunks, eff)))
        }
        let targetPerChunk = 64
        let dynamic = max(1, (eff + targetPerChunk - 1) / targetPerChunk)
        let defaultChunks = preferGQASWA ? defaultGQASWAChunks : defaultFullChunks
        return max(1, min(dynamic, min(defaultChunks, min(maxChunks, eff))))
    }

    static func splitGeometry(numQHeads: UInt32,
                                     numKVHeads: UInt32,
                                     headDim: UInt32,
                                     seqLen: UInt32,
                                     kvStart: UInt32,
                                     preferGQASWA: Bool) -> AttentionSplitGeometry {
        let qPerKV = Int(numQHeads / numKVHeads)
        // GQA-partial kernels process every Q head sharing a KV head in one
        // threadgroup, so their grid is [numKVHeads * numChunks] instead of
        // [numQHeads * numChunks]. The full path uses such a kernel whenever
        // the shape is 512/16/2 (8 Q per KV) — enable the grouped geometry
        // there too, otherwise encodeSplit would dispatch q_per_kv redundant
        // threadgroups that all write the same output region.
        let useGQAAny = preferGQASWA && qPerKV <= 2
        // The GQA-full kernel exists for the production full-attention shape
        // (512/16/2). Only that shape routes through it; every other full
        // shape keeps the generic per-q-head kernel (which is correct for any
        // head_dim, and avoids the GQA-full kernel's 512-only assumptions).
        let useGQAFull = !preferGQASWA && qPerKV > 2
            && numQHeads == 16 && numKVHeads == 2 && headDim == 512
        let useGroupedPartial = useGQAAny || useGQAFull
        let effectiveLength = Int(seqLen) - Int(kvStart)
        // Chunk default depends on the true path: SWA uses 8 base chunks,
        // full uses 16. The GQA-full kernel runs 8 Q heads per TG so it keeps
        // the full-16 chunking (do NOT amplify by q_per_kv — 16×8=128 would
        // exceed maxChunks; unlike the GQA-SWA kernel it needs no amplification
        // because its 8 Q heads already provide the parallelism).
        let baseChunks = Self.chunkCount(effLen: effectiveLength,
                                         preferGQASWA: useGQAAny)
        let numChunks = useGQAAny
            ? max(baseChunks, min(Self.maxChunks, baseChunks * qPerKV))
            : baseChunks
        let chunkLength = (max(1, effectiveLength) + numChunks - 1) / numChunks
        let partialHeadGroups = useGroupedPartial ? Int(numKVHeads) : Int(numQHeads)
        return AttentionSplitGeometry(effectiveLength: effectiveLength,
                                      numChunks: numChunks,
                                      chunkLength: chunkLength,
                                      partialThreadgroups: partialHeadGroups * numChunks,
                                      useSWAGroupedPartial: useGroupedPartial)
    }


    /// Sliding-window attention. `window` caps the K/V positions to the most
    /// recent `window` entries (`[max(0, seqLen-window), seqLen)`).
    /// `scale` defaults to `rsqrt(head_dim)` for generic callers;
    /// Gemma 4 callers pass 1.0 because the model's configured attention scale
    /// is 1.0.
    func encodeSWA(commandBuffer: MTLCommandBuffer,
                          q: MTLBuffer, qOffset: Int = 0,
                          k: MTLBuffer, kOffset: Int = 0,
                          v: MTLBuffer, vOffset: Int = 0,
                          out: MTLBuffer, outOffset: Int = 0,
                          headDim: UInt32,
                          numQHeads: UInt32,
                          numKVHeads: UInt32,
                          seqLen: UInt32,
                          window: UInt32,
                          scale: Float? = nil,
                          ringCapacity: UInt32 = 0) {
        precondition(numQHeads % numKVHeads == 0,
                     "numQHeads must be a multiple of numKVHeads for GQA")
        precondition(headDim <= 512,
                     "head_dim must be <= 512 (kernel scratch is sized for the full-attn case)")
        let sc = scale ?? Self.defaultScale(headDim: headDim)
        let kvStart = seqLen > window ? seqLen - window : 0

        // Plan B (2026-08-08): single-pass SWA kernel when enabled and the
        // shape matches (256/16/8, ring-free). One TG per Q head scanning the
        // whole visible KV range, direct write — no combine launch, no
        // partial scratch round-trip. Byte-identity must be verified per
        // prompt (FP accumulation order is position-sequential either way).
        if Self.singlePassEnabled
            && psoSingle != nil
            && headDim == 256 && numQHeads == 16 && numKVHeads == 8
            && ringCapacity == 0 {
            encodeSingle(commandBuffer: commandBuffer,
                         q: q, qOffset: qOffset,
                         k: k, kOffset: kOffset,
                         v: v, vOffset: vOffset,
                         out: out, outOffset: outOffset,
                         headDim: headDim, numQHeads: numQHeads,
                         numKVHeads: numKVHeads, seqLen: seqLen,
                         kvStart: kvStart, scale: sc)
            return
        }

        encodeSplit(commandBuffer: commandBuffer,
                    q: q, qOffset: qOffset, k: k, kOffset: kOffset,
                    v: v, vOffset: vOffset, out: out, outOffset: outOffset,
                    headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads,
                    seqLen: seqLen, kvStart: kvStart, scale: sc,
                    preferGQASWA: true,
                    ringCapacity: ringCapacity)
    }

    /// Plan B opt-in flag (static let: read once, not per SWA layer).
    private static let singlePassEnabled: Bool =
        ProcessInfo.processInfo.environment["TURBO_FIELDFARE_ATTN_SINGLE"] == "1"
    /// Cross-ref: keep the online-softmax recurrence in attention_decode_single
    /// (attention.metal) in sync with attention_decode_partial — same math. If
    /// one changes the recurrence, the other must too (both are off-by-default
    /// experimental variants of the production split path).
    private static let _syncNote: Void = ()

    /// Single-pass SWA decode attention: one TG per Q head, full visible K/V
    /// scan, direct FP16 write to `out`.
    private func encodeSingle(commandBuffer: MTLCommandBuffer,
                              q: MTLBuffer, qOffset: Int,
                              k: MTLBuffer, kOffset: Int,
                              v: MTLBuffer, vOffset: Int,
                              out: MTLBuffer, outOffset: Int,
                              headDim: UInt32, numQHeads: UInt32,
                              numKVHeads: UInt32, seqLen: UInt32,
                              kvStart: UInt32, scale: Float) {
        guard let pso = psoSingle else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var hd = headDim, nq = numQHeads, nkv = numKVHeads
        var sl = seqLen, ks = kvStart, sc = scale
        enc.setBytes(&hd,  length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&nq,  length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&nkv, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&sl,  length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&ks,  length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&sc,  length: MemoryLayout<Float>.size,  index: 9)
        enc.dispatchThreadgroups(
            MTLSize(width: Int(numQHeads), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadsPerGroup, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Full attention. Gemma 4 reuses the raw K projection as raw V input, but
    /// separate normalization and RoPE make the cache streams distinct here.
    /// `scale` mirrors `encodeSWA`.
    func encodeFull(commandBuffer: MTLCommandBuffer,
                           q: MTLBuffer, qOffset: Int = 0,
                           k: MTLBuffer, kOffset: Int = 0,
                           v: MTLBuffer, vOffset: Int = 0,
                           out: MTLBuffer, outOffset: Int = 0,
                           headDim: UInt32,
                           numQHeads: UInt32,
                           numKVHeads: UInt32,
                           seqLen: UInt32,
                           scale: Float? = nil) {
        precondition(numQHeads % numKVHeads == 0,
                     "numQHeads must be a multiple of numKVHeads for GQA")
        precondition(headDim <= 512,
                     "head_dim must be <= 512 (kernel scratch is sized for the full-attn case)")
        precondition(seqLen > 0, "full attention requires at least one KV position")
        let sc = scale ?? Self.defaultScale(headDim: headDim)

        // B3: single-pass MPP tensor-ops decode attention. Only the production
        // full-attention shape (512/16/2) matches the tensor kernel's
        // compile-time 8-output/64-key/512-dim geometry; SWA layers (256/16/8)
        // and all other shapes keep the tiled split path.
        if Self.decodeTensorOpsEnabled
            && psoDecodeTensorOps != nil
            && headDim == 512
            && numQHeads == 16
            && numKVHeads == 2
            && sc == 1.0 {
            // f16 MPP accumulation order differs from the tiled f32 path, so
            // borderline logits can flip on some prompts (output verified
            // byte-identical on the standard test prompts; env-gated opt-in).
            encodeDecodeTensorOps(commandBuffer: commandBuffer,
                                  q: q, qOffset: qOffset,
                                  k: k, kOffset: kOffset,
                                  v: v, vOffset: vOffset,
                                  out: out, outOffset: outOffset,
                                  headDim: headDim, numQHeads: numQHeads,
                                  numKVHeads: numKVHeads, seqLen: seqLen)
            return
        }

        encodeSplit(commandBuffer: commandBuffer,
                    q: q, qOffset: qOffset, k: k, kOffset: kOffset,
                    v: v, vOffset: vOffset, out: out, outOffset: outOffset,
                    headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads,
                    seqLen: seqLen, kvStart: 0, scale: sc,
                    preferGQASWA: false)
    }


    /// B3 opt-in flag: TURBO_FIELDFARE_ATTN_TENSOROPS=1 routes the full-attention
    /// 512/16/2 decode layers through the single-pass MPP tensor-ops kernel.
    /// `== "1"` so that `=0` is a true opt-out (presence alone is not enough).
    ///
    /// NOTE (2026-08-08): on this SDK the MPP kernel never compiles — the
    /// tensor kernel is guarded by `#if defined(__HAVE_TENSOR__)` in
    /// prefill.metal, and the local CLT SDK has no MetalPerformancePrimitives.h,
    /// so `__HAVE_TENSOR__` stays undefined, `psoDecodeTensorOps == nil`, and
    /// every decode layer falls through to the tiled split path regardless of
    /// this flag. Measured: ATTEN_TENSOROPS=0 vs =1 produce byte-identical
    /// output AND identical GPU time (both are split). The flag is inert here.
    private static var decodeTensorOpsEnabled: Bool {
        ProcessInfo.processInfo.environment["TURBO_FIELDFARE_ATTN_TENSOROPS"] == "1"
    }

    /// Single-pass decode attention for one query token via the MPP tensor-ops
    /// kernel (queryCount = 1, all keys visible). Writes `out` directly — no
    /// partial/combine passes, no split-KV scratch. The K/V pointer convention
    /// (kOffset/vOffset into the layer KV, [pos, kvHead, dim]) matches the
    /// tiled encodeSplit path.
    private func encodeDecodeTensorOps(commandBuffer: MTLCommandBuffer,
                                       q: MTLBuffer, qOffset: Int,
                                       k: MTLBuffer, kOffset: Int,
                                       v: MTLBuffer, vOffset: Int,
                                       out: MTLBuffer, outOffset: Int,
                                       headDim: UInt32, numQHeads: UInt32,
                                       numKVHeads: UInt32, seqLen: UInt32) {
        guard let pso = psoDecodeTensorOps else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var params = PrefillAttentionParams(
            startPosition: seqLen - 1,
            queryCount: 1,
            headDim: headDim,
            numQHeads: numQHeads,
            numKVHeads: numKVHeads,
            kvValidCount: seqLen,
            // The tensor kernel ignores slidingWindow (starts at key 0);
            // full-attention layers have no window.
            slidingWindow: 0,
            kvTokenStrideElements: numKVHeads * headDim,
            qTokenStrideElements: headDim,
            oTokenStrideElements: headDim,
            scale: 1.0)
        enc.setBytes(&params, length: MemoryLayout<PrefillAttentionParams>.stride, index: 4)
        // One threadgroup per KV-head group (8 Q heads each); 128 threads
        // matches the prefill tensor kernel's threadgroup size.
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: Int(numQHeads) / 8, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Two-pass split-KV (Flash-Decoding) dispatch shared by SWA and full
    /// attention — they differ only by `kvStart`. Pass 1 fans the head's
    /// `[kvStart, seqLen)` range across `chunkCount` threadgroups per head;
    /// pass 2 merges the partials. Both encoders go on the same command buffer
    /// so pass 2 hazard-tracks the partial scratch written by pass 1.
    private func encodeSplit(commandBuffer: MTLCommandBuffer,
                             q: MTLBuffer, qOffset: Int,
                             k: MTLBuffer, kOffset: Int,
                             v: MTLBuffer, vOffset: Int,
                             out: MTLBuffer, outOffset: Int,
                             headDim: UInt32, numQHeads: UInt32, numKVHeads: UInt32,
                             seqLen: UInt32, kvStart: UInt32, scale: Float,
                             preferGQASWA: Bool,
                             ringCapacity: UInt32 = 0) {
        precondition(Int(numQHeads) <= Self.maxQHeads,
                     "numQHeads \(numQHeads) exceeds split-KV scratch (max \(Self.maxQHeads))")
        precondition(Int(headDim) <= Self.maxHeadDim,
                     "head_dim \(headDim) exceeds split-KV scratch (max \(Self.maxHeadDim))")
        precondition(ringCapacity == 0 || preferGQASWA,
                     "FP16 KV ring is only valid for SWA attention")
        let geometry = Self.splitGeometry(numQHeads: numQHeads,
                                          numKVHeads: numKVHeads,
                                          headDim: headDim,
                                          seqLen: seqLen,
                                          kvStart: kvStart,
                                          preferGQASWA: preferGQASWA)
        let useSWAGQAPartial = geometry.useSWAGroupedPartial
        let nChunks = geometry.numChunks
        let chunkLen = geometry.chunkLength
        let partialPSO = partialPipeline(headDim: headDim,
                                         numQHeads: numQHeads,
                                         numKVHeads: numKVHeads,
                                         numChunks: nChunks,
                                         useGQAPartial: useSWAGQAPartial,
                                         ringCapacity: ringCapacity)
        let tgWidth = Int(partialPSO.maxTotalThreadsPerThreadgroup)

        guard let p1 = commandBuffer.makeComputeCommandEncoder() else { return }
        p1.setComputePipelineState(partialPSO)
        p1.setBuffer(q, offset: qOffset, index: 0)
        p1.setBuffer(k, offset: kOffset, index: 1)
        p1.setBuffer(v, offset: vOffset, index: 2)
        p1.setBuffer(mPartial, offset: 0, index: 3)
        p1.setBuffer(dPartial, offset: 0, index: 4)
        p1.setBuffer(oPartial, offset: 0, index: 5)
        var hd = headDim, nq = numQHeads, nkv = numKVHeads, sl = seqLen, ks = kvStart
        var cl = UInt32(chunkLen), nc = UInt32(nChunks), sc = scale
        p1.setBytes(&hd,  length: MemoryLayout<UInt32>.size, index: 6)
        p1.setBytes(&nq,  length: MemoryLayout<UInt32>.size, index: 7)
        p1.setBytes(&nkv, length: MemoryLayout<UInt32>.size, index: 8)
        p1.setBytes(&sl,  length: MemoryLayout<UInt32>.size, index: 9)
        p1.setBytes(&ks,  length: MemoryLayout<UInt32>.size, index: 10)
        p1.setBytes(&cl,  length: MemoryLayout<UInt32>.size, index: 11)
        p1.setBytes(&nc,  length: MemoryLayout<UInt32>.size, index: 12)
        p1.setBytes(&sc,  length: MemoryLayout<Float>.size,  index: 13)
        let partialGroups = geometry.partialThreadgroups
        p1.dispatchThreadgroups(MTLSize(width: partialGroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        p1.endEncoding()

        guard let p2 = commandBuffer.makeComputeCommandEncoder() else { return }
        let combinePSO = combinePipeline(headDim: headDim,
                                         numQHeads: numQHeads,
                                         numKVHeads: numKVHeads,
                                         numChunks: nChunks)
        p2.setComputePipelineState(combinePSO)
        p2.setBuffer(mPartial, offset: 0, index: 0)
        p2.setBuffer(dPartial, offset: 0, index: 1)
        p2.setBuffer(oPartial, offset: 0, index: 2)
        p2.setBuffer(out, offset: outOffset, index: 3)
        var hd2 = headDim, nc2 = UInt32(nChunks)
        p2.setBytes(&hd2, length: MemoryLayout<UInt32>.size, index: 4)
        p2.setBytes(&nc2, length: MemoryLayout<UInt32>.size, index: 5)
        let combineTGWidth = min(Self.threadsPerGroup,
                                 Int(combinePSO.maxTotalThreadsPerThreadgroup))
        p2.dispatchThreadgroups(MTLSize(width: Int(numQHeads), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: combineTGWidth, height: 1, depth: 1))
        p2.endEncoding()
    }

    /// `1 / sqrt(head_dim)` — the classic transformer scaling. Used as the
    /// default for non-Gemma callers (and for the existing tests that pre-date
    /// the runtime scale arg).
    static func defaultScale(headDim: UInt32) -> Float {
        Float(1.0) / Float(headDim).squareRoot()
    }

    private static func specializedPipeline(_ context: MetalContext,
                                            _ name: String,
                                            headDim: UInt32,
                                            numQHeads: UInt32,
                                            numKVHeads: UInt32,
                                            numChunks: UInt32? = nil,
                                            ringCapacity: UInt32? = nil) throws -> MTLComputePipelineState {
        var constants = [
            MetalFunctionConstant(index: 60, value: .uint32(headDim)),
            MetalFunctionConstant(index: 61, value: .uint32(numQHeads)),
            MetalFunctionConstant(index: 62, value: .uint32(numKVHeads)),
            MetalFunctionConstant(index: 63, value: .bool(true)),
        ]
        if let numChunks {
            constants.append(MetalFunctionConstant(index: 65, value: .uint32(numChunks)))
        }
        if let ringCapacity {
            constants.append(MetalFunctionConstant(index: 69, value: .uint32(ringCapacity)))
        }
        return try context.pipeline(name, constants: constants)
    }

    private func partialPipeline(headDim: UInt32,
                                 numQHeads: UInt32,
                                 numKVHeads: UInt32,
                                 numChunks: Int,
                                 useGQAPartial: Bool,
                                 ringCapacity: UInt32 = 0) -> MTLComputePipelineState {
        if ringCapacity > 0 {
            let name = useGQAPartial ? "attention_decode_gqa_swa_partial" : "attention_decode_partial"
            let specializedChunks = numChunks == 16 ? Optional(UInt32(numChunks)) : nil
            do {
                return try Self.specializedPipeline(ctx,
                                                    name,
                                                    headDim: headDim,
                                                    numQHeads: numQHeads,
                                                    numKVHeads: numKVHeads,
                                                    numChunks: specializedChunks,
                                                    ringCapacity: ringCapacity)
            } catch {
                preconditionFailure("failed to build FP16 KV ring attention pipeline: \(error)")
            }
        }
        if useGQAPartial && headDim == 256 && numQHeads == 16 && numKVHeads == 8 {
            if numChunks == 16 {
                return psoGQAPartialSWAChunks16
            }
            return psoGQAPartialSWA
        }
        if !useGQAPartial && headDim == 256 && numQHeads == 16 && numKVHeads == 8 {
            return psoPartialSWA
        }
        if useGQAPartial && headDim == 512 && numQHeads == 16 && numKVHeads == 2 {
            // Full attention: 8 Q heads share each KV head. The GQA-full
            // kernel reads each K/V row once per chunk and serves all 8 Q
            // heads from it (the generic kernel would re-read 8x).
            return numChunks == 16 ? psoGQAFullPartialFullChunks16 : psoGQAFullPartialFull
        }
        return useGQAPartial ? psoGQAPartial : psoPartial
    }

    private func combinePipeline(headDim: UInt32,
                                 numQHeads: UInt32,
                                 numKVHeads: UInt32,
                                 numChunks: Int) -> MTLComputePipelineState {
        if headDim == 256 && numQHeads == 16 && numKVHeads == 8 {
            return numChunks == 16 ? psoCombineSWAChunks16 : psoCombineSWA
        }
        if headDim == 512 && numQHeads == 16 && numKVHeads == 2 {
            return numChunks == 16 ? psoCombineFullChunks16 : psoCombineFull
        }
        return psoCombine
    }
}
