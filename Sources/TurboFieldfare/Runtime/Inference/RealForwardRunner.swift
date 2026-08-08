import Foundation
import Metal

public enum RDAdvicePolicyMode: String, Codable, Sendable, Equatable {
    case `default`
    case off
    case bounded
    case adaptive

    public static func parse(_ raw: String?) -> RDAdvicePolicyMode {
        switch raw?.lowercased() {
        case "off", "none", "disabled":
            return .off
        case "bounded":
            return .bounded
        case "adaptive":
            return .adaptive
        default:
            return .default
        }
    }
}

public struct RDAdviceAdaptivePolicyConfig: Sendable, Equatable {
    public var missCap: Int
    public var byteCap: UInt64
    public var slowCallNanos: UInt64

    public init(missCap: Int,
                byteCap: UInt64,
                slowCallNanos: UInt64) {
        self.missCap = missCap
        self.byteCap = byteCap
        self.slowCallNanos = slowCallNanos
    }

    public static let conservative = RDAdviceAdaptivePolicyConfig(
        missCap: 12,
        byteCap: 384 * 1_048_576,
        slowCallNanos: 1_000_000)
}

public struct PrefillTimingBreakdown: Sendable, Equatable {
    public var attentionNanos: UInt64
    public var routerFrontHalfGPUNanos: UInt64
    public var routerReadbackAndCPUNanos: UInt64
    public var sharedExpertNanos: UInt64
    public var streamedRoutedTilesWaitNanos: UInt64
    public var tailReduceLayerTailNanos: UInt64
    public var tilePlanNanos: UInt64
    public var tileFetchBindingNanos: UInt64
    public var tileArgumentBufferNanos: UInt64
    public var tileFetchOpenOrReadNanos: UInt64
    public var tileFetchReadWallNanos: UInt64
    public var tileFetchCacheSlotOverheadNanos: UInt64

    public init(attentionNanos: UInt64 = 0,
                routerFrontHalfGPUNanos: UInt64 = 0,
                routerReadbackAndCPUNanos: UInt64 = 0,
                sharedExpertNanos: UInt64 = 0,
                streamedRoutedTilesWaitNanos: UInt64 = 0,
                tailReduceLayerTailNanos: UInt64 = 0,
                tilePlanNanos: UInt64 = 0,
                tileFetchBindingNanos: UInt64 = 0,
                tileArgumentBufferNanos: UInt64 = 0,
                tileFetchOpenOrReadNanos: UInt64 = 0,
                tileFetchReadWallNanos: UInt64 = 0,
                tileFetchCacheSlotOverheadNanos: UInt64 = 0) {
        self.attentionNanos = attentionNanos
        self.routerFrontHalfGPUNanos = routerFrontHalfGPUNanos
        self.routerReadbackAndCPUNanos = routerReadbackAndCPUNanos
        self.sharedExpertNanos = sharedExpertNanos
        self.streamedRoutedTilesWaitNanos = streamedRoutedTilesWaitNanos
        self.tailReduceLayerTailNanos = tailReduceLayerTailNanos
        self.tilePlanNanos = tilePlanNanos
        self.tileFetchBindingNanos = tileFetchBindingNanos
        self.tileArgumentBufferNanos = tileArgumentBufferNanos
        self.tileFetchOpenOrReadNanos = tileFetchOpenOrReadNanos
        self.tileFetchReadWallNanos = tileFetchReadWallNanos
        self.tileFetchCacheSlotOverheadNanos = tileFetchCacheSlotOverheadNanos
    }

    public static let zero = PrefillTimingBreakdown()

    public func delta(since baseline: PrefillTimingBreakdown) -> PrefillTimingBreakdown {
        PrefillTimingBreakdown(
            attentionNanos: attentionNanos &- baseline.attentionNanos,
            routerFrontHalfGPUNanos: routerFrontHalfGPUNanos &- baseline.routerFrontHalfGPUNanos,
            routerReadbackAndCPUNanos: routerReadbackAndCPUNanos &- baseline.routerReadbackAndCPUNanos,
            sharedExpertNanos: sharedExpertNanos &- baseline.sharedExpertNanos,
            streamedRoutedTilesWaitNanos: streamedRoutedTilesWaitNanos &- baseline.streamedRoutedTilesWaitNanos,
            tailReduceLayerTailNanos: tailReduceLayerTailNanos &- baseline.tailReduceLayerTailNanos,
            tilePlanNanos: tilePlanNanos &- baseline.tilePlanNanos,
            tileFetchBindingNanos: tileFetchBindingNanos &- baseline.tileFetchBindingNanos,
            tileArgumentBufferNanos: tileArgumentBufferNanos &- baseline.tileArgumentBufferNanos,
            tileFetchOpenOrReadNanos: tileFetchOpenOrReadNanos &- baseline.tileFetchOpenOrReadNanos,
            tileFetchReadWallNanos: tileFetchReadWallNanos &- baseline.tileFetchReadWallNanos,
            tileFetchCacheSlotOverheadNanos: tileFetchCacheSlotOverheadNanos &- baseline.tileFetchCacheSlotOverheadNanos)
    }
}

public struct PrefillExpertAccessPattern: Sendable, Equatable {
    public var tileAccesses: UInt64
    public var expertReferences: UInt64
    public var uniqueExperts: UInt64
    public var consecutiveTileComparisons: UInt64
    public var consecutiveTileOverlapExperts: UInt64
    public var consecutiveTileRequestedExperts: UInt64
    public var contiguousNeighborPairs: UInt64
    public var contiguousNeighborOpportunities: UInt64
    public var contiguousRunCount: UInt64
    public var expertsInContiguousRuns: UInt64
    public var maxContiguousRunLength: UInt64

    public init(tileAccesses: UInt64 = 0,
                expertReferences: UInt64 = 0,
                uniqueExperts: UInt64 = 0,
                consecutiveTileComparisons: UInt64 = 0,
                consecutiveTileOverlapExperts: UInt64 = 0,
                consecutiveTileRequestedExperts: UInt64 = 0,
                contiguousNeighborPairs: UInt64 = 0,
                contiguousNeighborOpportunities: UInt64 = 0,
                contiguousRunCount: UInt64 = 0,
                expertsInContiguousRuns: UInt64 = 0,
                maxContiguousRunLength: UInt64 = 0) {
        self.tileAccesses = tileAccesses
        self.expertReferences = expertReferences
        self.uniqueExperts = uniqueExperts
        self.consecutiveTileComparisons = consecutiveTileComparisons
        self.consecutiveTileOverlapExperts = consecutiveTileOverlapExperts
        self.consecutiveTileRequestedExperts = consecutiveTileRequestedExperts
        self.contiguousNeighborPairs = contiguousNeighborPairs
        self.contiguousNeighborOpportunities = contiguousNeighborOpportunities
        self.contiguousRunCount = contiguousRunCount
        self.expertsInContiguousRuns = expertsInContiguousRuns
        self.maxContiguousRunLength = maxContiguousRunLength
    }

    public static let zero = PrefillExpertAccessPattern()
}

struct RDAdviceAdaptivePolicyState: Sendable, Equatable {
    var config: RDAdviceAdaptivePolicyConfig
    private var skipUntilPosition: Int = -1
    private(set) var recentSlowCallNanos: UInt64 = 0

    init(config: RDAdviceAdaptivePolicyConfig = .conservative) {
        self.config = config
    }

    mutating func reset() {
        skipUntilPosition = -1
        recentSlowCallNanos = 0
    }

    func shouldSkip(position: Int,
                    requestedMisses: Int,
                    estimatedBytes: UInt64,
                    canOverlapUsefulGPUWork: Bool) -> Bool {
        position <= skipUntilPosition ||
        !canOverlapUsefulGPUWork ||
        requestedMisses > config.missCap ||
        estimatedBytes > config.byteCap
    }

    mutating func update(after result: ExpertIOAdviceResult,
                                position: Int) {
        recentSlowCallNanos = max(recentSlowCallNanos, result.maxCallNanos)
        if result.maxCallNanos >= config.slowCallNanos {
            skipUntilPosition = max(skipUntilPosition, position)
        }
    }
}

/// Gemma 4 real-forward decode pass.
///
/// Composes the production kernels against the `.gturbo` model:
///
///   embed_lookup_int4(token) * sqrt(H)
///   for L in 0..<30:
///     a = rmsnorm_bf16w(h, input_layernorm)
///     Q = q_proj(a)    K = k_proj(a)    V = (SWA) v_proj(a) | (full) k_proj(a)
///     per-head q/k_norm (bf16w), per-head v_norm (no_scale)
///     NeoX RoPE on Q + K (default for SWA, proportional for full)
///     write K and V into separate cache slots
///     attn = attention(scale=1.0, SWA window or full causal)
///     attn = o_proj(attn)
///     h = h + rmsnorm_bf16w(attn, post_attention_layernorm)
///     h1 = rmsnorm_bf16w(h, pre_feedforward_layernorm)
///     h1 = SharedExpertInt8(h1)
///     h1 = rmsnorm_bf16w(h1, post_feedforward_layernorm_1)
///     // router + routed branch
///     xr   = rmsnorm_no_scale(h)
///     idx, w = router_topk_gemma4(xr, effective_scale[L], per_expert_scale[L])
///     h2 = rmsnorm_bf16w(h, pre_feedforward_layernorm_2)
///     h2 = moe_fused_ffn_streamed_routed(h2, residual=0, routedBlobs=fetch(idx), w)
///     h2 = rmsnorm_bf16w(h2, post_feedforward_layernorm_2)
///     h = h + rmsnorm_bf16w(h1 + h2, post_feedforward_layernorm)
///     h = h * layer_scalar[L]
///   logits = DequantInt4GEMV(rmsnorm_bf16w(h, model.norm), embed_table^T)
///   // final softcap and softmax happen in the Sampler.
///
/// Direct against `Model`; this is the only production decode forward path.
internal enum PrefillProjectionFamily: Sendable, Equatable {
    case q
    case kv
    case o
    case shared
    case routed
}

internal enum PrefillProjectionDispatch: Sendable, Equatable {
    case repeatedGEMV
    case qmm
}

internal enum PrefillProjectionDispatchPolicy {
    static func selectedDispatch(for family: PrefillProjectionFamily,
                                 chunkTokens: Int) -> PrefillProjectionDispatch {
        guard chunkTokens >= 32 else {
            return .repeatedGEMV
        }
        switch family {
        case .q:
            return .repeatedGEMV
        case .kv, .o, .shared, .routed:
            return .qmm
        }
    }
}

public final class RealForwardRunner: ChunkedPrefillRunner, ContextWindowReporting, ContinuableLogitProducer, RequestAwareLogitProducer, ExpertCacheTelemetryReporting, TargetTokenEmbeddingProvider, @unchecked Sendable {
    private struct LayerSharedExpertProjections {
        let gate: SharedExpertInt8Proj
        let up: SharedExpertInt8Proj
        let down: SharedExpertInt8Proj
        let postF1: TensorView
    }

    private struct PrefillExpertAccessPatternTracker {
        private struct ExpertPairKey: Hashable {
            let first: Int
            let second: Int
        }

        private struct LayerTraceState {
            var tileAccesses: UInt64 = 0
            var expertCounts: [Int: UInt64] = [:]
            var coAccessPairs: [ExpertPairKey: UInt64] = [:]
        }

        private(set) var metrics = PrefillExpertAccessPattern.zero
        private var seenExperts = Set<UInt64>()
        private var lastTileExpertsByLayer: [Int: Set<Int>] = [:]
        private var layerTraceStates: [Int: LayerTraceState] = [:]

        mutating func reset() {
            metrics = .zero
            seenExperts.removeAll(keepingCapacity: true)
            lastTileExpertsByLayer.removeAll(keepingCapacity: true)
            layerTraceStates.removeAll(keepingCapacity: true)
        }

        mutating func record(layer: Int,
                             expertIDs: [Int],
                             physicalOffsets: [UInt64],
                             expertStride: UInt64) {
            let validExpertIDs = expertIDs.filter { $0 >= 0 && $0 < physicalOffsets.count }
            guard !validExpertIDs.isEmpty else { return }

            metrics.tileAccesses &+= 1
            metrics.expertReferences &+= UInt64(validExpertIDs.count)

            let currentExperts = Set(validExpertIDs)
            var layerTrace = layerTraceStates[layer] ?? LayerTraceState()
            layerTrace.tileAccesses &+= 1
            for expertID in currentExperts {
                layerTrace.expertCounts[expertID, default: 0] &+= 1
            }
            let sortedExperts = currentExperts.sorted()
            if sortedExperts.count >= 2 {
                for firstIndex in 0..<(sortedExperts.count - 1) {
                    let first = sortedExperts[firstIndex]
                    for second in sortedExperts[(firstIndex + 1)...] {
                        let key = ExpertPairKey(first: first, second: second)
                        layerTrace.coAccessPairs[key, default: 0] &+= 1
                    }
                }
            }
            layerTraceStates[layer] = layerTrace

            for expertID in currentExperts {
                seenExperts.insert((UInt64(layer) << 32) | UInt64(expertID))
            }
            metrics.uniqueExperts = UInt64(seenExperts.count)

            if let previousExperts = lastTileExpertsByLayer[layer], !previousExperts.isEmpty {
                metrics.consecutiveTileComparisons &+= 1
                metrics.consecutiveTileOverlapExperts &+= UInt64(currentExperts.intersection(previousExperts).count)
                metrics.consecutiveTileRequestedExperts &+= UInt64(currentExperts.count)
            }
            lastTileExpertsByLayer[layer] = currentExperts

            let sortedOffsets = currentExperts
                .map { (expertID: $0, offset: physicalOffsets[$0]) }
                .sorted { lhs, rhs in lhs.offset < rhs.offset }
            guard sortedOffsets.count >= 2 else { return }

            metrics.contiguousNeighborOpportunities &+= UInt64(sortedOffsets.count - 1)
            var currentRunLength = 1
            for index in 1..<sortedOffsets.count {
                let previous = sortedOffsets[index - 1]
                let current = sortedOffsets[index]
                if current.offset == previous.offset &+ expertStride {
                    metrics.contiguousNeighborPairs &+= 1
                    currentRunLength += 1
                } else if currentRunLength > 1 {
                    metrics.contiguousRunCount &+= 1
                    metrics.expertsInContiguousRuns &+= UInt64(currentRunLength)
                    metrics.maxContiguousRunLength = max(metrics.maxContiguousRunLength,
                                                         UInt64(currentRunLength))
                    currentRunLength = 1
                } else {
                    currentRunLength = 1
                }
            }
            if currentRunLength > 1 {
                metrics.contiguousRunCount &+= 1
                metrics.expertsInContiguousRuns &+= UInt64(currentRunLength)
                metrics.maxContiguousRunLength = max(metrics.maxContiguousRunLength,
                                                     UInt64(currentRunLength))
            }
        }

        var trace: PrefillExpertTrace {
            PrefillExpertTrace(
                layers: layerTraceStates.keys.sorted().map { layer in
                    let state = layerTraceStates[layer] ?? LayerTraceState()
                    return PrefillExpertLayerTrace(
                        layer: layer,
                        tileAccesses: state.tileAccesses,
                        uniqueExperts: UInt64(state.expertCounts.count),
                        expertCounts: state.expertCounts
                            .map { PrefillExpertCountEntry(expert: $0.key, count: $0.value) }
                            .sorted { lhs, rhs in
                                if lhs.count != rhs.count { return lhs.count > rhs.count }
                                return lhs.expert < rhs.expert
                            },
                        coAccessPairs: state.coAccessPairs
                            .map {
                                PrefillExpertPairCountEntry(
                                    first: $0.key.first,
                                    second: $0.key.second,
                                    count: $0.value)
                            }
                            .sorted { lhs, rhs in
                                if lhs.count != rhs.count { return lhs.count > rhs.count }
                                if lhs.first != rhs.first { return lhs.first < rhs.first }
                                return lhs.second < rhs.second
                            })
                })
        }
    }

    private let model: Model
    private let ctx: MetalContext
    private let kv: KVCacheManager?
    private let cfg: ArchConfig

    // Kernels
    private let embedInt4: EmbedLookupInt4
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let attention: Attention
    private let shared: SharedExpertRuntime
    private let moe: MoE
    private let fusionHead: LMHeadChainInt4
    private let fusedQKVGEMV: FusedQKVGEMV
    private let fusedQKVEpilogue: FusedQKVEpilogue
    private let fusedPostAttentionSetup: FusedPostAttentionSetup
    private let fusedTail: FusedLayerTail

    // Prefill kernels. These are initialized once per runner so the chunk path
    // cannot accidentally rebuild PSOs inside a per-layer loop.
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let prefillRMS: PrefillRMSNorm
    private let prefillQMM: PrefillInt4QMM
    private let prefillMPPAffineInt4: MPPPrefillInt4QMM?
    private let prefillQKVEpilogue: PrefillQKVEpilogue
    private let prefillAttention: PrefillAttention
    private let prefillPostAttention: PrefillPostAttentionSetup
    private let prefillRouter: PrefillRouter
    private let prefillSharedExpert: PrefillSharedExpert
    private let prefillGroupedMoE: PrefillGroupedRoutedMoE
    private let prefillMoE: PrefillMoE
    private let prefillLayerTail: PrefillLayerTail
    private let prefillFinalRowHead: PrefillFinalRowHeadInt4

    // Scratch — preallocated per spec'd D / F / vocab.
    private let hidden: MTLBuffer        // [D] FP16
    private let normed: MTLBuffer        // [D] FP16
    private let attnOut: MTLBuffer       // [N_HEADS * head_dim] FP16
    private let qScratch: MTLBuffer      // [N_HEADS * head_dim] FP16
    private let kStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    private let vStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    private let oOut: MTLBuffer          // [D] FP16
    private let h1Buf: MTLBuffer         // [D] FP16 (dense MLP output)
    private let h2Buf: MTLBuffer         // [D] FP16 (routed output)
    private let routedX: MTLBuffer       // [D] FP16 (pre_feedforward_layernorm_2 output)
    private let denseX: MTLBuffer        // [D] FP16 (pre_feedforward_layernorm output)
    private let denseScratchGate: MTLBuffer // [F=2112] FP16
    private let denseScratchUp: MTLBuffer   // [F=2112] FP16
    private let denseScratchAct: MTLBuffer  // [F=2112] FP16
    private let routerInput: MTLBuffer   // [D] FP16 (rmsnorm_no_scale(h))
    private let zeroResidual: MTLBuffer  // [D] FP16 zeros — for routed branch base
    private let outIndices: MTLBuffer    // [topK] UInt32
    private let outWeights: MTLBuffer    // [topK] FP16
    /// Speculative next-layer routing, written by running layer `L+1`'s router
    /// against layer `L`'s post-attention hidden state. Expert I/O is only
    /// issuable once the real router has run, which puts a full SSD round trip
    /// on the critical path; if this cheap extrapolation is accurate enough the
    /// reads can be started a whole layer early and overlapped with GPU work.
    private let predIndices: MTLBuffer   // [topK] UInt32
    private let predWeights: MTLBuffer   // [topK] FP16
    // Persistent MoE scratch, allocated once; about 56 KiB at production shape.
    private let moeActs: MTLBuffer       // [topK * FmoE] FP16
    private let moeHitActiveSlots: MTLBuffer // [topK] UInt32
    private let moeMissActiveSlots: MTLBuffer // [topK] UInt32
    private let greedyTokenBuf: MTLBuffer // 4 B UInt32 fused-head output
    /// `[maxPerRowGreedyRows]` UInt32 — one greedy argmax per row of a prefill
    /// chunk, used by batched speculative verify.
    private let greedyRowTokensBuf: MTLBuffer
    /// Non-zero while a `verifyBatch` call is in flight; tells the final-head
    /// stage to emit argmax for every row instead of only the last one.
    private var perRowGreedyRowCount: Int = 0
    private var prefillChunkState = PrefillChunkCommitState()
    private var prefillScratch: PrefillChunkScratchBuffers?

    /// Upper bound on rows a single `verifyBatch` can score. Sized to clear the
    /// CLI's 32-draft ceiling plus the current token; the readback buffer is
    /// 256 bytes, so the cap exists to make the bound explicit rather than to
    /// save memory.
    public static let maxPerRowGreedyRows = 64

    private static let rdadviseBoundedMissCap = 12
    private static let rdadviseBoundedMaxCallNanos: UInt64 = 250_000
    private static let rdadviseAdaptiveMissCap = 12
    private static let rdadviseAdaptiveByteCap: UInt64 = 384 * 1_048_576
    private static let rdadviseAdaptiveSlowCallNanos: UInt64 = 1_000_000
    private static let prefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig()

    /// Per-layer `router.scale * D^-0.5` pre-folded into one BF16 buffer
    /// allocation per layer. ~168 KB total at 30 layers × 2816 BF16 — bounded
    /// host work done once at init.
    private let effectiveScaleBuffers: [MTLBuffer]
    private let sharedExpertProjections: [LayerSharedExpertProjections]

    public let maxContext: Int

    /// Per-instance head and RDADVISE modes. The fused head (default) skips the
    /// 512 KB logits write and leaves a greedy argmax in `lastGreedyToken`;
    /// callers that sample from the logits buffer (non-greedy configs) must pass
    /// `forceLogitsHead: true` or they read a never-written buffer.
    private let useFusedGreedyHead: Bool
    private let prefillAttentionPath: RuntimePrefillAttentionPath
    public let rdadviseEnabled: Bool
    public let rdadvisePolicyMode: RDAdvicePolicyMode
    private var rdadviseSkipUntilPosition: Int = -1
    private var rdadviseAdaptiveState: RDAdviceAdaptivePolicyState
    private var rdadviseAdaptivePosition: Int = -1
    private var rdadviseAdaptivePositionBytes: UInt64 = 0
    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        self.model = model
        self.ctx = context
        self.cfg = model.config
        self.maxContext = maxContext
        self.useFusedGreedyHead = runtimeConfiguration.headPath == .fusedRows
        self.prefillAttentionPath = runtimeConfiguration.prefillAttentionPath
        let useFP16Ring = runtimeConfiguration.fp16RingEnabled
        self.rdadvisePolicyMode = runtimeConfiguration.rdadvisePolicy
        self.rdadviseAdaptiveState = RDAdviceAdaptivePolicyState(
            config: RDAdviceAdaptivePolicyConfig(
                missCap: Self.rdadviseAdaptiveMissCap,
                byteCap: Self.rdadviseAdaptiveByteCap,
                slowCallNanos: Self.rdadviseAdaptiveSlowCallNanos))
        self.rdadviseEnabled = runtimeConfiguration.rdadviseEnabled
        self.kv = try KVCacheManager(device: context.device,
                                     config: cfg,
                                     maxContext: maxContext,
                                     fp16RingEnabled: useFP16Ring,
                                     slidingWindow: cfg.slidingWindow,
                                     maxPrefillChunkTokens: PrefillRuntimeConfig.maxChunkTokens)

        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.rms       = try RMSNorm(context: context)
        self.int4      = try DequantInt4GEMV(context: context)
        self.attention = try Attention(context: context)
        self.shared    = try SharedExpertRuntime(context: context,
                                                  weightBits: model.sharedExpertWeightBits)
        self.moe       = try MoE(context: context)
        self.fusionHead = try LMHeadChainInt4(context: context,
                                              maxD: cfg.hiddenSize,
                                              maxVocab: cfg.vocabSize)
        self.fusedQKVGEMV = try FusedQKVGEMV(context: context)
        self.fusedQKVEpilogue = try FusedQKVEpilogue(context: context)
        self.fusedPostAttentionSetup = try FusedPostAttentionSetup(context: context)
        self.fusedTail = try FusedLayerTail(context: context)
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillQMM = try PrefillInt4QMM(context: context)
        self.prefillMPPAffineInt4 = MPPPrefillInt4QMM(context: context)
        self.prefillQKVEpilogue = try PrefillQKVEpilogue(context: context)
        self.prefillAttention = try PrefillAttention(context: context)
        self.prefillPostAttention = try PrefillPostAttentionSetup(context: context)
        self.prefillRouter = try PrefillRouter(context: context)
        self.prefillSharedExpert = try PrefillSharedExpert(
            context: context,
            weightBits: model.sharedExpertWeightBits)
        self.prefillGroupedMoE = try PrefillGroupedRoutedMoE(context: context)
        self.prefillMoE = try PrefillMoE(context: context)
        self.prefillLayerTail = try PrefillLayerTail(context: context)
        self.prefillFinalRowHead = try PrefillFinalRowHeadInt4(context: context,
                                                               maxD: cfg.hiddenSize)

        let device = context.device
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        let maxQ = cfg.numHeads * max(cfg.headDim, cfg.fullHeadDim)

        func buf(_ count: Int, _ stride: Int = MemoryLayout<Float16>.size) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(count, 1) * stride,
                                            options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return b
        }
        self.hidden        = try buf(D)
        self.normed        = try buf(D)
        self.attnOut       = try buf(maxQ)
        self.qScratch      = try buf(maxQ)
        self.kStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim))
        self.vStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim))
        self.oOut          = try buf(D)
        self.h1Buf         = try buf(D)
        self.h2Buf         = try buf(D)
        self.routedX       = try buf(D)
        self.denseX        = try buf(D)
        self.denseScratchGate = try buf(F)
        self.denseScratchUp   = try buf(F)
        self.denseScratchAct  = try buf(F)
        self.routerInput   = try buf(D)
        self.zeroResidual  = try buf(D)
        // The routed MoE kernel seeds y[d] = residual[d]; pinning this buffer
        // to zero once at init makes the routed branch's residual contribution
        // exactly zero (it's combined with the dense MLP downstream).
        memset(self.zeroResidual.contents(), 0, self.zeroResidual.length)
        self.outIndices    = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.outWeights    = try buf(cfg.topKExperts)
        self.predIndices   = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.predWeights   = try buf(cfg.topKExperts)
        self.moeActs       = try buf(cfg.topKExperts * cfg.moeIntermediateSize)
        self.moeHitActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.moeMissActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        guard let tok = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                          options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        self.greedyTokenBuf = tok
        guard let rowToks = device.makeBuffer(
                  length: Self.maxPerRowGreedyRows * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        rowToks.label = "head.greedyRowTokens"
        self.greedyRowTokensBuf = rowToks

        func sharedProj(_ view: TensorView, rows: UInt32, cols: UInt32) -> SharedExpertProjection {
            SharedExpertProjection(weights: view.buffer,
                                 scales: view.buffer,
                                 biases: view.buffer,
                                 weightsOffset: Int(view.offset),
                                 scalesOffset: Int(view.scaleOffset),
                                 biasesOffset: Int(view.biasOffset),
                                 rows: rows,
                                 cols: cols)
        }
        var sharedViews: [LayerSharedExpertProjections] = []
        sharedViews.reserveCapacity(cfg.numLayers)
        for L in 0..<cfg.numLayers {
            let gate = try model.sharedExpertGate(layer: L)
            let up = try model.sharedExpertUp(layer: L)
            let down = try model.sharedExpertDown(layer: L)
            sharedViews.append(LayerSharedExpertProjections(
                gate: sharedProj(gate, rows: UInt32(F), cols: UInt32(D)),
                up: sharedProj(up, rows: UInt32(F), cols: UInt32(D)),
                down: sharedProj(down, rows: UInt32(D), cols: UInt32(F)),
                postF1: try model.postFFN1(layer: L)))
        }
        self.sharedExpertProjections = sharedViews

        // Pre-fold 1/sqrt(D) into router.scale per layer. Each layer gets its
        // own BF16 [D] buffer — the kernel reads `effective_scale[i]` and we
        // pay for the multiply once per generation, not per token.
        var perLayer: [MTLBuffer] = []
        perLayer.reserveCapacity(cfg.numLayers)
        let invSqrtD = Float(1.0) / Float(D).squareRoot()
        let dInts = D
        for L in 0..<cfg.numLayers {
            let scaleView = try model.routerScale(layer: L)
            guard let buf = device.makeBuffer(length: dInts * MemoryLayout<UInt16>.size,
                                              options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            let src = scaleView.buffer.contents()
                .advanced(by: Int(scaleView.offset))
                .assumingMemoryBound(to: UInt16.self)
            let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<dInts {
                let v = Quantization.bf16ToFloat(src[i]) * invSqrtD
                dst[i] = Quantization.bf16Bits(v)
            }
            buf.label = "effective_scale.L\(L)"
            perLayer.append(buf)
        }
        self.effectiveScaleBuffers = perLayer
    }

    public func reset() {
        kv?.reset()
        resetTransientState()
    }

    public var continuationPosition: Int {
        kv?.position ?? 0
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard let kv else {
            throw PrefillError.prefillCursorMismatch(
                "continuation requires an initialized KV cache")
        }
        guard expectedPosition > 0, kv.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected KV position \(expectedPosition), current \(kv.position)")
        }
        resetTransientState()
    }

    private func resetTransientState() {
        prefillChunkState.reset()
        rdadviseSkipUntilPosition = -1
        rdadviseAdaptiveState.reset()
        rdadviseAdaptivePosition = -1
        rdadviseAdaptivePositionBytes = 0
        currentDecodeStepIndex = nil
        currentPrefillExpertAccessPattern.reset()
    }

    public func beginRequest() {
        currentRequestID = UInt64.random(in: UInt64.min ... UInt64.max)
        currentDecodeStepIndex = nil
        isInDecodePhase = false
        currentPrefillExpertAccessPattern.reset()
    }

    public func beginPrefillPhase() {
        isInDecodePhase = false
    }

    public func beginDecodePhase() {
        isInDecodePhase = true
    }

    public func setDecodeStep(index: Int) {
        currentDecodeStepIndex = index
    }

    private func makePrefillAccessContext() -> ExpertCacheAccessContext {
        ExpertCacheAccessContext(ownerPhase: .prefillTransient,
                                 controlPlane: .prefill,
                                 requestID: currentRequestID,
                                 decodeStepIndex: nil)
    }

    private func makeDecodeAccessContext() -> ExpertCacheAccessContext {
        if isInDecodePhase {
            return ExpertCacheAccessContext(ownerPhase: .decodeProtected,
                                            controlPlane: .decode,
                                            requestID: currentRequestID,
                                            decodeStepIndex: currentDecodeStepIndex)
        }
        return makePrefillAccessContext()
    }

    /// Enables the next-layer routing extrapolation. Prefetching implies it,
    /// since the prediction is what gets prefetched.
    let expertLookaheadEnabled =
        ProcessInfo.processInfo.environment["TURBO_FIELDFARE_EXPERT_LOOKAHEAD"] != nil
        || ProcessInfo.processInfo.environment["TURBO_FIELDFARE_EXPERT_PREFETCH"] != nil
        || ProcessInfo.processInfo.environment["TURBO_FIELDFARE_MISS_PREFETCH"] != nil
    /// Predicted expert set awaiting comparison against the real router.
    private var pendingLookahead: (layer: Int, experts: [Int])?
    /// Predicted experts that turned out to be in the real top-K.
    public private(set) var lookaheadHits: UInt64 = 0
    /// Total predicted slots compared (`topK` per evaluated layer).
    public private(set) var lookaheadPredictions: UInt64 = 0
    /// Issues speculative reads for the predicted set instead of only scoring
    /// it. Separate from the probe so accuracy can be measured without paying
    /// for wasted I/O.
    let expertPrefetchEnabled = ProcessInfo.processInfo
        .environment["TURBO_FIELDFARE_EXPERT_PREFETCH"] != nil
    /// MISS-ONLY speculative prefetch (2026-08-08): reads the NEXT layer's
    /// predicted experts that are NOT pool members, dispatched right after the
    /// prediction is computed (readback, i.e. the cb1+shared+routed GPU window)
    /// instead of after the fetch. The pool filter fires the background read
    /// only on the few % of layers whose prediction contains long-tail experts
    /// — no per-layer lock churn (the earlier EXPERT_PREFETCH -41% root cause)
    /// and no redundant pool I/O (planExpertsCached skips residents). The next
    /// layer's plan finds the prefetched expert as a hit, removing the
    /// phase1Hit-after-miss exposure.
    let missPrefetchEnabled = ProcessInfo.processInfo
        .environment["TURBO_FIELDFARE_MISS_PREFETCH"] != nil
    /// Static per-layer pool membership for the lock-free filter.
    private lazy var hotPoolSets: [Set<Int>] = model.hotPoolExpertSets()
    /// Layers where a non-pool prediction triggered a speculative read.
    public private(set) var missPrefetchDispatches: UInt64 = 0
    /// Experts (per layer) that have actually missed in this run. The prefetch
    /// filter requires membership here: first-time long-tail misses stay real
    /// (unavoidable), but their RECURRENCES (the LRU-thrash source) get read
    /// ahead during the GPU window.
    private var knownMissersByLayer: [Int: Set<Int>] = [:]
    /// In-flight speculative read. Must be drained before the next layer plans
    /// its own fetch, otherwise two planners could hand the same free slot to
    /// different experts.
    private var expertPrefetchTask: Task<Int?, Never>?

    /// A one-shot speculative read parked on a plain GCD queue.
    ///
    /// `Task.detached` + `await task.value` pays a cooperative-executor
    /// suspension on *every* drain, even when the read finished a whole layer
    /// ago and returned without touching the disk (the common case at a 96%
    /// cache hit rate). At 30 drains per token that overhead lands directly on
    /// the GPU's critical path — measured at 0.86s of a 10s decode. A
    /// semaphore that is already signalled drains for free.
    final class ExpertPrefetchHandle: @unchecked Sendable {
        private let done = DispatchSemaphore(value: 0)
        private var readCount: Int?
        func finish(_ count: Int?) {
            readCount = count
            done.signal()
        }
        /// Blocks until the read finishes; returns how many experts it had to
        /// read, or nil when everything was already resident.
        func drain() -> Int? {
            done.wait()
            return readCount
        }
    }
    /// Set to "0" to fall back to the `Task.detached` drain for A/B runs.
    let syncPrefetchDrainEnabled =
        ProcessInfo.processInfo.environment["TURBO_FIELDFARE_SYNC_PREFETCH"] != "0"
    private let expertPrefetchQueue = DispatchQueue(
        label: "turbofieldfare.expert-prefetch", qos: .userInitiated)
    private var expertPrefetchHandle: ExpertPrefetchHandle?
    /// Experts actually read speculatively (misses only).
    public private(set) var prefetchedExperts: UInt64 = 0
    /// Wall time the critical path spent waiting on an unfinished prefetch.
    public private(set) var prefetchDrainNanos: UInt64 = 0

    /// Sums `gpuEndTime - gpuStartTime` per command buffer. CPU-side wall time
    /// cannot separate "GPU is busy" from "CPU is blocked waiting"; this can,
    /// and it is what decides whether the engine is I/O bound or compute bound.
    /// Commits the dense/shared FFN command buffer *before* the CPU blocks on
    /// the attention buffer. The shared MLP reads only `denseX` (written on-GPU
    /// by cb1's post-attention setup) and resident weights, so none of it needs
    /// a CPU readback. Queuing it early lets the GPU roll straight from cb1
    /// into the shared FFN while the CPU is still reading router indices,
    /// draining prefetch and planning routed experts. GPU submission order is
    /// unchanged. Set to "0" to restore the old late commit for A/B runs.
    /// B4: hit-only synchronous fetch fast path (TURBO_FIELDFARE_B4_HIT_ONLY_SYNC=1).
    /// With the hot pool at ~99% hit, every layer's fetch is pure CPU bookkeeping;
    /// the async path's continuation + dispatch hop shows up as a ~1.1ms GPU-idle
    /// gap after sharedFFN (3800 gaps / 4.26s per 128-token run, 2026-08-07 trace).
    /// B4: hit-only synchronous fetch fast path, DEFAULT ON. With the hot pool
    /// at ~99% hit, every layer's fetch is pure CPU bookkeeping; the async
    /// path's continuation + dispatch hop shows up as a ~1.1ms GPU-idle gap
    /// after sharedFFN (3800 gaps / 4.26s per 128-token run, 2026-08-07 trace).
    /// Same work, fewer hops, byte-identical output (md5-verified). Measured
    /// neutral on throughput (+0.8% noise) but strictly simpler; kept on.
    /// Opt out with TURBO_FIELDFARE_B4_HIT_ONLY_SYNC=0. NOTE: under
    /// TURBO_FIELDFARE_HOT_POOL_PRELOAD=async the sync path can block the
    /// decode thread briefly on cacheLock while the pool task publishes slot
    /// metadata; prod default is sync preload so this is latent.
    let b4HitOnlySyncEnabled = ProcessInfo.processInfo
        .environment["TURBO_FIELDFARE_B4_HIT_ONLY_SYNC"] != "0"
    let earlySharedCommitEnabled =
        ProcessInfo.processInfo.environment["TURBO_FIELDFARE_EARLY_SHARED"] != "0"
    /// EXPERIMENTAL (B1): encode the shared FFN into cb1 itself, removing one
    /// command-buffer commit per layer. FINAL VERDICT (2026-08-07, clean disk,
    /// strict interleaved 6-round A/B at 64 slots): a confirmed regression —
    /// split median 13.0 vs fused 8.3 tok/s on r3 (+57% split), 15.9 vs 8.5 on
    /// r4 (+88% split). The earlier "+25/+51%" reading was page-cache
    /// contamination (split ran right before fused and warmed the cache). The
    /// early-split path hides the shared GPU time under the CPU's
    /// router-readback window; fusion serialises it back onto the critical
    /// wait. Kept opt-in (TURBO_FIELDFARE_FUSE_SHARED=1) for future hardware
    /// re-checks; default stays the early split.
    let fuseSharedIntoCB1 =
        ProcessInfo.processInfo.environment["TURBO_FIELDFARE_FUSE_SHARED"] == "1"
    let gpuTimingEnabled = ProcessInfo.processInfo
        .environment["TURBO_FIELDFARE_GPU_TIMING"] != nil

    /// Per-batch-size expert-union coverage stats (TURBO_FIELDFARE_UNION_STATS=1):
    /// for each routed batch, the per-layer union of routed experts vs the
    /// hot-pool resident set. unionPool/unionTotal = union coverage;
    /// perTokenPool/perTokenTotal = per-token coverage (single-token baseline).
    public struct UnionStatsBucket {
        public var visits = 0
        public var unionTotal = 0
        public var unionPool = 0
        public var perTokenTotal = 0
        public var perTokenPool = 0
        public init() {}
    }
    let unionStatsEnabled = ProcessInfo.processInfo
        .environment["TURBO_FIELDFARE_UNION_STATS"] == "1"
    public private(set) var unionStats: [Int: UnionStatsBucket] = [:]
    public let unionDumpPath = ProcessInfo.processInfo
        .environment["TURBO_FIELDFARE_UNION_DUMP"]
    public private(set) var unionDumpLines: [String] = []
    /// Attention + QKV + o_proj + router (the pre-routing command buffer).
    public private(set) var gpuAttentionNanos: UInt64 = 0
    /// MoE phase1-miss + phase2 reduce + tail (the post-routing buffer).
    public private(set) var gpuRoutedNanos: UInt64 = 0
    /// Dense/shared FFN, issued so it overlaps the expert read.
    public private(set) var gpuSharedNanos: UInt64 = 0
    /// MoE phase1 restricted to experts already resident in the slot cache.
    public private(set) var gpuPhase1HitNanos: UInt64 = 0
    /// Final norm + LM head (or the fused greedy head).
    public private(set) var gpuHeadNanos: UInt64 = 0

    /// Wall time the CPU spends blocked in `waitUntilCompleted` on the
    /// pre-routing buffer. Compare against `gpuAttentionNanos`: the difference
    /// is pure submit/schedule/wake round-trip, which no kernel optimisation
    /// can remove — only issuing fewer, larger command buffers can.
    public private(set) var cb1WaitNanos: UInt64 = 0
    /// B4: per-cb1 CPU-side latency decomposition (GPU_TIMING only):
    /// schedule = commit -> gpuStart (queue/schedule latency), wake =
    /// gpuEnd -> waitUntilCompleted return (thread wake latency). The
    /// remainder of cb1WaitNanos is GPU execution of the cb1 itself plus any
    /// CBs queued ahead of it (the prior layer's routed CB).
    public private(set) var cb1ScheduleNanos: UInt64 = 0
    public private(set) var cb1WakeNanos: UInt64 = 0
    /// Same, for the post-routing (MoE) buffer chain.
    public private(set) var routedWaitNanos: UInt64 = 0
    /// Command buffers committed on the decode path.
    public private(set) var cbCommitCount: UInt64 = 0
    /// Sum of `gpuStartTime - kernelEndTime`: how long a committed buffer sits
    /// in the queue before the GPU actually starts it.
    public private(set) var gpuSchedGapNanos: UInt64 = 0

    /// Absolute GPU execution spans, in `MTLCommandBuffer` clock seconds. Kept
    /// only while GPU timing is on. Aggregate counters cannot distinguish "GPU
    /// idle in thousands of sub-millisecond gaps" (a command-buffer count
    /// problem) from "GPU idle in a few long stalls" (a pipeline problem), and
    /// the two call for opposite fixes — so record the timeline and measure the
    /// gap distribution directly.
    public private(set) var gpuSpanStarts: [Double] = []
    public private(set) var gpuSpanEnds: [Double] = []
    /// Which stage produced each span, so an idle gap can be attributed to the
    /// stage the GPU had just finished when it went quiet.
    public private(set) var gpuSpanLabels: [UInt8] = []
    public static let gpuStageNames = ["cb1", "sharedFFN", "phase1Hit", "routed", "head"]

    fileprivate func accumulateGPUTime(_ cb: MTLCommandBuffer,
                                       into total: inout UInt64,
                                       label: UInt8) {
        let elapsed = cb.gpuEndTime - cb.gpuStartTime
        guard elapsed > 0, elapsed.isFinite else { return }
        total &+= UInt64(elapsed * 1e9)
        let gap = cb.gpuStartTime - cb.kernelEndTime
        if gap > 0, gap.isFinite {
            gpuSchedGapNanos &+= UInt64(gap * 1e9)
        }
        gpuSpanStarts.append(cb.gpuStartTime)
        gpuSpanEnds.append(cb.gpuEndTime)
        gpuSpanLabels.append(label)
    }

    /// Union of all recorded GPU spans, plus the idle gaps between them.
    /// Returns (busySeconds, idleSeconds, sortedGapsDescending) over the window
    /// spanned by the first and last recorded buffer.
    public func gpuTimelineAnalysis()
        -> (busy: Double, idle: Double, gaps: [Double], gapAfter: [Double], gapAfterCount: [Int]) {
        let n = RealForwardRunner.gpuStageNames.count
        guard !gpuSpanStarts.isEmpty else {
            return (0, 0, [], [Double](repeating: 0, count: n),
                    [Int](repeating: 0, count: n))
        }
        let order = (0..<gpuSpanStarts.count).sorted {
            gpuSpanStarts[$0] < gpuSpanStarts[$1]
        }
        var busy = 0.0
        var gaps: [Double] = []
        var gapAfter = [Double](repeating: 0, count: n)
        var gapAfterCount = [Int](repeating: 0, count: n)
        var curStart = gpuSpanStarts[order[0]]
        var curEnd = gpuSpanEnds[order[0]]
        // The span that ends the current busy run owns the gap that follows it.
        var curEndLabel = gpuSpanLabels[order[0]]
        for i in order.dropFirst() {
            let s = gpuSpanStarts[i], e = gpuSpanEnds[i]
            if s > curEnd {
                busy += curEnd - curStart
                let g = s - curEnd
                gaps.append(g)
                let li = Int(curEndLabel)
                if li < n {
                    gapAfter[li] += g
                    gapAfterCount[li] += 1
                }
                curStart = s
                curEnd = e
                curEndLabel = gpuSpanLabels[i]
            } else if e > curEnd {
                curEnd = e
                curEndLabel = gpuSpanLabels[i]
            }
        }
        busy += curEnd - curStart
        let window = curEnd - gpuSpanStarts[order[0]]
        return (busy, max(0, window - busy), gaps.sorted(by: >), gapAfter, gapAfterCount)
    }

    public private(set) var totalIoNanos: UInt64 = 0
    public private(set) var fetchHitOnlyNanos: UInt64 = 0
    public private(set) var fetchHitOnlyCount: UInt64 = 0
    public private(set) var fetchAsyncPlanNanos: UInt64 = 0
    public private(set) var fetchAsyncPlanCount: UInt64 = 0
    public private(set) var fetchAsyncNoPlanNanos: UInt64 = 0
    public private(set) var fetchAsyncNoPlanCount: UInt64 = 0
    /// CPU wall time reading router indices back from GPU (decode loop).
    public private(set) var readbackNanos: UInt64 = 0
    /// CPU wall time planning routed experts (cache-hit classification).
    public private(set) var planNanos: UInt64 = 0
    /// Wall clock from waitForCompletion(cb) return to routedCB.commit().
    public private(set) var chainWallNanos: UInt64 = 0
    public private(set) var totalCb1Nanos: UInt64 = 0
    public private(set) var totalCb2Nanos: UInt64 = 0
    public private(set) var totalHeadNanos: UInt64 = 0
    public private(set) var totalHeadFusedNanos: UInt64 = 0
    public private(set) var lastGreedyToken: UInt32 = 0
    /// Greedy argmax per row of the most recent `verifyBatch` chunk.
    public private(set) var lastGreedyRowTokens: [UInt32] = []
    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }

    /// Score `tokens` in a single multi-token forward and return the target's
    /// greedy prediction after each one.
    ///
    /// This is the batched half of speculative decoding. Passing
    /// `[current, d0, d1, d2]` yields `[p0, p1, p2, p3]` where `p_i` is what
    /// the target would emit after consuming `tokens[0...i]`; draft `d_i` is
    /// accepted exactly when `p_i == d_i`. Running the whole candidate span
    /// through one forward is what makes speculation profitable: the routed
    /// MoE path de-duplicates each layer's expert set across the chunk
    /// (`PrefillMoEGrouping`), so N tokens cost far less expert I/O than N
    /// separate decode steps, and I/O — not FLOPs — is the bottleneck here.
    ///
    /// K/V for the whole span is written, including any rejected suffix; the
    /// caller rewinds with `rewindKV(to:)` and the next forward overwrites the
    /// stale slots.
    public func verifyBatch(tokens: [Int32],
                            startPosition: Int,
                            config: PrefillRuntimeConfig = .defaultChunked,
                            into logits: MTLBuffer) async throws -> [UInt32] {
        guard useFusedGreedyHead else {
            throw PrefillError.unsupportedPrefillSeed(
                "verifyBatch requires the fused greedy head")
        }
        guard !tokens.isEmpty else { return [] }
        guard tokens.count <= Self.maxPerRowGreedyRows else {
            throw PrefillError.chunkedUnsupported(
                "verifyBatch \(tokens.count) rows exceeds cap \(Self.maxPerRowGreedyRows)")
        }
        // One span keeps every row inside a single `scratch.hidden` block, so
        // the per-row head reads the rows it just produced.
        guard tokens.count <= config.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "verifyBatch needs a single span: \(tokens.count) > chunkTokens \(config.chunkTokens)")
        }
        perRowGreedyRowCount = tokens.count
        defer { perRowGreedyRowCount = 0 }
        _ = try await prefillChunked(tokens: tokens[0...],
                                     startPosition: startPosition,
                                     outputMode: .greedyIfAvailable,
                                     config: config,
                                     into: logits) { _ in }
        return lastGreedyRowTokens
    }

    /// Re-publish row `row` of the last verify chunk as the "last hidden state".
    ///
    /// After a partial accept the drafter must continue from the last *accepted*
    /// row, not the last row forwarded, so the row-`t-1` copy that
    /// `executePrefillChunk` performs has to be overridden.
    public func publishHiddenRow(_ row: Int) throws {
        guard let scratch = prefillScratch else {
            throw PrefillError.chunkedUnsupported(
                "publishHiddenRow requires an allocated prefill scratch block")
        }
        let rowBytes = cfg.hiddenSize * MemoryLayout<Float16>.stride
        guard row >= 0, (row + 1) * rowBytes <= scratch.hidden.length else {
            throw PrefillError.chunkedUnsupported(
                "publishHiddenRow row \(row) is outside the scratch hidden block")
        }
        runSync { cb in
            if let blit = cb.makeBlitCommandEncoder() {
                blit.copy(from: scratch.hidden,
                          sourceOffset: row * rowBytes,
                          to: hidden,
                          destinationOffset: 0,
                          size: rowBytes)
                blit.endEncoding()
            }
        }
    }

    /// Drop the K/V of a rejected speculative suffix. See `KVCacheManager.rewind`.
    public func rewindKV(to position: Int) throws {
        guard let kv else {
            throw PrefillError.prefillCursorMismatch(
                "rewindKV requires an initialized KV cache")
        }
        kv.rewind(to: position)
    }

    /// Copy the last decode hidden state as Float32 for MTP draft generation.
    public func copyLastHiddenState() -> [Float] {
        let count = cfg.hiddenSize
        var result = [Float](repeating: 0, count: count)
        let ptr = hidden.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { result[i] = Float(ptr[i]) }
        return result
    }

    /// Scratch for `copyTokenEmbedding` — must never alias `hidden`, which the
    /// main decode step owns.
    private var targetTokenEmbeddingScratch: MTLBuffer?

    /// Target-model token embedding, for the MTP drafter's fusion input.
    ///
    /// The assistant consumes `concat(target_embed(t), backbone_hidden)`. The
    /// embedding must come from the *target's* int4 table with Gemma's
    /// `sqrt(hidden_size)` scaling — the assistant's own `embed_tokens` is tied
    /// to its output head and lives in a different space, so substituting it
    /// (or leaving the provider nil) feeds the drafter a bogus fusion vector.
    public func copyTokenEmbedding(_ token: Int32) -> [Float]? {
        let d = cfg.hiddenSize
        if targetTokenEmbeddingScratch == nil {
            targetTokenEmbeddingScratch = ctx.device.makeBuffer(
                length: max(d, 1) * MemoryLayout<Float16>.stride,
                options: .storageModeShared)
            targetTokenEmbeddingScratch?.label = "assistant.targetTokenEmbedding"
        }
        guard let out = targetTokenEmbeddingScratch else { return nil }
        let emb = model.embedding
        let sqrtHidden = Float(d).squareRoot()
        runSync { cb in
            embedInt4.encode(commandBuffer: cb,
                             table: emb.buffer, tableOffset: Int(emb.offset),
                             scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                             biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                             out: out,
                             tokenId: UInt32(bitPattern: token),
                             d: UInt32(d),
                             outScale: sqrtHidden)
        }
        let ptr = out.contents().bindMemory(to: Float16.self, capacity: d)
        return (0..<d).map { Float(ptr[$0]) }
    }

    /// Build a bridge snapshot for the MTP drafter from current hidden state + KV cache.
    public func makeBridgeSnapshot(currentToken: Int32) -> AssistantBridgeSnapshot {
        let hiddenState = copyLastHiddenState()
        var fullKV: AssistantBridgeKVSnapshot?
        var slidingKV: AssistantBridgeKVSnapshot?
        if let kv = kv {
            let valid = kv.position
            for layer in 0..<cfg.numLayers {
                let kind = kv.layerKind(layer)
                let kvKey = kv.keyView(layer: layer, validTokenCount: valid)
                let kvVal = kv.valueView(layer: layer, validTokenCount: valid)
                let ringCapacity = kv.ringCapacity(layer: layer)
                let keyBuf = AssistantBridgeKVBuffer(
                    buffer: kvKey.buffer, offset: kvKey.offset, stride: kvKey.stride,
                    startSlot: kvKey.startSlot, validTokenCount: valid,
                    ringCapacity: ringCapacity)
                let valBuf = AssistantBridgeKVBuffer(
                    buffer: kvVal.buffer, offset: kvVal.offset, stride: kvVal.stride,
                    startSlot: kvVal.startSlot, validTokenCount: valid,
                    ringCapacity: ringCapacity)
                let snap = AssistantBridgeKVSnapshot(
                    kind: kind == .full ? .fullAttention : .slidingAttention,
                    headDim: kind == .full ? cfg.fullHeadDim : cfg.headDim,
                    numKVHeads: kind == .full ? cfg.numFullKVHeads : cfg.numKVHeads,
                    key: keyBuf, value: valBuf)
                // Mirrors `Gemma4TextAttention.store_full_length_kv`: publish the
                // LAST layer of each type (target has num_kv_shared_layers == 0,
                // so that is layer 28 sliding / layer 29 full). The original
                // first-match + early `break` handed the drafter layer-0
                // activations and pinned accept rate at 0%.
                if kind == .full { fullKV = snap } else if kind == .swa { slidingKV = snap }
            }
        }
        let tokenEmbedding = copyTokenEmbedding(currentToken)
        if ProcessInfo.processInfo.environment["TURBO_FIELDFARE_MTP_DEBUG"] != nil {
            func stats(_ v: [Float]) -> String {
                guard !v.isEmpty else { return "empty" }
                var sumSq: Double = 0
                var lo = Float.greatestFiniteMagnitude
                var hi = -Float.greatestFiniteMagnitude
                var nonFinite = 0
                for x in v {
                    if !x.isFinite { nonFinite += 1; continue }
                    sumSq += Double(x) * Double(x)
                    lo = min(lo, x); hi = max(hi, x)
                }
                let rms = (sumSq / Double(v.count)).squareRoot()
                return String(format: "n=%d rms=%.4f min=%.3f max=%.3f nonfinite=%d",
                              v.count, rms, lo, hi, nonFinite)
            }
            var msg = "[mtp] hidden  \(stats(hiddenState))\n"
            msg += "[mtp] embed   \(tokenEmbedding.map(stats) ?? "nil")\n"
            msg += "[mtp] kvValid=\(kv?.position ?? -1) full=\(fullKV != nil) sliding=\(slidingKV != nil)\n"
            FileHandle.standardError.write(Data(msg.utf8))
        }
        return AssistantBridgeSnapshot(
            lastHiddenState: hiddenState,
            lastTokenEmbedding: tokenEmbedding,
            fullAttentionKV: fullKV,
            slidingAttentionKV: slidingKV)
    }
    public private(set) var totalRDAdviseNanos: UInt64 = 0
    public private(set) var totalRDAdviseCalls: UInt64 = 0
    public private(set) var totalRDAdviseBytes: UInt64 = 0
    public private(set) var totalRDAdviseFailures: UInt64 = 0
    public private(set) var totalRDAdviseSkipped: UInt64 = 0
    public private(set) var totalPrefillAttentionNanos: UInt64 = 0
    public private(set) var totalPrefillRouterFrontHalfGPUNanos: UInt64 = 0
    public private(set) var totalPrefillRouterReadbackAndCPUNanos: UInt64 = 0
    public private(set) var totalPrefillSharedExpertNanos: UInt64 = 0
    public private(set) var totalPrefillStreamedRoutedTilesWaitNanos: UInt64 = 0
    public private(set) var totalPrefillTailReduceLayerTailNanos: UInt64 = 0
    public private(set) var totalPrefillTilePlanNanos: UInt64 = 0
    public private(set) var totalPrefillTileFetchBindingNanos: UInt64 = 0
    public private(set) var totalPrefillTileArgumentBufferNanos: UInt64 = 0
    public private(set) var totalPrefillTileFetchOpenOrReadNanos: UInt64 = 0
    public private(set) var totalPrefillTileFetchReadWallNanos: UInt64 = 0
    public private(set) var totalPrefillTileFetchCacheSlotOverheadNanos: UInt64 = 0
    private var currentPrefillExpertAccessPattern = PrefillExpertAccessPatternTracker()
    private var currentRequestID: UInt64?
    private var currentDecodeStepIndex: Int?
    private var isInDecodePhase = false

    public var routedExpertCacheTelemetrySnapshots: [RoutedExpertCacheLayerTelemetrySnapshot] {
        model.routedExpertCacheTelemetrySnapshots()
    }

    public var prefillTimingBreakdownSnapshot: PrefillTimingBreakdown {
        PrefillTimingBreakdown(
            attentionNanos: totalPrefillAttentionNanos,
            routerFrontHalfGPUNanos: totalPrefillRouterFrontHalfGPUNanos,
            routerReadbackAndCPUNanos: totalPrefillRouterReadbackAndCPUNanos,
            sharedExpertNanos: totalPrefillSharedExpertNanos,
            streamedRoutedTilesWaitNanos: totalPrefillStreamedRoutedTilesWaitNanos,
            tailReduceLayerTailNanos: totalPrefillTailReduceLayerTailNanos,
            tilePlanNanos: totalPrefillTilePlanNanos,
            tileFetchBindingNanos: totalPrefillTileFetchBindingNanos,
            tileArgumentBufferNanos: totalPrefillTileArgumentBufferNanos,
            tileFetchOpenOrReadNanos: totalPrefillTileFetchOpenOrReadNanos,
            tileFetchReadWallNanos: totalPrefillTileFetchReadWallNanos,
            tileFetchCacheSlotOverheadNanos: totalPrefillTileFetchCacheSlotOverheadNanos)
    }

    public var prefillExpertAccessPatternSnapshot: PrefillExpertAccessPattern {
        currentPrefillExpertAccessPattern.metrics
    }

    public var prefillExpertTraceSnapshot: PrefillExpertTrace {
        currentPrefillExpertAccessPattern.trace
    }

    private func recordRDAdvice(_ result: ExpertIOAdviceResult, wallNanos: UInt64) {
        totalRDAdviseNanos &+= wallNanos
        totalRDAdviseCalls &+= UInt64(result.calls)
        totalRDAdviseBytes &+= result.bytes
        totalRDAdviseFailures &+= UInt64(result.failed)
        totalRDAdviseSkipped &+= UInt64(result.skipped)
    }

    private func shouldSkipRDAdvice(position: Int,
                                    requestedMisses: Int,
                                    estimatedBytes: UInt64,
                                    canOverlapUsefulGPUWork: Bool) -> ExpertIOAdviceResult? {
        switch rdadvisePolicyMode {
        case .bounded:
            if position <= rdadviseSkipUntilPosition {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            if requestedMisses > Self.rdadviseBoundedMissCap {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            return nil
        case .adaptive:
            if position != rdadviseAdaptivePosition {
                rdadviseAdaptivePosition = position
                rdadviseAdaptivePositionBytes = 0
            }
            let cumulativeEstimatedBytes = rdadviseAdaptivePositionBytes &+ estimatedBytes
            let shouldSkip = rdadviseAdaptiveState.shouldSkip(
                position: position,
                requestedMisses: requestedMisses,
                estimatedBytes: cumulativeEstimatedBytes,
                canOverlapUsefulGPUWork: canOverlapUsefulGPUWork)
            rdadviseAdaptivePositionBytes = cumulativeEstimatedBytes
            guard shouldSkip else { return nil }
            return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                bytes: estimatedBytes)
        case .default, .off:
            return nil
        }
    }

    private func updateRDAdvicePolicy(after result: ExpertIOAdviceResult,
                                      position: Int) {
        switch rdadvisePolicyMode {
        case .bounded:
            if result.maxCallNanos > Self.rdadviseBoundedMaxCallNanos {
                rdadviseSkipUntilPosition = max(rdadviseSkipUntilPosition, position + 1)
            }
        case .adaptive:
            rdadviseAdaptiveState.update(after: result, position: position)
        case .default, .off:
            break
        }
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try await produceToken(token: token,
                               position: position,
                               into: logits,
                               emitHead: true,
                               outputMode: .greedyIfAvailable)
    }

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0 else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill startPosition must be non-negative")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }

        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        for (spanIndex, span) in spans.enumerated() {
            let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
            let upper = tokens.index(lower, offsetBy: span.tokenCount)
            try await executePrefillChunk(
                tokens: tokens[lower..<upper],
                startPosition: span.startPosition,
                outputMode: outputMode,
                logits: logits,
                scratch: scratch,
                config: config,
                writeFinalHead: spanIndex == spans.count - 1)
            onProgress(span.completedCount)
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    @discardableResult
    private func ensurePrefillScratch(config: PrefillRuntimeConfig) throws -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(config: cfg, runtime: config)
        if let scratch = prefillScratch, scratch.layout == layout {
            return scratch
        }
        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)
        prefillScratch = scratch
        return scratch
    }

    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     outputMode: PrefillOutputMode,
                                     logits: MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     writeFinalHead: Bool) async throws {
        guard !tokens.isEmpty else { return }
        guard kv != nil else {
            throw PrefillError.chunkedUnsupported("chunked prefill attention requires FP16 KV")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard startPosition >= 0, startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)")
        }
        guard tokens.count <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill token count \(tokens.count) exceeds scratch chunk size \(scratch.layout.chunkTokens)")
        }
        if let kv, kv.fp16RingEnabled, let ringLayer = (0..<cfg.numLayers).first(where: {
            kv.ringCapacity(layer: $0) > 0
        }) {
            let requiredCapacity = min(maxContext, cfg.slidingWindow + config.chunkTokens)
            let ringCapacity = kv.ringCapacity(layer: ringLayer)
            guard requiredCapacity <= ringCapacity else {
                throw PrefillError.chunkedUnsupported(
                    "FP16 KV ring capacity \(ringCapacity) cannot hold required capacity \(requiredCapacity) for maxContext \(maxContext), slidingWindow \(cfg.slidingWindow), and prefillChunkTokens \(config.chunkTokens)")
            }
        }

        struct LayerPrefillQKVViews {
            let inputNorm: TensorView
            let q: TensorView
            let k: TensorView
            let v: TensorView
            let o: TensorView
            let postAttention: TensorView
            let preFFN: TensorView
            let preFFN2: TensorView
            let postFFN2: TensorView
            let postFFN: TensorView
            let layerScalar: TensorView
            let qNorm: TensorView
            let kNorm: TensorView
            let router: TensorView
            let routerPerExpertScale: TensorView
        }

        let layerViews = try (0..<cfg.numLayers).map { L in
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            return LayerPrefillQKVViews(
                inputNorm: try model.inputNorm(layer: L),
                q: try model.qProj(layer: L),
                k: try model.kProj(layer: L),
                v: isFull ? (try model.kProj(layer: L)) : (try model.vProj(layer: L)),
                o: try model.oProj(layer: L),
                postAttention: try model.postAttnNorm(layer: L),
                preFFN: try model.preFFN(layer: L),
                preFFN2: try model.preFFN2(layer: L),
                postFFN2: try model.postFFN2(layer: L),
                postFFN: try model.postFFN(layer: L),
                layerScalar: try model.layerScalar(layer: L),
                qNorm: try model.qNorm(layer: L),
                kNorm: try model.kNorm(layer: L),
                router: try model.router(layer: L),
                routerPerExpertScale: try model.routerPerExpertScale(layer: L))
        }

        let tokenIDs = tokens.map { UInt32(bitPattern: $0) }
        guard let tokenBuffer = ctx.device.makeBuffer(bytes: tokenIDs,
                                                      length: tokenIDs.count * MemoryLayout<UInt32>.stride,
                                                      options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        let sqrtHidden = Float(D).squareRoot()
        let t = tokens.count
        let emb = model.embedding

        func encodeInt4Projection(commandBuffer: MTLCommandBuffer,
                                  family: PrefillProjectionFamily,
                                  bits: Int = 4,
                                  weights: TensorView,
                                  x: MTLBuffer,
                                  y: MTLBuffer,
                                  rows: Int,
                                  columns: Int,
                                  tokenCount: Int,
                                  xStrideElements: Int,
                                  yStrideElements: Int) {
            if tokenCount >= 32,
               bits == 4,
               family == .q || family == .kv || family == .o,
               let candidate = prefillMPPAffineInt4 {
                let path = candidate.encode(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    scales: weights.buffer,
                    scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer,
                    biasesOffset: Int(weights.biasOffset),
                    x: x,
                    y: y,
                    m: tokenCount,
                    n: rows,
                    k: columns)
                if path == .affineThreadgroupF16 {
                    return
                }
            }
            if PrefillProjectionDispatchPolicy.selectedDispatch(for: family,
                                                                chunkTokens: tokenCount) == .qmm {
                prefillQMM.encode(commandBuffer: commandBuffer,
                                  weights: weights.buffer,
                                  weightsOffset: Int(weights.offset),
                                  scales: weights.buffer,
                                  scalesOffset: Int(weights.scaleOffset),
                                  biases: weights.buffer,
                                  biasesOffset: Int(weights.biasOffset),
                                  x: x,
                                  y: y,
                                  t: tokenCount,
                                  n: rows,
                                  k: columns,
                                  bits: bits)
                return
            }
            for row in 0..<tokenCount {
                if bits == 3 {
                    int4.encodeTgx(commandBuffer: commandBuffer,
                                   weights: weights.buffer,
                                   weightsOffset: Int(weights.offset),
                                   scales: weights.buffer,
                                   scalesOffset: Int(weights.scaleOffset),
                                   biases: weights.buffer,
                                   biasesOffset: Int(weights.biasOffset),
                                   x: x,
                                   xOffset: row * xStrideElements * MemoryLayout<Float16>.stride,
                                   y: y,
                                   yOffset: row * yStrideElements * MemoryLayout<Float16>.stride,
                                   m: UInt32(rows),
                                   n: UInt32(columns),
                                   bits: 3)
                } else {
                    int4.encode(commandBuffer: commandBuffer,
                                weights: weights.buffer,
                                weightsOffset: Int(weights.offset),
                                scales: weights.buffer,
                                scalesOffset: Int(weights.scaleOffset),
                                biases: weights.buffer,
                                biasesOffset: Int(weights.biasOffset),
                                x: x,
                                xOffset: row * xStrideElements * MemoryLayout<Float16>.stride,
                                y: y,
                                yOffset: row * yStrideElements * MemoryLayout<Float16>.stride,
                                m: UInt32(rows),
                                n: UInt32(columns))
                }
            }
        }

        func copyPrefillKV(commandBuffer: MTLCommandBuffer,
                           source: MTLBuffer,
                           destination: (buffer: MTLBuffer, offset: Int, stride: Int),
                           sourceTokenOffset: Int,
                           tokenCount: Int,
                           bytesPerToken: Int) throws {
            guard tokenCount > 0 else { return }
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.copy(from: source,
                      sourceOffset: sourceTokenOffset * bytesPerToken,
                      to: destination.buffer,
                      destinationOffset: destination.offset,
                      size: tokenCount * bytesPerToken)
            blit.endEncoding()
        }

        func copyPrefillKVToCache(commandBuffer: MTLCommandBuffer,
                                  kv: KVCacheManager,
                                  layer: Int,
                                  startPosition: Int,
                                  tokenCount: Int,
                                  keySource: MTLBuffer,
                                  valueSource: MTLBuffer,
                                  bytesPerToken: Int) throws {
            let capacity = kv.capacity(layer: layer)
            let physicalStart = startPosition % capacity
            let firstSpan = min(tokenCount, capacity - physicalStart)
            let keyFirst = kv.kRange(layer: layer, start: startPosition, count: firstSpan)
            let valueFirst = kv.vRange(layer: layer, start: startPosition, count: firstSpan)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: keySource,
                              destination: keyFirst,
                              sourceTokenOffset: 0,
                              tokenCount: firstSpan,
                              bytesPerToken: bytesPerToken)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: valueSource,
                              destination: valueFirst,
                              sourceTokenOffset: 0,
                              tokenCount: firstSpan,
                              bytesPerToken: bytesPerToken)
            guard firstSpan < tokenCount else { return }

            let secondCount = tokenCount - firstSpan
            let secondStart = startPosition + firstSpan
            let keySecond = kv.kRange(layer: layer, start: secondStart, count: secondCount)
            let valueSecond = kv.vRange(layer: layer, start: secondStart, count: secondCount)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: keySource,
                              destination: keySecond,
                              sourceTokenOffset: firstSpan,
                              tokenCount: secondCount,
                              bytesPerToken: bytesPerToken)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: valueSource,
                              destination: valueSecond,
                              sourceTokenOffset: firstSpan,
                              tokenCount: secondCount,
                              bytesPerToken: bytesPerToken)
        }

        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: tokens.count)

        guard var attentionCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        prefillEmbed.encode(commandBuffer: attentionCB,
                            table: emb.buffer,
                            tableOffset: Int(emb.offset),
                            scales: emb.buffer,
                            scalesOffset: Int(emb.scaleOffset),
                            biases: emb.buffer,
                            biasesOffset: Int(emb.biasOffset),
                            tokens: tokenBuffer,
                            out: scratch.hidden,
                            t: UInt32(t),
                            d: UInt32(D),
                            outScale: sqrtHidden)

        for L in 0..<cfg.numLayers {
            model.beginOpeningRoutedExpertStreamer(layer: L)
            let views = layerViews[L]
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            let headDim = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVHeads = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim = cfg.numHeads * headDim
            let kvDim = numKVHeads * headDim

            prefillRMS.encodeBF16W(commandBuffer: attentionCB,
                                   x: scratch.hidden,
                                   weight: views.inputNorm.buffer,
                                   weightOffset: Int(views.inputNorm.offset),
                                   out: scratch.normed,
                                   t: UInt32(t),
                                   d: UInt32(D),
                                   eps: eps)
            encodeInt4Projection(commandBuffer: attentionCB,
                                 family: .q,
                                 bits: model.attentionWeightBits == 3 ? 3 : 4,
                                 weights: views.q,
                                 x: scratch.normed,
                                 y: scratch.q,
                                 rows: qDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: qDim)
            encodeInt4Projection(commandBuffer: attentionCB,
                                 family: .kv,
                                 bits: model.attentionWeightBits == 3 ? 3 : 4,
                                 weights: views.k,
                                 x: scratch.normed,
                                 y: scratch.kStage,
                                 rows: kvDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: kvDim)
            encodeInt4Projection(commandBuffer: attentionCB,
                                 family: .kv,
                                 bits: model.attentionWeightBits == 3 ? 3 : 4,
                                 weights: views.v,
                                 x: scratch.normed,
                                 y: scratch.vStage,
                                 rows: kvDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: kvDim)

            let rotatedPairs = isFull
                ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                : UInt32(headDim / 2)
            prefillQKVEpilogue.encode(commandBuffer: attentionCB,
                                       q: scratch.q,
                                       k: scratch.kStage,
                                       v: scratch.vStage,
                                       qWeight: views.qNorm.buffer,
                                       qWeightOffset: Int(views.qNorm.offset),
                                       kWeight: views.kNorm.buffer,
                                       kWeightOffset: Int(views.kNorm.offset),
                                       startPosition: UInt32(startPosition),
                                       queryCount: UInt32(t),
                                       headDim: UInt32(headDim),
                                       numQHeads: UInt32(cfg.numHeads),
                                       numKVHeads: UInt32(numKVHeads),
                                       qTokenStrideElements: UInt32(qDim),
                                       kvTokenStrideElements: UInt32(kvDim),
                                       theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                       rotatedPairs: rotatedPairs,
                                       eps: eps)

            if let kv {
                let bytes = t * kvDim * MemoryLayout<Float16>.stride
                  try copyPrefillKVToCache(commandBuffer: attentionCB,
                                         kv: kv,
                                         layer: L,
                                         startPosition: startPosition,
                                         tokenCount: t,
                                         keySource: scratch.kStage,
                                         valueSource: scratch.vStage,
                                         bytesPerToken: bytes / t)
            }
            let params = PrefillAttentionParams(
                    startPosition: UInt32(startPosition),
                    queryCount: UInt32(t),
                    headDim: UInt32(headDim),
                    numQHeads: UInt32(cfg.numHeads),
                    numKVHeads: UInt32(numKVHeads),
                    kvValidCount: UInt32(startPosition + t),
                    slidingWindow: isFull ? UInt32(startPosition + t) : UInt32(cfg.slidingWindow),
                    kvTokenStrideElements: UInt32(kvDim),
                    qTokenStrideElements: UInt32(qDim),
                    oTokenStrideElements: UInt32(qDim),
                    scale: 1.0)
            if let kv {
                    let keyBuffer = kv.keyBuffer(layer: L, validTokenCount: startPosition + t)
                    let valueBuffer = kv.valueBuffer(layer: L, validTokenCount: startPosition + t)
                    let ringCapacity = kv.ringCapacity(layer: L)
                    let activeRingCapacity = ringCapacity > 0 && startPosition + t > ringCapacity
                        ? UInt32(ringCapacity)
                        : 0
                      prefillAttention.encodeCausal(commandBuffer: attentionCB,
                                                  q: scratch.q,
                                                  k: keyBuffer,
                                                  v: valueBuffer,
                                                  out: scratch.attentionOutput,
                                                  params: params,
                                                  kvRingCapacity: activeRingCapacity,
                                                  path: prefillAttentionPath)
            } else {
                throw PrefillError.chunkedUnsupported(
                    "chunked prefill attention requires FP16 KV")
            }
              attentionCB.commit()
              totalPrefillAttentionNanos &+= timedWaitForCompletion(attentionCB)
              if let error = attentionCB.error {
                  throw error
              }

              guard let routerCB = ctx.queue.makeCommandBuffer() else {
                  throw ModelError.residentBufferWrapFailed
              }
              encodeInt4Projection(commandBuffer: routerCB,
                                     family: .o,
                                     bits: model.attentionWeightBits == 3 ? 3 : 4,
                                     weights: views.o,
                                     x: scratch.attentionOutput,
                                     y: scratch.h1,
                                     rows: D,
                                     columns: qDim,
                                     tokenCount: t,
                                     xStrideElements: qDim,
                                     yStrideElements: D)
              prefillPostAttention.encode(commandBuffer: routerCB,
                                            hidden: scratch.hidden,
                                            attn: scratch.h1,
                                            denseX: scratch.denseX,
                                            routedX: scratch.routedX,
                                            routerX: scratch.routerX,
                                            postAttentionWeight: views.postAttention.buffer,
                                            postAttentionWeightOffset: Int(views.postAttention.offset),
                                            preFFNWeight: views.preFFN.buffer,
                                            preFFNWeightOffset: Int(views.preFFN.offset),
                                            preFFN2Weight: views.preFFN2.buffer,
                                            preFFN2WeightOffset: Int(views.preFFN2.offset),
                                            queryCount: UInt32(t),
                                            d: UInt32(D),
                                            hiddenStrideElements: UInt32(D),
                                            attnStrideElements: UInt32(D),
                                            denseStrideElements: UInt32(D),
                                            routedStrideElements: UInt32(D),
                                            routerStrideElements: UInt32(D),
                                            eps: eps)
              prefillRouter.encodeGemma4Block(
                          commandBuffer: routerCB,
                        weights: views.router.buffer,
                        weightsOffset: Int(views.router.offset),
                        scales: views.router.buffer,
                        scalesOffset: Int(views.router.scaleOffset),
                        biases: views.router.buffer,
                        biasesOffset: Int(views.router.biasOffset),
                        hidden: scratch.routerX,
                        effectiveScale: effectiveScaleBuffers[L],
                        perExpertScale: views.routerPerExpertScale.buffer,
                        perExpertScaleOffset: Int(views.routerPerExpertScale.offset),
                        outIndices: scratch.routeIDs,
                        outWeights: scratch.routeWeights,
                        queryCount: UInt32(t),
                        numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D),
                        topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D))

                      routerCB.commit()
                      totalPrefillRouterFrontHalfGPUNanos &+= timedWaitForCompletion(routerCB)
                      if let error = routerCB.error {
                        throw error
                    }

                      let routerReadbackAndCPUStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    let routeCount = t * cfg.topKExperts
                    let idPtr = scratch.routeIDs.contents()
                        .bindMemory(to: UInt32.self, capacity: routeCount)
                    let weightPtr = scratch.routeWeights.contents()
                        .bindMemory(to: Float16.self, capacity: routeCount)
                    var routeIDs = [UInt32]()
                    routeIDs.reserveCapacity(routeCount)
                    var routeWeights = [Float16]()
                    routeWeights.reserveCapacity(routeCount)
                    for i in 0..<routeCount {
                        routeIDs.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
                        routeWeights.append(weightPtr[i])
                    }
                    if unionStatsEnabled, t <= 16 {
                        let poolSet = L < hotPoolSets.count ? hotPoolSets[L] : []
                        var union = Set<Int>()
                        var perTokenPool = 0
                        for token in 0..<t {
                            var covered = 0
                            for r in 0..<cfg.topKExperts {
                                let id = Int(routeIDs[token * cfg.topKExperts + r])
                                union.insert(id)
                                if poolSet.contains(id) { covered += 1 }
                            }
                            perTokenPool += covered
                        }
                        if unionDumpPath != nil {
                            let ids = union.sorted().map(String.init).joined(separator: " ")
                            unionDumpLines.append("t=\(t),L=\(L),\(ids)")
                        }
                        let poolCover = union.filter { poolSet.contains($0) }.count
                        var b = unionStats[t, default: UnionStatsBucket()]
                        b.visits += 1
                        b.unionTotal += union.count
                        b.unionPool += poolCover
                        b.perTokenTotal += t * cfg.topKExperts
                        b.perTokenPool += perTokenPool
                        unionStats[t] = b
                    }
                    let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs,
                                                                   weights: routeWeights,
                                                                   queryCount: t,
                                                                   topK: cfg.topKExperts)
                    let schedulerConfig = Self.prefillRoutedTileSchedulerConfig
                    let routeTileExpertCount: Int
                    if let slotCount = model.routedExpertCacheSlotCount(layer: L) {
                        guard schedulerConfig.fitsSlotBudget(slotCount: slotCount) else {
                            throw PrefillError.chunkedUnsupported(
                                "prefill routed tile depth \(schedulerConfig.maxPendingDepth) with \(schedulerConfig.tileExperts) experts/tile needs \((schedulerConfig.maxPendingDepth + 1) * schedulerConfig.tileExperts) slots, has \(slotCount)")
                        }
                        routeTileExpertCount = min(schedulerConfig.tileExperts, slotCount)
                    } else {
                        routeTileExpertCount = schedulerConfig.tileExperts
                    }
                      let routedPhysicalOffsets = model.routedExpertPhysicalOffsets(layer: L)
                      let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                        pairs,
                        queryCount: t,
                        topK: cfg.topKExperts,
                        numExperts: cfg.numExperts,
                        tileExpertCount: routeTileExpertCount,
                          expertSortKeys: routedPhysicalOffsets)
                      totalPrefillRouterReadbackAndCPUNanos &+=
                          clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - routerReadbackAndCPUStart

                    guard let sharedCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    let sharedProj = sharedExpertProjections[L]
                    try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
                                                        x: scratch.denseX,
                                                        y: scratch.h1,
                                                        gate: sharedProj.gate,
                                                        up: sharedProj.up,
                                                        down: sharedProj.down,
                                                        scratchGate: scratch.sharedGateScratch,
                                                        scratchUp: scratch.sharedUpScratch,
                                                        scratchAct: scratch.sharedActScratch,
                                                        queryCount: t,
                                                        d: D,
                                                        intermediate: cfg.intermediateSize,
                                                        xStrideElements: D,
                                                        yStrideElements: D)
                    prefillRMS.encodeBF16W(commandBuffer: sharedCB,
                                           x: scratch.h1,
                                           weight: sharedProj.postF1.buffer,
                                           weightOffset: Int(sharedProj.postF1.offset),
                                           out: scratch.h1,
                                           t: UInt32(t),
                                           d: UInt32(D),
                                           eps: eps)
                    sharedCB.commit()
                      totalPrefillSharedExpertNanos &+= timedWaitForCompletion(sharedCB)
                    if let error = sharedCB.error {
                        throw error
                    }

                      let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
                        device: ctx.device,
                        routes: routes)
                    let routedOffsets = model.routedExpertOffsets(layer: L)
                    struct PendingPrefillTile {
                        let tileIndex: Int
                        let commandBuffer: MTLCommandBuffer
                        let fetch: PrefillStreamedTileFetchResult
                        let argumentBuffer: PrefillStreamedTileArgumentBuffer
                    }
                    var pendingTiles: [PendingPrefillTile] = []
                    var tileLifetime = PrefillStreamedTileSlotLifetime()
                    func drainOldestPendingTile() throws {
                        guard !pendingTiles.isEmpty else { return }
                        let pending = pendingTiles.removeFirst()
                        withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                              totalPrefillStreamedRoutedTilesWaitNanos &+=
                                  timedWaitForCompletion(pending.commandBuffer)
                        }
                        if let error = pending.commandBuffer.error {
                            throw error
                        }
                        if !pending.fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.complete(tileIndex: pending.tileIndex)
                        }
                    }

                    let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
                    for (tileIndex, tile) in routes.tiles.enumerated() {
                        let prefillAccessContext = makePrefillAccessContext()
                        let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                            forTile: tileIndex,
                            routes: routes)
                          currentPrefillExpertAccessPattern.record(
                              layer: L,
                              expertIDs: expertIDs,
                              physicalOffsets: routedPhysicalOffsets,
                              expertStride: model.packedExpertsLayout.expertStride)
                        var plannedFetch: RoutedExpertFetchPlan?
                          var avoidingSlots = Set<Int>()
                        if !pendingTiles.isEmpty {
                            let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                            if !pendingAssignedSlots.isEmpty {
                                  let pendingSlots = Set(pendingAssignedSlots)
                                  avoidingSlots = pendingSlots
                                  let planStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                                let plan = try model.planRoutedExpertsIfPossible(
                                    layer: L,
                                    experts: expertIDs,
                                    avoidingSlots: pendingSlots,
                                    accessContext: prefillAccessContext)
                                  totalPrefillTilePlanNanos &+=
                                      clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - planStart
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: pendingAssignedSlots,
                                        avoidingSlotPlanAvailable: plan != nil))
                                switch decision {
                                  case .prefetchNext(let schedulerAvoidingSlots):
                                    guard let plan else {
                                        throw ModelError.indexCorrupt(
                                            detail: "routed tile scheduler requested missing plan")
                                    }
                                      avoidingSlots = Set(schedulerAvoidingSlots)
                                    plannedFetch = plan
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                      avoidingSlots = Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots))
                                case .issueWithoutPending:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler ignored pending tile")
                                }
                            } else {
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: [],
                                        avoidingSlotPlanAvailable: false))
                                switch decision {
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                      avoidingSlots = Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots))
                                case .issueWithoutPending, .prefetchNext:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler failed to drain empty-slot pending tile")
                                }
                            }
                        } else {
                            let decision = routedTileScheduler.decide(
                                PrefillRoutedTileSchedulerInput(
                                    hasPendingTile: false,
                                    pendingAssignedSlots: [],
                                    avoidingSlotPlanAvailable: false))
                            switch decision {
                            case .issueWithoutPending:
                                break
                            case .prefetchNext, .drainBeforeIssue:
                                throw ModelError.indexCorrupt(
                                    detail: "routed tile scheduler requested pending action without pending tile")
                            }
                        }
                        let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                            model: model,
                            layer: L,
                            tileIndex: tileIndex,
                            routes: routes,
                            expertIDs: expertIDs,
                            plannedFetch: plannedFetch,
                              avoidingSlots: avoidingSlots,
                            accessContext: prefillAccessContext)
                          totalPrefillTilePlanNanos &+= fetch.timing.planNanos
                          totalPrefillTileFetchOpenOrReadNanos &+= fetch.timing.fetchOpenOrReadNanos
                          totalPrefillTileFetchBindingNanos &+= fetch.timing.fetchBindingNanos
                          totalPrefillTileFetchReadWallNanos &+= fetch.timing.fetchReadWallNanos
                          totalPrefillTileFetchCacheSlotOverheadNanos &+=
                              fetch.timing.fetchCacheSlotOverheadNanos
                        try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                              pairStart: Int(tile.pairStart),
                                                              pairCount: Int(tile.pairCount))
                        if !fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.begin(tileIndex: tileIndex,
                                                   plannedSlots: fetch.plannedMissSlots)
                        }
                          let argumentBufferStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                          let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                            device: ctx.device,
                            binding: fetch.binding)
                          totalPrefillTileArgumentBufferNanos &+=
                              clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - argumentBufferStart
                        let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                            pairStart: tile.pairStart,
                            pairCount: tile.pairCount,
                            d: UInt32(D),
                            routedIntermediate: UInt32(cfg.moeIntermediateSize),
                            topK: UInt32(cfg.topKExperts),
                            hiddenStrideElements: UInt32(D),
                            binding: fetch.binding,
                            offsets: routedOffsets)
                        guard let tileCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        _ = prefillGroupedMoE.encodeStreamedBatched(
                            commandBuffer: tileCB,
                            hidden: scratch.routedX,
                            sortedPairs: metadata.sortedPairs,
                            routePartials: scratch.routePartials,
                            gateUpActScratch: scratch.routedGateUpActScratch,
                            downScratch: scratch.routedDownScratch,
                            argumentBuffer: argumentBuffer,
                            binding: fetch.binding,
                            params: streamedParams,
                            pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows,
                            bits: model.routedExpertWeightBits)
                        tileCB.commit()
                        pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                               commandBuffer: tileCB,
                                                               fetch: fetch,
                                                               argumentBuffer: argumentBuffer))
                        while pendingTiles.count > schedulerConfig.maxPendingDepth {
                            try drainOldestPendingTile()
                        }
                    }
                    while !pendingTiles.isEmpty {
                        try drainOldestPendingTile()
                    }
                    guard let tailCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                                      routePartials: scratch.routePartials,
                                                      routeWeights: scratch.routeWeights,
                                                      h2: scratch.h2,
                                                      queryCount: UInt32(t),
                                                      topK: UInt32(cfg.topKExperts),
                                                      d: UInt32(D))
                    let scalarBits = views.layerScalar.buffer.contents()
                        .advanced(by: Int(views.layerScalar.offset))
                        .assumingMemoryBound(to: UInt16.self)[0]
                    prefillLayerTail.encode(commandBuffer: tailCB,
                                            h2: scratch.h2,
                                            h1: scratch.h1,
                                            hidden: scratch.hidden,
                                            postFFN2Weight: views.postFFN2.buffer,
                                            postFFN2WeightOffset: Int(views.postFFN2.offset),
                                            postFFNWeight: views.postFFN.buffer,
                                            postFFNWeightOffset: Int(views.postFFN.offset),
                                            queryCount: UInt32(t),
                                            d: UInt32(D),
                                            h2StrideElements: UInt32(D),
                                            h1StrideElements: UInt32(D),
                                            hiddenStrideElements: UInt32(D),
                                            eps: eps,
                                            layerScalar: Quantization.bf16ToFloat(scalarBits))
                    tailCB.commit()
                    withExtendedLifetime(metadata) {
                          totalPrefillTailReduceLayerTailNanos &+=
                              timedWaitForCompletion(tailCB)
                    }
                    if let error = tailCB.error {
                        throw error
                    }
                    if L + 1 < cfg.numLayers {
                        guard let nextCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                          attentionCB = nextCB
                    }
                    continue
        }

        if writeFinalHead {
            let finalNorm = model.finalNorm
            let lm = model.lmHead
            guard let finalCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            // Batched speculative verify needs the greedy argmax of *every*
            // row: row r answers "what does the target predict after
            // tokens[0...r]", which is exactly the accept test for draft r.
            // Ordinary prefill only needs the last row.
            let perRowGreedy = perRowGreedyRowCount > 0
            if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                let rowCount = perRowGreedy ? t : 1
                for slot in 0..<rowCount {
                    let row = perRowGreedy ? slot : t - 1
                    fusionHead.encodeGreedyDecode(
                        commandBuffer: finalCB,
                        hidden: scratch.hidden,
                        hiddenOffset: row * D * MemoryLayout<Float16>.stride,
                        normWeight: finalNorm.buffer,
                        normOffset: Int(finalNorm.offset),
                        weights: lm.buffer,
                        weightsOffset: Int(lm.offset),
                        scales: lm.buffer,
                        scalesOffset: Int(lm.scaleOffset),
                        biases: lm.buffer,
                        biasesOffset: Int(lm.biasOffset),
                        outToken: perRowGreedy ? greedyRowTokensBuf : greedyTokenBuf,
                        outTokenOffset: perRowGreedy ? slot * MemoryLayout<UInt32>.stride : 0,
                        d: UInt32(D),
                        vocab: UInt32(cfg.vocabSize),
                        rmsEps: eps)
                }
            } else {
                prefillFinalRowHead.encodeLogits(commandBuffer: finalCB,
                                                 hiddenBlock: scratch.hidden,
                                                 row: t - 1,
                                                 rowStrideElements: D,
                                                 normWeight: finalNorm.buffer,
                                                 normWeightOffset: Int(finalNorm.offset),
                                                 weights: lm.buffer,
                                                 weightsOffset: Int(lm.offset),
                                                 scales: lm.buffer,
                                                 scalesOffset: Int(lm.scaleOffset),
                                                 biases: lm.buffer,
                                                 biasesOffset: Int(lm.biasOffset),
                                                 logits: logits,
                                                 d: UInt32(D),
                                                 vocab: UInt32(cfg.vocabSize),
                                                 rmsEps: eps)
            }
            // Publish the final prefill row into the single-token `hidden`
            // buffer that `copyLastHiddenState()` reads. Prefill writes into
            // `scratch.hidden` (a [tokens, D] block) while decode writes into
            // `hidden`, so without this copy the MTP drafter's very first
            // speculation was fed an all-zero backbone vector (observable as
            // `[mtp] hidden rms=0.0000`) and had to fall back to the token
            // embedding alone. Both buffers hold pre-final-norm activations,
            // so the copy is semantically exact.
            if let blit = finalCB.makeBlitCommandEncoder() {
                let rowBytes = D * MemoryLayout<Float16>.stride
                blit.copy(from: scratch.hidden,
                          sourceOffset: (t - 1) * rowBytes,
                          to: hidden,
                          destinationOffset: 0,
                          size: rowBytes)
                blit.endEncoding()
            }
            finalCB.commit()
            waitForCompletion(finalCB)
            if let error = finalCB.error {
                throw error
            }
            if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                if perRowGreedy {
                    let base = greedyRowTokensBuf.contents()
                        .bindMemory(to: UInt32.self, capacity: t)
                    lastGreedyRowTokens = Array(UnsafeBufferPointer(start: base, count: t))
                    lastGreedyToken = lastGreedyRowTokens[t - 1]
                } else {
                    lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
                }
            }
        }

        kv?.advance(by: tokens.count)
        prefillChunkState.markCommitted()
    }

    private func produceToken(token: Int32,
                              position: Int,
                              into logits: MTLBuffer,
                              emitHead: Bool,
                              outputMode: PrefillOutputMode) async throws {
        let kvPosition = kv?.position ?? 0
        guard kvPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kvPosition) != position \(position)")
        }
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
        let D    = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let eps: Float = 1e-6
        let sqrtHidden = Float(cfg.hiddenSize).squareRoot()
        struct PendingRoutedCommand {
            let cb: MTLCommandBuffer
            let sharedCB: MTLCommandBuffer?
            let phase1HitCB: MTLCommandBuffer?
            let encodeAndCommitNanos: UInt64
        }
        var pendingRoutedCommand: PendingRoutedCommand?

        func finishPendingRoutedCommand(_ pending: PendingRoutedCommand,
                                        waitIfNeeded: Bool) {
            if waitIfNeeded {
                let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                func wait(_ cb: MTLCommandBuffer) {
                    waitForCompletion(cb)
                }
                if let sharedCB = pending.sharedCB {
                    wait(sharedCB)
                }
                if let phase1HitCB = pending.phase1HitCB {
                    wait(phase1HitCB)
                }
                wait(pending.cb)
                routedWaitNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
            } else if let err = pending.cb.error {
                print("CB error: \(err)")
            }
            if let sharedCB = pending.sharedCB {
                if let err = sharedCB.error {
                    print("CB error: \(err)")
                }
            }
            if let phase1HitCB = pending.phase1HitCB,
               let err = phase1HitCB.error {
                print("CB error: \(err)")
            }
            totalCb2Nanos &+= pending.encodeAndCommitNanos
            if gpuTimingEnabled {
                accumulateGPUTime(pending.cb, into: &gpuRoutedNanos, label: 3)
                if let sharedCB = pending.sharedCB {
                    accumulateGPUTime(sharedCB, into: &gpuSharedNanos, label: 1)
                }
                if let phase1HitCB = pending.phase1HitCB {
                    accumulateGPUTime(phase1HitCB, into: &gpuPhase1HitNanos, label: 2)
                }
            }
        }

        func writeActiveSlots(_ slots: [UInt32], into buffer: MTLBuffer) {
            let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
            for i in 0..<slots.count { ptr[i] = slots[i] }
        }

        // Embed lookup + sqrt(H) fused.
        let emb = model.embedding
        do {
            runSync { cb in
                embedInt4.encode(commandBuffer: cb,
                                 table:  emb.buffer, tableOffset:  Int(emb.offset),
                                 scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                 biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                 out: hidden,
                                 tokenId: UInt32(bitPattern: token),
                                 d: D,
                                 outScale: sqrtHidden)
            }
        }

        for L in 0..<cfg.numLayers {
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            let headDimL = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVL   = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim     = UInt32(cfg.numHeads * headDimL)
            let kvDim    = UInt32(numKVL * headDimL)
            let kSlot    = kv?.kSlot(layer: L, position: position) ?? (buffer: kStage, offset: 0)
            let vSlot    = kv?.vSlot(layer: L, position: position) ?? (buffer: vStage, offset: 0)
            let seqLen   = UInt32(position + 1)

            let inNorm   = try model.inputNorm(layer: L)
            let q        = try model.qProj(layer: L)
            let k        = try model.kProj(layer: L)
            // v_proj only exists on SWA layers; full layers reuse k_proj.
            let vProj    = isFull ? k : (try model.vProj(layer: L))
            let o        = try model.oProj(layer: L)
            let postAttn = try model.postAttnNorm(layer: L)
            let qNorm    = try model.qNorm(layer: L)
            let kNorm    = try model.kNorm(layer: L)
            let preFFN   = try model.preFFN(layer: L)
            let preFFN2  = try model.preFFN2(layer: L)
            let sharedProj = sharedExpertProjections[L]
            let postF2   = try model.postFFN2(layer: L)
            let postF    = try model.postFFN(layer: L)
            let routerW  = try model.router(layer: L)
            let perExpertScale = try model.routerPerExpertScale(layer: L)
            let layerScalarView = try model.layerScalar(layer: L)

            let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // Everything up to and including the router runs in a single CB:
            // the only reason to break is the CPU readback of router indices
            // needed to issue I/O for the routed-expert blobs.
            let gInputNorm: (MTLCommandBuffer) -> Void = { [self] cb in
                rms.encodeBF16W(commandBuffer: cb,
                                x: hidden,
                                weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                                out: normed,
                                d: D, eps: eps)
            }

            let gQKV: (MTLCommandBuffer) -> Void = { [self] cb in
                fusedQKVGEMV.encode(commandBuffer: cb,
                                    qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                                    qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                                    qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                                    kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                                    kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                                    kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                                    vWeights: vProj.buffer, vWeightsOffset: Int(vProj.offset),
                                    vScales: vProj.buffer, vScalesOffset: Int(vProj.scaleOffset),
                                    vBiases: vProj.buffer, vBiasesOffset: Int(vProj.biasOffset),
                                    x: normed,
                                    qOut: qScratch,
                                    kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                                    vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                                    qRows: qDim,
                                    kvRows: kvDim,
                                    n: D,
                                    bits: model.attentionWeightBits == 3 ? 3 : 4)
            }

            let gQKVEpilogue: (MTLCommandBuffer) -> Void = { [self] cb in
                let rotated = isFull
                    ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                    : UInt32(headDimL / 2)
                fusedQKVEpilogue.encode(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer,
                                        kOffset: kSlot.offset,
                                        v: vSlot.buffer,
                                        vOffset: vSlot.offset,
                                        qWeight: qNorm.buffer,
                                        qWeightOffset: Int(qNorm.offset),
                                        kWeight: kNorm.buffer,
                                        kWeightOffset: Int(kNorm.offset),
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        position: UInt32(position),
                                        theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                        rotatedPairs: rotated,
                                        eps: eps)
            }

            let gAttention: (MTLCommandBuffer) -> Void = { [self] cb in
                guard kv != nil else {
                    preconditionFailure("FP16 attention requires an FP16 KV cache")
                }
                if Self.skipAttentionCore || (isFull && Self.skipAttentionFull)
                    || (!isFull && Self.skipAttentionSWA) {
                    // ROI probe: skip the core; downstream reads stale attnOut.
                    return
                }
                if isFull {
                    attention.encodeFull(commandBuffer: cb,
                                         q: qScratch,
                                         k: kSlot.buffer, kOffset: 0,
                                         v: vSlot.buffer, vOffset: 0,
                                         out: attnOut,
                                         headDim: UInt32(headDimL),
                                         numQHeads: UInt32(cfg.numHeads),
                                         numKVHeads: UInt32(numKVL),
                                         seqLen: seqLen,
                                         scale: 1.0)
                } else {
                    let ringCapacity = kv?.ringCapacity(layer: L) ?? 0
                    let activeRingCapacity = ringCapacity > 0 && Int(seqLen) > ringCapacity
                        ? UInt32(ringCapacity)
                        : 0
                    attention.encodeSWA(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer, kOffset: 0,
                                        v: vSlot.buffer, vOffset: 0,
                                        out: attnOut,
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        seqLen: seqLen,
                                        window: UInt32(cfg.slidingWindow),
                                        scale: 1.0,
                                        ringCapacity: activeRingCapacity)
                }
            }
            let gOProj: (MTLCommandBuffer) -> Void = { [self] cb in
                int4.encodeTgx(commandBuffer: cb,
                               weights: o.buffer, weightsOffset: Int(o.offset),
                               scales:  o.buffer, scalesOffset:  Int(o.scaleOffset),
                               biases:  o.buffer, biasesOffset:  Int(o.biasOffset),
                               x: attnOut, y: oOut, m: D, n: qDim,
                               bits: model.attentionWeightBits == 3 ? 3 : 4)
            }

            let gPostAttnSetup: (MTLCommandBuffer) -> Void = { [self] cb in
                fusedPostAttentionSetup.encode(commandBuffer: cb,
                                               hidden: hidden,
                                               attn: oOut,
                                               denseX: denseX,
                                               routedX: routedX,
                                               routerX: routerInput,
                                               postAttentionWeight: postAttn.buffer,
                                               postAttentionWeightOffset: Int(postAttn.offset),
                                               preFFNWeight: preFFN.buffer,
                                               preFFNWeightOffset: Int(preFFN.offset),
                                               preFFN2Weight: preFFN2.buffer,
                                               preFFN2WeightOffset: Int(preFFN2.offset),
                                               d: D,
                                               eps: eps)
            }

            let gRouter: (MTLCommandBuffer) -> Void = { [self] cb in
                moe.encodeRouterGemma4(commandBuffer: cb,
                    weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                    scales:  routerW.buffer, scalesOffset:  Int(routerW.scaleOffset),
                    biases:  routerW.buffer, biasesOffset:  Int(routerW.biasOffset),
                    hidden: routerInput,
                    effectiveScale: effectiveScaleBuffers[L],
                    perExpertScale: perExpertScale.buffer,
                    perExpertScaleOffset: Int(perExpertScale.offset),
                    outIndices: outIndices, outWeights: outWeights,
                    numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts))
            }

            // Speculative router for layer L+1, encoded into the same command
            // buffer so it costs one extra 2048x128 GEMV and no extra sync. It
            // reads the *current* layer's post-attention hidden state, which
            // differs from the real input by only this layer's MLP delta — in a
            // deep residual stream that delta is small, so the top-K sets
            // largely agree.
            let nextRouter: (weights: TensorView, perExpertScale: TensorView)? = {
                guard expertLookaheadEnabled, L + 1 < cfg.numLayers else { return nil }
                guard let w = try? model.router(layer: L + 1),
                      let s = try? model.routerPerExpertScale(layer: L + 1) else { return nil }
                return (w, s)
            }()

            // The shared dense MLP depends only on denseX, not on the routed
            // experts. denseX is produced on-GPU by cb1's post-attention setup
            // and the weights are resident, so this block can be encoded before
            // cb1 has even run — no CPU readback is involved.
            let gSharedFFN: (MTLCommandBuffer) -> Void = { [self] cb in
                try! shared.encode(commandBuffer: cb,
                                   x: denseX,
                                   gate: sharedProj.gate,
                                   up: sharedProj.up,
                                   down: sharedProj.down,
                                   y: h1Buf,
                                   scratchGate: denseScratchGate,
                                   scratchUp: denseScratchUp,
                                   scratchAct: denseScratchAct)
            }
            let gSharedNorm: (MTLCommandBuffer) -> Void = { [self] cb in
                rms.encodeBF16W(commandBuffer: cb, x: h1Buf,
                                weight: sharedProj.postF1.buffer,
                                weightOffset: Int(sharedProj.postF1.offset),
                                out: h1Buf, d: D, eps: eps)
            }
            let encodeAndCommitSharedCB: () -> MTLCommandBuffer = { [self] in
                let scb = ctx.queue.makeCommandBuffer()!
                gSharedFFN(scb)
                gSharedNorm(scb)
                scb.commit()
                cbCommitCount &+= 1
                return scb
            }
            // B1 experiment: optionally fuse the shared MLP into cb1
            // (TURBO_FIELDFARE_FUSE_SHARED=1). Confirmed regression on a clean
            // machine (see knob doc); off by default.
            let fusedSharedIntoCB1 = fuseSharedIntoCB1

            let cb = ctx.queue.makeCommandBuffer()!
            gInputNorm(cb)
            gQKV(cb)
            gQKVEpilogue(cb)
            gAttention(cb)
            gOProj(cb)
            gPostAttnSetup(cb)
            gRouter(cb)
            if let nextRouter {
                moe.encodeRouterGemma4(commandBuffer: cb,
                    weights: nextRouter.weights.buffer,
                    weightsOffset: Int(nextRouter.weights.offset),
                    scales: nextRouter.weights.buffer,
                    scalesOffset: Int(nextRouter.weights.scaleOffset),
                    biases: nextRouter.weights.buffer,
                    biasesOffset: Int(nextRouter.weights.biasOffset),
                    hidden: routerInput,
                    effectiveScale: effectiveScaleBuffers[L + 1],
                    perExpertScale: nextRouter.perExpertScale.buffer,
                    perExpertScaleOffset: Int(nextRouter.perExpertScale.offset),
                    outIndices: predIndices, outWeights: predWeights,
                    numExperts: UInt32(cfg.numExperts), d: D,
                    topK: UInt32(cfg.topKExperts))
            }
            if fusedSharedIntoCB1 {
                gSharedFFN(cb)
                gSharedNorm(cb)
            }
            // gpuStartTime (Metal, mach_absolute_time seconds) and
            // CLOCK_UPTIME_RAW are on the same time base, so the schedule/wake
            // arithmetic below is meaningful. Captured pre-commit (the only
            // unconditional hot-path cost; ~10-20ns clock read).
            let tCommit = gpuTimingEnabled ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0
            cb.commit()
            cbCommitCount &+= 1
            // Split path: queue the shared FFN immediately behind cb1 on the
            // same queue so the GPU has work waiting the instant attention
            // retires while the CPU reads router indices and plans experts.
            var sharedCB: MTLCommandBuffer? = fusedSharedIntoCB1 ? nil
                : (earlySharedCommitEnabled ? encodeAndCommitSharedCB() : nil)
            let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            waitForCompletionPolling(cb)
            let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
            cb1WaitNanos &+= waitNanos
            let tCb1Done = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if gpuTimingEnabled {
                accumulateGPUTime(cb, into: &gpuAttentionNanos, label: 0)
                let nowN = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                let gpuStartN = UInt64(cb.gpuStartTime * 1e9)
                let gpuEndN = UInt64(cb.gpuEndTime * 1e9)
                if gpuStartN > tCommit { cb1ScheduleNanos &+= gpuStartN - tCommit }
                if nowN > gpuEndN { cb1WakeNanos &+= nowN - gpuEndN }
                // NOTE: in the fused path the shared kernels run inside cb1, so
                // there is no separate shared span to attribute — the whole cb1
                // span is (correctly) counted as attention. gpuSharedNanos stays
                // 0 in fused mode; do NOT double-attribute the cb1 span to it
                // (that inflated sharedFFN to ~= attn, see 2026-08-07 review).
            }
            if let pending = pendingRoutedCommand {
                finishPendingRoutedCommand(pending, waitIfNeeded: false)
                pendingRoutedCommand = nil
            }
            totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start - waitNanos

            // CPU readback to fetch routed-expert blobs from disk.
            let tReadback = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let idxPtr = outIndices.contents().bindMemory(to: UInt32.self,
                                                          capacity: cfg.topKExperts)
            var experts = [Int](repeating: 0, count: cfg.topKExperts)
            for i in 0..<cfg.topKExperts {
                experts[i] = min(Int(idxPtr[i]), cfg.numExperts - 1)
            }
            if unionStatsEnabled {
                let poolSet = L < hotPoolSets.count ? hotPoolSets[L] : []
                if unionDumpPath != nil {
                    let ids = experts.sorted().map(String.init).joined(separator: " ")
                    unionDumpLines.append("t=1,L=\(L),\(ids)")
                }
                let covered = experts.filter { poolSet.contains($0) }.count
                var b = unionStats[1, default: UnionStatsBucket()]
                b.visits += 1
                b.unionTotal += cfg.topKExperts
                b.unionPool += covered
                b.perTokenTotal += cfg.topKExperts
                b.perTokenPool += covered
                unionStats[1] = b
            }
            readbackNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tReadback

            if expertLookaheadEnabled {
                if let predicted = pendingLookahead, predicted.layer == L {
                    let actual = Set(experts)
                    let overlap = predicted.experts.filter { actual.contains($0) }.count
                    lookaheadHits &+= UInt64(overlap)
                    lookaheadPredictions &+= UInt64(cfg.topKExperts)
                }
                pendingLookahead = nil
                if nextRouter != nil {
                    let predPtr = predIndices.contents().bindMemory(
                        to: UInt32.self, capacity: cfg.topKExperts)
                    var predicted = [Int](repeating: 0, count: cfg.topKExperts)
                    for i in 0..<cfg.topKExperts {
                        predicted[i] = min(Int(predPtr[i]), cfg.numExperts - 1)
                    }
                    pendingLookahead = (layer: L + 1, experts: predicted)
                    if missPrefetchEnabled,
                       let predicted = pendingLookahead,
                       predicted.layer < hotPoolSets.count {
                        // Long-tail only: pool members are already resident and
                        // the plan skips LRU residents, so this filter keeps the
                        // background dispatch off the per-layer hot path.
                        let known = knownMissersByLayer[predicted.layer, default: []]
                        let nonPool = predicted.experts.filter {
                            !hotPoolSets[predicted.layer].contains($0)
                                && known.contains($0)
                        }
                        if !nonPool.isEmpty {
                            let ctxForPrefetch = makeDecodeAccessContext()
                            let capturedModel = model
                            let handle = ExpertPrefetchHandle()
                            expertPrefetchHandle = handle
                            missPrefetchDispatches &+= 1
                            expertPrefetchQueue.async {
                                handle.finish((try? capturedModel.prefetchRoutedExperts(
                                    layer: predicted.layer,
                                    experts: nonPool,
                                    accessContext: ctxForPrefetch)) ?? nil)
                            }
                        }
                    }
                }
            }

            // A speculative read for this layer may still be in flight. It owns
            // slots that the planner below is about to hand out, so drain it
            // before planning. Ideally this costs nothing: the read was issued
            // a layer ago and had the whole GPU pass to complete.
            if let handle = expertPrefetchHandle {
                let tDrain = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                let readCount = handle.drain()
                prefetchDrainNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tDrain
                prefetchedExperts &+= UInt64(readCount ?? 0)
                expertPrefetchHandle = nil
            } else if let task = expertPrefetchTask {
                let tDrain = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                let readCount = await task.value
                prefetchDrainNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tDrain
                prefetchedExperts &+= UInt64(readCount ?? 0)
                expertPrefetchTask = nil
            }

            let routedOffsets = model.routedExpertOffsets(layer: L)
            let topK = UInt32(cfg.topKExperts)
            let canPlanPhase1HitSplit =
                cfg.topKExperts <= MoE.maxStreamedExperts
            let tPlanStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let plannedFetch = canPlanPhase1HitSplit
                ? try model.planRoutedExperts(layer: L,
                                             experts: experts,
                                             accessContext: makeDecodeAccessContext())
                : nil
            planNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tPlanStart
            if let plan = plannedFetch, !plan.misses.isEmpty {
                var set = knownMissersByLayer[L, default: []]
                for idx in plan.misses { set.insert(experts[idx]) }
                knownMissersByLayer[L] = set
            }
            var phase1HitCB: MTLCommandBuffer?
            var phase1HitSplitArgBuf: MTLBuffer?
            var phase1HitSplitRoutedBufs: [MoE.RoutedBlob] = []
            var phase1HitSlots: [UInt32] = []
            var phase1MissSlots: [UInt32] = []

            if let plan = plannedFetch {
                let missSet = Set(plan.misses)
                phase1HitSlots = (0..<cfg.topKExperts)
                    .filter { !missSet.contains($0) }
                    .map { UInt32($0) }
                phase1MissSlots = plan.misses.map { UInt32($0) }
            }
            func encodeRoutedPhase1Full(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [MoE.RoutedBlob]
            ) {
                moe.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                                                        routedArgBuffer: argBuf,
                                                        routedBlobs: routedBufs,
                                                        routedOffsets: routedOffsets,
                                                        x: routedX,
                                                        acts: moeActs,
                                                        d: D,
                                                        f: FmoE,
                                                        topK: topK,
                                                        bits: model.routedExpertWeightBits)
            }

            func encodeRoutedPhase1Subset(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [MoE.RoutedBlob],
                activeSlots: MTLBuffer,
                activeSlotIndices: [UInt32],
                activeCount: UInt32
            ) {
                moe.encodeRoutedPersistentPhase1SubsetU16Load(
                    commandBuffer: cb,
                    routedArgBuffer: argBuf,
                    routedBlobs: routedBufs,
                    routedOffsets: routedOffsets,
                    x: routedX,
                    acts: moeActs,
                    activeSlots: activeSlots,
                    activeSlotIndices: activeSlotIndices,
                    activeCount: activeCount,
                    d: D,
                    f: FmoE,
                    topK: topK,
                    bits: model.routedExpertWeightBits)
            }

            // INVARIANT: this pre-fetch arg buffer is only ever consumed by the
            // SUBSET phase1 encoder below (active indices = hits only). Miss
            // slots carry placeholder offsets here (slotExpert is still -1
            // until the fetch runs) and are never dereferenced by phase1;
            // phase2 re-encodes from post-fetch pairs. Do not switch this CB
            // to a full encode without rebuilding the arg buffer after fetch.
            if let plan = plannedFetch,
               plan.hits > 0,
               !plan.misses.isEmpty {
                let plannedBlobs = try model.routedExpertBuffers(for: plan)
                phase1HitSplitRoutedBufs = plannedBlobs.map {
                    MoE.RoutedBlob(buffer: $0.buffer, offset: $0.offset)
                }
                phase1HitSplitArgBuf = moe.makeRoutedArgumentBuffer(
                    routedBlobs: phase1HitSplitRoutedBufs,
                    topK: topK)
                if let argBuf = phase1HitSplitArgBuf, plan.hits > 0, !plan.misses.isEmpty {
                    writeActiveSlots(phase1HitSlots, into: moeHitActiveSlots)
                    let cb = ctx.queue.makeCommandBuffer()!
                    encodeRoutedPhase1Subset(
                        cb,
                        argBuf: argBuf,
                        routedBufs: phase1HitSplitRoutedBufs,
                        activeSlots: moeHitActiveSlots,
                        activeSlotIndices: phase1HitSlots,
                        activeCount: UInt32(phase1HitSlots.count))
                    phase1HitCB = cb
                }
            }

            // Late-split path (TURBO_FIELDFARE_EARLY_SHARED=0): issue the shared
            // FFN here, still without waiting, so its GPU work overlaps the
            // routed-expert pread. The routed CB follows it on the same queue,
            // so the combine sees h1Buf. (Fused path already ran it inside cb1.)
            if sharedCB == nil, !fusedSharedIntoCB1 {
                sharedCB = encodeAndCommitSharedCB()
            }
            if let cb = phase1HitCB {
                cb.commit()
                cbCommitCount &+= 1
            }
            if rdadviseEnabled && rdadvisePolicyMode != .off {
                let requestedMisses = plannedFetch?.misses.count ?? experts.count
                let estimatedAdviceBytes = try model.routedExpertAdviceByteEstimate(
                    layer: L,
                    missCount: requestedMisses)
                if let skipped = shouldSkipRDAdvice(position: position,
                                                    requestedMisses: requestedMisses,
                                                    estimatedBytes: estimatedAdviceBytes,
                                                    canOverlapUsefulGPUWork: true) {
                    recordRDAdvice(skipped, wallNanos: 0)
                } else {
                    let tAdvice = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    let result: ExpertIOAdviceResult
                    if let plannedFetch {
                        result = try model.adviseRoutedExperts(plan: plannedFetch)
                    } else {
                        result = try model.adviseRoutedExperts(layer: L, experts: experts)
                    }
                    let wallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tAdvice
                    recordRDAdvice(result, wallNanos: wallNanos)
                    updateRDAdvicePolicy(after: result, position: position)
                }
            }

            // Routed-expert pread — overlaps the shared MLP GPU work above.
            // B4: when the plan has zero misses every expert is already in a
            // slot, so the fetch is pure CPU bookkeeping. The generic async
            // path still pays a continuation + global-queue dispatch hop on
            // every layer (~100s of us); the sync fast path runs it inline on
            // the decode thread. Hit rate is ~99% with the hot pool, so this
            // is the common case. Miss-bearing layers keep the async path.
            let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let blobs: [TensorView]
            if let plannedFetch, plannedFetch.misses.isEmpty,
               b4HitOnlySyncEnabled {
                let tF = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                blobs = try model.fetchRoutedExpertsHitOnlySync(
                    plan: plannedFetch,
                    accessContext: makeDecodeAccessContext()).views
                fetchHitOnlyNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) &- tF
                fetchHitOnlyCount &+= 1
            } else if let plannedFetch {
                let tF = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                blobs = try await model.fetchRoutedExperts(
                    plan: plannedFetch,
                    accessContext: makeDecodeAccessContext())
                fetchAsyncPlanNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) &- tF
                fetchAsyncPlanCount &+= 1
            } else {
                let tF = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                blobs = try await model.fetchRoutedExperts(
                    layer: L,
                    experts: experts,
                    accessContext: makeDecodeAccessContext())
                fetchAsyncNoPlanNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) &- tF
                fetchAsyncNoPlanCount &+= 1
            }
            let layerIo = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart
            totalIoNanos &+= layerIo

            // Speculatively warm the next layer now that this layer's blocking
            // read is done — the drive is idle from here until the next real
            // fetch, and the GPU has a full MoE pass to chew on. Issued after
            // the critical read so it never competes with it for queue depth.
            if expertPrefetchEnabled, !missPrefetchEnabled,
               let predicted = pendingLookahead,
               predicted.layer == L + 1 {
                let ctxForPrefetch = makeDecodeAccessContext()
                let capturedModel = model
                if syncPrefetchDrainEnabled {
                    let handle = ExpertPrefetchHandle()
                    expertPrefetchHandle = handle
                    expertPrefetchQueue.async {
                        handle.finish((try? capturedModel.prefetchRoutedExperts(
                            layer: predicted.layer,
                            experts: predicted.experts,
                            accessContext: ctxForPrefetch)) ?? nil)
                    }
                } else {
                    expertPrefetchTask = Task.detached(priority: .userInitiated) {
                        (try? capturedModel.prefetchRoutedExperts(
                            layer: predicted.layer,
                            experts: predicted.experts,
                            accessContext: ctxForPrefetch)) ?? nil
                    }
                }
            }

            let routedBufs = blobs.map {
                MoE.RoutedBlob(buffer: $0.buffer, offset: $0.offset)
            }
            let tCb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let scalarPtr = layerScalarView.buffer.contents()
                .advanced(by: Int(layerScalarView.offset))
                .assumingMemoryBound(to: UInt16.self)
            let layerScalar = Quantization.bf16ToFloat(scalarPtr[0])

            let gTail: (MTLCommandBuffer) -> Void = { [self] cb in
                fusedTail.encode(commandBuffer: cb,
                                 h2: h2Buf,
                                 h1: h1Buf,
                                 hidden: hidden,
                                 postFFN2Weight: postF2.buffer,
                                 postFFN2WeightOffset: Int(postF2.offset),
                                 postFFNWeight: postF.buffer,
                                 postFFNWeightOffset: Int(postF.offset),
                                 d: D,
                                 eps: eps,
                                 layerScalar: layerScalar)
            }
            let routedCB = ctx.queue.makeCommandBuffer()!
            // Re-encode the split arg buffer from the *fresh* `routedBufs` (the
            // post-fetch views). In mmap mode slot→expert mapping only becomes
            // final after fetchRoutedExperts loads the misses, so offsets
            // captured before (phase1HitSplitArgBuf) are stale for miss slots;
            // pread mode is unaffected (slot buffers are stable). Re-encoding
            // here costs the same as the non-split path.
            let splitArgBuf = phase1HitCB != nil && !phase1MissSlots.isEmpty
                ? moe.makeReusedRoutedArgumentBuffer(routedBlobs: routedBufs, topK: topK)
                : nil
            let argBuf = splitArgBuf ?? moe.makeReusedRoutedArgumentBuffer(
                routedBlobs: routedBufs,
                topK: topK)
            if splitArgBuf != nil {
                writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
                encodeRoutedPhase1Subset(
                    routedCB,
                    argBuf: argBuf,
                    routedBufs: routedBufs,
                    activeSlots: moeMissActiveSlots,
                    activeSlotIndices: phase1MissSlots,
                    activeCount: UInt32(phase1MissSlots.count))
            } else {
                encodeRoutedPhase1Full(routedCB,
                                       argBuf: argBuf,
                                       routedBufs: routedBufs)
            }
            moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: routedCB,
                                                   routedArgBuffer: argBuf,
                                                   routedBlobs: routedBufs,
                                                   routedOffsets: routedOffsets,
                                                   acts: moeActs,
                                                   routingWeights: outWeights,
                                                   residual: zeroResidual,
                                                   y: h2Buf,
                                                   d: D,
                                                   f: FmoE,
                                                   topK: topK,
                                                   bits: model.routedExpertWeightBits,
                                                   chunk: Self.phase2Chunk)
            gTail(routedCB)
            routedCB.commit()
            cbCommitCount &+= 1
            chainWallNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Done
            precondition(pendingRoutedCommand == nil,
                         "routed command-buffer pipeline drained before queuing the next layer")
            pendingRoutedCommand = PendingRoutedCommand(
                cb: routedCB,
                sharedCB: sharedCB,
                phase1HitCB: phase1HitCB,
                encodeAndCommitNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb2Start)
            continue
        }
        if let pending = pendingRoutedCommand {
            finishPendingRoutedCommand(pending, waitIfNeeded: true)
            pendingRoutedCommand = nil
        }

        // The fused head skips the vocab buffer and leaves a greedy token in
        // greedyTokenBuf; the logits path writes the complete vector.
        let fNorm = model.finalNorm
        let lm    = model.embedding
        let gFinalNorm: (MTLCommandBuffer) -> Void = { cb in
            self.rms.encodeBF16W(commandBuffer: cb, x: self.hidden,
                                 weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                                 out: self.normed, d: D, eps: eps)
        }
        let gLmHead: (MTLCommandBuffer) -> Void = { cb in
            self.int4.encodeTgx(commandBuffer: cb,
                                weights: lm.buffer, weightsOffset: Int(lm.offset),
                                scales:  lm.buffer, scalesOffset:  Int(lm.scaleOffset),
                                biases:  lm.buffer, biasesOffset:  Int(lm.biasOffset),
                                x: self.normed, y: logits, m: UInt32(self.cfg.vocabSize), n: D)
        }
        let gFusionHead: (MTLCommandBuffer) -> Void = { cb in
            self.fusionHead.encodeGreedyDecode(
                commandBuffer: cb,
                hidden: self.hidden,
                normWeight: fNorm.buffer, normOffset: Int(fNorm.offset),
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: self.greedyTokenBuf,
                d: D, vocab: UInt32(self.cfg.vocabSize),
                rmsEps: eps)
        }
        if emitHead {
            let useFusedHeadForThisToken = useFusedGreedyHead && outputMode == .greedyIfAvailable
            let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if useFusedHeadForThisToken {
                runSync(gFusionHead)
                totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            } else {
                runSync { cb in
                    gFinalNorm(cb)
                    gLmHead(cb)
                }
                totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            }
        }

        kv?.advance()
    }

    private func runSync(_ body: (MTLCommandBuffer) -> Void) {
        let cb = ctx.queue.makeCommandBuffer()!
        body(cb)
        cb.commit()
        cbCommitCount &+= 1
        cb.waitUntilCompleted()
        if let err = cb.error {
            print("CB error: \(err)")
        }
        if gpuTimingEnabled {
            accumulateGPUTime(cb, into: &gpuHeadNanos, label: 4)
        }
    }

    /// Optional bounded spin-wait on MTLCommandBuffer.status before falling
    /// back to waitUntilCompleted. The decode thread has no other work during
    /// the cb1 wait (serial chain), so polling `status` (a mapped field) lets
    /// the chain start ~100us sooner than the semaphore wake path; the GPU is
    /// busy with the early-committed shared FFN meanwhile, so the wake delay
    /// directly pushes the routedCB commit later. Env: TURBO_FIELDFARE_WAKE_POLL_US
    /// (0 = off, default). sched_yield() between polls keeps other threads runnable.
    /// B3 phase-2 ROI probe (documented in §13.17): skip the attention
    /// core kernels entirely (QKV / epilogue / OProj still run). Delta in
    /// [gpu] attn isolates how much of cb1's GPU time is the core. These
    /// are diagnostic-only timing switches, default off, NOT wired to prod.
    private static let skipAttentionCore: Bool = {
        ProcessInfo.processInfo.environment["TURBO_FIELDFARE_SKIP_ATTN_CORE"] == "1"
    }()
    private static let skipAttentionSWA: Bool = {
        ProcessInfo.processInfo.environment["TURBO_FIELDFARE_SKIP_ATTN_SWA"] == "1"
    }()
    private static let skipAttentionFull: Bool = {
        ProcessInfo.processInfo.environment["TURBO_FIELDFARE_SKIP_ATTN_FULL"] == "1"
    }()

    private static let wakePollSpinNanos: UInt64 = {
        guard let v = ProcessInfo.processInfo.environment["TURBO_FIELDFARE_WAKE_POLL_US"],
              let us = Int(v), us > 0 else { return 0 }
        return UInt64(us) &* 1000
    }()

    /// Phase-2 chunked decode (smem acts reuse). TURBO_FIELDFARE_PHASE2_CHUNK
    /// > 1 switches to the moe_phase2_down_reduce_k8_chunked kernel (each TG
    /// computes `chunk` output dims; acts read once per TG instead of per d).
    private static let phase2Chunk: UInt32 = {
        guard let v = ProcessInfo.processInfo.environment["TURBO_FIELDFARE_PHASE2_CHUNK"],
              let n = UInt32(v), n > 1 else { return 1 }
        return n
    }()

    private nonisolated func waitForCompletionPolling(_ cb: MTLCommandBuffer) {
        let spin = Self.wakePollSpinNanos
        if spin > 0 {
            let deadline = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) &+ spin
            repeat {
                // .completed -> the chain can start now. .error is also
                // terminal: fall through so waitForCompletion prints it.
                if cb.status == .completed {
                    if let err = cb.error {
                        print("CB error: \(err)")
                    }
                    return
                }
                if cb.status == .error {
                    break
                }
                sched_yield()
            } while clock_gettime_nsec_np(CLOCK_UPTIME_RAW) < deadline
        }
        waitForCompletion(cb)
    }

    private nonisolated func waitForCompletion(_ cb: MTLCommandBuffer) {
        cb.waitUntilCompleted()
        if let err = cb.error {
            print("CB error: \(err)")
        }
    }

    private nonisolated func timedWaitForCompletion(_ cb: MTLCommandBuffer) -> UInt64 {
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        waitForCompletion(cb)
        return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start
    }

}
