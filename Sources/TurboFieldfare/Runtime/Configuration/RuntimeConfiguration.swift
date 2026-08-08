public enum RuntimeHeadPath: String, Codable, Sendable {
    case fusedRows = "fused-rows"
    case logits
}

public enum RuntimePrefillPolicy: String, Codable, Sendable {
    case off
    case chunked
}

public enum RuntimePrefillAttentionPath: String, Codable, Sendable {
    case causalTiled = "causal-tiled"
    case fullTensorOps2DPreferred = "full-tensorops-2d-preferred"
    case fullTensorOps2DValidityV2 = "full-tensorops-2d-validity-v2"
}

public enum RuntimeExpertCachePolicy: String, Codable, Sendable {
    case lfu
    case lru
}

public enum RuntimePrefillExpertReadMode: String, Codable, Sendable, Equatable {
    case baseline
    case coalesced
    case layerLocalReadahead = "layer-local-readahead"
}

public struct RuntimeConfiguration: Sendable, Equatable {
    /// Slots are per layer, so resident expert bytes scale as
    /// `slots * layers * expertStride` (3.2MB stride over 30 layers is ~96MB
    /// per slot). The large values only pay off on machines that can hold the
    /// hot working set; a trace-driven replay showed LRU hit rate climbing
    /// 57.6% -> 81.5% -> ~93% across 16 -> 32 -> 128 slots.
    public static let allowedExpertCacheSlots = [8, 16, 24, 32, 48, 64, 96, 128]
    public static let allowedPrefillChunkTokens = [32, 64, 128]

    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let prefillAttentionPath: RuntimePrefillAttentionPath
    public let prefillExpertReadMode: RuntimePrefillExpertReadMode
    public let prefillExpertLayerLocalReadaheadExperts: Int
    public let prefillExpertBoundedCoalescedRunExperts: Int
    public let prefillExpertBoundedParallelMissReadWorkers: Int
    public let headPath: RuntimeHeadPath

    public init(expertCacheSlots: Int = 16,
                expertCachePolicy: RuntimeExpertCachePolicy = .lfu,
                rdadvisePolicy: RDAdvicePolicyMode = .off,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 128,
                prefillAttentionPath: RuntimePrefillAttentionPath = .fullTensorOps2DPreferred,
                prefillExpertReadMode: RuntimePrefillExpertReadMode = .baseline,
                prefillExpertLayerLocalReadaheadExperts: Int = 16,
                prefillExpertBoundedCoalescedRunExperts: Int = 4,
                prefillExpertBoundedParallelMissReadWorkers: Int = 2,
                forceLogitsHead: Bool = false) {
        precondition(Self.allowedExpertCacheSlots.contains(expertCacheSlots),
                     "unsupported expert-cache slot count")
        precondition(Self.allowedPrefillChunkTokens.contains(prefillChunkTokens),
                     "unsupported prefill chunk size")
        precondition(prefillExpertLayerLocalReadaheadExperts > 0,
                     "prefill expert layer-local readahead experts must be positive")
        precondition(prefillExpertBoundedCoalescedRunExperts > 0,
                     "prefill expert bounded coalesced run experts must be positive")
        precondition(prefillExpertBoundedParallelMissReadWorkers > 0,
                     "prefill expert bounded parallel miss read workers must be positive")
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.prefillPolicy = prefillEnabled ? .chunked : .off
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillAttentionPath = prefillAttentionPath
        self.prefillExpertReadMode = prefillExpertReadMode
        self.prefillExpertLayerLocalReadaheadExperts = prefillExpertLayerLocalReadaheadExperts
        self.prefillExpertBoundedCoalescedRunExperts = prefillExpertBoundedCoalescedRunExperts
        self.prefillExpertBoundedParallelMissReadWorkers = prefillExpertBoundedParallelMissReadWorkers
        self.headPath = forceLogitsHead ? .logits : .fusedRows
    }

    public static var production: RuntimeConfiguration {
        RuntimeConfiguration()
    }

    public var fp16RingEnabled: Bool { true }
    public var rdadviseEnabled: Bool { rdadvisePolicy != .off }
    public var prefillConfig: PrefillRuntimeConfig {
        switch prefillPolicy {
        case .off:
            return .off
        case .chunked:
            return .production(chunkTokens: prefillChunkTokens)
        }
    }
    public var modelExpertCachePolicy: ExpertCachePolicy {
        expertCachePolicy == .lru ? .lru : .lfu
    }
}
