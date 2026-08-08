import Foundation
import Metal

/// Attention kind for a Gemma 4 assistant layer.
public enum AssistantBridgeAttentionKind: String, Sendable {
    case slidingAttention
    case fullAttention
}

/// A snapshot of one KV stream (key or value) from the target model,
/// captured for the assistant drafter to read during speculation.
public struct AssistantBridgeKVBuffer: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: Int
    public let stride: Int
    public let startSlot: Int
    public let validTokenCount: Int
    /// Ring slot count for this layer, i.e. `KVCacheManager.ringCapacity(layer:)`
    /// (0 for non-ring layers). Must be carried explicitly: `buffer.length /
    /// stride` overestimates it because Metal rounds allocations up to a page
    /// boundary, which would give the drafter a wrong modulus once the sliding
    /// window actually wraps.
    public let ringCapacity: Int

    public init(buffer: MTLBuffer, offset: Int = 0, stride: Int,
                startSlot: Int = 0, validTokenCount: Int,
                ringCapacity: Int = 0) {
        self.buffer = buffer
        self.offset = offset
        self.stride = stride
        self.startSlot = startSlot
        self.validTokenCount = validTokenCount
        self.ringCapacity = ringCapacity
    }
}

/// A key/value pair snapshot for one attention kind.
public struct AssistantBridgeKVSnapshot: Sendable {
    public let kind: AssistantBridgeAttentionKind
    public let headDim: Int
    public let numKVHeads: Int
    public let key: AssistantBridgeKVBuffer
    public let value: AssistantBridgeKVBuffer

    public init(kind: AssistantBridgeAttentionKind,
                headDim: Int,
                numKVHeads: Int,
                key: AssistantBridgeKVBuffer,
                value: AssistantBridgeKVBuffer) {
        self.kind = kind
        self.headDim = headDim
        self.numKVHeads = numKVHeads
        self.key = key
        self.value = value
    }
}

/// Full bridge snapshot passed from the target model to the MTP drafter.
///
/// The assistant declares `num_kv_shared_layers == num_hidden_layers`, i.e. it
/// owns no `k_proj`/`v_proj` and instead reuses the target's KV cache.
///
/// The sharing is **by layer type, not by position**. The HF reference
/// (`Gemma4AssistantForCausalLM.forward`) takes a `shared_kv_states` dict with
/// exactly two entries — `"full_attention"` and `"sliding_attention"` — and
/// `Gemma4TextAttention.store_full_length_kv` publishes only the *last*
/// non-KV-shared layer of each type. For the 26B target (`layer_types` ending
/// `… 26 sliding, 27 sliding, 28 sliding, 29 full`, `num_kv_shared_layers = 0`)
/// that is layer 28 for sliding and layer 29 for full. Every assistant sliding
/// layer reads the *same* layer-28 KV; its single full layer reads layer 29.
///
/// An earlier revision mapped assistant layer `i` onto target layer
/// `targetLayers - assistantLayers + i` (26/27/28/29). That is not what the
/// reference does and fed layers 0–1 the wrong (shallower) KV.
public struct AssistantBridgeSnapshot: Sendable {
    public let lastHiddenState: [Float]
    public let lastTokenEmbedding: [Float]?
    public let fullAttentionKV: AssistantBridgeKVSnapshot?
    public let slidingAttentionKV: AssistantBridgeKVSnapshot?

    public init(lastHiddenState: [Float],
                lastTokenEmbedding: [Float]? = nil,
                fullAttentionKV: AssistantBridgeKVSnapshot? = nil,
                slidingAttentionKV: AssistantBridgeKVSnapshot? = nil) {
        self.lastHiddenState = lastHiddenState
        self.lastTokenEmbedding = lastTokenEmbedding
        self.fullAttentionKV = fullAttentionKV
        self.slidingAttentionKV = slidingAttentionKV
    }

    /// KV the assistant should attend to for a layer of the given type.
    public func kv(kind: AssistantBridgeAttentionKind) -> AssistantBridgeKVSnapshot? {
        kind == .fullAttention ? fullAttentionKV : slidingAttentionKV
    }
}
