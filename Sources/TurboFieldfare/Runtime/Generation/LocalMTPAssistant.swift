import Accelerate
import Foundation
import Metal

// MARK: - Error

public enum GemmaAssistantError: Error, CustomStringConvertible {
    case missingFile(String)
    case invalidConfig(String)
    case invalidTensor(name: String, detail: String)
    case unsupportedDType(name: String, dtype: String)
    case hiddenStateMismatch(expected: Int, actual: Int)
    case tokenOutOfRange(Int32)
    case allocationFailed(String)

    public var description: String {
        switch self {
        case .missingFile(let path):
            return "gemma assistant file is missing: \(path)"
        case .invalidConfig(let detail):
            return "gemma assistant config is invalid: \(detail)"
        case .invalidTensor(let name, let detail):
            return "gemma assistant tensor \(name) is invalid: \(detail)"
        case .unsupportedDType(let name, let dtype):
            return "gemma assistant tensor \(name) uses unsupported dtype \(dtype)"
        case .hiddenStateMismatch(let expected, let actual):
            return "gemma assistant hidden state size mismatch: expected \(expected), got \(actual)"
        case .tokenOutOfRange(let token):
            return "gemma assistant token id \(token) is out of range"
        case .allocationFailed(let detail):
            return "gemma assistant metal allocation failed: \(detail)"
        }
    }
}

// MARK: - Config (parsed from config.json)

struct GemmaAssistantConfig: Decodable {
    struct TextConfig: Decodable {
        let hiddenSize: Int
        let intermediateSize: Int
        let numHiddenLayers: Int
        let numAttentionHeads: Int
        let headDim: Int
        let globalHeadDim: Int
        let numKeyValueHeads: Int
        let numGlobalKeyValueHeads: Int
        let vocabSize: Int
        let rmsNormEps: Float
        let layerTypes: [String]
        let slidingWindow: Int
    }

    let backboneHiddenSize: Int
    let textConfig: TextConfig

    var hiddenSize: Int { textConfig.hiddenSize }
    var intermediateSize: Int { textConfig.intermediateSize }
    var vocabSize: Int { textConfig.vocabSize }
    var eps: Float { textConfig.rmsNormEps }

    static func load(from modelDirectory: URL) throws -> GemmaAssistantConfig {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw GemmaAssistantError.missingFile(configURL.path)
        }
        let configData = try Data(contentsOf: configURL, options: [.mappedIfSafe])
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(GemmaAssistantConfig.self, from: configData)
        guard config.textConfig.hiddenSize > 0,
              config.backboneHiddenSize > 0,
              config.textConfig.numHiddenLayers >= 0,
              config.textConfig.numAttentionHeads > 0,
              config.textConfig.headDim > 0,
              config.textConfig.vocabSize > 0 else {
            throw GemmaAssistantError.invalidConfig("non-positive dimensions")
        }
        return config
    }

    struct LayerInfo {
        let attentionKind: AssistantBridgeAttentionKind
        let headDim: Int
        let numHeads: Int
        let numKVHeads: Int
        let qRows: Int
        let layerScalar: Float
        let hasPostFeedforwardNorm: Bool
    }

    func layerInfo(index: Int) -> LayerInfo {
        let layerType = index < textConfig.layerTypes.count
            ? textConfig.layerTypes[index]
            : "sliding_attention"
        let isFull = layerType == "full_attention"
        let layerHeadDim = isFull ? textConfig.globalHeadDim : textConfig.headDim
        let qProjRows = textConfig.numAttentionHeads * layerHeadDim
        return LayerInfo(
            attentionKind: isFull ? .fullAttention : .slidingAttention,
            headDim: layerHeadDim,
            numHeads: textConfig.numAttentionHeads,
            numKVHeads: isFull
                ? textConfig.numGlobalKeyValueHeads
                : textConfig.numKeyValueHeads,
            qRows: qProjRows,
            layerScalar: 1.0, // read from safetensors
            hasPostFeedforwardNorm: false // detected at load time
        )
    }
}

// MARK: - Safetensors archive reader

struct GemmaAssistantTensorArchive {
    struct HeaderEntry: Decodable {
        let dataOffsets: [Int]?
        let dtype: String?
        let shape: [Int]?
    }

    private let data: Data
    private let entries: [String: HeaderEntry]
    private let headerSize: Int

    init(url: URL) throws {
        self.data = try Data(contentsOf: url, options: [.mappedIfSafe])
        guard data.count >= 8 else {
            throw GemmaAssistantError.invalidConfig("safetensors header prefix is truncated")
        }
        let rawHeaderSize = data.prefix(8).withUnsafeBytes { raw -> UInt64 in
            raw.loadUnaligned(as: UInt64.self).littleEndian
        }
        let headerSize = Int(rawHeaderSize)
        guard headerSize > 0, data.count >= 8 + headerSize else {
            throw GemmaAssistantError.invalidConfig("safetensors header size is out of range")
        }
        let headerData = data.subdata(in: 8..<(8 + headerSize))
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        self.entries = try decoder.decode([String: HeaderEntry].self, from: headerData)
        self.headerSize = headerSize
    }

    func hasTensor(_ name: String) -> Bool {
        entries[name] != nil
    }

    func scalar(name: String) throws -> Float {
        let (_, _, start, _) = try validatedTensor(name: name, expectedShape: [1])
        return try data.withUnsafeBytes { raw -> Float in
            guard let base = raw.baseAddress else {
                throw GemmaAssistantError.invalidTensor(name: name, detail: "raw buffer is unavailable")
            }
            let tensorBase = base.advanced(by: start)
            switch entries[name]?.dtype {
            case "BF16":
                let values = tensorBase.bindMemory(to: UInt16.self, capacity: 1)
                return Float(bitPattern: UInt32(values[0]) << 16)
            case "F16":
                let values = tensorBase.bindMemory(to: UInt16.self, capacity: 1)
                return Float(Float16(bitPattern: values[0]))
            case "F32":
                return tensorBase.bindMemory(to: Float.self, capacity: 1)[0]
            default:
                throw GemmaAssistantError.unsupportedDType(name: name, dtype: entries[name]?.dtype ?? "unknown")
            }
        }
    }

    func optionalBuffer(device: MTLDevice,
                        name: String,
                        expectedShape: [Int]) throws -> MTLBuffer? {
        guard entries[name] != nil else { return nil }
        return try buffer(device: device, name: name, expectedShape: expectedShape)
    }

    func buffer(device: MTLDevice,
                name: String,
                expectedShape: [Int]) throws -> MTLBuffer {
        let (count, _, start, end) = try validatedTensor(name: name, expectedShape: expectedShape)
        let byteCount = end - start
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw GemmaAssistantError.allocationFailed(name)
        }
        buffer.label = name
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            memcpy(buffer.contents(), base.advanced(by: start), byteCount)
        }
        precondition(count > 0)
        return buffer
    }

    private func validatedTensor(name: String,
                                 expectedShape: [Int]) throws -> (Int, HeaderEntry, Int, Int) {
        guard let entry = entries[name] else {
            throw GemmaAssistantError.invalidTensor(name: name, detail: "missing")
        }
        guard let shape = entry.shape else {
            throw GemmaAssistantError.invalidTensor(name: name, detail: "shape is missing")
        }
        guard shape == expectedShape else {
            throw GemmaAssistantError.invalidTensor(
                name: name,
                detail: "expected shape \(expectedShape), got \(shape)")
        }
        guard let dataOffsets = entry.dataOffsets,
              dataOffsets.count == 2 else {
            throw GemmaAssistantError.invalidTensor(name: name, detail: "data_offsets must have two values")
        }
        guard let dtype = entry.dtype else {
            throw GemmaAssistantError.invalidTensor(name: name, detail: "dtype is missing")
        }
        let start = 8 + headerSize + dataOffsets[0]
        let end = 8 + headerSize + dataOffsets[1]
        guard start >= 0, end <= data.count, end >= start else {
            throw GemmaAssistantError.invalidTensor(name: name, detail: "offset range is invalid")
        }
        let count = expectedShape.reduce(1, *)
        return (count, HeaderEntry(dataOffsets: dataOffsets, dtype: dtype, shape: shape), start, end)
    }
}

// MARK: - Metal kernel wrappers

private final class AssistantEmbedLookup {
    private let pso: MTLComputePipelineState
    init(context: MetalContext) throws {
        self.pso = try context.pipeline("assistant_embed_lookup_bf16")
    }
    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, token: MTLBuffer, output: MTLBuffer,
                rowWidth: UInt32) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: 0, index: 0)
        enc.setBuffer(token, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var w = rowWidth
        enc.setBytes(&w, length: MemoryLayout<UInt32>.size, index: 3)
        let threads = MTLSize(width: Int(rowWidth), height: 1, depth: 1)
        let tg = MTLSize(width: min(Int(pso.maxTotalThreadsPerThreadgroup), 256), height: 1, depth: 1)
        enc.dispatchThreads(threads, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
}

private final class AssistantConcat {
    private let pso: MTLComputePipelineState
    init(context: MetalContext) throws {
        self.pso = try context.pipeline("assistant_concat_halves")
    }
    func encode(commandBuffer: MTLCommandBuffer,
                lhs: MTLBuffer, rhs: MTLBuffer, output: MTLBuffer, count: UInt32) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(lhs, offset: 0, index: 0)
        enc.setBuffer(rhs, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var c = count
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)
        let threads = MTLSize(width: Int(count), height: 1, depth: 1)
        let tg = MTLSize(width: min(Int(pso.maxTotalThreadsPerThreadgroup), 256), height: 1, depth: 1)
        enc.dispatchThreads(threads, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
}

/// `output = (base + delta) * scale` — the residual add of
/// `Gemma4TextDecoderLayer`, with `layer_scalar` applied to the whole layer
/// output once. Use `scale: 1.0` for the intermediate residual add.
private final class AssistantAddThenScale {
    private let pso: MTLComputePipelineState
    init(context: MetalContext) throws {
        self.pso = try context.pipeline("assistant_add_then_scale")
    }
    func encode(commandBuffer: MTLCommandBuffer,
                base: MTLBuffer, delta: MTLBuffer, output: MTLBuffer,
                count: UInt32, scale: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(base, offset: 0, index: 0)
        enc.setBuffer(delta, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var c = count; var s = scale
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&s, length: MemoryLayout<Float>.size, index: 4)
        let threads = MTLSize(width: Int(count), height: 1, depth: 1)
        let tg = MTLSize(width: min(Int(pso.maxTotalThreadsPerThreadgroup), 256), height: 1, depth: 1)
        enc.dispatchThreads(threads, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
}

/// `output = gelu_tanh(gate) * up` — Gemma 4 uses `gelu_pytorch_tanh`, not SiLU.
private final class AssistantGELUMul {
    private let pso: MTLComputePipelineState
    init(context: MetalContext) throws {
        self.pso = try context.pipeline("assistant_gelu_mul")
    }
    func encode(commandBuffer: MTLCommandBuffer,
                gate: MTLBuffer, up: MTLBuffer, output: MTLBuffer, count: UInt32) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(gate, offset: 0, index: 0)
        enc.setBuffer(up, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var c = count
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)
        let threads = MTLSize(width: Int(count), height: 1, depth: 1)
        let tg = MTLSize(width: min(Int(pso.maxTotalThreadsPerThreadgroup), 256), height: 1, depth: 1)
        enc.dispatchThreads(threads, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
}

private final class AssistantBF16GEMV {
    private let pso: MTLComputePipelineState
    init(context: MetalContext) throws {
        self.pso = try context.pipeline("assistant_bf16_gemv")
    }
    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, input: MTLBuffer, output: MTLBuffer,
                rows: UInt32, cols: UInt32) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: 0, index: 0)
        enc.setBuffer(input, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)
        var r = rows; var c = cols
        enc.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 4)
        let tg = MTLSize(width: min(Int(pso.maxTotalThreadsPerThreadgroup), 256), height: 1, depth: 1)
        enc.dispatchThreadgroups(MTLSize(width: Int(rows), height: 1, depth: 1), threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
}

private final class AssistantArgmax {
    private let pso: MTLComputePipelineState
    init(context: MetalContext) throws {
        self.pso = try context.pipeline("assistant_argmax_half")
    }
    func encode(commandBuffer: MTLCommandBuffer,
                values: MTLBuffer, outputToken: MTLBuffer, count: UInt32) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(values, offset: 0, index: 0)
        enc.setBuffer(outputToken, offset: 0, index: 1)
        var c = count
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 2)
        let tg = MTLSize(width: min(Int(pso.maxTotalThreadsPerThreadgroup), 256), height: 1, depth: 1)
        enc.dispatchThreads(tg, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }
}

// MARK: - GPU Metal State

private final class GemmaAssistantMetalState {
    private struct GPULayer {
        let attentionKind: AssistantBridgeAttentionKind
        let inputLayernorm: MTLBuffer
        let postAttentionLayernorm: MTLBuffer
        let qProj: MTLBuffer
        let qNorm: MTLBuffer
        let oProj: MTLBuffer
        let gateProj: MTLBuffer
        let upProj: MTLBuffer
        let downProj: MTLBuffer
        let preFeedforwardLayernorm: MTLBuffer
        let postFeedforwardLayernorm: MTLBuffer?
        let layerScalar: Float
        let headDim: UInt32
        let numHeads: Int
        let numKVHeads: Int
        let qRows: UInt32
    }

    private let config: GemmaAssistantConfig
    private let context: MetalContext
    private let rms: RMSNorm
    private let rope: RoPE
    private let attention: Attention
    private let gemv: AssistantBF16GEMV
    private let embedLookup: AssistantEmbedLookup
    private let concat: AssistantConcat
    private let addThenScale: AssistantAddThenScale
    private let geluMul: AssistantGELUMul
    private let argmax: AssistantArgmax

    // Model weight buffers (BF16)
    private let embedWeight: MTLBuffer
    private let preProjection: MTLBuffer
    private let postProjection: MTLBuffer
    private let finalNorm: MTLBuffer
    private let layers: [GPULayer]

    // Working buffers (F16)
    private let backboneA: MTLBuffer
    private let backboneB: MTLBuffer
    private let tokenEmbed: MTLBuffer
    private let proxyEmbed: MTLBuffer
    private let fusedInput: MTLBuffer
    private let xCurrent: MTLBuffer
    private let xTemp: MTLBuffer
    private let normed: MTLBuffer
    private let qBuffer: MTLBuffer
    private let qNormed: MTLBuffer
    private let attnContext: MTLBuffer
    private let attnOut: MTLBuffer
    private let ffInput: MTLBuffer
    private let gate: MTLBuffer
    private let up: MTLBuffer
    private let activated: MTLBuffer
    private let ff: MTLBuffer
    private let ffNormed: MTLBuffer
    private let finalHidden: MTLBuffer
    private let logits: MTLBuffer
    private let tokenBuffer: MTLBuffer

    init(modelDirectory: URL, context: MetalContext) throws {
        let cfg = try GemmaAssistantConfig.load(from: modelDirectory)
        let archive = try GemmaAssistantTensorArchive(
            url: modelDirectory.appendingPathComponent("model.safetensors"))
        self.config = cfg
        self.context = context
        self.rms = try RMSNorm(context: context)
        self.rope = try RoPE(context: context)
        self.attention = try Attention(context: context)
        self.gemv = try AssistantBF16GEMV(context: context)
        self.embedLookup = try AssistantEmbedLookup(context: context)
        self.concat = try AssistantConcat(context: context)
        self.addThenScale = try AssistantAddThenScale(context: context)
        self.geluMul = try AssistantGELUMul(context: context)
        self.argmax = try AssistantArgmax(context: context)

        let hidden = cfg.hiddenSize
        let backbone = cfg.backboneHiddenSize
        let intermediate = cfg.intermediateSize
        let vocab = cfg.vocabSize
        let maxQRows = (0..<cfg.textConfig.numHiddenLayers).map { cfg.layerInfo(index: $0).qRows }.max() ?? 0

        // Load weight buffers
        self.embedWeight = try archive.buffer(
            device: context.device,
            name: "model.embed_tokens.weight",
            expectedShape: [vocab, hidden])
        self.preProjection = try archive.buffer(
            device: context.device,
            name: "pre_projection.weight",
            expectedShape: [hidden, backbone * 2])
        self.postProjection = try archive.buffer(
            device: context.device,
            name: "post_projection.weight",
            expectedShape: [backbone, hidden])
        self.finalNorm = try archive.buffer(
            device: context.device,
            name: "model.norm.weight",
            expectedShape: [hidden])

        // Load layer weights
        var gpuLayers: [GPULayer] = []
        gpuLayers.reserveCapacity(cfg.textConfig.numHiddenLayers)
        for index in 0..<cfg.textConfig.numHiddenLayers {
            let info = cfg.layerInfo(index: index)
            let prefix = "model.layers.\(index)"
            let postFFNName = "\(prefix).post_feedforward_layernorm.weight"
            let scalar = try archive.scalar(name: "\(prefix).layer_scalar")
            gpuLayers.append(GPULayer(
                attentionKind: info.attentionKind,
                inputLayernorm: try archive.buffer(
                    device: context.device, name: "\(prefix).input_layernorm.weight",
                    expectedShape: [hidden]),
                postAttentionLayernorm: try archive.buffer(
                    device: context.device, name: "\(prefix).post_attention_layernorm.weight",
                    expectedShape: [hidden]),
                qProj: try archive.buffer(
                    device: context.device, name: "\(prefix).self_attn.q_proj.weight",
                    expectedShape: [info.qRows, hidden]),
                qNorm: try archive.buffer(
                    device: context.device, name: "\(prefix).self_attn.q_norm.weight",
                    expectedShape: [info.headDim]),
                oProj: try archive.buffer(
                    device: context.device, name: "\(prefix).self_attn.o_proj.weight",
                    expectedShape: [hidden, info.qRows]),
                gateProj: try archive.buffer(
                    device: context.device, name: "\(prefix).mlp.gate_proj.weight",
                    expectedShape: [intermediate, hidden]),
                upProj: try archive.buffer(
                    device: context.device, name: "\(prefix).mlp.up_proj.weight",
                    expectedShape: [intermediate, hidden]),
                downProj: try archive.buffer(
                    device: context.device, name: "\(prefix).mlp.down_proj.weight",
                    expectedShape: [hidden, intermediate]),
                preFeedforwardLayernorm: try archive.buffer(
                    device: context.device, name: "\(prefix).pre_feedforward_layernorm.weight",
                    expectedShape: [hidden]),
                postFeedforwardLayernorm: try archive.optionalBuffer(
                    device: context.device, name: postFFNName,
                    expectedShape: [hidden]),
                layerScalar: scalar,
                headDim: UInt32(info.headDim),
                numHeads: info.numHeads,
                numKVHeads: info.numKVHeads,
                qRows: UInt32(info.qRows)))
        }
        self.layers = gpuLayers

        // Allocate working buffers
        self.backboneA = try Self.makeHalfBuffer(context: context, count: backbone, label: "assistant.backboneA")
        self.backboneB = try Self.makeHalfBuffer(context: context, count: backbone, label: "assistant.backboneB")
        self.tokenEmbed = try Self.makeHalfBuffer(context: context, count: hidden, label: "assistant.tokenEmbed")
        self.proxyEmbed = try Self.makeHalfBuffer(context: context, count: backbone, label: "assistant.proxyEmbed")
        self.fusedInput = try Self.makeHalfBuffer(context: context, count: backbone * 2, label: "assistant.fusedInput")
        self.xCurrent = try Self.makeHalfBuffer(context: context, count: hidden, label: "assistant.xCurrent")
        self.xTemp = try Self.makeHalfBuffer(context: context, count: hidden, label: "assistant.xTemp")
        self.normed = try Self.makeHalfBuffer(context: context, count: hidden, label: "assistant.normed")
        self.qBuffer = try Self.makeHalfBuffer(context: context, count: maxQRows, label: "assistant.q")
        self.qNormed = try Self.makeHalfBuffer(context: context, count: maxQRows, label: "assistant.qNormed")
        self.attnContext = try Self.makeHalfBuffer(context: context, count: maxQRows, label: "assistant.attnContext")
        self.attnOut = try Self.makeHalfBuffer(context: context, count: hidden, label: "assistant.attnOut")
        self.ffInput = try Self.makeHalfBuffer(context: context, count: hidden, label: "assistant.ffInput")
        self.gate = try Self.makeHalfBuffer(context: context, count: intermediate, label: "assistant.gate")
        self.up = try Self.makeHalfBuffer(context: context, count: intermediate, label: "assistant.up")
        self.activated = try Self.makeHalfBuffer(context: context, count: intermediate, label: "assistant.activated")
        self.ff = try Self.makeHalfBuffer(context: context, count: hidden, label: "assistant.ff")
        self.ffNormed = try Self.makeHalfBuffer(context: context, count: hidden, label: "assistant.ffNormed")
        self.finalHidden = try Self.makeHalfBuffer(context: context, count: hidden, label: "assistant.finalHidden")
        self.logits = try Self.makeHalfBuffer(context: context, count: vocab, label: "assistant.logits")
        self.tokenBuffer = try Self.makeSharedBuffer(
            context: context, byteCount: MemoryLayout<UInt32>.stride, label: "assistant.token")
    }

    func draftTokens(hiddenState: [Float],
                     currentToken: Int32,
                     initialTokenEmbedding: [Float]?,
                     bridgeSnapshot: AssistantBridgeSnapshot,
                     maxDraftTokens: Int,
                     targetTokenEmbeddingProvider: (any TargetTokenEmbeddingProvider)?) throws -> [Int32] {
        guard hiddenState.count == config.backboneHiddenSize else {
            throw GemmaAssistantError.hiddenStateMismatch(
                expected: config.backboneHiddenSize, actual: hiddenState.count)
        }
        guard maxDraftTokens > 0 else { return [] }

        writeHalf(hiddenState, to: backboneA)
        tokenBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)[0] = UInt32(bitPattern: currentToken)

        var draft: [Int32] = []
        draft.reserveCapacity(maxDraftTokens)
        var currentBackbone = backboneA
        var nextBackbone = backboneB
        var currentTokenEmbedding = initialTokenEmbedding
        // The HF reference computes `position_ids = input_ids.shape[1] - 1`
        // ONCE, *outside* the drafter's autoregressive loop, and reuses it for
        // every chained draft — the assistant is a single-position model reading
        // the target's KV, not a model that walks forward through positions.
        // `input_ids` there already carries the freshly sampled (not yet
        // KV-committed) token, so `shape[1] - 1` is the index that token will
        // occupy, which equals the number of tokens the target has committed.
        let draftPosition = UInt32(
            bridgeSnapshot.fullAttentionKV?.key.validTokenCount
                ?? bridgeSnapshot.slidingAttentionKV?.key.validTokenCount
                ?? 0)

        for _ in 0..<maxDraftTokens {
            if let currentTokenEmbedding {
                guard currentTokenEmbedding.count == config.backboneHiddenSize else {
                    throw GemmaAssistantError.hiddenStateMismatch(
                        expected: config.backboneHiddenSize, actual: currentTokenEmbedding.count)
                }
                writeHalf(currentTokenEmbedding, to: proxyEmbed)
            }
            guard let cb = context.queue.makeCommandBuffer() else {
                throw MetalError.noQueue
            }
            encodeStep(commandBuffer: cb,
                       currentBackbone: currentBackbone,
                       nextBackbone: nextBackbone,
                       bridgeSnapshot: bridgeSnapshot,
                       position: draftPosition,
                       useProvidedTokenEmbedding: currentTokenEmbedding != nil)
            cb.commit()
            cb.waitUntilCompleted()
            let nextToken = Int32(bitPattern: tokenBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)[0])
            draft.append(nextToken)
            swap(&currentBackbone, &nextBackbone)
            currentTokenEmbedding = targetTokenEmbeddingProvider?.copyTokenEmbedding(nextToken)
        }
        return draft
    }

    private func encodeStep(commandBuffer: MTLCommandBuffer,
                            currentBackbone: MTLBuffer,
                            nextBackbone: MTLBuffer,
                            bridgeSnapshot: AssistantBridgeSnapshot,
                            position: UInt32,
                            useProvidedTokenEmbedding: Bool) {
        let hiddenSize = UInt32(config.hiddenSize)
        let backboneHidden = UInt32(config.backboneHiddenSize)
        let intermediate = UInt32(config.intermediateSize)
        let vocabSize = UInt32(config.vocabSize)

        // Token embedding -> backbone projection
        if !useProvidedTokenEmbedding {
            embedLookup.encode(commandBuffer: commandBuffer,
                               weights: embedWeight, token: tokenBuffer, output: tokenEmbed,
                               rowWidth: hiddenSize)
            gemv.encode(commandBuffer: commandBuffer,
                        weights: postProjection, input: tokenEmbed, output: proxyEmbed,
                        rows: backboneHidden, cols: hiddenSize)
        }

        // Concat [proxyEmbed, currentBackbone] -> fusedInput
        concat.encode(commandBuffer: commandBuffer,
                      lhs: proxyEmbed, rhs: currentBackbone, output: fusedInput,
                      count: backboneHidden)

        // Pre-projection: fusedInput -> xCurrent
        gemv.encode(commandBuffer: commandBuffer,
                    weights: preProjection, input: fusedInput, output: xCurrent,
                    rows: hiddenSize, cols: backboneHidden * 2)

        // Each layer reads `currentHidden`, stages the post-attention residual in
        // `nextHidden`, and writes its final output back into `currentHidden`, so
        // no ping-pong swap is needed between layers.
        let currentHidden = xCurrent
        let nextHidden = xTemp

        for layer in layers {
            // The assistant owns no k_proj/v_proj. Sharing is *by layer type*:
            // every sliding layer reads the target's last sliding-layer KV, and
            // the full layer reads the target's last full-attention KV — exactly
            // the two entries the reference's `shared_kv_states` dict carries.
            let sharedKV = bridgeSnapshot.kv(kind: layer.attentionKind)
            // Input layernorm
            rms.encodeBF16W(commandBuffer: commandBuffer,
                            x: currentHidden, weight: layer.inputLayernorm,
                            out: normed, d: hiddenSize, eps: config.eps)
            // Q projection
            gemv.encode(commandBuffer: commandBuffer,
                        weights: layer.qProj, input: normed, output: qBuffer,
                        rows: layer.qRows, cols: hiddenSize)
            // Q per-head norm
            rms.encodeBF16WPerHead(commandBuffer: commandBuffer,
                                   x: qBuffer, weight: layer.qNorm, out: qNormed,
                                   headDim: layer.headDim, numHeads: layer.numHeads,
                                   eps: config.eps)

            // Attention
            switch layer.attentionKind {
            case .fullAttention:
                rope.encodeProportionalNeox(commandBuffer: commandBuffer,
                                            data: qNormed, position: position,
                                            headDim: layer.headDim, numHeads: UInt32(layer.numHeads),
                                            rotatedPairs: layer.headDim / 8)
                if let snapshot = sharedKV {
                    attention.encodeFull(commandBuffer: commandBuffer,
                                         q: qNormed,
                                         k: snapshot.key.buffer, kOffset: snapshot.key.offset,
                                         v: snapshot.value.buffer, vOffset: snapshot.value.offset,
                                         out: attnContext,
                                         headDim: layer.headDim,
                                         numQHeads: UInt32(layer.numHeads),
                                         numKVHeads: UInt32(layer.numKVHeads),
                                         seqLen: UInt32(snapshot.key.validTokenCount),
                                         scale: 1.0)
                    gemv.encode(commandBuffer: commandBuffer,
                                weights: layer.oProj, input: attnContext, output: attnOut,
                                rows: hiddenSize, cols: layer.qRows)
                } else {
                    gemv.encode(commandBuffer: commandBuffer,
                                weights: layer.oProj, input: qNormed, output: attnOut,
                                rows: hiddenSize, cols: layer.qRows)
                }
            case .slidingAttention:
                rope.encodeDefaultNeox(commandBuffer: commandBuffer,
                                       data: qNormed, position: position,
                                       headDim: layer.headDim, numHeads: UInt32(layer.numHeads),
                                       theta: 10_000.0)
                if let snapshot = sharedKV {
                    // Mirror the main decode path: only pass a ring capacity
                    // once the window has actually wrapped, otherwise 0 (linear
                    // indexing). Deriving it from `buffer.length / stride` is
                    // wrong — Metal pads allocations to a page boundary.
                    let capacity = snapshot.key.ringCapacity
                    let ringCap = capacity > 0 && snapshot.key.validTokenCount > capacity
                        ? UInt32(capacity)
                        : 0
                    attention.encodeSWA(commandBuffer: commandBuffer,
                                        q: qNormed,
                                        k: snapshot.key.buffer, kOffset: snapshot.key.offset,
                                        v: snapshot.value.buffer, vOffset: snapshot.value.offset,
                                        out: attnContext,
                                        headDim: layer.headDim,
                                        numQHeads: UInt32(layer.numHeads),
                                        numKVHeads: UInt32(layer.numKVHeads),
                                        seqLen: UInt32(snapshot.key.validTokenCount),
                                        window: UInt32(config.textConfig.slidingWindow),
                                        scale: 1.0,
                                        ringCapacity: ringCap)
                    gemv.encode(commandBuffer: commandBuffer,
                                weights: layer.oProj, input: attnContext, output: attnOut,
                                rows: hiddenSize, cols: layer.qRows)
                } else {
                    gemv.encode(commandBuffer: commandBuffer,
                                weights: layer.oProj, input: qNormed, output: attnOut,
                                rows: hiddenSize, cols: layer.qRows)
                }
            }

            // `post_attention_layernorm` normalises the *attention output*, and
            // the residual is then added UNSCALED. `layer_scalar` is not applied
            // here — see the end of the layer.
            rms.encodeBF16W(commandBuffer: commandBuffer,
                            x: attnOut, weight: layer.postAttentionLayernorm,
                            out: ffInput, d: hiddenSize, eps: config.eps)
            addThenScale.encode(commandBuffer: commandBuffer,
                                base: currentHidden, delta: ffInput, output: nextHidden,
                                count: hiddenSize, scale: 1.0)

            // FFN
            rms.encodeBF16W(commandBuffer: commandBuffer,
                            x: nextHidden, weight: layer.preFeedforwardLayernorm,
                            out: ffInput, d: hiddenSize, eps: config.eps)
            gemv.encode(commandBuffer: commandBuffer,
                        weights: layer.gateProj, input: ffInput, output: gate,
                        rows: intermediate, cols: hiddenSize)
            gemv.encode(commandBuffer: commandBuffer,
                        weights: layer.upProj, input: ffInput, output: up,
                        rows: intermediate, cols: hiddenSize)
            geluMul.encode(commandBuffer: commandBuffer,
                           gate: gate, up: up, output: activated, count: intermediate)
            gemv.encode(commandBuffer: commandBuffer,
                        weights: layer.downProj, input: activated, output: ff,
                        rows: hiddenSize, cols: intermediate)

            // `post_feedforward_layernorm` normalises the *MLP output*, BEFORE
            // the residual add — not the post-residual hidden state.
            var ffDelta = ff
            if let post = layer.postFeedforwardLayernorm {
                rms.encodeBF16W(commandBuffer: commandBuffer,
                                x: ff, weight: post, out: ffNormed,
                                d: hiddenSize, eps: config.eps)
                ffDelta = ffNormed
            }

            // Second residual add, then the single `layer_scalar` multiply that
            // scales the whole layer output (residual included) exactly once:
            //     hidden_states = residual + hidden_states
            //     hidden_states *= self.layer_scalar
            addThenScale.encode(commandBuffer: commandBuffer,
                                base: nextHidden, delta: ffDelta, output: currentHidden,
                                count: hiddenSize, scale: layer.layerScalar)
        }

        // Final norm + lm_head (tied embedding)
        rms.encodeBF16W(commandBuffer: commandBuffer,
                        x: currentHidden, weight: finalNorm, out: finalHidden,
                        d: hiddenSize, eps: config.eps)
        gemv.encode(commandBuffer: commandBuffer,
                    weights: embedWeight, input: finalHidden, output: logits,
                    rows: vocabSize, cols: hiddenSize)
        argmax.encode(commandBuffer: commandBuffer,
                      values: logits, outputToken: tokenBuffer, count: vocabSize)

        // Post-projection: finalHidden -> nextBackbone
        gemv.encode(commandBuffer: commandBuffer,
                    weights: postProjection, input: finalHidden, output: nextBackbone,
                    rows: backboneHidden, cols: hiddenSize)
    }

    private func writeHalf(_ values: [Float], to buffer: MTLBuffer) {
        let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: values.count)
        for (index, value) in values.enumerated() {
            ptr[index] = Float16(value)
        }
    }

    private static func makeHalfBuffer(context: MetalContext,
                                       count: Int, label: String) throws -> MTLBuffer {
        try makeSharedBuffer(context: context,
                             byteCount: max(count, 1) * MemoryLayout<Float16>.stride,
                             label: label)
    }

    private static func makeSharedBuffer(context: MetalContext,
                                         byteCount: Int, label: String) throws -> MTLBuffer {
        guard let buffer = context.device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw GemmaAssistantError.allocationFailed(label)
        }
        buffer.label = label
        return buffer
    }
}

// MARK: - Public GPU Drafter

public final class GemmaAssistantMetalDrafter: @unchecked Sendable, LocalMTPDrafter {
    private let state: GemmaAssistantMetalState
    private let lock = NSLock()

    public init(modelDirectory: URL, context: MetalContext) throws {
        self.state = try GemmaAssistantMetalState(modelDirectory: modelDirectory, context: context)
    }

    public func draftTokens(bridgeSnapshot: AssistantBridgeSnapshot,
                            history _: [Int32],
                            currentToken: Int32,
                            maxDraftTokens: Int,
                            targetTokenEmbeddingProvider: (any TargetTokenEmbeddingProvider)?) throws -> [Int32] {
        lock.lock()
        defer { lock.unlock() }
        return try state.draftTokens(
            hiddenState: bridgeSnapshot.lastHiddenState,
            currentToken: currentToken,
            initialTokenEmbedding: bridgeSnapshot.lastTokenEmbedding,
            bridgeSnapshot: bridgeSnapshot,
            maxDraftTokens: maxDraftTokens,
            targetTokenEmbeddingProvider: targetTokenEmbeddingProvider)
    }
}
