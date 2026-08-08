import Foundation
import Metal

@frozen
public struct MoEExpertOffsets {
    public var gateWOff: UInt32
    public var gateSOff: UInt32
    public var gateBOff: UInt32
    public var upWOff: UInt32
    public var upSOff: UInt32
    public var upBOff: UInt32
    public var downWOff: UInt32
    public var downSOff: UInt32
    public var downBOff: UInt32

    public init(gateWOff: UInt32, gateSOff: UInt32, gateBOff: UInt32,
                upWOff: UInt32, upSOff: UInt32, upBOff: UInt32,
                downWOff: UInt32, downSOff: UInt32, downBOff: UInt32) {
        self.gateWOff = gateWOff
        self.gateSOff = gateSOff
        self.gateBOff = gateBOff
        self.upWOff = upWOff
        self.upSOff = upSOff
        self.upBOff = upBOff
        self.downWOff = downWOff
        self.downSOff = downSOff
        self.downBOff = downBOff
    }
}

final class MoE {
    static let maxStreamedExperts = 8

    private static let realDecodeD: UInt32 = 2816
    private static let realDecodeF: UInt32 = 704
    private static let realDecodeTopK: UInt32 = 8
    private static let realDecodeNumExperts: UInt32 = 128
    /// Matches kMoEPhase2MaxChunk in moe.metal: the chunked phase-2 kernel
    /// clamps `chunk` internally, so the Swift-side grid MUST use the same
    /// clamped value or a too-large env chunk would dispatch too few TGs and
    /// silently leave the tail of y unwritten.
    private static let maxPhase2Chunk: UInt32 = 64
    private static let realDecodeMoEConstants: [MetalFunctionConstant] = [
        MetalFunctionConstant(index: 0, value: .uint32(realDecodeD)),
        MetalFunctionConstant(index: 1, value: .uint32(realDecodeF)),
        MetalFunctionConstant(index: 2, value: .uint32(realDecodeTopK)),
        MetalFunctionConstant(index: 3, value: .bool(true)),
    ]
    private static let realDecodeRouterConstants: [MetalFunctionConstant] = [
        MetalFunctionConstant(index: 40, value: .uint32(realDecodeNumExperts)),
        MetalFunctionConstant(index: 41, value: .uint32(realDecodeD)),
        MetalFunctionConstant(index: 42, value: .uint32(realDecodeTopK)),
        MetalFunctionConstant(index: 43, value: .bool(true)),
    ]

    private let routerGemvPSO: MTLComputePipelineState
    private let routerGemvSpecializedPSO: MTLComputePipelineState
    private let routerSelectK8PSO: MTLComputePipelineState
    private let routerSelectK8SpecializedPSO: MTLComputePipelineState
    private let routerLogits: MTLBuffer
    private let phase1U16PSO: MTLComputePipelineState
    private let phase1U16SpecializedPSO: MTLComputePipelineState
    private let phase1SubsetU16PSO: MTLComputePipelineState
    private let phase1SubsetU16SpecializedPSO: MTLComputePipelineState
    private let phase2ReduceK8PSO: MTLComputePipelineState
    private let phase2ReduceK8SpecializedPSO: MTLComputePipelineState
    private let phase2ChunkedPSO: MTLComputePipelineState
    private let phase2ChunkedSpecializedPSO: MTLComputePipelineState
    private var bitsPhase1PSO: [Int: MTLComputePipelineState] = [:]
    private var bitsPhase1SpecializedPSO: [Int: MTLComputePipelineState] = [:]
    private var bitsPhase1SubsetPSO: [Int: MTLComputePipelineState] = [:]
    private var bitsPhase1SubsetSpecializedPSO: [Int: MTLComputePipelineState] = [:]
    private var bitsPhase2PSO: [Int: MTLComputePipelineState] = [:]
    private var bitsPhase2SpecializedPSO: [Int: MTLComputePipelineState] = [:]
    private let routedArgEncoder: MTLArgumentEncoder
    private let reusableRoutedArgBuffer: MTLBuffer

    init(context: MetalContext) throws {
        let routerName = "router_gemv_gemma4_r4"
        self.routerGemvPSO = try context.pipeline(
            routerName,
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)
        self.routerGemvSpecializedPSO = try context.pipeline(
            routerName,
            constants: Self.realDecodeRouterConstants,
            maxTotalThreadsPerThreadgroup: 512)
        self.routerSelectK8PSO = try context.pipeline("router_topk_select_k8")
        self.routerSelectK8SpecializedPSO = try context.pipeline(
            "router_topk_select_k8",
            constants: Self.realDecodeRouterConstants)
        self.phase1U16PSO = try context.pipeline("moe_phase1_gate_up_act_u16load")
        self.phase1U16SpecializedPSO = try context.pipeline(
            "moe_phase1_gate_up_act_u16load",
            constants: Self.realDecodeMoEConstants)
        self.phase1SubsetU16PSO = try context.pipeline("moe_phase1_gate_up_act_subset_u16load")
        self.phase1SubsetU16SpecializedPSO = try context.pipeline(
            "moe_phase1_gate_up_act_subset_u16load",
            constants: Self.realDecodeMoEConstants)
        self.phase2ReduceK8PSO = try context.pipeline("moe_phase2_down_reduce_k8")
        self.phase2ReduceK8SpecializedPSO = try context.pipeline(
            "moe_phase2_down_reduce_k8",
            constants: Self.realDecodeMoEConstants)
        self.phase2ChunkedPSO = try context.pipeline("moe_phase2_down_reduce_k8_chunked")
        self.phase2ChunkedSpecializedPSO = try context.pipeline(
            "moe_phase2_down_reduce_k8_chunked",
            constants: Self.realDecodeMoEConstants)
        // 2/3-bit routed-expert variants (weightBits = 2 | 3).
        for bits in [2, 3] {
            self.bitsPhase1PSO[bits] = try context.pipeline(
                "moe_phase1_gate_up_act_b\(bits)",
                constants: [],
                maxTotalThreadsPerThreadgroup: 512)
            self.bitsPhase1SpecializedPSO[bits] = try context.pipeline(
                "moe_phase1_gate_up_act_b\(bits)",
                constants: Self.realDecodeMoEConstants,
                maxTotalThreadsPerThreadgroup: 512)
            self.bitsPhase1SubsetPSO[bits] = try context.pipeline(
                "moe_phase1_gate_up_act_subset_b\(bits)")
            self.bitsPhase1SubsetSpecializedPSO[bits] = try context.pipeline(
                "moe_phase1_gate_up_act_subset_b\(bits)",
                constants: Self.realDecodeMoEConstants)
            self.bitsPhase2PSO[bits] = try context.pipeline(
                "moe_phase2_down_reduce_b\(bits)")
            self.bitsPhase2SpecializedPSO[bits] = try context.pipeline(
                "moe_phase2_down_reduce_b\(bits)",
                constants: Self.realDecodeMoEConstants)
        }

        guard let logits = context.device.makeBuffer(
            length: 256 * MemoryLayout<Float>.stride,
            options: .storageModeShared),
              let phase1Function = context.library.makeFunction(
                name: "moe_phase1_gate_up_act_u16load") else {
            throw MetalError.noDevice
        }
        self.routerLogits = logits
        self.routedArgEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 0)
        guard let reusable = context.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.reusableRoutedArgBuffer = reusable
    }

    func encodeRouterGemma4(commandBuffer: MTLCommandBuffer,
                                   weights: MTLBuffer, weightsOffset: Int = 0,
                                   scales: MTLBuffer, scalesOffset: Int = 0,
                                   biases: MTLBuffer, biasesOffset: Int = 0,
                                   hidden: MTLBuffer,
                                   effectiveScale: MTLBuffer, effectiveScaleOffset: Int = 0,
                                   perExpertScale: MTLBuffer, perExpertScaleOffset: Int = 0,
                                   outIndices: MTLBuffer,
                                   outWeights: MTLBuffer,
                                   numExperts: UInt32,
                                   d: UInt32,
                                   topK: UInt32) {
        precondition(d.isMultiple(of: UInt32(Quantization.groupSize)))
        precondition(numExperts <= 256)
        precondition(topK == UInt32(Self.maxStreamedExperts))

        var expertCount = numExperts
        var dimension = d
        let useSpecialized = numExperts == Self.realDecodeNumExperts
            && d == Self.realDecodeD
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(
                useSpecialized ? routerGemvSpecializedPSO : routerGemvPSO)
            encoder.setBuffer(weights, offset: weightsOffset, index: 0)
            encoder.setBuffer(scales, offset: scalesOffset, index: 1)
            encoder.setBuffer(biases, offset: biasesOffset, index: 2)
            encoder.setBuffer(hidden, offset: 0, index: 3)
            encoder.setBuffer(effectiveScale, offset: effectiveScaleOffset, index: 4)
            encoder.setBuffer(routerLogits, offset: 0, index: 5)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(width: (Int(numExperts) + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(
                useSpecialized ? routerSelectK8SpecializedPSO : routerSelectK8PSO)
            encoder.setBuffer(routerLogits, offset: 0, index: 0)
            encoder.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 1)
            encoder.setBuffer(outIndices, offset: 0, index: 2)
            encoder.setBuffer(outWeights, offset: 0, index: 3)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    typealias RoutedBlob = (buffer: MTLBuffer, offset: UInt64)

    /// The GPU residency pass for file-backed bytesNoCopy buffers costs ~85us
    /// per buffer. In mmap mode every blob is the same stream buffer, so
    /// collapse the array to unique objects before useResources — one
    /// residency registration per command buffer instead of per expert.
    private static func uniqueBlobBuffers(_ blobs: [RoutedBlob]) -> [MTLBuffer] {
        var seen = Set<ObjectIdentifier>()
        return blobs.compactMap { blob in
            let id = ObjectIdentifier(blob.buffer)
            guard !seen.contains(id) else { return nil }
            seen.insert(id)
            return blob.buffer
        }
    }

    func makeRoutedArgumentBuffer(routedBlobs: [RoutedBlob],
                                         topK: UInt32) -> MTLBuffer? {
        validate(routedBlobs: routedBlobs, topK: topK)
        guard let buffer = routedBlobs.first?.buffer.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            return nil
        }
        encodeRoutedArgumentBuffer(buffer, routedBlobs: routedBlobs)
        return buffer
    }

    func makeReusedRoutedArgumentBuffer(routedBlobs: [RoutedBlob],
                                               topK: UInt32) -> MTLBuffer {
        validate(routedBlobs: routedBlobs, topK: topK)
        encodeRoutedArgumentBuffer(reusableRoutedArgBuffer, routedBlobs: routedBlobs)
        return reusableRoutedArgBuffer
    }

    func encodeRoutedPersistentPhase1U16Load(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [RoutedBlob],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        acts: MTLBuffer,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        bits: Int = 4
    ) {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        var expertCount = topK
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            phase1PSO(specialized: useRealDecodeConstants(d: d, f: f), bits: bits))
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        encoder.useResources(Self.uniqueBlobBuffers(routedBlobs), usage: .read)
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(topK * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase1SubsetU16Load(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [RoutedBlob],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        acts: MTLBuffer,
        activeSlots: MTLBuffer,
        activeSlotIndices: [UInt32],
        activeCount: UInt32,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        bits: Int = 4
    ) {
        guard activeCount > 0 else { return }
        validate(routedBlobs: routedBlobs, topK: topK)
        precondition(activeSlotIndices.count == Int(activeCount))
        var dimension = d
        var intermediate = f
        var expertCount = topK
        var active = activeCount
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            phase1SubsetPSO(specialized: useRealDecodeConstants(d: d, f: f), bits: bits))
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        encoder.useResources(
            Self.uniqueBlobBuffers(activeSlotIndices.map { routedBlobs[Int($0)] }),
            usage: .read)
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(activeSlots, offset: 0, index: 7)
        encoder.setBytes(&active, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(activeCount * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase2Reduce(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [RoutedBlob],
        routedOffsets: MoEExpertOffsets,
        acts: MTLBuffer,
        routingWeights: MTLBuffer,
        residual: MTLBuffer,
        y: MTLBuffer,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        bits: Int = 4,
        chunk: UInt32 = 1
    ) {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        // F > 1024 would read past the 16KB acts smem copy (kMoEPhase2MaxActs).
        let useChunked = chunk > 1 && bits == 4 && Int(f) <= 1024
        var chunkValue = useChunked ? min(chunk, Self.maxPhase2Chunk) : 1
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        if useChunked {
            encoder.setComputePipelineState(
                useRealDecodeConstants(d: d, f: f) ? phase2ChunkedSpecializedPSO : phase2ChunkedPSO)
        } else {
            encoder.setComputePipelineState(
                phase2PSO(specialized: useRealDecodeConstants(d: d, f: f), bits: bits))
        }
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        encoder.useResources(Self.uniqueBlobBuffers(routedBlobs), usage: .read)
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(acts, offset: 0, index: 2)
        encoder.setBuffer(routingWeights, offset: 0, index: 3)
        encoder.setBuffer(residual, offset: 0, index: 4)
        encoder.setBuffer(y, offset: 0, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 7)
        if useChunked {
            encoder.setBytes(&chunkValue, length: MemoryLayout<UInt32>.stride, index: 8)
            encoder.dispatchThreadgroups(
                MTLSize(width: Int((d + chunkValue - 1) / chunkValue), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        } else {
            encoder.dispatchThreadgroups(
                MTLSize(width: Int(d), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        encoder.endEncoding()
    }

    /// Pick the routed-expert decode PSO for a given weight bit width.
    /// `bits == 4` uses the original kernels; 2/3 use the A/B variants.
    private func phase1PSO(specialized: Bool, bits: Int) -> MTLComputePipelineState {
        switch bits {
        case 2, 3:
            return specialized ? bitsPhase1SpecializedPSO[bits]! : bitsPhase1PSO[bits]!
        default:
            return specialized ? phase1U16SpecializedPSO : phase1U16PSO
        }
    }

    private func phase1SubsetPSO(specialized: Bool, bits: Int) -> MTLComputePipelineState {
        switch bits {
        case 2, 3:
            return specialized ? bitsPhase1SubsetSpecializedPSO[bits]! : bitsPhase1SubsetPSO[bits]!
        default:
            return specialized ? phase1SubsetU16SpecializedPSO : phase1SubsetU16PSO
        }
    }

    private func phase2PSO(specialized: Bool, bits: Int) -> MTLComputePipelineState {
        switch bits {
        case 2, 3:
            return specialized ? bitsPhase2SpecializedPSO[bits]! : bitsPhase2PSO[bits]!
        default:
            return specialized ? phase2ReduceK8SpecializedPSO : phase2ReduceK8PSO
        }
    }

    private func validate(routedBlobs: [RoutedBlob], topK: UInt32) {
        precondition(topK == UInt32(Self.maxStreamedExperts))
        precondition(routedBlobs.count == Int(topK))
    }

    private func encodeRoutedArgumentBuffer(_ buffer: MTLBuffer,
                                            routedBlobs: [RoutedBlob]) {
        routedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for (index, blob) in routedBlobs.enumerated() {
            // Per-expert base address = stream buffer + expert offset (mmap)
            // or slot buffer + 0 (pread). The kernel's `blob[slot]` pointers
            // resolve to exactly the expert's window either way.
            routedArgEncoder.setBuffer(blob.buffer,
                                       offset: Int(blob.offset),
                                       index: index)
        }
    }

    private func useRealDecodeConstants(d: UInt32, f: UInt32) -> Bool {
        d == Self.realDecodeD && f == Self.realDecodeF
    }
}
