import Metal

/// MLX-affine INT4 matrix-vector multiplication.
/// Eight SIMD groups process eight output rows per threadgroup.
final class DequantInt4GEMV {
    private struct Shape: Hashable {
        var m: UInt32
        var n: UInt32
    }

    private static let rowsPerThreadgroup = 8
    private static let realDecodeShapes: [Shape] = [
        Shape(m: 4096, n: 2816),
        Shape(m: 2048, n: 2816),
        Shape(m: 2816, n: 4096),
        Shape(m: 8192, n: 2816),
        Shape(m: 1024, n: 2816),
        Shape(m: 2816, n: 8192),
    ]

    /// The 3-bit tgx GEMV kernels are selected per-tensor via the `bits:`
    /// parameter (driven by manifest quant.attention.weightBits), never by a
    /// global env — a global switch would misread 4-bit attention as 3-bit.
    private let tgxB3Pipeline: MTLComputePipelineState?
    private let tgxB3Specialized: [Shape: MTLComputePipelineState]

    private let pipeline: MTLComputePipelineState
    private let specializedPipelines: [Shape: MTLComputePipelineState]
    private let tgxPipeline: MTLComputePipelineState
    private let tgxSpecialized: [Shape: MTLComputePipelineState]

    init(context: MetalContext) throws {
        self.pipeline = try context.pipeline(
            "dequant_int4_gemv_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)

        self.tgxPipeline = try context.pipeline(
            "dequant_int4_gemv_tgx",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)

        var specializedPipelines: [Shape: MTLComputePipelineState] = [:]
        var tgxSpecialized: [Shape: MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            let constants = [
                MetalFunctionConstant(index: 20, value: .uint32(shape.m)),
                MetalFunctionConstant(index: 21, value: .uint32(shape.n)),
                MetalFunctionConstant(index: 22, value: .bool(true)),
            ]
            specializedPipelines[shape] = try context.pipeline(
                "dequant_int4_gemv_simd",
                constants: constants,
                maxTotalThreadsPerThreadgroup: 512)
            tgxSpecialized[shape] = try context.pipeline(
                "dequant_int4_gemv_tgx",
                constants: constants,
                maxTotalThreadsPerThreadgroup: 512)
        }
        self.specializedPipelines = specializedPipelines
        self.tgxSpecialized = tgxSpecialized
        // Always build the 3-bit tgx pipelines: selection is now per-tensor
        // (bits: param driven by manifest quant.attention.weightBits), so the
        // env is no longer required to enable the b3 path.
        self.tgxB3Pipeline = try? context.pipeline("dequant_int4_gemv_tgx_b3",
                                                   constants: [],
                                                   maxTotalThreadsPerThreadgroup: 512)
        var b3s: [Shape: MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            let constants = [
                MetalFunctionConstant(index: 20, value: .uint32(shape.m)),
                MetalFunctionConstant(index: 21, value: .uint32(shape.n)),
                MetalFunctionConstant(index: 22, value: .bool(true)),
            ]
            if let p = try? context.pipeline("dequant_int4_gemv_tgx_b3",
                                             constants: constants,
                                             maxTotalThreadsPerThreadgroup: 512) {
                b3s[shape] = p
            }
        }
        self.tgxB3Specialized = b3s
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer,
                weightsOffset: Int = 0,
                scales: MTLBuffer,
                scalesOffset: Int = 0,
                biases: MTLBuffer,
                biasesOffset: Int = 0,
                x: MTLBuffer,
                xOffset: Int = 0,
                y: MTLBuffer,
                yOffset: Int = 0,
                m: UInt32,
                n: UInt32) {
        precondition(n % UInt32(Quantization.groupSize) == 0,
                     "N must be a multiple of \(Quantization.groupSize)")
        // The kernel reads packed weights through a `ushort*`; the repacker
        // guarantees two-byte sub-tensor alignment but not four-byte alignment.
        precondition(weightsOffset % 2 == 0,
                     "dequant_int4_gemv_simd needs a 2-aligned weightsOffset, got \(weightsOffset)")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(
            specializedPipelines[Shape(m: m, n: n)] ?? pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = m
        var nValue = n
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)

        let threadgroupSize = MTLSize(
            width: 32 * Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        let threadgroupCount = MTLSize(
            width: (Int(m) + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        encoder.dispatchThreadgroups(threadgroupCount,
                                     threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    /// Threadgroup-cached-x GEMV: x is loaded into threadgroup memory once per
    /// threadgroup (8 rows share it), so the hot loop reads x from shared
    /// memory instead of device memory.
    func encodeTgx(commandBuffer: MTLCommandBuffer,
                   weights: MTLBuffer,
                   weightsOffset: Int = 0,
                   scales: MTLBuffer,
                   scalesOffset: Int = 0,
                   biases: MTLBuffer,
                   biasesOffset: Int = 0,
                   x: MTLBuffer,
                   xOffset: Int = 0,
                   y: MTLBuffer,
                   yOffset: Int = 0,
                   m: UInt32,
                   n: UInt32,
                   bits: Int = 4) {
        precondition(n % UInt32(Quantization.groupSize) == 0,
                     "N must be a multiple of \(Quantization.groupSize)")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        precondition(bits == 3 || bits == 4,
                     "unsupported attention weight bits \(bits)")
        if bits == 3, let b3 = tgxB3Specialized[Shape(m: m, n: n)] {
            encoder.setComputePipelineState(b3)
        } else if bits == 3, let b3 = tgxB3Pipeline {
            encoder.setComputePipelineState(b3)
        } else {
            encoder.setComputePipelineState(
                tgxSpecialized[Shape(m: m, n: n)] ?? tgxPipeline)
        }
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = m
        var nValue = n
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)

        let threadgroupSize = MTLSize(
            width: 32 * Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        let threadgroupCount = MTLSize(
            width: (Int(m) + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
            height: 1,
            depth: 1)
        encoder.dispatchThreadgroups(threadgroupCount,
                                     threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
}
