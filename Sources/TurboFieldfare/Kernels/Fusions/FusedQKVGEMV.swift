import Foundation
import Metal

final class FusedQKVGEMV {
    private struct Shape: Hashable {
        var qRows: UInt32
        var kvRows: UInt32
        var n: UInt32
    }

    private let pso: MTLComputePipelineState
    private let specializedPSOs: [Shape: MTLComputePipelineState]
    private let psoSmemX: MTLComputePipelineState?
    /// 3-bit fused QKV kernels are selected per-tensor via the `bits:`
    /// parameter (manifest quant.attention.weightBits), never a global env.
    private let psoB3: MTLComputePipelineState?
    private let specializedB3PSOs: [Shape: MTLComputePipelineState]
    private let specializedSmemXPSOs: [Shape: MTLComputePipelineState]

    /// Opt-in: threadgroup-x smem variant of the QKV GEMV
    /// (dequant_int4_qkv_gemv_simd_smemx). Same math, byte-identical output;
    /// A/B probe for the §13.29.3 x-re-read finding.
    private static let smemXEnabled: Bool =
        ProcessInfo.processInfo.environment["TURBO_FIELDFARE_QKV_SMEM_X"] == "1"

    private static let realDecodeShapes: [Shape] = [
        Shape(qRows: 4096, kvRows: 2048, n: 2816),
        Shape(qRows: 8192, kvRows: 1024, n: 2816),
    ]

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("dequant_int4_qkv_gemv_simd",
                                        constants: [],
                                        maxTotalThreadsPerThreadgroup: 512)
        var variants: [Shape: MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            variants[shape] = try context.pipeline(
                "dequant_int4_qkv_gemv_simd",
                constants: [
                    MetalFunctionConstant(index: 23, value: .uint32(shape.qRows)),
                    MetalFunctionConstant(index: 24, value: .uint32(shape.kvRows)),
                    MetalFunctionConstant(index: 25, value: .uint32(shape.n)),
                    MetalFunctionConstant(index: 26, value: .bool(true)),
                ],
                maxTotalThreadsPerThreadgroup: 512)
        }
        self.specializedPSOs = variants
        if Self.smemXEnabled {
            self.psoSmemX = try? context.pipeline("dequant_int4_qkv_gemv_simd_smemx",
                                                   constants: [],
                                                   maxTotalThreadsPerThreadgroup: 512)
            var smemVariants: [Shape: MTLComputePipelineState] = [:]
            for shape in Self.realDecodeShapes {
                if let p = try? context.pipeline(
                    "dequant_int4_qkv_gemv_simd_smemx",
                    constants: [
                        MetalFunctionConstant(index: 23, value: .uint32(shape.qRows)),
                        MetalFunctionConstant(index: 24, value: .uint32(shape.kvRows)),
                        MetalFunctionConstant(index: 25, value: .uint32(shape.n)),
                        MetalFunctionConstant(index: 26, value: .bool(true)),
                    ],
                    maxTotalThreadsPerThreadgroup: 512) {
                    smemVariants[shape] = p
                }
            }
            self.specializedSmemXPSOs = smemVariants
        } else {
            self.psoSmemX = nil
            self.specializedSmemXPSOs = [:]
        }
        // Always build the 3-bit QKV pipelines (per-tensor bits selection).
        self.psoB3 = try? context.pipeline("dequant_int4_qkv_gemv_simd_b3",
                                           constants: [],
                                           maxTotalThreadsPerThreadgroup: 512)
        var b3v: [Shape: MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            if let p = try? context.pipeline(
                "dequant_int4_qkv_gemv_simd_b3",
                constants: [
                    MetalFunctionConstant(index: 23, value: .uint32(shape.qRows)),
                    MetalFunctionConstant(index: 24, value: .uint32(shape.kvRows)),
                    MetalFunctionConstant(index: 25, value: .uint32(shape.n)),
                    MetalFunctionConstant(index: 26, value: .bool(true)),
                ],
                maxTotalThreadsPerThreadgroup: 512) {
                b3v[shape] = p
            }
        }
        self.specializedB3PSOs = b3v
    }

    func encode(commandBuffer: MTLCommandBuffer,
                       qWeights: MTLBuffer, qWeightsOffset: Int = 0,
                       qScales: MTLBuffer, qScalesOffset: Int = 0,
                       qBiases: MTLBuffer, qBiasesOffset: Int = 0,
                       kWeights: MTLBuffer, kWeightsOffset: Int = 0,
                       kScales: MTLBuffer, kScalesOffset: Int = 0,
                       kBiases: MTLBuffer, kBiasesOffset: Int = 0,
                       vWeights: MTLBuffer, vWeightsOffset: Int = 0,
                       vScales: MTLBuffer, vScalesOffset: Int = 0,
                       vBiases: MTLBuffer, vBiasesOffset: Int = 0,
                       x: MTLBuffer,
                       qOut: MTLBuffer, qOutOffset: Int = 0,
                       kOut: MTLBuffer, kOutOffset: Int = 0,
                       vOut: MTLBuffer, vOutOffset: Int = 0,
                       qRows: UInt32,
                       kvRows: UInt32,
                       n: UInt32,
                       bits: Int = 4) {
        precondition(n % UInt32(Quantization.groupSize) == 0,
                     "N must be a multiple of \(Quantization.groupSize)")
        precondition(qWeightsOffset % 2 == 0 &&
                     kWeightsOffset % 2 == 0 &&
                     vWeightsOffset % 2 == 0,
                     "FusedQKVGEMV needs 2-aligned weights offsets")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        let shape = Shape(qRows: qRows, kvRows: kvRows, n: n)
        precondition(bits == 3 || bits == 4,
                     "unsupported attention weight bits \(bits)")
        if bits == 3, let b3 = specializedB3PSOs[shape] {
            enc.setComputePipelineState(b3)
        } else if bits == 3, let b3 = psoB3 {
            enc.setComputePipelineState(b3)
        } else if Self.smemXEnabled, let smemPso = psoSmemX {
            enc.setComputePipelineState(specializedSmemXPSOs[shape] ?? smemPso)
        } else {
            enc.setComputePipelineState(specializedPSOs[shape] ?? pso)
        }
        enc.setBuffer(qWeights, offset: qWeightsOffset, index: 0)
        enc.setBuffer(qScales, offset: qScalesOffset, index: 1)
        enc.setBuffer(qBiases, offset: qBiasesOffset, index: 2)
        enc.setBuffer(kWeights, offset: kWeightsOffset, index: 3)
        enc.setBuffer(kScales, offset: kScalesOffset, index: 4)
        enc.setBuffer(kBiases, offset: kBiasesOffset, index: 5)
        enc.setBuffer(vWeights, offset: vWeightsOffset, index: 6)
        enc.setBuffer(vScales, offset: vScalesOffset, index: 7)
        enc.setBuffer(vBiases, offset: vBiasesOffset, index: 8)
        enc.setBuffer(x, offset: 0, index: 9)
        enc.setBuffer(qOut, offset: qOutOffset, index: 10)
        enc.setBuffer(kOut, offset: kOutOffset, index: 11)
        enc.setBuffer(vOut, offset: vOutOffset, index: 12)
        var qVar = qRows
        var kvVar = kvRows
        var nVar = n
        enc.setBytes(&qVar, length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&kvVar, length: MemoryLayout<UInt32>.size, index: 14)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 15)
        let totalRows = Int(qRows + 2 * kvRows)
        enc.dispatchThreadgroups(MTLSize(width: (totalRows + 7) / 8,
                                         height: 1,
                                         depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256,
                                                                 height: 1,
                                                                 depth: 1))
        enc.endEncoding()
    }
}
