// ============================================================================
// DeltaNetRunner.swift — Qwen3.6 DeltaNet 层的最小 Metal 调度器 (v2, 真实 API)
//
// 独立于 RealForwardRunner: 先证明 4 个 kernel 可调用、输出与 golden 一致,
// 再集成进主 runner。对照 golden_layer0.npz 验证。
//
// 权重绑定: 通过 TensorView (resident 原名, e.g.
//   "model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
//   → enc.setBuffer(view.buffer, offset: Int(view.offset), index:)
//
// 状态: h [32,128,128] FP32 (跨 token 持久), conv_state [8192,3] FP16
// ============================================================================
import Metal
import Foundation

public final class DeltaNetRunner {
    public struct Config {
        public var hiddenSize: Int = 2048
        public var numVHeads: Int = 32
        public var numKHeads: Int = 16
        public var headDim: Int = 128
        public var convDim: Int = 8192
        public var convKernel: Int = 4
        public init() {}
    }

    private let device: MTLDevice
    private let config: Config

    // PSOs (对应 deltanet.metal 5 个 kernel)
    private let psoProject: MTLComputePipelineState
    private let psoConv1d: MTLComputePipelineState
    private let psoPrepare: MTLComputePipelineState
    private let psoDeltaRule: MTLComputePipelineState
    private let psoOutput: MTLComputePipelineState

    // 常驻状态 (跨 token)
    private var hState: MTLBuffer          // [32, 128, 128] FP32
    private var convState: MTLBuffer       // [8192, 3] FP16
    private var convStateTmp: MTLBuffer    // 双 buffer 防 in-place 竞争

    // 中间 buffer
    private var qkvBuf: MTLBuffer          // [8192] FP16
    private var zBuf: MTLBuffer            // [4096] FP16
    private var bBuf: MTLBuffer            // [32] FP16
    private var aBuf: MTLBuffer            // [32] FP16
    private var qkvPostBuf: MTLBuffer      // [8192] FP16
    private var qBuf: MTLBuffer            // [32,128] FP16 (L2norm+GQA 后)
    private var kBuf: MTLBuffer            // [32,128] FP16
    private var vBuf: MTLBuffer            // [32,128] FP16
    private var gBuf: MTLBuffer            // [32] FP32
    private var betaBuf: MTLBuffer         // [32] FP32
    private var oBuf: MTLBuffer            // [32,128] FP32 (Delta Rule 输出)
    private var oNormedBuf: MTLBuffer      // [32,128] FP16 (norm 后)
    private var yBuf: MTLBuffer            // [2048] FP16

    public init(context: MetalContext, model: Model) throws {
        self.device = context.device
        self.config = Config()
        // deltanet.metal 编译进 module library
        self.psoProject = try context.pipeline("deltanet_project")
        self.psoConv1d = try context.pipeline("deltanet_conv1d")
        self.psoPrepare = try context.pipeline("deltanet_prepare")
        self.psoDeltaRule = try context.pipeline("deltanet_delta_rule")
        self.psoOutput = try context.pipeline("deltanet_output")

        // 状态分配
        hState = device.makeBuffer(length: 32 * 128 * 128 * MemoryLayout<Float>.size,
                                   options: .storageModePrivate)!
        hState.label = "deltanet.h"
        convState = device.makeBuffer(length: 8192 * 4 * MemoryLayout<Float16>.size,
                                      options: .storageModePrivate)!
        convState.label = "deltanet.conv_state"
        convStateTmp = device.makeBuffer(length: 8192 * 4 * MemoryLayout<Float16>.size,
                                         options: .storageModePrivate)!
        qkvBuf = device.makeBuffer(length: 8192 * MemoryLayout<Float16>.size, options: .storageModePrivate)!
        zBuf = device.makeBuffer(length: 4096 * MemoryLayout<Float16>.size, options: .storageModePrivate)!
        bBuf = device.makeBuffer(length: 32 * MemoryLayout<Float16>.size, options: .storageModePrivate)!
        aBuf = device.makeBuffer(length: 32 * MemoryLayout<Float16>.size, options: .storageModePrivate)!
        qkvPostBuf = device.makeBuffer(length: 8192 * MemoryLayout<Float16>.size, options: .storageModePrivate)!
        qBuf = device.makeBuffer(length: 32 * 128 * MemoryLayout<Float16>.size, options: .storageModePrivate)!
        kBuf = device.makeBuffer(length: 32 * 128 * MemoryLayout<Float16>.size, options: .storageModePrivate)!
        vBuf = device.makeBuffer(length: 32 * 128 * MemoryLayout<Float16>.size, options: .storageModePrivate)!
        gBuf = device.makeBuffer(length: 32 * MemoryLayout<Float>.size, options: .storageModePrivate)!
        betaBuf = device.makeBuffer(length: 32 * MemoryLayout<Float>.size, options: .storageModePrivate)!
        oBuf = device.makeBuffer(length: 32 * 128 * MemoryLayout<Float>.size, options: .storageModePrivate)!
        oNormedBuf = device.makeBuffer(length: 32 * 128 * MemoryLayout<Float16>.size, options: .storageModePrivate)!
        yBuf = device.makeBuffer(length: 2048 * MemoryLayout<Float16>.size, options: .storageModePrivate)!
    }

    /// 从 Model 的 resident 索引取权重 TensorView (原名)
    private func weight(_ model: Model, _ name: String) throws -> TensorView {
        try model.resident(name: name)
    }

    /// 单 token 前向: x [2048] FP16 → y [2048] FP16
    /// 注意: 完整实现还需 (1) qkv 分头+L2norm+g/beta 预处理 kernel,
    ///       (2) MoE 路径 (MTP 层/主模型共享), (3) shared expert.
    ///       本版本先验证 4 个核心 kernel 的调用与 golden 对齐。
    public func encodeToken(commandBuffer: MTLCommandBuffer,
                            x: MTLBuffer, xOffset: Int,
                            y: MTLBuffer, yOffset: Int,
                            model: Model) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        let threads = MTLSize(width: 256, height: 1, depth: 1)

        // ---- Kernel 1: 投影 GEMV (qkv/z/b/a, 4 次 dispatch) ----
        let wQKV = try weight(model, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
        let wZ   = try weight(model, "model.language_model.layers.0.linear_attn.in_proj_z.weight")
        let wB   = try weight(model, "model.language_model.layers.0.linear_attn.in_proj_b.weight")
        let wA   = try weight(model, "model.language_model.layers.0.linear_attn.in_proj_a.weight")

        // qkv (8192 输出)
        enc.setComputePipelineState(psoProject)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(wQKV.buffer, offset: Int(wQKV.offset), index: 1)
        enc.setBuffer(wZ.buffer, offset: Int(wZ.offset), index: 2)
        enc.setBuffer(wB.buffer, offset: Int(wB.offset), index: 3)
        enc.setBuffer(wA.buffer, offset: Int(wA.offset), index: 4)
        enc.setBuffer(qkvBuf, offset: 0, index: 5)
        enc.setBuffer(zBuf, offset: 0, index: 6)
        enc.setBuffer(bBuf, offset: 0, index: 7)
        enc.setBuffer(aBuf, offset: 0, index: 8)
        var mode: UInt32 = 0
        enc.setBytes(&mode, length: 4, index: 9)
        enc.dispatchThreads(MTLSize(width: 8192, height: 1, depth: 1),
                            threadsPerThreadgroup: threads)

        // ---- Kernel 2: conv1d (8192 通道, 用双 buffer 防竞争) ----
        let wConv = try weight(model, "model.language_model.layers.0.linear_attn.conv1d.weight")
        enc.setComputePipelineState(psoConv1d)
        enc.setBuffer(qkvBuf, offset: 0, index: 0)
        enc.setBuffer(wConv.buffer, offset: Int(wConv.offset), index: 1)
        enc.setBuffer(convState, offset: 0, index: 2)
        enc.setBuffer(qkvPostBuf, offset: 0, index: 3)
        enc.setBuffer(convStateTmp, offset: 0, index: 4)
        enc.dispatchThreads(MTLSize(width: 8192, height: 1, depth: 1),
                            threadsPerThreadgroup: threads)
        // 交换 conv state (tmp → active)
        swap(&convState, &convStateTmp)

        // ---- Kernel 2.5: 预处理 (分头+L2norm+GQA 展开 + g/beta) ----
        // 注意: A_log/dt_bias 是标量张量 [32], b/a 是 GEMV 输出 (bBuf/aBuf FP16)
        let wALog = try weight(model, "model.language_model.layers.0.linear_attn.A_log")
        let wDt   = try weight(model, "model.language_model.layers.0.linear_attn.dt_bias")
        enc.setComputePipelineState(psoPrepare)
        enc.setBuffer(qkvPostBuf, offset: 0, index: 0)
        enc.setBuffer(aBuf, offset: 0, index: 1)
        enc.setBuffer(bBuf, offset: 0, index: 2)
        enc.setBuffer(wALog.buffer, offset: Int(wALog.offset), index: 3)
        enc.setBuffer(wDt.buffer, offset: Int(wDt.offset), index: 4)
        enc.setBuffer(qBuf, offset: 0, index: 5)
        enc.setBuffer(kBuf, offset: 0, index: 6)
        enc.setBuffer(vBuf, offset: 0, index: 7)
        enc.setBuffer(gBuf, offset: 0, index: 8)
        enc.setBuffer(betaBuf, offset: 0, index: 9)
        enc.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        // ---- Kernel 3: Delta Rule (32 V 头, 每头 128 线程) ----
        enc.setComputePipelineState(psoDeltaRule)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kBuf, offset: 0, index: 1)
        enc.setBuffer(vBuf, offset: 0, index: 2)
        enc.setBuffer(gBuf, offset: 0, index: 3)
        enc.setBuffer(betaBuf, offset: 0, index: 4)
        enc.setBuffer(hState, offset: 0, index: 5)
        enc.setBuffer(oBuf, offset: 0, index: 6)
        // 32 TGs × 128 线程 (每 V 头 1 TG), v_head = threadgroup_position_in_grid
        enc.dispatchThreads(MTLSize(width: 32 * 128, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

        // ---- Kernel 4a: Gated RMSNorm (32 头 × 128 = 4096 线程) ----
        let wNorm = try weight(model, "model.language_model.layers.0.linear_attn.norm.weight")
        let wOut  = try weight(model, "model.language_model.layers.0.linear_attn.out_proj.weight")
        enc.setComputePipelineState(psoOutput)
        enc.setBuffer(oBuf, offset: 0, index: 0)
        enc.setBuffer(zBuf, offset: 0, index: 1)
        enc.setBuffer(wNorm.buffer, offset: Int(wNorm.offset), index: 2)
        enc.setBuffer(wOut.buffer, offset: Int(wOut.offset), index: 3)
        enc.setBuffer(yBuf, offset: 0, index: 4)
        enc.setBuffer(oNormedBuf, offset: 0, index: 5)
        var modeOut: UInt32 = 0
        enc.setBytes(&modeOut, length: 4, index: 6)
        // mode 0: 32 TGs × 128 线程 (每头 1 TG)
        enc.dispatchThreads(MTLSize(width: 32 * 128, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

        // ---- Kernel 4b: out_proj (2048 行并行) ----
        modeOut = 1
        enc.setBytes(&modeOut, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: 2048, height: 1, depth: 1),
                            threadsPerThreadgroup: threads)

        // y 输出 → 调用方 buffer
        enc.setComputePipelineState(psoProject)  // 复用 GEMV kernel 做一次 memcpy? 不 — 用 blit
        enc.endEncoding()
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.copy(from: yBuf, sourceOffset: 0,
                      to: y, destinationOffset: yOffset,
                      size: 2048 * MemoryLayout<Float16>.size)
            blit.endEncoding()
        }
    }
}
