import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Micro-benchmark for the decode attention kernels at the real Gemma 4
/// shapes. Not a correctness test — it measures per-call GPU time (command
/// buffer spans, not wall clock) so we can tell whether the `attn` share of
/// the GPU timeline is kernel-bound (memory/barriers) or launch-bound
/// (two-pass split-KV + tiny grids).
///
/// Run with: swift test --filter AttentionBench
@Suite(.enabled(if: ProcessInfo.processInfo.environment["RUN_ATTN_BENCH"] == "1"))
struct AttentionBench {

    private struct Config: CustomStringConvertible {
        let name: String
        let headDim: Int
        let numQHeads: Int
        let numKVHeads: Int
        let seqLen: Int
        let window: Int?   // nil = full attention
        var description: String { name }
    }

    private static let cfgs: [Config] = [
        // Gemma 4 real shapes
        Config(name: "full-512",  headDim: 512, numQHeads: 16, numKVHeads: 2,  seqLen: 128, window: nil),
        Config(name: "full-512",  headDim: 512, numQHeads: 16, numKVHeads: 2,  seqLen: 256, window: nil),
        Config(name: "full-512",  headDim: 512, numQHeads: 16, numKVHeads: 2,  seqLen: 1024, window: nil),
        Config(name: "swa-256",   headDim: 256, numQHeads: 16, numKVHeads: 8,  seqLen: 128, window: 128),
        Config(name: "swa-256",   headDim: 256, numQHeads: 16, numKVHeads: 8,  seqLen: 1024, window: 1024),
    ]

    @Test func benchAttentionKernels() throws {
        let ctx = try MetalContext()
        let attention = try Attention(context: ctx)
        var rng = SeedTree(0xBEEF).key("attn-bench")

        for cfg in Self.cfgs {
            let qCount = cfg.numQHeads * cfg.headDim
            let kvCount = cfg.seqLen * cfg.numKVHeads * cfg.headDim
            let qFp16 = (0..<qCount).map { _ in Float16(rng.uniform(-0.25, 0.25)) }
            let kFp16 = (0..<kvCount).map { _ in Float16(rng.uniform(-0.25, 0.25)) }
            let vFp16 = (0..<kvCount).map { _ in Float16(rng.uniform(-0.25, 0.25)) }
            guard let qBuf = Fp16Buffer.make(ctx.device, halves: qFp16),
                  let kBuf = Fp16Buffer.make(ctx.device, halves: kFp16),
                  let vBuf = Fp16Buffer.make(ctx.device, halves: vFp16),
                  let outBuf = Fp16Buffer.make(ctx.device, count: qCount) else {
                Issue.record("alloc failed for \(cfg)"); continue
            }

            let iterations = 200
            let cb = ctx.queue.makeCommandBuffer()!
            for _ in 0..<iterations {
                if let window = cfg.window {
                    attention.encodeSWA(commandBuffer: cb,
                                        q: qBuf, k: kBuf, v: vBuf, out: outBuf,
                                        headDim: UInt32(cfg.headDim),
                                        numQHeads: UInt32(cfg.numQHeads),
                                        numKVHeads: UInt32(cfg.numKVHeads),
                                        seqLen: UInt32(cfg.seqLen),
                                        window: UInt32(window),
                                        scale: 1.0)
                } else {
                    attention.encodeFull(commandBuffer: cb,
                                         q: qBuf, k: kBuf, v: vBuf, out: outBuf,
                                         headDim: UInt32(cfg.headDim),
                                         numQHeads: UInt32(cfg.numQHeads),
                                         numKVHeads: UInt32(cfg.numKVHeads),
                                         seqLen: UInt32(cfg.seqLen),
                                         scale: 1.0)
                }
            }
            // A trailing 1-dispatch buffer to read precise GPU timestamps.
            let cb1 = ctx.queue.makeCommandBuffer()!
            if let window = cfg.window {
                attention.encodeSWA(commandBuffer: cb1,
                                    q: qBuf, k: kBuf, v: vBuf, out: outBuf,
                                    headDim: UInt32(cfg.headDim),
                                    numQHeads: UInt32(cfg.numQHeads),
                                    numKVHeads: UInt32(cfg.numKVHeads),
                                    seqLen: UInt32(cfg.seqLen),
                                    window: UInt32(window),
                                    scale: 1.0)
            } else {
                attention.encodeFull(commandBuffer: cb1,
                                     q: qBuf, k: kBuf, v: vBuf, out: outBuf,
                                     headDim: UInt32(cfg.headDim),
                                     numQHeads: UInt32(cfg.numQHeads),
                                     numKVHeads: UInt32(cfg.numKVHeads),
                                     seqLen: UInt32(cfg.seqLen),
                                     scale: 1.0)
            }

            cb.commit()
            cb1.commit()
            let start = DispatchTime.now()
            cb.waitUntilCompleted()
            let warm = DispatchTime.now()
            let bulk = Double(warm.uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
            let singleGpu: Double = {
                let gs = cb1.gpuStartTime, ge = cb1.gpuEndTime
                guard gs > 0, ge > gs, ge.isFinite else { return -1 }
                return ge - gs
            }()
            let perCallWall = bulk / Double(iterations)

            let q = qFp16.map { Float($0) }
            let k = kFp16.map { Float($0) }
            let v = vFp16.map { Float($0) }
            let ref = AttentionRef.apply(
                q: q, k: k, v: v,
                headDim: cfg.headDim, numQHeads: cfg.numQHeads,
                numKVHeads: cfg.numKVHeads, seqLen: cfg.seqLen,
                window: cfg.window, scale: 1.0)
            let actual = Fp16Buffer.read(outBuf, count: qCount)
            let rel = RelError.compute(actual: actual, reference: ref)

            print(String(format: "[attn-bench] %@ T=%d  bulk=%.1fus/call  singleGPU=%.1fus  rel=%.2e",
                         cfg.name, cfg.seqLen, perCallWall * 1e6, singleGpu * 1e6, rel))
            // Sanity: output must still match the reference on the last call.
            #expect(rel < Tolerance.fp16ChainedReduction,
                    "bench \(cfg) produced garbage rel=\(rel)")
        }
    }
}
