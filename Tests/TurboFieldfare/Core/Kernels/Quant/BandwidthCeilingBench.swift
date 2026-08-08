import Testing
import Foundation
import Metal
@testable import TurboFieldfare

/// Measures the M4's practical DRAM read bandwidth ceiling with a trivial
/// read-only kernel (sum bytes, write 1 value) so we know how close the GEMV
/// kernel is to the hardware limit.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["RUN_GEMV_BENCH"] == "1"))
struct BandwidthCeilingBench {

    @Test func readBandwidthCeiling() throws {
        let ctx = try MetalContext()
        let sizes = [6_494_208, 12_988_416, 26_000_000]
        for size in sizes {
            guard let buf = ctx.device.makeBuffer(length: size, options: .storageModePrivate),
                  let out = ctx.device.makeBuffer(length: 4, options: .storageModeShared) else {
                Issue.record("alloc"); continue
            }
            // Fill.
            let stage = ctx.device.makeBuffer(length: size, options: .storageModeShared)!
            let cb0 = ctx.queue.makeCommandBuffer()!
            let b0 = cb0.makeBlitCommandEncoder()!
            b0.fill(buffer: buf, range: 0..<size, value: 0xAB)
            b0.endEncoding()
            cb0.commit(); cb0.waitUntilCompleted()

            let src = try ctx.pipeline("bw_ceiling_read")
            let iterations = 200
            let cb = ctx.queue.makeCommandBuffer()!
            for _ in 0..<iterations {
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(src)
                enc.setBuffer(buf, offset: 0, index: 0)
                enc.setBuffer(out, offset: 0, index: 1)
                var n = UInt32(size / 4)
                enc.setBytes(&n, length: 4, index: 2)
                // Each thread reads 256 uints (1 KiB); threads = n/256.
                let threads = (size / 4 + 255) / 256
                let tg = (threads + 1023) / 1024
                enc.dispatchThreadgroups(MTLSize(width: tg, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
                enc.endEncoding()
            }
            cb.commit()
            let w0 = DispatchTime.now()
            cb.waitUntilCompleted()
            let wall = Double(DispatchTime.now().uptimeNanoseconds - w0.uptimeNanoseconds) / 1e9
            let perCall = wall / Double(iterations)
            let bw = Double(size) / perCall / 1e9
            print(String(format: "[bw-ceil] size=%d  %.1f GB/s (%.2f MB/call, %.1fus)",
                         size, bw, Double(size) / 1e6, perCall * 1e6))
        }
    }
}
