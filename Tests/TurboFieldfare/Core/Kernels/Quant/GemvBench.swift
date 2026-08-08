import Testing
import Foundation
import Metal
@testable import TurboFieldfare

/// Micro-benchmark for the int4 affine GEMV kernel at the real Gemma 4 decode
/// shapes. Measures sustained DRAM bandwidth (weights + scale/bias bytes read
/// per call). Run with: RUN_GEMV_BENCH=1 swift test --filter GemvBench
@Suite(.enabled(if: ProcessInfo.processInfo.environment["RUN_GEMV_BENCH"] == "1"))
struct GemvBench {

    private struct Shape: CustomStringConvertible {
        let m: Int, n: Int
        var description: String { "\(m)x\(n)" }
        var weightBytes: Int { m * (n / 2) }                     // int4 packed
        var auxBytes: Int { m * (n / 64) * 2 * 2 }               // BF16 s+b
        var totalBytes: Int { weightBytes + auxBytes }
    }

    private static let shapes: [Shape] = [
        Shape(m: 4096, n: 2816),   // gate/up at N=2816
        Shape(m: 2816, n: 4096),   // gate_proj→neighbors
        Shape(m: 2816, n: 8192),   // gate_proj 8192
        Shape(m: 8192, n: 2816),   // up 8192 rows
    ]

    @Test func benchInt4Gemv() throws {
        let ctx = try MetalContext()
        let gemv = try DequantInt4GEMV(context: ctx)
        var rng = SystemRandomNumberGenerator()

        for shape in Self.shapes {
            // Buffers. W is [M][N/2] bytes, scales/biases [M][N/64] BF16.
            let wLen = shape.m * (shape.n / 2)
            let auxLen = shape.m * (shape.n / 64)
            guard let wBuf = ctx.device.makeBuffer(length: wLen, options: .storageModePrivate),
                  let sBuf = ctx.device.makeBuffer(length: auxLen * 2, options: .storageModePrivate),
                  let bBuf = ctx.device.makeBuffer(length: auxLen * 2, options: .storageModePrivate),
                  let xBuf = ctx.device.makeBuffer(length: shape.n * 2, options: .storageModePrivate),
                  let yBuf = ctx.device.makeBuffer(length: shape.m * 2, options: .storageModeShared) else {
                Issue.record("alloc failed \(shape)"); continue
            }
            // Fill with plausible data via shared staging.
            func fill(_ buf: MTLBuffer, _ count: Int, _ gen: (Int) -> UInt8) {
                let bytes = (0..<count).map(gen)
                let stage = ctx.device.makeBuffer(bytes: bytes, length: count, options: .storageModeShared)!
                let cb = ctx.queue.makeCommandBuffer()!
                let blit = cb.makeBlitCommandEncoder()!
                blit.copy(from: stage, sourceOffset: 0, to: buf, destinationOffset: 0, size: count)
                blit.endEncoding()
                cb.commit(); cb.waitUntilCompleted()
            }
            fill(wBuf, wLen, { _ in UInt8.random(in: 0...255) })
            fill(sBuf, auxLen * 2, { _ in UInt8.random(in: 0...255) })
            fill(bBuf, auxLen * 2, { _ in UInt8.random(in: 0...255) })
            fill(xBuf, shape.n * 2, { _ in UInt8.random(in: 0...255) })

            // Warmup + correctness cross-check: tgx vs simd output must match.
            var cb0 = ctx.queue.makeCommandBuffer()!
            gemv.encode(commandBuffer: cb0, weights: wBuf, scales: sBuf, biases: bBuf,
                        x: xBuf, y: yBuf, m: UInt32(shape.m), n: UInt32(shape.n))
            cb0.commit(); cb0.waitUntilCompleted()
            // Read via pointer:
            let refPtr = yBuf.contents().assumingMemoryBound(to: Float16.self)
            var refVals = [Float16](repeating: 0, count: shape.m)
            for i in 0..<shape.m { refVals[i] = refPtr[i] }
            let yTgx = ctx.device.makeBuffer(length: shape.m * 2, options: .storageModeShared)!
            cb0 = ctx.queue.makeCommandBuffer()!
            gemv.encodeTgx(commandBuffer: cb0, weights: wBuf, scales: sBuf, biases: bBuf,
                           x: xBuf, y: yTgx, m: UInt32(shape.m), n: UInt32(shape.n))
            cb0.commit(); cb0.waitUntilCompleted()
            let tgxPtr = yTgx.contents().assumingMemoryBound(to: Float16.self)
            var maxDiff = Float(0)
            for i in 0..<shape.m {
                maxDiff = max(maxDiff, abs(Float(refVals[i]) - Float(tgxPtr[i])))
            }
            print("[gemv] \(shape) correctness tgx-vs-simd maxDiff=\(maxDiff)")
            #expect(maxDiff == 0, "tgx mismatch \(shape)")

            for _ in 0..<5 {
                let cb = ctx.queue.makeCommandBuffer()!
                gemv.encode(commandBuffer: cb, weights: wBuf, scales: sBuf, biases: bBuf,
                            x: xBuf, y: yBuf, m: UInt32(shape.m), n: UInt32(shape.n))
                cb.commit(); cb.waitUntilCompleted()
            }

            let iterations = 100
            func timed(_ enc: (MTLCommandBuffer) -> Void) -> Double {
                let cb = ctx.queue.makeCommandBuffer()!
                let t0 = DispatchTime.now()
                for _ in 0..<iterations { enc(cb) }
                cb.commit()
                let w0 = DispatchTime.now()
                cb.waitUntilCompleted()
                return Double(DispatchTime.now().uptimeNanoseconds - w0.uptimeNanoseconds) / 1e9 * 1e6 / Double(iterations)
            }
            let wallUs = timed { gemv.encode(commandBuffer: $0, weights: wBuf, scales: sBuf,
                                             biases: bBuf, x: xBuf, y: yBuf,
                                             m: UInt32(shape.m), n: UInt32(shape.n)) }
            let tgxUs = timed { gemv.encodeTgx(commandBuffer: $0, weights: wBuf, scales: sBuf,
                                               biases: bBuf, x: xBuf, y: yTgx,
                                               m: UInt32(shape.m), n: UInt32(shape.n)) }

            let gbPerS = Double(shape.totalBytes) / (wallUs * 1e-6) / 1e9
            let gbPerSTgx = Double(shape.totalBytes) / (tgxUs * 1e-6) / 1e9
            print(String(format: "[gemv] %@  simd=%.1fus (%.1f GB/s)  tgx=%.1fus (%.1f GB/s)  speedup=%.2fx",
                         shape.description, wallUs, gbPerS, tgxUs, gbPerSTgx, wallUs / tgxUs))
        }
    }
}
