import Foundation

public enum Quantization {

    public static let groupSize: Int = 64

    // MARK: - BF16 helpers
    //
    // BF16 = top 16 bits of FP32 (8-bit exponent, 7-bit mantissa). Stored on
    // disk and on the GPU as the native `bfloat` type; in Swift we carry the
    // raw bits as `UInt16` and convert via this pair. Round-half-to-even on
    // encode matches the IEEE-754 default. No NaN/Inf special-case: fixtures
    // are bounded and finite, and the .gturbo importer copies BF16 bytes
    // through unchanged so we never call the encoder on weight data.

    @inline(__always)
    public static func bf16Bits(_ x: Float) -> UInt16 {
        let bits = x.bitPattern
        let lsb  = (bits >> 16) & 1
        let roundingBias: UInt32 = 0x7FFF &+ lsb
        let rounded = (bits &+ roundingBias) >> 16
        return UInt16(truncatingIfNeeded: rounded)
    }

    @inline(__always)
    public static func bf16ToFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    // MARK: - INT4 affine

    /// MLX `affine` 4-bit row. Packed unsigned nibbles, BF16 scale + bias per
    /// group of 64. `scales` and `biases` carry BF16 bit patterns as `UInt16`
    /// so the same buffer can be uploaded to a Metal `device const bfloat*`.
    public struct Int4AffineRow {
        public let packed: [UInt8]   // N / 2 bytes; low nibble = even index, high = odd
        public let scales: [UInt16]  // N / 64 BF16 bits
        public let biases: [UInt16]  // N / 64 BF16 bits

        public init(packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
            self.packed = packed
            self.scales = scales
            self.biases = biases
        }
    }

    /// Affine 4-bit quantize: `q ∈ [0..15]`, `w ≈ q * scale + bias`.
    /// Scale and bias are computed from per-group min/max, then rounded to BF16.
    /// Test-fixture only — the runtime importer never calls this.
    public static func quantizeInt4Affine(_ row: [Float]) -> Int4AffineRow {
        precondition(row.count % groupSize == 0,
                     "row length \(row.count) is not a multiple of \(groupSize)")

        let nGroups = row.count / groupSize
        var packed = [UInt8](repeating: 0, count: row.count / 2)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for g in 0..<nGroups {
            var wmin: Float =  .infinity
            var wmax: Float = -.infinity
            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            // Constant group: scale=1, bias=value preserves exact reconstruction.
            let scaleF: Float
            let biasF:  Float
            if wmax == wmin {
                scaleF = 1
                biasF  = wmin
            } else {
                scaleF = (wmax - wmin) / 15.0
                biasF  = wmin
            }
            // Round through BF16 first, then quantize against the rounded
            // values so the runtime decode (which reads BF16) reproduces the
            // same q the fixture stored.
            let sBits = bf16Bits(scaleF)
            let bBits = bf16Bits(biasF)
            scales[g] = sBits
            biases[g] = bBits
            let scale = bf16ToFloat(sBits)
            let bias  = bf16ToFloat(bBits)
            let invScale = scale == 0 ? Float(0) : 1.0 / scale

            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                var q = Int(((w - bias) * invScale).rounded())
                q = max(0, min(15, q))
                let nibble = UInt8(q) & 0x0F
                let byteIdx = g * (groupSize / 2) + (k / 2)
                if (k & 1) == 0 {
                    packed[byteIdx] = (packed[byteIdx] & 0xF0) | nibble
                } else {
                    packed[byteIdx] = (packed[byteIdx] & 0x0F) | (nibble << 4)
                }
            }
        }
        return Int4AffineRow(packed: packed, scales: scales, biases: biases)
    }

    public static func dequantizeInt4Affine(_ r: Int4AffineRow, n: Int) -> [Float] {
        precondition(n == r.packed.count * 2)
        var out = [Float](repeating: 0, count: n)
        let nGroups = n / groupSize
        for g in 0..<nGroups {
            let scale = bf16ToFloat(r.scales[g])
            let bias  = bf16ToFloat(r.biases[g])
            for k in 0..<groupSize {
                let byteIdx = g * (groupSize / 2) + (k / 2)
                let b = r.packed[byteIdx]
                let nibble: Int = (k & 1) == 0 ? Int(b & 0x0F) : Int(b >> 4)
                out[g * groupSize + k] = Float(nibble) * scale + bias
            }
        }
        return out
    }

    // MARK: - INT8 affine

    public struct Int8AffineRow {
        public let packed: [UInt8]   // N unsigned bytes
        public let scales: [UInt16]  // N / 64 BF16 bits
        public let biases: [UInt16]  // N / 64 BF16 bits

        public init(packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
            self.packed = packed
            self.scales = scales
            self.biases = biases
        }
    }

    public static func quantizeInt8Affine(_ row: [Float]) -> Int8AffineRow {
        precondition(row.count % groupSize == 0,
                     "row length \(row.count) is not a multiple of \(groupSize)")

        let nGroups = row.count / groupSize
        var packed = [UInt8](repeating: 0, count: row.count)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for g in 0..<nGroups {
            var wmin: Float =  .infinity
            var wmax: Float = -.infinity
            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            let scaleF: Float
            let biasF:  Float
            if wmax == wmin {
                scaleF = 1
                biasF  = wmin
            } else {
                scaleF = (wmax - wmin) / 255.0
                biasF  = wmin
            }
            let sBits = bf16Bits(scaleF)
            let bBits = bf16Bits(biasF)
            scales[g] = sBits
            biases[g] = bBits
            let scale = bf16ToFloat(sBits)
            let bias  = bf16ToFloat(bBits)
            let invScale = scale == 0 ? Float(0) : 1.0 / scale

            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                var q = Int(((w - bias) * invScale).rounded())
                q = max(0, min(255, q))
                packed[g * groupSize + k] = UInt8(q)
            }
        }
        return Int8AffineRow(packed: packed, scales: scales, biases: biases)
    }

    public static func dequantizeInt8Affine(_ r: Int8AffineRow, n: Int) -> [Float] {
        precondition(n == r.packed.count)
        var out = [Float](repeating: 0, count: n)
        let nGroups = n / groupSize
        for g in 0..<nGroups {
            let scale = bf16ToFloat(r.scales[g])
            let bias  = bf16ToFloat(r.biases[g])
            for k in 0..<groupSize {
                out[g * groupSize + k] = Float(r.packed[g * groupSize + k]) * scale + bias
            }
        }
        return out
    }

    // MARK: - Generic N-bit affine (2 / 3 / 4)

    /// Bit-packed affine row for any supported bit width. Codes are unsigned
    /// in `[0, 2^bits-1]`; scale + bias are BF16 per `groupSize` elements.
    ///
    /// Packing (low element index = least significant bits):
    ///   - 4-bit: 2 codes/byte (low nibble = even index) — identical to `Int4AffineRow`.
    ///   - 2-bit: 4 codes/byte.
    ///   - 3-bit: 8 codes packed little-endian into 3 bytes (24 bits).
    public struct QuantizedAffineRow {
        public let bits: Int
        public let packed: [UInt8]
        public let scales: [UInt16]  // BF16 bits per group
        public let biases: [UInt16]  // BF16 bits per group

        public init(bits: Int, packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
            precondition(bits == 2 || bits == 3 || bits == 4,
                         "unsupported weight bits \(bits)")
            self.bits = bits
            self.packed = packed
            self.scales = scales
            self.biases = biases
        }

        /// Number of packed bytes for `elements` at `bits`.
        public static func packedRowBytes(elements: Int, bits: Int) -> Int {
            precondition(bits == 2 || bits == 3 || bits == 4)
            return (elements * bits + 7) / 8
        }
    }

    /// Affine N-bit quantize with the same rounding discipline as the 4-bit
    /// path: scale/bias are rounded through BF16 first, then codes are computed
    /// against the rounded scale/bias so the runtime decode (which reads BF16)
    /// reproduces the stored codes exactly.
    public static func quantizeAffine(
        _ row: [Float],
        bits: Int,
        groupSize: Int = Quantization.groupSize
    ) -> QuantizedAffineRow {
        precondition(bits == 2 || bits == 3 || bits == 4,
                     "unsupported weight bits \(bits)")
        precondition(row.count % groupSize == 0,
                     "row length \(row.count) is not a multiple of \(groupSize)")
        let qMax = (1 << bits) - 1
        let nGroups = row.count / groupSize
        let packedCount = QuantizedAffineRow.packedRowBytes(elements: row.count, bits: bits)
        var packed = [UInt8](repeating: 0, count: packedCount)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for g in 0..<nGroups {
            var wmin: Float =  .infinity
            var wmax: Float = -.infinity
            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            let scaleF: Float
            let biasF:  Float
            if wmax == wmin {
                scaleF = 1
                biasF  = wmin
            } else {
                scaleF = (wmax - wmin) / Float(qMax)
                biasF  = wmin
            }
            let sBits = bf16Bits(scaleF)
            let bBits = bf16Bits(biasF)
            scales[g] = sBits
            biases[g] = bBits
            let scale = bf16ToFloat(sBits)
            let bias  = bf16ToFloat(bBits)
            let invScale = scale == 0 ? Float(0) : 1.0 / scale

            for k in 0..<groupSize {
                let w = row[g * groupSize + k]
                var q = Int(((w - bias) * invScale).rounded())
                q = max(0, min(qMax, q))
                packCode(&packed, q, at: g * groupSize + k, bits: bits)
            }
        }
        return QuantizedAffineRow(bits: bits, packed: packed, scales: scales, biases: biases)
    }

    public static func dequantizeAffine(
        _ r: QuantizedAffineRow,
        count: Int,
        groupSize: Int = Quantization.groupSize
    ) -> [Float] {
        precondition(r.bits == 2 || r.bits == 3 || r.bits == 4)
        precondition(count % groupSize == 0)
        var out = [Float](repeating: 0, count: count)
        let nGroups = count / groupSize
        for g in 0..<nGroups {
            let scale = bf16ToFloat(r.scales[g])
            let bias  = bf16ToFloat(r.biases[g])
            for k in 0..<groupSize {
                let code = unpackCode(r.packed, at: g * groupSize + k, bits: r.bits)
                out[g * groupSize + k] = Float(code) * scale + bias
            }
        }
        return out
    }

    /// Write one code into the bit-packed buffer (low index = LSB).
    public static func packCode(_ buf: inout [UInt8], _ code: Int, at index: Int, bits: Int) {
        switch bits {
        case 4:
            let byteIdx = index / 2
            let nibble = UInt8(code) & 0x0F
            if (index & 1) == 0 {
                buf[byteIdx] = (buf[byteIdx] & 0xF0) | nibble
            } else {
                buf[byteIdx] = (buf[byteIdx] & 0x0F) | (nibble << 4)
            }
        case 2:
            let byteIdx = index / 4
            let shift = UInt8((index % 4) * 2)
            buf[byteIdx] = (buf[byteIdx] & ~(0x03 << shift)) | (UInt8(code & 0x03) << shift)
        case 3:
            let bit = index * 3
            let byteIdx = bit / 8
            let shift = UInt8(bit % 8)
            let v = UInt16(code & 0x07) << shift
            buf[byteIdx] |= UInt8(v & 0xFF)
            if byteIdx + 1 < buf.count {
                buf[byteIdx + 1] |= UInt8((v >> 8) & 0xFF)
            }
        default:
            fatalError("unsupported bits \(bits)")
        }
    }

    /// Read one code from the bit-packed buffer (low index = LSB).
    public static func unpackCode(_ buf: [UInt8], at index: Int, bits: Int) -> Int {
        switch bits {
        case 4:
            let byte = buf[index / 2]
            return (index & 1) == 0 ? Int(byte & 0x0F) : Int(byte >> 4)
        case 2:
            let byte = buf[index / 4]
            return Int((byte >> UInt8((index % 4) * 2)) & 0x03)
        case 3:
            let bit = index * 3
            let byteIdx = bit / 8
            let shift = bit % 8
            let b0 = UInt32(buf[byteIdx])
            let b1 = UInt32(byteIdx + 1 < buf.count ? buf[byteIdx + 1] : 0)
            let b2 = UInt32(byteIdx + 2 < buf.count ? buf[byteIdx + 2] : 0)
            let v = b0 | (b1 << 8) | (b2 << 16)
            return Int((v >> UInt32(shift)) & 0x07)
        default:
            fatalError("unsupported bits \(bits)")
        }
    }
}
