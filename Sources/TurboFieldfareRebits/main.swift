import Foundation
import CryptoKit
import TurboFieldfare

// ============================================================================
// TurboFieldfareRebits — routed-expert bit-width A/B tool.
//
// Reads an existing .gturbo model whose routed experts are 4-bit affine
// (group 64, BF16 scale+bias), dequantizes them, re-quantizes at 2 or 3 bits
// with the same affine scheme, and writes a new model directory with an
// updated layout.json / manifest.json / verified-install.json.
//
// This is the "weightBits=2/3 repack path": it produces a model the runtime
// can load and decode with the b2/b3 Metal kernels, while leaving attention /
// embedding / router / shared experts untouched at their original bit widths.
//
// Usage:
//   TurboFieldfareRebits --input <gturbo-dir> --output <dir> --routed-bits 2|3
// ============================================================================

private struct Args {
    var input: String?
    var output: String?
    var routedBits: Int = 3
    var groupSize: Int = Quantization.groupSize

    static func parse(_ values: [String]) throws -> Args {
        var a = Args()
        var i = 1
        while i < values.count {
            switch values[i] {
            case "--input", "--output":
                guard i + 1 < values.count else {
                    throw NSError(domain: "rebits", code: 2,
                                  userInfo: [NSLocalizedDescriptionKey: "\(values[i]) needs a value"])
                }
                if values[i] == "--input" { a.input = values[i + 1] } else { a.output = values[i + 1] }
                i += 2
            case "--routed-bits":
                guard i + 1 < values.count, let b = Int(values[i + 1]), b == 2 || b == 3 else {
                    throw NSError(domain: "rebits", code: 2,
                                  userInfo: [NSLocalizedDescriptionKey: "--routed-bits must be 2 or 3"])
                }
                a.routedBits = b
                i += 2
            case "--help":
                print("""
                Usage: TurboFieldfareRebits --input <gturbo-dir> --output <dir> --routed-bits 2|3
                Re-quantizes routed experts of an existing 4-bit .gturbo to 2 or 3 bits.
                """)
                exit(0)
            default:
                throw NSError(domain: "rebits", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "unknown flag \(values[i])"])
            }
        }
        guard let input = a.input, let output = a.output else {
            throw NSError(domain: "rebits", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "--input and --output are required"])
        }
        a.input = input
        a.output = output
        return a
    }
}

private struct TensorRegion {
    let offset: UInt64
    let size: UInt64
    let dtype: String
    let shape: [Int]
    let bits: Int?
}

private struct ExpertInfo {
    let expert: Int
    let physicalRank: Int
    let offset: UInt64
    let tensors: [String: TensorRegion]
}

private struct LayerInfo {
    let layer: Int
    let file: String
    let experts: [ExpertInfo]
}

private struct QualityAccumulator {
    var n = 0
    var sumSqErr: Double = 0
    var maxAbsErr: Double = 0

    mutating func add(row4: [Float], rowN: [Float]) {
        var s: Double = 0
        var m: Double = 0
        for i in 0..<row4.count {
            let d = Double(row4[i] - rowN[i])
            s += d * d
            if abs(d) > m { m = abs(d) }
        }
        n += row4.count
        sumSqErr += s
        if m > maxAbsErr { maxAbsErr = m }
    }

    var rmse: Double { n > 0 ? (sumSqErr / Double(n)).squareRoot() : 0 }
}

private func readFile(_ path: String) throws -> Data {
    try Data(contentsOf: URL(fileURLWithPath: path))
}

private func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

// ---- parsing ----

private func parseLayout(_ data: Data) throws -> (numLayers: Int, expertsPerLayer: Int, layers: [LayerInfo]) {
    let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
    guard let numLayers = root["numLayers"] as? Int,
          let expertsPerLayer = root["expertsPerLayer"] as? Int,
          let layersArr = root["layers"] as? [[String: Any]] else {
        throw NSError(domain: "rebits", code: 3,
                      userInfo: [NSLocalizedDescriptionKey: "layout.json malformed"])
    }
    var layers: [LayerInfo] = []
    for lo in layersArr {
        guard let layer = lo["layer"] as? Int,
              let file = lo["file"] as? String,
              let exps = lo["experts"] as? [[String: Any]] else { continue }
        var infos: [ExpertInfo] = []
        for eo in exps {
            guard let expert = eo["expert"] as? Int,
                  let physicalRank = eo["physicalRank"] as? Int,
                  let blobOffset = (eo["offset"] as? NSNumber)?.uint64Value,
                  let tens = eo["tensors"] as? [String: [String: Any]] else { continue }
            var tensors: [String: TensorRegion] = [:]
            for (role, to) in tens {
                tensors[role] = TensorRegion(
                    offset: (to["offset"] as? NSNumber)?.uint64Value ?? 0,
                    size: (to["size"] as? NSNumber)?.uint64Value ?? 0,
                    dtype: to["dtype"] as? String ?? "",
                    shape: (to["shape"] as? [Int]) ?? [],
                    bits: to["bits"] as? Int)
            }
            infos.append(ExpertInfo(expert: expert, physicalRank: physicalRank,
                                     offset: blobOffset, tensors: tensors))
        }
        layers.append(LayerInfo(layer: layer, file: file, experts: infos))
    }
    return (numLayers, expertsPerLayer, layers)
}

// ---- per-role requant ----

/// Dequantize a 4-bit affine region row by row, re-quantize at `bits`, write
/// the new packed region; accumulate quality stats vs the 4-bit reference.
private func requantRegion(
    blob: Data,
    region: TensorRegion,
    scalesRegion: TensorRegion,
    biasesRegion: TensorRegion,
    expertOffset: UInt64,
    bits: Int,
    groupSize: Int,
    stats: inout QualityAccumulator
) throws -> (w: Data, s: Data, b: Data) {
    guard region.shape.count == 2 else {
        throw NSError(domain: "rebits", code: 4,
                      userInfo: [NSLocalizedDescriptionKey: "unexpected shape \(region.shape)"])
    }
    let rows = region.shape[0]
    let n = region.shape[1]
    guard n % groupSize == 0 else {
        throw NSError(domain: "rebits", code: 4,
                      userInfo: [NSLocalizedDescriptionKey: "N \(n) not multiple of group \(groupSize)"])
    }
    let srcRowBytes = n / 2                       // 4-bit: N/2 bytes per row
    let dstRowBytes = Quantization.QuantizedAffineRow.packedRowBytes(elements: n, bits: bits)
    let groups = n / groupSize

    // layout.json sub-tensor offsets are relative to the expert blob's start;
    // add the blob's absolute file offset so each expert reads ITS OWN bytes.
    let wStart = Int(region.offset + expertOffset)
    let sStart = Int(scalesRegion.offset + expertOffset)
    let bStart = Int(biasesRegion.offset + expertOffset)
    let wBase = [UInt8](blob[wStart ..< wStart + Int(region.size)])
    let sBase = [UInt8](blob[sStart ..< sStart + Int(scalesRegion.size)])
    let bBase = [UInt8](blob[bStart ..< bStart + Int(biasesRegion.size)])

    var outBytes = [UInt8](repeating: 0, count: rows * dstRowBytes)
    var scalesOut = [UInt8](repeating: 0, count: rows * groups * 2)
    var biasesOut = [UInt8](repeating: 0, count: rows * groups * 2)
    var row4 = [Float](repeating: 0, count: n)

    for r in 0..<rows {
        // dequant 4-bit row (BF16 scale/bias, low nibble = even index)
        for g in 0..<groups {
            let so = (r * groups + g) * 2
            let s = Float(bitPattern: UInt32(UInt16(sBase[so]) | (UInt16(sBase[so + 1]) << 8)) << 16)
            let bo = (r * groups + g) * 2
            let b = Float(bitPattern: UInt32(UInt16(bBase[bo]) | (UInt16(bBase[bo + 1]) << 8)) << 16)
            for k in 0..<groupSize {
                let byte = wBase[r * srcRowBytes + g * (groupSize / 2) + (k / 2)]
                let code = (k & 1) == 0 ? Int(byte & 0x0F) : Int(byte >> 4)
                row4[g * groupSize + k] = Float(code) * s + b
            }
        }
        // requant at target bits (same rounding discipline as the runtime)
        let q = Quantization.quantizeAffine(row4, bits: bits, groupSize: groupSize)
        let rowN = Quantization.dequantizeAffine(q, count: n, groupSize: groupSize)
        for k in 0..<dstRowBytes {
            outBytes[r * dstRowBytes + k] = q.packed[k]
        }
        // New scales/biases for THIS row (groups x BF16), little-endian, matching
        // the 4-bit region layout (2 bytes per group).
        for g in 0..<groups {
            let sb = q.scales[g]   // BF16 bit pattern
            let bb = q.biases[g]
            let so = (r * groups + g) * 2
            scalesOut[so] = UInt8(sb & 0xFF)
            scalesOut[so + 1] = UInt8(sb >> 8)
            biasesOut[so] = UInt8(bb & 0xFF)
            biasesOut[so + 1] = UInt8(bb >> 8)
        }
        stats.add(row4: row4, rowN: rowN)
    }
    return (w: Data(outBytes), s: Data(scalesOut), b: Data(biasesOut))
}

// ---- main ----

private func run() throws {
    let args = try Args.parse(CommandLine.arguments)
    let inputDir = args.input!
    let outputDir = args.output!
    let bits = args.routedBits

    let fm = FileManager.default
    try fm.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
    let packedOut = (outputDir as NSString).appendingPathComponent("packed_experts")
    try fm.createDirectory(atPath: packedOut, withIntermediateDirectories: true)

    let manifestData = try readFile((inputDir as NSString).appendingPathComponent("manifest.json"))
    let layoutData = try readFile((inputDir as NSString).appendingPathComponent("packed_experts/layout.json"))
    let manifestRoot = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any] ?? [:]
    let manifestFiles = manifestRoot["files"] as? [String: [String: Any]] ?? [:]
    let oldStride = (manifestRoot["expertStride"] as? NSNumber)?.uint64Value ?? 0
    let (numLayers, expertsPerLayer, layers) = try parseLayout(layoutData)

    var newFiles = manifestFiles
    var newLayersJSON: [[String: Any]] = []
    var totalIn = UInt64(0)
    var totalOut = UInt64(0)
    var gateStats = QualityAccumulator()
    var upStats = QualityAccumulator()
    var downStats = QualityAccumulator()
    var expertStrideOut: UInt64 = 0

    for layerInfo in layers {
        let layerPath = (inputDir as NSString).appendingPathComponent("packed_experts/\(layerInfo.file)")
        let blob = try readFile(layerPath)
        let oldLayerSize = blob.count
        totalIn += UInt64(oldLayerSize)

        var expertJSONs: [[String: Any]] = []
        var layerOut = Data()
        layerOut.reserveCapacity(layerInfo.experts.count * 1_500_000)

        for exp in layerInfo.experts.sorted(by: { $0.physicalRank < $1.physicalRank }) {
            let roles = ["gate", "up", "down"]
            var rebuilt = Data()
            var cursor = 0
            var tensorJSON: [String: [String: Any]] = [:]

            for role in roles {
                guard let w = exp.tensors[role],
                      let s = exp.tensors["\(role)_scales"],
                      let b = exp.tensors["\(role)_biases"] else {
                    throw NSError(domain: "rebits", code: 5,
                                  userInfo: [NSLocalizedDescriptionKey: "missing regions for \(role)"])
                }
                var roleStats: QualityAccumulator
                switch role {
                case "gate": roleStats = gateStats
                case "up": roleStats = upStats
                default: roleStats = downStats
                }
                let newRegions = try requantRegion(blob: blob, region: w, scalesRegion: s,
                                                    biasesRegion: b, expertOffset: exp.offset,
                                                    bits: bits,
                                                    groupSize: args.groupSize,
                                                    stats: &roleStats)
                switch role {
                case "gate": gateStats = roleStats
                case "up": upStats = roleStats
                default: downStats = roleStats
                }
                appendRegion(out: &rebuilt, cursor: &cursor,
                             tensorJSON: &tensorJSON, role: role, w: w, s: s, b: b,
                             newW: newRegions.w, newS: newRegions.s, newB: newRegions.b,
                             bits: bits)
            }

            // pad to page alignment so expertStride % pageSize == 0 (runtime requirement:
            // getpagesize() == 16384 on Apple Silicon, 4096 elsewhere)
            let pageSize = Int(getpagesize())
            var padded = rebuilt.count
            if padded % pageSize != 0 { padded += pageSize - (padded % pageSize) }
            while rebuilt.count < padded { rebuilt.append(0) }
            layerOut.append(rebuilt)

            expertJSONs.append([
                "expert": exp.expert,
                "physicalRank": exp.physicalRank,
                "offset": UInt64(exp.physicalRank * padded),
                "size": UInt64(padded),
                "tensors": tensorJSON,
            ])
            if expertStrideOut == 0 { expertStrideOut = UInt64(padded) }
        }

        let layerSize = layerOut.count
        totalOut += UInt64(layerSize)
        let outPath = (outputDir as NSString).appendingPathComponent("packed_experts/\(layerInfo.file)")
        try layerOut.write(to: URL(fileURLWithPath: outPath))
        newFiles["packed_experts/\(layerInfo.file)"] = ["sha256": sha256(layerOut), "size": layerSize]
        newLayersJSON.append([
            "layer": layerInfo.layer,
            "file": layerInfo.file,
            "experts": expertJSONs,
        ])
        print("[layer \(layerInfo.layer)] \(oldLayerSize) -> \(layerSize) bytes "
              + "(\(String(format: "%.1f", Double(layerSize) / Double(oldLayerSize) * 100))%)")
    }

    // write layout.json
    let newLayout: [String: Any] = [
        "expertsPerLayer": expertsPerLayer,
        "numLayers": numLayers,
        "expertStride": expertStrideOut,
        "layers": newLayersJSON,
    ]
    let layoutJSONData = try JSONSerialization.data(withJSONObject: newLayout,
                                                    options: [.prettyPrinted, .sortedKeys])
    try layoutJSONData.write(to: URL(fileURLWithPath:
        (outputDir as NSString).appendingPathComponent("packed_experts/layout.json")))
    newFiles["packed_experts/layout.json"] = [
        "sha256": sha256(layoutJSONData), "size": layoutJSONData.count,
    ]

    // copy tokenizer + model_weights.bin unchanged
    var modelWeightsDigest = ""
    for name in ["model_weights.bin", "tokenizer"] {
        let src = (inputDir as NSString).appendingPathComponent(name)
        let dst = (outputDir as NSString).appendingPathComponent(name)
        if fm.fileExists(atPath: src) {
            if fm.fileExists(atPath: dst) { try fm.removeItem(atPath: dst) }
            try fm.copyItem(atPath: src, toPath: dst)
            if name == "model_weights.bin" {
                modelWeightsDigest = sha256(try readFile(dst))
            }
        }
    }

    // update manifest
    var newManifest = manifestRoot
    newManifest["expertStride"] = expertStrideOut
    if var quant = newManifest["quant"] as? [String: Any],
       var routed = quant["routedExpert"] as? [String: Any] {
        routed["weightBits"] = bits
        quant["routedExpert"] = routed
        newManifest["quant"] = quant
    }
    newManifest["files"] = newFiles
    let newManifestData = try JSONSerialization.data(withJSONObject: newManifest,
                                                     options: [.prettyPrinted, .sortedKeys])
    try newManifestData.write(to: URL(fileURLWithPath:
        (outputDir as NSString).appendingPathComponent("manifest.json")))
    let manifestDigest = sha256(newManifestData)

    // write verified-install.json bound to the new directory
    var receiptFiles: [String: [String: Any]] = [:]
    for (path, entry) in newFiles {
        receiptFiles[path] = ["sha256": entry["sha256"] ?? "", "size": entry["size"] ?? 0]
    }
    receiptFiles["manifest.json"] = ["sha256": manifestDigest, "size": newManifestData.count]
    if let mw = newFiles["model_weights.bin"] {
        receiptFiles["model_weights.bin"] = mw
    } else if !modelWeightsDigest.isEmpty, let size = try? fm.attributesOfItem(
        atPath: (outputDir as NSString).appendingPathComponent("model_weights.bin"))[.size] as? Int {
        receiptFiles["model_weights.bin"] = ["sha256": modelWeightsDigest, "size": size]
    }
    let receipt: [String: Any] = [
        "schemaVersion": 1,
        "modelDirectoryPath": URL(fileURLWithPath: outputDir).standardizedFileURL.path,
        "sourceRepoID": "rebits",
        "sourceRevision": "routed-bits=\(bits) (source stride \(oldStride))",
        "toolVersion": "TurboFieldfareRebits (routed-expert \(bits)-bit affine, group \(args.groupSize))",
        "verificationTimestamp": ISO8601DateFormatter().string(from: Date()),
        "manifestSha256": manifestDigest,
        "files": receiptFiles,
    ]
    let receiptData = try JSONSerialization.data(withJSONObject: receipt,
                                                 options: [.prettyPrinted, .sortedKeys])
    try receiptData.write(to: URL(fileURLWithPath:
        (outputDir as NSString).appendingPathComponent("verified-install.json")))

    print()
    print("=== TurboFieldfareRebits summary ===")
    print("routed experts: 4-bit -> \(bits)-bit (affine, group \(args.groupSize))")
    print("packed_experts: \(totalIn) -> \(totalOut) bytes "
          + "(-\(String(format: "%.1f", (1 - Double(totalOut) / Double(totalIn)) * 100))%)")
    print("expertStride:   \(oldStride) -> \(expertStrideOut)")
    print("quality (4-bit dequant vs \(bits)-bit dequant, weight-space):")
    print("  gate RMSE=\(String(format: "%.6f", gateStats.rmse)) maxAbs=\(String(format: "%.6f", gateStats.maxAbsErr)) (n=\(gateStats.n))")
    print("  up   RMSE=\(String(format: "%.6f", upStats.rmse)) maxAbs=\(String(format: "%.6f", upStats.maxAbsErr)) (n=\(upStats.n))")
    print("  down RMSE=\(String(format: "%.6f", downStats.rmse)) maxAbs=\(String(format: "%.6f", downStats.maxAbsErr)) (n=\(downStats.n))")
    print("output: \(outputDir)")
}

/// Append a rebuilt W region + unchanged scales/biases to the new expert blob.
private func appendRegion(out: inout Data,
                          cursor: inout Int,
                          tensorJSON: inout [String: [String: Any]],
                          role: String,
                          w: TensorRegion,
                          s: TensorRegion,
                          b: TensorRegion,
                          newW: Data,
                          newS: Data,
                          newB: Data,
                          bits: Int) {
    let wOff = cursor
    out.append(newW)
    let sOff = wOff + newW.count
    out.append(newS)
    let bOff = sOff + newS.count
    out.append(newB)
    cursor = bOff + newB.count

    tensorJSON[role] = [
        "offset": wOff, "size": newW.count,
        "dtype": w.dtype, "shape": w.shape, "bits": bits,
    ]
    tensorJSON["\(role)_scales"] = [
        "offset": sOff, "size": newS.count, "dtype": s.dtype, "shape": s.shape,
    ]
    tensorJSON["\(role)_biases"] = [
        "offset": bOff, "size": newB.count, "dtype": b.dtype, "shape": b.shape,
    ]
}

do {
    try run()
} catch {
    fputs("TurboFieldfareRebits error: \(error)\n", stderr)
    exit(1)
}
