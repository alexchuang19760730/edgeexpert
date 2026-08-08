import CryptoKit
import Foundation
import TurboFieldfare

actor ServerPrefillExpertTraceWriter {
    private let handle: FileHandle

    init(path: String) throws {
        let fileURL = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true,
            attributes: nil)
        if !FileManager.default.fileExists(atPath: fileURL.path) {
            FileManager.default.createFile(atPath: fileURL.path, contents: nil)
        }
        self.handle = try FileHandle(forWritingTo: fileURL)
        try self.handle.seekToEnd()
    }

    deinit {
        try? handle.close()
    }

    func append(modelID: String,
                sourceSnapshotHash: String?,
                runtimeProfileHash: String,
                request: ValidatedChatRequest,
                renderedPromptIDs: [Int32],
                effectivePromptIDs: [Int32],
                completion: ServerCompletion,
                trace: PrefillExpertTrace) throws {
        let record: [String: Any] = [
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "model_id": modelID,
            "source_snapshot_hash": Self.json(sourceSnapshotHash),
            "runtime_profile_hash": runtimeProfileHash,
            "request": [
                "message_count": request.messages.count,
                "tool_count": request.tools.count,
                "stream": request.stream,
                "maximum_completion_tokens": request.maximumCompletionTokens,
                "rendered_prompt_token_count": renderedPromptIDs.count,
                "effective_prompt_token_count": effectivePromptIDs.count,
                "cached_prompt_tokens": completion.usage.promptTokensDetails.cachedTokens,
                "rendered_prompt_sha256": Self.sha256Hex(renderedPromptIDs),
                "effective_prompt_sha256": Self.sha256Hex(effectivePromptIDs),
            ],
            "timing": [
                "prefill_ms": completion.prefillMs,
                "decode_ms": completion.decodeMs,
                "prefill_tile_plan_ms": Self.json(Self.milliseconds(
                    completion.prefillTimingBreakdown?.tilePlanNanos)),
                "prefill_tile_fetch_open_or_read_ms": Self.json(Self.milliseconds(
                    completion.prefillTimingBreakdown?.tileFetchOpenOrReadNanos)),
                "prefill_tile_fetch_read_wall_ms": Self.json(Self.milliseconds(
                    completion.prefillTimingBreakdown?.tileFetchReadWallNanos)),
            ],
            "access_pattern": [
                "tile_accesses": Self.json(completion.prefillExpertAccessPattern?.tileAccesses),
                "expert_references": Self.json(completion.prefillExpertAccessPattern?.expertReferences),
                "unique_experts": Self.json(completion.prefillExpertAccessPattern?.uniqueExperts),
                "consecutive_tile_overlap_ratio": Self.json(Self.ratio(
                    completion.prefillExpertAccessPattern?.consecutiveTileOverlapExperts,
                    completion.prefillExpertAccessPattern?.consecutiveTileRequestedExperts)),
                "contiguous_neighbor_ratio": Self.json(Self.ratio(
                    completion.prefillExpertAccessPattern?.contiguousNeighborPairs,
                    completion.prefillExpertAccessPattern?.contiguousNeighborOpportunities)),
                "max_contiguous_run_length": Self.json(completion.prefillExpertAccessPattern?.maxContiguousRunLength),
            ],
            "layers": trace.layers.map(Self.encodeLayer(_:)),
        ]
        let data = try JSONSerialization.data(
            withJSONObject: record,
            options: [.sortedKeys, .withoutEscapingSlashes])
        try handle.seekToEnd()
        try handle.write(contentsOf: data + Data([0x0a]))
    }

    private static func encodeLayer(_ layer: PrefillExpertLayerTrace) -> [String: Any] {
        [
            "layer": layer.layer,
            "tile_accesses": layer.tileAccesses,
            "unique_experts": layer.uniqueExperts,
            "expert_counts": layer.expertCounts.map {
                ["expert": $0.expert, "count": $0.count]
            },
            "coaccess_pairs": layer.coAccessPairs.map {
                ["first": $0.first, "second": $0.second, "count": $0.count]
            },
        ]
    }

    private static func ratio(_ numerator: UInt64?, _ denominator: UInt64?) -> Double? {
        guard let numerator, let denominator, denominator > 0 else { return nil }
        return Double(numerator) / Double(denominator)
    }

    private static func milliseconds(_ nanos: UInt64?) -> Double? {
        guard let nanos else { return nil }
        return Double(nanos) / 1_000_000.0
    }

    private static func json(_ value: Any?) -> Any {
        value ?? NSNull()
    }

    private static func sha256Hex(_ values: [Int32]) -> String {
        let data: Data
        if values.isEmpty {
            data = Data()
        } else {
            data = values.withUnsafeBufferPointer { buffer in
                Data(bytes: buffer.baseAddress!,
                     count: buffer.count * MemoryLayout<Int32>.size)
            }
        }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
