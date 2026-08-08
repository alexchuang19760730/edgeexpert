import Foundation

struct LocalSnapshot {
    let metadata: IndexLoader.SourceMetadata
    let arch: ArchInfo
    let shardHeaders: [Safetensors.Header]
    let modelDirectory: String
}

enum LocalSnapshotLoader {
    static func load(modelDirectory: String,
                     requireKnownSource: Bool) throws -> LocalSnapshot {
        let root = URL(fileURLWithPath: modelDirectory).standardizedFileURL.path
        let metadata = try IndexLoader.load(snapshotDir: root)
        if requireKnownSource && SourceFingerprint.modelID(forIndexSha256: metadata.indexSha256Hex) == nil {
            throw RepackError.sourceFingerprintRejected(path: metadata.indexPath,
                                                        sha256: metadata.indexSha256Hex)
        }
        let arch = try ArchInfo.load(configPath: metadata.configPath)
        let headers = try metadata.shardFilenames.map { shard in
            try loadShardHeader(root: root, shard: shard)
        }
        return LocalSnapshot(metadata: metadata,
                             arch: arch,
                             shardHeaders: headers,
                             modelDirectory: root)
    }

    private static func loadShardHeader(root: String, shard: String) throws -> Safetensors.Header {
        let path = URL(fileURLWithPath: root)
            .appendingPathComponent(shard)
            .standardizedFileURL
            .path
        let fd = try Posix.openReadNoFollow(path)
        defer { close(fd) }
        let fileSize = try Posix.fileSize(fd: fd, path: path)

        var prefix = Data(count: 8)
        try prefix.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            try Posix.preadAll(fd: fd, path: path, buf: base, count: 8, offset: 0)
        }
        let headerSize = prefix.withUnsafeBytes { raw -> UInt64 in
            var value: UInt64 = 0
            for i in 0..<8 {
                value |= UInt64(raw[i]) << UInt64(i * 8)
            }
            return value
        }
        if headerSize > Safetensors.maxHeaderBytes || headerSize > fileSize - 8 {
            throw RepackError.safetensorsHeaderTooLarge(path: path, size: headerSize)
        }

        let headerCount = Int(headerSize)
        var headerData = Data(count: headerCount)
        try headerData.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress, headerCount > 0 else { return }
            try Posix.preadAll(fd: fd,
                               path: path,
                               buf: base,
                               count: headerCount,
                               offset: 8)
        }
        return try Safetensors.parseHeaderBytes(path: path,
                                                fileSize: fileSize,
                                                headerBytes: headerData)
    }
}
