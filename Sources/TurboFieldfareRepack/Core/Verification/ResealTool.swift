import Foundation
import Darwin

/// Re-issues `verified-install.json` for an existing `.gturbo` directory.
///
/// A receipt is bound to the absolute path it was signed for, so copying or
/// moving a verified model invalidates it. Without a way to re-seal, the
/// runtime falls back to re-hashing every expert shard on each launch
/// (12+ GiB for Gemma 4 26B), which dominates time-to-first-token.
///
/// Two modes are offered, and they carry different trust semantics:
///
/// - ``ResealMode/full`` re-reads and re-hashes every file declared in
///   `manifest.json`. This is the same work `--verify-install` performs and is
///   the only mode that can detect silent data corruption.
/// - ``ResealMode/rebind`` trusts the digests already recorded in the previous
///   receipt and only confirms that every file is still present at its
///   recorded byte length. It is intended for the narrow case where a
///   *previously verified* directory was relocated as a whole. It is orders of
///   magnitude faster but cannot detect content changes that preserve size.
///
/// The emitted receipt records which mode produced it in `toolVersion`, so a
/// rebound receipt never masquerades as a freshly hashed one.
public enum ResealMode: String, Sendable {
    case full
    case rebind
}

public struct ResealOptions: Sendable {
    public let inputGTurbo: String
    public let mode: ResealMode

    public init(inputGTurbo: String, mode: ResealMode) {
        self.inputGTurbo = inputGTurbo
        self.mode = mode
    }
}

public struct ResealResult: Sendable {
    public let receiptPath: String
    public let fileCount: Int
    public let bytesVerified: UInt64
    public let unexpectedEntries: [String]
    public let mode: ResealMode
    public let previousPath: String?
    public let elapsedSeconds: Double
}

public enum ResealTool {
    public static func run(options: ResealOptions) throws -> ResealResult {
        let started = Date()
        let root = URL(fileURLWithPath: options.inputGTurbo).standardizedFileURL
        let previousPath = try? existingReceiptPath(root: root)

        switch options.mode {
        case .full:
            let verified = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: root.path))
            return ResealResult(receiptPath: verified.receiptPath,
                                fileCount: verified.fileCount,
                                bytesVerified: verified.bytesVerified,
                                unexpectedEntries: verified.unexpectedEntries,
                                mode: .full,
                                previousPath: previousPath,
                                elapsedSeconds: Date().timeIntervalSince(started))
        case .rebind:
            return try rebind(root: root,
                              previousPath: previousPath,
                              started: started)
        }
    }

    // MARK: - Rebind

    private static func rebind(root: URL,
                               previousPath: String?,
                               started: Date) throws -> ResealResult {
        let receiptURL = root.appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
        guard FileManager.default.fileExists(atPath: receiptURL.path) else {
            throw RepackError.configurationInvalid(
                detail: """
                    no existing \(VerifiedInstallReceiptWriter.fileName) to rebind; \
                    run --reseal without --rebind to hash the model from scratch
                    """)
        }

        let previous = try loadReceipt(path: receiptURL.path)

        // The layout invariants are cheap to re-check and catch a repack that
        // was interrupted or produced with an incompatible tool version.
        try VerifiedInstallTool.validatePackedExpertLayout(inputGTurbo: root.path)

        var files: [RepackAudit.OutputFile] = []
        files.reserveCapacity(previous.files.count)
        var bytesVerified: UInt64 = 0
        var manifestSha: String?
        var manifestSize: UInt64?

        for relativePath in previous.files.keys.sorted() {
            guard let entry = previous.files[relativePath] else { continue }
            let path = root.appendingPathComponent(relativePath).path
            let actualSize = try fileSize(path: path, relativePath: relativePath)
            guard actualSize == entry.size else {
                throw RepackError.configurationInvalid(
                    detail: """
                        \(relativePath) size \(actualSize) != receipt \(entry.size); \
                        contents changed — run --reseal without --rebind
                        """)
            }
            bytesVerified &+= actualSize
            if relativePath == "manifest.json" {
                // Re-hash the manifest: it is tiny, and it is the root of trust
                // that pins every other file's expected digest.
                let actualSha = try Sha256Stream.hashFile(path: path, noCache: true)
                guard actualSha.lowercased() == entry.sha256.lowercased() else {
                    throw RepackError.configurationInvalid(
                        detail: "manifest.json SHA mismatch — run --reseal without --rebind")
                }
                manifestSha = actualSha
                manifestSize = actualSize
            } else {
                files.append(RepackAudit.OutputFile(relativePath: relativePath,
                                                    size: actualSize,
                                                    sha256: entry.sha256))
            }
        }

        guard let manifestSha, let manifestSize else {
            throw RepackError.configurationInvalid(
                detail: "previous receipt does not cover manifest.json")
        }

        // Cross-check the receipt against the manifest so a rebind cannot
        // quietly bless a directory whose file set has drifted.
        let manifest = try loadManifestFileSet(
            path: root.appendingPathComponent("manifest.json").path)
        let receiptSet = Set(files.map(\.relativePath))
        let missing = manifest.subtracting(receiptSet).sorted()
        guard missing.isEmpty else {
            throw RepackError.configurationInvalid(
                detail: """
                    receipt is missing \(missing.count) file(s) declared in manifest \
                    (e.g. \(missing.prefix(3).joined(separator: ", "))) — \
                    run --reseal without --rebind
                    """)
        }

        let receiptData = try VerifiedInstallReceiptWriter.encode(
            outputDir: root.path,
            manifestSha256: manifestSha,
            manifestSize: manifestSize,
            sourceRepoID: previous.sourceRepoID,
            sourceRevision: previous.sourceRevision,
            toolVersion: "TurboFieldfareRepack reseal --rebind (size-only, digests inherited)",
            files: files)
        try receiptData.write(to: receiptURL, options: .atomic)

        return ResealResult(receiptPath: receiptURL.path,
                            fileCount: files.count + 1,
                            bytesVerified: bytesVerified,
                            unexpectedEntries: [],
                            mode: .rebind,
                            previousPath: previousPath,
                            elapsedSeconds: Date().timeIntervalSince(started))
    }

    // MARK: - Receipt loading

    private struct PreviousReceipt {
        let modelDirectoryPath: String?
        let sourceRepoID: String?
        let sourceRevision: String?
        let files: [String: FileEntry]
    }

    private struct FileEntry {
        let size: UInt64
        let sha256: String
    }

    private static func existingReceiptPath(root: URL) throws -> String? {
        let path = root.appendingPathComponent(VerifiedInstallReceiptWriter.fileName).path
        guard FileManager.default.fileExists(atPath: path) else { return nil }
        return try loadReceipt(path: path).modelDirectoryPath
    }

    private static func loadReceipt(path: String) throws -> PreviousReceipt {
        let size = try fileSize(path: path, relativePath: VerifiedInstallReceiptWriter.fileName)
        guard size <= VerifiedInstallTool.metadataMaxBytes else {
            throw RepackError.configurationInvalid(
                detail: "\(VerifiedInstallReceiptWriter.fileName) exceeds metadata cap")
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configurationInvalid(
                detail: "\(VerifiedInstallReceiptWriter.fileName) is not a JSON object")
        }
        guard let rawFiles = object["files"] as? [String: Any] else {
            throw RepackError.configurationInvalid(
                detail: "\(VerifiedInstallReceiptWriter.fileName) has no files map")
        }
        var files: [String: FileEntry] = [:]
        files.reserveCapacity(rawFiles.count)
        for (relativePath, value) in rawFiles {
            guard let entry = value as? [String: Any],
                  let sha = entry["sha256"] as? String,
                  let sizeNumber = entry["size"] as? NSNumber else {
                throw RepackError.configurationInvalid(
                    detail: "receipt entry for \(relativePath) is malformed")
            }
            files[relativePath] = FileEntry(size: sizeNumber.uint64Value, sha256: sha)
        }
        return PreviousReceipt(modelDirectoryPath: object["modelDirectoryPath"] as? String,
                               sourceRepoID: object["sourceRepoID"] as? String,
                               sourceRevision: object["sourceRevision"] as? String,
                               files: files)
    }

    private static func loadManifestFileSet(path: String) throws -> Set<String> {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let files = object["files"] as? [String: Any] else {
            throw RepackError.configurationInvalid(detail: "manifest.json invalid")
        }
        return Set(files.keys)
    }

    private static func fileSize(path: String, relativePath: String) throws -> UInt64 {
        var st = stat()
        guard stat(path, &st) == 0 else {
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        guard st.st_size >= 0 else {
            throw RepackError.configurationInvalid(
                detail: "\(relativePath) has negative file size")
        }
        return UInt64(st.st_size)
    }
}
