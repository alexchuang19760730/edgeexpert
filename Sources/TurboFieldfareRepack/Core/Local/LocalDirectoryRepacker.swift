import Darwin
import Foundation

public struct LocalDirectoryRepackOptions: Sendable {
    public let inputDir: String
    public let outputDir: String
    public let expertLayoutOrder: ExpertLayoutOrder?
    public let requireKnownSource: Bool
    public let copyAuditPath: String?
    public let rangeChunkBytes: Int
    public let writeTileBytes: Int
    public let minFreeReserveBytes: UInt64
    public let overwrite: Bool
    public let dryRunSpaceCheck: Bool

    public init(inputDir: String,
                outputDir: String,
                expertLayoutOrder: ExpertLayoutOrder? = nil,
                requireKnownSource: Bool = false,
                copyAuditPath: String? = nil,
                rangeChunkBytes: Int = RemoteChunkPolicy.defaultBytes,
                writeTileBytes: Int = WriterCore.tileBytes,
                minFreeReserveBytes: UInt64 = 1 * 1024 * 1024 * 1024,
                overwrite: Bool = false,
                dryRunSpaceCheck: Bool = false) {
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.expertLayoutOrder = expertLayoutOrder
        self.requireKnownSource = requireKnownSource
        self.copyAuditPath = copyAuditPath
        self.rangeChunkBytes = rangeChunkBytes
        self.writeTileBytes = writeTileBytes
        self.minFreeReserveBytes = minFreeReserveBytes
        self.overwrite = overwrite
        self.dryRunSpaceCheck = dryRunSpaceCheck
    }
}

public struct LocalDirectoryRepackResult: Sendable {
    public let outputDir: String
    public let sourceIndexSHA256: String
    public let modelID: String
    public let outputBytes: UInt64
    public let copiedBytes: UInt64
    public let dryRun: Bool
}

final class LocalFileSourceByteProvider {
    private let writeTileBytes: Int

    init(writeTileBytes: Int = WriterCore.tileBytes) {
        self.writeTileBytes = writeTileBytes
    }

    func copyBatch(_ copies: [RangeCopy],
                   audit: RepackAudit,
                   progress: @escaping @Sendable (UInt64) -> Void) throws {
        var shardMaps: [String: MmapHandle] = [:]
        var outputFDs: [String: Int32] = [:]
        defer { outputFDs.values.forEach { close($0) } }

        var copied: UInt64 = 0
        for copy in copies where copy.size > 0 {
            let shard = try mappedShard(path: copy.shardID, cache: &shardMaps)
            let destinationFD: Int32
            if let existing = outputFDs[copy.destinationPath] {
                destinationFD = existing
            } else {
                destinationFD = try Posix.openExistingRW(copy.destinationPath)
                outputFDs[copy.destinationPath] = destinationFD
            }
            try copyBytes(from: shard,
                          sourceOffset: copy.sourceOffset,
                          destinationFD: destinationFD,
                          destinationPath: copy.destinationPath,
                          destinationOffset: copy.destinationOffset,
                          size: copy.size,
                          audit: audit)
            copied += copy.size
            progress(copied)
        }

        for (path, fd) in outputFDs {
            try Posix.fsync(fd, path: path)
        }
    }

    private func mappedShard(path: String,
                             cache: inout [String: MmapHandle]) throws -> MmapHandle {
        if let handle = cache[path] {
            return handle
        }
        let handle = try MmapHandle(path: path)
        cache[path] = handle
        return handle
    }

    private func copyBytes(from shard: MmapHandle,
                           sourceOffset: UInt64,
                           destinationFD: Int32,
                           destinationPath: String,
                           destinationOffset: UInt64,
                           size: UInt64,
                           audit: RepackAudit) throws {
        var remaining = Int(size)
        var srcOff = sourceOffset
        var dstOff = destinationOffset
        while remaining > 0 {
            let count = min(remaining, writeTileBytes)
            let base = shard.base.advanced(by: Int(srcOff))
            try Posix.pwriteAll(fd: destinationFD,
                                path: destinationPath,
                                buf: base,
                                count: count,
                                offset: dstOff)
            audit.recordTile(bytes: count)
            audit.recordRead(bytes: count)
            audit.recordWrite(bytes: count)
            shard.adviseDontNeed(offset: srcOff, count: count)
            remaining -= count
            srcOff += UInt64(count)
            dstOff += UInt64(count)
        }
    }
}

public final class LocalDirectoryRepacker {
    private let options: LocalDirectoryRepackOptions
    private let audit: RepackAudit
    private let startTime = Date()

    public init(options: LocalDirectoryRepackOptions,
                audit: RepackAudit = RepackAudit()) {
        self.options = options
        self.audit = audit
    }

    public func run(progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }) async throws
        -> LocalDirectoryRepackResult {
        try validateOptions()
        let installLock = try InstallLock.acquire(outputDirectory: options.outputDir)
        defer { withExtendedLifetime(installLock) {} }
        let paths = installLock.paths
        try resetOutputState(paths: paths)

        progress(.downloadingMetadata)
        let snapshot = try LocalSnapshotLoader.load(modelDirectory: options.inputDir,
                                                    requireKnownSource: options.requireKnownSource)
        let plan = try RepackPlanner.plan(meta: snapshot.metadata,
                                          arch: snapshot.arch,
                                          shardHeaders: snapshot.shardHeaders,
                                          outputDir: paths.partialDirectory,
                                          expertLayoutOrder: options.expertLayoutOrder)
        let rangePlan = try RangeCopyPlanner.plan(repackPlan: plan,
                                                  rangeChunkBytes: options.rangeChunkBytes,
                                                  layoutMode: options.expertLayoutOrder?.strategy ?? "identity",
                                                  layoutOrderSha256: options.expertLayoutOrder?.sha256Hex)
        let outputBytes = plan.resident.totalSize
            + plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize }
        progress(.planning(downloadBytes: rangePlan.scalarCopies.reduce(UInt64(0)) { $0 + $1.size },
                           outputBytes: outputBytes))
        let diskRequirement = try DiskSpaceChecker.requireAvailable(
            path: paths.parentDirectory,
            bytes: outputBytes,
            reserveBytes: options.minFreeReserveBytes)
        progress(.checkingDisk(diskRequirement))

        audit.sourceSnapshotSha256 = snapshot.metadata.indexSha256Hex
        audit.bitWidthOverridesHonored = snapshot.metadata.bitsOverrides.count
        audit.tensorsDroppedMultimodal = plan.excludedMultimodalTensorNames
        audit.packedExpertLayoutMode = options.expertLayoutOrder?.strategy ?? "identity"
        audit.packedExpertLayoutOrderPath = options.expertLayoutOrder?.sourcePath
        audit.packedExpertLayoutOrderSha256 = options.expertLayoutOrder?.sha256Hex
        audit.packedExpertLayoutStrategy = options.expertLayoutOrder?.strategy
        audit.packedExpertReorderedLayerCount = options.expertLayoutOrder?.reorderedLayerCount ?? 0
        audit.remoteRangeStreamingSupported = false

        let modelID = plan.matchedModelID ?? "local/\(URL(fileURLWithPath: snapshot.modelDirectory).lastPathComponent)"
        if options.dryRunSpaceCheck {
            return LocalDirectoryRepackResult(outputDir: options.outputDir,
                                              sourceIndexSHA256: snapshot.metadata.indexSha256Hex,
                                              modelID: modelID,
                                              outputBytes: outputBytes,
                                              copiedBytes: 0,
                                              dryRun: true)
        }

        progress(.reservingOutput(bytes: outputBytes))
        try createOutputFiles(plan: plan, partialDirectory: paths.partialDirectory)

        let provider = LocalFileSourceByteProvider(writeTileBytes: options.writeTileBytes)
        let totalCopyBytes = rangePlan.scalarCopies.reduce(UInt64(0)) { $0 + $1.size }
        progress(.copyingPayload(reusedBytes: 0,
                                 downloadedThisRunBytes: 0,
                                 totalBytes: totalCopyBytes))
        try provider.copyBatch(rangePlan.scalarCopies,
                               audit: audit,
                               progress: { copied in
                                   progress(.copyingPayload(reusedBytes: 0,
                                                            downloadedThisRunBytes: copied,
                                                            totalBytes: totalCopyBytes))
                               })

        try recordOutputFile(relativePath: "model_weights.bin",
                             path: plan.resident.path,
                             progress: progress)
        for layer in plan.layers where layer.expertsPerLayer > 0 {
            try Task.checkCancellation()
            let relativePath = "packed_experts/" + (layer.path as NSString).lastPathComponent
            try recordOutputFile(relativePath: relativePath,
                                 path: layer.path,
                                 progress: progress)
        }

        let layoutPath = ((paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts") as NSString)
            .appendingPathComponent("layout.json")
        let expertStride = plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertStride ?? 0
        let layoutData = try GTurboJSON.encodeLayout(plan: plan, expertStride: expertStride)
        try writeSmall(path: layoutPath, data: layoutData)
        try GTurboLayoutValidator.validate(path: layoutPath, plan: plan)
        audit.packedExpertLayoutAuditLogicalIDCount =
            plan.layers.reduce(0) { $0 + $1.expertsPerLayer }
        audit.packedExpertLayoutOffsetValidationPassed = true
        try recordOutputFile(relativePath: "packed_experts/layout.json",
                             path: layoutPath,
                             progress: progress)

        try copyLocalMetadataSidecars(sourceDir: snapshot.modelDirectory,
                                      partialDir: paths.partialDirectory,
                                      progress: progress)

        progress(.finalizing)
        try writeManifest(plan: plan,
                          partialDir: paths.partialDirectory,
                          metadata: snapshot.metadata,
                          expertStride: expertStride,
                          modelID: modelID,
                          sourceDescriptor: snapshot.modelDirectory)

        if try Posix.entryKind(paths.finalDirectory) == .directory {
            try FileManager.default.removeItem(atPath: paths.finalDirectory)
        }
        try Posix.rename(from: paths.partialDirectory, to: paths.finalDirectory)
        try Posix.fsyncDirectory(paths.parentDirectory)
        if try Posix.entryKind(paths.checkpointFile) != .absent {
            try FileManager.default.removeItem(atPath: paths.checkpointFile)
        }

        audit.wallTimeSeconds = Date().timeIntervalSince(startTime)
        audit.wholeFileHeapBuffers = false
        if let auditPath = options.copyAuditPath {
            let data = try audit.toJSONData(outputDir: options.outputDir)
            try Posix.mkdirP((auditPath as NSString).deletingLastPathComponent)
            try data.write(to: URL(fileURLWithPath: auditPath))
        }

        return LocalDirectoryRepackResult(outputDir: options.outputDir,
                                          sourceIndexSHA256: snapshot.metadata.indexSha256Hex,
                                          modelID: modelID,
                                          outputBytes: outputBytes,
                                          copiedBytes: totalCopyBytes,
                                          dryRun: false)
    }

    private func validateOptions() throws {
        var isDirectory: ObjCBool = false
        let inputExists = FileManager.default.fileExists(atPath: options.inputDir,
                                                         isDirectory: &isDirectory)
        guard inputExists, isDirectory.boolValue else {
            throw RepackError.configurationInvalid(detail: "local input directory is missing: \(options.inputDir)")
        }
        guard options.rangeChunkBytes > 0,
              options.rangeChunkBytes <= RemoteChunkPolicy.maxBytes else {
            throw RepackError.configurationInvalid(detail: "bad range chunk bytes \(options.rangeChunkBytes)")
        }
        guard options.writeTileBytes > 0,
              options.writeTileBytes <= BoundedScratch.defaultLimitBytes else {
            throw RepackError.configurationInvalid(detail: "bad write tile bytes \(options.writeTileBytes)")
        }
    }

    private func resetOutputState(paths: RemoteInstallPaths) throws {
        let finalKind = try Posix.entryKind(paths.finalDirectory)
        let partialKind = try Posix.entryKind(paths.partialDirectory)
        let checkpointKind = try Posix.entryKind(paths.checkpointFile)

        if (finalKind != .absent || partialKind != .absent || checkpointKind != .absent) && !options.overwrite {
            throw RepackError.configurationInvalid(
                detail: "output directory already exists: \(paths.finalDirectory); rerun with --overwrite")
        }

        for path in [paths.finalDirectory, paths.partialDirectory, paths.checkpointFile] {
            if try Posix.entryKind(path) != .absent {
                try FileManager.default.removeItem(atPath: path)
            }
        }
    }

    private func createOutputFiles(plan: RepackPlan,
                                   partialDirectory: String) throws {
        try Posix.mkdirP((partialDirectory as NSString).appendingPathComponent("packed_experts"))
        let residentFD = try ResidentWriter.createAndWriteIndex(plan: plan.resident, audit: audit)
        try Posix.fsync(residentFD, path: plan.resident.path)
        close(residentFD)
        for layer in plan.layers where layer.expertsPerLayer > 0 {
            try Task.checkCancellation()
            let descriptor = try Posix.openCreateRW(layer.path)
            try Posix.ftruncate(descriptor, path: layer.path, size: layer.fileSize)
            try Posix.fsync(descriptor, path: layer.path)
            close(descriptor)
        }
        try Posix.fsyncDirectory(partialDirectory)
    }

    private func recordOutputFile(relativePath: String,
                                  path: String,
                                  progress: @Sendable (ModelInstallProgress) -> Void) throws {
        progress(.hashingOutput(relativePath))
        try Task.checkCancellation()
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        let size = try Posix.fileSize(fd: fd, path: path)
        let sha = try WriterCore.hashEntireFile(path: path,
                                                size: size,
                                                audit: audit,
                                                cancellationCheck: Task.checkCancellation)
        audit.outputFiles.append(.init(relativePath: relativePath, size: size, sha256: sha))
    }

    private func writeSmall(path: String, data: Data) throws {
        let directory = (path as NSString).deletingLastPathComponent
        try Posix.mkdirP(directory)
        try Posix.atomicWrite(data, to: path, durableIn: directory)
        audit.recordWrite(bytes: data.count)
    }

    private func copyLocalMetadataSidecars(sourceDir: String,
                                           partialDir: String,
                                           progress: @Sendable (ModelInstallProgress) -> Void) throws {
        let tokenizerDir = (partialDir as NSString).appendingPathComponent("tokenizer")
        try Posix.mkdirP(tokenizerDir)
        let sourceRoots = [
            (sourceDir as NSString).appendingPathComponent("tokenizer"),
            sourceDir,
        ]
        let files: [(name: String, required: Bool)] = [
            ("config.json", true),
            ("tokenizer.json", true),
            ("tokenizer_config.json", true),
            ("special_tokens_map.json", false),
            ("chat_template.jinja", true),
            ("chat_template.json", false),
        ]
        for file in files {
            try Task.checkCancellation()
            guard let sourcePath = firstExistingPath(named: file.name, roots: sourceRoots) else {
                if file.required {
                    throw RepackError.configurationInvalid(detail:
                        "local source is missing required tokenizer sidecar \(file.name)")
                }
                continue
            }
            let destinationPath = (tokenizerDir as NSString).appendingPathComponent(file.name)
            if FileManager.default.fileExists(atPath: destinationPath) {
                try FileManager.default.removeItem(atPath: destinationPath)
            }
            let data = try Data(contentsOf: URL(fileURLWithPath: sourcePath))
            try writeSmall(path: destinationPath, data: data)
            try recordOutputFile(relativePath: "tokenizer/\(file.name)",
                                 path: destinationPath,
                                 progress: progress)
        }
    }

    private func firstExistingPath(named filename: String,
                                   roots: [String]) -> String? {
        for root in roots {
            let candidate = (root as NSString).appendingPathComponent(filename)
            if FileManager.default.fileExists(atPath: candidate) {
                return candidate
            }
        }
        return nil
    }

    private func writeManifest(plan: RepackPlan,
                               partialDir: String,
                               metadata: IndexLoader.SourceMetadata,
                               expertStride: UInt64,
                               modelID: String,
                               sourceDescriptor: String) throws {
        var bits = GTurboJSON.QuantBitWidths(
            embedding: 4,
            attention: 4,
            router: 8,
            sharedExpert: 8,
            routedExpert: 4)
        for entry in plan.resident.entries {
            if entry.name == "language_model.model.embed_tokens.weight", let spec = entry.quantSpec {
                bits.embedding = spec.bits
            }
            if entry.name.hasSuffix(".self_attn.q_proj.weight"), let spec = entry.quantSpec {
                bits.attention = spec.bits
            }
            if entry.name.hasSuffix(".router.proj.weight"), let spec = entry.quantSpec {
                bits.router = spec.bits
            }
            if entry.name.hasSuffix(".mlp.gate_proj.weight"), let spec = entry.quantSpec {
                bits.sharedExpert = spec.bits
            }
        }
        if let layer = plan.layers.first(where: { !$0.subTensors.isEmpty }),
           let routedBits = layer.subTensors.first?.bitsForWeights {
            bits.routedExpert = routedBits
        }
        let files = audit.outputFiles.map {
            ($0.relativePath, GTurboJSON.FileEntry(size: $0.size, sha256: $0.sha256))
        }
        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: modelID,
            sourceSnapshotHash: "sha256:" + metadata.indexSha256Hex,
            files: files,
            expertsPerLayer: plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertsPerLayer ?? 0,
            numLayers: plan.arch.numLayers,
            expertStride: expertStride,
            bitWidths: bits)
        let manifestTmp = (partialDir as NSString).appendingPathComponent("manifest.json.tmp")
        let manifestPath = (partialDir as NSString).appendingPathComponent("manifest.json")
        try writeSmall(path: manifestTmp, data: data)
        try Posix.rename(from: manifestTmp, to: manifestPath)
        let manifestSha = try Sha256Stream.hashFile(path: manifestPath)
        let receipt = try VerifiedInstallReceiptWriter.encode(
            outputDir: options.outputDir,
            manifestSha256: manifestSha,
            manifestSize: UInt64(data.count),
            sourceRepoID: "local-hf-snapshot",
            sourceRevision: sourceDescriptor,
            files: audit.outputFiles)
        let receiptPath = (partialDir as NSString)
            .appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
        try writeSmall(path: receiptPath, data: receipt)
    }
}
