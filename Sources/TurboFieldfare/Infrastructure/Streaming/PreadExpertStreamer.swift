import Darwin
import Foundation
import Metal

public struct ExpertIOAdviceResult: Sendable, Equatable {
    public let requested: Int
    public let failed: Int
    public let calls: Int
    public let bytes: UInt64
    public let skipped: Int
    public let maxCallNanos: UInt64

    public init(requested: Int,
                failed: Int,
                calls: Int? = nil,
                bytes: UInt64 = 0,
                skipped: Int = 0,
                maxCallNanos: UInt64 = 0) {
        self.requested = requested
        self.failed = failed
        self.calls = calls ?? requested
        self.bytes = bytes
        self.skipped = skipped
        self.maxCallNanos = maxCallNanos
    }

    public static func skipped(requested: Int, bytes: UInt64 = 0) -> ExpertIOAdviceResult {
        ExpertIOAdviceResult(requested: requested,
                             failed: 0,
                             calls: 0,
                             bytes: bytes,
                             skipped: requested)
    }

}

public struct ExpertCachePlan: Sendable, Equatable {
    public let experts: [Int]
    public let assignedSlots: [Int]
    public let misses: [Int]
    public let hits: Int

    public init(experts: [Int], assignedSlots: [Int], misses: [Int], hits: Int) {
        self.experts = experts
        self.assignedSlots = assignedSlots
        self.misses = misses
        self.hits = hits
    }
}

public struct ExpertCacheExecutionTiming: Sendable, Equatable {
    public let totalNanos: UInt64
    public let readWallNanos: UInt64
    public let cacheSlotOverheadNanos: UInt64

    public init(totalNanos: UInt64 = 0,
                readWallNanos: UInt64 = 0,
                cacheSlotOverheadNanos: UInt64 = 0) {
        self.totalNanos = totalNanos
        self.readWallNanos = readWallNanos
        self.cacheSlotOverheadNanos = cacheSlotOverheadNanos
    }
}

public struct ExpertCacheExecutionResult {
    public let buffers: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)]
    public let timing: ExpertCacheExecutionTiming

    public init(buffers: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)],
                timing: ExpertCacheExecutionTiming) {
        self.buffers = buffers
        self.timing = timing
    }
}

public enum ExpertCacheSlotOwnerPhase: String, Sendable, Equatable {
    case unassigned
    case prefillTransient
    case decodeProtected
    case sharedResident
}

public enum ExpertCacheControlPlane: String, Sendable, Equatable {
    case prefill
    case decode
    case sharedPool
}

public struct ExpertCacheAccessContext: Sendable, Equatable {
    public let ownerPhase: ExpertCacheSlotOwnerPhase
    public let controlPlane: ExpertCacheControlPlane
    public let requestID: UInt64?
    public let decodeStepIndex: Int?

    public init(ownerPhase: ExpertCacheSlotOwnerPhase,
                controlPlane: ExpertCacheControlPlane,
                requestID: UInt64? = nil,
                decodeStepIndex: Int? = nil) {
        self.ownerPhase = ownerPhase
        self.controlPlane = controlPlane
        self.requestID = requestID
        self.decodeStepIndex = decodeStepIndex
    }

    public static func phaseOnly(_ ownerPhase: ExpertCacheSlotOwnerPhase) -> ExpertCacheAccessContext {
        let controlPlane: ExpertCacheControlPlane = switch ownerPhase {
        case .prefillTransient:
            .prefill
        case .decodeProtected:
            .decode
        case .sharedResident, .unassigned:
            .sharedPool
        }
        return ExpertCacheAccessContext(ownerPhase: ownerPhase, controlPlane: controlPlane)
    }
}

public struct ExpertCacheSlotSnapshot: Sendable, Equatable {
    public let slot: Int
    public let expert: Int?
    public let ownerPhase: ExpertCacheSlotOwnerPhase
    public let hitCount: Int
    public let lastUseClock: Int

    public init(slot: Int,
                expert: Int?,
                ownerPhase: ExpertCacheSlotOwnerPhase,
                hitCount: Int,
                lastUseClock: Int) {
        self.slot = slot
        self.expert = expert
        self.ownerPhase = ownerPhase
        self.hitCount = hitCount
        self.lastUseClock = lastUseClock
    }
}

public struct ExpertCacheRequestDeltaSnapshot: Sendable, Equatable {
    public let evictions: UInt64
    public let prefillTransientEvictions: UInt64
    public let decodeProtectedEvictions: UInt64
    public let sharedResidentEvictions: UInt64
    public let decodeProtectedPromotions: UInt64
    public let decodeProtectedDemotions: UInt64
    public let decodeProtectedAdmissionRejected: UInt64
    public let prefillSharedResidentPromotions: UInt64
    public let decodeSharedPoolHandoffRequests: UInt64
    public let decodeSharedPoolHandoffHits: UInt64
    public let decodeSharedPoolHandoffMisses: UInt64

    public init(evictions: UInt64,
                prefillTransientEvictions: UInt64,
                decodeProtectedEvictions: UInt64,
                sharedResidentEvictions: UInt64,
                decodeProtectedPromotions: UInt64,
                decodeProtectedDemotions: UInt64,
                decodeProtectedAdmissionRejected: UInt64,
                prefillSharedResidentPromotions: UInt64,
                decodeSharedPoolHandoffRequests: UInt64,
                decodeSharedPoolHandoffHits: UInt64,
                decodeSharedPoolHandoffMisses: UInt64) {
        self.evictions = evictions
        self.prefillTransientEvictions = prefillTransientEvictions
        self.decodeProtectedEvictions = decodeProtectedEvictions
        self.sharedResidentEvictions = sharedResidentEvictions
        self.decodeProtectedPromotions = decodeProtectedPromotions
        self.decodeProtectedDemotions = decodeProtectedDemotions
        self.decodeProtectedAdmissionRejected = decodeProtectedAdmissionRejected
        self.prefillSharedResidentPromotions = prefillSharedResidentPromotions
        self.decodeSharedPoolHandoffRequests = decodeSharedPoolHandoffRequests
        self.decodeSharedPoolHandoffHits = decodeSharedPoolHandoffHits
        self.decodeSharedPoolHandoffMisses = decodeSharedPoolHandoffMisses
    }

    public static let zero = ExpertCacheRequestDeltaSnapshot(
        evictions: 0,
        prefillTransientEvictions: 0,
        decodeProtectedEvictions: 0,
        sharedResidentEvictions: 0,
        decodeProtectedPromotions: 0,
        decodeProtectedDemotions: 0,
        decodeProtectedAdmissionRejected: 0,
        prefillSharedResidentPromotions: 0,
        decodeSharedPoolHandoffRequests: 0,
        decodeSharedPoolHandoffHits: 0,
        decodeSharedPoolHandoffMisses: 0)
}

public struct ExpertCacheRawCounterStepSnapshot: Sendable, Equatable {
    public let decodeStepIndex: Int
    public let handoffHits: UInt64
    public let handoffRequests: UInt64
    public let planMisses: Int
    public let readWallNanos: UInt64

    public init(decodeStepIndex: Int,
                handoffHits: UInt64,
                handoffRequests: UInt64,
                planMisses: Int,
                readWallNanos: UInt64) {
        self.decodeStepIndex = decodeStepIndex
        self.handoffHits = handoffHits
        self.handoffRequests = handoffRequests
        self.planMisses = planMisses
        self.readWallNanos = readWallNanos
    }
}

public struct ExpertCacheTelemetrySnapshot: Sendable, Equatable {
    public let slotCount: Int
    public let occupiedSlots: Int
    public let prefillTransientSlots: Int
    public let decodeProtectedSlots: Int
    public let sharedResidentSlots: Int
    public let unassignedSlots: Int
    public let totalPrefillRequests: UInt64
    public let totalPrefillHits: UInt64
    public let totalPrefillMisses: UInt64
    public let totalDecodeRequests: UInt64
    public let totalDecodeHits: UInt64
    public let totalDecodeMisses: UInt64
    public let totalSharedResidentRequests: UInt64
    public let totalSharedResidentHits: UInt64
    public let totalSharedResidentMisses: UInt64
    public let totalLoads: UInt64
    public let totalEvictions: UInt64
    public let totalPrefillTransientEvictions: UInt64
    public let totalDecodeProtectedEvictions: UInt64
    public let totalSharedResidentEvictions: UInt64
    public let totalDecodeProtectedPromotions: UInt64
    public let totalDecodeProtectedDemotions: UInt64
    public let totalDecodeProtectedAdmissionRejected: UInt64
    public let coldStartGuardActive: Bool
    public let effectiveDecodeProtectedLimit: Int
    public let effectiveDecodeProtectedCap: Int
    public let effectiveDecodeProtectedMinHitCount: Int
    public let currentRequestID: UInt64?
    public let currentRequestPrefillAccessCount: UInt64
    public let currentRequestDecodeAccessCount: UInt64
    public let currentRequestSharedPoolAccessCount: UInt64
    public let currentRequestDecodeStepIndex: Int?
    public let currentRequestDelta: ExpertCacheRequestDeltaSnapshot
    public let currentRequestFocusLayer: Bool
    public let currentRequestRawCounterSteps: [ExpertCacheRawCounterStepSnapshot]
    public let slotSnapshots: [ExpertCacheSlotSnapshot]
    /// Wall-clock nanoseconds spent inside miss reads, summed over every
    /// executed plan. Compared against decode wall time this separates
    /// "IO-bound" from "compute-bound" instead of inferring it from throughput.
    public let totalReadWallNanos: UInt64
    /// Bytes actually pulled through `pread` for misses.
    public let totalReadBytes: UInt64

    public init(slotCount: Int,
                occupiedSlots: Int,
                prefillTransientSlots: Int,
                decodeProtectedSlots: Int,
                sharedResidentSlots: Int,
                unassignedSlots: Int,
                totalPrefillRequests: UInt64,
                totalPrefillHits: UInt64,
                totalPrefillMisses: UInt64,
                totalDecodeRequests: UInt64,
                totalDecodeHits: UInt64,
                totalDecodeMisses: UInt64,
                totalSharedResidentRequests: UInt64,
                totalSharedResidentHits: UInt64,
                totalSharedResidentMisses: UInt64,
                totalLoads: UInt64,
                totalEvictions: UInt64,
                totalPrefillTransientEvictions: UInt64,
                totalDecodeProtectedEvictions: UInt64,
                totalSharedResidentEvictions: UInt64,
                totalDecodeProtectedPromotions: UInt64,
                totalDecodeProtectedDemotions: UInt64,
                totalDecodeProtectedAdmissionRejected: UInt64,
                coldStartGuardActive: Bool,
                effectiveDecodeProtectedLimit: Int,
                effectiveDecodeProtectedCap: Int,
                effectiveDecodeProtectedMinHitCount: Int,
                currentRequestID: UInt64?,
                currentRequestPrefillAccessCount: UInt64,
                currentRequestDecodeAccessCount: UInt64,
                currentRequestSharedPoolAccessCount: UInt64,
                currentRequestDecodeStepIndex: Int?,
                currentRequestDelta: ExpertCacheRequestDeltaSnapshot,
                currentRequestFocusLayer: Bool,
                currentRequestRawCounterSteps: [ExpertCacheRawCounterStepSnapshot],
                slotSnapshots: [ExpertCacheSlotSnapshot],
                totalReadWallNanos: UInt64 = 0,
                totalReadBytes: UInt64 = 0) {
        self.totalReadWallNanos = totalReadWallNanos
        self.totalReadBytes = totalReadBytes
        self.slotCount = slotCount
        self.occupiedSlots = occupiedSlots
        self.prefillTransientSlots = prefillTransientSlots
        self.decodeProtectedSlots = decodeProtectedSlots
        self.sharedResidentSlots = sharedResidentSlots
        self.unassignedSlots = unassignedSlots
        self.totalPrefillRequests = totalPrefillRequests
        self.totalPrefillHits = totalPrefillHits
        self.totalPrefillMisses = totalPrefillMisses
        self.totalDecodeRequests = totalDecodeRequests
        self.totalDecodeHits = totalDecodeHits
        self.totalDecodeMisses = totalDecodeMisses
        self.totalSharedResidentRequests = totalSharedResidentRequests
        self.totalSharedResidentHits = totalSharedResidentHits
        self.totalSharedResidentMisses = totalSharedResidentMisses
        self.totalLoads = totalLoads
        self.totalEvictions = totalEvictions
        self.totalPrefillTransientEvictions = totalPrefillTransientEvictions
        self.totalDecodeProtectedEvictions = totalDecodeProtectedEvictions
        self.totalSharedResidentEvictions = totalSharedResidentEvictions
        self.totalDecodeProtectedPromotions = totalDecodeProtectedPromotions
        self.totalDecodeProtectedDemotions = totalDecodeProtectedDemotions
        self.totalDecodeProtectedAdmissionRejected = totalDecodeProtectedAdmissionRejected
        self.coldStartGuardActive = coldStartGuardActive
        self.effectiveDecodeProtectedLimit = effectiveDecodeProtectedLimit
        self.effectiveDecodeProtectedCap = effectiveDecodeProtectedCap
        self.effectiveDecodeProtectedMinHitCount = effectiveDecodeProtectedMinHitCount
        self.currentRequestID = currentRequestID
        self.currentRequestPrefillAccessCount = currentRequestPrefillAccessCount
        self.currentRequestDecodeAccessCount = currentRequestDecodeAccessCount
        self.currentRequestSharedPoolAccessCount = currentRequestSharedPoolAccessCount
        self.currentRequestDecodeStepIndex = currentRequestDecodeStepIndex
        self.currentRequestDelta = currentRequestDelta
        self.currentRequestFocusLayer = currentRequestFocusLayer
        self.currentRequestRawCounterSteps = currentRequestRawCounterSteps
        self.slotSnapshots = slotSnapshots
    }
}

public enum ExpertCachePolicy: String, Sendable {
    case lru
    case lfu
}

/// `pread`-based routed-expert streamer with a fixed per-layer slot cache.
public final class PreadExpertStreamer: @unchecked Sendable {
    public static let scratchAlignment = 2 * 1024 * 1024
    public static var cachePolicyDefault: ExpertCachePolicy { .lfu }
    public static let boundedCoalescedRunExpertsDefault = 4
    public static let boundedParallelMissReadWorkersDefault = 2

    public let layout: StreamLayout
    public let slotCount: Int
    public let cachePolicy: ExpertCachePolicy
    public let prefillExpertReadMode: RuntimePrefillExpertReadMode
    public let prefillExpertLayerLocalReadaheadExperts: Int
    public let boundedCoalescedRunExperts: Int
    /// When true, expert slots are read-only views into one `mmap` of the layer
    /// file (zero-copy page-cache reads) instead of `pread`-filled private
    /// buffers. First GPU touch of a cold page faults it in lazily; the
    /// `F_RDADVISE` prefetch machinery still applies. Slots become free views,
    /// so effective residency is the page cache, not the slot budget.
    /// Enabled via TURBO_FIELDFARE_EXPERT_MMAP=1 (default off).
    public let useMmap: Bool

    private let device: MTLDevice
    private var mappedBase: UnsafeMutableRawPointer?
    private var mappedLength: Int = 0
    private let fd: Int32
    /// Short human label (the layer file's basename) used only by
    /// `ExpertAccessTrace`, so a streamer can identify itself without the
    /// caller having to thread a layer index down here.
    private let traceLabel: String
    private let slotPointers: [UnsafeMutableRawPointer]
    private var slotBuffers: [MTLBuffer]
    /// Pinned anonymous hot-pool experts (pread mode only): preloaded once at
    /// init into dedicated slots that the planner never evicts. Profile-driven
    /// (top-N per layer from a prior trace); the MTP verify union mostly lands
    /// in these slots, so batched-verify IO collapses to page-cache reads.
    private let hotPoolExperts: [Int]
    /// The pinned hot-pool expert ids (deduped profile). Static after preload;
    /// read lock-free by the miss-prefetch filter in the runner.
    public var poolExperts: [Int] { hotPoolExperts }
    private var slotPinned: [Bool]
    /// mmap mode only: ONE buffer wrapping the whole layer stream. Every slot
    /// references it; per-expert addressing is carried by per-slot OFFSETS in
    /// the argument buffer (setBuffer(offset:)). A single buffer means the
    /// GPU residency pass is registered once per command buffer instead of
    /// once per expert — file-backed bytesNoCopy residency is ~85us per
    /// buffer, which dominated decode overhead when every slot wrapped its own
    /// window.
    private var streamBuffer: MTLBuffer?
    private let reusableCoalescedScratch: UnsafeMutableRawPointer
    private let reusableCoalescedScratchBytes: Int
    private let reusableCoalescedScratchLock = NSLock()
    private let boundedParallelMissReadWorkers: Int

    private var nextSlot = 0
    private let cursorLock = NSLock()

    private var slotExpert: [Int]
    private var slotOwnerPhase: [ExpertCacheSlotOwnerPhase]
    private var slotHitCount: [Int]
    private var slotLastUse: [Int]
    private var expertUseCount: [Int]
    private var useClock = 0
    private var totalPrefillRequests: UInt64 = 0
    private var totalPrefillHits: UInt64 = 0
    private var totalPrefillMisses: UInt64 = 0
    private var totalDecodeRequests: UInt64 = 0
    private var totalDecodeHits: UInt64 = 0
    private var totalDecodeMisses: UInt64 = 0
    private var totalSharedResidentRequests: UInt64 = 0
    private var totalSharedResidentHits: UInt64 = 0
    private var totalSharedResidentMisses: UInt64 = 0
    private var totalLoads: UInt64 = 0
    private var totalReadWallNanos: UInt64 = 0
    private var totalReadBytes: UInt64 = 0
    /// Decode/verify plan residency, bucketed by plan expert count (== the
    /// per-layer union size for batched verify). Lets us measure the REAL
    /// residency hit rate per batch size (t=1 / t=3 / t=5) and compare it to
    /// the static-pool union coverage the runner already reports.
    private var planResidencyBuckets: [Int: (requests: Int, hits: Int)] = [:]
    private var totalEvictions: UInt64 = 0
    private var totalPrefillTransientEvictions: UInt64 = 0
    private var totalDecodeProtectedEvictions: UInt64 = 0
    private var totalSharedResidentEvictions: UInt64 = 0
    private var currentRequestID: UInt64?
    private var currentRequestPrefillAccessCount: UInt64 = 0
    private var currentRequestDecodeAccessCount: UInt64 = 0
    private var currentRequestSharedPoolAccessCount: UInt64 = 0
    private var currentRequestDecodeStepIndex: Int?
    private var currentRequestBaseEvictions: UInt64 = 0
    private var currentRequestBasePrefillTransientEvictions: UInt64 = 0
    private var currentRequestBaseDecodeProtectedEvictions: UInt64 = 0
    private var currentRequestBaseSharedResidentEvictions: UInt64 = 0
    private var currentRequestRawCounterSteps: [ExpertCacheRawCounterStepSnapshot] = []
    private let cacheLock = NSLock()

    public init(layout: StreamLayout,
                device: MTLDevice,
                slotCount: Int,
                useMmap: Bool = false,
                hotPoolExperts: [Int] = [],
                cachePolicy: ExpertCachePolicy = .lfu,
                prefillExpertReadMode: RuntimePrefillExpertReadMode = .baseline,
                prefillExpertLayerLocalReadaheadExperts: Int = 16,
                boundedCoalescedRunExperts: Int = PreadExpertStreamer.boundedCoalescedRunExpertsDefault,
                boundedParallelMissReadWorkers: Int = PreadExpertStreamer.boundedParallelMissReadWorkersDefault) throws {
        precondition(slotCount > 0, "slotCount must be positive")
        precondition(prefillExpertLayerLocalReadaheadExperts > 0,
                     "prefillExpertLayerLocalReadaheadExperts must be positive")
        precondition(boundedCoalescedRunExperts > 0,
                     "boundedCoalescedRunExperts must be positive")
        precondition(boundedParallelMissReadWorkers > 0,
                     "boundedParallelMissReadWorkers must be positive")
        self.layout = layout
        self.traceLabel = (layout.path as NSString).lastPathComponent
        self.slotCount = slotCount
        self.useMmap = useMmap
        var seenPool = Set<Int>()
        self.hotPoolExperts = hotPoolExperts.filter {
            $0 >= 0 && $0 < layout.expertsPerLayer && seenPool.insert($0).inserted
        }
        self.device = device
        self.cachePolicy = cachePolicy
        self.prefillExpertReadMode = prefillExpertReadMode
        self.prefillExpertLayerLocalReadaheadExperts = prefillExpertLayerLocalReadaheadExperts
        self.boundedCoalescedRunExperts = boundedCoalescedRunExperts
        self.boundedParallelMissReadWorkers = max(
            1,
            min(boundedParallelMissReadWorkers,
                ProcessInfo.processInfo.activeProcessorCount))
        let pageSize = Int(getpagesize())
        let reusableScratchBytes = ((Int(layout.expertStride) * boundedCoalescedRunExperts
            + pageSize - 1) / pageSize) * pageSize

        let openedFD = open(layout.path, O_RDONLY)
        guard openedFD >= 0 else {
            throw StreamerError.openFailed(path: layout.path, errno: errno)
        }
        self.fd = openedFD

        var fileStats = stat()
        if fstat(openedFD, &fileStats) == 0 {
            let required = layout.streamOffset + layout.streamSize
            if UInt64(fileStats.st_size) < required {
                close(openedFD)
                throw StreamerError.sizeMismatch(
                    expected: required,
                    actual: UInt64(fileStats.st_size))
            }
        }

        var pointers: [UnsafeMutableRawPointer] = []
        var buffers: [MTLBuffer] = []
        pointers.reserveCapacity(slotCount)
        buffers.reserveCapacity(slotCount)

        if useMmap {
            // One read-only mapping of the whole layer stream. Every expert is
            // a zero-copy window into it; the page cache is the cache.
            let streamSizeInt = Int(layout.streamSize)
            let mapped = mmap(nil, streamSizeInt, PROT_READ, MAP_PRIVATE,
                              openedFD, off_t(layout.streamOffset))
            guard mapped != MAP_FAILED else {
                close(openedFD)
                throw StreamerError.mmapFailed(errno: errno)
            }
            let mappedPtr = mapped!
            self.mappedBase = mappedPtr
            self.mappedLength = streamSizeInt

            // Dummy scratch (unused in mmap mode; the coalesced prefill paths
            // are bypassed). Kept non-optional to match the pread layout.
            var dummyScratch: UnsafeMutableRawPointer?
            let dRes = posix_memalign(&dummyScratch, Self.scratchAlignment, max(pageSize, 1))
            guard dRes == 0, let dummyScratch else {
                munmap(mapped, streamSizeInt)
                close(openedFD)
                throw StreamerError.allocFailed(errno: dRes)
            }
            self.reusableCoalescedScratch = dummyScratch
            self.reusableCoalescedScratchBytes = max(pageSize, 1)

            // One stream buffer for the whole layer stream; every slot is a
            // reference to it. Per-expert data is addressed via offsets in the
            // argument buffer, so no per-expert Metal buffer exists at all.
            guard let stream = device.makeBuffer(
                bytesNoCopy: mappedPtr,
                length: streamSizeInt,
                options: .storageModeShared) else {
                munmap(mapped, streamSizeInt)
                free(dummyScratch)
                close(openedFD)
                throw StreamerError.bufferWrapFailed
            }
            self.streamBuffer = stream
            for _ in 0..<slotCount {
                buffers.append(stream)
                pointers.append(mappedPtr)
            }
        } else {
            var coalescedScratchRaw: UnsafeMutableRawPointer?
            let coalescedScratchResult = posix_memalign(
                &coalescedScratchRaw,
                Self.scratchAlignment,
                reusableScratchBytes)
            guard coalescedScratchResult == 0, let coalescedScratchRaw else {
                close(openedFD)
                throw StreamerError.allocFailed(errno: coalescedScratchResult)
            }
            self.reusableCoalescedScratch = coalescedScratchRaw
            self.reusableCoalescedScratchBytes = reusableScratchBytes

            let allocationSize = ((Int(layout.expertStride) + pageSize - 1) / pageSize) * pageSize

            func unwind() {
                for index in buffers.count..<pointers.count {
                    free(pointers[index])
                }
                free(coalescedScratchRaw)
                close(openedFD)
            }

            for _ in 0..<slotCount {
                var raw: UnsafeMutableRawPointer?
                let result = posix_memalign(&raw, Self.scratchAlignment, allocationSize)
                guard result == 0, let pointer = raw else {
                    unwind()
                    throw StreamerError.allocFailed(errno: result)
                }
                pointers.append(pointer)
                nonisolated(unsafe) let capturedPointer = pointer
                guard let buffer = device.makeBuffer(
                    bytesNoCopy: pointer,
                    length: allocationSize,
                    options: .storageModeShared,
                    deallocator: { _, _ in free(capturedPointer) })
                else {
                    unwind()
                    throw StreamerError.bufferWrapFailed
                }
                buffers.append(buffer)
            }
        }

        self.slotPointers = pointers
        self.slotBuffers = buffers
        self.slotExpert = [Int](repeating: -1, count: slotCount)
        self.slotOwnerPhase = [ExpertCacheSlotOwnerPhase](repeating: .unassigned, count: slotCount)
        self.slotHitCount = [Int](repeating: 0, count: slotCount)
        self.slotLastUse = [Int](repeating: 0, count: slotCount)
        self.expertUseCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
        self.slotPinned = [Bool](repeating: false, count: slotCount)
        if !useMmap && !hotPoolExperts.isEmpty {
            // Pinned pool: dedicated anonymous slots that are never evicted.
            // Leave >= 16 LRU slots so any single-token plan (topK<=8) and
            // MTP verify unions (<=16) can always be placed.
            // INVARIANT: keep >= minLruSlots evictable slots. The decode plan
            // calls planExpertsCached (preconditionFailure on overflow), so any
            // plan needing more misses than the remaining LRU slots crashes.
            // topK=4 single token needs <= 4; MTP 4-row verify union <= 16.
            // This coupling is what ties the margin to 16.
            if Self.adaptivePoolEnabled {
                // Adaptive pool: warm-start fill only, no pinning. Use the
                // full slot budget (nothing is reserved, so the minLruSlots
                // margin is unnecessary). slotOwnerPhase marks them as pool
                // residents for telemetry; the planner may still evict them
                // (promote-on-miss) once the run's routing diverges.
                let poolSize = min(hotPoolExperts.count, slotCount)
                let poolExperts = Array(hotPoolExperts.prefix(poolSize))
                for slot in 0..<poolExperts.count {
                    slotOwnerPhase[slot] = .sharedResident
                }
                if Self.asyncPoolPreloadEnabled {
                    let selfRef = self
                    Self.poolPreloadQueue.async {
                        Self.poolPreloadSemaphore.wait()
                        defer { Self.poolPreloadSemaphore.signal() }
                        do {
                            try selfRef.loadPoolExperts(poolExperts)
                        } catch {
                            let line = "[hot-pool] async preload failed: \(error)\n"
                            FileHandle.standardError.write(Data(line.utf8))
                        }
                    }
                } else {
                    try loadPoolExperts(poolExperts)
                }
            } else {
                let minLruSlots = 16
                let poolSize = min(hotPoolExperts.count, max(0, slotCount - minLruSlots))
                let poolExperts = Array(hotPoolExperts.prefix(poolSize))
                // Reserve the pool slots NOW (zero reads) so the planner never
                // hands them out. slotExpert stays -1 until each expert lands, so
                // an unloaded pool expert is just a normal miss (loaded into an
                // LRU slot) — the pool simply is not hot yet. Safe either way.
                for slot in 0..<poolExperts.count {
                    slotPinned[slot] = true
                    slotOwnerPhase[slot] = .sharedResident
                }
                if Self.asyncPoolPreloadEnabled {
                    let selfRef = self
                    Self.poolPreloadQueue.async {
                        Self.poolPreloadSemaphore.wait()
                        defer { Self.poolPreloadSemaphore.signal() }
                        do {
                            try selfRef.loadPoolExperts(poolExperts)
                        } catch {
                            let line = "[hot-pool] async preload failed: \(error)\n"
                            FileHandle.standardError.write(Data(line.utf8))
                        }
                    }
                } else {
                    try loadPoolExperts(poolExperts)
                }
            }
        }
    }

    /// Bounded (2-deep) background queue for async hot-pool preloads across
    /// all layer streamers, so 30 layers never thrash the SSD at once.
    private static let poolPreloadQueue = DispatchQueue(
        label: "turbofieldfare.hotpool-preload",
        qos: .utility,
        attributes: .concurrent)
    private static let poolPreloadSemaphore = DispatchSemaphore(value: 2)
    private static let asyncPoolPreloadEnabled =
        ProcessInfo.processInfo.environment["TURBO_FIELDFARE_HOT_POOL_PRELOAD"] == "async"
    /// TURBO_FIELDFARE_ADAPTIVE_POOL=1: pool slots are warm-start fill only
    /// (NOT pinned). The LFU/LRU eviction then promotes hot non-pool experts
    /// and demotes cold pool members = the offline-sim "pin-as-you-go" model.
    /// Static profile still preloads them (warm start); they just become
    /// evictable once the run's real routing differs from the profile.
    private static let adaptivePoolEnabled =
        ProcessInfo.processInfo.environment["TURBO_FIELDFARE_ADAPTIVE_POOL"] == "1"

    /// Load a set of experts into their pinned pool slots. Slots are already
    /// pinned by the caller; this only fills data + slot metadata under the
    /// cache lock (the planner reads the same fields under it).
    private func loadPoolExperts(_ experts: [Int]) throws {
        for (slot, expert) in experts.enumerated() {
            let offset = layout.expertOffset(layer: 0, expert: expert)
            try readFull(
                into: slotPointers[slot],
                fileOffset: layout.streamOffset + offset,
                count: Int(layout.expertStride))
            cacheLock.lock()
            slotExpert[slot] = expert
            slotLastUse[slot] = 0
            slotHitCount[slot] = 1
            expertUseCount[expert] = 1
            cacheLock.unlock()
        }
    }

    deinit {
        free(reusableCoalescedScratch)
        close(fd)
        // Views are released after this body runs; at teardown no command
        // buffer is in flight (decode completed), so unmapping is safe.
        if let mappedBase {
            munmap(mappedBase, mappedLength)
        }
    }

    public func loadExpert(layer: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        cursorLock.lock()
        // Never hand out a pinned hot-pool slot to a new expert: that would
        // silently break the pin (slotPinned stays true but the data changes).
        var slot = nextSlot
        for _ in 0..<slotCount where slotPinned[slot] {
            slot = (slot + 1) % slotCount
        }
        nextSlot = (slot + 1) % slotCount
        cursorLock.unlock()
        return try loadExpert(layer: layer, expert: expert, slot: slot)
    }

    public func loadExpert(layer: Int, expert: Int, slot: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        guard slot >= 0 && slot < slotCount else {
            throw StreamerError.slotOutOfRange(slot)
        }
        let regionOffset = layout.expertOffset(layer: layer, expert: expert)
        guard regionOffset + layout.expertStride <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        if useMmap {
            // Zero-copy: wrap the expert's page-aligned window in the mapping.
            // No pread, no copy; a cold page faults in on first GPU touch.
            guard let mappedBase else {
                throw StreamerError.mmapFailed(errno: EINVAL)
            }
            let windowStart = mappedBase.advanced(by: Int(regionOffset))
            // Pull the window into the page cache with kernel readahead. The
            // GPU can fault cold file-backed pages correctly on its own (blit
            // probe), but a resident window avoids mid-command-buffer stalls.
            // WILLNEED reads in large sequential chunks — unlike a per-page
            // touch loop, which issues one no-readahead 16K read per fault.
            let strideInt = Int(layout.expertStride)
            guard let streamBuffer else {
                throw StreamerError.mmapFailed(errno: EINVAL)
            }
            // WILLNEED issues a bulk readahead (efficient large sequential
            // reads); the per-page touch then guarantees every page is
            // resident. The touch is REQUIRED: the GPU cannot fault truly-cold
            // file-backed pages — it reads them as garbage (a blit probe that
            // "proved" otherwise was flawed: its file had just been written,
            // so the pages were already resident in the page cache).
            _ = posix_madvise(windowStart, strideInt, POSIX_MADV_WILLNEED)
            let pageSize = Int(getpagesize())
            let pageCount = (strideInt + pageSize - 1) / pageSize
            var touchSum: UInt8 = 0
            var touch = windowStart
            for _ in 0..<pageCount {
                touchSum &+= touch.load(as: UInt8.self)
                touch = touch.advanced(by: pageSize)
            }
            if touchSum == 0 && strideInt > 0 {
                // Keep the touch observable (real windows are never all-zero).
            }
            slotBuffers[slot] = streamBuffer
            return (streamBuffer, UInt64(regionOffset), layout.expertStride)
        }
        try readFull(
            into: slotPointers[slot],
            fileOffset: layout.streamOffset + regionOffset,
            count: Int(layout.expertStride))
        return (slotBuffers[slot], 0, layout.expertStride)
    }

    public func loadExpertsCached(experts: [Int]) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        try executeExpertCachePlan(planExpertsCached(experts: experts))
    }

    public func loadExpertsCached(experts: [Int],
                                  accessContext: ExpertCacheAccessContext) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        try executeExpertCachePlan(planExpertsCached(experts: experts, accessContext: accessContext),
                                   accessContext: accessContext)
    }

    public func planExpertsCached(experts: [Int],
                                  avoidingSlots: Set<Int> = []) -> ExpertCachePlan {
        guard let plan = makeExpertCachePlan(experts: experts, avoidingSlots: avoidingSlots) else {
            preconditionFailure("expert cache cannot place requested misses")
        }
        return plan
    }

    public func planExpertsCached(experts: [Int],
                                  avoidingSlots: Set<Int> = [],
                                  accessContext: ExpertCacheAccessContext) -> ExpertCachePlan {
        guard let plan = makeExpertCachePlan(
            experts: experts,
            avoidingSlots: avoidingSlots,
            accessContext: accessContext)
        else {
            preconditionFailure("expert cache cannot place requested misses")
        }
        return plan
    }

    public func planExpertsCachedIfPossible(experts: [Int],
                                            avoidingSlots: Set<Int> = []) -> ExpertCachePlan? {
        makeExpertCachePlan(experts: experts, avoidingSlots: avoidingSlots)
    }

    public func planExpertsCachedIfPossible(experts: [Int],
                                            avoidingSlots: Set<Int> = [],
                                            accessContext: ExpertCacheAccessContext) -> ExpertCachePlan? {
        makeExpertCachePlan(experts: experts,
                            avoidingSlots: avoidingSlots,
                            accessContext: accessContext)
    }

    private func makeExpertCachePlan(experts: [Int],
                                     avoidingSlots rawAvoidingSlots: Set<Int>) -> ExpertCachePlan? {
        makeExpertCachePlan(experts: experts, avoidingSlots: rawAvoidingSlots, accessContext: nil)
    }

    private func makeExpertCachePlan(experts: [Int],
                                     avoidingSlots rawAvoidingSlots: Set<Int>,
                                     accessContext: ExpertCacheAccessContext?) -> ExpertCachePlan? {
        precondition(experts.count <= slotCount,
                     "expert cache needs at least \(experts.count) slots")
        let avoidingSlots = Set(rawAvoidingSlots.filter { $0 >= 0 && $0 < slotCount })

        cacheLock.lock()
        defer { cacheLock.unlock() }

        let clock = useClock + 1
        var assignedSlots = [Int](repeating: -1, count: experts.count)
        var reserved = [Bool](repeating: false, count: slotCount)

        for index in experts.indices {
            for slot in 0..<slotCount
                where !reserved[slot] && slotExpert[slot] == experts[index] {
                assignedSlots[index] = slot
                reserved[slot] = true
                break
            }
        }
        for slot in avoidingSlots where !reserved[slot] {
            reserved[slot] = true
        }

        let misses = experts.indices.filter { assignedSlots[$0] == -1 }
        let evictable = (0..<slotCount)
            .filter { !reserved[$0] && !slotPinned[$0] }
            .sorted { shouldEvictSlot($0, before: $1) }
        guard misses.count <= evictable.count else { return nil }

        if let accessContext {
            beginRequestTrackingIfNeeded(accessContext)
            recordAccessCounts(accessContext: accessContext, requestCount: experts.count)
            recordRequestStats(accessContext: accessContext,
                               hits: experts.count - misses.count,
                               misses: misses.count)
            // Bucket by plan expert count (== per-layer union size): t=1
            // decode plans carry topK (8) experts; verify batches carry the
            // union (~18 for t=3, ~27 for t=5). Including prefill lets MTP
            // verify (which runs on the prefill control plane) be measured.
            let hits = experts.count - misses.count
            var bucket = planResidencyBuckets[experts.count, default: (0, 0)]
            bucket.requests += experts.count
            bucket.hits += hits
            planResidencyBuckets[experts.count] = bucket
            if let decodeStepIndex = accessContext.decodeStepIndex {
                updateRawCounterStep(decodeStepIndex: decodeStepIndex,
                                     handoffHitsDelta: 0,
                                     handoffRequestsDelta: 0,
                                     planMissesDelta: misses.count,
                                     readWallNanosDelta: 0)
            }
        }

        useClock = clock
        for expert in experts where expert >= 0 && expert < expertUseCount.count {
            expertUseCount[expert] &+= 1
        }
        for slot in assignedSlots where slot >= 0 {
            slotLastUse[slot] = clock
            slotHitCount[slot] &+= 1
            if let accessContext {
                slotOwnerPhase[slot] = accessContext.ownerPhase
            }
        }
        for (offset, index) in misses.enumerated() {
            let slot = evictable[offset]
            let previousExpert = slotExpert[slot]
            let previousOwnerPhase = slotOwnerPhase[slot]
            assignedSlots[index] = slot
            reserved[slot] = true
            slotExpert[slot] = -1
            slotHitCount[slot] = 0
            slotLastUse[slot] = clock
            if previousExpert >= 0 {
                totalEvictions &+= 1
                switch previousOwnerPhase {
                case .prefillTransient:
                    totalPrefillTransientEvictions &+= 1
                case .decodeProtected:
                    totalDecodeProtectedEvictions &+= 1
                case .sharedResident, .unassigned:
                    totalSharedResidentEvictions &+= 1
                }
            }
            if let accessContext {
                slotOwnerPhase[slot] = accessContext.ownerPhase
            } else {
                slotOwnerPhase[slot] = .unassigned
            }
        }

        if ExpertAccessTrace.shared.isEnabled {
            ExpertAccessTrace.shared.record(
                layerLabel: traceLabel,
                phase: accessContext?.ownerPhase.rawValue ?? "none",
                experts: experts,
                hits: experts.count - misses.count,
                misses: misses.count)
        }

        return ExpertCachePlan(
            experts: experts,
            assignedSlots: assignedSlots,
            misses: misses,
            hits: experts.count - misses.count)
    }

    public func executeExpertCachePlan(_ plan: ExpertCachePlan) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        try executeExpertCachePlanDetailed(plan, accessContext: nil).buffers
    }

    public func executeExpertCachePlan(_ plan: ExpertCachePlan,
                                       accessContext: ExpertCacheAccessContext?) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        try executeExpertCachePlanDetailed(plan, accessContext: accessContext).buffers
    }

    public func executeExpertCachePlanDetailed(_ plan: ExpertCachePlan,
                                               accessContext: ExpertCacheAccessContext?) throws
        -> ExpertCacheExecutionResult {
        precondition(plan.experts.count <= slotCount,
                     "expert cache plan exceeds slot count")
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")

        let executionStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let readStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        if useMmap {
            // Instant: loading a "miss" is just wrapping a window of the
            // mapping. The slot table still drives hit/miss accounting and the
            // phase1-hit/miss GPU split; both sides are free here.
            for index in plan.misses {
                _ = try loadExpert(layer: 0,
                                   expert: plan.experts[index],
                                   slot: plan.assignedSlots[index])
            }
        } else if accessContext?.controlPlane == .prefill {
            switch prefillExpertReadMode {
            case .baseline:
                // Prefill keeps a low read depth: TTFT measured +13.8% with
                // decode-tuned workers (8) because the parallel preads fight
                // the hot-pool sync preload for SSD bandwidth. Decode keeps
                // the full env-tuned depth (w8 = +24% decode, 2026-08-07).
                try executeParallelMissReads(
                    plan,
                    workerLimit: min(PreadExpertStreamer.boundedParallelMissReadWorkersDefault,
                                     boundedParallelMissReadWorkers))
            case .coalesced:
                try executeCoalescedPrefillMissReads(plan)
            case .layerLocalReadahead:
                try executeLayerLocalReadaheadPrefillMissReads(plan)
            }
        } else {
            try executeParallelMissReads(plan)
        }
        let readWallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - readStart

        cacheLock.lock()
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            slotExpert[slot] = plan.experts[index]
            slotHitCount[slot] = 1
            slotOwnerPhase[slot] = accessContext?.ownerPhase ?? .sharedResident
        }
        totalLoads &+= UInt64(plan.misses.count)
        totalReadWallNanos &+= readWallNanos
        totalReadBytes &+= UInt64(plan.misses.count) &* layout.expertStride
        if let decodeStepIndex = accessContext?.decodeStepIndex {
            updateRawCounterStep(decodeStepIndex: decodeStepIndex,
                                 handoffHitsDelta: 0,
                                 handoffRequestsDelta: 0,
                                 planMissesDelta: 0,
                                 readWallNanosDelta: readWallNanos)
        }
        cacheLock.unlock()
        let totalNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - executionStart
        let buffers = expertCachePlanBuffers(plan)
        let timing = ExpertCacheExecutionTiming(
            totalNanos: totalNanos,
            readWallNanos: readWallNanos,
            cacheSlotOverheadNanos: totalNanos &- readWallNanos)
        return ExpertCacheExecutionResult(buffers: buffers, timing: timing)
    }

    private struct MissReadEntry {
        let index: Int
        let slot: Int
        let fileOffset: UInt64
    }

    private func makeSortedMissReadEntries(_ plan: ExpertCachePlan) -> [MissReadEntry] {
        plan.misses.map { index in
            MissReadEntry(
                index: index,
                slot: plan.assignedSlots[index],
                fileOffset: layout.streamOffset + layout.expertOffset(layer: 0, expert: plan.experts[index]))
        }.sorted { lhs, rhs in
            if lhs.fileOffset == rhs.fileOffset { return lhs.slot < rhs.slot }
            return lhs.fileOffset < rhs.fileOffset
        }
    }

    private func executeParallelMissReads(_ plan: ExpertCachePlan,
                                          workerLimit: Int? = nil) throws {
        guard !plan.misses.isEmpty else { return }
        let limit = workerLimit ?? boundedParallelMissReadWorkers
        let workerCount = min(plan.misses.count, limit)
        guard workerCount > 1 else {
            for index in plan.misses {
                _ = try loadExpert(layer: 0, expert: plan.experts[index], slot: plan.assignedSlots[index])
            }
            return
        }
        final class MissReadCursor: @unchecked Sendable {
            private let lock = NSLock()
            private var nextOffset = 0

            func take(limit: Int) -> Int? {
                lock.lock()
                defer { lock.unlock() }
                guard nextOffset < limit else { return nil }
                let result = nextOffset
                nextOffset += 1
                return result
            }
        }
        let errorLock = NSLock()
        let cursor = MissReadCursor()
        nonisolated(unsafe) var firstError: Error?
        DispatchQueue.concurrentPerform(iterations: workerCount) { _ in
            while true {
                guard let missOffset = cursor.take(limit: plan.misses.count) else { break }
                let index = plan.misses[missOffset]
                do {
                    _ = try self.loadExpert(
                        layer: 0,
                        expert: plan.experts[index],
                        slot: plan.assignedSlots[index])
                } catch {
                    errorLock.lock()
                    if firstError == nil { firstError = error }
                    errorLock.unlock()
                }
            }
        }
        if let firstError { throw firstError }
    }

    private func executeCoalescedPrefillMissReads(_ plan: ExpertCachePlan) throws {
        try executeCoalescedSortedMissReadEntries(makeSortedMissReadEntries(plan))
    }

    private func executeLayerLocalReadaheadPrefillMissReads(_ plan: ExpertCachePlan) throws {
        let sortedEntries = makeSortedMissReadEntries(plan)
        issueLayerLocalReadahead(for: sortedEntries)
        try executeCoalescedSortedMissReadEntries(sortedEntries)
    }

    private func executeCoalescedSortedMissReadEntries(_ sortedEntries: [MissReadEntry]) throws {
        guard !sortedEntries.isEmpty else { return }
        // mmap mode bypasses this path (see executeExpertCachePlanDetailed);
        // slotPointers is empty there, so never fall through.
        if useMmap { return }
        let stride = Int(layout.expertStride)

        var runStart = 0
        while runStart < sortedEntries.count {
            var runEnd = runStart + 1
            while runEnd < sortedEntries.count,
                  runEnd - runStart < boundedCoalescedRunExperts,
                  sortedEntries[runEnd].fileOffset ==
                      sortedEntries[runEnd - 1].fileOffset + layout.expertStride {
                runEnd += 1
            }

            let run = Array(sortedEntries[runStart..<runEnd])
            if run.count == 1 {
                let entry = run[0]
                try readFull(
                    into: slotPointers[entry.slot],
                    fileOffset: entry.fileOffset,
                    count: stride)
            } else {
                let totalBytes = run.count * stride
                  try withReusableCoalescedScratch(totalBytes: totalBytes) { scratch in
                      try readFull(
                          into: scratch,
                          fileOffset: run[0].fileOffset,
                          count: totalBytes)
                      for (offset, entry) in run.enumerated() {
                          memcpy(slotPointers[entry.slot], scratch.advanced(by: offset * stride), stride)
                      }
                  }
            }
            runStart = runEnd
        }
    }

    private func withReusableCoalescedScratch(
        totalBytes: Int,
        body: (UnsafeMutableRawPointer) throws -> Void) throws {
        guard totalBytes > 0 else { return }
        if totalBytes <= reusableCoalescedScratchBytes {
            reusableCoalescedScratchLock.lock()
            defer { reusableCoalescedScratchLock.unlock() }
            try body(reusableCoalescedScratch)
            return
        }
        let scratch = UnsafeMutableRawPointer.allocate(
            byteCount: totalBytes,
            alignment: MemoryLayout<UInt64>.alignment)
        defer { scratch.deallocate() }
        try body(scratch)
    }

    private func issueLayerLocalReadahead(for sortedEntries: [MissReadEntry]) {
        guard !sortedEntries.isEmpty else { return }
        let stride = layout.expertStride
        let span = max(1, prefillExpertLayerLocalReadaheadExperts)
        var ranges: [(offset: UInt64, count: UInt64)] = []
        ranges.reserveCapacity(sortedEntries.count)
        for index in sortedEntries.indices {
            let endIndex = min(sortedEntries.count - 1, index + span - 1)
            let offset = sortedEntries[index].fileOffset
            let end = sortedEntries[endIndex].fileOffset &+ stride
            ranges.append((offset: offset, count: end &- offset))
        }
        _ = adviseRanges(ranges, requested: sortedEntries.count)
    }

    public func expertCachePlanBuffers(_ plan: ExpertCachePlan)
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")
        return plan.assignedSlots.map { slot in
            if useMmap {
                // All slots share the stream buffer; the argument buffer gets
                // the per-expert offset via setBuffer(offset:).
                let expert = min(max(slotExpert[slot], 0),
                                 max(layout.expertsPerLayer - 1, 0))
                return (slotBuffers[slot],
                        UInt64(layout.expertOffset(layer: 0, expert: expert)),
                        layout.expertStride)
            }
            return (slotBuffers[slot], UInt64(0), layout.expertStride)
        }
    }

    public func adviseExpertCachePlanMisses(_ plan: ExpertCachePlan) -> ExpertIOAdviceResult {
        let experts = plan.misses.map { plan.experts[$0] }
        return adviseRanges(expertAdviceRanges(experts: experts), requested: experts.count)
    }

    public func adviseExperts(experts: [Int]) -> ExpertIOAdviceResult {
        adviseRanges(expertAdviceRanges(experts: experts), requested: experts.count)
    }

    public func adviseExpertMisses(experts: [Int]) -> ExpertIOAdviceResult {
        cacheLock.lock()
        let misses = experts.filter { !slotExpert.contains($0) }
        cacheLock.unlock()
        return adviseRanges(expertAdviceRanges(experts: misses), requested: misses.count)
    }

    static func coalescedAdjacentAdviceRanges(_ ranges: [(offset: UInt64, count: UInt64)])
        -> [(offset: UInt64, count: UInt64)] {
        let sorted = ranges.filter { $0.count > 0 }.sorted {
            $0.offset == $1.offset ? $0.count < $1.count : $0.offset < $1.offset
        }
        var result: [(offset: UInt64, count: UInt64)] = []
        for range in sorted {
            guard var last = result.popLast() else {
                result.append(range)
                continue
            }
            let lastEnd = last.offset &+ last.count
            let rangeEnd = range.offset &+ range.count
            if range.offset <= lastEnd {
                last.count = max(lastEnd, rangeEnd) - last.offset
                result.append(last)
            } else {
                result.append(last)
                result.append(range)
            }
        }
        return result
    }

    /// Bucketed decode/verify plan residency: [batchSize: (requests, hits)].
    /// batchSize == per-layer union size (topK for t=1; the union for verify).
    public func planResidencySummary() -> [Int: (requests: Int, hits: Int)] {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return planResidencyBuckets
    }

    public func telemetrySnapshot() -> ExpertCacheTelemetrySnapshot {
        cacheLock.lock()
        let slotSnapshots = (0..<slotCount).map { slot in
            ExpertCacheSlotSnapshot(
                slot: slot,
                expert: slotExpert[slot] >= 0 ? slotExpert[slot] : nil,
                ownerPhase: slotOwnerPhase[slot],
                hitCount: slotHitCount[slot],
                lastUseClock: slotLastUse[slot])
        }
        let occupiedSlots = slotSnapshots.reduce(into: 0) { count, snapshot in
            if snapshot.expert != nil { count += 1 }
        }
        let prefillTransientSlots = slotSnapshots.reduce(into: 0) { count, snapshot in
            if snapshot.ownerPhase == .prefillTransient { count += 1 }
        }
        let decodeProtectedSlots = slotSnapshots.reduce(into: 0) { count, snapshot in
            if snapshot.ownerPhase == .decodeProtected { count += 1 }
        }
        let sharedResidentSlots = slotSnapshots.reduce(into: 0) { count, snapshot in
            if snapshot.ownerPhase == .sharedResident { count += 1 }
        }
        let unassignedSlots = slotSnapshots.reduce(into: 0) { count, snapshot in
            if snapshot.ownerPhase == .unassigned { count += 1 }
        }
        let currentRequestDelta: ExpertCacheRequestDeltaSnapshot
        if currentRequestID != nil {
            currentRequestDelta = ExpertCacheRequestDeltaSnapshot(
                evictions: totalEvictions &- currentRequestBaseEvictions,
                prefillTransientEvictions: totalPrefillTransientEvictions
                    &- currentRequestBasePrefillTransientEvictions,
                decodeProtectedEvictions: totalDecodeProtectedEvictions
                    &- currentRequestBaseDecodeProtectedEvictions,
                sharedResidentEvictions: totalSharedResidentEvictions
                    &- currentRequestBaseSharedResidentEvictions,
                decodeProtectedPromotions: 0,
                decodeProtectedDemotions: 0,
                decodeProtectedAdmissionRejected: 0,
                prefillSharedResidentPromotions: 0,
                decodeSharedPoolHandoffRequests: 0,
                decodeSharedPoolHandoffHits: 0,
                decodeSharedPoolHandoffMisses: 0)
        } else {
            currentRequestDelta = .zero
        }
        let snapshot = ExpertCacheTelemetrySnapshot(
            slotCount: slotCount,
            occupiedSlots: occupiedSlots,
            prefillTransientSlots: prefillTransientSlots,
            decodeProtectedSlots: decodeProtectedSlots,
            sharedResidentSlots: sharedResidentSlots,
            unassignedSlots: unassignedSlots,
            totalPrefillRequests: totalPrefillRequests,
            totalPrefillHits: totalPrefillHits,
            totalPrefillMisses: totalPrefillMisses,
            totalDecodeRequests: totalDecodeRequests,
            totalDecodeHits: totalDecodeHits,
            totalDecodeMisses: totalDecodeMisses,
            totalSharedResidentRequests: totalSharedResidentRequests,
            totalSharedResidentHits: totalSharedResidentHits,
            totalSharedResidentMisses: totalSharedResidentMisses,
            totalLoads: totalLoads,
            totalEvictions: totalEvictions,
            totalPrefillTransientEvictions: totalPrefillTransientEvictions,
            totalDecodeProtectedEvictions: totalDecodeProtectedEvictions,
            totalSharedResidentEvictions: totalSharedResidentEvictions,
            totalDecodeProtectedPromotions: 0,
            totalDecodeProtectedDemotions: 0,
            totalDecodeProtectedAdmissionRejected: 0,
            coldStartGuardActive: false,
            effectiveDecodeProtectedLimit: 0,
            effectiveDecodeProtectedCap: 0,
            effectiveDecodeProtectedMinHitCount: 0,
            currentRequestID: currentRequestID,
            currentRequestPrefillAccessCount: currentRequestPrefillAccessCount,
            currentRequestDecodeAccessCount: currentRequestDecodeAccessCount,
            currentRequestSharedPoolAccessCount: currentRequestSharedPoolAccessCount,
            currentRequestDecodeStepIndex: currentRequestDecodeStepIndex,
            currentRequestDelta: currentRequestDelta,
            currentRequestFocusLayer: currentRequestID != nil,
            currentRequestRawCounterSteps: currentRequestRawCounterSteps.sorted {
                $0.decodeStepIndex < $1.decodeStepIndex
            },
            slotSnapshots: slotSnapshots,
            totalReadWallNanos: totalReadWallNanos,
            totalReadBytes: totalReadBytes)
        cacheLock.unlock()
        return snapshot
    }

    private func beginRequestTrackingIfNeeded(_ accessContext: ExpertCacheAccessContext) {
        if currentRequestID != accessContext.requestID {
            currentRequestID = accessContext.requestID
            currentRequestPrefillAccessCount = 0
            currentRequestDecodeAccessCount = 0
            currentRequestSharedPoolAccessCount = 0
            currentRequestDecodeStepIndex = accessContext.decodeStepIndex
            currentRequestBaseEvictions = totalEvictions
            currentRequestBasePrefillTransientEvictions = totalPrefillTransientEvictions
            currentRequestBaseDecodeProtectedEvictions = totalDecodeProtectedEvictions
            currentRequestBaseSharedResidentEvictions = totalSharedResidentEvictions
            currentRequestRawCounterSteps = []
            return
        }
        if let decodeStepIndex = accessContext.decodeStepIndex {
            currentRequestDecodeStepIndex = max(currentRequestDecodeStepIndex ?? decodeStepIndex,
                                                decodeStepIndex)
        }
    }

    private func recordAccessCounts(accessContext: ExpertCacheAccessContext,
                                    requestCount: Int) {
        switch accessContext.controlPlane {
        case .prefill:
            currentRequestPrefillAccessCount &+= UInt64(requestCount)
        case .decode:
            currentRequestDecodeAccessCount &+= UInt64(requestCount)
        case .sharedPool:
            currentRequestSharedPoolAccessCount &+= UInt64(requestCount)
        }
    }

    private func recordRequestStats(accessContext: ExpertCacheAccessContext,
                                    hits: Int,
                                    misses: Int) {
        switch accessContext.controlPlane {
        case .prefill:
            totalPrefillRequests &+= UInt64(hits + misses)
            totalPrefillHits &+= UInt64(hits)
            totalPrefillMisses &+= UInt64(misses)
        case .decode:
            totalDecodeRequests &+= UInt64(hits + misses)
            totalDecodeHits &+= UInt64(hits)
            totalDecodeMisses &+= UInt64(misses)
        case .sharedPool:
            totalSharedResidentRequests &+= UInt64(hits + misses)
            totalSharedResidentHits &+= UInt64(hits)
            totalSharedResidentMisses &+= UInt64(misses)
        }
    }

    private func updateRawCounterStep(decodeStepIndex: Int,
                                      handoffHitsDelta: UInt64,
                                      handoffRequestsDelta: UInt64,
                                      planMissesDelta: Int,
                                      readWallNanosDelta: UInt64) {
        if let index = currentRequestRawCounterSteps.firstIndex(where: {
            $0.decodeStepIndex == decodeStepIndex
        }) {
            let current = currentRequestRawCounterSteps[index]
            currentRequestRawCounterSteps[index] = ExpertCacheRawCounterStepSnapshot(
                decodeStepIndex: decodeStepIndex,
                handoffHits: current.handoffHits &+ handoffHitsDelta,
                handoffRequests: current.handoffRequests &+ handoffRequestsDelta,
                planMisses: current.planMisses + planMissesDelta,
                readWallNanos: current.readWallNanos &+ readWallNanosDelta)
            return
        }
        currentRequestRawCounterSteps.append(
            ExpertCacheRawCounterStepSnapshot(
                decodeStepIndex: decodeStepIndex,
                handoffHits: handoffHitsDelta,
                handoffRequests: handoffRequestsDelta,
                planMisses: planMissesDelta,
                readWallNanos: readWallNanosDelta))
    }

    private func shouldEvictSlot(_ lhs: Int, before rhs: Int) -> Bool {
        if cachePolicy == .lru {
            return slotLastUse[lhs] < slotLastUse[rhs]
        }
        let lhsExpert = slotExpert[lhs]
        let rhsExpert = slotExpert[rhs]
        if lhsExpert < 0 || rhsExpert < 0 {
            return lhsExpert < rhsExpert
        }
        let lhsCount = lhsExpert < expertUseCount.count ? expertUseCount[lhsExpert] : 0
        let rhsCount = rhsExpert < expertUseCount.count ? expertUseCount[rhsExpert] : 0
        if lhsCount != rhsCount { return lhsCount < rhsCount }
        return slotLastUse[lhs] < slotLastUse[rhs]
    }

    private func expertAdviceRanges(experts: [Int]) -> [(offset: UInt64, count: UInt64)] {
        experts.compactMap { expert in
            let regionOffset = layout.expertOffset(layer: 0, expert: expert)
            guard regionOffset + layout.expertStride <= layout.streamSize else { return nil }
            return (layout.streamOffset + regionOffset, layout.expertStride)
        }
    }

    private func adviseRanges(_ ranges: [(offset: UInt64, count: UInt64)],
                              requested: Int) -> ExpertIOAdviceResult {
        let coalesced = Self.coalescedAdjacentAdviceRanges(ranges)
        var failed = 0
        var bytes: UInt64 = 0
        var maxCallNanos: UInt64 = 0
        for range in coalesced {
            let result = RDAdvice.call(fd: fd, offset: range.offset, byteCount: range.count)
            if !result.succeeded { failed += 1 }
            bytes &+= result.requestedBytes
            maxCallNanos = max(maxCallNanos, result.elapsedNanos)
        }
        return ExpertIOAdviceResult(
            requested: requested,
            failed: failed,
            calls: coalesced.count,
            bytes: bytes,
            maxCallNanos: maxCallNanos)
    }

    private func readFull(into destination: UnsafeMutableRawPointer,
                          fileOffset: UInt64,
                          count: Int) throws {
        var filled = 0
        while filled < count {
            let readCount = pread(
                fd,
                destination.advanced(by: filled),
                count - filled,
                off_t(fileOffset) + off_t(filled))
            if readCount < 0 {
                throw StreamerError.preadFailed(errno: errno)
            }
            if readCount == 0 {
                throw StreamerError.sizeMismatch(expected: UInt64(count), actual: UInt64(filled))
            }
            filled += readCount
        }
    }
}
