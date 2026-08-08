import Foundation
import Metal

public struct RoutedExpertFetchPlan: Sendable {
    public let layer: Int
    public let cachePlan: ExpertCachePlan

    public var experts: [Int] { cachePlan.experts }
    public var misses: [Int] { cachePlan.misses }
    public var hits: Int { cachePlan.hits }
    public var assignedSlots: [Int] { cachePlan.assignedSlots }

    public init(layer: Int, cachePlan: ExpertCachePlan) {
        self.layer = layer
        self.cachePlan = cachePlan
    }
}

public struct RoutedExpertFetchResult: Sendable {
    public let views: [TensorView]
    public let executionTiming: ExpertCacheExecutionTiming

    public init(views: [TensorView], executionTiming: ExpertCacheExecutionTiming) {
        self.views = views
        self.executionTiming = executionTiming
    }
}

public struct RoutedExpertCacheLayerTelemetrySnapshot: Sendable, Equatable {
    public let layer: Int
    public let snapshot: ExpertCacheTelemetrySnapshot

    public init(layer: Int, snapshot: ExpertCacheTelemetrySnapshot) {
        self.layer = layer
        self.snapshot = snapshot
    }
}

extension Model {
    public func routedExpertOffsets(layer: Int) -> MoEExpertOffsets {
        let expert = packedExpertsLayout.expert(layer: layer, expert: 0)
        func offset(_ role: String) -> UInt32 {
            UInt32(expert.subTensors[role]?.offset ?? 0)
        }
        if ProcessInfo.processInfo.environment["TURBO_FIELDFARE_DEBUG_OFFSETS"] != nil {
            fputs("[rebits-debug] layer=\(layer) stride=\(packedExpertsLayout.expertStride) "
                + "gate=\(offset("gate")) up=\(offset("up")) down=\(offset("down")) "
                + "gate_s=\(offset("gate_scales")) down_s=\(offset("down_scales")) "
                + "routedBits=\(routedExpertWeightBits)\n", stderr)
        }
        return MoEExpertOffsets(
            gateWOff: offset("gate"),
            gateSOff: offset("gate_scales"),
            gateBOff: offset("gate_biases"),
            upWOff: offset("up"),
            upSOff: offset("up_scales"),
            upBOff: offset("up_biases"),
            downWOff: offset("down"),
            downSOff: offset("down_scales"),
            downBOff: offset("down_biases"))
    }

    public func routedExpertPhysicalOffsets(layer: Int) -> [UInt64] {
        packedExpertsLayout.layers[layer].experts.map(\.offset)
    }

    public func adviseRoutedExperts(layer: Int,
                                    experts: [Int]) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return streamer.adviseExpertMisses(experts: experts)
    }

    public func routedExpertAdviceByteEstimate(layer: Int,
                                               missCount: Int) throws -> UInt64 {
        guard missCount > 0 else { return 0 }
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return UInt64(missCount) * streamer.layout.expertStride
    }

    public func planRoutedExperts(layer: Int,
                                  experts: [Int],
                                  avoidingSlots: Set<Int> = []) throws -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        return RoutedExpertFetchPlan(
            layer: layer,
            cachePlan: streamer.planExpertsCached(experts: experts, avoidingSlots: validSlots))
    }

    public func planRoutedExperts(layer: Int,
                                  experts: [Int],
                                  avoidingSlots: Set<Int> = [],
                                  accessContext: ExpertCacheAccessContext? = nil) throws
        -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        let cachePlan = if let accessContext {
            streamer.planExpertsCached(experts: experts,
                                       avoidingSlots: validSlots,
                                       accessContext: accessContext)
        } else {
            streamer.planExpertsCached(experts: experts, avoidingSlots: validSlots)
        }
        return RoutedExpertFetchPlan(layer: layer, cachePlan: cachePlan)
    }

    public func planRoutedExpertsIfPossible(layer: Int,
                                            experts: [Int],
                                            avoidingSlots: Set<Int> = []) throws
        -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        guard let cachePlan = streamer.planExpertsCachedIfPossible(
            experts: experts,
            avoidingSlots: validSlots)
        else {
            return nil
        }
        return RoutedExpertFetchPlan(layer: layer, cachePlan: cachePlan)
    }

    public func planRoutedExpertsIfPossible(layer: Int,
                                            experts: [Int],
                                            avoidingSlots: Set<Int> = [],
                                            accessContext: ExpertCacheAccessContext? = nil) throws
        -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        let cachePlan = if let accessContext {
            streamer.planExpertsCachedIfPossible(experts: experts,
                                                 avoidingSlots: validSlots,
                                                 accessContext: accessContext)
        } else {
            streamer.planExpertsCachedIfPossible(experts: experts,
                                                 avoidingSlots: validSlots)
        }
        guard let cachePlan else { return nil }
        return RoutedExpertFetchPlan(layer: layer, cachePlan: cachePlan)
    }

    public func routedExpertCacheSlotCount(layer _: Int) -> Int? {
        guard case .pread(let slotCount) = streamingMode else { return nil }
        return slotCount
    }

    public func routedExpertBuffers(for plan: RoutedExpertFetchPlan) throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return Self.makeExpertViews(
            streamer.expertCachePlanBuffers(plan.cachePlan),
            layer: plan.layer,
            experts: plan.experts)
    }

    public func adviseRoutedExperts(plan: RoutedExpertFetchPlan) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return streamer.adviseExpertCachePlanMisses(plan.cachePlan)
    }

    public func fetchRoutedExperts(plan: RoutedExpertFetchPlan) async throws -> [TensorView] {
        try await fetchRoutedExpertsDetailed(plan: plan).views
    }

    /// Warms `layer`'s slot cache with `experts` and reports how many bytes had
    /// to be read.
    ///
    /// This is the same work `fetchRoutedExperts` does minus the tensor-view
    /// construction: the caller has no use for the buffers yet, it only wants
    /// the slots resident so the *real* fetch a layer later turns into hits.
    /// Runs synchronously on whatever thread calls it — the caller is expected
    /// to be a detached task overlapping GPU execution.
    ///
    /// Returns `nil` when every expert was already cached (no work done).
    @discardableResult
    public func prefetchRoutedExperts(layer: Int,
                                      experts: [Int],
                                      accessContext: ExpertCacheAccessContext) throws
        -> Int? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let cachePlan = streamer.planExpertsCached(experts: experts,
                                                   avoidingSlots: [],
                                                   accessContext: accessContext)
        guard !cachePlan.misses.isEmpty else { return nil }
        _ = try streamer.executeExpertCachePlanDetailed(cachePlan,
                                                        accessContext: accessContext)
        return cachePlan.misses.count
    }

    public func fetchRoutedExpertsDetailed(plan: RoutedExpertFetchPlan) async throws -> RoutedExpertFetchResult {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try streamer.executeExpertCachePlanDetailed(plan.cachePlan, accessContext: nil)
                    continuation.resume(returning: RoutedExpertFetchResult(
                        views: Self.makeExpertViews(
                            result.buffers,
                            layer: plan.layer,
                            experts: plan.experts),
                        executionTiming: result.timing))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// B4: hit-only fast path. When the plan has zero misses every expert is
    /// already resident in the slot cache, so the fetch is pure CPU bookkeeping
    /// — no pread, no dispatch hop. Running it synchronously on the decode
    /// thread removes the withCheckedThrowingContinuation + global-queue async
    /// hop the generic path pays on every layer (~100s of us each; 3800
    /// ~1.1ms GPU-idle gaps after sharedFFN per 128-token run traced
    /// 2026-08-07). The async path stays for any layer with real I/O.
    public func fetchRoutedExpertsHitOnlySync(
        plan: RoutedExpertFetchPlan,
        accessContext: ExpertCacheAccessContext?
    ) throws -> RoutedExpertFetchResult {
        guard plan.misses.isEmpty else {
            throw ModelError.hitOnlySyncWithMisses
        }
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        let result = try streamer.executeExpertCachePlanDetailed(
            plan.cachePlan,
            accessContext: accessContext)
        return RoutedExpertFetchResult(
            views: Self.makeExpertViews(
                result.buffers,
                layer: plan.layer,
                experts: plan.experts),
            executionTiming: result.timing)
    }

    public func fetchRoutedExperts(plan: RoutedExpertFetchPlan,
                                   accessContext: ExpertCacheAccessContext) async throws
        -> [TensorView] {
        try await fetchRoutedExpertsDetailed(plan: plan, accessContext: accessContext).views
    }

    public func fetchRoutedExpertsDetailed(plan: RoutedExpertFetchPlan,
                                           accessContext: ExpertCacheAccessContext) async throws
        -> RoutedExpertFetchResult {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try streamer.executeExpertCachePlanDetailed(
                        plan.cachePlan,
                        accessContext: accessContext)
                    continuation.resume(returning: RoutedExpertFetchResult(
                        views: Self.makeExpertViews(
                            result.buffers,
                            layer: plan.layer,
                            experts: plan.experts),
                        executionTiming: result.timing))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    public func fetchRoutedExperts(layer: Int, experts: [Int]) async throws -> [TensorView] {
        try await fetchRoutedExpertsDetailed(layer: layer, experts: experts).views
    }

    public func fetchRoutedExpertsDetailed(layer: Int,
                                           experts: [Int]) async throws -> RoutedExpertFetchResult {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try streamer.executeExpertCachePlanDetailed(
                        streamer.planExpertsCached(experts: experts),
                        accessContext: nil)
                    continuation.resume(returning: RoutedExpertFetchResult(
                        views: Self.makeExpertViews(
                            result.buffers,
                            layer: layer,
                            experts: experts),
                        executionTiming: result.timing))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    public func fetchRoutedExperts(layer: Int,
                                   experts: [Int],
                                   accessContext: ExpertCacheAccessContext) async throws
        -> [TensorView] {
        try await fetchRoutedExpertsDetailed(
            layer: layer,
            experts: experts,
            accessContext: accessContext).views
    }

    public func fetchRoutedExpertsDetailed(layer: Int,
                                           experts: [Int],
                                           accessContext: ExpertCacheAccessContext) async throws
        -> RoutedExpertFetchResult {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try streamer.executeExpertCachePlanDetailed(
                        streamer.planExpertsCached(experts: experts, accessContext: accessContext),
                        accessContext: accessContext)
                    continuation.resume(returning: RoutedExpertFetchResult(
                        views: Self.makeExpertViews(
                            result.buffers,
                            layer: layer,
                            experts: experts),
                        executionTiming: result.timing))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func makeExpertViews(
        _ buffers: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)],
        layer: Int,
        experts: [Int]
    ) -> [TensorView] {
        buffers.enumerated().map { index, entry in
            TensorView(
                buffer: entry.buffer,
                offset: entry.offset,
                length: entry.size,
                scaleOffset: 0,
                scaleLength: 0,
                biasOffset: 0,
                biasLength: 0,
                shape: (UInt32(layer), UInt32(experts[index]), 0, 0),
                dtype: 0)
        }
    }

    public func routedExpertCacheTelemetrySnapshots() -> [RoutedExpertCacheLayerTelemetrySnapshot] {
        streamersQueue.sync {
            streamersBox.streamers.enumerated().compactMap { layer, streamer in
                guard let streamer else { return nil }
                return RoutedExpertCacheLayerTelemetrySnapshot(
                    layer: layer,
                    snapshot: streamer.telemetrySnapshot())
            }
        }
    }

    /// Per-layer pinned hot-pool expert sets for the lock-free miss-prefetch
    /// filter. Static after the streamers open (prefill touches every layer).
    /// Aggregated decode/verify plan residency across all layer streamers,
    /// keyed by batch size. Used by the CLI footer to verify adaptive-pool
    /// union coverage vs the static-pool coverage the runner measures.
    public func routedExpertCachePlanResidency()
        -> [Int: (requests: Int, hits: Int)] {
        var out: [Int: (requests: Int, hits: Int)] = [:]
        streamersQueue.sync {
            for streamer in streamersBox.streamers {
                guard let streamer else { continue }
                for (batch, pair) in streamer.planResidencySummary() {
                    var acc = out[batch, default: (0, 0)]
                    acc.requests += pair.requests
                    acc.hits += pair.hits
                    out[batch] = acc
                }
            }
        }
        return out
    }

    public func hotPoolExpertSets() -> [Set<Int>] {
        streamersQueue.sync {
            streamersBox.streamers.map { streamer in
                Set(streamer?.poolExperts ?? [])
            }
        }
    }
}
