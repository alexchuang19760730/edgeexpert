import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Synchronization
import TurboFieldfare

public actor TurboFieldfareHTTPServer {
    public static let maximumBodyBytes = 1_048_576

    private let group: MultiThreadedEventLoopGroup
    private let modelID: String
    private let backend: any ServerInferenceBackend
    private let coordinator: ServerCoordinator
    private let heartbeatInterval: TimeAmount
    private let childChannels = ChildChannelRegistry()
    private var channel: Channel?
    private var shutdownTask: Task<Void, any Error>?

    public init(modelID: String,
                queueLimit: Int,
                backend: any ServerInferenceBackend,
                heartbeatInterval: TimeAmount = .seconds(5),
                group: MultiThreadedEventLoopGroup = .init(numberOfThreads: 1)) {
        self.group = group
        self.modelID = modelID
        self.backend = backend
        self.coordinator = ServerCoordinator(queueLimit: queueLimit)
        self.heartbeatInterval = heartbeatInterval
    }

    public func start(port: Int) async throws -> Channel {
        let modelID = self.modelID
        let backend = self.backend
        let coordinator = self.coordinator
        let heartbeatInterval = self.heartbeatInterval
        let childChannels = self.childChannels
        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(ChannelOptions.backlog, value: 16)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                childChannels.insert(channel)
                return channel.pipeline.configureHTTPServerPipeline(
                    withPipeliningAssistance: true,
                    withErrorHandling: true
                ).flatMap {
                    channel.pipeline.addHandler(ServerHTTPHandler(
                        modelID: modelID,
                        backend: backend,
                        coordinator: coordinator,
                        heartbeatInterval: heartbeatInterval,
                        childChannels: childChannels))
                }
            }
            .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
        let channel = try await bootstrap.bind(host: "127.0.0.1", port: port).get()
        self.channel = channel
        return channel
    }

    public func shutdown() async throws {
        if let shutdownTask {
            try await shutdownTask.value
            return
        }

        let listeningChannel = channel
        channel = nil
        let childChannels = self.childChannels
        let coordinator = self.coordinator
        let group = self.group
        let task = Task { @Sendable in
            var firstError: (any Error)?
            await coordinator.shutdown()
            if let listeningChannel {
                do {
                    try await listeningChannel.close().get()
                } catch ChannelError.alreadyClosed {
                } catch {
                    firstError = error
                }
            }
            await childChannels.closeAll()
            do {
                try await group.shutdownGracefully()
            } catch {
                if firstError == nil {
                    firstError = error
                }
            }
            if let firstError {
                throw firstError
            }
        }
        shutdownTask = task
        try await task.value
    }

    var queuedRequestCount: Int {
        get async { await coordinator.queuedCount }
    }

    var hasActiveRequest: Bool {
        get async { await coordinator.isActive }
    }

    var acceptedConnectionCount: Int {
        childChannels.count
    }
}

private final class ServerHTTPHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private let modelID: String
    private let backend: any ServerInferenceBackend
    private let coordinator: ServerCoordinator
    private let heartbeatInterval: TimeAmount
    private let childChannels: ChildChannelRegistry
    private var head: HTTPRequestHead?
    private var body = ByteBuffer()
    private var oversized = false
    private var activeTask: Task<Void, Never>?

    init(modelID: String,
         backend: any ServerInferenceBackend,
         coordinator: ServerCoordinator,
         heartbeatInterval: TimeAmount,
         childChannels: ChildChannelRegistry) {
        self.modelID = modelID
        self.backend = backend
        self.coordinator = coordinator
        self.heartbeatInterval = heartbeatInterval
        self.childChannels = childChannels
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        switch unwrapInboundIn(data) {
        case .head(let head):
            self.head = head
            body.clear()
            oversized = false
        case .body(var part):
            if body.readableBytes + part.readableBytes > TurboFieldfareHTTPServer.maximumBodyBytes {
                oversized = true
            } else {
                body.writeBuffer(&part)
            }
        case .end:
            guard let head else { return }
            self.head = nil
            if oversized {
                writeError(context, status: .payloadTooLarge,
                           OpenAIErrorEnvelope(message: "request body is too large",
                                               code: "request_too_large"))
                return
            }
            route(head: head, body: body, context: context)
        }
    }

    func channelInactive(context: ChannelHandlerContext) {
        activeTask?.cancel()
        activeTask = nil
        childChannels.remove(context.channel)
        context.fireChannelInactive()
    }

    private func route(head: HTTPRequestHead,
                       body: ByteBuffer,
                       context: ChannelHandlerContext) {
        let path = head.uri.split(separator: "?", maxSplits: 1,
                                  omittingEmptySubsequences: false).first.map(String.init) ?? head.uri
        switch (head.method, path) {
        case (.GET, "/health"):
            writeJSON(context, status: .ok, object: ["status": "ok"])
        case (.GET, "/v1/models"):
            let response = OpenAIModelList(
                object: "list",
                data: [.init(id: modelID,
                             object: "model",
                             created: 0,
                             ownedBy: "turbofieldfare")])
            writeCodable(context, status: .ok, response)
        case (.POST, "/v1/chat/completions"):
            guard head.headers.first(name: "content-type")?
                .lowercased().hasPrefix("application/json") == true else {
                writeError(context, status: .unsupportedMediaType,
                           OpenAIErrorEnvelope(message: "content-type must be application/json",
                                               code: "unsupported_media_type"))
                return
            }
            handleCompletion(body: body, context: context)
        case (_, "/health"), (_, "/v1/models"), (_, "/v1/chat/completions"):
            writeError(context, status: .methodNotAllowed,
                       OpenAIErrorEnvelope(message: "method not allowed",
                                           code: "method_not_allowed"))
        default:
            writeError(context, status: .notFound,
                       OpenAIErrorEnvelope(message: "route not found",
                                           code: "not_found"))
        }
    }

    private func handleCompletion(body: ByteBuffer,
                                  context: ChannelHandlerContext) {
        do {
            let bytes = body.getBytes(at: body.readerIndex, length: body.readableBytes) ?? []
            let decoded = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(bytes))
            let request = try OpenAIRequestValidator.validate(decoded, modelID: modelID)
            let responseID = "chatcmpl-" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
            let created = Int(Date().timeIntervalSince1970)
            let contextBox = SendableContext(context)
            let streamState = StreamState()
            let phaseState = RequestPhaseState()
            let startStream: @Sendable () -> Void = {
                guard request.stream,
                      streamState.start(eventLoop: contextBox.value.eventLoop,
                                        interval: self.heartbeatInterval,
                                        ping: {
                          self.writeHeartbeat(contextBox.value)
                      }) else { return }
                let future = self.beginStream(
                    contextBox.value,
                    self.chunk(id: responseID, created: created,
                               delta: ["role": "assistant"],
                               finishReason: nil))
                streamState.setStartFuture(future)
            }
            let onQueued: @Sendable () -> Void = {
                phaseState.set("queued")
                ServerLog.queued(id: responseID)
                startStream()
            }
            activeTask = childChannels.startTask {
                defer { streamState.stop() }
                let started = ContinuousClock.now
                ServerLog.accepted(id: responseID, streaming: request.stream)
                do {
                    let completion = try await self.coordinator.runPreparing(
                        onQueued: onQueued,
                        prepare: {
                            let prepared = try await self.backend.prepare(request)
                            phaseState.set("prepared")
                            ServerLog.prepared(id: responseID,
                                               promptTokens: prepared.promptTokenCount)
                            return prepared
                        },
                        operation: { prepared in
                            try Task.checkCancellation()
                            startStream()
                            try await streamState.waitUntilStarted()
                            try Task.checkCancellation()
                            phaseState.set("generating")
                            ServerLog.generating(id: responseID)
                            return try await self.backend.generate(prepared) { event in
                                guard request.stream else { return }
                                switch event {
                                case .content(let text):
                                    self.writeStreamChunk(
                                        contextBox.value,
                                        self.chunk(id: responseID, created: created,
                                                   delta: ["content": text],
                                                   finishReason: nil))
                                case .toolCall(let call):
                                    self.writeToolCall(contextBox.value,
                                                       id: responseID,
                                                       created: created,
                                                       toolIndex: streamState.nextToolIndex(),
                                                       call: call)
                                }
                            }
                    })
                    ServerLog.completed(id: responseID,
                                        duration: started.duration(to: .now),
                                        completion: completion)
                    if request.stream {
                        streamState.stop()
                        self.finishStream(contextBox.value,
                                          id: responseID,
                                          created: created,
                                          completion: completion,
                                          includeUsage: request.includeUsage)
                    } else {
                        self.writeCompletion(contextBox.value,
                                             id: responseID,
                                             created: created,
                                             completion: completion)
                    }
                } catch {
                    streamState.stop()
                    self.handleAsyncError(error,
                                          context: contextBox.value,
                                          id: responseID,
                                          phase: phaseState.value,
                                          stream: streamState.isStarted)
                }
            }
        } catch let error as ServerRequestError {
            writeError(context,
                       status: error == .unknownModel ? .notFound : .badRequest,
                       error.envelope)
        } catch {
            writeError(context, status: .badRequest,
                       OpenAIErrorEnvelope(message: "malformed JSON request",
                                           code: "invalid_json"))
        }
    }

    private func writeCompletion(_ context: ChannelHandlerContext,
                                 id: String,
                                 created: Int,
                                 completion: ServerCompletion) {
        let encodedContent: Any =
            completion.content.isEmpty && !completion.toolCalls.isEmpty
                ? NSNull()
                : completion.content
        var message: [String: Any] = [
            "role": "assistant",
            "content": encodedContent,
        ]
        if !completion.toolCalls.isEmpty {
            message["tool_calls"] = completion.toolCalls.map(toolCallObject)
        }
        let object: [String: Any] = [
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": modelID,
            "choices": [[
                "index": 0,
                "message": message,
                "finish_reason": completion.finishReason,
            ]],
            "usage": usageObject(completion.usage),
            "diagnostics": diagnosticsObject(completion),
        ]
        writeJSON(context, status: .ok, object: object)
    }

    private func beginStream(
        _ context: ChannelHandlerContext,
        _ initialChunk: [String: Any]
    ) -> EventLoopFuture<Void> {
        guard let data = try? JSONSerialization.data(withJSONObject: initialChunk) else {
            return context.eventLoop.makeFailedFuture(ServerRequestError.invalid(
                message: "stream response could not be encoded",
                param: nil,
                code: "internal_error"))
        }
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "text/event-stream")
        headers.add(name: "cache-control", value: "no-cache")
        headers.add(name: "connection", value: "keep-alive")
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        let contextBox = SendableContext(context)
        let promise = context.eventLoop.makePromise(of: Void.self)
        context.eventLoop.execute {
            contextBox.value.write(self.wrapOutboundOut(.head(head)),
                promise: nil)
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count + 8)
            buffer.writeString("data: ")
            buffer.writeBytes(data)
            buffer.writeString("\n\n")
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))),
                promise: promise)
        }
        return promise.futureResult
    }

    private func writeToolCall(_ context: ChannelHandlerContext,
                               id: String,
                               created: Int,
                               toolIndex: Int,
                               call: ParsedToolCall) {
        let fragments = utf8Fragments(call.argumentsJSON, maximumBytes: 1024)
        for (index, fragment) in fragments.enumerated() {
            var function: [String: Any] = ["arguments": fragment]
            var tool: [String: Any] = ["index": toolIndex, "function": function]
            if index == 0 {
                function["name"] = call.name
                tool["id"] = call.id
                tool["type"] = "function"
                tool["function"] = function
            }
            writeStreamChunk(
                context,
                chunk(id: id, created: created,
                      delta: ["tool_calls": [tool]],
                      finishReason: nil))
        }
    }

    private func finishStream(_ context: ChannelHandlerContext,
                              id: String,
                              created: Int,
                              completion: ServerCompletion,
                              includeUsage: Bool) {
        writeStreamChunk(
            context,
            chunk(id: id, created: created,
                  delta: [:],
                  finishReason: completion.finishReason))
        if includeUsage {
            writeStreamChunk(context, [
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": modelID,
                "choices": [],
                "usage": usageObject(completion.usage),
                "diagnostics": diagnosticsObject(completion),
            ])
        }
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            let buffer = contextBox.value.channel.allocator.buffer(string: "data: [DONE]\n\n")
            contextBox.value.write(self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
        }
    }

    private func chunk(id: String,
                       created: Int,
                       delta: [String: Any],
                       finishReason: String?) -> [String: Any] {
        let encodedReason: Any = finishReason.map { $0 as Any } ?? NSNull()
        return [
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": modelID,
            "choices": [[
                "index": 0,
                "delta": delta,
                "finish_reason": encodedReason,
            ]],
        ]
    }

    private func writeStreamChunk(_ context: ChannelHandlerContext,
                                  _ object: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count + 8)
            buffer.writeString("data: ")
            buffer.writeBytes(data)
            buffer.writeString("\n\n")
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
        }
    }

    private func writeHeartbeat(_ context: ChannelHandlerContext) {
        let buffer = context.channel.allocator.buffer(string: ": ping\n\n")
        context.writeAndFlush(
            wrapOutboundOut(.body(.byteBuffer(buffer))),
            promise: nil)
    }

    private func handleAsyncError(_ error: Error,
                                  context: ChannelHandlerContext,
                                  id: String,
                                  phase: String,
                                  stream: Bool) {
        let envelope: OpenAIErrorEnvelope
        let status: HTTPResponseStatus
        if let requestError = error as? ServerRequestError {
            status = requestError == .queueFull ? .tooManyRequests : .badRequest
            envelope = requestError.envelope
        } else {
            status = .internalServerError
            envelope = OpenAIErrorEnvelope(
                message: "generation failed; see TurboFieldfareServer stderr",
                code: "internal_error",
                type: "server_error")
        }
        if !(error is CancellationError) {
            ServerLog.failed(id: id, phase: phase, status: status.code, error: error)
        }
        if stream {
            finishStreamWithError(context, envelope: envelope)
            return
        }
        writeError(context, status: status, envelope)
    }

    private func finishStreamWithError(_ context: ChannelHandlerContext,
                                       envelope: OpenAIErrorEnvelope) {
        guard let data = try? JSONEncoder().encode(envelope) else { return }
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            guard contextBox.value.channel.isActive else { return }
            var buffer = contextBox.value.channel.allocator.buffer(
                capacity: data.count + 32)
            buffer.writeString("data: ")
            buffer.writeBytes(data)
            buffer.writeString("\n\ndata: [DONE]\n\n")
            contextBox.value.write(
                self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            contextBox.value.writeAndFlush(
                self.wrapOutboundOut(.end(nil)), promise: nil)
        }
    }

    private func writeCodable<T: Encodable>(_ context: ChannelHandlerContext,
                                            status: HTTPResponseStatus,
                                            _ value: T) {
        guard let data = try? JSONEncoder().encode(value) else { return }
        writeData(context, status: status, data: data)
    }

    private func writeError(_ context: ChannelHandlerContext,
                            status: HTTPResponseStatus,
                            _ error: OpenAIErrorEnvelope) {
        writeCodable(context, status: status, error)
    }

    private func writeJSON(_ context: ChannelHandlerContext,
                           status: HTTPResponseStatus,
                           object: Any) {
        guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
        writeData(context, status: status, data: data)
    }

    private func writeData(_ context: ChannelHandlerContext,
                           status: HTTPResponseStatus,
                           data: Data) {
        let contextBox = SendableContext(context)
        context.eventLoop.execute {
            var headers = HTTPHeaders()
            headers.add(name: "content-type", value: "application/json")
            headers.add(name: "content-length", value: "\(data.count)")
            contextBox.value.write(self.wrapOutboundOut(.head(
                HTTPResponseHead(version: .http1_1, status: status, headers: headers))),
                promise: nil)
            var buffer = contextBox.value.channel.allocator.buffer(capacity: data.count)
            buffer.writeBytes(data)
            contextBox.value.write(self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            contextBox.value.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
        }
    }

    private func usageObject(_ usage: OpenAIUsage) -> [String: Any] {
        [
            "prompt_tokens": usage.promptTokens,
            "completion_tokens": usage.completionTokens,
            "total_tokens": usage.totalTokens,
            "prompt_tokens_details": [
                "cached_tokens": usage.promptTokensDetails.cachedTokens,
            ],
        ]
    }

    private func diagnosticsObject(_ completion: ServerCompletion) -> [String: Any] {
        let prefillBreakdownMs: [String: Any]? = completion.prefillTimingBreakdown.map { breakdown in
            func ms(_ nanos: UInt64) -> Double {
                Double(nanos) / 1_000_000.0
            }
            return [
                "attention_ms": ms(breakdown.attentionNanos),
                "router_front_half_gpu_ms": ms(breakdown.routerFrontHalfGPUNanos),
                "router_readback_cpu_ms": ms(breakdown.routerReadbackAndCPUNanos),
                "shared_expert_ms": ms(breakdown.sharedExpertNanos),
                "streamed_routed_tiles_wait_ms": ms(breakdown.streamedRoutedTilesWaitNanos),
                "tail_reduce_layer_tail_ms": ms(breakdown.tailReduceLayerTailNanos),
                "prefill_tile_plan_ms": ms(breakdown.tilePlanNanos),
                "prefill_tile_fetch_binding_ms": ms(breakdown.tileFetchBindingNanos),
                "prefill_tile_argument_buffer_ms": ms(breakdown.tileArgumentBufferNanos),
                "prefill_tile_fetch_open_or_read_ms": ms(breakdown.tileFetchOpenOrReadNanos),
                "prefill_tile_fetch_read_wall_ms": ms(breakdown.tileFetchReadWallNanos),
                "prefill_tile_fetch_cache_slot_overhead_ms": ms(breakdown.tileFetchCacheSlotOverheadNanos),
            ]
        }
        let prefillExpertAccessPattern: [String: Any]? = completion.prefillExpertAccessPattern.map { pattern in
            func ratio(_ numerator: UInt64, _ denominator: UInt64) -> Double? {
                guard denominator > 0 else { return nil }
                return Double(numerator) / Double(denominator)
            }

            let duplicateExpertReferences = pattern.expertReferences >= pattern.uniqueExperts
                ? pattern.expertReferences - pattern.uniqueExperts
                : 0
            return [
                "tile_accesses": pattern.tileAccesses,
                "expert_references": pattern.expertReferences,
                "unique_experts": pattern.uniqueExperts,
                "duplicate_expert_references": duplicateExpertReferences,
                "duplicate_reference_ratio": ratio(duplicateExpertReferences, pattern.expertReferences) as Any,
                "mean_experts_per_tile": ratio(pattern.expertReferences, pattern.tileAccesses) as Any,
                "consecutive_tile_comparisons": pattern.consecutiveTileComparisons,
                "consecutive_tile_overlap_experts": pattern.consecutiveTileOverlapExperts,
                "consecutive_tile_requested_experts": pattern.consecutiveTileRequestedExperts,
                "consecutive_tile_overlap_ratio": ratio(pattern.consecutiveTileOverlapExperts,
                                                         pattern.consecutiveTileRequestedExperts) as Any,
                "contiguous_neighbor_pairs": pattern.contiguousNeighborPairs,
                "contiguous_neighbor_opportunities": pattern.contiguousNeighborOpportunities,
                "contiguous_neighbor_ratio": ratio(pattern.contiguousNeighborPairs,
                                                    pattern.contiguousNeighborOpportunities) as Any,
                "contiguous_run_count": pattern.contiguousRunCount,
                "experts_in_contiguous_runs": pattern.expertsInContiguousRuns,
                "max_contiguous_run_length": pattern.maxContiguousRunLength,
            ]
        }
        let layers = completion.routedExpertCacheTelemetrySnapshots
            .filter { layerSnapshot in
                let stats = layerSnapshot.snapshot
                return stats.totalPrefillRequests > 0
                    || stats.totalDecodeRequests > 0
                    || stats.totalSharedResidentRequests > 0
                    || stats.totalEvictions > 0
                    || stats.occupiedSlots > 0
            }
            .sorted { $0.layer < $1.layer }
            .map { layerSnapshot in
                let stats = layerSnapshot.snapshot
                return [
                    "layer": layerSnapshot.layer,
                    "slot_count": stats.slotCount,
                    "occupied_slots": stats.occupiedSlots,
                    "cold_start_guard": stats.coldStartGuardActive,
                    "decode_protected_cap": stats.effectiveDecodeProtectedCap,
                    "decode_protected_budget": stats.effectiveDecodeProtectedLimit,
                    "decode_protected_min_hits": stats.effectiveDecodeProtectedMinHitCount,
                    "current_request_id": stats.currentRequestID as Any,
                    "current_request_prefill_accesses": stats.currentRequestPrefillAccessCount,
                    "current_request_decode_accesses": stats.currentRequestDecodeAccessCount,
                    "current_request_shared_pool_accesses": stats.currentRequestSharedPoolAccessCount,
                    "current_request_decode_step": stats.currentRequestDecodeStepIndex as Any,
                    "current_request_delta": [
                        "evictions": stats.currentRequestDelta.evictions,
                        "prefill_transient_evictions": stats.currentRequestDelta.prefillTransientEvictions,
                        "decode_protected_evictions": stats.currentRequestDelta.decodeProtectedEvictions,
                        "shared_resident_evictions": stats.currentRequestDelta.sharedResidentEvictions,
                        "decode_protected_promotions": stats.currentRequestDelta.decodeProtectedPromotions,
                        "decode_protected_demotions": stats.currentRequestDelta.decodeProtectedDemotions,
                        "decode_protected_admission_rejected":
                            stats.currentRequestDelta.decodeProtectedAdmissionRejected,
                        "prefill_shared_resident_promotions":
                            stats.currentRequestDelta.prefillSharedResidentPromotions,
                        "decode_shared_pool_handoff_requests":
                            stats.currentRequestDelta.decodeSharedPoolHandoffRequests,
                        "decode_shared_pool_handoff_hits":
                            stats.currentRequestDelta.decodeSharedPoolHandoffHits,
                        "decode_shared_pool_handoff_misses":
                            stats.currentRequestDelta.decodeSharedPoolHandoffMisses,
                    ],
                    "current_request_focus_layer": stats.currentRequestFocusLayer,
                    "current_request_raw_counter_steps": stats.currentRequestRawCounterSteps.map { step in
                        [
                            "decode_step_index": step.decodeStepIndex,
                            "handoff_hits": step.handoffHits,
                            "handoff_requests": step.handoffRequests,
                            "plan_misses": step.planMisses,
                            "read_wall_nanos": step.readWallNanos,
                        ]
                    },
                    "decode_protected_slots": stats.decodeProtectedSlots,
                    "prefill_transient_slots": stats.prefillTransientSlots,
                    "shared_resident_slots": stats.sharedResidentSlots,
                    "prefill_requests": stats.totalPrefillRequests,
                    "prefill_hits": stats.totalPrefillHits,
                    "prefill_misses": stats.totalPrefillMisses,
                    "decode_requests": stats.totalDecodeRequests,
                    "decode_hits": stats.totalDecodeHits,
                    "decode_misses": stats.totalDecodeMisses,
                    "shared_resident_requests": stats.totalSharedResidentRequests,
                    "shared_resident_hits": stats.totalSharedResidentHits,
                    "shared_resident_misses": stats.totalSharedResidentMisses,
                    "total_loads": stats.totalLoads,
                    "total_evictions": stats.totalEvictions,
                    "prefill_transient_evictions": stats.totalPrefillTransientEvictions,
                    "decode_protected_evictions": stats.totalDecodeProtectedEvictions,
                    "decode_protected_promotions": stats.totalDecodeProtectedPromotions,
                    "decode_protected_demotions": stats.totalDecodeProtectedDemotions,
                    "decode_protected_admission_rejected": stats.totalDecodeProtectedAdmissionRejected,
                    "shared_resident_evictions": stats.totalSharedResidentEvictions,
                ] as [String: Any]
            }
        let aggregate: [String: Any] = [
            "layers": layers.count,
            "cold_start_guard_layers": layers.reduce(0) {
                $0 + (((($1["cold_start_guard"] as? Bool) ?? false) ? 1 : 0))
            },
            "decode_protected_cap": layers.reduce(0) {
                $0 + (Int(truncating: ($1["decode_protected_cap"] as? NSNumber) ?? 0))
            },
            "decode_protected_budget": layers.reduce(0) {
                $0 + (Int(truncating: ($1["decode_protected_budget"] as? NSNumber) ?? 0))
            },
            "decode_protected_slots": layers.reduce(0) {
                $0 + (Int(truncating: ($1["decode_protected_slots"] as? NSNumber) ?? 0))
            },
            "prefill_transient_slots": layers.reduce(0) {
                $0 + (Int(truncating: ($1["prefill_transient_slots"] as? NSNumber) ?? 0))
            },
            "shared_resident_slots": layers.reduce(0) {
                $0 + (Int(truncating: ($1["shared_resident_slots"] as? NSNumber) ?? 0))
            },
            "prefill_requests": layers.reduce(UInt64(0)) {
                $0 + (UInt64(truncating: ($1["prefill_requests"] as? NSNumber) ?? 0))
            },
            "decode_requests": layers.reduce(UInt64(0)) {
                $0 + (UInt64(truncating: ($1["decode_requests"] as? NSNumber) ?? 0))
            },
            "total_evictions": layers.reduce(UInt64(0)) {
                $0 + (UInt64(truncating: ($1["total_evictions"] as? NSNumber) ?? 0))
            },
            "decode_protected_promotions": layers.reduce(UInt64(0)) {
                $0 + (UInt64(truncating: ($1["decode_protected_promotions"] as? NSNumber) ?? 0))
            },
            "decode_protected_demotions": layers.reduce(UInt64(0)) {
                $0 + (UInt64(truncating: ($1["decode_protected_demotions"] as? NSNumber) ?? 0))
            },
            "decode_protected_admission_rejected": layers.reduce(UInt64(0)) {
                $0 + (UInt64(truncating: ($1["decode_protected_admission_rejected"] as? NSNumber) ?? 0))
            },
            "current_request_delta": [
                "evictions": layers.reduce(UInt64(0)) {
                    $0 + (UInt64(truncating: (($1["current_request_delta"] as? [String: Any])?["evictions"] as? NSNumber) ?? 0))
                },
                "prefill_transient_evictions": layers.reduce(UInt64(0)) {
                    $0 + (UInt64(truncating: (($1["current_request_delta"] as? [String: Any])?["prefill_transient_evictions"] as? NSNumber) ?? 0))
                },
                "decode_protected_evictions": layers.reduce(UInt64(0)) {
                    $0 + (UInt64(truncating: (($1["current_request_delta"] as? [String: Any])?["decode_protected_evictions"] as? NSNumber) ?? 0))
                },
                "shared_resident_evictions": layers.reduce(UInt64(0)) {
                    $0 + (UInt64(truncating: (($1["current_request_delta"] as? [String: Any])?["shared_resident_evictions"] as? NSNumber) ?? 0))
                },
                "decode_protected_promotions": layers.reduce(UInt64(0)) {
                    $0 + (UInt64(truncating: (($1["current_request_delta"] as? [String: Any])?["decode_protected_promotions"] as? NSNumber) ?? 0))
                },
                "decode_protected_demotions": layers.reduce(UInt64(0)) {
                    $0 + (UInt64(truncating: (($1["current_request_delta"] as? [String: Any])?["decode_protected_demotions"] as? NSNumber) ?? 0))
                },
                "decode_protected_admission_rejected": layers.reduce(UInt64(0)) {
                    $0 + (UInt64(truncating: (($1["current_request_delta"] as? [String: Any])?["decode_protected_admission_rejected"] as? NSNumber) ?? 0))
                },
                "prefill_shared_resident_promotions": layers.reduce(UInt64(0)) {
                    $0 + (UInt64(truncating: (($1["current_request_delta"] as? [String: Any])?["prefill_shared_resident_promotions"] as? NSNumber) ?? 0))
                },
                "decode_shared_pool_handoff_requests": layers.reduce(UInt64(0)) {
                    $0 + (UInt64(truncating: (($1["current_request_delta"] as? [String: Any])?["decode_shared_pool_handoff_requests"] as? NSNumber) ?? 0))
                },
                "decode_shared_pool_handoff_hits": layers.reduce(UInt64(0)) {
                    $0 + (UInt64(truncating: (($1["current_request_delta"] as? [String: Any])?["decode_shared_pool_handoff_hits"] as? NSNumber) ?? 0))
                },
                "decode_shared_pool_handoff_misses": layers.reduce(UInt64(0)) {
                    $0 + (UInt64(truncating: (($1["current_request_delta"] as? [String: Any])?["decode_shared_pool_handoff_misses"] as? NSNumber) ?? 0))
                },
            ],
            "current_request_raw_counters": [
                "focus_layers": layers.reduce(0) {
                    $0 + (((($1["current_request_focus_layer"] as? Bool) ?? false) ? 1 : 0))
                },
                "tracked_steps": layers.reduce(0) {
                    $0 + (((($1["current_request_raw_counter_steps"] as? [[String: Any]])?.count) ?? 0))
                },
            ],
        ]
        return [
            "timing": [
                "prefill_ms": completion.prefillMs,
                "decode_ms": completion.decodeMs,
                "prefill_breakdown_ms": prefillBreakdownMs as Any,
            ],
            "prefill_expert_access_pattern": prefillExpertAccessPattern as Any,
            "routed_expert_cache": [
                "aggregate": aggregate,
                "layers": layers,
            ],
        ]
    }

    private func toolCallObject(_ call: ParsedToolCall) -> [String: Any] {
        [
            "id": call.id,
            "type": "function",
            "function": [
                "name": call.name,
                "arguments": call.argumentsJSON,
            ],
        ]
    }

    private func utf8Fragments(_ text: String, maximumBytes: Int) -> [String] {
        guard !text.isEmpty else { return [""] }
        var result: [String] = []
        var current = ""
        var bytes = 0
        for character in text {
            let size = String(character).utf8.count
            if bytes + size > maximumBytes, !current.isEmpty {
                result.append(current)
                current = ""
                bytes = 0
            }
            current.append(character)
            bytes += size
        }
        if !current.isEmpty { result.append(current) }
        return result
    }
}

private final class ChildChannelRegistry: Sendable {
    private struct State {
        var channels: [ObjectIdentifier: Channel] = [:]
        var tasks: [UUID: Task<Void, Never>] = [:]
        var shuttingDown = false
    }

    private let state = Mutex(State())

    func insert(_ channel: Channel) {
        let shouldClose = state.withLock {
            guard !$0.shuttingDown else { return true }
            $0.channels[ObjectIdentifier(channel)] = channel
            return false
        }
        if shouldClose {
            channel.close(promise: nil)
        }
    }

    func remove(_ channel: Channel) {
        _ = state.withLock {
            $0.channels.removeValue(forKey: ObjectIdentifier(channel))
        }
    }

    func startTask(
        _ operation: @escaping @Sendable () async -> Void
    ) -> Task<Void, Never> {
        state.withLock { state in
            let id = UUID()
            let task = Task { [self] in
                defer {
                    _ = self.state.withLock {
                        $0.tasks.removeValue(forKey: id)
                    }
                }
                await operation()
            }
            state.tasks[id] = task
            if state.shuttingDown {
                task.cancel()
            }
            return task
        }
    }

    func closeAll() async {
        let channels = state.withLock {
            $0.shuttingDown = true
            return Array($0.channels.values)
        }
        for channel in channels {
            try? await channel.close().get()
        }
        let tasks = state.withLock { Array($0.tasks.values) }
        for task in tasks {
            task.cancel()
        }
        for task in tasks {
            await task.value
        }
    }

    var count: Int {
        state.withLock { $0.channels.count }
    }
}

private final class SendableContext: @unchecked Sendable {
    let value: ChannelHandlerContext

    init(_ value: ChannelHandlerContext) {
        self.value = value
    }
}

private final class RequestPhaseState: Sendable {
    private let state = Mutex("accepted")

    var value: String { state.withLock { $0 } }

    func set(_ value: String) {
        state.withLock { $0 = value }
    }
}

private final class StreamState: @unchecked Sendable {
    private let lock = NSLock()
    private var started = false
    private var stopped = false
    private var heartbeat: RepeatedTask?
    private var startFuture: EventLoopFuture<Void>?
    private var toolIndex = 0

    var isStarted: Bool {
        lock.withLock { started }
    }

    func start(eventLoop: EventLoop,
               interval: TimeAmount,
               ping: @escaping @Sendable () -> Void) -> Bool {
        lock.withLock {
            guard !started else { return false }
            started = true
            stopped = false
            startFuture = nil
            heartbeat = eventLoop.scheduleRepeatedTask(
                initialDelay: interval,
                delay: interval) { [weak self] _ in
                    guard self?.shouldPing == true else { return }
                    ping()
                }
            return true
        }
    }

    func setStartFuture(_ future: EventLoopFuture<Void>) {
        lock.withLock { startFuture = future }
    }

    func waitUntilStarted() async throws {
        let future = lock.withLock { startFuture }
        if let future {
            try await future.get()
        }
    }

    private var shouldPing: Bool {
        lock.withLock { started && !stopped }
    }

    func stop() {
        lock.withLock {
            stopped = true
            heartbeat?.cancel()
            heartbeat = nil
        }
    }

    func nextToolIndex() -> Int {
        lock.withLock {
            defer { toolIndex += 1 }
            return toolIndex
        }
    }
}
