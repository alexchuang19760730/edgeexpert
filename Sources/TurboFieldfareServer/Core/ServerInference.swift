import CryptoKit
import Foundation
import TurboFieldfare

public enum ServerInferenceEvent: Equatable, Sendable {
    case content(String)
    case toolCall(ParsedToolCall)
}

public struct ServerCompletion: Equatable, Sendable {
    public let content: String
    public let toolCalls: [ParsedToolCall]
    public let finishReason: String
    public let usage: OpenAIUsage
    public let prefillMs: Double
    public let decodeMs: Double
    public let prefillTimingBreakdown: PrefillTimingBreakdown?
    public let prefillExpertAccessPattern: PrefillExpertAccessPattern?
    public let routedExpertCacheTelemetrySnapshots: [RoutedExpertCacheLayerTelemetrySnapshot]

    public init(content: String,
                toolCalls: [ParsedToolCall],
                finishReason: String,
                usage: OpenAIUsage,
                prefillMs: Double,
                decodeMs: Double,
                prefillTimingBreakdown: PrefillTimingBreakdown? = nil,
                 prefillExpertAccessPattern: PrefillExpertAccessPattern? = nil,
                routedExpertCacheTelemetrySnapshots: [RoutedExpertCacheLayerTelemetrySnapshot] = []) {
        self.content = content
        self.toolCalls = toolCalls
        self.finishReason = finishReason
        self.usage = usage
        self.prefillMs = prefillMs
        self.decodeMs = decodeMs
        self.prefillTimingBreakdown = prefillTimingBreakdown
        self.prefillExpertAccessPattern = prefillExpertAccessPattern
        self.routedExpertCacheTelemetrySnapshots = routedExpertCacheTelemetrySnapshots
    }
}

public struct ServerPreparedRequest: Sendable {
    public let request: ValidatedChatRequest
    fileprivate let promptIDs: [Int32]?

    public var promptTokenCount: Int? { promptIDs?.count }

    init(request: ValidatedChatRequest, promptIDs: [Int32]? = nil) {
        self.request = request
        self.promptIDs = promptIDs
    }
}

public protocol ServerInferenceBackend: Sendable {
    func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest
    func generate(_ request: ValidatedChatRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion
    func generate(_ prepared: ServerPreparedRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion
}

public extension ServerInferenceBackend {
    func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest {
        ServerPreparedRequest(request: request)
    }

    func generate(
        _ prepared: ServerPreparedRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        try await generate(prepared.request, onEvent: onEvent)
    }
}

public actor ServerCoordinator {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Error>
    }

    private let queueLimit: Int
    private var admittedCount = 0
    private var active = false
    private var waiters: [Waiter] = []
    private var shuttingDown = false

    public init(queueLimit: Int) {
        self.queueLimit = queueLimit
    }

    public func run<T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await runPreparing(
            onQueued: onQueued,
            prepare: { () },
            operation: { _ in try await operation() })
    }

    func runPreparing<Prepared: Sendable, T: Sendable>(
        onQueued: @escaping @Sendable () -> Void = {},
        prepare: @escaping @Sendable () async throws -> Prepared,
        operation: @escaping @Sendable (Prepared) async throws -> T
    ) async throws -> T {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        guard admittedCount <= queueLimit else { throw ServerRequestError.queueFull }
        admittedCount += 1
        defer { admittedCount -= 1 }

        let prepared = try await prepare()
        try Task.checkCancellation()
        try await acquire(onQueued: onQueued)
        defer { release() }
        return try await operation(prepared)
    }

    private func acquire(onQueued: @escaping @Sendable () -> Void) async throws {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        if !active {
            active = true
            return
        }
        guard waiters.count < queueLimit else { throw ServerRequestError.queueFull }
        onQueued()
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(Waiter(id: id, continuation: continuation))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
        if Task.isCancelled {
            release()
            throw CancellationError()
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }

    private func release() {
        if waiters.isEmpty {
            active = false
        } else {
            waiters.removeFirst().continuation.resume()
        }
    }

    public func shutdown() {
        shuttingDown = true
        let queued = waiters
        waiters.removeAll()
        for waiter in queued {
            waiter.continuation.resume(throwing: CancellationError())
        }
    }

    public var queuedCount: Int { waiters.count }
    public var isActive: Bool { active }
}

public actor ServerModelSession: ServerInferenceBackend {
    private let context: MetalContext
    private let model: Model
    private let tokenizer: GFTokenizer
    private let runner: RealForwardRunner
    private let scratch: RawCompletionScratch
    private let prefillConfig: PrefillRuntimeConfig
    private let maxContext: Int
    private let promptCacheMode: ServerPromptCacheMode
    private let diagnosticsMode: ServerDiagnosticsMode
    private let promptCacheDomain: ServerPromptCacheDomain
    private let prefillExpertTraceWriter: ServerPrefillExpertTraceWriter?
    private var promptCache = ServerPromptCache()

    public static func load(modelDirectory: URL,
                            maxContext: Int,
                            expertStreamingMode: ServerExpertStreamingMode = .pread,
                            pdServiceMode: ServerPDServiceMode = .on,
                            promptCacheMode: ServerPromptCacheMode = .singlePrefix,
                            promptCachePrimingMode: ServerPromptCachePrimingMode = .off,
                            stickyQuotaMode: ServerStickyQuotaMode = .on,
                            diagnosticsMode: ServerDiagnosticsMode = .on,
                            prefillExpertReadMode: ServerPrefillExpertReadMode = .baseline,
                            prefillExpertLayerLocalReadaheadExperts: Int = 16,
                            prefillExpertBoundedCoalescedRunExperts: Int = 4,
                            prefillExpertBoundedParallelMissReadWorkers: Int = 2,
                            prefillExpertTraceOutput: String? = nil) async throws -> ServerModelSession {
        let tokenizerFolder = GFTokenizer.tokenizerFolder(forModelDirectory: modelDirectory)
        guard let tokenizerFolder else {
            throw GFTokenizerError.missingToolTemplate
        }
        let templateURL = tokenizerFolder.appendingPathComponent("chat_template.jinja")
        guard FileManager.default.fileExists(atPath: templateURL.path) else {
            throw GFTokenizerError.missingToolTemplate
        }
        let tokenizer = try await GFTokenizer.load(from: tokenizerFolder)
        let context = try MetalContext()
        let runtime = RuntimeConfiguration(
            prefillExpertReadMode: {
                switch prefillExpertReadMode {
                case .baseline: .baseline
                case .coalesced: .coalesced
                case .layerLocalReadahead: .layerLocalReadahead
                }
            }(),
            prefillExpertLayerLocalReadaheadExperts: prefillExpertLayerLocalReadaheadExperts,
            prefillExpertBoundedCoalescedRunExperts: prefillExpertBoundedCoalescedRunExperts,
            prefillExpertBoundedParallelMissReadWorkers: prefillExpertBoundedParallelMissReadWorkers,
            forceLogitsHead: true)
        let effectiveStreamingMode: ExpertStreamingMode
        switch expertStreamingMode {
        case .pread:
            effectiveStreamingMode = .pread(slotCount: runtime.expertCacheSlots)
        case .mmap:
            throw ServerRequestError.invalid(
                message: "expert streaming mode mmap is not supported on this baseline",
                param: "expert_streaming_mode",
                code: "unsupported_value")
        }
        let model = try Model.load(
            directoryURL: modelDirectory,
            device: context.device,
            streamingMode: effectiveStreamingMode,
            expertCachePolicy: runtime.modelExpertCachePolicy,
            prefillExpertReadMode: runtime.prefillExpertReadMode,
            prefillExpertLayerLocalReadaheadExperts: runtime.prefillExpertLayerLocalReadaheadExperts,
            prefillExpertBoundedCoalescedRunExperts: runtime.prefillExpertBoundedCoalescedRunExperts,
            prefillExpertBoundedParallelMissReadWorkers: runtime.prefillExpertBoundedParallelMissReadWorkers,
            integrityPolicy: .fullSha256)
        let runner = try RealForwardRunner(model: model,
                                           context: context,
                                           maxContext: maxContext,
                                           runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize)
        let templateDigest = SHA256.hash(data: try Data(contentsOf: templateURL))
            .map { String(format: "%02x", $0) }
            .joined()
        let runtimeIdentity = [
            String(runtime.expertCacheSlots),
            runtime.expertCachePolicy.rawValue,
            runtime.rdadvisePolicy.rawValue,
            runtime.prefillPolicy.rawValue,
            String(runtime.prefillChunkTokens),
            runtime.headPath.rawValue,
            "expert_streaming=\(expertStreamingMode.rawValue)",
            "pd_service=\(pdServiceMode.rawValue)",
            "prompt_cache=\(promptCacheMode.rawValue)",
            "prompt_cache_priming=\(promptCachePrimingMode.rawValue)",
            "sticky_quota=\(stickyQuotaMode.rawValue)",
            "diagnostics=\(diagnosticsMode.rawValue)",
            "prefill_expert_read=\(prefillExpertReadMode.rawValue)",
            "prefill_expert_readahead_experts=\(prefillExpertLayerLocalReadaheadExperts)",
            "prefill_expert_bounded_run=\(prefillExpertBoundedCoalescedRunExperts)",
            "prefill_expert_bounded_workers=\(prefillExpertBoundedParallelMissReadWorkers)",
        ].joined(separator: ":")
        let runtimeDigest = SHA256.hash(data: Data(runtimeIdentity.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        let promptCacheDomain = ServerPromptCacheDomain(
            modelID: model.modelID,
            sourceSnapshotHash: model.sourceSnapshotHash,
            runtimeProfileHash: runtimeDigest,
            maximumContext: maxContext,
            kvStorage: PrefillKVStorageMode.fp16.rawValue,
            fp16RingEnabled: runtime.fp16RingEnabled,
            templateSHA256: templateDigest)
        let prefillExpertTraceWriter: ServerPrefillExpertTraceWriter?
        if let prefillExpertTraceOutput {
            prefillExpertTraceWriter = try ServerPrefillExpertTraceWriter(path: prefillExpertTraceOutput)
        } else {
            prefillExpertTraceWriter = nil
        }
        return ServerModelSession(context: context,
                                  model: model,
                                  tokenizer: tokenizer,
                                  runner: runner,
                                  scratch: scratch,
                                  prefillConfig: runtime.prefillConfig,
                                  maxContext: maxContext,
                                  promptCacheMode: promptCacheMode,
                                  diagnosticsMode: diagnosticsMode,
                                  promptCacheDomain: promptCacheDomain,
                                  prefillExpertTraceWriter: prefillExpertTraceWriter)
    }

    private init(context: MetalContext,
                 model: Model,
                 tokenizer: GFTokenizer,
                 runner: RealForwardRunner,
                 scratch: RawCompletionScratch,
                 prefillConfig: PrefillRuntimeConfig,
                 maxContext: Int,
                 promptCacheMode: ServerPromptCacheMode,
                 diagnosticsMode: ServerDiagnosticsMode,
                 promptCacheDomain: ServerPromptCacheDomain,
                 prefillExpertTraceWriter: ServerPrefillExpertTraceWriter?) {
        self.context = context
        self.model = model
        self.tokenizer = tokenizer
        self.runner = runner
        self.scratch = scratch
        self.prefillConfig = prefillConfig
        self.maxContext = maxContext
        self.promptCacheMode = promptCacheMode
        self.diagnosticsMode = diagnosticsMode
        self.promptCacheDomain = promptCacheDomain
        self.prefillExpertTraceWriter = prefillExpertTraceWriter
    }

    public func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let prepared = try await prepare(request)
        return try await generate(prepared, onEvent: onEvent)
    }

    public func prepare(_ request: ValidatedChatRequest) async throws -> ServerPreparedRequest {
        ServerPreparedRequest(request: request, promptIDs: try renderPrompt(request))
    }

    public func generate(
        _ prepared: ServerPreparedRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let request = prepared.request
        var completed = false
        defer {
            if !completed {
                promptCache.invalidate()
                runner.reset()
            }
        }
        let needsToolTemplate = usesToolTemplate(request)
        let promptIDs = try prepared.promptIDs ?? renderPrompt(request)

        let effectivePromptIDs: [Int32]
        let completionStart: RawCompletionStart
        if promptCacheMode == .singlePrefix {
            switch promptCache.match(
                domain: promptCacheDomain,
                request: request,
                renderedPromptIDs: promptIDs,
                tokenizer: tokenizer) {
            case .miss:
                promptCache.invalidate()
                effectivePromptIDs = promptIDs
                completionStart = .reset
            case .hit(let effective, let cached):
                effectivePromptIDs = effective
                completionStart = .resume(cachedPromptTokens: cached)
            }
        } else {
            promptCache.invalidate()
            effectivePromptIDs = promptIDs
            completionStart = .reset
        }
        guard effectivePromptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "effective prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }

        var config = request.generationConfig
        config.maxNewTokens = min(
            request.maximumCompletionTokens,
            maxContext - effectivePromptIDs.count)
        config.stopStrings = []

        let decoder = needsToolTemplate
            ? StructuredAssistantDecoder(
                tokenizer: tokenizer,
                allowedTools: Set(request.tools.map(\.name)))
            : nil
        var stopMatcher = StreamingStopMatcher(stops: request.generationConfig.stopStrings)
        var content = ""
        var calls: [ParsedToolCall] = []
        var decodingError: Error?
        var shouldStop = false
        let prefillTimingBaseline = diagnosticsMode == .on
            ? runner.prefillTimingBreakdownSnapshot
            : .zero

        let result = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: effectivePromptIDs,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: prefillConfig,
            start: completionStart,
            shouldStop: { shouldStop }) { progress in
                guard decodingError == nil else { return }
                do {
                    switch progress {
                    case .prefill:
                        break
                    case .token(_, let tokenID, let delta):
                        let events = if let decoder {
                            try decoder.consume(tokenID: tokenID, delta: delta)
                        } else {
                            delta.isEmpty ? [] : [StructuredAssistantEvent.content(delta)]
                        }
                        for event in events {
                            switch event {
                            case .content(let text):
                                let visible = stopMatcher.push(text)
                                if !visible.isEmpty {
                                    content += visible
                                    onEvent(.content(visible))
                                }
                                if stopMatcher.isStopped { shouldStop = true }
                            case .toolCall(let call):
                                calls.append(call)
                                onEvent(.toolCall(call))
                            }
                        }
                    case .tail(let text):
                        let visible = stopMatcher.push(text)
                        if !visible.isEmpty {
                            content += visible
                            onEvent(.content(visible))
                        }
                    }
                } catch {
                    decodingError = error
                    shouldStop = true
                }
        }
        if let decodingError { throw decodingError }
        try decoder?.finish()
        if needsToolTemplate, result.reason == .toolCalls, calls.isEmpty {
            throw GemmaToolCallParserError.malformed
        }
        let tail = stopMatcher.finish()
        if !tail.isEmpty {
            content += tail
            onEvent(.content(tail))
        }
        let reason: String
        if !calls.isEmpty {
            reason = "tool_calls"
        } else if result.reason == .maxTokens {
            reason = "length"
        } else {
            reason = "stop"
        }
        if promptCacheMode == .singlePrefix {
            promptCache.publish(
                domain: promptCacheDomain,
                request: request,
                content: content,
                calls: calls,
                result: result,
                stopStringFiltered: stopMatcher.isStopped)
        }
        completed = true
          let prefillTimingBreakdown = diagnosticsMode == .on
              ? runner.prefillTimingBreakdownSnapshot.delta(since: prefillTimingBaseline)
              : nil
          let prefillExpertAccessPattern = diagnosticsMode == .on
              ? runner.prefillExpertAccessPatternSnapshot
              : nil
        let prefillExpertTrace = runner.prefillExpertTraceSnapshot
        let completion = ServerCompletion(
            content: content,
            toolCalls: calls,
            finishReason: reason,
            usage: OpenAIUsage(promptTokens: result.prefillTokens,
                               completionTokens: result.newTokens,
                               totalTokens: result.prefillTokens + result.newTokens,
                               cachedTokens: result.cachedPromptTokens),
            prefillMs: result.prefillSeconds * 1000.0,
            decodeMs: result.decodeSeconds * 1000.0,
                prefillTimingBreakdown: prefillTimingBreakdown,
              prefillExpertAccessPattern: prefillExpertAccessPattern,
            routedExpertCacheTelemetrySnapshots: diagnosticsMode == .on
                ? result.routedExpertCacheTelemetrySnapshots
                : [])
        if let prefillExpertTraceWriter {
            try await prefillExpertTraceWriter.append(
                modelID: model.modelID,
                sourceSnapshotHash: model.sourceSnapshotHash,
                runtimeProfileHash: promptCacheDomain.runtimeProfileHash,
                request: request,
                renderedPromptIDs: promptIDs,
                effectivePromptIDs: effectivePromptIDs,
                completion: completion,
                trace: prefillExpertTrace)
        }
        return completion
    }

    private func renderPrompt(_ request: ValidatedChatRequest) throws -> [Int32] {
        let promptIDs: [Int32]
        if usesToolTemplate(request) {
            promptIDs = try tokenizer.encodeToolChat(
                messages: request.messages,
                tools: request.tools)
        } else {
            let rendered = try tokenizer.applyChatTemplate(request.messages)
            promptIDs = tokenizer.encode(rendered, addBOS: false)
        }
        guard promptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }
        return promptIDs
    }

    private func usesToolTemplate(_ request: ValidatedChatRequest) -> Bool {
        !request.tools.isEmpty || request.messages.contains {
            $0.role == .developer || $0.role == .tool || !$0.toolCalls.isEmpty
        }
    }
}
