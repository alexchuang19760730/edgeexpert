import Foundation
import Metal

/// Adaptive-gate summary for the CLI footer (nil when the gate is disabled).
public struct MTPAdaptiveStats: Sendable {
    public let stepsAtMaxDraft: Int
    public let stepsReduced: Int
    public let stepsDisabled: Int
    public let rowShare: Double
    public let rollingAcceptance: Double

    public init(stepsAtMaxDraft: Int, stepsReduced: Int, stepsDisabled: Int,
                rowShare: Double, rollingAcceptance: Double) {
        self.stepsAtMaxDraft = stepsAtMaxDraft
        self.stepsReduced = stepsReduced
        self.stepsDisabled = stepsDisabled
        self.rowShare = rowShare
        self.rollingAcceptance = rollingAcceptance
    }
}

public struct MTPDecodeResult: Sendable {
    public let prefillTokens: Int
    public let newTokens: Int
    public let prefillSeconds: Double
    public let decodeSeconds: Double
    public let reason: StopReason
    public let mtpDrafted: Int
    public let mtpAccepted: Int
    public let adaptive: MTPAdaptiveStats?

    public init(prefillTokens: Int, newTokens: Int, prefillSeconds: Double,
                decodeSeconds: Double, reason: StopReason, mtpDrafted: Int, mtpAccepted: Int,
                adaptive: MTPAdaptiveStats? = nil) {
        self.prefillTokens = prefillTokens
        self.newTokens = newTokens
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds
        self.reason = reason
        self.mtpDrafted = mtpDrafted
        self.mtpAccepted = mtpAccepted
        self.adaptive = adaptive
    }
}

/// Speculative decode loop using an MTP assistant drafter.
/// Requires a fused-greedy-head RealForwardRunner and pure-greedy config.
public func runRawCompletionWithMTP(
    producer: RealForwardRunner,
    tokenizer: GFTokenizer,
    promptIds: [Int32],
    config: GenerationConfig,
    context: MetalContext,
    scratch: RawCompletionScratch,
    prefillConfig: PrefillRuntimeConfig = .defaultChunked,
    drafter: any LocalMTPDrafter,
    maxDraftTokens: Int,
    onProgress: (RawDecodeProgress) -> Void
) async throws -> MTPDecodeResult {
    try config.validate()
    guard !promptIds.isEmpty else { throw GeneratorError.emptyPrompt }

    let fusedGreedy = producer.usesFusedGreedyHead
    guard fusedGreedy else {
        throw PrefillError.unsupportedPrefillSeed(
            "MTP speculation requires fused greedy head")
    }
    // Verify scores `current + drafts` as one chunk, so the span must fit both
    // the per-row head's output buffer and a single prefill span.
    let maxVerifySpan = min(RealForwardRunner.maxPerRowGreedyRows, prefillConfig.chunkTokens)
    guard maxDraftTokens >= 0, maxDraftTokens + 1 <= maxVerifySpan else {
        throw PrefillError.chunkedUnsupported(
            "maxDraftTokens \(maxDraftTokens) exceeds verify span capacity \(maxVerifySpan - 1)")
    }

    var detok = GFDetokenizer(tokenizer: tokenizer)
    var history = Array(promptIds)
    history.reserveCapacity(promptIds.count + config.maxNewTokens)

    producer.reset()
    if let ra = producer as? any RequestAwareLogitProducer {
        ra.beginRequest()
        ra.beginPrefillPhase()
    }

    let prefillStart = Date()
    var position = 0
    var prefillSeed: PrefillSeed?

    let result = try await producer.prefillChunked(
        tokens: promptIds[0...],
        startPosition: 0,
        outputMode: .greedyIfAvailable,
        config: prefillConfig,
        into: scratch.logits) { done in
        onProgress(.prefill(done: done, total: promptIds.count))
    }
    position = result.newPosition
    prefillSeed = result.seed

    let decodeStart = Date()
    if let ra = producer as? any RequestAwareLogitProducer {
        ra.beginDecodePhase()
    }

    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var mtpDrafted = 0
    var mtpAccepted = 0
    var currentToken: Int32
    // Set TURBO_FIELDFARE_MTP_DEBUG=1 to dump per-step draft/target token IDs.
    let mtpDebug = ProcessInfo.processInfo.environment["TURBO_FIELDFARE_MTP_DEBUG"] != nil

    // Adaptive gate (TURBO_FIELDFARE_MTP_ADAPTIVE, default ON): picks the
    // per-step draft count from a rolling acceptance window + calibrated
    // cost ratio, and drops to draft 0 (plain greedy) when MTP stops paying
    // on this machine/prompt. Set the env var to 0 for fixed-draft behavior.
    let adaptiveEnabled =
        (ProcessInfo.processInfo.environment["TURBO_FIELDFARE_MTP_ADAPTIVE"] ?? "1") != "0"
    let adaptiveDebug =
        ProcessInfo.processInfo.environment["TURBO_FIELDFARE_MTP_ADAPTIVE_DEBUG"] != nil
    var adaptive: MTPAdaptiveController? =
        adaptiveEnabled ? MTPAdaptiveController(maxDraft: maxDraftTokens) : nil

    // Per-phase timing accumulation (MTP_DEBUG only).
    var phaseDraftSecs = 0.0
    var phaseVerifySecs = 0.0
    var phaseRewindSecs = 0.0
    var mtpLastPhaseEnd: Date?

    // First token from prefill
    if case .greedyToken(let token) = prefillSeed {
        currentToken = Int32(bitPattern: token)
    } else {
        currentToken = Int32(bitPattern: producer.lastGreedyToken)
    }

    while generated < config.maxNewTokens {
        try Task.checkCancellation()

        // Check stop for current token
        if tokenizer.stopTokenIDs.contains(currentToken) || config.extraStopTokens.contains(currentToken) {
            if currentToken == tokenizer.endOfTurnID { reason = .endOfTurn }
            else if currentToken == tokenizer.toolResponseID { reason = .toolCalls }
            else { reason = .eos }
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            break
        }

        let delta = detok.push(currentToken)
        let visible = stopMatcher.push(delta)
        onProgress(.token(index: generated, id: currentToken, delta: visible))
        if stopMatcher.isStopped {
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            reason = .stopString
            break
        }

        generated += 1
        history.append(currentToken)
        if generated >= config.maxNewTokens { reason = .maxTokens; break }

        // Adaptive gate decides the draft count. A budget of 0 means the gate
        // (or the startup calibration) wants a plain-greedy step — route it
        // through the single-token DECODE path (`produce`) instead of the
        // batched verify, so a disabled gate really returns to base speed.
        // Both paths keep the same invariants: the chunk state is clean after
        // verifyBatch (markCommitted), kv.position is rewound to `position`
        // after each verify step (rewindKV), and `produce` checks
        // kv.position == position — so the two can be interleaved freely.
        let stepStart = Date()
        var stepGapMs: Double = 0
        if let lastPhaseEnd = mtpLastPhaseEnd {
            stepGapMs = stepStart.timeIntervalSince(lastPhaseEnd) * 1000
        }
        let draftBudget = adaptive?.draftCountForNextStep() ?? maxDraftTokens
        if draftBudget == 0 {
            try await producer.produce(token: currentToken, position: position,
                                       into: scratch.logits)
            position += 1
            currentToken = Int32(bitPattern: producer.lastGreedyToken)
            adaptive?.report(
                stepSeconds: Date().timeIntervalSince(stepStart),
                accepted: 0, drafted: 0)
            mtpLastPhaseEnd = Date()
            if adaptiveDebug, let ad = adaptive {
                FileHandle.standardError.write(
                    Data("[mtp-adaptive] phase=\(ad.phase) draft=0 (decode path) step=\(String(format: "%.1f", stepStart.timeIntervalSinceNow * -1000))ms gap=\(String(format: "%.0f", stepGapMs))ms\n".utf8))
            }
            continue
        }
        let bridge = producer.makeBridgeSnapshot(currentToken: currentToken)
        let drafts = try drafter.draftTokens(
            bridgeSnapshot: bridge, history: history,
            currentToken: currentToken, maxDraftTokens: draftBudget,
            // Chained drafts need the target's embedding for each freshly
            // drafted token; passing nil silently degraded the fusion input.
            targetTokenEmbeddingProvider: producer)
        mtpDrafted += drafts.count
        let draftEnd = Date()
        if mtpDebug {
            FileHandle.standardError.write(
                Data("[mtp] ctx=\(currentToken) drafts=\(drafts)\n".utf8))
        }

        // Verify every candidate in ONE forward.
        //
        // The previous loop called `produce()` once per draft, so N accepted
        // drafts cost N+1 target forwards — exactly what plain decoding pays
        // for the same N+1 tokens. Acceptance was high (~88%) yet throughput
        // was *below* baseline, because all speculation bought was the extra
        // drafter cost. Scoring `[current, drafts…]` as a single chunk is what
        // turns acceptance into speed: the chunk shares one pass of routed
        // expert I/O, which dominates runtime.
        //
        // `predictions[i]` is the target's greedy token after `span[0...i]`,
        // so draft `i` is accepted iff `predictions[i] == drafts[i]`, and
        // `predictions[accepted]` is the free bonus token that follows the
        // accepted prefix.
        let span = [currentToken] + drafts
        let predictions = try await producer.verifyBatch(
            tokens: span,
            startPosition: position,
            config: prefillConfig,
            into: scratch.logits)
        guard predictions.count == span.count else {
            throw PrefillError.unsupportedPrefillSeed(
                "verifyBatch returned \(predictions.count) rows for \(span.count) tokens")
        }
        let verifyEnd = Date()

        var accepted = 0
        while accepted < drafts.count,
              Int32(bitPattern: predictions[accepted]) == drafts[accepted] {
            accepted += 1
        }
        mtpAccepted += accepted
        if mtpDebug {
            let verdicts = drafts.enumerated().map { i, d -> String in
                let t = Int32(bitPattern: predictions[i])
                return t == d ? "\(d)=ACCEPT" : "\(d)!=\(t)"
            }.joined(separator: " ")
            FileHandle.standardError.write(
                Data("[mtp]   \(verdicts) -> accepted=\(accepted)/\(drafts.count)\n".utf8))
        }

        // Tokens `span[0...accepted]` are on the real path; the rejected
        // suffix's K/V is dropped by rewinding the cursor, and the drafter
        // has to continue from the last accepted row, not the last row
        // forwarded.
        let committed = position + accepted + 1
        try producer.rewindKV(to: committed)
        position = committed
        try producer.publishHiddenRow(accepted)

        let phaseEnd = Date()
        phaseDraftSecs += draftEnd.timeIntervalSince(stepStart)
        phaseVerifySecs += verifyEnd.timeIntervalSince(draftEnd)
        phaseRewindSecs += phaseEnd.timeIntervalSince(verifyEnd)
        adaptive?.report(
            stepSeconds: phaseEnd.timeIntervalSince(stepStart),
            accepted: accepted, drafted: drafts.count)
        if adaptiveDebug, let ad = adaptive {
            FileHandle.standardError.write(
                Data("[mtp-adaptive] phase=\(ad.phase) draft=\(draftBudget) acc=\(String(format: "%.2f", ad.rollingAcceptance)) row=\(String(format: "%.2f", ad.rowShare)) step=\(String(format: "%.1f", phaseEnd.timeIntervalSince(stepStart) * 1000))ms gap=\(String(format: "%.0f", stepGapMs))ms\n".utf8))
            mtpLastPhaseEnd = phaseEnd
        }

        var emissionStopped = false
        for index in 0..<accepted {
            let token = drafts[index]
            if tokenizer.stopTokenIDs.contains(token) || config.extraStopTokens.contains(token) {
                generated += 1
                history.append(token)
                let d = detok.push(token)
                onProgress(.token(index: generated - 1, id: token, delta: stopMatcher.push(d)))
                if token == tokenizer.endOfTurnID { reason = .endOfTurn }
                else if token == tokenizer.toolResponseID { reason = .toolCalls }
                else { reason = .eos }
                emissionStopped = true
                break
            }
            let d = detok.push(token)
            let v = stopMatcher.push(d)
            onProgress(.token(index: generated, id: token, delta: v))
            generated += 1
            history.append(token)
            if stopMatcher.isStopped { reason = .stopString; emissionStopped = true; break }
            if generated >= config.maxNewTokens { reason = .maxTokens; emissionStopped = true; break }
        }
        if emissionStopped { break }

        currentToken = Int32(bitPattern: predictions[accepted])
    }

    if mtpDebug {
        let total = max(phaseDraftSecs + phaseVerifySecs + phaseRewindSecs, 1e-9)
        let pct = { (x: Double) -> Int in Int((x / total * 100).rounded()) }
        FileHandle.standardError.write(
            Data(String(format: "[mtp] phases draft=%.2fs verify=%.2fs rewind=%.2fs (d=%d%% v=%d%% r=%d%%)\n",
                        phaseDraftSecs, phaseVerifySecs, phaseRewindSecs,
                        pct(phaseDraftSecs), pct(phaseVerifySecs), pct(phaseRewindSecs)).utf8))
    }

    let adaptiveStats = adaptive.map {
        MTPAdaptiveStats(
            stepsAtMaxDraft: $0.stepsAtMaxDraft,
            stepsReduced: $0.stepsReduced,
            stepsDisabled: $0.stepsDisabled,
            rowShare: $0.rowShare,
            rollingAcceptance: $0.rollingAcceptance)
    }
    return MTPDecodeResult(
        prefillTokens: promptIds.count,
        newTokens: generated,
        prefillSeconds: decodeStart.timeIntervalSince(prefillStart),
        decodeSeconds: Date().timeIntervalSince(decodeStart),
        reason: reason,
        mtpDrafted: mtpDrafted,
        mtpAccepted: mtpAccepted,
        adaptive: adaptiveStats)
}
