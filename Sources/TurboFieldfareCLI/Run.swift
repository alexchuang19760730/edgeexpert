import Foundation
import Metal
import TurboFieldfare

private struct MessageJSON: Decodable {
    let role: String
    let content: String
}

public struct RunResult: Equatable, Sendable {
    public let exitCode: Int32
    public init(exitCode: Int32) { self.exitCode = exitCode }
}

public func run(args: Args,
                stdout: FileHandle = .standardOutput,
                stderr: FileHandle = .standardError) async -> RunResult {
    do {
        let modelURL = URL(fileURLWithPath: args.model)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
        let promptIds: [Int32]
        if let rawPrompt = args.prompt {
            promptIds = tokenizer.encode(rawPrompt, addBOS: true)
        } else if let messagesFile = args.messagesFile {
            let data = try Data(contentsOf: URL(fileURLWithPath: messagesFile),
                                options: [.mappedIfSafe])
            let rows = try JSONDecoder().decode([MessageJSON].self, from: data)
            let messages = try rows.map { row -> GFTokenizer.Message in
                guard let role = GFTokenizer.Role(rawValue: row.role) else {
                    throw GFTokenizerError.invalidChatTemplate("unsupported role \(row.role)")
                }
                return GFTokenizer.Message(role: role, content: row.content)
            }
            let rendered = try tokenizer.applyChatTemplate(messages)
            promptIds = tokenizer.encode(rendered, addBOS: false)
        } else if args.perplexityFile != nil {
            // Perplexity eval feeds its own corpus; no prompt needed.
            promptIds = []
        } else {
            return errored(stderr, "one of --prompt or --messages-file is required", 2)
        }
        guard !promptIds.isEmpty || args.perplexityFile != nil else {
            return errored(stderr, "empty prompt", 2)
        }
        guard promptIds.count < args.maxContext else {
            return errored(
                stderr,
                "context overflow: prompt \(promptIds.count) reaches maxContext \(args.maxContext)",
                2)
        }
        let effectiveMaxNew = min(args.maxNew, args.maxContext - promptIds.count)
        let config = GenerationConfig(
            maxNewTokens: effectiveMaxNew,
            temperature: args.temperature,
            topK: args.topK,
            topP: args.topP,
            repetitionPenalty: args.repetitionPenalty,
            seed: args.seed,
            stopStrings: args.stops,
            extraStopTokens: [])
        // Expert-cache tuning knobs are exposed via environment variables so
        // benchmarks can sweep them without changing the CLI surface.
        let env = ProcessInfo.processInfo.environment
        let slots: Int = {
            guard let raw = env["TURBO_FIELDFARE_EXPERT_SLOTS"],
                  let parsed = Int(raw),
                  RuntimeConfiguration.allowedExpertCacheSlots.contains(parsed) else { return 16 }
            return parsed
        }()
        let readMode: RuntimePrefillExpertReadMode = {
            guard let raw = env["TURBO_FIELDFARE_PREFILL_EXPERT_READ"],
                  let parsed = RuntimePrefillExpertReadMode(rawValue: raw) else { return .baseline }
            return parsed
        }()
        // Concurrent pread depth for expert cache misses. This is the SSD
        // queue depth: decode misses up to topK experts per layer, so a depth
        // below topK leaves the drive under-utilised.
        let readWorkers: Int = {
            guard let raw = env["TURBO_FIELDFARE_EXPERT_READ_WORKERS"],
                  let parsed = Int(raw), parsed > 0 else { return 2 }
            return min(parsed, 64)
        }()
        let runtime = RuntimeConfiguration(
            expertCacheSlots: slots,
            prefillExpertReadMode: readMode,
            forceLogitsHead: args.perplexityFile != nil || !config.isPureGreedy)

        guard MTLCreateSystemDefaultDevice() != nil else {
            return errored(stderr, "no Metal device", 1)
        }
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            prefillExpertBoundedParallelMissReadWorkers: readWorkers,
            integrityPolicy: args.trustReceipt ? .sizeCheckTrustedReceipt
                                               : .fullSha256)
        let runner = try RealForwardRunner(
            model: model,
            context: context,
            maxContext: args.maxContext,
            runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize)

        if let corpusPath = args.perplexityFile {
            let vocab = model.config.vocabSize
            guard let evalLogits = context.device.makeBuffer(
                length: vocab * MemoryLayout<Float16>.size,
                options: .storageModeShared) else {
                stderr.write(Data("[perplexity] could not allocate logits buffer\n".utf8))
                return RunResult(exitCode: 1)
            }
            return try await runPerplexityEval(corpusPath: corpusPath,
                                               runner: runner,
                                               tokenizer: tokenizer,
                                               logits: evalLogits,
                                               vocab: vocab,
                                               stderr: stderr)
        }

        // If MTP model is specified, run speculative decode loop
        if let mtpPath = args.mtpModel {
            let mtpURL = URL(fileURLWithPath: mtpPath)
            let drafter = try GemmaAssistantMetalDrafter(
                modelDirectory: mtpURL, context: context)
            let mtpStats = try await runRawCompletionWithMTP(
                producer: runner,
                tokenizer: tokenizer,
                promptIds: promptIds,
                config: config,
                context: context,
                scratch: scratch,
                prefillConfig: runtime.prefillConfig,
                drafter: drafter,
                maxDraftTokens: args.mtpMaxDraft) { progress in
                switch progress {
                case .prefill:
                    break
                case .token(_, _, let delta):
                    if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
                case .tail(let tail):
                    stdout.write(Data(tail.utf8))
                }
            }
            if !args.quiet {
                let tps = mtpStats.decodeSeconds > 0
                    ? Double(mtpStats.newTokens) / mtpStats.decodeSeconds : 0
                let accRate = mtpStats.mtpDrafted > 0
                    ? Double(mtpStats.mtpAccepted) / Double(mtpStats.mtpDrafted) * 100 : 0
                var footer = "\n[stop=\(String(describing: mtpStats.reason)) prefill=\(mtpStats.prefillTokens)tok new=\(mtpStats.newTokens)tok ttft=\(String(format: "%.2f", mtpStats.prefillSeconds))s decode=\(String(format: "%.2f", mtpStats.decodeSeconds))s tok/s=\(String(format: "%.3f", tps)) mtp=\(mtpStats.mtpAccepted)/\(mtpStats.mtpDrafted)(\(String(format: "%.0f", accRate))%)"
                if let ad = mtpStats.adaptive {
                    footer += " adaptive(d=\(ad.stepsAtMaxDraft) red=\(ad.stepsReduced) off=\(ad.stepsDisabled) row=\(String(format: "%.2f", ad.rowShare)) acc=\(String(format: "%.0f", ad.rollingAcceptance * 100))%)"
                }
                footer += "]\n"
                stderr.write(Data(footer.utf8))
            }
            finalizeDiagnostics(model: model,
                                runner: runner,
                                stderr: stderr,
                                decodeSeconds: mtpStats.decodeSeconds,
                                totalSeconds: mtpStats.prefillSeconds + mtpStats.decodeSeconds)
            return RunResult(exitCode: 0)
        }

        let stats = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: promptIds,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: runtime.prefillConfig) { progress in
                switch progress {
                case .prefill:
                    break
                case .token(_, _, let delta):
                    if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
                case .tail(let tail):
                    stdout.write(Data(tail.utf8))
                }
            }

        if !args.quiet {
            let tokensPerSecond = stats.decodeSeconds > 0
                ? Double(stats.newTokens) / stats.decodeSeconds
                : 0
            let footer = "\n[stop=\(String(describing: stats.reason)) prefill=\(stats.prefillTokens)tok new=\(stats.newTokens)tok ttft=\(String(format: "%.2f", stats.prefillSeconds))s decode=\(String(format: "%.2f", stats.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
            stderr.write(Data(footer.utf8))
        }
        finalizeDiagnostics(model: model,
                            runner: runner,
                            stderr: stderr,
                            decodeSeconds: stats.decodeSeconds,
                            totalSeconds: stats.prefillSeconds + stats.decodeSeconds)
        return RunResult(exitCode: 0)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}

/// Emits opt-in expert-cache diagnostics and drains any pending access trace.
///
/// Both are off by default: `TURBO_FIELDFARE_EXPERT_STATS` prints a hit/miss
/// summary aggregated across every layer streamer, and the trace only holds
/// buffered data when `TURBO_FIELDFARE_EXPERT_TRACE` named an output file.
private func finalizeDiagnostics(model: Model,
                                 runner: RealForwardRunner? = nil,
                                 stderr: FileHandle,
                                 decodeSeconds: Double = 0,
                                 totalSeconds: Double = 0) {
    ExpertAccessTrace.shared.flush()
    guard ProcessInfo.processInfo.environment["TURBO_FIELDFARE_EXPERT_STATS"] != nil else {
        return
    }

    var prefillRequests: UInt64 = 0
    var prefillHits: UInt64 = 0
    var decodeRequests: UInt64 = 0
    var decodeHits: UInt64 = 0
    var sharedRequests: UInt64 = 0
    var sharedHits: UInt64 = 0
    var evictions: UInt64 = 0
    var slotCount = 0
    var readWallNanos: UInt64 = 0
    var readBytes: UInt64 = 0

    for entry in model.routedExpertCacheTelemetrySnapshots() {
        let snapshot = entry.snapshot
        prefillRequests += snapshot.totalPrefillRequests
        prefillHits += snapshot.totalPrefillHits
        decodeRequests += snapshot.totalDecodeRequests
        decodeHits += snapshot.totalDecodeHits
        sharedRequests += snapshot.totalSharedResidentRequests
        sharedHits += snapshot.totalSharedResidentHits
        evictions += snapshot.totalEvictions
        slotCount = max(slotCount, snapshot.slotCount)
        readWallNanos += snapshot.totalReadWallNanos
        readBytes += snapshot.totalReadBytes
    }

    let totalRequests = prefillRequests + decodeRequests + sharedRequests
    let totalHits = prefillHits + decodeHits + sharedHits
    let totalMisses = totalRequests - totalHits

    func rate(_ hits: UInt64, _ requests: UInt64) -> String {
        guard requests > 0 else { return "n/a" }
        return String(format: "%.1f%%", Double(hits) / Double(requests) * 100)
    }

    var lines = "[expert-cache] slots/layer=\(slotCount)\n"
    lines += "[expert-cache] prefill req=\(prefillRequests) hit=\(prefillHits)"
    lines += " rate=\(rate(prefillHits, prefillRequests))\n"
    lines += "[expert-cache] decode  req=\(decodeRequests) hit=\(decodeHits)"
    lines += " rate=\(rate(decodeHits, decodeRequests))\n"
    lines += "[expert-cache] shared  req=\(sharedRequests) hit=\(sharedHits)"
    lines += " rate=\(rate(sharedHits, sharedRequests))\n"
    lines += "[expert-cache] total   req=\(totalRequests) hit=\(totalHits)"
    lines += " miss=\(totalMisses) rate=\(rate(totalHits, totalRequests))"
    lines += " evictions=\(evictions)\n"

    // The decisive number: how much of the run is actually blocked on expert
    // reads. Layer streamers execute serially within a forward pass, so summing
    // their read wall time approximates total IO stall.
    let readSeconds = Double(readWallNanos) / 1e9
    let readGiB = Double(readBytes) / 1_073_741_824
    lines += String(format: "[expert-io] readWall=%.3fs bytes=%.2fGiB",
                    readSeconds, readGiB)
    if readSeconds > 0 {
        lines += String(format: " throughput=%.2fGiB/s", readGiB / readSeconds)
    }
    if totalSeconds > 0 {
        lines += String(format: " ofTotal=%.1f%%", readSeconds / totalSeconds * 100)
    }
    if decodeSeconds > 0 {
        lines += String(format: " ofDecode=%.1f%%", readSeconds / decodeSeconds * 100)
    }
    lines += "\n"

    if let runner, let path = runner.unionDumpPath, !runner.unionDumpLines.isEmpty {
        do {
            try runner.unionDumpLines.joined(separator: "\n").write(
                toFile: path, atomically: true, encoding: .utf8)
        } catch {
            // best-effort diagnostic dump
        }
    }
    if let runner, !runner.unionStats.isEmpty {
        var unionLine = "[union-stats]"
        for t in runner.unionStats.keys.sorted() {
            let b = runner.unionStats[t]!
            let uRate = b.unionTotal > 0 ? Double(b.unionPool) / Double(b.unionTotal) : 0
            let pRate = b.perTokenTotal > 0 ? Double(b.perTokenPool) / Double(b.perTokenTotal) : 0
            unionLine += String(format: " t=%d n=%d unionCov=%.1f%% perTokCov=%.1f%%",
                                t, b.visits, uRate * 100, pRate * 100)
        }
        lines += unionLine + "\n"
    }

    // Real residency (pool + LRU combined) per batch size, from the plan
    // itself. This is the decisive adaptive-pool number: union coverage vs
    // the static-pool unionCov above.
    let planResidency = model.routedExpertCachePlanResidency()
    if !planResidency.isEmpty {
        var resLine = "[plan-residency]"
        for batch in planResidency.keys.sorted() {
            let pair = planResidency[batch]!
            let rate = pair.requests > 0
                ? Double(pair.hits) / Double(pair.requests) * 100 : 0
            resLine += String(format: " t=%d req=%d resHit=%.1f%%",
                              batch, pair.requests, rate)
        }
        lines += resLine + "\n"
    }

    // Per-stage CPU wall clock already tracked by the forward runner. `cb1`
    // covers attention+router encode (the GPU round trip is excluded); `io` is
    // the awaited expert fetch; `cb2` is the MoE/dense tail encode; `head` is
    // the LM head. Whatever the four fail to cover is time blocked in
    // `waitUntilCompleted`, i.e. actual GPU execution.
    if let runner {
        let stages: [(String, UInt64)] = [
            ("cb1", runner.totalCb1Nanos),
            ("io", runner.totalIoNanos),
            ("cb2", runner.totalCb2Nanos),
            ("head", runner.totalHeadNanos &+ runner.totalHeadFusedNanos)]
        let stageTotal = stages.reduce(UInt64(0)) { $0 &+ $1.1 }
        var stageLine = "[stage]"
        for (name, nanos) in stages {
            let seconds = Double(nanos) / 1e9
            stageLine += String(format: " %@=%.2fs", name, seconds)
            if totalSeconds > 0 {
                stageLine += String(format: "(%.0f%%)", seconds / totalSeconds * 100)
            }
        }
        if totalSeconds > 0 {
            let gpuWait = totalSeconds - Double(stageTotal) / 1e9
            stageLine += String(format: " gpuWait=%.2fs(%.0f%%) wall=%.2fs",
                                gpuWait, gpuWait / totalSeconds * 100, totalSeconds)
        }
        lines += stageLine + "\n"
        // Per-phase CPU decode-chain breakdown: how the sharedFFN->routed
        // GPU-idle gap (see [gpu-idle-after]) is spent on the CPU side.
        // readback = router index readback, plan = cache-hit classification,
        // io = expert fetch, cb2 = MoE/tail encode. Gated on the decode loop.
        if runner.totalCb1Nanos > 0 || runner.totalIoNanos > 0 {
            let chainStages: [(String, UInt64)] = [
                ("readback", runner.readbackNanos),
                ("plan", runner.planNanos),
                ("io", runner.totalIoNanos),
                ("hitIo", runner.fetchHitOnlyNanos),
                ("missIo", runner.fetchAsyncPlanNanos &+ runner.fetchAsyncNoPlanNanos),
                ("cb2", runner.totalCb2Nanos)]
            let chainTotal = chainStages.reduce(UInt64(0)) { $0 &+ $1.1 }
            var chainLine = "[cpu-chain]"
            for (name, nanos) in chainStages {
                chainLine += String(format: " %@=%.2fs", name, Double(nanos) / 1e9)
            }
            chainLine += String(format: " total=%.2fs", Double(chainTotal) / 1e9)
            // chainWall = cb1 done -> routedCB commit (full serial chain).
            // unaccounted = chainWall - (readback+plan+io+cb2). Meaningful
            // BECAUSE io includes the async fetch suspension on miss layers
            // and chainWall includes the same suspension, so they cancel —
            // the remainder is phase1Hit encode/commit + prefetch drain +
            // RD-advice + lookahead (non-fetch serial work).
            let chainWall = Double(runner.chainWallNanos) / 1e9
            chainLine += String(format: " chainWall=%.2fs", chainWall)
            if chainWall > 0 {
                let lookaheadPct = runner.lookaheadPredictions > 0
                ? Double(runner.lookaheadHits) / Double(runner.lookaheadPredictions) : 0
            chainLine += String(format: " missPrefetch=%llu lookaheadAcc=%.2f drain=%.2fs",
                                runner.missPrefetchDispatches, lookaheadPct,
                                Double(runner.prefetchDrainNanos) / 1e9)
            chainLine += String(format: " unaccounted=%.2fs",
                                    chainWall - Double(chainTotal) / 1e9)
            }
            if runner.cb1WaitNanos > 0 {
                chainLine += String(format: " cb1Wait=%.2fs",
                                    Double(runner.cb1WaitNanos) / 1e9)
            }
            if totalSeconds > 0 {
                chainLine += String(format: " ofWall=%.0f%%",
                                    Double(chainTotal) / 1e9 / totalSeconds * 100)
            }
            lines += chainLine + "\n"
        }
        // GPU-side truth. If `busy` approaches `wall`, no amount of I/O work
        // can help — the drive is already hiding behind the GPU.
        let gpuStages: [(String, UInt64)] = [
            ("attn", runner.gpuAttentionNanos),
            ("routedMoE", runner.gpuRoutedNanos),
            ("sharedFFN", runner.gpuSharedNanos),
            ("phase1Hit", runner.gpuPhase1HitNanos),
            ("head", runner.gpuHeadNanos)]
        let gpuTotal = gpuStages.reduce(UInt64(0)) { $0 &+ $1.1 }
        if gpuTotal > 0 {
            var gpuLine = "[gpu]"
            for (name, nanos) in gpuStages {
                gpuLine += String(format: " %@=%.2fs", name, Double(nanos) / 1e9)
            }
            gpuLine += String(format: " busy=%.2fs", Double(gpuTotal) / 1e9)
            if totalSeconds > 0 {
                gpuLine += String(format: " ofWall=%.0f%%",
                                  Double(gpuTotal) / 1e9 / totalSeconds * 100)
            }
            lines += gpuLine + "\n"
        }
        // Round-trip cost. `blocked` is wall time the CPU sat in
        // waitUntilCompleted; subtracting the GPU busy inside those same
        // buffers leaves `overhead` — submit, schedule and wake latency that
        // only a smaller command-buffer count can remove.
        if runner.cbCommitCount > 0 {
            let cb1Wait = Double(runner.cb1WaitNanos) / 1e9
            let routedWait = Double(runner.routedWaitNanos) / 1e9
            let blocked = cb1Wait + routedWait
            let gpuInBlocked = Double(runner.gpuAttentionNanos &+ runner.gpuRoutedNanos
                &+ runner.gpuSharedNanos &+ runner.gpuPhase1HitNanos) / 1e9
            var syncLine = String(
                format: "[sync] cbs=%llu cb1Wait=%.2fs routedWait=%.2fs blocked=%.2fs",
                runner.cbCommitCount, cb1Wait, routedWait, blocked)
            syncLine += String(format: " sched=%.2fs",
                               Double(runner.gpuSchedGapNanos) / 1e9)
            let overhead = blocked - gpuInBlocked
            if overhead > 0 {
                syncLine += String(format: " overhead=%.2fs(%.0fus/cb)",
                                   overhead,
                                   overhead * 1e6 / Double(runner.cbCommitCount))
            }
            if totalSeconds > 0 {
                syncLine += String(format: " ofWall=%.0f%%", blocked / totalSeconds * 100)
            }
            lines += syncLine + "\n"
        }
        // B4: decompose the cb1 wait (the dominant per-layer blocking wait)
        // into GPU execution vs CPU-side schedule/wake latency.
        if runner.cb1WaitNanos > 0, runner.cb1ScheduleNanos > 0 || runner.cb1WakeNanos > 0 {
            let waitS = Double(runner.cb1WaitNanos) / 1e9
            let gpuS = Double(runner.gpuAttentionNanos) / 1e9
            let schedS = Double(runner.cb1ScheduleNanos) / 1e9
            let wakeS = Double(runner.cb1WakeNanos) / 1e9
            // Remainder = GPU work queued ahead of cb1 (prior layer's routed).
            let otherS = max(0, waitS - gpuS - schedS - wakeS)
            var latLine = String(
                format: "[cb-latency] wait=%.2fs gpu=%.2fs wake=%.2fs sched=%.2fs other=%.2fs",
                waitS, gpuS, wakeS, schedS, otherS)
            if totalSeconds > 0 {
                latLine += String(format: " ofWall=%.0f%%", waitS / totalSeconds * 100)
            }
            lines += latLine + "\n"
        }
        // Where the GPU actually sits idle. Aggregate counters say the GPU is
        // busy ~40% of the wall clock but cannot say whether the missing 60% is
        // thousands of tiny submit gaps or a handful of long stalls. Bucketing
        // the real gaps separates "fewer command buffers" from "fix the
        // pipeline" — they are not the same fix.
        let tl = runner.gpuTimelineAnalysis()
        if !tl.gaps.isEmpty || tl.busy > 0 {
            var buckets = [0, 0, 0, 0]      // <0.2ms, 0.2-1ms, 1-5ms, >5ms
            var bucketTime = [0.0, 0.0, 0.0, 0.0]
            for g in tl.gaps {
                let i = g < 0.0002 ? 0 : (g < 0.001 ? 1 : (g < 0.005 ? 2 : 3))
                buckets[i] += 1
                bucketTime[i] += g
            }
            var tlLine = String(format: "[timeline] gpuBusy=%.2fs gpuIdle=%.2fs gaps=%d",
                                tl.busy, tl.idle, tl.gaps.count)
            let names = ["<0.2ms", "0.2-1ms", "1-5ms", ">5ms"]
            for i in 0..<4 where buckets[i] > 0 {
                tlLine += String(format: " %@:%dx=%.2fs", names[i], buckets[i], bucketTime[i])
            }
            if let biggest = tl.gaps.first {
                let top5 = tl.gaps.prefix(5).reduce(0, +)
                tlLine += String(format: " max=%.1fms top5=%.2fs", biggest * 1000, top5)
            }
            lines += tlLine + "\n"
            // Attribute each idle gap to the stage the GPU had just finished.
            // This is what says whether the CPU stalls after routing (expert
            // I/O), after the head (sampling and the token loop), or elsewhere.
            var idleLine = "[gpu-idle-after]"
            let stageNames = RealForwardRunner.gpuStageNames
            for i in 0..<stageNames.count where tl.gapAfterCount[i] > 0 {
                idleLine += String(format: " %@=%.2fs(%dx,%.1fms avg)",
                                   stageNames[i], tl.gapAfter[i], tl.gapAfterCount[i],
                                   tl.gapAfter[i] * 1000 / Double(tl.gapAfterCount[i]))
            }
            lines += idleLine + "\n"
        }
        // Full span timeline dump (TURBO_FIELDFARE_GPU_TIMELINE_CSV=<path>):
        // start,end,label per GPU command buffer in gpuStartTime order. Lets an
        // external tool bucket the idle gaps, find long-stall contexts, and
        // detect periodicity (per-layer rhythms) without touching the binary.
        if let csvPath = ProcessInfo.processInfo
            .environment["TURBO_FIELDFARE_GPU_TIMELINE_CSV"],
           !runner.gpuSpanStarts.isEmpty {
            let order = (0..<runner.gpuSpanStarts.count).sorted {
                runner.gpuSpanStarts[$0] < runner.gpuSpanStarts[$1]
            }
            var csv = "idx,start_s,end_s,dur_ms,label\n"
            var prevEnd = -1.0
            var gapS = 0.0
            for (j, i) in order.enumerated() {
                let st = runner.gpuSpanStarts[i]
                let en = runner.gpuSpanEnds[i]
                if prevEnd >= 0 {
                    gapS = max(0, st - prevEnd)
                }
                let label = Int(runner.gpuSpanLabels[i]) < RealForwardRunner.gpuStageNames.count
                    ? RealForwardRunner.gpuStageNames[Int(runner.gpuSpanLabels[i])]
                    : "?"
                csv += String(format: "%d,%.6f,%.6f,%.3f,%@,gap=%.3fms\n",
                              j, st, en, (en - st) * 1000, label, gapS * 1000)
                prevEnd = en
            }
            try? Data(csv.utf8).write(to: URL(fileURLWithPath: csvPath))
        }

        if runner.lookaheadPredictions > 0 {
            let rate = Double(runner.lookaheadHits)
                / Double(runner.lookaheadPredictions) * 100
            lines += String(format: "[lookahead] predicted=%llu hit=%llu rate=%.1f%%",
                            runner.lookaheadPredictions, runner.lookaheadHits, rate)
            if runner.prefetchedExperts > 0 {
                lines += String(format: " prefetchRead=%llu drain=%.2fs",
                                runner.prefetchedExperts,
                                Double(runner.prefetchDrainNanos) / 1e9)
            }
            lines += "\n"
        }
    }
    stderr.write(Data(lines.utf8))
}

private func errored(_ stderr: FileHandle, _ message: String, _ code: Int32) -> RunResult {
    stderr.write(Data("error: \(message)\n".utf8))
    return RunResult(exitCode: code)
}




/// Log-perplexity eval. Feeds a UTF-8 corpus through the logits head one token
/// at a time, accumulating the mean negative log-likelihood of the actual next
/// token (plain logsumexp softmax over the raw head logits — no softcap, which
/// is uniform across bit-width variants and therefore fair for comparisons).
/// Sequential produce() is slower than chunked prefill but gives per-position
/// logprobs; produce() waits for GPU completion, so the buffer is safe to read.
private func runPerplexityEval(corpusPath: String,
                               runner: RealForwardRunner,
                               tokenizer: GFTokenizer,
                               logits: MTLBuffer,
                               vocab: Int,
                               stderr: FileHandle) async throws -> RunResult {
    let text = try String(contentsOfFile: corpusPath, encoding: .utf8)
    let ids = tokenizer.encode(text, addBOS: true)
    // Fresh-start contract (parity with runRawCompletion): the runner may
    // carry stale per-row counters / lastGreedyToken from construction.
    runner.reset()
    guard ids.count >= 2 else {
        stderr.write(Data("[perplexity] corpus too short\n".utf8))
        return RunResult(exitCode: 1)
    }
    let t0 = DispatchTime.now().uptimeNanoseconds
    var totalNLL = 0.0
    var count = 0
    var argmaxHits = 0
    var nlls: [Double] = []
    nlls.reserveCapacity(max(ids.count - 1, 1))
    var position = 0
    for i in 0..<(ids.count - 1) {
        try Task.checkCancellation()
        try await runner.produce(token: ids[i], position: position, into: logits)
        position += 1
        let target = Int(ids[i + 1])
        let ptr = logits.contents().assumingMemoryBound(to: Float16.self)
        var maxL = -Float.greatestFiniteMagnitude
        var argmax = 0
        for v in 0..<vocab {
            let l = Float(ptr[v])
            if l > maxL { maxL = l; argmax = v }
        }
        var sum = 0.0
        for v in 0..<vocab {
            sum += Double(exp(Double(Float(ptr[v]) - maxL)))
        }
        let lse = Double(maxL) + log(sum)
        let nll = lse - Double(Float(ptr[target]))
        totalNLL += nll
        nlls.append(nll)
        if argmax == target { argmaxHits += 1 }
        count += 1
    }
    let meanNLL = totalNLL / Double(max(count, 1))
    let ppl = exp(meanNLL)
    let argmaxAcc = Double(argmaxHits) / Double(max(count, 1))
    // Robust stats: the geometric mean is dominated by rare-token outliers, so
    // report median + p95 per-token NLL as well (median ppl is the fair number
    // for bit-width comparisons).
    let sorted = nlls.sorted()
    let medianNLL = sorted[sorted.count / 2]
    let p95NLL = sorted[Int(Double(sorted.count - 1) * 0.95)]
    let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
    let msg = String(format: "[perplexity] tokens=%d vocab=%d ppl=%.4f medPPL=%.2f meanNLL=%.4f medNLL=%.4f p95NLL=%.4f argmaxAcc=%.1f%% elapsed=%.1fs\n",
                     count, vocab, ppl, exp(medianNLL), meanNLL, medianNLL, p95NLL, argmaxAcc * 100, dt)
    stderr.write(Data(msg.utf8))
    return RunResult(exitCode: 0)
}

