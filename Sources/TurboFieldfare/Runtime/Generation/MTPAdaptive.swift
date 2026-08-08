import Foundation

/// Per-run adaptive speculative-decode controller.
///
/// Decides the draft count for every verify step from *empirically measured
/// throughput*, so MTP only engages where it actually pays on this machine and
/// this prompt.
///
/// Startup calibration (to defeat the IO-cold-start bias — the first decode
/// steps after prefill run much slower than steady state):
///   1. `warmupSteps` plain-greedy steps (discarded),
///   2. `calibMTPSteps` full-MTP steps — compute-bound (the verify batch shares
///      its expert union), so their wall time is representative even cold, and
///      they warm the expert cache,
///   3. `warmBaselineSteps` plain-greedy steps measured AFTER the MTP steps,
///      when the cache is warm — this is the honest single-token baseline.
///
/// Running policy (decision every `decideEvery` steps, ±`gainHysteresis`
/// band to avoid thrash):
///   - the window rate is EMA-smoothed (`emaAlpha`) to defeat machine noise;
///   - above baseline×1.05 → escalate the draft count (up to `maxDraft`);
///   - below baseline×0.95 → shrink; disabling/ramping-down requires TWO
///     consecutive losing decisions so a single noisy window cannot kill MTP;
///   - inside the band → keep the current draft count.
/// A periodic `baselineProbeEvery` plain-greedy step refreshes the baseline
/// against page-cache/machine drift, and every re-probe resets the rate EMA so
/// revival compares a fresh window (never a stale pre-disable value).
///
/// Draft 0 is handled by the caller passing `maxDraftTokens: 0` to the
/// drafter, which returns no drafts and degrades the verify step to a single
/// forward — no separate non-MTP code path is needed.
public struct MTPAdaptiveController {

    public enum Phase: Sendable, CustomStringConvertible {
        case warmup(remaining: Int)
        case calibratingMTP(remaining: Int)
        case warmBaseline(remaining: Int)
        case running

        public var description: String {
            switch self {
            case .warmup(let r): return "warmup(\(r))"
            case .calibratingMTP(let r): return "calib-mtp(\(r))"
            case .warmBaseline(let r): return "warm-baseline(\(r))"
            case .running: return "running"
            }
        }
    }

    public let maxDraft: Int
    public let windowSize: Int
    public let warmupSteps: Int
    public let calibMTPSteps: Int
    public let warmBaselineSteps: Int
    public let decideEvery: Int
    public let baselineProbeEvery: Int
    public let reprobeEvery: Int
    public let probeDraft: Int
    public let probeSteps: Int
    public let gainHysteresis: Double
    public let hardDisableRatio: Double
    public let emaAlpha: Double

    public private(set) var phase: Phase
    public private(set) var rowShare: Double = 0

    // Rolling acceptance window (informational; last `windowSize` draft steps).
    public private(set) var windowAccepted = 0
    public private(set) var windowDrafted = 0
    private var recent: [(accepted: Int, drafted: Int)] = []

    // Baseline single-token throughput (tokens/sec), warm-measured.
    public private(set) var baselineTokPerSec: Double = 0

    // Throughput window accumulators (reset at each decision).
    private var windowTokens = 0
    private var windowSeconds: Double = 0

    // Calibration accumulators (seeding only).
    private var mtpTotal: Double = 0
    private var mtpSamples = 0
    private var mtpCalibSeen = false
    private var baselineTotal: Double = 0
    private var baselineSamples = 0
    private var baselineMin: Double = 0

    // Policy state.
    private var stepsSinceDecide = 0
    private var stepsSinceBaselineProbe = 0
    private var disabledSteps = 0
    private var probeRemaining = 0
    public private(set) var draftForNext = 0

    // Noise-resistant decision state: the per-window raw rate is EMA-smoothed
    // (machine load swings +/-30% make a single 8-16 step window meaningless),
    // and disabling requires TWO consecutive losing decisions so one noisy
    // window cannot kill MTP on a high-acceptance prompt. rateEMA is reset to
    // zero whenever a probe starts so a revive decision compares a FRESH probe
    // window, never a stale EMA frozen during a long disabled stretch.
    private var rateEMA: Double = 0
    private var losingDecisions = 0

    // Stats for the CLI footer.
    public private(set) var stepsAtMaxDraft = 0
    public private(set) var stepsReduced = 0
    public private(set) var stepsDisabled = 0

    public init(maxDraft: Int,
                windowSize: Int = 64,
                warmupSteps: Int = 3,
                calibMTPSteps: Int = 2,
                warmBaselineSteps: Int = 4,
                decideEvery: Int = 16,
                baselineProbeEvery: Int = 32,
                reprobeEvery: Int = 64,
                probeDraft: Int = 2,
                probeSteps: Int = 8,
                gainHysteresis: Double = 0.05,
                hardDisableRatio: Double = 0.75,
                emaAlpha: Double = 0.35) {
        precondition(maxDraft >= 0, "maxDraft must be >= 0")
        precondition(windowSize > 0, "windowSize must be > 0")
        self.maxDraft = max(1, maxDraft)
        self.windowSize = windowSize
        self.warmupSteps = max(0, warmupSteps)
        self.calibMTPSteps = max(1, calibMTPSteps)
        self.warmBaselineSteps = max(1, warmBaselineSteps)
        self.decideEvery = max(1, decideEvery)
        self.baselineProbeEvery = max(1, baselineProbeEvery)
        self.reprobeEvery = max(1, reprobeEvery)
        self.probeDraft = min(max(1, probeDraft), self.maxDraft)
        self.probeSteps = max(1, probeSteps)
        self.gainHysteresis = min(max(0.0, gainHysteresis), 0.5)
        self.hardDisableRatio = min(max(0.4, hardDisableRatio), 0.95)
        self.emaAlpha = min(max(0.05, emaAlpha), 0.9)
        self.phase = .warmup(remaining: self.warmupSteps)
    }

    /// Draft count to use for the *next* verify step.
    public mutating func draftCountForNextStep() -> Int {
        switch phase {
        case .warmup, .warmBaseline:
            return 0
        case .calibratingMTP:
            return maxDraft
        case .running:
            if probeRemaining > 0 {
                probeRemaining -= 1
                return probeDraft
            }
            // Periodic baseline probe: one plain-greedy step keeps the
            // baseline honest against page-cache/machine drift.
            stepsSinceBaselineProbe += 1
            if stepsSinceBaselineProbe >= baselineProbeEvery {
                stepsSinceBaselineProbe = 0
                return 0
            }
            stepsSinceDecide += 1
            if stepsSinceDecide < decideEvery {
                return draftForNext
            }
            stepsSinceDecide = 0
            decide()
            return draftForNext
        }
    }

    /// Feed the measured result of the step that just ran.
    public mutating func report(stepSeconds: Double, accepted: Int, drafted: Int) {
        switch phase {
        case .warmup(let remaining):
            phase = remaining > 1
                ? .warmup(remaining: remaining - 1)
                : .calibratingMTP(remaining: calibMTPSteps)
        case .calibratingMTP(let remaining):
            // Skip the FIRST sample: the first batched verify after warmup is a
            // cold-start outlier (expert-union IO can cost 7s vs ~100ms steady)
            // and would poison the seed rate/rowShare.
            if drafted > 0, mtpCalibSeen {
                mtpTotal += stepSeconds
                mtpSamples += 1
            }
            if drafted > 0 {
                mtpCalibSeen = true
            }
            phase = remaining > 1
                ? .calibratingMTP(remaining: remaining - 1)
                : .warmBaseline(remaining: warmBaselineSteps)
        case .warmBaseline(let remaining):
            baselineTotal += stepSeconds
            baselineSamples += 1
            if baselineMin == 0 || stepSeconds < baselineMin {
                baselineMin = stepSeconds
            }
            if remaining > 1 {
                phase = .warmBaseline(remaining: remaining - 1)
            } else {
                finalizeCalibration()
                phase = .running
                draftForNext = min(maxDraft, probeDraft)
                stepsSinceDecide = decideEvery
            }
        case .running:
            if drafted > 0 {
                windowAccepted += accepted
                windowDrafted += drafted
                recent.append((accepted, drafted))
                while recent.count > windowSize {
                    let drop = recent.removeFirst()
                    windowAccepted -= drop.accepted
                    windowDrafted -= drop.drafted
                }
            }
            if drafted == 0 {
                // A plain-greedy step is a free baseline refresh. It must NOT
                // enter the MTP throughput window: mixing fast single-token
                // steps into the window would mask a slow draft count and keep
                // the gate engaged when MTP is losing.
                if stepSeconds > 0, baselineTokPerSec > 0 {
                    let singleRate = 1.0 / stepSeconds
                    baselineTokPerSec = emaAlpha * singleRate + (1 - emaAlpha) * baselineTokPerSec
                }
            } else {
                // Throughput window: MTP steps only (tokens per wall second).
                windowTokens += accepted + 1
                windowSeconds += stepSeconds
            }
        }
    }

    public var rollingAcceptance: Double {
        windowDrafted > 0 ? Double(windowAccepted) / Double(windowDrafted) : 0
    }

    // MARK: - Private

    private mutating func decide() {
        // Only MTP steps (d > 0) enter the throughput window.
        let hasMTP = windowSeconds > 0
        let rawRate = hasMTP ? Double(windowTokens) / windowSeconds : 0
        windowTokens = 0
        windowSeconds = 0

        // EMA the measured rate: a single window's raw rate is noise-dominated
        // on a shared machine (+/-30% load swings); the EMA keeps recent
        // history so one bad window cannot look like a real trend.
        if hasMTP, rawRate > 0 {
            if rateEMA <= 0 {
                rateEMA = rawRate
            } else {
                rateEMA = emaAlpha * rawRate + (1 - emaAlpha) * rateEMA
            }
        }
        let rate = rateEMA

        let low = baselineTokPerSec * (1.0 - gainHysteresis)
        let high = baselineTokPerSec * (1.0 + gainHysteresis)

        if draftForNext == 0 {
            // A just-finished probe can revive MTP: if the probe window's MTP
            // rate beats the baseline, re-engage at the probe draft count.
            // rateEMA was zeroed when the probe started, so `rate` here is the
            // fresh probe-window EMA, not stale pre-disable history.
            if hasMTP, rate > high {
                draftForNext = probeDraft
                stepsReduced += 1
                disabledSteps = 0
                return
            }
            // Disabled: keep probing on schedule even when no MTP ran.
            stepsDisabled += 1
            disabledSteps += 1
            if disabledSteps >= reprobeEvery {
                disabledSteps = 0
                probeRemaining = probeSteps
                // Fresh probe window: drop the stale EMA so the revive
                // decision at the next decide() is seeded from current samples.
                rateEMA = 0
            }
            return
        }

        guard hasMTP, baselineTokPerSec > 0 else { return }
        let hardLow = baselineTokPerSec * (1.0 - 2.0 * gainHysteresis)
        // Clearly losing by a LARGE margin: disable immediately (no two-
        // decision grace). Waiting for two 16-step windows burns ~13s of
        // generation at MTP speed when the verdict is already obvious.
        if rate < baselineTokPerSec * hardDisableRatio {
            draftForNext = 0
            stepsDisabled += 1
            disabledSteps = 0
            losingDecisions = 0
            return
        }
        if rate < hardLow {
            // Clearly losing: require TWO consecutive losing decisions before
            // disabling, so a single noisy window (e.g. an iCloud spike) cannot
            // kill MTP on a prompt whose acceptance is actually healthy.
            losingDecisions += 1
            if losingDecisions >= 2 {
                draftForNext = 0
                stepsDisabled += 1
                disabledSteps = 0
                losingDecisions = 0
            }
        } else if rate > high, draftForNext < maxDraft {
            losingDecisions = 0
            draftForNext += 1
            stepsAtMaxDraft += (draftForNext == maxDraft) ? 1 : 0
            stepsReduced += (draftForNext < maxDraft) ? 1 : 0
            disabledSteps = 0
        } else if rate < low {
            losingDecisions += 1
            if losingDecisions >= 2 {
                if draftForNext > 1 {
                    draftForNext -= 1
                    stepsReduced += 1
                } else {
                    draftForNext = 0
                    stepsDisabled += 1
                    disabledSteps = 0
                }
                losingDecisions = 0
            }
        } else {
            // Inside the hysteresis band: hold the current draft count.
            losingDecisions = 0
        }
    }

    private mutating func finalizeCalibration() {
        // Warm single-token baseline measured AFTER the MTP steps warmed the
        // expert cache — this is the honest steady-state single-step rate.
        // Use the MIN of the samples: in the MTP loop the d=0 steps degrade
        // after a cold verify batch (loop-context noise), so the mean is a
        // ~2x under-estimate of the true base rate. The min matches the real
        // RawCompletion decode rate; a too-slow baseline is exactly what kept
        // drafts on when MTP was losing.
        let singleRate = baselineMin > 0 ? 1.0 / baselineMin : 0
        if singleRate > 0 { baselineTokPerSec = singleRate }
        let singleAvg = baselineSamples > 0 ? baselineTotal / Double(baselineSamples) : 0
        guard mtpTotal > 0, mtpSamples > 0, singleAvg > 0 else {
            rowShare = 0.6
            return
        }
        let mtpAvg = mtpTotal / Double(mtpSamples)
        let rMax = mtpAvg / singleAvg
        // R(d) = 1 + d·rowShare with R(maxDraft) ≈ rMax; informational only.
        rowShare = max(0.1, min(1.5, (rMax - 1.0) / Double(maxDraft)))
    }
}
