import Foundation

public enum ServerExpertStreamingMode: String, Sendable, Equatable {
    case mmap
    case pread
}

public enum ServerPDServiceMode: String, Sendable, Equatable {
    case on
    case off
}

public enum ServerPromptCachePrimingMode: String, Sendable, Equatable {
    case off
    case sync
}

public enum ServerStickyQuotaMode: String, Sendable, Equatable {
    case on
    case off
}

public enum ServerDiagnosticsMode: String, Sendable, Equatable {
    case on
    case off
}

public enum ServerPrefillExpertReadMode: String, Sendable, Equatable {
    case baseline
    case coalesced
    case layerLocalReadahead = "layer-local-readahead"
}

public enum ServerRuntimeProfile: String, Sendable, Equatable {
    case current
    case v1Compat = "v1-compat"

    func applyDefaults(expertStreamingMode: inout ServerExpertStreamingMode,
                       pdServiceMode: inout ServerPDServiceMode,
                       promptCacheMode: inout ServerPromptCacheMode,
                       promptCachePrimingMode: inout ServerPromptCachePrimingMode,
                       stickyQuotaMode: inout ServerStickyQuotaMode,
                       diagnosticsMode: inout ServerDiagnosticsMode) {
        switch self {
        case .current:
            expertStreamingMode = .pread
            pdServiceMode = .on
            promptCacheMode = .singlePrefix
            promptCachePrimingMode = .off
            stickyQuotaMode = .on
            diagnosticsMode = .on
        case .v1Compat:
            expertStreamingMode = .pread
            pdServiceMode = .on
            promptCacheMode = .singlePrefix
            promptCachePrimingMode = .off
            stickyQuotaMode = .off
            diagnosticsMode = .on
        }
    }
}

public struct ServerArguments: Equatable, Sendable {
    public let model: String
    public let port: Int
    public let modelID: String
    public let maxContext: Int
    public let queueLimit: Int
    public let runtimeProfile: ServerRuntimeProfile
    public let expertStreamingMode: ServerExpertStreamingMode
    public let pdServiceMode: ServerPDServiceMode
    public let promptCacheMode: ServerPromptCacheMode
    public let promptCachePrimingMode: ServerPromptCachePrimingMode
    public let stickyQuotaMode: ServerStickyQuotaMode
    public let diagnosticsMode: ServerDiagnosticsMode
    public let prefillExpertReadMode: ServerPrefillExpertReadMode
    public let prefillExpertLayerLocalReadaheadExperts: Int
    public let prefillExpertBoundedCoalescedRunExperts: Int
    public let prefillExpertBoundedParallelMissReadWorkers: Int
    public let prefillExpertTraceOutput: String?

    public static let usage = """
    usage: TurboFieldfareServer --model <completed .gturbo directory> [options]

      --model <dir>          Required model directory.
      --port <1...65535>     Loopback port (default 8080).
      --model-id <id>        API model identifier (default gemma-4-26b-a4b-it).
      --max-context <tokens> 4096, 8192, 16384, 32768, or 65536 (default 16384).
      --queue-limit <count>  Maximum queued requests (default 4).
      --runtime-profile <current|v1-compat>
                             Apply a named runtime profile. Later flags override profile defaults.
      --expert-streaming-mode <mmap|pread>
                             Routed expert backend for A/B runtime comparison (default pread).
      --pd-service-mode <on|off>
                             Enable single-host PD handoff service path (default on).
      --prompt-cache-mode <off|single-prefix>
                             Prompt KV reuse mode (default single-prefix).
      --prompt-cache-priming <off|sync>
                             Prompt continuation priming mode (default off).
      --sticky-quota-mode <on|off>
                             Enable sticky shared-resident and focus-layer quota policy (default on).
      --diagnostics-mode <on|off>
                             Enable raw counters and diagnostics reporting paths (default on).
      --prefill-expert-read-mode <baseline|coalesced|layer-local-readahead>
                              Prefill routed expert read strategy for runtime A/B comparison (default baseline).
      --prefill-expert-layer-local-readahead-experts <count>
                              Expert span for layer-local prefill readahead mode (default 16).
      --prefill-expert-bounded-coalesced-run-experts <count>
                              Max contiguous expert count per bounded coalesced prefill read (default 4).
      --prefill-expert-bounded-parallel-miss-read-workers <count>
                              Max worker count for bounded parallel prefill miss reads (default 2).
      --prefill-expert-trace-output <path>
                             Append request-end prefill expert trace JSONL to this path.
      --help                 Show this help.
    """

    public static func parse(_ input: [String]) throws -> ServerArguments {
        var model: String?
        var port = 8080
        var modelID = "gemma-4-26b-a4b-it"
        var maxContext = 16_384
        var queueLimit = 4
        var runtimeProfile: ServerRuntimeProfile = .current
        var expertStreamingMode: ServerExpertStreamingMode = .pread
        var pdServiceMode: ServerPDServiceMode = .on
        var promptCacheMode: ServerPromptCacheMode = .singlePrefix
        var promptCachePrimingMode: ServerPromptCachePrimingMode = .off
        var stickyQuotaMode: ServerStickyQuotaMode = .on
        var diagnosticsMode: ServerDiagnosticsMode = .on
        var prefillExpertReadMode: ServerPrefillExpertReadMode = .baseline
        var prefillExpertLayerLocalReadaheadExperts = 16
        var prefillExpertBoundedCoalescedRunExperts = 4
        var prefillExpertBoundedParallelMissReadWorkers = 2
        var prefillExpertTraceOutput: String?
        var index = 0
        while index < input.count {
            let flag = input[index]
            if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
            guard index + 1 < input.count else {
                throw ServerArgumentError.invalid("\(flag) requires a value")
            }
            let value = input[index + 1]
            index += 2
            switch flag {
            case "--model":
                model = value
            case "--port":
                guard let parsed = Int(value), (1...65_535).contains(parsed) else {
                    throw ServerArgumentError.invalid("--port must be between 1 and 65535")
                }
                port = parsed
            case "--model-id":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--model-id must not be empty")
                }
                modelID = value
            case "--max-context":
                guard let parsed = Int(value),
                      [4_096, 8_192, 16_384, 32_768, 65_536].contains(parsed) else {
                    throw ServerArgumentError.invalid("--max-context is not supported")
                }
                maxContext = parsed
            case "--queue-limit":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid("--queue-limit must be positive")
                }
                queueLimit = parsed
            case "--runtime-profile":
                guard let parsed = ServerRuntimeProfile(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--runtime-profile must be current or v1-compat")
                }
                runtimeProfile = parsed
                parsed.applyDefaults(
                    expertStreamingMode: &expertStreamingMode,
                    pdServiceMode: &pdServiceMode,
                    promptCacheMode: &promptCacheMode,
                    promptCachePrimingMode: &promptCachePrimingMode,
                    stickyQuotaMode: &stickyQuotaMode,
                    diagnosticsMode: &diagnosticsMode)
            case "--expert-streaming-mode":
                guard let parsed = ServerExpertStreamingMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--expert-streaming-mode must be mmap or pread")
                }
                expertStreamingMode = parsed
            case "--pd-service-mode":
                guard let parsed = ServerPDServiceMode(rawValue: value) else {
                    throw ServerArgumentError.invalid("--pd-service-mode must be on or off")
                }
                pdServiceMode = parsed
            case "--prompt-cache-mode":
                guard let parsed = ServerPromptCacheMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-mode must be off or single-prefix")
                }
                promptCacheMode = parsed
            case "--prompt-cache-priming":
                guard let parsed = ServerPromptCachePrimingMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-priming must be off or sync")
                }
                promptCachePrimingMode = parsed
            case "--sticky-quota-mode":
                guard let parsed = ServerStickyQuotaMode(rawValue: value) else {
                    throw ServerArgumentError.invalid("--sticky-quota-mode must be on or off")
                }
                stickyQuotaMode = parsed
            case "--diagnostics-mode":
                guard let parsed = ServerDiagnosticsMode(rawValue: value) else {
                    throw ServerArgumentError.invalid("--diagnostics-mode must be on or off")
                }
                diagnosticsMode = parsed
            case "--prefill-expert-read-mode":
                guard let parsed = ServerPrefillExpertReadMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--prefill-expert-read-mode must be baseline, coalesced, or layer-local-readahead")
                }
                prefillExpertReadMode = parsed
            case "--prefill-expert-layer-local-readahead-experts":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid(
                        "--prefill-expert-layer-local-readahead-experts must be positive")
                }
                prefillExpertLayerLocalReadaheadExperts = parsed
            case "--prefill-expert-bounded-coalesced-run-experts":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid(
                        "--prefill-expert-bounded-coalesced-run-experts must be positive")
                }
                prefillExpertBoundedCoalescedRunExperts = parsed
            case "--prefill-expert-bounded-parallel-miss-read-workers":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid(
                        "--prefill-expert-bounded-parallel-miss-read-workers must be positive")
                }
                prefillExpertBoundedParallelMissReadWorkers = parsed
            case "--prefill-expert-trace-output":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid(
                        "--prefill-expert-trace-output must not be empty")
                }
                prefillExpertTraceOutput = value
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        guard let model else { throw ServerArgumentError.invalid("--model is required") }
        return ServerArguments(model: model,
                               port: port,
                               modelID: modelID,
                               maxContext: maxContext,
                               queueLimit: queueLimit,
                               runtimeProfile: runtimeProfile,
                               expertStreamingMode: expertStreamingMode,
                               pdServiceMode: pdServiceMode,
                               promptCacheMode: promptCacheMode,
                               promptCachePrimingMode: promptCachePrimingMode,
                               stickyQuotaMode: stickyQuotaMode,
                                 diagnosticsMode: diagnosticsMode,
                                 prefillExpertReadMode: prefillExpertReadMode,
                                   prefillExpertLayerLocalReadaheadExperts: prefillExpertLayerLocalReadaheadExperts,
                                   prefillExpertBoundedCoalescedRunExperts: prefillExpertBoundedCoalescedRunExperts,
                                   prefillExpertBoundedParallelMissReadWorkers: prefillExpertBoundedParallelMissReadWorkers,
                                   prefillExpertTraceOutput: prefillExpertTraceOutput)
    }
}

public enum ServerArgumentError: Error, Equatable, CustomStringConvertible {
    case help
    case invalid(String)

    public var description: String {
        switch self {
        case .help: "help"
        case .invalid(let message): message
        }
    }
}
