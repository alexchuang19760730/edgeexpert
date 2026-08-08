import Foundation
public struct Args: Equatable, Sendable {
    public var model: String
    public var prompt: String?
    public var messagesFile: String?
    public var maxNew: Int
    public var maxContext: Int
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var seed: UInt64?
    public var stops: [String]
    public var quiet: Bool
    public var mtpModel: String?
    public var mtpMaxDraft: Int
    /// Path to a UTF-8 corpus file for a log-perplexity eval (logits head).
    public var perplexityFile: String?
    /// Re-hash every expert shard on open (`full`), or trust the
    /// `verified-install.json` receipt and only check file sizes (`trusted`).
    /// Full hashing re-reads the entire 13 GB model on every launch, which
    /// costs seconds of TTFT and evicts freshly-read experts from the page
    /// cache, so benchmarks should prefer `trusted`.
    public var trustReceipt: Bool
    /// Requested routed-expert bit width (2|3|4). When 2/3, the `--model` path
    /// is resolved to its `-r2`/`-r3` sibling variant directory.
    public var routedBits: Int?

    public init(model: String,
                prompt: String? = nil,
                messagesFile: String? = nil,
                maxNew: Int = 1_024,
                maxContext: Int = 4096,
                temperature: Float = 0.2,
                topK: Int? = 64,
                topP: Float? = 0.95,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                stops: [String] = [],
                quiet: Bool = false,
                mtpModel: String? = nil,
                mtpMaxDraft: Int = 4,
                perplexityFile: String? = nil,
                trustReceipt: Bool = false,
                routedBits: Int? = nil) {
        self.model = model
        self.prompt = prompt
        self.messagesFile = messagesFile
        self.maxNew = maxNew
        self.maxContext = maxContext
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.stops = stops
        self.quiet = quiet
        self.mtpModel = mtpModel
        self.mtpMaxDraft = mtpMaxDraft
        self.perplexityFile = perplexityFile
        self.trustReceipt = trustReceipt
        self.routedBits = routedBits
    }
}

public enum ArgsError: Error, Equatable, CustomStringConvertible {
    case helpRequested
    case unknownFlag(String)
    case missingValue(flag: String)
    case invalidValue(flag: String, value: String)
    case requiredMissing(String)
    case mutuallyExclusive(String, String)
    case modeMissing
    case variantMissing(base: String, variant: String, bits: Int)
    case variantConflict(variant: String, found: Int?, bits: Int)

    public var description: String {
        switch self {
        case .helpRequested: return "help requested"
        case .unknownFlag(let flag): return "unknown flag: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .invalidValue(let flag, let value): return "invalid value for \(flag): \(value)"
        case .requiredMissing(let flag): return "required flag missing: \(flag)"
        case .mutuallyExclusive(let a, let b): return "\(a) and \(b) are mutually exclusive"
        case .modeMissing: return "one of --prompt or --messages-file is required"
        case .variantMissing(let base, let variant, let bits):
            return """
        no \(bits)-bit routed-expert variant found at: \(variant)
        generate it with:
          TurboFieldfareRebits --input \(base) --output \(variant) --routed-bits \(bits)
        """
        case .variantConflict(let variant, let found, let bits):
            if let found {
                return "\(variant) already exists but declares \(found)-bit routed experts (requested \(bits)-bit). "
                    + "Use a different --output or regenerate it with TurboFieldfareRebits."
            }
            return "\(variant) already exists but has no readable manifest.json, so it cannot be confirmed as \(bits)-bit."
        }
    }
}

extension Args {
    public static let usage = """
    TurboFieldfareCLI — Gemma 4 26B-A4B text generation

    usage: TurboFieldfareCLI --model <dir> (--prompt <string> | --messages-file <path>) [options]

    required:
      --model <dir>             Path to a .gturbo model directory.
      --prompt <string>         Raw-completion prompt.
      --messages-file <path>    JSON chat messages with role and content fields.

    options:
      --max-new <int>           Generated-token limit (default 1024).
      --max-context <int>       Context limit in tokens (default 4096).
      --temperature <float>     Sampling temperature (default 0.2; 0 = greedy).
      --top-k <int>             Top-k truncation, 1...256 (default 64; 0 = off).
      --top-p <float>           Nucleus truncation (default 0.95).
      --repetition-penalty <f>  Repetition penalty (default 1.0).
      --seed <uint64>           Deterministic sampling seed (default off).
      --stop <string>           Stop substring (repeatable).
      --mtp-model <dir>         MTP assistant model dir for speculative decode.
      --mtp-max-draft <int>     Max draft tokens per step (default 4).
      --quiet                   Suppress the timing footer.
      --routed-bits <2|3|4>     Select the routed-expert bit width. With 2 or 3,
                                resolves --model to its sibling variant directory
                                (gemma4.gturbo -> gemma4-r3.gturbo for 3). 4 = use
                                the given model as-is. If the variant is missing,
                                prints the TurboFieldfareRebits command to build it.
      --trust-receipt           Skip full SHA-256 re-hashing of expert shards
                                and trust verified-install.json (size check
                                only). Saves seconds of TTFT per launch.
      --help                    Show this message.
    """

    public static func parse(_ argv: [String]) throws -> Args {
        var model: String?
        var prompt: String?
        var messagesFile: String?
        var maxNew = 1_024
        var maxContext = 4096
        var temperature: Float = 0.2
        var topK: Int? = 64
        var topP: Float? = 0.95
        var repetitionPenalty: Float = 1.0
        var seed: UInt64?
        var stops: [String] = []
        var quiet = false
        var mtpModel: String?
        var mtpMaxDraft = 4
        var perplexityFile: String? = nil
        var trustReceipt = false
        var routedBits: Int?

        var index = 0
        while index < argv.count {
            let flag = argv[index]
            switch flag {
            case "--help":
                throw ArgsError.helpRequested
            case "--quiet":
                quiet = true
                index += 1
            case "--trust-receipt":
                trustReceipt = true
                index += 1
            case "--model":
                model = try takeValue(argv, &index, flag: flag)
            case "--prompt":
                prompt = try takeValue(argv, &index, flag: flag)
            case "--messages-file":
                messagesFile = try takeValue(argv, &index, flag: flag)
            case "--max-new":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxNew = parsed
            case "--max-context":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                maxContext = parsed
            case "--temperature":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed >= 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                temperature = parsed
            case "--top-k":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), (0...256).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topK = parsed == 0 ? nil : parsed
            case "--top-p":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed > 0, parsed <= 1 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                topP = parsed
            case "--repetition-penalty":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Float(value), parsed > 0 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                repetitionPenalty = parsed
            case "--seed":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = UInt64(value) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                seed = parsed
            case "--stop":
                stops.append(try takeValue(argv, &index, flag: flag))
            case "--perplexity":
                perplexityFile = try takeValue(argv, &index, flag: flag)
            case "--mtp-model":
                mtpModel = try takeValue(argv, &index, flag: flag)
            case "--routed-bits":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), (2...4).contains(parsed) else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                routedBits = parsed
            case "--mtp-max-draft":
                let value = try takeValue(argv, &index, flag: flag)
                guard let parsed = Int(value), parsed > 0, parsed <= 32 else {
                    throw ArgsError.invalidValue(flag: flag, value: value)
                }
                mtpMaxDraft = parsed
            default:
                throw ArgsError.unknownFlag(flag)
            }
        }

        guard let model else { throw ArgsError.requiredMissing("--model") }
        let resolvedModel = try Self.resolveRoutedBits(base: model, bits: routedBits)
        if prompt != nil && messagesFile != nil {
            throw ArgsError.mutuallyExclusive("--prompt", "--messages-file")
        }
        if perplexityFile != nil && (prompt != nil || messagesFile != nil) {
            throw ArgsError.mutuallyExclusive("--perplexity", "--prompt/--messages-file")
        }
        if prompt == nil && messagesFile == nil && perplexityFile == nil {
            throw ArgsError.modeMissing
        }
        if temperature > 0, topK == nil, let topP, topP < 1 {
            throw ArgsError.invalidValue(
                flag: "--top-p",
                value: "\(topP) requires --top-k between 1 and 256")
        }
        return Args(model: resolvedModel,
                    prompt: prompt,
                    messagesFile: messagesFile,
                    maxNew: maxNew,
                    maxContext: maxContext,
                    temperature: temperature,
                    topK: topK,
                    topP: topP,
                    repetitionPenalty: repetitionPenalty,
                    seed: seed,
                    stops: stops,
                    quiet: quiet,
                    mtpModel: mtpModel,
                    mtpMaxDraft: mtpMaxDraft,
                    perplexityFile: perplexityFile,
                    trustReceipt: trustReceipt,
                    routedBits: routedBits)
    }

    /// Resolve `--routed-bits`: 4 keeps the base model as-is; 2/3 point at the
    /// `-r2`/`-r3` sibling variant directory. Idempotent when the given path is
    /// already a variant declaring the requested bits.
    private static func resolveRoutedBits(base: String, bits: Int?) throws -> String {
        guard let bits else { return base }
        if bits == 4 { return base }
        if manifestWeightBits(at: base) == bits { return base }
        let variant = variantPath(for: base, bits: bits)
        if let found = manifestWeightBits(at: variant), found == bits { return variant }
        if FileManager.default.fileExists(atPath: variant) {
            throw ArgsError.variantConflict(variant: variant,
                                            found: manifestWeightBits(at: variant),
                                            bits: bits)
        }
        throw ArgsError.variantMissing(base: base, variant: variant, bits: bits)
    }

    private static func variantPath(for base: String, bits: Int) -> String {
        let url = URL(fileURLWithPath: base)
        var stem = url.deletingPathExtension().lastPathComponent
        let ext = url.pathExtension
        // Normalize an existing "-rN" variant suffix so requesting a different
        // bit width maps to the sibling directory (gemma4-r3 -> gemma4-r2),
        // never a stacked "gemma4-r3-r2" name.
        if let range = stem.range(of: #"-r[234]+\z"#, options: .regularExpression) {
            stem = String(stem[..<range.lowerBound])
        }
        let name = ext.isEmpty ? "\(stem)-r\(bits)" : "\(stem)-r\(bits).\(ext)"
        return url.deletingLastPathComponent().appendingPathComponent(name).path
    }

    private static func manifestWeightBits(at dir: String) -> Int? {
        let url = URL(fileURLWithPath: dir).appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: url),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let quant = root["quant"] as? [String: Any],
              let routed = quant["routedExpert"] as? [String: Any],
              let bits = routed["weightBits"] as? Int else { return nil }
        return bits
    }

    private static func takeValue(_ argv: [String],
                                  _ index: inout Int,
                                  flag: String) throws -> String {
        guard index + 1 < argv.count else { throw ArgsError.missingValue(flag: flag) }
        let value = argv[index + 1]
        index += 2
        return value
    }
}
