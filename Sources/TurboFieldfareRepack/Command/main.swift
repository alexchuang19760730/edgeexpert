import Foundation
import TurboFieldfareRepackCore

private let usage = """
Usage:
  TurboFieldfareRepack --output <model.gturbo> [--overwrite] [--resume]
  TurboFieldfareRepack --local-input-hf <model-dir> --output <model.gturbo> [--overwrite]
                       [--expert-layout-order <layout-order.json>]
  TurboFieldfareRepack --discard-partial --output <model.gturbo>
  TurboFieldfareRepack --verify-install --input-gturbo <model.gturbo>
  TurboFieldfareRepack --reseal --input-gturbo <model.gturbo> [--rebind]
  TurboFieldfareRepack --help

The installer streams the supported Gemma 4 checkpoint from Hugging Face and
repackages it without materializing the source checkpoint on disk. Set HF_TOKEN
only if Hugging Face requests authentication. A cancelled or interrupted
download can be continued with --resume or removed with --discard-partial.

Re-sealing:
  A verified-install.json receipt is bound to the absolute path it was signed
  for, so copying or moving a model invalidates it and the runtime falls back
  to re-hashing every expert shard on each launch. --reseal re-issues the
  receipt for the model's current location.

  --reseal            Re-read and re-hash every file, then sign. This is the
                      only mode that detects silent data corruption.
  --reseal --rebind   Fast path for a model that was relocated as a whole after
                      it had already been verified: confirms every file is
                      present at its recorded byte length, re-hashes only
                      manifest.json, and inherits the remaining digests. Takes
                      seconds instead of minutes. The receipt records that it
                      was produced this way.
"""

private struct Arguments {
    var output: String?
    var overwrite = false
    var resume = false
    var discardPartial = false
    var verifyInstall = false
    var reseal = false
    var rebind = false
    var inputGTurbo: String?
    var localInputHF: String?
    var expertLayoutOrderPath: String?

    static func parse(_ values: [String]) throws -> Arguments {
        var parsed = Arguments()
        var index = 1
        while index < values.count {
            let flag = values[index]
            switch flag {
            case "--help":
                throw ParseError.help
            case "--overwrite":
                parsed.overwrite = true
                index += 1
            case "--resume":
                parsed.resume = true
                index += 1
            case "--discard-partial":
                parsed.discardPartial = true
                index += 1
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--reseal":
                parsed.reseal = true
                index += 1
            case "--rebind":
                parsed.rebind = true
                index += 1
            case "--output", "--input-gturbo", "--local-input-hf", "--expert-layout-order":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                if flag == "--output" {
                    parsed.output = values[index + 1]
                } else if flag == "--local-input-hf" {
                    parsed.localInputHF = values[index + 1]
                } else if flag == "--expert-layout-order" {
                    parsed.expertLayoutOrderPath = values[index + 1]
                } else {
                    parsed.inputGTurbo = values[index + 1]
                }
                index += 2
            default:
                throw ParseError.unknown(flag)
            }
        }

        guard !(parsed.resume && parsed.discardPartial) else {
            throw ParseError.invalidMode("--resume and --discard-partial are mutually exclusive")
        }
        guard !(parsed.reseal && parsed.verifyInstall) else {
            throw ParseError.invalidMode("--reseal and --verify-install are mutually exclusive")
        }
        guard !parsed.rebind || parsed.reseal else {
            throw ParseError.invalidMode("--rebind requires --reseal")
        }
        if parsed.reseal {
            guard parsed.inputGTurbo != nil else {
                throw ParseError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil,
                  parsed.localInputHF == nil,
                  parsed.expertLayoutOrderPath == nil,
                  !parsed.overwrite,
                  !parsed.resume,
                  !parsed.discardPartial else {
                throw ParseError.invalidMode("--reseal accepts only --input-gturbo and --rebind")
            }
            return parsed
        }
        if parsed.discardPartial {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil,
                  parsed.localInputHF == nil,
                  parsed.expertLayoutOrderPath == nil,
                  !parsed.overwrite,
                  !parsed.verifyInstall else {
                throw ParseError.invalidMode("--discard-partial only accepts --output")
            }
            return parsed
        }
        if parsed.verifyInstall {
            guard parsed.inputGTurbo != nil else {
                throw ParseError.missingRequired("--input-gturbo")
            }
            guard parsed.output == nil,
                  parsed.localInputHF == nil,
                  parsed.expertLayoutOrderPath == nil,
                  !parsed.overwrite,
                  !parsed.resume else {
                throw ParseError.invalidMode("verification accepts only --input-gturbo")
            }
        } else {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputGTurbo == nil else {
                throw ParseError.invalidMode("--input-gturbo requires --verify-install")
            }
            if parsed.localInputHF != nil && parsed.resume {
                throw ParseError.invalidMode("--local-input-hf does not support --resume")
            }
            if parsed.expertLayoutOrderPath != nil && parsed.output == nil {
                throw ParseError.invalidMode("--expert-layout-order requires --output")
            }
        }
        return parsed
    }
}

private enum ParseError: Error, CustomStringConvertible {
    case help
    case unknown(String)
    case missingValue(String)
    case missingRequired(String)
    case invalidMode(String)

    var description: String {
        switch self {
        case .help: return "help"
        case .unknown(let flag): return "unknown argument: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .missingRequired(let flag): return "missing required argument: \(flag)"
        case .invalidMode(let message): return message
        }
    }
}

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func formatBytes(_ bytes: UInt64) -> String {
    guard bytes >= 1024 else { return "\(bytes) B" }
    let units = ["KiB", "MiB", "GiB", "TiB"]
    var value = Double(bytes) / 1024
    var unit = 0
    while value >= 1024 && unit < units.count - 1 {
        value /= 1024
        unit += 1
    }
    return String(format: "%.2f %@", value, units[unit])
}

/// Surfaces files sitting in the model directory that no manifest entry claims.
/// These were previously computed and then dropped on the floor, so a stray or
/// leftover file could never be noticed.
private func reportUnexpected(_ entries: [String]) {
    guard !entries.isEmpty else { return }
    printError("warning: \(entries.count) entry(s) present but not declared in manifest.json:")
    for entry in entries.prefix(10) {
        printError("  \(entry)")
    }
    if entries.count > 10 {
        printError("  ... and \(entries.count - 10) more")
    }
}

private func repackBaseURL(from environment: [String: String]) throws -> URL {
    guard let raw = environment["HF_BASE_URL"], !raw.isEmpty else {
        return SupportedModelSource.defaultBaseURL
    }
    guard let url = URL(string: raw), url.scheme != nil, url.host != nil else {
        throw ParseError.invalidMode("HF_BASE_URL must be an absolute URL")
    }
    return url
}

private func run(_ values: [String]) async -> Int32 {
    let arguments: Arguments
    do {
        arguments = try Arguments.parse(values)
    } catch ParseError.help {
        print(usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(usage)")
        return 2
    }

    if arguments.discardPartial, let output = arguments.output {
        do {
            try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
            print("Discarded saved download for \(output)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.verifyInstall, let input = arguments.inputGTurbo {
        do {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: input))
            print("Verified \(result.fileCount) files (\(formatBytes(result.bytesVerified)))")
            print("Receipt: \(result.receiptPath)")
            reportUnexpected(result.unexpectedEntries)
            return 0
        } catch {
            printError("verification failed: \(error)")
            return 1
        }
    }

    if arguments.reseal, let input = arguments.inputGTurbo {
        do {
            let mode: ResealMode = arguments.rebind ? .rebind : .full
            let result = try ResealTool.run(
                options: ResealOptions(inputGTurbo: input, mode: mode))
            switch result.mode {
            case .full:
                print("Re-sealed \(result.fileCount) files "
                    + "(\(formatBytes(result.bytesVerified)) hashed) "
                    + String(format: "in %.1fs", result.elapsedSeconds))
            case .rebind:
                print("Rebound \(result.fileCount) files "
                    + "(\(formatBytes(result.bytesVerified)) size-checked, digests inherited) "
                    + String(format: "in %.1fs", result.elapsedSeconds))
            }
            if let previous = result.previousPath,
               previous != URL(fileURLWithPath: input).standardizedFileURL.path {
                print("Path: \(previous)")
                print("  ->  \(URL(fileURLWithPath: input).standardizedFileURL.path)")
            }
            print("Receipt: \(result.receiptPath)")
            reportUnexpected(result.unexpectedEntries)
            return 0
        } catch {
            printError("reseal failed: \(error)")
            return 1
        }
    }

    guard let output = arguments.output else { return 2 }
    let expertLayoutOrder: ExpertLayoutOrder?
    do {
        if let path = arguments.expertLayoutOrderPath {
            expertLayoutOrder = try ExpertLayoutOrder.load(path: path)
        } else {
            expertLayoutOrder = nil
        }
    } catch {
        printError("invalid expert layout order: \(error)")
        return 2
    }
    if let localInput = arguments.localInputHF {
        let options = LocalDirectoryRepackOptions(
            inputDir: localInput,
            outputDir: output,
            expertLayoutOrder: expertLayoutOrder,
            overwrite: arguments.overwrite)
        do {
            let result = try await LocalDirectoryRepacker(options: options).run()
            print("Installed local HF snapshot")
            print("Source snapshot: \(result.sourceIndexSHA256)")
            print("Model ID: \(result.modelID)")
            print("Model: \(result.outputDir)")
            return 0
        } catch {
            printError("local install failed: \(error)")
            return 1
        }
    }
    let environment = ProcessInfo.processInfo.environment
    let baseURL: URL
    do {
        baseURL = try repackBaseURL(from: environment)
    } catch {
        printError("install failed: \(error)")
        return 1
    }
    let options = SupportedModelSource.installOptions(
        outputDirectory: URL(fileURLWithPath: output),
        overwrite: arguments.overwrite,
        token: environment["HF_TOKEN"],
        expertLayoutOrder: expertLayoutOrder,
        resume: arguments.resume,
        baseURL: baseURL)
    do {
        let result = try await RemoteStreamingRepacker(options: options).run()
        print("Installed \(SupportedModelSource.displayName)")
        print("Source revision: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        printError("install failed: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))
