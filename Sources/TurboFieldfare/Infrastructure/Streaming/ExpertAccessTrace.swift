import Foundation

/// Opt-in tracing of routed-expert selections, one line per cache plan.
///
/// Enabled by setting `TURBO_FIELDFARE_EXPERT_TRACE` to an output path. The
/// trace exists to answer capacity-planning questions offline — expert reuse
/// across decode steps, access skew, and the hit rate an ideal cache of a
/// given size would reach — without perturbing the hot path when disabled
/// (a single `Bool` load).
///
/// Line format (CSV, no header):
///   `<layerLabel>,<phase>,<hits>,<misses>,<expert ids separated by spaces>`
public final class ExpertAccessTrace: @unchecked Sendable {
    public static let shared = ExpertAccessTrace()

    /// Flush threshold. Trace lines are tiny; batching keeps write syscalls
    /// off the per-layer critical path.
    private static let flushThresholdBytes = 1 << 20

    private let lock = NSLock()
    private let path: String?
    private var pending = Data()
    private var handle: FileHandle?
    private var openFailed = false

    /// Hoisted so callers can branch without touching the lock.
    public let isEnabled: Bool

    private init() {
        let raw = ProcessInfo.processInfo.environment["TURBO_FIELDFARE_EXPERT_TRACE"]
        let trimmed = raw?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let trimmed, !trimmed.isEmpty {
            path = trimmed
            isEnabled = true
        } else {
            path = nil
            isEnabled = false
        }
    }

    public func record(layerLabel: String,
                       phase: String,
                       experts: [Int],
                       hits: Int,
                       misses: Int) {
        guard isEnabled else { return }
        var line = layerLabel
        line += ","
        line += phase
        line += ","
        line += String(hits)
        line += ","
        line += String(misses)
        line += ","
        for (index, expert) in experts.enumerated() {
            if index > 0 { line += " " }
            line += String(expert)
        }
        line += "\n"

        lock.lock()
        pending.append(contentsOf: line.utf8)
        let shouldFlush = pending.count >= Self.flushThresholdBytes
        lock.unlock()

        if shouldFlush { flush() }
    }

    /// Writes buffered lines out. Safe to call when disabled or already empty.
    public func flush() {
        guard isEnabled, let path else { return }
        lock.lock()
        defer { lock.unlock() }
        guard !pending.isEmpty, !openFailed else { return }

        if handle == nil {
            FileManager.default.createFile(atPath: path, contents: nil)
            guard let opened = FileHandle(forWritingAtPath: path) else {
                openFailed = true
                pending.removeAll(keepingCapacity: false)
                return
            }
            handle = opened
        }
        handle?.write(pending)
        pending.removeAll(keepingCapacity: true)
    }
}
