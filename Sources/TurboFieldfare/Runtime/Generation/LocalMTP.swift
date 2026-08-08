import Foundation

// MARK: - Protocols

public protocol TargetTokenEmbeddingProvider: Sendable {
    func copyTokenEmbedding(_ token: Int32) -> [Float]?
}

public protocol HiddenStateLogitProducer: LogitProducer {
    func copyLastHiddenState() -> [Float]
}

public protocol AssistantBridgeSnapshotProducer: HiddenStateLogitProducer, TargetTokenEmbeddingProvider {
    func copyAssistantBridgeSnapshot(currentToken: Int32) -> AssistantBridgeSnapshot
}

public protocol LocalMTPDrafter: Sendable {
    func draftTokens(bridgeSnapshot: AssistantBridgeSnapshot,
                     history: [Int32],
                     currentToken: Int32,
                     maxDraftTokens: Int,
                     targetTokenEmbeddingProvider: (any TargetTokenEmbeddingProvider)?) throws -> [Int32]
}

// MARK: - Type-erased wrapper

public final class AnyLocalMTPDrafter: @unchecked Sendable {
    private let draftImpl: @Sendable (
        AssistantBridgeSnapshot,
        [Int32],
        Int32,
        Int,
        (any TargetTokenEmbeddingProvider)?
    ) throws -> [Int32]

    public init<D: LocalMTPDrafter>(_ drafter: D) {
        self.draftImpl = drafter.draftTokens(
            bridgeSnapshot:history:currentToken:maxDraftTokens:targetTokenEmbeddingProvider:)
    }

    public func draftTokens(bridgeSnapshot: AssistantBridgeSnapshot,
                            history: [Int32],
                            currentToken: Int32,
                            maxDraftTokens: Int,
                            targetTokenEmbeddingProvider: (any TargetTokenEmbeddingProvider)? = nil) throws -> [Int32] {
        try draftImpl(
            bridgeSnapshot,
            history,
            currentToken,
            maxDraftTokens,
            targetTokenEmbeddingProvider)
    }
}

// MARK: - Speculation spec

public struct LocalMTPSpeculation: Sendable {
    public let drafter: AnyLocalMTPDrafter
    public let maxDraftTokens: Int

    public init(drafter: AnyLocalMTPDrafter, maxDraftTokens: Int) {
        precondition(maxDraftTokens > 0, "maxDraftTokens must be positive")
        self.drafter = drafter
        self.maxDraftTokens = maxDraftTokens
    }

    public static func ngram(maxDraftTokens: Int,
                             maxOrder: Int = 4) -> LocalMTPSpeculation {
        LocalMTPSpeculation(
            drafter: AnyLocalMTPDrafter(NGramLocalMTPDrafter(maxOrder: maxOrder)),
            maxDraftTokens: maxDraftTokens)
    }

    public static func assistant(modelDirectory: URL,
                                 context: MetalContext,
                                 maxDraftTokens: Int) throws -> LocalMTPSpeculation {
        try LocalMTPSpeculation(
            drafter: AnyLocalMTPDrafter(
                GemmaAssistantMetalDrafter(modelDirectory: modelDirectory, context: context)),
            maxDraftTokens: maxDraftTokens)
    }
}

// MARK: - N-Gram drafter (fallback, no model needed)

public struct NGramLocalMTPDrafter: LocalMTPDrafter, Sendable {
    public let maxOrder: Int

    public init(maxOrder: Int = 4) {
        precondition(maxOrder > 0, "maxOrder must be positive")
        self.maxOrder = maxOrder
    }

    public func draftTokens(bridgeSnapshot _: AssistantBridgeSnapshot,
                            history: [Int32],
                            currentToken: Int32,
                            maxDraftTokens: Int,
                            targetTokenEmbeddingProvider _: (any TargetTokenEmbeddingProvider)?) throws -> [Int32] {
        guard maxDraftTokens > 0 else { return [] }
        guard !history.isEmpty, history.last == currentToken else { return [] }
        let upperOrder = min(maxOrder, history.count)
        guard upperOrder > 0 else { return [] }

        for order in stride(from: upperOrder, through: 1, by: -1) {
            let suffixStart = history.count - order
            guard let matchStart = bestMatchStart(
                history: history,
                suffixStart: suffixStart,
                order: order)
            else {
                continue
            }
            let continuationStart = matchStart + order
            let continuationEnd = min(continuationStart + maxDraftTokens, history.count)
            let draft = Array(history[continuationStart..<continuationEnd])
            if !draft.isEmpty {
                return draft
            }
        }
        return []
    }

    private func bestMatchStart(history: [Int32],
                                suffixStart: Int,
                                order: Int) -> Int? {
        guard suffixStart > 0 else { return nil }
        let suffix = history[suffixStart..<history.count]
        var best: Int?
        for start in 0..<suffixStart {
            let end = start + order
            guard end < history.count else { continue }
            guard history[start..<end].elementsEqual(suffix) else { continue }
            best = start
        }
        return best
    }
}
