import Foundation

public struct PrefillExpertCountEntry: Sendable, Equatable {
    public let expert: Int
    public let count: UInt64

    public init(expert: Int, count: UInt64) {
        self.expert = expert
        self.count = count
    }
}

public struct PrefillExpertPairCountEntry: Sendable, Equatable {
    public let first: Int
    public let second: Int
    public let count: UInt64

    public init(first: Int, second: Int, count: UInt64) {
        self.first = first
        self.second = second
        self.count = count
    }
}

public struct PrefillExpertLayerTrace: Sendable, Equatable {
    public let layer: Int
    public let tileAccesses: UInt64
    public let uniqueExperts: UInt64
    public let expertCounts: [PrefillExpertCountEntry]
    public let coAccessPairs: [PrefillExpertPairCountEntry]

    public init(layer: Int,
                tileAccesses: UInt64,
                uniqueExperts: UInt64,
                expertCounts: [PrefillExpertCountEntry],
                coAccessPairs: [PrefillExpertPairCountEntry]) {
        self.layer = layer
        self.tileAccesses = tileAccesses
        self.uniqueExperts = uniqueExperts
        self.expertCounts = expertCounts
        self.coAccessPairs = coAccessPairs
    }
}

public struct PrefillExpertTrace: Sendable, Equatable {
    public let layers: [PrefillExpertLayerTrace]

    public init(layers: [PrefillExpertLayerTrace]) {
        self.layers = layers
    }

    public static let empty = PrefillExpertTrace(layers: [])
}
