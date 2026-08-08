import CryptoKit
import Foundation

public struct ExpertLayoutOrder: Sendable, Equatable {
    public struct LayerOrder: Sendable, Equatable {
        public let layer: Int
        public let order: [Int]

        public init(layer: Int, order: [Int]) {
            self.layer = layer
            self.order = order
        }
    }

    public let strategy: String
    public let sourcePath: String?
    public let sha256Hex: String
    public let layers: [LayerOrder]
    private let ordersByLayer: [Int: [Int]]

    public var reorderedLayerCount: Int { layers.count }

    public init(strategy: String,
                sourcePath: String?,
                sha256Hex: String,
                layers: [LayerOrder]) {
        self.strategy = strategy
        self.sourcePath = sourcePath
        self.sha256Hex = sha256Hex
        self.layers = layers.sorted { $0.layer < $1.layer }
        self.ordersByLayer = Dictionary(uniqueKeysWithValues: layers.map { ($0.layer, $0.order) })
    }

    public func order(for layer: Int, expertsPerLayer: Int) throws -> [Int]? {
        guard let order = ordersByLayer[layer] else { return nil }
        try Self.validateOrder(order, layer: layer, expertsPerLayer: expertsPerLayer)
        return order
    }

    public static func load(path: String,
                            expectedLayers: Int? = nil,
                            expectedExpertsPerLayer: Int? = nil) throws -> ExpertLayoutOrder {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rawLayers = root["layers"] as? [[String: Any]] else {
            throw RepackError.configurationInvalid(detail: "expert layout order must contain a layers array")
        }
        let strategy = root["strategy"] as? String ?? "custom"
        var layers: [LayerOrder] = []
        layers.reserveCapacity(rawLayers.count)
        var seenLayers = Set<Int>()
        for layerObject in rawLayers {
            guard let layer = layerObject["layer"] as? Int,
                  let order = layerObject["order"] as? [Int] else {
                throw RepackError.configurationInvalid(detail: "expert layout order layer entry is malformed")
            }
            guard seenLayers.insert(layer).inserted else {
                throw RepackError.configurationInvalid(detail: "expert layout order duplicates layer \(layer)")
            }
            if let expectedExpertsPerLayer {
                try validateOrder(order, layer: layer, expertsPerLayer: expectedExpertsPerLayer)
            } else {
                try validateDistinct(order, layer: layer)
            }
            layers.append(LayerOrder(layer: layer, order: order))
        }
        if let expectedLayers {
            for layer in layers {
                guard (0..<expectedLayers).contains(layer.layer) else {
                    throw RepackError.configurationInvalid(detail:
                        "expert layout order layer \(layer.layer) exceeds layer count \(expectedLayers)")
                }
            }
        }
        return ExpertLayoutOrder(strategy: strategy,
                                 sourcePath: path,
                                 sha256Hex: digest,
                                 layers: layers)
    }

    private static func validateDistinct(_ order: [Int], layer: Int) throws {
        guard order.allSatisfy({ $0 >= 0 }) else {
            throw RepackError.configurationInvalid(detail:
                "expert layout order layer \(layer) contains a negative expert id")
        }
        guard Set(order).count == order.count else {
            throw RepackError.configurationInvalid(detail:
                "expert layout order layer \(layer) contains duplicate expert ids")
        }
    }

    private static func validateOrder(_ order: [Int],
                                      layer: Int,
                                      expertsPerLayer: Int) throws {
        guard order.count == expertsPerLayer else {
            throw RepackError.configurationInvalid(detail:
                "expert layout order layer \(layer) count \(order.count) != expertsPerLayer \(expertsPerLayer)")
        }
        try validateDistinct(order, layer: layer)
        guard Set(order) == Set(0..<expertsPerLayer) else {
            throw RepackError.configurationInvalid(detail:
                "expert layout order layer \(layer) is not a full permutation")
        }
    }
}
