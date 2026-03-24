/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
import Foundation

// MARK: - DSL factory

/// Opens or creates a PistaDB database using a trailing-closure DSL.
///
/// ```swift
/// let db = try pistaDB(path: dbPath, dim: 384) {
///     $0.metric    = .cosine
///     $0.indexType = .hnsw
///     $0.params.hnswEfSearch = 100
/// }
/// ```
public func pistaDB(
    path: String,
    dim: Int,
    configure: (inout PistaDBConfig) -> Void = { _ in }
) throws -> PistaDB {
    var cfg = PistaDBConfig(dim: dim)
    configure(&cfg)
    return try PistaDB(path: path, dim: dim,
                       metric: cfg.metric,
                       indexType: cfg.indexType,
                       params: cfg.params)
}

/// Mutable configuration value used in the ``pistaDB(path:dim:configure:)`` DSL.
public struct PistaDBConfig {
    public var metric:    Metric        = .l2
    public var indexType: IndexType     = .hnsw
    public var params:    PistaDBParams = PistaDBParams()

    public init(dim: Int) {}

    /// Inline params shorthand: `cfg.params { $0.hnswEfSearch = 100 }`.
    public mutating func params(_ configure: (inout PistaDBParams) -> Void) {
        configure(&params)
    }
}

// MARK: - withDatabase (scoped use)

/// Opens a database, executes `body`, saves, then closes – even if `body` throws.
///
/// ```swift
/// try withDatabase(path: dbPath, dim: 384) { db in
///     try db.insert(id: 1, vector: vec, label: "hello")
/// }  // saved & closed automatically
/// ```
public func withDatabase<T>(
    path: String,
    dim: Int,
    metric: Metric = .l2,
    indexType: IndexType = .hnsw,
    params: PistaDBParams? = nil,
    body: (PistaDB) throws -> T
) throws -> T {
    let db = try PistaDB(path: path, dim: dim,
                         metric: metric, indexType: indexType, params: params)
    defer { db.close() }
    return try body(db)
}

// MARK: - PistaDB subscript

public extension PistaDB {
    /// Retrieves a stored entry by id using subscript syntax.
    ///
    /// ```swift
    /// let entry = try db[42]
    /// ```
    subscript(id: UInt64) -> VectorEntry {
        get throws { try get(id: id) }
    }
}

// MARK: - Batch from Dictionary

public extension PistaDB {
    /// Inserts a dictionary of `[id: (vector, label)]`.
    func insertBatch(_ dict: [UInt64: (vector: [Float], label: String)]) throws {
        for (id, pair) in dict {
            try insert(id: id, vector: pair.vector, label: pair.label)
        }
    }

    /// Inserts a dictionary of `[id: vector]` with empty labels.
    func insertBatch(_ dict: [UInt64: [Float]]) throws {
        for (id, vec) in dict {
            try insert(id: id, vector: vec)
        }
    }
}

// MARK: - PistaDBParams presets

public extension PistaDBParams {
    /// HNSW preset for high-recall scenarios.
    static var highRecall: PistaDBParams {
        var p = PistaDBParams()
        p.hnswM              = 32
        p.hnswEfConstruction = 400
        p.hnswEfSearch       = 200
        return p
    }

    /// HNSW preset for low-latency / on-device scenarios.
    static var lowLatency: PistaDBParams {
        var p = PistaDBParams()
        p.hnswM              = 16
        p.hnswEfConstruction = 100
        p.hnswEfSearch       = 20
        return p
    }
}

// MARK: - SearchResult helpers

public extension Array where Element == SearchResult {
    /// Returns only the ids in order.
    var ids: [UInt64] { map(\.id) }

    /// Returns only the distances in order.
    var distances: [Float] { map(\.distance) }

    /// Returns only the labels in order.
    var labels: [String] { map(\.label) }

    /// Returns the closest result (first element).
    var nearest: SearchResult? { first }
}
