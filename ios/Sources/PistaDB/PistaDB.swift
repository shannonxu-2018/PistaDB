/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
import Foundation
import PistaDBObjC

/// PistaDB – embedded vector database for iOS and macOS.
///
/// Wraps the Objective-C ``PSTDatabase`` layer which in turn calls the
/// underlying C library.  Use try-with `defer` or a ``withDatabase`` block
/// to ensure the handle is closed cleanly.
///
/// ## Quick-start
/// ```swift
/// let db = try PistaDB(path: dbPath, dim: 384, metric: .cosine)
/// defer { db.close() }
///
/// try db.insert(id: 1, vector: embedding, label: "My document")
/// let hits = try db.search(queryVec, k: 5)
/// try db.save()
/// ```
///
/// ## SwiftUI / async
/// ```swift
/// let results = try await db.search(queryVec, k: 10)
/// ```
public final class PistaDB {

    private let db: PSTDatabase

    // MARK: - Init

    /// Opens or creates a database with default parameters and HNSW index.
    ///
    /// - Parameters:
    ///   - path: Path to the `.pst` file (created if absent).
    ///   - dim:  Vector dimensionality.
    public convenience init(path: String, dim: Int) throws {
        try self.init(path: path, dim: dim, metric: .l2, indexType: .hnsw, params: nil)
    }

    /// Opens or creates a database.
    ///
    /// - Parameters:
    ///   - path:      Path to the `.pst` file (created if absent).
    ///   - dim:       Vector dimensionality.
    ///   - metric:    Distance metric (default `.l2`).
    ///   - indexType: Index algorithm (default `.hnsw`).
    ///   - params:    Tuning parameters; `nil` uses C-library defaults.
    public init(path: String,
                dim: Int,
                metric: Metric = .l2,
                indexType: IndexType = .hnsw,
                params: PistaDBParams? = nil) throws
    {
        var nsErr: NSError?
        guard let database = PSTDatabase(path: path,
                                         dim: dim,
                                         metric: metric.objc,
                                         indexType: indexType.objc,
                                         params: params?.toPST(),
                                         error: &nsErr)
        else {
            throw nsErr?.toPistaDBError() ?? PistaDBError.openFailed("unknown")
        }
        self.db = database
    }

    // MARK: - Lifecycle

    /// Persists the database to disk without closing the handle.
    public func save() throws {
        var nsErr: NSError?
        guard db.save(&nsErr) else { throw nsErr!.toPistaDBError() }
    }

    /// Saves then frees the native handle. Safe to call multiple times.
    public func close() { db.close() }

    // MARK: - Insert

    /// Inserts a vector with an optional label.
    ///
    /// - Parameters:
    ///   - id:     Unique identifier (user-managed).
    ///   - vector: Float array of length ``dim``.
    ///   - label:  Optional human-readable label (max 255 bytes).
    public func insert(id: UInt64, vector: [Float], label: String = "") throws {
        var nsErr: NSError?
        let ok = vector.withUnsafeBufferPointer {
            db.insert(id: id,
                      floatArray: $0.baseAddress!,
                      count: vector.count,
                      label: label,
                      error: &nsErr)
        }
        if !ok { throw nsErr!.toPistaDBError() }
    }

    // MARK: - Delete

    /// Soft-deletes a vector by id. Space is reclaimed on the next ``save()``.
    public func delete(id: UInt64) throws {
        var nsErr: NSError?
        guard db.delete(id: id, error: &nsErr) else { throw nsErr!.toPistaDBError() }
    }

    // MARK: - Update

    /// Replaces the vector data for an existing id.
    ///
    /// - Parameters:
    ///   - id:     Identifier to update.
    ///   - vector: Replacement float array of length ``dim``.
    public func update(id: UInt64, vector: [Float]) throws {
        var nsErr: NSError?
        let ok = vector.withUnsafeBufferPointer {
            db.update(id: id,
                      floatArray: $0.baseAddress!,
                      count: vector.count,
                      error: &nsErr)
        }
        if !ok { throw nsErr!.toPistaDBError() }
    }

    // MARK: - Get

    /// Retrieves a stored vector and its label.
    public func get(id: UInt64) throws -> VectorEntry {
        var nsErr: NSError?
        guard let entry = db.entry(id: id, error: &nsErr) else {
            throw nsErr!.toPistaDBError()
        }
        return VectorEntry(entry)
    }

    // MARK: - Batch insert

    /// Inserts multiple vectors in one call.
    ///
    /// - Parameters:
    ///   - entries: Sequence of `(id, vector, label)` triples.
    public func insertBatch<S: Sequence>(_ entries: S) throws
        where S.Element == (id: UInt64, vector: [Float], label: String)
    {
        for (id, vector, label) in entries {
            try insert(id: id, vector: vector, label: label)
        }
    }

    // MARK: - Search

    /// K-nearest-neighbour search.
    ///
    /// - Parameters:
    ///   - query: Float array of length ``dim``.
    ///   - k:     Number of results (default 10).
    /// - Returns: Up to `k` results ordered by ascending distance.
    public func search(_ query: [Float], k: Int = 10) throws -> [SearchResult] {
        var nsErr: NSError?
        let hits: NSArray? = query.withUnsafeBufferPointer {
            db.search(floatArray: $0.baseAddress!,
                      count: query.count,
                      k: k,
                      error: &nsErr) as NSArray?
        }
        guard let hits else { throw nsErr!.toPistaDBError() }
        return (hits as! [PSTSearchResult]).map(SearchResult.init)
    }

    // MARK: - Train

    /// Trains the index on currently inserted vectors.
    ///
    /// Required for `IVF`, `IVF_PQ`, and `ScaNN` before inserting.
    /// Optional for `HNSW` / `DiskANN` (triggers a rebuild pass).
    public func train() throws {
        var nsErr: NSError?
        guard db.train(&nsErr) else { throw nsErr!.toPistaDBError() }
    }

    // MARK: - Metadata

    /// Number of active (non-deleted) vectors.
    public var count: Int { db.count }

    /// Vector dimensionality.
    public var dim: Int { db.dim }

    /// Distance metric in use.
    public var metric: Metric { Metric(db.metric) }

    /// Index algorithm in use.
    public var indexType: IndexType { IndexType(db.indexType) }

    /// Human-readable description of the last error.
    public var lastError: String { db.lastError }

    /// Library version string, e.g. `"1.0.0"`.
    public static var version: String { PSTDatabase.version() }
}
