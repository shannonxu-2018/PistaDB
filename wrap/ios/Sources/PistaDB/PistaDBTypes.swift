/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
import Foundation
import PistaDBObjC

// MARK: - Metric

/// Distance metric used for vector comparisons.
public enum Metric: Int, CaseIterable, Sendable {
    /// Euclidean distance – general purpose, image and multimodal embeddings.
    case l2      = 0
    /// Cosine similarity stored as `1 − similarity` – text embeddings.
    case cosine  = 1
    /// Inner product stored as negative dot product – normalised vectors.
    case ip      = 2
    /// Manhattan / L1 distance – sparse vectors.
    case l1      = 3
    /// Hamming distance – binary embeddings, deduplication.
    case hamming = 4

    var objc: PSTMetric { PSTMetric(rawValue: rawValue)! }

    init(_ objc: PSTMetric) { self.init(rawValue: objc.rawValue)! }
}

// MARK: - IndexType

/// Index algorithm used to organise and search vectors.
public enum IndexType: Int, CaseIterable, Sendable {
    /// Brute-force linear scan – exact results, no training needed.
    case linear  = 0
    /// Hierarchical Navigable Small World graphs (recommended default).
    case hnsw    = 1
    /// Inverted File Index – requires ``PistaDB/train()`` before inserting.
    case ivf     = 2
    /// IVF + Product Quantisation – requires ``PistaDB/train()`` before inserting.
    case ivfPQ   = 3
    /// Vamana / DiskANN – billion-scale datasets.
    case diskANN = 4
    /// Locality-Sensitive Hashing – ultra-low memory.
    case lsh     = 5
    /// ScaNN: Anisotropic Vector Quantisation – requires ``PistaDB/train()``.
    case scaNN   = 6

    var objc: PSTIndexType { PSTIndexType(rawValue: rawValue)! }

    init(_ objc: PSTIndexType) { self.init(rawValue: objc.rawValue)! }
}

// MARK: - SearchResult

/// A single k-NN result returned by ``PistaDB/search(_:k:)``.
/// Results are ordered by ascending distance (nearest first).
public struct SearchResult: Sendable {
    /// The vector's unique identifier.
    public let id: UInt64
    /// Distance from the query (interpretation depends on metric).
    public let distance: Float
    /// Human-readable label supplied at insert time. Empty if none.
    public let label: String

    init(_ pst: PSTSearchResult) {
        self.id       = pst.vectorId
        self.distance = pst.distance
        self.label    = pst.label
    }
}

// MARK: - VectorEntry

/// A stored vector with its label, returned by ``PistaDB/get(id:)``.
public struct VectorEntry: Sendable {
    /// The raw float vector (length == `dim`).
    public let vector: [Float]
    /// Human-readable label supplied at insert time. Empty if none.
    public let label: String

    init(_ pst: PSTVectorEntry) {
        self.label  = pst.label
        self.vector = pst.vectorData.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
    }
}

// MARK: - PistaDBParams

/// Index tuning parameters passed when opening a database.
///
/// ```swift
/// var p = PistaDBParams()
/// p.hnswM = 32
/// p.hnswEfSearch = 100
/// ```
public struct PistaDBParams: Sendable {

    // HNSW
    public var hnswM:              Int   = 16
    public var hnswEfConstruction: Int   = 200
    public var hnswEfSearch:       Int   = 50

    // IVF / IVF_PQ
    public var ivfNlist:           Int   = 128
    public var ivfNprobe:          Int   = 8
    public var pqM:                Int   = 8
    public var pqNbits:            Int   = 8

    // DiskANN
    public var diskannR:           Int   = 32
    public var diskannL:           Int   = 100
    public var diskannAlpha:       Float = 1.2

    // LSH
    public var lshL:               Int   = 10
    public var lshK:               Int   = 8
    public var lshW:               Float = 10.0

    // ScaNN
    public var scannNlist:         Int   = 128
    public var scannNprobe:        Int   = 32
    public var scannPqM:           Int   = 8
    public var scannPqBits:        Int   = 8
    public var scannRerankK:       Int   = 100
    public var scannAqEta:         Float = 0.2

    public init() {}

    func toPST() -> PSTParams {
        let p = PSTParams.default()
        p.hnswM               = hnswM
        p.hnswEfConstruction  = hnswEfConstruction
        p.hnswEfSearch        = hnswEfSearch
        p.ivfNlist            = ivfNlist
        p.ivfNprobe           = ivfNprobe
        p.pqM                 = pqM
        p.pqNbits             = pqNbits
        p.diskannR            = diskannR
        p.diskannL            = diskannL
        p.diskannAlpha        = diskannAlpha
        p.lshL                = lshL
        p.lshK                = lshK
        p.lshW                = lshW
        p.scannNlist          = scannNlist
        p.scannNprobe         = scannNprobe
        p.scannPqM            = scannPqM
        p.scannPqBits         = scannPqBits
        p.scannRerankK        = scannRerankK
        p.scannAqEta          = scannAqEta
        return p
    }
}

// MARK: - PistaDBError

/// Errors thrown by ``PistaDB`` operations.
public enum PistaDBError: LocalizedError, Sendable {
    case openFailed(String)
    case notFound
    case alreadyExists
    case invalidArgument(String)
    case notTrained
    case ioError(String)
    case corrupt
    case closed
    case operationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .openFailed(let m):     return "Failed to open database: \(m)"
        case .notFound:              return "Vector not found"
        case .alreadyExists:         return "Vector ID already exists"
        case .invalidArgument(let m): return "Invalid argument: \(m)"
        case .notTrained:            return "Index not trained – call train() before inserting"
        case .ioError(let m):        return "I/O error: \(m)"
        case .corrupt:               return "Database file is corrupt"
        case .closed:                return "Database is already closed"
        case .operationFailed(let m): return "Operation failed: \(m)"
        }
    }
}

// MARK: - Internal helpers

extension NSError {
    func toPistaDBError() -> PistaDBError {
        guard domain == PSTDatabaseErrorDomain else {
            return .operationFailed(localizedDescription)
        }
        // Swift strips the common "PSTDatabaseError" prefix from NS_ENUM cases:
        // PSTDatabaseErrorNotFound → PSTDatabaseErrorCode.notFound
        switch PSTDatabaseErrorCode(rawValue: code) {
        case .notFound:      return .notFound
        case .alreadyExists: return .alreadyExists
        case .notTrained:    return .notTrained
        case .corrupt:       return .corrupt
        case .io:            return .ioError(localizedDescription)
        case .invalidArg:    return .invalidArgument(localizedDescription)
        case .noMemory:      return .operationFailed("Out of memory")
        default:             return .operationFailed(localizedDescription)
        }
    }
}
