/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * Milvus-style schema layer for the Swift / Objective-C wrapper.
 *
 * Adds DataType, FieldSchema, CollectionSchema, Collection, plus the
 * `createCollection` / `loadCollection` factories.  Vectors stay in the
 * underlying `.pst` file; non-vector scalar fields go to a JSON sidecar
 * (`<path>.meta.json`) handled by `Foundation.JSONSerialization`.
 */
import Foundation

// MARK: - DataType

/// Per-field data type.  Mirrors `pymilvus.DataType`.
public enum DataType: Int, Sendable {
    case bool         = 1
    case int8         = 2
    case int16        = 3
    case int32        = 4
    case int64        = 5
    case float        = 10
    case double       = 11
    case varchar      = 21
    case json         = 23
    case floatVector  = 101

    /// Wire string used in the JSON sidecar.
    public var wire: String {
        switch self {
        case .bool:        return "BOOL"
        case .int8:        return "INT8"
        case .int16:       return "INT16"
        case .int32:       return "INT32"
        case .int64:       return "INT64"
        case .float:       return "FLOAT"
        case .double:      return "DOUBLE"
        case .varchar:     return "VARCHAR"
        case .json:        return "JSON"
        case .floatVector: return "FLOAT_VECTOR"
        }
    }

    /// Parse a wire string back to a `DataType`.
    public static func fromWire(_ s: String) throws -> DataType {
        switch s {
        case "BOOL":         return .bool
        case "INT8":         return .int8
        case "INT16":        return .int16
        case "INT32":        return .int32
        case "INT64":        return .int64
        case "FLOAT":        return .float
        case "DOUBLE":       return .double
        case "VARCHAR":      return .varchar
        case "JSON":         return .json
        case "FLOAT_VECTOR": return .floatVector
        default:
            throw PistaDBError.invalidArgument("unknown DataType: \(s)")
        }
    }

    var isInt:   Bool { self == .int8 || self == .int16 || self == .int32 || self == .int64 }
    var isFloat: Bool { self == .float || self == .double }
}

// MARK: - FieldSchema

/// Description of a single field in a `CollectionSchema`.
public struct FieldSchema: Sendable {
    public let name: String
    public let dtype: DataType
    public let isPrimary: Bool
    public let autoId: Bool
    public let maxLength: Int?     // VARCHAR only
    public let dim: Int?           // FLOAT_VECTOR only
    public let description: String

    public init(
        name: String,
        dtype: DataType,
        isPrimary: Bool = false,
        autoId: Bool = false,
        maxLength: Int? = nil,
        dim: Int? = nil,
        description: String = ""
    ) throws {
        precondition(!name.isEmpty, "FieldSchema: name must not be empty")
        self.name        = name
        self.dtype       = dtype
        self.isPrimary   = isPrimary
        self.autoId      = autoId
        self.maxLength   = maxLength
        self.dim         = dim
        self.description = description
        try validate()
    }

    private func validate() throws {
        if dtype == .floatVector && (dim ?? 0) <= 0 {
            throw PistaDBError.invalidArgument(
                "FLOAT_VECTOR field '\(name)' requires a positive dim")
        }
        if dtype == .varchar, let m = maxLength, m <= 0 {
            throw PistaDBError.invalidArgument(
                "VARCHAR field '\(name)': maxLength must be positive")
        }
        if isPrimary && dtype != .int64 {
            throw PistaDBError.invalidArgument(
                "primary key '\(name)' must be INT64")
        }
        if autoId && !isPrimary {
            throw PistaDBError.invalidArgument(
                "autoId only valid on the primary field (got '\(name)')")
        }
    }

    func toJSON() -> [String: Any] {
        var d: [String: Any] = [
            "name":       name,
            "dtype":      dtype.wire,
            "is_primary": isPrimary,
            "auto_id":    autoId,
        ]
        if let m = maxLength      { d["max_length"]  = m }
        if let v = dim            { d["dim"]         = v }
        if !description.isEmpty   { d["description"] = description }
        return d
    }

    static func fromJSON(_ obj: [String: Any]) throws -> FieldSchema {
        guard
            let name  = obj["name"] as? String,
            let wire  = obj["dtype"] as? String
        else {
            throw PistaDBError.invalidArgument("sidecar: bad field entry")
        }
        return try FieldSchema(
            name:        name,
            dtype:       try DataType.fromWire(wire),
            isPrimary:   obj["is_primary"] as? Bool ?? false,
            autoId:      obj["auto_id"]    as? Bool ?? false,
            maxLength:   obj["max_length"] as? Int,
            dim:         obj["dim"]        as? Int,
            description: obj["description"] as? String ?? ""
        )
    }
}

// MARK: - CollectionSchema

public struct CollectionSchema: Sendable {
    public let fields: [FieldSchema]
    public let description: String

    public init(_ fields: [FieldSchema], description: String = "") throws {
        self.fields      = fields
        self.description = description

        var seen = Set<String>()
        var primaryCount = 0
        var vectorCount  = 0
        for f in fields {
            if !seen.insert(f.name).inserted {
                throw PistaDBError.invalidArgument("duplicate field name '\(f.name)'")
            }
            if f.isPrimary { primaryCount += 1 }
            if f.dtype == .floatVector { vectorCount += 1 }
        }
        if primaryCount != 1 {
            throw PistaDBError.invalidArgument(
                "schema must have exactly one primary key (found \(primaryCount))")
        }
        if vectorCount != 1 {
            throw PistaDBError.invalidArgument(
                "schema must have exactly one FLOAT_VECTOR field (found \(vectorCount))")
        }
    }

    public var primary: FieldSchema {
        fields.first(where: { $0.isPrimary })!
    }

    public var vector: FieldSchema {
        fields.first(where: { $0.dtype == .floatVector })!
    }

    public var scalarFields: [FieldSchema] {
        fields.filter { !$0.isPrimary && $0.dtype != .floatVector }
    }

    public func field(_ name: String) throws -> FieldSchema {
        guard let f = fields.first(where: { $0.name == name }) else {
            throw PistaDBError.invalidArgument("no field named '\(name)'")
        }
        return f
    }
}

// MARK: - Hit

/// A single search hit enriched with the projected scalar fields.
///
/// `fields` holds heterogeneous values, so `Hit` is not `Sendable` —
/// transferring across actors requires copying out the typed values you need.
public struct Hit {
    public let id: UInt64
    public let distance: Float
    public let fields: [String: Any]

    /// Get a projected field value, or nil if absent.
    public func get(_ name: String) -> Any? { fields[name] }
}

// MARK: - Collection options

/// Configuration passed to `createCollection` / `loadCollection`.
public struct CollectionOptions: Sendable {
    public var metric:     Metric        = .l2
    public var indexType:  IndexType     = .hnsw
    public var params:     PistaDBParams?
    /// Directory where `<name>.pst` is created.  Ignored if `path` is set.
    public var baseDir:    String?
    /// Explicit path overrides `baseDir + name`.
    public var path:       String?
    /// If true, replace any existing `.pst` / `.meta.json`; otherwise fail.
    public var overwrite:  Bool          = false

    public init(
        metric: Metric = .l2,
        indexType: IndexType = .hnsw,
        params: PistaDBParams? = nil,
        baseDir: String? = nil,
        path: String? = nil,
        overwrite: Bool = false
    ) {
        self.metric    = metric
        self.indexType = indexType
        self.params    = params
        self.baseDir   = baseDir
        self.path      = path
        self.overwrite = overwrite
    }

    func resolvePath(_ name: String) -> String {
        if let p = path { return p }
        if let d = baseDir, !d.isEmpty {
            return (d as NSString).appendingPathComponent("\(name).pst")
        }
        return "\(name).pst"
    }
}

// MARK: - Collection

/// Schema-backed collection wrapping a `PistaDB` plus a JSON sidecar.
public final class Collection {

    public let name: String
    public let path: String
    public let schema: CollectionSchema

    private let db: PistaDB
    private let metaPath: String
    private let metric: Metric
    private let indexType: IndexType
    private var rows: [UInt64: [String: Any]]
    private var nextID: UInt64

    private static let sidecarVersion = 1

    private init(
        name: String,
        path: String,
        schema: CollectionSchema,
        db: PistaDB,
        metric: Metric,
        indexType: IndexType,
        rows: [UInt64: [String: Any]],
        nextID: UInt64
    ) {
        self.name      = name
        self.path      = path
        self.metaPath  = path + ".meta.json"
        self.schema    = schema
        self.db        = db
        self.metric    = metric
        self.indexType = indexType
        self.rows      = rows
        self.nextID    = nextID
    }

    // MARK: Properties

    /// The underlying `PistaDB` handle (advanced use).
    public var database: PistaDB { db }

    public var numEntities: Int { db.count }

    // MARK: Lifecycle

    /// Persist both the .pst file and the JSON sidecar.
    public func flush() throws {
        try db.save()
        try saveSidecar()
    }

    /// Alias for `flush()`.
    public func save() throws { try flush() }

    public func close() { db.close() }

    // MARK: Insert

    /// Insert one or more rows.  Each row is a `[String: Any]` keyed by field
    /// name; the vector field accepts `[Float]` or `[Double]` or numeric `[Any]`.
    /// Returns the assigned primary ids in order.
    @discardableResult
    public func insert(_ rowsToInsert: [[String: Any]]) throws -> [UInt64] {
        let pk      = schema.primary
        let vec     = schema.vector
        let scalars = schema.scalarFields
        let known   = Set(schema.fields.map(\.name))

        var out = [UInt64]()
        for row in rowsToInsert {
            for k in row.keys where !known.contains(k) {
                throw PistaDBError.invalidArgument("unknown field '\(k)'")
            }

            let id: UInt64
            if pk.autoId {
                if let v = row[pk.name], !(v is NSNull) {
                    throw PistaDBError.invalidArgument(
                        "autoId enabled on '\(pk.name)' — do not supply it")
                }
                id = nextID
                nextID += 1
            } else {
                guard let v = row[pk.name], !(v is NSNull) else {
                    throw PistaDBError.invalidArgument("missing primary key '\(pk.name)'")
                }
                id = try Self.coerceUInt64(v)
                if rows[id] != nil {
                    throw PistaDBError.invalidArgument("duplicate primary id=\(id)")
                }
                if id >= nextID { nextID = id + 1 }
            }
            if id == 0 {
                throw PistaDBError.invalidArgument("primary key must be > 0")
            }

            guard let vraw = row[vec.name] else {
                throw PistaDBError.invalidArgument("missing vector field '\(vec.name)'")
            }
            let v = try Self.coerceFloatArray(vraw, dim: vec.dim ?? 0)

            var scalarVals: [String: Any] = [:]
            for f in scalars {
                if let raw = row[f.name], !(raw is NSNull) {
                    scalarVals[f.name] = try Self.coerceScalar(raw, field: f)
                } else {
                    scalarVals[f.name] = NSNull()
                }
            }

            try db.insert(id: id, vector: v, label: "")
            rows[id] = scalarVals
            out.append(id)
        }
        return out
    }

    // MARK: Delete / Get

    /// Delete a single row by primary id.  Returns true if it existed.
    @discardableResult
    public func delete(id: UInt64) -> Bool {
        do {
            try db.delete(id: id)
            rows.removeValue(forKey: id)
            return true
        } catch { return false }
    }

    /// Delete multiple rows; returns the number actually removed.
    @discardableResult
    public func delete(ids: [UInt64]) -> Int {
        ids.reduce(0) { $0 + (delete(id: $1) ? 1 : 0) }
    }

    /// Get the full row (all fields, including vector) by primary id.
    public func get(id: UInt64) throws -> [String: Any] {
        guard let meta = rows[id] else {
            throw PistaDBError.notFound
        }
        let entry = try db.get(id: id)
        var out = meta
        out[schema.primary.name] = id
        out[schema.vector.name]  = entry.vector
        return out
    }

    // MARK: Search

    /// k-NN search.  Pass `outputFields = nil` to project all scalar fields,
    /// or a list of names to project only those.
    public func search(
        _ query: [Float],
        k: Int = 10,
        outputFields: [String]? = nil
    ) throws -> [Hit] {
        let vec = schema.vector
        if query.count != (vec.dim ?? 0) {
            throw PistaDBError.invalidArgument(
                "query length \(query.count) != dim \(vec.dim ?? 0)")
        }

        let want: [String]
        if let outputFields {
            for n in outputFields where n != schema.primary.name && n != vec.name {
                _ = try schema.field(n)
            }
            want = outputFields
        } else {
            want = schema.scalarFields.map(\.name)
        }

        let raw = try db.search(query, k: k)
        var hits = [Hit]()
        hits.reserveCapacity(raw.count)
        for r in raw {
            let meta = rows[r.id]
            var fields: [String: Any] = [:]
            for n in want {
                if n == schema.primary.name {
                    fields[n] = r.id
                } else if n == vec.name {
                    if let entry = try? db.get(id: r.id) {
                        fields[n] = entry.vector
                    } else {
                        fields[n] = NSNull()
                    }
                } else {
                    fields[n] = meta?[n] ?? NSNull()
                }
            }
            hits.append(Hit(id: r.id, distance: r.distance, fields: fields))
        }
        return hits
    }

    // MARK: Sidecar I/O

    fileprivate func saveSidecar() throws {
        var sortedRows: [String: Any] = [:]
        for (k, v) in rows {
            sortedRows[String(k)] = v
        }

        let payload: [String: Any] = [
            "version":     Self.sidecarVersion,
            "name":        name,
            "description": schema.description,
            "metric":      Self.metricName(metric),
            "index":       Self.indexName(indexType),
            "next_id":     nextID,
            "fields":      schema.fields.map { $0.toJSON() },
            "rows":        sortedRows,
        ]

        let data: Data
        do {
            data = try JSONSerialization.data(
                withJSONObject: payload,
                options: [.prettyPrinted, .sortedKeys])
        } catch {
            throw PistaDBError.ioError("serialize sidecar: \(error.localizedDescription)")
        }

        let tmp = metaPath + ".tmp"
        do {
            try data.write(to: URL(fileURLWithPath: tmp), options: .atomic)
        } catch {
            throw PistaDBError.ioError("write \(tmp): \(error.localizedDescription)")
        }
        let fm = FileManager.default
        if fm.fileExists(atPath: metaPath) {
            try? fm.removeItem(atPath: metaPath)
        }
        do {
            try fm.moveItem(atPath: tmp, toPath: metaPath)
        } catch {
            throw PistaDBError.ioError("rename \(tmp) -> \(metaPath): \(error.localizedDescription)")
        }
    }

    // MARK: Static helpers

    fileprivate static func metricName(_ m: Metric) -> String {
        switch m {
        case .l2:      return "L2"
        case .cosine:  return "Cosine"
        case .ip:      return "IP"
        case .l1:      return "L1"
        case .hamming: return "Hamming"
        }
    }

    fileprivate static func metricFromName(_ s: String) throws -> Metric {
        switch s {
        case "L2":                  return .l2
        case "Cosine":              return .cosine
        case "IP", "InnerProduct":  return .ip
        case "L1":                  return .l1
        case "Hamming":             return .hamming
        default: throw PistaDBError.invalidArgument("unknown metric: \(s)")
        }
    }

    fileprivate static func indexName(_ i: IndexType) -> String {
        switch i {
        case .linear:  return "Linear"
        case .hnsw:    return "HNSW"
        case .ivf:     return "IVF"
        case .ivfPQ:   return "IVF_PQ"
        case .diskANN: return "DiskANN"
        case .lsh:     return "LSH"
        case .scaNN:   return "ScaNN"
        }
    }

    fileprivate static func indexFromName(_ s: String) throws -> IndexType {
        switch s {
        case "Linear":  return .linear
        case "HNSW":    return .hnsw
        case "IVF":     return .ivf
        case "IVF_PQ":  return .ivfPQ
        case "DiskANN": return .diskANN
        case "LSH":     return .lsh
        case "ScaNN":   return .scaNN
        default: throw PistaDBError.invalidArgument("unknown index type: \(s)")
        }
    }

    private static func coerceUInt64(_ v: Any) throws -> UInt64 {
        if let u = v as? UInt64 { return u }
        if let n = v as? NSNumber {
            let i = n.int64Value
            if i < 0 { throw PistaDBError.invalidArgument("negative primary key") }
            return UInt64(i)
        }
        if let s = v as? String, let u = UInt64(s) { return u }
        throw PistaDBError.invalidArgument("cannot coerce \(type(of: v)) to UInt64")
    }

    private static func coerceInt64(_ v: Any) throws -> Int64 {
        if let i = v as? Int64    { return i }
        if let i = v as? Int      { return Int64(i) }
        if let n = v as? NSNumber { return n.int64Value }
        if let s = v as? String, let i = Int64(s) { return i }
        throw PistaDBError.invalidArgument("cannot coerce \(type(of: v)) to Int64")
    }

    private static func coerceDouble(_ v: Any) throws -> Double {
        if let d = v as? Double   { return d }
        if let f = v as? Float    { return Double(f) }
        if let i = v as? Int      { return Double(i) }
        if let n = v as? NSNumber { return n.doubleValue }
        if let s = v as? String, let d = Double(s) { return d }
        throw PistaDBError.invalidArgument("cannot coerce \(type(of: v)) to Double")
    }

    private static func coerceFloatArray(_ v: Any, dim: Int) throws -> [Float] {
        if let f = v as? [Float] {
            guard f.count == dim else {
                throw PistaDBError.invalidArgument("vector length \(f.count) != dim \(dim)")
            }
            return f
        }
        if let d = v as? [Double] {
            guard d.count == dim else {
                throw PistaDBError.invalidArgument("vector length \(d.count) != dim \(dim)")
            }
            return d.map { Float($0) }
        }
        if let arr = v as? [Any] {
            guard arr.count == dim else {
                throw PistaDBError.invalidArgument("vector length \(arr.count) != dim \(dim)")
            }
            return try arr.map { Float(try coerceDouble($0)) }
        }
        if let arr = v as? [NSNumber] {
            guard arr.count == dim else {
                throw PistaDBError.invalidArgument("vector length \(arr.count) != dim \(dim)")
            }
            return arr.map { $0.floatValue }
        }
        throw PistaDBError.invalidArgument(
            "cannot coerce \(type(of: v)) to vector")
    }

    private static func coerceScalar(_ v: Any, field f: FieldSchema) throws -> Any {
        switch f.dtype {
        case .bool:
            if let b = v as? Bool { return b }
            throw PistaDBError.invalidArgument("field \(f.name): expected Bool")
        case .varchar:
            let s: String
            if let str = v as? String { s = str } else { s = "\(v)" }
            if let m = f.maxLength, s.utf8.count > m {
                throw PistaDBError.invalidArgument(
                    "field \(f.name): exceeds maxLength=\(m)")
            }
            return s
        case .json:
            return v
        case .floatVector:
            throw PistaDBError.invalidArgument("vector cannot appear as scalar")
        default:
            if f.dtype.isInt   { return try coerceInt64(v) }
            if f.dtype.isFloat { return try coerceDouble(v) }
            throw PistaDBError.invalidArgument("unsupported dtype \(f.dtype)")
        }
    }

    // MARK: Internal factory used by `createCollection` / `loadCollection`

    fileprivate static func make(
        name: String,
        path: String,
        schema: CollectionSchema,
        db: PistaDB,
        metric: Metric,
        indexType: IndexType,
        rows: [UInt64: [String: Any]],
        nextID: UInt64
    ) -> Collection {
        Collection(
            name:      name,
            path:      path,
            schema:    schema,
            db:        db,
            metric:    metric,
            indexType: indexType,
            rows:      rows,
            nextID:    nextID
        )
    }
}

// MARK: - Factories

/// Create a new collection.  Errors out if the .pst or .meta.json already
/// exist, unless `options.overwrite` is true.
public func createCollection(
    name: String,
    fields: [FieldSchema],
    description: String = "",
    options: CollectionOptions = CollectionOptions()
) throws -> Collection {
    let schema   = try CollectionSchema(fields, description: description)
    let pstPath  = options.resolvePath(name)
    let metaPath = pstPath + ".meta.json"
    let fm       = FileManager.default

    if options.overwrite {
        try? fm.removeItem(atPath: pstPath)
        try? fm.removeItem(atPath: metaPath)
    } else {
        if fm.fileExists(atPath: pstPath) {
            throw PistaDBError.invalidArgument("\(pstPath) already exists")
        }
        if fm.fileExists(atPath: metaPath) {
            throw PistaDBError.invalidArgument("\(metaPath) already exists")
        }
    }
    if let dir = (pstPath as NSString).deletingLastPathComponent.nilIfEmpty,
       !fm.fileExists(atPath: dir) {
        try fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
    }

    let db = try PistaDB(
        path:      pstPath,
        dim:       schema.vector.dim ?? 0,
        metric:    options.metric,
        indexType: options.indexType,
        params:    options.params)

    let coll = Collection.make(
        name: name, path: pstPath, schema: schema, db: db,
        metric: options.metric, indexType: options.indexType,
        rows: [:], nextID: 1)
    try coll.saveSidecar()
    return coll
}

/// Re-open a previously created collection.
public func loadCollection(
    name: String,
    options: CollectionOptions = CollectionOptions()
) throws -> Collection {
    let pstPath  = options.resolvePath(name)
    let metaPath = pstPath + ".meta.json"
    let fm       = FileManager.default

    guard fm.fileExists(atPath: metaPath) else {
        throw PistaDBError.invalidArgument("sidecar not found: \(metaPath)")
    }

    let data: Data
    do {
        data = try Data(contentsOf: URL(fileURLWithPath: metaPath))
    } catch {
        throw PistaDBError.ioError("read \(metaPath): \(error.localizedDescription)")
    }
    guard
        let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
        throw PistaDBError.invalidArgument("sidecar: not a JSON object")
    }

    guard let fa = parsed["fields"] as? [[String: Any]] else {
        throw PistaDBError.invalidArgument("sidecar: missing 'fields' array")
    }
    var fields = [FieldSchema]()
    fields.reserveCapacity(fa.count)
    for o in fa { fields.append(try FieldSchema.fromJSON(o)) }

    let schema = try CollectionSchema(
        fields,
        description: parsed["description"] as? String ?? "")

    guard
        let mName = parsed["metric"] as? String,
        let iName = parsed["index"]  as? String
    else {
        throw PistaDBError.invalidArgument("sidecar: missing metric/index")
    }
    let metric    = try Collection.metricFromName(mName)
    let indexType = try Collection.indexFromName(iName)

    let nextID: UInt64 = {
        if let n = parsed["next_id"] as? NSNumber { return n.uint64Value }
        return 1
    }()
    let savedName = parsed["name"] as? String ?? name

    var rows = [UInt64: [String: Any]]()
    if let jr = parsed["rows"] as? [String: [String: Any]] {
        for (k, v) in jr {
            if let id = UInt64(k) { rows[id] = v }
        }
    }

    let db = try PistaDB(
        path:      pstPath,
        dim:       schema.vector.dim ?? 0,
        metric:    metric,
        indexType: indexType,
        params:    options.params)

    return Collection.make(
        name: savedName, path: pstPath, schema: schema, db: db,
        metric: metric, indexType: indexType,
        rows: rows, nextID: max(nextID, 1))
}

// MARK: - Internal helpers

private extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}
