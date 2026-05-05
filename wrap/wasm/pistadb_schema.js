/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * pistadb_schema.js — Milvus-style schema layer for the WASM binding.
 *
 * Pure-JS module that wraps an initialised PistaDB WebAssembly Module:
 *
 *   import PistaDB from './pistadb.js';
 *   import { attachSchema } from './pistadb_schema.js';
 *
 *   const M = await PistaDB();
 *   attachSchema(M);   // adds M.DataType, M.FieldSchema, M.Collection,
 *                      // M.createCollection, M.loadCollection
 *
 *   const fields = [
 *       new M.FieldSchema('lc_id',     M.DataType.INT64,  { isPrimary: true, autoId: true }),
 *       new M.FieldSchema('lc_section',M.DataType.VARCHAR,{ maxLength: 100 }),
 *       new M.FieldSchema('lc_vector', M.DataType.FLOAT_VECTOR, { dim: 1536 }),
 *   ];
 *   const coll = M.createCollection('common_text', fields, 'Common text search', {
 *       metric: M.Metric.Cosine, indexType: M.IndexType.HNSW,
 *   });
 *   const ids = coll.insert([
 *       { lc_section: 'common', lc_vector: new Float32Array(1536) },
 *   ]);
 *   const hits = coll.search(query, 10);
 *   coll.flush();
 *   coll.close();
 *
 * The vector field is stored in the underlying .pst file via the C library;
 * remaining scalar fields go to a JSON sidecar (`<path>.meta.json`) on the
 * Emscripten virtual filesystem.
 */

// ── DataType ──────────────────────────────────────────────────────────────────

const DataType = Object.freeze({
    BOOL:         1,
    INT8:         2,
    INT16:        3,
    INT32:        4,
    INT64:        5,
    FLOAT:       10,
    DOUBLE:      11,
    VARCHAR:     21,
    JSON:        23,
    FLOAT_VECTOR: 101,
});

const _wireByCode = Object.freeze({
    [DataType.BOOL]:         'BOOL',
    [DataType.INT8]:         'INT8',
    [DataType.INT16]:        'INT16',
    [DataType.INT32]:        'INT32',
    [DataType.INT64]:        'INT64',
    [DataType.FLOAT]:        'FLOAT',
    [DataType.DOUBLE]:       'DOUBLE',
    [DataType.VARCHAR]:      'VARCHAR',
    [DataType.JSON]:         'JSON',
    [DataType.FLOAT_VECTOR]: 'FLOAT_VECTOR',
});

const _codeByWire = Object.freeze(
    Object.fromEntries(Object.entries(_wireByCode).map(([k, v]) => [v, Number(k)])),
);

function _isInt(d)   { return d === DataType.INT8 || d === DataType.INT16 ||
                              d === DataType.INT32 || d === DataType.INT64; }
function _isFloat(d) { return d === DataType.FLOAT || d === DataType.DOUBLE; }

// ── FieldSchema ───────────────────────────────────────────────────────────────

class FieldSchema {
    /**
     * @param {string} name
     * @param {number} dtype  one of DataType.*
     * @param {object} [opts] { isPrimary, autoId, maxLength, dim, description }
     */
    constructor(name, dtype, opts = {}) {
        if (!name || typeof name !== 'string')
            throw new Error('FieldSchema: name must be a non-empty string');
        if (!_wireByCode[dtype])
            throw new Error(`FieldSchema: unknown dtype ${dtype}`);

        this.name        = name;
        this.dtype       = dtype;
        this.isPrimary   = !!opts.isPrimary;
        this.autoId      = !!opts.autoId;
        this.maxLength   = opts.maxLength ?? null;
        this.dim         = opts.dim       ?? null;
        this.description = opts.description ?? '';

        if (dtype === DataType.FLOAT_VECTOR && (!this.dim || this.dim <= 0))
            throw new Error(`FLOAT_VECTOR field '${name}' requires a positive dim`);
        if (dtype === DataType.VARCHAR && this.maxLength != null && this.maxLength <= 0)
            throw new Error(`VARCHAR field '${name}': maxLength must be positive`);
        if (this.isPrimary && dtype !== DataType.INT64)
            throw new Error(`primary key '${name}' must be INT64`);
        if (this.autoId && !this.isPrimary)
            throw new Error(`autoId only valid on the primary field (got '${name}')`);
    }

    _toJSON() {
        const o = {
            name:       this.name,
            dtype:      _wireByCode[this.dtype],
            is_primary: this.isPrimary,
            auto_id:    this.autoId,
        };
        if (this.maxLength != null)            o.max_length  = this.maxLength;
        if (this.dim != null)                  o.dim         = this.dim;
        if (this.description)                  o.description = this.description;
        return o;
    }

    static _fromJSON(o) {
        const code = _codeByWire[o.dtype];
        if (code == null) throw new Error(`unknown DataType: ${o.dtype}`);
        return new FieldSchema(o.name, code, {
            isPrimary:   !!o.is_primary,
            autoId:      !!o.auto_id,
            maxLength:   o.max_length ?? null,
            dim:         o.dim        ?? null,
            description: o.description ?? '',
        });
    }
}

// ── CollectionSchema ──────────────────────────────────────────────────────────

class CollectionSchema {
    /**
     * @param {FieldSchema[]} fields
     * @param {string} [description]
     */
    constructor(fields, description = '') {
        if (!Array.isArray(fields) || fields.length === 0)
            throw new Error('CollectionSchema: at least one field required');

        const seen = new Set();
        let primary = null, vector = null;
        for (const f of fields) {
            if (!(f instanceof FieldSchema))
                throw new Error('CollectionSchema: each entry must be a FieldSchema');
            if (seen.has(f.name)) throw new Error(`duplicate field name '${f.name}'`);
            seen.add(f.name);
            if (f.isPrimary) {
                if (primary) throw new Error('schema must have exactly one primary key');
                primary = f;
            }
            if (f.dtype === DataType.FLOAT_VECTOR) {
                if (vector) throw new Error('schema must have exactly one FLOAT_VECTOR field');
                vector = f;
            }
        }
        if (!primary) throw new Error('schema must have a primary key');
        if (!vector)  throw new Error('schema must have a FLOAT_VECTOR field');

        this.fields      = fields.slice();
        this.description = description;
        this.primary     = primary;
        this.vector      = vector;
    }

    get scalarFields() {
        return this.fields.filter(
            f => !f.isPrimary && f.dtype !== DataType.FLOAT_VECTOR);
    }

    field(name) {
        const f = this.fields.find(x => x.name === name);
        if (!f) throw new Error(`no field named '${name}'`);
        return f;
    }
}

// ── Hit (search row) ──────────────────────────────────────────────────────────

class Hit {
    constructor(id, distance, fields) {
        this.id       = id;
        this.distance = distance;
        this.fields   = fields;
    }
    get(name) { return this.fields[name]; }
}

// ── Sidecar / FS helpers ──────────────────────────────────────────────────────

const SIDECAR_VERSION = 1;
const TEXT_ENCODER    = (typeof TextEncoder !== 'undefined') ? new TextEncoder() : null;
const TEXT_DECODER    = (typeof TextDecoder !== 'undefined') ? new TextDecoder() : null;

function _writeFile(M, path, str) {
    if (TEXT_ENCODER) {
        M.FS.writeFile(path, TEXT_ENCODER.encode(str));
    } else {
        M.FS.writeFile(path, str);
    }
}

function _readTextFile(M, path) {
    const data = M.FS.readFile(path);
    if (typeof data === 'string') return data;
    if (TEXT_DECODER) return TEXT_DECODER.decode(data);
    // Fallback for older runtimes.
    let s = '';
    for (const b of data) s += String.fromCharCode(b);
    return s;
}

function _exists(M, path) {
    try { M.FS.lookupPath(path); return true; }
    catch (_) { return false; }
}

function _unlinkIfExists(M, path) {
    try { M.FS.unlink(path); } catch (_) {}
}

function _ensureParent(M, path) {
    const idx = path.lastIndexOf('/');
    if (idx <= 0) return;
    const parts = path.slice(0, idx).split('/');
    let cur = '';
    for (const p of parts) {
        if (!p) { cur += '/'; continue; }
        cur += (cur && cur !== '/' ? '/' : '') + p;
        if (!_exists(M, cur)) {
            try { M.FS.mkdir(cur); } catch (_) {}
        }
    }
}

function _resolvePath(name, opts) {
    if (opts.path) return opts.path;
    if (opts.baseDir) {
        const d = opts.baseDir.replace(/\/+$/, '');
        return `${d}/${name}.pst`;
    }
    return `${name}.pst`;
}

function _metricName(M, m) {
    switch (m) {
        case M.Metric.L2:      return 'L2';
        case M.Metric.Cosine:  return 'Cosine';
        case M.Metric.IP:      return 'IP';
        case M.Metric.L1:      return 'L1';
        case M.Metric.Hamming: return 'Hamming';
    }
    throw new Error(`unknown metric value: ${m}`);
}

function _metricFromName(M, s) {
    switch (s) {
        case 'L2':                  return M.Metric.L2;
        case 'Cosine':              return M.Metric.Cosine;
        case 'IP': case 'InnerProduct': return M.Metric.IP;
        case 'L1':                  return M.Metric.L1;
        case 'Hamming':             return M.Metric.Hamming;
    }
    throw new Error(`unknown metric: ${s}`);
}

function _indexName(M, i) {
    switch (i) {
        case M.IndexType.Linear:  return 'Linear';
        case M.IndexType.HNSW:    return 'HNSW';
        case M.IndexType.IVF:     return 'IVF';
        case M.IndexType.IVF_PQ:  return 'IVF_PQ';
        case M.IndexType.DiskANN: return 'DiskANN';
        case M.IndexType.LSH:     return 'LSH';
        case M.IndexType.ScaNN:   return 'ScaNN';
    }
    throw new Error(`unknown index type value: ${i}`);
}

function _indexFromName(M, s) {
    switch (s) {
        case 'Linear':  return M.IndexType.Linear;
        case 'HNSW':    return M.IndexType.HNSW;
        case 'IVF':     return M.IndexType.IVF;
        case 'IVF_PQ':  return M.IndexType.IVF_PQ;
        case 'DiskANN': return M.IndexType.DiskANN;
        case 'LSH':     return M.IndexType.LSH;
        case 'ScaNN':   return M.IndexType.ScaNN;
    }
    throw new Error(`unknown index type: ${s}`);
}

// ── Coercion ──────────────────────────────────────────────────────────────────

function _coerceFloat32Array(v, dim) {
    if (v instanceof Float32Array) {
        if (v.length !== dim) throw new Error(`vector length ${v.length} != dim ${dim}`);
        return v;
    }
    if (v instanceof Float64Array) {
        if (v.length !== dim) throw new Error(`vector length ${v.length} != dim ${dim}`);
        return Float32Array.from(v);
    }
    if (Array.isArray(v)) {
        if (v.length !== dim) throw new Error(`vector length ${v.length} != dim ${dim}`);
        return Float32Array.from(v.map(Number));
    }
    throw new Error(`cannot coerce ${typeof v} to vector`);
}

function _coerceScalar(v, f) {
    if (v == null) return null;
    if (f.dtype === DataType.BOOL) {
        if (typeof v === 'boolean') return v;
        throw new Error(`field '${f.name}': expected boolean`);
    }
    if (f.dtype === DataType.VARCHAR) {
        const s = (typeof v === 'string') ? v : String(v);
        if (f.maxLength != null && s.length > f.maxLength) {
            // Note: pymilvus measures bytes; here we approximate with code-unit length.
            throw new Error(`field '${f.name}': exceeds maxLength=${f.maxLength}`);
        }
        return s;
    }
    if (f.dtype === DataType.JSON) return v;
    if (f.dtype === DataType.FLOAT_VECTOR)
        throw new Error(`vector cannot appear as scalar`);
    if (_isInt(f.dtype)) {
        const n = Number(v);
        if (!Number.isFinite(n)) throw new Error(`field '${f.name}': not a number`);
        return Math.trunc(n);
    }
    if (_isFloat(f.dtype)) {
        const n = Number(v);
        if (!Number.isFinite(n)) throw new Error(`field '${f.name}': not a number`);
        return n;
    }
    throw new Error(`unsupported dtype for field '${f.name}'`);
}

// ── Collection ────────────────────────────────────────────────────────────────

class Collection {
    /** @internal */
    constructor(M, opts) {
        this._M       = M;
        this.name     = opts.name;
        this.path     = opts.path;
        this._meta    = opts.path + '.meta.json';
        this.schema   = opts.schema;
        this._db      = opts.db;
        this._metric  = opts.metric;
        this._index   = opts.indexType;
        this._rows    = opts.rows || {};
        this._nextId  = opts.nextId ?? 1;
    }

    /** Number of active (non-deleted) rows. */
    get numEntities() { return this._db.count(); }

    /** Underlying Database handle (advanced use). */
    get database() { return this._db; }

    /** Persist both the .pst and the JSON sidecar. */
    flush() {
        this._db.save();
        this._saveSidecar();
    }

    /** Alias for flush(). */
    save() { this.flush(); }

    /** Free the underlying C++ Database object.  Does not auto-save. */
    close() {
        if (this._db) {
            this._db.delete();
            this._db = null;
        }
    }

    // ── Insert ────────────────────────────────────────────────────────────

    /**
     * Insert one or more rows.  Returns an array of assigned primary ids.
     *
     * @param {Array<object>} rows  Each row is an object keyed by field name;
     *                              the vector field accepts a Float32Array,
     *                              Float64Array, or numeric array.
     */
    insert(rows) {
        if (!Array.isArray(rows)) throw new Error('insert: rows must be an array');
        const pk      = this.schema.primary;
        const vec     = this.schema.vector;
        const scalars = this.schema.scalarFields;
        const known   = new Set(this.schema.fields.map(f => f.name));

        const out = [];
        for (const row of rows) {
            for (const k of Object.keys(row))
                if (!known.has(k))
                    throw new Error(`unknown field '${k}'`);

            // Primary id
            let id;
            if (pk.autoId) {
                if (row[pk.name] != null)
                    throw new Error(`autoId enabled on '${pk.name}' — do not supply it`);
                id = this._nextId++;
            } else {
                if (row[pk.name] == null)
                    throw new Error(`missing primary key '${pk.name}'`);
                id = Number(row[pk.name]);
                if (!Number.isFinite(id) || id <= 0 || Math.floor(id) !== id)
                    throw new Error(`primary key must be a positive integer`);
                if (this._rows[id] !== undefined)
                    throw new Error(`duplicate primary id=${id}`);
                if (id >= this._nextId) this._nextId = id + 1;
            }

            // Vector
            if (row[vec.name] == null)
                throw new Error(`missing vector field '${vec.name}'`);
            const v = _coerceFloat32Array(row[vec.name], vec.dim);

            // Scalars
            const scalarVals = {};
            for (const f of scalars) {
                scalarVals[f.name] = (row[f.name] == null)
                    ? null
                    : _coerceScalar(row[f.name], f);
            }

            this._db.insert(id, v, '');
            this._rows[id] = scalarVals;
            out.push(id);
        }
        return out;
    }

    // ── Delete / Get ──────────────────────────────────────────────────────

    /** Delete one or more rows; returns the number actually removed. */
    delete(idOrIds) {
        const ids = Array.isArray(idOrIds) ? idOrIds : [idOrIds];
        let removed = 0;
        for (const id of ids) {
            const n = Number(id);
            try {
                this._db.remove(n);
                delete this._rows[n];
                removed += 1;
            } catch (_) { /* ignore missing */ }
        }
        return removed;
    }

    /** Get the full row (all fields, including vector) by primary id. */
    get(id) {
        const n = Number(id);
        const meta = this._rows[n];
        if (!meta) throw new Error(`id=${n} not found`);
        const entry = this._db.get(n);
        const out = Object.assign({}, meta);
        out[this.schema.primary.name] = n;
        out[this.schema.vector.name]  = entry.vector;
        return out;
    }

    // ── Search ────────────────────────────────────────────────────────────

    /**
     * k-NN search.
     * @param {Float32Array|number[]} query
     * @param {number} k
     * @param {string[]} [outputFields]  Pass undefined for all scalar fields.
     */
    search(query, k, outputFields) {
        const vec = this.schema.vector;
        const q   = _coerceFloat32Array(query, vec.dim);

        const want = (outputFields == null)
            ? this.schema.scalarFields.map(f => f.name)
            : outputFields.slice();

        if (outputFields != null) {
            for (const n of outputFields)
                if (n !== this.schema.primary.name && n !== vec.name)
                    this.schema.field(n);   // throws if unknown
        }

        const raw = this._db.search(q, k);
        const out = [];
        const pkName  = this.schema.primary.name;
        const vecName = vec.name;
        for (const r of raw) {
            const meta = this._rows[r.id];
            const fields = {};
            for (const n of want) {
                if (n === pkName)        fields[n] = r.id;
                else if (n === vecName)  fields[n] = this._db.get(r.id).vector;
                else                      fields[n] = meta ? meta[n] ?? null : null;
            }
            out.push(new Hit(r.id, r.distance, fields));
        }
        return out;
    }

    // ── Sidecar I/O ───────────────────────────────────────────────────────

    _saveSidecar() {
        const sortedRows = {};
        for (const k of Object.keys(this._rows).sort((a, b) => Number(a) - Number(b)))
            sortedRows[k] = this._rows[k];

        const payload = {
            version:     SIDECAR_VERSION,
            name:        this.name,
            description: this.schema.description,
            metric:      _metricName(this._M, this._metric),
            index:       _indexName(this._M, this._index),
            next_id:     this._nextId,
            fields:      this.schema.fields.map(f => f._toJSON()),
            rows:        sortedRows,
        };

        const json = JSON.stringify(payload, null, 2);
        const tmp  = this._meta + '.tmp';
        _ensureParent(this._M, tmp);
        _writeFile(this._M, tmp, json);
        _unlinkIfExists(this._M, this._meta);
        this._M.FS.rename(tmp, this._meta);
    }
}

// ── Factories ────────────────────────────────────────────────────────────────

/**
 * Create a new collection.  Throws if files already exist unless
 * `opts.overwrite` is true.
 *
 * @param {object} M           initialised PistaDB module
 * @param {string} name
 * @param {FieldSchema[]} fields
 * @param {string} [description]
 * @param {object} [opts] { metric, indexType, params, baseDir, path, overwrite }
 */
function createCollection(M, name, fields, description = '', opts = {}) {
    const schema   = new CollectionSchema(fields, description);
    const path     = _resolvePath(name, opts);
    const metaPath = path + '.meta.json';
    const metric   = opts.metric    ?? M.Metric.L2;
    const index    = opts.indexType ?? M.IndexType.HNSW;
    const params   = opts.params    ?? null;

    if (opts.overwrite) {
        _unlinkIfExists(M, path);
        _unlinkIfExists(M, metaPath);
    } else {
        if (_exists(M, path))     throw new Error(`${path} already exists`);
        if (_exists(M, metaPath)) throw new Error(`${metaPath} already exists`);
    }
    _ensureParent(M, path);

    const db = new M.Database(path, schema.vector.dim, metric, index, params);

    const coll = new Collection(M, {
        name, path, schema, db,
        metric, indexType: index,
        rows: {}, nextId: 1,
    });
    coll._saveSidecar();
    return coll;
}

/**
 * Re-open an existing collection from disk.
 */
function loadCollection(M, name, opts = {}) {
    const path     = _resolvePath(name, opts);
    const metaPath = path + '.meta.json';
    if (!_exists(M, metaPath))
        throw new Error(`sidecar not found: ${metaPath}`);

    const sc       = JSON.parse(_readTextFile(M, metaPath));
    const fields   = (sc.fields || []).map(FieldSchema._fromJSON);
    const schema   = new CollectionSchema(fields, sc.description ?? '');
    const metric   = _metricFromName(M, sc.metric);
    const index    = _indexFromName(M,  sc.index);
    const nextId   = Math.max(Number(sc.next_id ?? 1), 1);
    const savedNm  = sc.name ?? name;

    const rows = {};
    if (sc.rows && typeof sc.rows === 'object') {
        for (const [k, v] of Object.entries(sc.rows)) rows[Number(k)] = v;
    }

    const db = new M.Database(path, schema.vector.dim, metric, index, opts.params ?? null);
    return new Collection(M, {
        name: savedNm, path, schema, db,
        metric, indexType: index,
        rows, nextId,
    });
}

/**
 * Attach the schema API onto an initialised PistaDB module so callers can
 * write `M.createCollection(...)` instead of `createCollection(M, ...)`.
 */
function attachSchema(M) {
    M.DataType         = DataType;
    M.FieldSchema      = FieldSchema;
    M.CollectionSchema = CollectionSchema;
    M.Hit              = Hit;
    M.Collection       = Collection;
    M.createCollection = (name, fields, description, opts) =>
        createCollection(M, name, fields, description, opts);
    M.loadCollection   = (name, opts) =>
        loadCollection(M, name, opts);
    return M;
}

// ── Exports (UMD-ish: works in ESM and CJS) ──────────────────────────────────

const _api = {
    DataType,
    FieldSchema,
    CollectionSchema,
    Hit,
    Collection,
    createCollection,
    loadCollection,
    attachSchema,
};

// CommonJS
if (typeof module !== 'undefined' && module.exports) {
    module.exports = _api;
}

// ESM (re-exports also work when consumed via `import`)
export {
    DataType,
    FieldSchema,
    CollectionSchema,
    Hit,
    Collection,
    createCollection,
    loadCollection,
    attachSchema,
};
export default _api;
