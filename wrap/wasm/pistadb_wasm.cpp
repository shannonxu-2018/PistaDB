/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * pistadb_wasm.cpp — Emscripten / Embind binding for PistaDB
 *
 * Exposes the full PistaDB C API to JavaScript/TypeScript via Embind.
 * Compiled with emcmake / emmake — see build.sh.
 *
 * Generated output:
 *   pistadb.js   — module factory (async, returns a Promise<PistaDBModule>)
 *   pistadb.wasm — WebAssembly binary
 *
 * Usage (ESM):
 *   import PistaDB from './pistadb.js';
 *   const M = await PistaDB();
 *   const db = new M.Database('data.pst', 128, M.Metric.Cosine, M.IndexType.HNSW, null);
 *   db.insert(1, new Float32Array(128).fill(0.1), 'hello');
 *   const results = db.search(new Float32Array(128).fill(0.1), 5);
 *   db.save();
 *   db.delete();   // free C++ object (Embind RAII)
 */

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// C API — add PistaDB/src to your include path (CMakeLists.txt does this).
#include "pistadb.h"

using namespace emscripten;

// ── Memory helpers ────────────────────────────────────────────────────────────

/**
 * Efficiently copy a JavaScript Float32Array into a C++ std::vector<float>.
 *
 * Uses TypedArray.set() to bulk-copy via the WASM heap view — equivalent to
 * a memcpy, not element-by-element iteration.
 */
static std::vector<float> js_to_floats(const val& arr) {
    const size_t n = arr["length"].as<size_t>();
    std::vector<float> v(n);
    if (n == 0) return v;

    // Create a Float32Array view into WASM linear memory at our vector's address.
    val view = val::global("Float32Array").new_(
        val::module_property("HEAPU8")["buffer"],
        reinterpret_cast<uintptr_t>(v.data()),
        static_cast<unsigned>(n));

    // TypedArray.set() does a fast bulk copy from the JS side.
    view.call<void>("set", arr);
    return v;
}

/**
 * Copy a C++ vector<float> into a new, owned JS Float32Array.
 *
 * The returned Float32Array owns its memory independently of the WASM heap.
 */
static val floats_to_js(const float* data, size_t n) {
    val arr = val::global("Float32Array").new_(static_cast<unsigned>(n));
    if (n == 0) return arr;

    // Create a temporary view into WASM heap memory, then .set() into arr.
    val view = val::global("Float32Array").new_(
        val::module_property("HEAPU8")["buffer"],
        reinterpret_cast<uintptr_t>(data),
        static_cast<unsigned>(n));
    arr.call<void>("set", view);
    return arr;
}

// ── Parameter extraction ──────────────────────────────────────────────────────

/**
 * Build a PistaDBParams from a JavaScript object.
 *
 * Any field that is present and numeric overrides the library default.
 * Pass null or undefined to use all defaults.
 *
 * Accepted JS keys (camelCase or snake_case variants shown in pistadb.d.ts):
 *   hnsw_m, hnsw_ef_construction, hnsw_ef_search,
 *   ivf_nlist, ivf_nprobe, pq_m, pq_nbits,
 *   diskann_r, diskann_l, diskann_alpha,
 *   lsh_l, lsh_k, lsh_w,
 *   scann_nlist, scann_nprobe, scann_pq_m, scann_pq_bits,
 *   scann_rerank_k, scann_aq_eta
 */
static PistaDBParams extract_params(const val& obj) {
    PistaDBParams p = pistadb_default_params();
    if (obj.isNull() || obj.isUndefined()) return p;

    auto gi = [&](const char* key, int& dst) {
        val v = obj[key]; if (v.isNumber()) dst = v.as<int>();
    };
    auto gf = [&](const char* key, float& dst) {
        val v = obj[key]; if (v.isNumber()) dst = v.as<float>();
    };

    gi("hnsw_m",               p.hnsw_M);
    gi("hnsw_ef_construction", p.hnsw_ef_construction);
    gi("hnsw_ef_search",       p.hnsw_ef_search);
    gi("ivf_nlist",            p.ivf_nlist);
    gi("ivf_nprobe",           p.ivf_nprobe);
    gi("pq_m",                 p.pq_M);
    gi("pq_nbits",             p.pq_nbits);
    gi("diskann_r",            p.diskann_R);
    gi("diskann_l",            p.diskann_L);
    gf("diskann_alpha",        p.diskann_alpha);
    gi("lsh_l",                p.lsh_L);
    gi("lsh_k",                p.lsh_K);
    gf("lsh_w",                p.lsh_w);
    gi("scann_nlist",          p.scann_nlist);
    gi("scann_nprobe",         p.scann_nprobe);
    gi("scann_pq_m",           p.scann_pq_M);
    gi("scann_pq_bits",        p.scann_pq_bits);
    gi("scann_rerank_k",       p.scann_rerank_k);
    gf("scann_aq_eta",         p.scann_aq_eta);
    return p;
}

// ── DatabaseJS wrapper ────────────────────────────────────────────────────────

/**
 * JavaScript-facing wrapper over the native PistaDB C API.
 *
 * RAII: the C handle is opened in the constructor and closed in the destructor.
 * Call .delete() from JavaScript to trigger the destructor and free memory.
 *
 * IDs are accepted as JS numbers (double). Safe for values up to 2^53.
 */
class DatabaseJS {
public:
    /**
     * @param path        File path on the Emscripten virtual filesystem.
     * @param dim         Vector dimension.
     * @param metric      Distance metric (use M.Metric.* constants).
     * @param index_type  Index algorithm (use M.IndexType.* constants).
     * @param params_obj  JS object with optional parameter overrides, or null.
     */
    DatabaseJS(const std::string& path,
               int dim,
               int metric,
               int index_type,
               val params_obj)
    {
        PistaDBParams p = extract_params(params_obj);
        db_ = pistadb_open(path.c_str(), dim, metric, index_type, &p);
        if (!db_) {
            throw std::runtime_error(
                std::string("pistadb_open failed — ") + safe_error());
        }
    }

    ~DatabaseJS() {
        if (db_) { pistadb_close(db_); db_ = nullptr; }
    }

    DatabaseJS(const DatabaseJS&)            = delete;
    DatabaseJS& operator=(const DatabaseJS&) = delete;

    // ── Lifecycle ──────────────────────────────────────────────────────────

    /** Persist the database to the .pst file. */
    void save() { chk(pistadb_save(db_)); }

    // ── CRUD ──────────────────────────────────────────────────────────────

    /**
     * Insert a vector.
     * @param id    Unique numeric id (safe for values < 2^53).
     * @param vec   Float32Array of length dim.
     * @param label Optional string label (pass "" for none).
     */
    void insert(double id, val vec_js, const std::string& label) {
        auto vec = js_to_floats(vec_js);
        const char* lbl = label.empty() ? nullptr : label.c_str();
        chk(pistadb_insert(db_, static_cast<uint64_t>(id), lbl, vec.data()));
    }

    /**
     * Logically delete a vector by id.
     * @note Named 'remove' — 'delete' is reserved in JavaScript.
     */
    void remove(double id) {
        chk(pistadb_delete(db_, static_cast<uint64_t>(id)));
    }

    /** Replace the vector data for an existing id. */
    void update(double id, val vec_js) {
        auto vec = js_to_floats(vec_js);
        chk(pistadb_update(db_, static_cast<uint64_t>(id), vec.data()));
    }

    /**
     * Retrieve the stored vector and label for a given id.
     * @returns { vector: Float32Array, label: string }
     */
    val get(double id) {
        const int d = pistadb_dim(db_);
        std::vector<float> vec(d);
        char label_buf[256] = {};
        chk(pistadb_get(db_, static_cast<uint64_t>(id), vec.data(), label_buf));

        val result = val::object();
        result.set("vector", floats_to_js(vec.data(), d));
        result.set("label",  std::string(label_buf));
        return result;
    }

    // ── Search ────────────────────────────────────────────────────────────

    /**
     * K-nearest-neighbour search.
     * @param query Float32Array of length dim.
     * @param k     Number of results requested.
     * @returns     Array of { id: number, distance: number, label: string }
     */
    val search(val query_js, int k) {
        auto query = js_to_floats(query_js);
        std::vector<PistaDBResult> buf(static_cast<size_t>(k));
        const int n = pistadb_search(db_, query.data(), k, buf.data());
        if (n < 0) throw std::runtime_error(safe_error());

        val results = val::array();
        for (int i = 0; i < n; ++i) {
            val r = val::object();
            // Convert uint64_t id to double (safe for id < 2^53)
            r.set("id",       static_cast<double>(buf[i].id));
            r.set("distance", buf[i].distance);
            r.set("label",    std::string(buf[i].label));
            results.call<void>("push", r);
        }
        return results;
    }

    // ── Index management ──────────────────────────────────────────────────

    /**
     * Train the index. Required before insert() for IVF and IVF_PQ.
     * Optional for HNSW/DiskANN — triggers a rebuild pass.
     */
    void train() { chk(pistadb_train(db_)); }

    // ── Properties ────────────────────────────────────────────────────────

    int count()           const { return pistadb_count(db_);      }
    int dim()             const { return pistadb_dim(db_);         }
    int metric()          const { return pistadb_metric(db_);      }
    int indexType()       const { return pistadb_index_type(db_);  }
    std::string lastError() const { return safe_error();           }

    static std::string version() {
        const char* v = pistadb_version();
        return v ? v : "";
    }

private:
    PistaDB* db_ = nullptr;

    std::string safe_error() const {
        if (!db_) return "database not open";
        const char* msg = pistadb_last_error(db_);
        return (msg && *msg) ? msg : "unknown error";
    }

    void chk(int rc) {
        if (rc != 0) throw std::runtime_error(safe_error());
    }
};

// ── Embind registration ───────────────────────────────────────────────────────

EMSCRIPTEN_BINDINGS(pistadb) {

    // ── Metric enum ───────────────────────────────────────────────────────
    enum_<PistaDBMetric>("Metric")
        .value("L2",      METRIC_L2)
        .value("Cosine",  METRIC_COSINE)
        .value("IP",      METRIC_IP)
        .value("L1",      METRIC_L1)
        .value("Hamming", METRIC_HAMMING);

    // ── IndexType enum ────────────────────────────────────────────────────
    enum_<PistaDBIndexType>("IndexType")
        .value("Linear",  INDEX_LINEAR)
        .value("HNSW",    INDEX_HNSW)
        .value("IVF",     INDEX_IVF)
        .value("IVF_PQ",  INDEX_IVF_PQ)
        .value("DiskANN", INDEX_DISKANN)
        .value("LSH",     INDEX_LSH)
        .value("ScaNN",   INDEX_SCANN);

    // ── Database class ────────────────────────────────────────────────────
    class_<DatabaseJS>("Database")
        .constructor<std::string, int, int, int, val>()
        .function("save",       &DatabaseJS::save)
        .function("insert",     &DatabaseJS::insert)
        .function("remove",     &DatabaseJS::remove)
        .function("update",     &DatabaseJS::update)
        .function("get",        &DatabaseJS::get)
        .function("search",     &DatabaseJS::search)
        .function("train",      &DatabaseJS::train)
        .function("count",      &DatabaseJS::count)
        .function("dim",        &DatabaseJS::dim)
        .function("metric",     &DatabaseJS::metric)
        .function("indexType",  &DatabaseJS::indexType)
        .function("lastError",  &DatabaseJS::lastError)
        .class_function("version", &DatabaseJS::version);

    // ── Filesystem helpers exposed to JS ──────────────────────────────────
    // Users can call Module.FS.* directly for advanced filesystem operations.
    // Example: mounting IDBFS for persistent browser storage.
}
