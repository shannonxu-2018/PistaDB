/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * pistadb.hpp — C++17 header-only wrapper for PistaDB
 *
 * Requirements:
 *   - C++17 or later
 *   - PistaDB's src/ directory on your include path (for pistadb.h)
 *   - Link against the compiled pistadb shared/static library
 *
 * Quick start:
 *   pistadb::Database db("my.pst", 128, pistadb::Metric::Cosine,
 *                        pistadb::IndexType::HNSW);
 *   db.insert(1, embedding, "hello world");
 *   auto results = db.search(query, 5);
 *   db.save();
 *   // ~Database() auto-closes — no need for explicit close()
 *
 * Thread safety:
 *   Database is thread-safe. An internal std::mutex serialises all native calls.
 *   Share across threads with std::shared_ptr<Database> or wrap in std::mutex yourself.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// ── C API include ─────────────────────────────────────────────────────────────
// The C header already has extern "C" guards, so a plain include works.
// To override the include path, define PISTADB_C_HEADER before including this file:
//   #define PISTADB_C_HEADER "path/to/pistadb.h"
#ifndef PISTADB_C_HEADER
#  define PISTADB_C_HEADER "pistadb.h"
#endif
#include PISTADB_C_HEADER

// ── Namespace ─────────────────────────────────────────────────────────────────

namespace pistadb {

// ── Enums ─────────────────────────────────────────────────────────────────────

/// Vector distance metric.
enum class Metric : int {
    L2      = METRIC_L2,      ///< Euclidean distance. Default.
    Cosine  = METRIC_COSINE,  ///< Cosine similarity (as distance). Ideal for text embeddings.
    IP      = METRIC_IP,      ///< Inner product (stored negative). For dot-product similarity.
    L1      = METRIC_L1,      ///< Manhattan distance.
    Hamming = METRIC_HAMMING, ///< Hamming distance. For binary/integer vectors.
};

/// ANN index algorithm.
enum class IndexType : int {
    Linear  = INDEX_LINEAR,   ///< Brute-force exact scan.
    HNSW    = INDEX_HNSW,     ///< Hierarchical NSW. Recommended for RAG.
    IVF     = INDEX_IVF,      ///< Inverted File Index. Requires train() before insert.
    IVF_PQ  = INDEX_IVF_PQ,  ///< IVF + Product Quantization. Low memory, lossy.
    DiskANN = INDEX_DISKANN,  ///< Vamana / DiskANN. Scales to billion vectors.
    LSH     = INDEX_LSH,      ///< Locality-Sensitive Hashing. Ultra-low memory.
    ScaNN   = INDEX_SCANN,    ///< Anisotropic Vector Quantization. Best recall.
};

// ── Result types ──────────────────────────────────────────────────────────────

/// A single KNN search result.
struct SearchResult {
    uint64_t    id;        ///< User-supplied vector id.
    float       distance;  ///< Distance from the query (lower = more similar).
    std::string label;     ///< Optional human-readable label.
};

/// A stored vector entry returned by Database::get().
struct VectorEntry {
    std::vector<float> vector; ///< Raw float vector.
    std::string        label;  ///< Optional label.
};

// ── Parameters ────────────────────────────────────────────────────────────────

/// Index tuning parameters. Unused fields for the chosen IndexType are ignored.
/// Default values match the library defaults from pistadb_default_params().
struct Params {
    // ── HNSW ──────────────────────────────────────────────────────────────
    int   hnsw_m               = 16;  ///< Max connections per layer.  Higher = better recall.
    int   hnsw_ef_construction = 200; ///< Build-time search width.    Higher = better graph.
    int   hnsw_ef_search       = 50;  ///< Query-time search width.    Higher = better recall.

    // ── IVF / IVF_PQ ──────────────────────────────────────────────────────
    int   ivf_nlist  = 128; ///< Number of IVF centroids.
    int   ivf_nprobe = 8;   ///< Centroids to search at query time.
    int   pq_m       = 8;   ///< PQ sub-spaces.
    int   pq_nbits   = 8;   ///< Bits per PQ sub-code (4 or 8).

    // ── DiskANN / Vamana ──────────────────────────────────────────────────
    int   diskann_r     = 32;   ///< Max graph out-degree.
    int   diskann_l     = 100;  ///< Build-time search list size.
    float diskann_alpha = 1.2f; ///< Pruning parameter (≥ 1.0).

    // ── LSH ───────────────────────────────────────────────────────────────
    int   lsh_l = 10;    ///< Number of hash tables.
    int   lsh_k = 8;     ///< Hash functions per table.
    float lsh_w = 10.0f; ///< Bucket width (E2LSH).

    // ── ScaNN ─────────────────────────────────────────────────────────────
    int   scann_nlist    = 128;  ///< Coarse IVF partitions.
    int   scann_nprobe   = 32;   ///< Partitions to probe.
    int   scann_pq_m     = 8;    ///< PQ sub-spaces.
    int   scann_pq_bits  = 8;    ///< Bits per sub-code (4 or 8).
    int   scann_rerank_k = 100;  ///< Candidates to exact-rerank.
    float scann_aq_eta   = 0.2f; ///< Anisotropic penalty η.

    // ── Presets ───────────────────────────────────────────────────────────

    /// High-recall preset: large M and ef values for HNSW.
    static Params high_recall() noexcept {
        Params p;
        p.hnsw_m               = 32;
        p.hnsw_ef_construction = 400;
        p.hnsw_ef_search       = 200;
        return p;
    }

    /// Low-latency preset: smaller ef values for HNSW.
    static Params low_latency() noexcept {
        Params p;
        p.hnsw_m               = 16;
        p.hnsw_ef_construction = 100;
        p.hnsw_ef_search       = 20;
        return p;
    }

    /// Convert to the C struct for passing to native calls.
    ::PistaDBParams to_c() const noexcept {
        ::PistaDBParams p;
        p.hnsw_M               = hnsw_m;
        p.hnsw_ef_construction = hnsw_ef_construction;
        p.hnsw_ef_search       = hnsw_ef_search;
        p.ivf_nlist            = ivf_nlist;
        p.ivf_nprobe           = ivf_nprobe;
        p.pq_M                 = pq_m;
        p.pq_nbits             = pq_nbits;
        p.diskann_R            = diskann_r;
        p.diskann_L            = diskann_l;
        p.diskann_alpha        = diskann_alpha;
        p.lsh_L                = lsh_l;
        p.lsh_K                = lsh_k;
        p.lsh_w                = lsh_w;
        p.scann_nlist          = scann_nlist;
        p.scann_nprobe         = scann_nprobe;
        p.scann_pq_M           = scann_pq_m;
        p.scann_pq_bits        = scann_pq_bits;
        p.scann_rerank_k       = scann_rerank_k;
        p.scann_aq_eta         = scann_aq_eta;
        return p;
    }
};

// ── Exception ─────────────────────────────────────────────────────────────────

/// Thrown when a PistaDB native operation fails.
class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& msg) : std::runtime_error(msg) {}
    explicit Exception(const char* msg)         : std::runtime_error(msg) {}
};

// ── Database ──────────────────────────────────────────────────────────────────

/**
 * Embedded vector database backed by a single .pst file.
 *
 * - RAII: constructor opens the database; destructor calls pistadb_close().
 * - Thread-safe: all public methods are protected by an internal std::mutex.
 * - Move-only: copying is disabled; use std::shared_ptr for shared ownership.
 * - Does NOT auto-save on destruction — call save() before destroying.
 *
 * @note 'delete' is a C++ keyword. Use remove() to delete a vector by id.
 */
class Database {
public:
    // ── Construction ──────────────────────────────────────────────────────

    /**
     * Open an existing .pst database or create a new one.
     *
     * @param path        File path. Created if it does not exist.
     * @param dim         Vector dimension (must match file if loading).
     * @param metric      Distance metric. Default: L2.
     * @param index_type  Index algorithm. Default: HNSW.
     * @param params      Optional index parameters. nullptr → library defaults.
     * @throws Exception  If pistadb_open returns null.
     */
    explicit Database(
        const std::string& path,
        int                dim,
        Metric             metric     = Metric::L2,
        IndexType          index_type = IndexType::HNSW,
        const Params*      params     = nullptr)
    {
        const ::PistaDBParams* c_params_ptr = nullptr;
        ::PistaDBParams c_params;
        if (params) {
            c_params     = params->to_c();
            c_params_ptr = &c_params;
        }
        db_ = ::pistadb_open(
            path.c_str(),
            dim,
            static_cast<int>(metric),
            static_cast<int>(index_type),
            c_params_ptr);
        if (!db_) {
            throw Exception(
                "pistadb_open failed — check path, permissions, and parameters");
        }
    }

    /// Destructor calls pistadb_close(). Does NOT auto-save.
    ~Database() {
        if (db_) {
            ::pistadb_close(db_);
            db_ = nullptr;
        }
    }

    // Non-copyable.
    Database(const Database&)            = delete;
    Database& operator=(const Database&) = delete;

    // Movable. The mutex is not moved — a fresh one is default-constructed in *this.
    Database(Database&& other) noexcept : db_(other.db_) {
        other.db_ = nullptr;
    }

    Database& operator=(Database&& other) noexcept {
        if (this != &other) {
            if (db_) ::pistadb_close(db_);
            db_       = other.db_;
            other.db_ = nullptr;
        }
        return *this;
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    /// Persist the database to the .pst file.
    /// @throws Exception on I/O failure.
    void save() {
        std::lock_guard<std::mutex> lk(mutex_);
        check(::pistadb_save(db_));
    }

    // ── CRUD ──────────────────────────────────────────────────────────────

    /**
     * Insert a vector.
     *
     * @param id     Unique user-supplied identifier.
     * @param vec    Float vector (length must equal dim()).
     * @param label  Optional human-readable label (max 255 bytes).
     * @throws Exception on duplicate id or other error.
     */
    void insert(uint64_t id,
                const std::vector<float>& vec,
                std::optional<std::string_view> label = std::nullopt)
    {
        std::lock_guard<std::mutex> lk(mutex_);
        insert_impl(id, vec.data(), label);
    }

    /// Raw-pointer overload of insert() — avoids copying vec into a std::vector.
    void insert(uint64_t id,
                const float* vec,
                std::optional<std::string_view> label = std::nullopt)
    {
        std::lock_guard<std::mutex> lk(mutex_);
        insert_impl(id, vec, label);
    }

    /**
     * Logically delete a vector by id.
     * Space is reclaimed on the next save/rebuild.
     * @note Named 'remove' because 'delete' is a C++ keyword.
     * @throws Exception if id is not found.
     */
    void remove(uint64_t id) {
        std::lock_guard<std::mutex> lk(mutex_);
        check(::pistadb_delete(db_, id));
    }

    /**
     * Replace the vector data for an existing id.
     * @throws Exception if id is not found.
     */
    void update(uint64_t id, const std::vector<float>& vec) {
        std::lock_guard<std::mutex> lk(mutex_);
        check(::pistadb_update(db_, id, vec.data()));
    }

    /// Raw-pointer overload of update().
    void update(uint64_t id, const float* vec) {
        std::lock_guard<std::mutex> lk(mutex_);
        check(::pistadb_update(db_, id, vec));
    }

    /**
     * Retrieve the stored vector and label for a given id.
     * @throws Exception if id is not found.
     */
    VectorEntry get(uint64_t id) const {
        std::lock_guard<std::mutex> lk(mutex_);
        const int d = ::pistadb_dim(db_);
        std::vector<float> vec(static_cast<std::size_t>(d));
        char label_buf[256] = {};
        check(::pistadb_get(db_, id, vec.data(), label_buf));
        return { std::move(vec), std::string(label_buf) };
    }

    // ── Search ────────────────────────────────────────────────────────────

    /**
     * K-nearest-neighbour search.
     *
     * @param query  Query vector (length must equal dim()).
     * @param k      Number of results requested.
     * @return       Up to k results ordered by ascending distance.
     * @throws Exception on index error.
     */
    std::vector<SearchResult> search(const std::vector<float>& query, int k) const {
        std::lock_guard<std::mutex> lk(mutex_);
        return search_impl(query.data(), k);
    }

    /// Raw-pointer overload of search() — avoids copying query into a std::vector.
    std::vector<SearchResult> search(const float* query, int k) const {
        std::lock_guard<std::mutex> lk(mutex_);
        return search_impl(query, k);
    }

    // ── Index management ──────────────────────────────────────────────────

    /**
     * Train the index on currently inserted vectors.
     * Required before insert() for IndexType::IVF and IndexType::IVF_PQ.
     * Optional for HNSW/DiskANN — triggers a rebuild pass.
     * @throws Exception on training failure.
     */
    void train() {
        std::lock_guard<std::mutex> lk(mutex_);
        check(::pistadb_train(db_));
    }

    // ── Properties ────────────────────────────────────────────────────────

    /// Number of active (non-deleted) vectors.
    int count() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return ::pistadb_count(db_);
    }

    /// Vector dimension this database was opened with.
    int dim() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return ::pistadb_dim(db_);
    }

    /// Distance metric in use.
    Metric metric() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return static_cast<Metric>(::pistadb_metric(db_));
    }

    /// Index algorithm in use.
    IndexType index_type() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return static_cast<IndexType>(::pistadb_index_type(db_));
    }

    /// Human-readable description of the last native error. Never throws.
    std::string last_error() const {
        std::lock_guard<std::mutex> lk(mutex_);
        const char* msg = ::pistadb_last_error(db_);
        return msg ? msg : "";
    }

    /// Native library version string (e.g. "1.0.0"). Static — no handle needed.
    static std::string version() {
        const char* v = ::pistadb_version();
        return v ? v : "";
    }

private:
    // ── Private helpers (all called while holding mutex_) ─────────────────

    void insert_impl(uint64_t id, const float* vec,
                     std::optional<std::string_view> label)
    {
        // string_view is not guaranteed null-terminated; copy to string first.
        std::string label_storage;
        const char* lbl = nullptr;
        if (label.has_value()) {
            label_storage = std::string(*label);
            lbl           = label_storage.c_str();
        }
        check(::pistadb_insert(db_, id, lbl, vec));
    }

    std::vector<SearchResult> search_impl(const float* query, int k) const {
        std::vector<::PistaDBResult> buf(static_cast<std::size_t>(k));
        const int n = ::pistadb_search(db_, query, k, buf.data());
        if (n < 0) throw_last_error();

        std::vector<SearchResult> results;
        results.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            results.push_back({
                buf[i].id,
                buf[i].distance,
                std::string(buf[i].label), // stops at first '\0'
            });
        }
        return results;
    }

    void check(int rc) const {
        if (rc != 0) throw_last_error();
    }

    [[noreturn]] void throw_last_error() const {
        const char* msg = ::pistadb_last_error(db_);
        throw Exception(msg && *msg ? msg : "unknown PistaDB error");
    }

    // ── Members ───────────────────────────────────────────────────────────

    ::PistaDB*         db_    = nullptr;
    mutable std::mutex mutex_;
};

} // namespace pistadb
