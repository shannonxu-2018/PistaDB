//! Public types: enums, result types, parameters, and error.

use std::fmt;
use std::os::raw::c_char;

use crate::ffi;

// ── Distance metric ───────────────────────────────────────────────────────────

/// Vector distance metric used by the database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    /// Euclidean distance (L2). Default; good for most embeddings.
    L2      = 0,
    /// Cosine similarity expressed as distance. Ideal for text embeddings.
    Cosine  = 1,
    /// Inner product (stored as negative). For dot-product similarity.
    IP      = 2,
    /// Manhattan distance (L1).
    L1      = 3,
    /// Hamming distance. Suitable for binary or integer vectors.
    Hamming = 4,
}

impl TryFrom<i32> for Metric {
    type Error = Error;
    fn try_from(v: i32) -> Result<Self> {
        match v {
            0 => Ok(Metric::L2),
            1 => Ok(Metric::Cosine),
            2 => Ok(Metric::IP),
            3 => Ok(Metric::L1),
            4 => Ok(Metric::Hamming),
            _ => Err(Error::Native(format!("unknown metric value: {v}"))),
        }
    }
}

// ── Index algorithm ───────────────────────────────────────────────────────────

/// ANN index algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexType {
    /// Brute-force exact scan. Perfect accuracy, O(n) query.
    Linear  = 0,
    /// HNSW — fast approximate search. Recommended for RAG workloads.
    HNSW    = 1,
    /// Inverted File Index. Requires [`Database::train`](crate::Database::train) before insert.
    IVF     = 2,
    /// IVF + Product Quantization. Low memory, lossy. Requires train.
    IVF_PQ  = 3,
    /// DiskANN / Vamana. Scales to billion-vector datasets.
    DiskANN = 4,
    /// Locality-Sensitive Hashing. Ultra-low memory footprint.
    LSH     = 5,
    /// ScaNN — Anisotropic Vector Quantization. Two-phase reranking.
    ScaNN   = 6,
}

impl TryFrom<i32> for IndexType {
    type Error = Error;
    fn try_from(v: i32) -> Result<Self> {
        match v {
            0 => Ok(IndexType::Linear),
            1 => Ok(IndexType::HNSW),
            2 => Ok(IndexType::IVF),
            3 => Ok(IndexType::IVF_PQ),
            4 => Ok(IndexType::DiskANN),
            5 => Ok(IndexType::LSH),
            6 => Ok(IndexType::ScaNN),
            _ => Err(Error::Native(format!("unknown index type value: {v}"))),
        }
    }
}

// ── Result types ──────────────────────────────────────────────────────────────

/// A single KNN search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// User-supplied vector id.
    pub id: u64,
    /// Distance from the query (lower is more similar for L2/L1/Hamming/Cosine).
    pub distance: f32,
    /// Optional human-readable label associated with the vector.
    pub label: String,
}

impl fmt::Display for SearchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SearchResult {{ id: {}, distance: {:.4}, label: {:?} }}",
            self.id, self.distance, self.label
        )
    }
}

/// A stored vector entry returned by [`Database::get`](crate::Database::get).
#[derive(Debug, Clone)]
pub struct VectorEntry {
    /// The raw float vector.
    pub vector: Vec<f32>,
    /// Optional label stored alongside the vector.
    pub label: String,
}

// ── Parameters ────────────────────────────────────────────────────────────────

/// Index parameters passed when opening a database.
///
/// Unused parameters for the selected [`IndexType`] are ignored by the library.
/// Use [`Default::default()`] to get the library defaults, then override fields.
#[derive(Debug, Clone)]
pub struct Params {
    // ── HNSW ──────────────────────────────────────────────────────────────
    /// Max bi-directional connections per layer. Higher = better recall, more memory. Default 16.
    pub hnsw_m: i32,
    /// Build-time search width. Higher = better graph quality, slower build. Default 200.
    pub hnsw_ef_construction: i32,
    /// Query-time search width. Higher = better recall, slower query. Default 50.
    pub hnsw_ef_search: i32,

    // ── IVF / IVF_PQ ──────────────────────────────────────────────────────
    /// Number of IVF centroids (clusters). Default 128.
    pub ivf_nlist: i32,
    /// Number of centroids to search at query time. Default 8.
    pub ivf_nprobe: i32,
    /// Number of PQ sub-spaces. Default 8.
    pub pq_m: i32,
    /// Bits per PQ sub-code (4 or 8). Default 8.
    pub pq_nbits: i32,

    // ── DiskANN / Vamana ──────────────────────────────────────────────────
    /// Max graph out-degree. Default 32.
    pub diskann_r: i32,
    /// Build-time search list size. Default 100.
    pub diskann_l: i32,
    /// Pruning parameter (≥ 1.0). Default 1.2.
    pub diskann_alpha: f32,

    // ── LSH ───────────────────────────────────────────────────────────────
    /// Number of hash tables. Default 10.
    pub lsh_l: i32,
    /// Hash functions per table. Default 8.
    pub lsh_k: i32,
    /// Bucket width (E2LSH). Default 10.0.
    pub lsh_w: f32,

    // ── ScaNN ─────────────────────────────────────────────────────────────
    /// Coarse IVF partitions. Default 128.
    pub scann_nlist: i32,
    /// Partitions to probe during search. Default 32.
    pub scann_nprobe: i32,
    /// PQ sub-spaces. Default 8.
    pub scann_pq_m: i32,
    /// Bits per PQ sub-code (4 or 8). Default 8.
    pub scann_pq_bits: i32,
    /// Candidates to rerank with exact distances. Default 100.
    pub scann_rerank_k: i32,
    /// Anisotropic penalty η. Default 0.2.
    pub scann_aq_eta: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            ivf_nlist: 128,
            ivf_nprobe: 8,
            pq_m: 8,
            pq_nbits: 8,
            diskann_r: 32,
            diskann_l: 100,
            diskann_alpha: 1.2,
            lsh_l: 10,
            lsh_k: 8,
            lsh_w: 10.0,
            scann_nlist: 128,
            scann_nprobe: 32,
            scann_pq_m: 8,
            scann_pq_bits: 8,
            scann_rerank_k: 100,
            scann_aq_eta: 0.2,
        }
    }
}

impl Params {
    /// High-recall preset: large M and ef values for HNSW.
    pub fn high_recall() -> Self {
        Self {
            hnsw_m: 32,
            hnsw_ef_construction: 400,
            hnsw_ef_search: 200,
            ..Default::default()
        }
    }

    /// Low-latency preset: smaller ef values for HNSW.
    pub fn low_latency() -> Self {
        Self {
            hnsw_m: 16,
            hnsw_ef_construction: 100,
            hnsw_ef_search: 20,
            ..Default::default()
        }
    }

    /// Convert to the FFI struct for passing to native calls.
    pub(crate) fn to_ffi(&self) -> ffi::PistaDBParams {
        ffi::PistaDBParams {
            hnsw_M:               self.hnsw_m,
            hnsw_ef_construction: self.hnsw_ef_construction,
            hnsw_ef_search:       self.hnsw_ef_search,
            ivf_nlist:            self.ivf_nlist,
            ivf_nprobe:           self.ivf_nprobe,
            pq_M:                 self.pq_m,
            pq_nbits:             self.pq_nbits,
            diskann_R:            self.diskann_r,
            diskann_L:            self.diskann_l,
            diskann_alpha:        self.diskann_alpha,
            lsh_L:                self.lsh_l,
            lsh_K:                self.lsh_k,
            lsh_w:                self.lsh_w,
            scann_nlist:          self.scann_nlist,
            scann_nprobe:         self.scann_nprobe,
            scann_pq_M:           self.scann_pq_m,
            scann_pq_bits:        self.scann_pq_bits,
            scann_rerank_k:       self.scann_rerank_k,
            scann_aq_eta:         self.scann_aq_eta,
        }
    }
}

// ── Error ─────────────────────────────────────────────────────────────────────

/// Error type for all PistaDB operations.
#[derive(Debug)]
pub enum Error {
    /// A native C library error; message from `pistadb_last_error()`.
    Native(String),
    /// `pistadb_open` returned a null pointer.
    OpenFailed(String),
    /// A string argument contained an interior null byte.
    NulError(std::ffi::NulError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Native(msg)     => write!(f, "PistaDB error: {msg}"),
            Error::OpenFailed(msg) => write!(f, "Failed to open database: {msg}"),
            Error::NulError(e)     => write!(f, "String contains interior null byte: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::NulError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::ffi::NulError> for Error {
    fn from(e: std::ffi::NulError) -> Self {
        Error::NulError(e)
    }
}

/// Convenience `Result` type alias.
pub type Result<T> = std::result::Result<T, Error>;

// ── String helpers ────────────────────────────────────────────────────────────

/// Convert a null-terminated C `char` array to a Rust `String`.
pub(crate) fn chars_to_string(chars: &[c_char; 256]) -> String {
    let end = chars.iter().position(|&c| c == 0).unwrap_or(256);
    // Safety: reinterpreting i8 bytes as u8 is always valid.
    let bytes = unsafe { std::slice::from_raw_parts(chars.as_ptr() as *const u8, end) };
    String::from_utf8_lossy(bytes).into_owned()
}

/// Convert a raw C string pointer to a Rust `String`.
///
/// # Safety
/// `ptr` must be null or point to a valid null-terminated C string.
pub(crate) unsafe fn ptr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    std::ffi::CStr::from_ptr(ptr)
        .to_string_lossy()
        .into_owned()
}
