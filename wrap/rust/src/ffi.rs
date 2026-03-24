//! Raw FFI bindings — unsafe, mirrors `pistadb.h` exactly.
//!
//! Do **not** use these directly; prefer the safe [`crate::Database`] API.

use std::os::raw::{c_char, c_float, c_int};

/// Opaque database handle. Never construct — only pass through raw pointers.
#[repr(C)]
pub struct PistaDBOpaque {
    _private: [u8; 0],
}

/// Mirrors C struct `PistaDBParams` (Sequential layout, field order must match
/// `pistadb_types.h`).
#[repr(C)]
pub struct PistaDBParams {
    // ── HNSW ──────────────────────────────────────────────────────────────
    pub hnsw_M: c_int,
    pub hnsw_ef_construction: c_int,
    pub hnsw_ef_search: c_int,
    // ── IVF / IVF_PQ ──────────────────────────────────────────────────────
    pub ivf_nlist: c_int,
    pub ivf_nprobe: c_int,
    pub pq_M: c_int,
    pub pq_nbits: c_int,
    // ── DiskANN / Vamana ──────────────────────────────────────────────────
    pub diskann_R: c_int,
    pub diskann_L: c_int,
    pub diskann_alpha: c_float,
    // ── LSH ───────────────────────────────────────────────────────────────
    pub lsh_L: c_int,
    pub lsh_K: c_int,
    pub lsh_w: c_float,
    // ── ScaNN ─────────────────────────────────────────────────────────────
    pub scann_nlist: c_int,
    pub scann_nprobe: c_int,
    pub scann_pq_M: c_int,
    pub scann_pq_bits: c_int,
    pub scann_rerank_k: c_int,
    pub scann_aq_eta: c_float,
}

/// Mirrors C struct `PistaDBResult`.
/// `label` is a null-terminated UTF-8 string in a fixed 256-byte buffer.
#[repr(C)]
pub struct PistaDBResult {
    pub id: u64,
    pub distance: c_float,
    pub label: [c_char; 256],
}

extern "C" {
    // ── Lifecycle ─────────────────────────────────────────────────────────

    pub fn pistadb_open(
        path: *const c_char,
        dim: c_int,
        metric: c_int,
        index: c_int,
        params: *const PistaDBParams, // NULL → library defaults
    ) -> *mut PistaDBOpaque;

    pub fn pistadb_close(db: *mut PistaDBOpaque);

    pub fn pistadb_save(db: *mut PistaDBOpaque) -> c_int;

    // ── CRUD ──────────────────────────────────────────────────────────────

    pub fn pistadb_insert(
        db: *mut PistaDBOpaque,
        id: u64,
        label: *const c_char, // NULL → no label
        vec: *const c_float,
    ) -> c_int;

    pub fn pistadb_delete(db: *mut PistaDBOpaque, id: u64) -> c_int;

    pub fn pistadb_update(
        db: *mut PistaDBOpaque,
        id: u64,
        vec: *const c_float,
    ) -> c_int;

    pub fn pistadb_get(
        db: *mut PistaDBOpaque,
        id: u64,
        out_vec: *mut c_float,
        out_label: *mut c_char, // caller-supplied 256-byte buffer
    ) -> c_int;

    // ── Search ────────────────────────────────────────────────────────────

    pub fn pistadb_search(
        db: *mut PistaDBOpaque,
        query: *const c_float,
        k: c_int,
        results: *mut PistaDBResult,
    ) -> c_int; // actual count (≤ k) or negative on error

    // ── Index management ──────────────────────────────────────────────────

    pub fn pistadb_train(db: *mut PistaDBOpaque) -> c_int;

    // ── Metadata ──────────────────────────────────────────────────────────

    pub fn pistadb_count(db: *mut PistaDBOpaque) -> c_int;
    pub fn pistadb_dim(db: *mut PistaDBOpaque) -> c_int;
    pub fn pistadb_metric(db: *mut PistaDBOpaque) -> c_int;
    pub fn pistadb_index_type(db: *mut PistaDBOpaque) -> c_int;

    pub fn pistadb_last_error(db: *mut PistaDBOpaque) -> *const c_char;

    // ── Version ───────────────────────────────────────────────────────────

    pub fn pistadb_version() -> *const c_char;
}
