//!    ___ _    _        ___  ___
//!   | _ (_)__| |_ __ _|   \| _ )
//!   |  _/ (_-<  _/ _` | |) | _ \
//!   |_| |_/__/\__\__,_|___/|___/
//!
//! # PistaDB Rust Binding
//!
//! Safe, thread-safe Rust wrapper over the native PistaDB C library via FFI.
//!
//! ## Quick start
//!
//! ```no_run
//! use pistadb::{Database, Metric, IndexType, Params};
//!
//! // Open or create a database
//! let db = Database::open("knowledge.pst", 384, Metric::Cosine, IndexType::HNSW, None)?;
//!
//! // Insert a vector with an optional label
//! let embedding = vec![0.1_f32; 384];
//! db.insert(1, &embedding, Some("My first document"))?;
//!
//! // K-nearest-neighbour search
//! let results = db.search(&embedding, 5)?;
//! for r in &results {
//!     println!("id={} dist={:.4} label={:?}", r.id, r.distance, r.label);
//! }
//!
//! db.save()?;
//! // `db` is dropped here — automatically calls pistadb_close()
//! # Ok::<(), pistadb::Error>(())
//! ```
//!
//! ## Thread safety
//!
//! [`Database`] is `Send + Sync`. An internal [`Mutex`](std::sync::Mutex) serialises all
//! native calls, matching the behaviour of the Java `synchronized` wrapper.  For
//! multi-threaded workloads wrap it in [`Arc`](std::sync::Arc):
//!
//! ```no_run
//! use std::sync::Arc;
//! use pistadb::{Database, Metric, IndexType};
//!
//! let db = Arc::new(Database::open("db.pst", 128, Metric::L2, IndexType::HNSW, None)?);
//!
//! let db2 = Arc::clone(&db);
//! std::thread::spawn(move || {
//!     db2.insert(42, &[0.0_f32; 128], None).unwrap();
//! });
//! # Ok::<(), pistadb::Error>(())
//! ```

mod ffi;
mod types;

pub use types::{Error, IndexType, Metric, Params, Result, SearchResult, VectorEntry};

use std::ffi::CString;
use std::sync::Mutex;

// ── Handle wrapper ────────────────────────────────────────────────────────────

/// Newtype so we can implement `Send` for a raw pointer.
struct RawHandle(*mut ffi::PistaDBOpaque);

// Safety: the C library serialises all access internally (and we add a Mutex on
// top). Moving the pointer between threads is safe.
unsafe impl Send for RawHandle {}

// ── Database ──────────────────────────────────────────────────────────────────

/// Embedded vector database backed by a single `.pst` file.
///
/// - **Thread-safe** — protected internally by a [`Mutex`](std::sync::Mutex).
/// - **`Drop`** — automatically calls `pistadb_close` when dropped.
/// - **Does not auto-save** — call [`save`](Database::save) before dropping.
pub struct Database {
    inner: Mutex<RawHandle>,
}

// Mutex<RawHandle> is Send (RawHandle: Send) + Sync (Mutex guarantees exclusion).
unsafe impl Sync for Database {}

impl Database {
    // ── Factory ───────────────────────────────────────────────────────────

    /// Open an existing `.pst` database or create a new one.
    ///
    /// # Arguments
    ///
    /// - `path`       — File path. Created if it does not exist.
    /// - `dim`        — Vector dimension (must match file if loading).
    /// - `metric`     — Distance metric.
    /// - `index_type` — Index algorithm.
    /// - `params`     — Optional index parameters. `None` uses library defaults.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OpenFailed`] if the native call returns a null pointer.
    pub fn open(
        path: impl AsRef<std::path::Path>,
        dim: i32,
        metric: Metric,
        index_type: IndexType,
        params: Option<&Params>,
    ) -> Result<Self> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| Error::Native("path is not valid UTF-8".into()))?;
        let c_path = CString::new(path_str)?;

        let handle = unsafe {
            match params {
                Some(p) => {
                    let ffi_p = p.to_ffi();
                    ffi::pistadb_open(
                        c_path.as_ptr(),
                        dim,
                        metric as i32,
                        index_type as i32,
                        &ffi_p,
                    )
                }
                None => ffi::pistadb_open(
                    c_path.as_ptr(),
                    dim,
                    metric as i32,
                    index_type as i32,
                    std::ptr::null(),
                ),
            }
        };

        if handle.is_null() {
            return Err(Error::OpenFailed(
                "pistadb_open returned null — check path, permissions, and parameters".into(),
            ));
        }

        Ok(Self {
            inner: Mutex::new(RawHandle(handle)),
        })
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// Execute a closure with exclusive access to the native handle.
    fn with_handle<F, T>(&self, f: F) -> T
    where
        F: FnOnce(*mut ffi::PistaDBOpaque) -> T,
    {
        let guard = self.inner.lock().expect("PistaDB mutex poisoned");
        f(guard.0)
    }

    /// Retrieve `pistadb_last_error` as a Rust String.
    fn last_error_string(&self) -> String {
        self.with_handle(|h| unsafe { types::ptr_to_string(ffi::pistadb_last_error(h)) })
    }

    /// Return `Ok(())` if `rc == 0`, otherwise fetch and wrap the last error.
    fn check(&self, rc: i32) -> Result<()> {
        if rc == 0 {
            Ok(())
        } else {
            Err(Error::Native(self.last_error_string()))
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    /// Persist the database to its `.pst` file.
    ///
    /// **Must be called before dropping** to avoid losing in-memory state.
    pub fn save(&self) -> Result<()> {
        let rc = self.with_handle(|h| unsafe { ffi::pistadb_save(h) });
        self.check(rc)
    }

    // ── CRUD ──────────────────────────────────────────────────────────────

    /// Insert a vector with an optional label.
    ///
    /// `id` must be unique within this database.
    /// `vec` must have length equal to [`dim`](Database::dim).
    pub fn insert(&self, id: u64, vec: &[f32], label: Option<&str>) -> Result<()> {
        let c_label = label.map(CString::new).transpose()?;
        let label_ptr = c_label.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
        let rc = self.with_handle(|h| unsafe {
            ffi::pistadb_insert(h, id, label_ptr, vec.as_ptr())
        });
        self.check(rc)
    }

    /// Logically delete a vector by id.
    ///
    /// Space is reclaimed on the next save/rebuild.
    pub fn delete(&self, id: u64) -> Result<()> {
        let rc = self.with_handle(|h| unsafe { ffi::pistadb_delete(h, id) });
        self.check(rc)
    }

    /// Replace the vector data for an existing id.
    pub fn update(&self, id: u64, vec: &[f32]) -> Result<()> {
        let rc =
            self.with_handle(|h| unsafe { ffi::pistadb_update(h, id, vec.as_ptr()) });
        self.check(rc)
    }

    /// Retrieve the stored vector and label for a given id.
    pub fn get(&self, id: u64) -> Result<VectorEntry> {
        // Fetch dim first (releases lock), then allocate, then call get.
        let dim = self.dim() as usize;
        let mut vec = vec![0.0_f32; dim];
        let mut label_buf = [0_i8; 256];
        let rc = self.with_handle(|h| unsafe {
            ffi::pistadb_get(h, id, vec.as_mut_ptr(), label_buf.as_mut_ptr())
        });
        self.check(rc)?;
        Ok(VectorEntry {
            vector: vec,
            label: types::chars_to_string(&label_buf),
        })
    }

    // ── Search ────────────────────────────────────────────────────────────

    /// K-nearest-neighbour search.
    ///
    /// Returns up to `k` results ordered by ascending distance.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // Initialise result buffer (pistadb_search writes into it).
        let mut buf: Vec<ffi::PistaDBResult> = (0..k)
            .map(|_| ffi::PistaDBResult {
                id: 0,
                distance: 0.0,
                label: [0; 256],
            })
            .collect();

        let n = self.with_handle(|h| unsafe {
            ffi::pistadb_search(h, query.as_ptr(), k as i32, buf.as_mut_ptr())
        });

        if n < 0 {
            return Err(Error::Native(self.last_error_string()));
        }

        let results = buf[..n as usize]
            .iter()
            .map(|r| SearchResult {
                id: r.id,
                distance: r.distance,
                label: types::chars_to_string(&r.label),
            })
            .collect();

        Ok(results)
    }

    // ── Index management ──────────────────────────────────────────────────

    /// Train the index on currently inserted vectors.
    ///
    /// Required before inserting for [`IndexType::IVF`] and [`IndexType::IVF_PQ`].
    /// Optional for HNSW/DiskANN (triggers a rebuild pass).
    pub fn train(&self) -> Result<()> {
        let rc = self.with_handle(|h| unsafe { ffi::pistadb_train(h) });
        self.check(rc)
    }

    // ── Properties ────────────────────────────────────────────────────────

    /// Number of active (non-deleted) vectors.
    pub fn count(&self) -> i32 {
        self.with_handle(|h| unsafe { ffi::pistadb_count(h) })
    }

    /// Vector dimension this database was opened with.
    pub fn dim(&self) -> i32 {
        self.with_handle(|h| unsafe { ffi::pistadb_dim(h) })
    }

    /// Distance metric in use.
    pub fn metric(&self) -> Metric {
        let v = self.with_handle(|h| unsafe { ffi::pistadb_metric(h) });
        Metric::try_from(v).unwrap_or(Metric::L2)
    }

    /// Index algorithm in use.
    pub fn index_type(&self) -> IndexType {
        let v = self.with_handle(|h| unsafe { ffi::pistadb_index_type(h) });
        IndexType::try_from(v).unwrap_or(IndexType::HNSW)
    }

    /// Human-readable description of the last native error.
    pub fn last_error(&self) -> String {
        self.last_error_string()
    }

    /// Native library version string (e.g. `"1.0.0"`).
    pub fn version() -> String {
        unsafe { types::ptr_to_string(ffi::pistadb_version()) }
    }
}

// ── Drop ──────────────────────────────────────────────────────────────────────

impl Drop for Database {
    fn drop(&mut self) {
        // get_mut() does not lock — safe with &mut self (exclusive access).
        let inner = self
            .inner
            .get_mut()
            .unwrap_or_else(|e| e.into_inner());
        if !inner.0.is_null() {
            unsafe { ffi::pistadb_close(inner.0) }
        }
    }
}
