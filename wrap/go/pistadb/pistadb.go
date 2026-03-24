// Package pistadb provides Go bindings for PistaDB, a lightweight embedded
// vector database with support for multiple ANN index algorithms and distance
// metrics.
//
// # Building
//
// Build the C library first:
//
//	cd /path/to/PistaDB
//	mkdir -p build && cd build
//	cmake .. -DCMAKE_BUILD_TYPE=Release
//	cmake --build . --parallel
//
// Then build Go packages normally. The CGO flags in this file point to
// ../../build relative to this source file. Override with environment
// variables if your build directory differs:
//
//	export CGO_LDFLAGS="-L/custom/build/path -lpistadb"
//	go build pistadb.io/go/pistadb
//
// # Basic usage
//
//	db, err := pistadb.Open("mydb.pst", 128, pistadb.MetricL2, pistadb.IndexHNSW, nil)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer db.Close()
//
//	if err := db.Insert(1, "dog", embedding); err != nil {
//	    log.Fatal(err)
//	}
//
//	results, err := db.Search(queryVec, 10)
//	for _, r := range results {
//	    fmt.Printf("id=%d dist=%.4f label=%q\n", r.ID, r.Distance, r.Label)
//	}
package pistadb

/*
#cgo CFLAGS: -I${SRCDIR}/../../src
#cgo linux   LDFLAGS: -L${SRCDIR}/../../build -lpistadb -lm -lpthread
#cgo darwin  LDFLAGS: -L${SRCDIR}/../../build -lpistadb -lpthread
#cgo windows LDFLAGS: -L${SRCDIR}/../../build -lpistadb

#include "pistadb.h"
#include "pistadb_types.h"
#include <stdlib.h>
#include <string.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// ── Error codes ───────────────────────────────────────────────────────────────

// ErrCode is a PistaDB C error code that implements the error interface.
type ErrCode int

const (
	ErrGeneric    ErrCode = -1
	ErrNoMemory   ErrCode = -2
	ErrIO         ErrCode = -3
	ErrNotFound   ErrCode = -4
	ErrExist      ErrCode = -5
	ErrInvalid    ErrCode = -6
	ErrNotTrained ErrCode = -7
	ErrCorrupt    ErrCode = -8
	ErrVersion    ErrCode = -9
)

func (e ErrCode) Error() string {
	switch e {
	case ErrGeneric:
		return "pistadb: generic error"
	case ErrNoMemory:
		return "pistadb: out of memory"
	case ErrIO:
		return "pistadb: I/O error"
	case ErrNotFound:
		return "pistadb: vector not found"
	case ErrExist:
		return "pistadb: vector ID already exists"
	case ErrInvalid:
		return "pistadb: invalid argument"
	case ErrNotTrained:
		return "pistadb: index not trained"
	case ErrCorrupt:
		return "pistadb: corrupted file"
	case ErrVersion:
		return "pistadb: incompatible file version"
	default:
		return fmt.Sprintf("pistadb: error code %d", int(e))
	}
}

func codeToErr(rc C.int) error {
	if rc == C.PISTADB_OK {
		return nil
	}
	return ErrCode(int(rc))
}

// ── Metric ────────────────────────────────────────────────────────────────────

// Metric is a distance / similarity metric used by the database.
type Metric int

const (
	MetricL2      Metric = 0 // Euclidean distance
	MetricCosine  Metric = 1 // Cosine similarity expressed as a distance (1 − cos θ)
	MetricIP      Metric = 2 // Inner product (negated, so smaller ⇒ more similar)
	MetricL1      Metric = 3 // Manhattan / L1 distance
	MetricHamming Metric = 4 // Hamming distance (element-wise mismatch count)
)

func (m Metric) String() string {
	switch m {
	case MetricL2:
		return "L2"
	case MetricCosine:
		return "Cosine"
	case MetricIP:
		return "InnerProduct"
	case MetricL1:
		return "L1"
	case MetricHamming:
		return "Hamming"
	default:
		return fmt.Sprintf("Metric(%d)", int(m))
	}
}

// ── IndexType ─────────────────────────────────────────────────────────────────

// IndexType selects the ANN index algorithm.
type IndexType int

const (
	IndexLinear  IndexType = 0 // Brute-force linear scan — exact, slow for large n
	IndexHNSW    IndexType = 1 // Hierarchical Navigable Small World — fast, high recall
	IndexIVF     IndexType = 2 // Inverted File Index — requires Train before Insert
	IndexIVFPQ   IndexType = 3 // IVF + Product Quantization — compressed, requires Train
	IndexDiskANN IndexType = 4 // Vamana / DiskANN — graph-based, disk-friendly
	IndexLSH     IndexType = 5 // Locality-Sensitive Hashing — randomised, low memory
	IndexSCANN   IndexType = 6 // ScaNN / Anisotropic Vector Quantization
	IndexSQ      IndexType = 7 // Scalar Quantization (uint8) — 4x compression
)

func (t IndexType) String() string {
	switch t {
	case IndexLinear:
		return "Linear"
	case IndexHNSW:
		return "HNSW"
	case IndexIVF:
		return "IVF"
	case IndexIVFPQ:
		return "IVF_PQ"
	case IndexDiskANN:
		return "DiskANN"
	case IndexLSH:
		return "LSH"
	case IndexSCANN:
		return "ScaNN"
	case IndexSQ:
		return "SQ"
	default:
		return fmt.Sprintf("IndexType(%d)", int(t))
	}
}

// ── Params ────────────────────────────────────────────────────────────────────

// Params holds index-specific tuning parameters.
// The zero value is not valid; use DefaultParams to obtain a sensible baseline.
type Params struct {
	// HNSW
	HNSWM              int // Max connections per layer (default 16)
	HNSWEfConstruction int // Build-time beam width (default 200)
	HNSWEfSearch       int // Query-time beam width (default 50)

	// IVF / IVF_PQ
	IVFNList  int // Number of Voronoi centroids (default 128)
	IVFNProbe int // Centroids to search at query time (default 8)
	PQM       int // Product-quantisation subspaces (default 8)
	PQNBits   int // Bits per sub-code: 4 or 8 (default 8)

	// DiskANN / Vamana
	DiskANNR     int     // Max graph out-degree (default 32)
	DiskANNL     int     // Build-time candidate list size (default 100)
	DiskANNAlpha float32 // Pruning aggressiveness (default 1.2)

	// LSH
	LSHL int     // Number of hash tables (default 10)
	LSHK int     // Hash functions per table (default 8)
	LSHW float32 // Bucket width for E2LSH (default 4.0)

	// ScaNN
	SCANNNList   int     // Coarse IVF partitions (default 128)
	SCANNNProbe  int     // Partitions to probe at query time (default 32)
	SCANNPqM     int     // PQ sub-spaces (default 8)
	SCANNPqBits  int     // Bits per sub-code: 4 or 8 (default 8)
	SCANNRerankK int     // Candidates for exact re-ranking (default 100)
	SCANNAqEta   float32 // Anisotropic penalty η (default 0.2)
}

// DefaultParams returns a Params initialised with PistaDB's built-in defaults.
func DefaultParams() Params {
	cp := C.pistadb_default_params()
	return Params{
		HNSWM:              int(cp.hnsw_M),
		HNSWEfConstruction: int(cp.hnsw_ef_construction),
		HNSWEfSearch:       int(cp.hnsw_ef_search),
		IVFNList:           int(cp.ivf_nlist),
		IVFNProbe:          int(cp.ivf_nprobe),
		PQM:                int(cp.pq_M),
		PQNBits:            int(cp.pq_nbits),
		DiskANNR:           int(cp.diskann_R),
		DiskANNL:           int(cp.diskann_L),
		DiskANNAlpha:       float32(cp.diskann_alpha),
		LSHL:               int(cp.lsh_L),
		LSHK:               int(cp.lsh_K),
		LSHW:               float32(cp.lsh_w),
		SCANNNList:         int(cp.scann_nlist),
		SCANNNProbe:        int(cp.scann_nprobe),
		SCANNPqM:           int(cp.scann_pq_M),
		SCANNPqBits:        int(cp.scann_pq_bits),
		SCANNRerankK:       int(cp.scann_rerank_k),
		SCANNAqEta:         float32(cp.scann_aq_eta),
	}
}

func (p *Params) toC() C.PistaDBParams {
	var cp C.PistaDBParams
	cp.hnsw_M = C.int(p.HNSWM)
	cp.hnsw_ef_construction = C.int(p.HNSWEfConstruction)
	cp.hnsw_ef_search = C.int(p.HNSWEfSearch)
	cp.ivf_nlist = C.int(p.IVFNList)
	cp.ivf_nprobe = C.int(p.IVFNProbe)
	cp.pq_M = C.int(p.PQM)
	cp.pq_nbits = C.int(p.PQNBits)
	cp.diskann_R = C.int(p.DiskANNR)
	cp.diskann_L = C.int(p.DiskANNL)
	cp.diskann_alpha = C.float(p.DiskANNAlpha)
	cp.lsh_L = C.int(p.LSHL)
	cp.lsh_K = C.int(p.LSHK)
	cp.lsh_w = C.float(p.LSHW)
	cp.scann_nlist = C.int(p.SCANNNList)
	cp.scann_nprobe = C.int(p.SCANNNProbe)
	cp.scann_pq_M = C.int(p.SCANNPqM)
	cp.scann_pq_bits = C.int(p.SCANNPqBits)
	cp.scann_rerank_k = C.int(p.SCANNRerankK)
	cp.scann_aq_eta = C.float(p.SCANNAqEta)
	return cp
}

// ── SearchResult ──────────────────────────────────────────────────────────────

// SearchResult is a single KNN search result.
type SearchResult struct {
	ID       uint64
	Distance float32
	Label    string
}

// ── Database ──────────────────────────────────────────────────────────────────

// Database is an opaque handle to an open PistaDB database.
//
// Database is not goroutine-safe for concurrent writes. Use the Batch API for
// high-throughput multi-goroutine inserts. Concurrent reads (Search, Get) are
// safe if no writes are in progress.
//
// Always call Close when done. Save first if you want to persist data.
type Database struct {
	ptr *C.PistaDB
	dim int
}

// Open opens an existing PistaDB file or creates a new one.
//
// If path already holds a valid .pst file, it is loaded (dim, metric, and idx
// must match). Otherwise a new empty database is created at that path on the
// first Save call.
//
// Pass nil for params to use PistaDB's built-in defaults.
func Open(path string, dim int, metric Metric, idx IndexType, params *Params) (*Database, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	var cparams *C.PistaDBParams
	if params != nil {
		cp := params.toC()
		cparams = &cp
	}

	ptr := C.pistadb_open(cpath, C.int(dim),
		C.PistaDBMetric(metric), C.PistaDBIndexType(idx), cparams)
	if ptr == nil {
		return nil, errors.New("pistadb: failed to open database (check path and parameters)")
	}

	db := &Database{ptr: ptr, dim: dim}
	runtime.SetFinalizer(db, (*Database).Close)
	return db, nil
}

// Close closes the database and frees all native resources.
// It does NOT auto-save; call Save first to persist data.
// Safe to call multiple times.
func (db *Database) Close() {
	if db.ptr != nil {
		C.pistadb_close(db.ptr)
		db.ptr = nil
		runtime.SetFinalizer(db, nil)
	}
}

// Save persists the database to its .pst file.
func (db *Database) Save() error {
	return codeToErr(C.pistadb_save(db.ptr))
}

// Insert adds a vector to the database.
//
// id must be unique within this database.
// label is an optional human-readable tag; pass "" for no label.
// vec must have exactly Dim() elements.
func (db *Database) Insert(id uint64, label string, vec []float32) error {
	if len(vec) != db.dim {
		return fmt.Errorf("pistadb: vec length %d != dim %d", len(vec), db.dim)
	}
	var clabel *C.char
	if label != "" {
		clabel = C.CString(label)
		defer C.free(unsafe.Pointer(clabel))
	}
	return codeToErr(C.pistadb_insert(db.ptr,
		C.uint64_t(id), clabel,
		(*C.float)(unsafe.Pointer(&vec[0]))))
}

// Delete logically removes a vector by id.
// Disk space is reclaimed on the next Save.
func (db *Database) Delete(id uint64) error {
	return codeToErr(C.pistadb_delete(db.ptr, C.uint64_t(id)))
}

// Update replaces the vector for the given id.
// vec must have exactly Dim() elements.
func (db *Database) Update(id uint64, vec []float32) error {
	if len(vec) != db.dim {
		return fmt.Errorf("pistadb: vec length %d != dim %d", len(vec), db.dim)
	}
	return codeToErr(C.pistadb_update(db.ptr,
		C.uint64_t(id),
		(*C.float)(unsafe.Pointer(&vec[0]))))
}

// Get retrieves the vector and label for the given id.
func (db *Database) Get(id uint64) (vec []float32, label string, err error) {
	vec = make([]float32, db.dim)
	var labelBuf [256]C.char
	rc := C.pistadb_get(db.ptr,
		C.uint64_t(id),
		(*C.float)(unsafe.Pointer(&vec[0])),
		&labelBuf[0])
	if rc != C.PISTADB_OK {
		return nil, "", codeToErr(rc)
	}
	return vec, C.GoString(&labelBuf[0]), nil
}

// Search performs a k-nearest-neighbour query.
// Returns up to k results in ascending distance order.
// query must have exactly Dim() elements.
func (db *Database) Search(query []float32, k int) ([]SearchResult, error) {
	if len(query) != db.dim {
		return nil, fmt.Errorf("pistadb: query length %d != dim %d", len(query), db.dim)
	}
	if k <= 0 {
		return nil, ErrInvalid
	}
	cres := make([]C.PistaDBResult, k)
	n := C.pistadb_search(db.ptr,
		(*C.float)(unsafe.Pointer(&query[0])),
		C.int(k),
		&cres[0])
	if n < 0 {
		return nil, codeToErr(n)
	}
	results := make([]SearchResult, int(n))
	for i := range results {
		results[i] = SearchResult{
			ID:       uint64(cres[i].id),
			Distance: float32(cres[i].distance),
			Label:    C.GoString(&cres[i].label[0]),
		}
	}
	return results, nil
}

// Train trains the index on all currently inserted vectors.
// Required for IVF and IVF_PQ indexes before the first Search.
// Optional for graph-based indexes (triggers a rebuild pass).
func (db *Database) Train() error {
	return codeToErr(C.pistadb_train(db.ptr))
}

// Count returns the number of active (non-deleted) vectors.
func (db *Database) Count() int {
	return int(C.pistadb_count(db.ptr))
}

// Dim returns the vector dimension this database was opened with.
func (db *Database) Dim() int {
	return int(C.pistadb_dim(db.ptr))
}

// Metric returns the distance metric in use.
func (db *Database) Metric() Metric {
	return Metric(C.pistadb_metric(db.ptr))
}

// IndexType returns the ANN index algorithm in use.
func (db *Database) IndexType() IndexType {
	return IndexType(C.pistadb_index_type(db.ptr))
}

// LastError returns the human-readable error message from the most recent
// failing operation on this database handle.
func (db *Database) LastError() string {
	return C.GoString(C.pistadb_last_error(db.ptr))
}

// Version returns the PistaDB C library version string (e.g. "1.0.0").
func Version() string {
	return C.GoString(C.pistadb_version())
}
