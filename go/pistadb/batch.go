package pistadb

/*
#include "pistadb_batch.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// ── Batch ─────────────────────────────────────────────────────────────────────

// Batch is a multi-threaded batch insert context that owns an internal thread
// pool and a bounded work queue.
//
// Multiple goroutines may call Push concurrently. All other methods (Flush,
// ErrorCount, Destroy) must be called from a single owner goroutine.
//
// Do not mix direct db.Insert calls on the same Database handle while a Batch
// is active on it — index writes are serialised internally by the Batch.
//
// Always call Destroy when done.
type Batch struct {
	ptr *C.PistaDBBatch
}

// NewBatch creates a batch insert context for db.
//
// nThreads is the number of worker goroutines (0 = auto-detect, up to 32).
// queueCap is the maximum number of items in-flight at once (0 = default 4096).
// Push blocks when the queue is full, providing automatic back-pressure.
func NewBatch(db *Database, nThreads, queueCap int) (*Batch, error) {
	ptr := C.pistadb_batch_create(db.ptr, C.int(nThreads), C.int(queueCap))
	if ptr == nil {
		return nil, ErrNoMemory
	}
	b := &Batch{ptr: ptr}
	runtime.SetFinalizer(b, (*Batch).Destroy)
	return b, nil
}

// Push enqueues one vector for insertion.
//
// Thread-safe — any number of goroutines may call Push concurrently.
// Blocks (with back-pressure) if the internal queue is at capacity.
// vec is copied internally; the caller may reuse it immediately after return.
// Pass label = "" for no label.
func (b *Batch) Push(id uint64, label string, vec []float32) error {
	var clabel *C.char
	if label != "" {
		clabel = C.CString(label)
		defer C.free(unsafe.Pointer(clabel))
	}
	return codeToErr(C.pistadb_batch_push(b.ptr,
		C.uint64_t(id), clabel,
		(*C.float)(unsafe.Pointer(&vec[0]))))
}

// Flush blocks until all previously pushed items have been inserted.
// Resets the per-flush error counter.
// Returns the number of failed inserts since the last Flush call (0 on success).
func (b *Batch) Flush() int {
	return int(C.pistadb_batch_flush(b.ptr))
}

// ErrorCount returns the total number of insert errors accumulated since
// NewBatch. Unlike the value returned by Flush, this counter is never reset.
func (b *Batch) ErrorCount() int {
	return int(C.pistadb_batch_error_count(b.ptr))
}

// Destroy flushes remaining items, shuts down worker threads, and frees all
// memory. Safe to call even if Flush was not called first.
// Safe to call multiple times.
func (b *Batch) Destroy() {
	if b.ptr != nil {
		C.pistadb_batch_destroy(b.ptr)
		b.ptr = nil
		runtime.SetFinalizer(b, nil)
	}
}

// ── Convenience: offline bulk insert ─────────────────────────────────────────

// BatchInsert inserts n vectors into db using a temporary thread pool and
// blocks until all inserts complete. It is equivalent to creating a Batch,
// pushing all items, flushing, and destroying it.
//
// ids, labels, and vecs must each have exactly n elements. Individual labels
// may be "". Pass nil for labels to skip labels for all entries.
// nThreads = 0 for auto-detection.
//
// Returns the number of failed inserts (0 = full success) and any argument
// validation error.
func BatchInsert(db *Database, ids []uint64, labels []string, vecs [][]float32, nThreads int) (int, error) {
	n := len(ids)
	if len(vecs) != n {
		return 0, ErrInvalid
	}
	if n == 0 {
		return 0, nil
	}

	// Flatten vecs into a C-compatible row-major array.
	dim := db.dim
	flat := make([]float32, n*dim)
	for i, v := range vecs {
		if len(v) != dim {
			return 0, ErrInvalid
		}
		copy(flat[i*dim:], v)
	}

	// Build C id array.
	cids := make([]C.uint64_t, n)
	for i, id := range ids {
		cids[i] = C.uint64_t(id)
	}

	// Build C label pointer array; individual elements may be nil (no label).
	var clabelPtr **C.char
	if len(labels) == n {
		clabels := make([]*C.char, n)
		for i, l := range labels {
			if l != "" {
				clabels[i] = C.CString(l)
			}
		}
		defer func() {
			for _, cl := range clabels {
				if cl != nil {
					C.free(unsafe.Pointer(cl))
				}
			}
		}()
		clabelPtr = &clabels[0]
	}
	// clabelPtr == nil means "no labels" — the C API accepts NULL here.

	failures := C.pistadb_batch_insert(
		db.ptr,
		&cids[0],
		clabelPtr,
		(*C.float)(unsafe.Pointer(&flat[0])),
		C.int(n),
		C.int(nThreads),
	)
	return int(failures), nil
}
