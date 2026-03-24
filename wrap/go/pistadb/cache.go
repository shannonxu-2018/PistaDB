package pistadb

/*
#include "pistadb_cache.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// ── CacheStats ────────────────────────────────────────────────────────────────

// CacheStats is a point-in-time snapshot of embedding cache statistics.
type CacheStats struct {
	Hits       uint64 // Lookups that returned a cached vector
	Misses     uint64 // Lookups that found nothing
	Evictions  uint64 // LRU evictions triggered by Put
	Count      int    // Entries currently in the cache
	MaxEntries int    // Configured capacity limit (0 = unlimited)
}

// ── Cache ─────────────────────────────────────────────────────────────────────

// Cache is an LRU-evicting, file-backed embedding cache that maps text strings
// to float32 embedding vectors.
//
// Typical use-case: avoid re-calling an expensive embedding API for text that
// has already been encoded.
//
//	vec, ok := cache.Get(text)
//	if !ok {
//	    vec = myModel.Encode(text)   // expensive
//	    cache.Put(text, vec)
//	}
//
// All methods are thread-safe.
// Call Save before Close to persist data across restarts.
type Cache struct {
	ptr *C.PistaDBCache
	dim int
}

// OpenCache opens (or creates) an embedding cache file.
//
// If path points to a valid .pcc file with matching dim, the cache is loaded
// from disk. Otherwise an empty cache is created; the file is written on the
// first Save call.
//
// maxEntries caps the number of in-memory entries (0 = unlimited). When full,
// the least-recently-used entry is evicted to make room.
func OpenCache(path string, dim, maxEntries int) (*Cache, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	ptr := C.pistadb_cache_open(cpath, C.int(dim), C.int(maxEntries))
	if ptr == nil {
		return nil, ErrNoMemory
	}
	c := &Cache{ptr: ptr, dim: dim}
	runtime.SetFinalizer(c, (*Cache).Close)
	return c, nil
}

// Save persists the cache to its .pcc file.
func (c *Cache) Save() error {
	return codeToErr(C.pistadb_cache_save(c.ptr))
}

// Close frees all resources.  Does NOT auto-save.
// Safe to call multiple times.
func (c *Cache) Close() {
	if c.ptr != nil {
		C.pistadb_cache_close(c.ptr)
		c.ptr = nil
		runtime.SetFinalizer(c, nil)
	}
}

// Get looks up the cached embedding for text.
//
// On a hit the vector is returned and the entry is promoted to the MRU
// position. Returns (nil, false) on a miss.
func (c *Cache) Get(text string) ([]float32, bool) {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))

	vec := make([]float32, c.dim)
	hit := C.pistadb_cache_get(c.ptr, ctext,
		(*C.float)(unsafe.Pointer(&vec[0])))
	if hit == 0 {
		return nil, false
	}
	return vec, true
}

// Put stores an embedding in the cache.
//
// If text is already cached, its vector is updated and the entry is promoted
// to MRU. If the cache is at capacity, the LRU entry is evicted first.
// vec is copied internally; the caller may reuse it immediately.
// vec must have exactly dim elements.
func (c *Cache) Put(text string, vec []float32) error {
	if len(vec) != c.dim {
		return ErrInvalid
	}
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))

	return codeToErr(C.pistadb_cache_put(c.ptr, ctext,
		(*C.float)(unsafe.Pointer(&vec[0]))))
}

// Contains reports whether text is cached without touching the LRU order or
// copying any vector data.
func (c *Cache) Contains(text string) bool {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	return C.pistadb_cache_contains(c.ptr, ctext) != 0
}

// Evict removes a specific entry from the cache.
// Returns true if the entry existed and was removed, false if not found.
func (c *Cache) Evict(text string) bool {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	return C.pistadb_cache_evict_key(c.ptr, ctext) != 0
}

// Clear removes all entries, keeping the file path and capacity settings.
func (c *Cache) Clear() {
	C.pistadb_cache_clear(c.ptr)
}

// Stats returns a snapshot of current cache statistics.
func (c *Cache) Stats() CacheStats {
	var cs C.PistaDBCacheStats
	C.pistadb_cache_stats(c.ptr, &cs)
	return CacheStats{
		Hits:       uint64(cs.hits),
		Misses:     uint64(cs.misses),
		Evictions:  uint64(cs.evictions),
		Count:      int(cs.count),
		MaxEntries: int(cs.max_entries),
	}
}

// Count returns the number of entries currently in the cache.
func (c *Cache) Count() int {
	return int(C.pistadb_cache_count(c.ptr))
}
