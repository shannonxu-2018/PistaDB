/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - pistadb_cache.h
 * Embedding cache: persistent, LRU-evicting key-value store
 *   text string  →  float32 embedding vector
 *
 * Purpose
 * ───────
 * Embedding APIs (OpenAI, Cohere, local models) are expensive to call.
 * When the same text appears more than once — corpus deduplication, repeated
 * queries, cached document chunks — the cache returns the stored vector
 * instantly instead of re-running the model.
 *
 * Design
 * ──────
 *  • Hash map  (FNV-1a 64-bit, separate chaining)  — O(1) lookup
 *  • LRU list  (doubly-linked)                     — O(1) eviction
 *  • Binary file  (.pcc)                           — survives restarts
 *  • Thread-safe — all public functions protected by an internal mutex
 *
 * File format  (.pcc)
 * ───────────────────
 *  [Header – 64 bytes]
 *    magic[4]        "PCCH"
 *    ver_major[1]    1
 *    ver_minor[1]    0
 *    dim[4]          uint32  vector dimension
 *    max_entries[4]  uint32  capacity limit  (0 = unlimited)
 *    n_entries[8]    uint64  number of stored entries
 *    hits[8]         uint64  cumulative cache hits
 *    misses[8]       uint64  cumulative cache misses
 *    evictions[8]    uint64  cumulative LRU evictions
 *    reserved[18]    zeros
 *
 *  [Entries – repeated n_entries times, stored LRU-first]
 *    text_len[4]     uint32  byte length of text (including '\0')
 *    text[text_len]  char[]  null-terminated string
 *    vec[dim×4]      float[] embedding vector  (native byte order)
 *
 * Usage
 * ─────
 *   PistaDBCache *c = pistadb_cache_open("embed.pcc", 384, 100000);
 *   float vec[384];
 *   if (!pistadb_cache_get(c, text, vec)) {
 *       my_model_encode(text, vec);           // ← expensive call, skipped on hit
 *       pistadb_cache_put(c, text, vec);
 *   }
 *   // use vec …
 *   pistadb_cache_save(c);
 *   pistadb_cache_close(c);
 */
#ifndef PISTADB_CACHE_H
#define PISTADB_CACHE_H

#include "pistadb_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque handle ────────────────────────────────────────────────────────── */

typedef struct PistaDBCache PistaDBCache;

/* ── Stats snapshot ───────────────────────────────────────────────────────── */

typedef struct {
    uint64_t hits;        /**< Lookups that found a cached vector.    */
    uint64_t misses;      /**< Lookups that found nothing.            */
    uint64_t evictions;   /**< LRU evictions triggered by put().      */
    int      count;       /**< Entries currently in the cache.        */
    int      max_entries; /**< Capacity limit (0 = unlimited).        */
} PistaDBCacheStats;

/* ── Lifecycle ────────────────────────────────────────────────────────────── */

/**
 * Open (or create) an embedding cache file.
 *
 * If `path` points to an existing valid .pcc file whose dimension matches
 * `dim`, the cache is loaded from disk.  Otherwise a fresh empty cache is
 * created and the file is written on the first pistadb_cache_save() call.
 *
 * @param path        File path (.pcc).  Created on first save() call.
 * @param dim         Embedding vector dimension.
 * @param max_entries Maximum entries to hold in memory.
 *                    0 = unlimited (grow until memory runs out).
 *                    When full, the least-recently-used entry is evicted.
 * @return            Cache handle, or NULL on allocation failure.
 */
PistaDBCache *pistadb_cache_open(const char *path, int dim, int max_entries);

/**
 * Persist the cache to its .pcc file.
 * Thread-safe.
 * @return PISTADB_OK on success, PISTADB_EIO on I/O failure.
 */
int pistadb_cache_save(PistaDBCache *c);

/**
 * Close and free all resources.  Does NOT auto-save.
 * Call pistadb_cache_save() first if you want to keep the data.
 */
void pistadb_cache_close(PistaDBCache *c);

/* ── Lookup / store ───────────────────────────────────────────────────────── */

/**
 * Look up the cached embedding for a text string.
 * On a hit, the cached vector is written to `out_vec` and the entry is
 * promoted to the MRU position in the LRU list.
 * Thread-safe.
 *
 * @param text    Null-terminated input string (the cache key).
 * @param out_vec Output buffer of at least `dim` floats.  Written on hit.
 * @return        1 on cache hit, 0 on cache miss.
 */
int pistadb_cache_get(PistaDBCache *c, const char *text, float *out_vec);

/**
 * Store an embedding in the cache.
 * Thread-safe.
 *
 * - If `text` is already cached, its vector is updated and the entry is
 *   promoted to MRU.
 * - If the cache is at `max_entries` capacity, the LRU entry is evicted
 *   before the new one is inserted.
 * - `vec` is copied internally; caller may free it immediately.
 *
 * @param text  Null-terminated input string.
 * @param vec   Float array of length `dim`.
 * @return      PISTADB_OK on success, PISTADB_ENOMEM on allocation failure.
 */
int pistadb_cache_put(PistaDBCache *c, const char *text, const float *vec);

/**
 * Check whether a text string is cached without copying the vector or
 * touching the LRU order.
 * Thread-safe.
 * @return 1 if present, 0 if not.
 */
int pistadb_cache_contains(PistaDBCache *c, const char *text);

/**
 * Remove a specific entry from the cache.
 * Thread-safe.
 * @return 1 if the entry existed and was removed, 0 if not found.
 */
int pistadb_cache_evict_key(PistaDBCache *c, const char *text);

/**
 * Remove all entries (keeps file path and settings).
 * Thread-safe.
 */
void pistadb_cache_clear(PistaDBCache *c);

/* ── Metadata ─────────────────────────────────────────────────────────────── */

/** Fill `out` with a snapshot of current cache statistics.  Thread-safe. */
void pistadb_cache_stats(PistaDBCache *c, PistaDBCacheStats *out);

/** Number of entries currently in the cache.  Thread-safe. */
int pistadb_cache_count(PistaDBCache *c);

#ifdef __cplusplus
}
#endif

#endif /* PISTADB_CACHE_H */
