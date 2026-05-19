/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - pager.h
 * SQLite-style read pager with a bounded LRU page cache.
 *
 * Motivation
 * ──────────
 * The default open path malloc()s and fread()s the entire vector section,
 * then each index copies it again into its own heap — peak ≈ 2× file size,
 * steady-state ≈ 1× file size, and a single >2 GB malloc fails outright on
 * Windows.  This pager instead keeps the file open and serves fixed-size
 * pages on demand through a fixed-capacity cache: resident memory is capped
 * at `cache_bytes` regardless of how large the database file is, exactly
 * like SQLite's page cache.
 *
 * Scope
 * ─────
 *  - Read-only.  The .pst vector region is immutable once written; saves go
 *    through the existing atomic temp-file rewrite, never through the pager.
 *  - Pure C99 + stdio (no mmap): portable, no platform branching for the
 *    mapping itself, and a *hard* memory ceiling (mmap would let the OS keep
 *    an unbounded resident set under no memory pressure).
 *  - Thread-safe: a single mutex guards the cache so concurrent searches on
 *    one handle cannot corrupt it (mirrors pistadb_cache.c).
 *
 * Single-page guarantee
 * ─────────────────────
 * Callers pin a byte range and receive a stable, contiguous pointer into a
 * cache frame.  The range must lie within ONE page.  The paged .pst writer
 * guarantees this by padding each vector record so it never straddles a page
 * boundary, so the pager never has to stitch a record across two frames.
 */
#ifndef PISTADB_PAGER_H
#define PISTADB_PAGER_H

#include <stdint.h>
#include <stddef.h>

typedef struct PistaPager PistaPager;

typedef struct {
    uint64_t hits;        /* pin() served from cache                       */
    uint64_t misses;      /* pin() that had to read the file               */
    uint64_t evictions;   /* LRU frames reclaimed to make room             */
    uint64_t fetches;     /* file reads issued (== misses, kept for clarity)*/
    int      n_frames;    /* cache capacity, in pages                      */
    int      n_resident;  /* frames currently holding a valid page         */
    uint32_t page_size;
} PistaPagerStats;

/**
 * Open a read-only pager over the byte range
 *   [region_off, region_off + region_len)
 * of the file at `path`.
 *
 * @param page_size    Power of two, >= 4096.  0 selects the 64 KiB default.
 * @param cache_bytes  Hard upper bound on resident page memory.  Rounded
 *                      down to a whole number of pages; a small floor is
 *                      always honoured so the working set of one search
 *                      fits.  0 selects a 64 MiB default.
 * @return Handle, or NULL on failure (file open, allocation, bad args).
 */
PistaPager *pista_pager_open(const char *path,
                             uint64_t region_off, uint64_t region_len,
                             uint32_t page_size, size_t cache_bytes);

void pista_pager_close(PistaPager *pg);

/**
 * Pin the page holding region-relative range [off, off + span); fetch it
 * from disk if not resident, evicting the LRU unpinned frame if the cache
 * is full.
 *
 * @param off   Byte offset from the start of the region.
 * @param span  Byte length; [off, off+span) MUST fit inside one page.
 * @param tok   Out: opaque token to pass to pista_pager_unpin().
 * @return Pointer to the first byte (valid until the matching unpin), or
 *         NULL on error (out of range, span straddles a page, or every
 *         frame is pinned — i.e. the working set exceeds the cache).
 */
const void *pista_pager_pin(PistaPager *pg, uint64_t off, uint32_t span,
                            uint64_t *tok);

/** Release a pin previously returned by pista_pager_pin(). */
void pista_pager_unpin(PistaPager *pg, uint64_t tok);

/** Snapshot cache counters (thread-safe). */
void pista_pager_stats(PistaPager *pg, PistaPagerStats *out);

#endif /* PISTADB_PAGER_H */
