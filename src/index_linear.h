/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - index_linear.h
 * Brute-force (linear scan) index.
 */
#ifndef PISTADB_INDEX_LINEAR_H
#define PISTADB_INDEX_LINEAR_H

#include "pistadb_types.h"
#include "vec_store.h"
#include "distance.h"
#include <stdint.h>

typedef struct {
    VecStore     vs;        /* chunked vector + label storage  */
    uint64_t    *ids;       /* parallel id array               */
    uint8_t     *deleted;   /* 1 = logically deleted           */
    int          size;      /* current number of entries       */
    int          cap;       /* allocated capacity              */
    int          dim;
    DistFn       dist_fn;
    BatchDistFn  batch_fn;  /* may be NULL → falls back to dist_fn loop */
} LinearIndex;

int  linear_create(LinearIndex *idx, int dim, DistFn dist_fn, int initial_cap);
void linear_free(LinearIndex *idx);

int  linear_insert(LinearIndex *idx, uint64_t id, const char *label, const float *vec);
int  linear_delete(LinearIndex *idx, uint64_t id);
int  linear_update(LinearIndex *idx, uint64_t id, const float *vec);

/** KNN search. results must hold at least k elements. Returns actual count. */
int  linear_search(const LinearIndex *idx, const float *query, int k,
                   PistaDBResult *results);

/** Find internal slot by id; returns -1 if not found. */
int  linear_find_id(const LinearIndex *idx, uint64_t id);

/* Serialization */
int  linear_save(const LinearIndex *idx, void **out_buf, size_t *out_size);
int  linear_load(LinearIndex *idx, const void *buf, size_t size,
                 int dim, DistFn dist_fn);

/**
 * SQLite-style paged load: the vector section stays in the file and is
 * served on demand through a bounded LRU page cache, so resident memory is
 * capped at `cache_bytes` instead of growing with the database.  Reads the
 * existing on-disk format unchanged (no migration).  The resulting index is
 * read-only for inserts/updates.
 *
 * @param path        .pst file path (kept open by the pager).
 * @param vec_off     File offset of the vector section (hdr.vec_offset).
 * @param vec_size    Byte size of the vector section (hdr.vec_size).
 * @param cache_bytes Page-cache budget; 0 selects the 64 MiB default.
 */
int  linear_load_paged(LinearIndex *idx, const char *path,
                        uint64_t vec_off, uint64_t vec_size,
                        int dim, DistFn dist_fn, size_t cache_bytes);

#endif /* PISTADB_INDEX_LINEAR_H */
