/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - vec_store.h
 * Chunked vector + label storage.
 *
 * Replaces the flat `float *vectors` + `char (*labels)[256]` pattern used by
 * all indices.  Each chunk holds VS_CHUNK_SIZE = 131,072 entries, so the
 * largest single allocation is 131,072 × dim × 4 bytes (≈ 64 MB at dim=128),
 * eliminating the ~10 GB contiguous-realloc OOM that Windows hits when
 * doubling a flat array past ~9.4 M records.
 *
 * Modes
 * -----
 *  dim > 0  : allocate both vchunks (float) and lchunks (label)
 *  dim == 0 : label-only mode; vchunks is always NULL
 *             (used by IVF-PQ and ScaNN which store quantised codes instead)
 */
#ifndef PISTADB_VEC_STORE_H
#define PISTADB_VEC_STORE_H

#include "pistadb_types.h"
#include <stdlib.h>
#include <string.h>

/* ── Chunk geometry ─────────────────────────────────────────────────────── */
#define VS_CHUNK_BITS  17
#define VS_CHUNK_SIZE  (1 << VS_CHUNK_BITS)   /* 131,072 entries per chunk */
#define VS_CHUNK_MASK  (VS_CHUNK_SIZE - 1)

/* ── Storage struct ─────────────────────────────────────────────────────── */
typedef struct {
    float    **vchunks;       /* [n_chunks] float arrays; NULL when dim==0 */
    char   (**lchunks)[256];  /* [n_chunks] label arrays                    */
    int       n_chunks;
    int       dim;            /* 0 = label-only mode                        */
} VecStore;

/* Total slot capacity covered by current chunks. */
#define VS_CAP(vs)  ((vs)->n_chunks << VS_CHUNK_BITS)

/* Pointer to the float vector at slot `slot` (dim > 0 only). */
#define VS_VEC(vs, slot) \
    ((vs)->vchunks[(int)((slot) >> VS_CHUNK_BITS)] + \
     (size_t)((unsigned)(slot) & (unsigned)VS_CHUNK_MASK) * (size_t)(vs)->dim)

/* char[256] label at slot `slot`.  Decays to char* in most contexts. */
#define VS_LABEL(vs, slot) \
    ((vs)->lchunks[(int)((slot) >> VS_CHUNK_BITS)][(unsigned)(slot) & (unsigned)VS_CHUNK_MASK])

/* ── vs_ensure ───────────────────────────────────────────────────────────── */
/* Ensure at least `required_cap` slots are backed by chunks.
 * Existing data is untouched; new chunks are zero-initialised. */
static inline int vs_ensure(VecStore *vs, int required_cap) {
    if (required_cap <= 0) required_cap = 1;
    int needed = (required_cap + VS_CHUNK_SIZE - 1) >> VS_CHUNK_BITS;
    if (needed <= vs->n_chunks) return PISTADB_OK;

    /* Grow label chunk-pointer array */
    char (**nl)[256] = (char (**)[256])realloc(
        vs->lchunks, sizeof(*nl) * (size_t)needed);
    if (!nl) return PISTADB_ENOMEM;
    vs->lchunks = nl;

    /* Grow vector chunk-pointer array (dim > 0 only) */
    if (vs->dim > 0) {
        float **nv = (float **)realloc(
            vs->vchunks, sizeof(float *) * (size_t)needed);
        if (!nv) return PISTADB_ENOMEM;
        vs->vchunks = nv;
    }

    /* Allocate new chunks */
    for (int i = vs->n_chunks; i < needed; i++) {
        vs->lchunks[i] = (char (*)[256])calloc(VS_CHUNK_SIZE, 256);
        if (!vs->lchunks[i]) return PISTADB_ENOMEM;
        if (vs->dim > 0) {
            vs->vchunks[i] = (float *)malloc(
                sizeof(float) * (size_t)(VS_CHUNK_SIZE * vs->dim));
            if (!vs->vchunks[i]) {
                free(vs->lchunks[i]);
                vs->lchunks[i] = NULL;
                return PISTADB_ENOMEM;
            }
        }
        vs->n_chunks = i + 1;
    }
    return PISTADB_OK;
}

/* ── vs_init ─────────────────────────────────────────────────────────────── */
/* Initialise and allocate enough chunks for `initial_cap` entries.
 * dim == 0 activates label-only mode. */
static inline int vs_init(VecStore *vs, int dim, int initial_cap) {
    vs->dim      = dim;
    vs->n_chunks = 0;
    vs->vchunks  = NULL;
    vs->lchunks  = NULL;
    return vs_ensure(vs, initial_cap > 0 ? initial_cap : 1);
}

/* ── vs_free ─────────────────────────────────────────────────────────────── */
/* Free all chunks and reset the struct. */
static inline void vs_free(VecStore *vs) {
    for (int i = 0; i < vs->n_chunks; i++) {
        if (vs->dim > 0 && vs->vchunks) free(vs->vchunks[i]);
        free(vs->lchunks[i]);
    }
    free(vs->vchunks);
    free(vs->lchunks);
    vs->vchunks  = NULL;
    vs->lchunks  = NULL;
    vs->n_chunks = 0;
}

#endif /* PISTADB_VEC_STORE_H */
