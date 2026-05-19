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
#include "pager.h"
#include <stdlib.h>
#include <string.h>

/* ── Chunk geometry ─────────────────────────────────────────────────────── */
#define VS_CHUNK_BITS  17
#define VS_CHUNK_SIZE  (1 << VS_CHUNK_BITS)   /* 131,072 entries per chunk */
#define VS_CHUNK_MASK  (VS_CHUNK_SIZE - 1)

/* ── Paged-mode staging rings ────────────────────────────────────────────────
 * In paged mode VS_VEC/VS_LABEL stage one record out of the page cache into a
 * small rotating buffer and return a stable pointer, so every existing call
 * site keeps working with no changes.  The vector ring must exceed the largest
 * number of VS_VEC pointers held live simultaneously — the batch-gather paths
 * (linear/ivf/lsh) hold up to BATCH=256; 512 leaves margin.  Labels are only
 * ever read one at a time. */
#define VS_PG_VRING  512
#define VS_PG_LRING   32

/* ── Storage struct ─────────────────────────────────────────────────────── */
typedef struct {
    float    **vchunks;       /* [n_chunks] float arrays; NULL when dim==0 */
    char   (**lchunks)[256];  /* [n_chunks] label arrays                    */
    int       n_chunks;
    int       dim;            /* 0 = label-only mode                        */

    /* ── Paged mode (all zero/NULL in the default resident mode) ─────────
     * When `pager` is non-NULL the vectors+labels live in the file and are
     * served on demand through the LRU page cache; vchunks/lchunks are NULL.
     * Layout per slot in the region: pg_base + slot*pg_stride, with the
     * label at +pg_lbl_rel and the float vector at +pg_vec_rel. */
    PistaPager *pager;
    uint64_t    pg_base;      /* region offset of slot 0 (fixed-stride)     */
    uint64_t    pg_stride;    /* bytes between consecutive slots            */
    uint64_t   *pg_off;       /* OR: per-slot region offsets (variable-     */
                              /* stride formats: HNSW/DiskANN). NULL ⇒ use  */
                              /* the pg_base + slot*pg_stride formula.      */
    uint32_t    pg_lbl_rel;   /* label offset within a slot/node            */
    uint32_t    pg_vec_rel;   /* vector offset within a slot/node           */
    float     (*pg_vbuf);     /* VS_PG_VRING * dim floats                   */
    char     (*pg_lbuf)[256]; /* VS_PG_LRING labels                         */
    int         pg_vnext;     /* next vector ring slot                      */
    int         pg_lnext;     /* next label ring slot                       */
} VecStore;

/* ── Paged fetch: copy `len` bytes at region-relative `off` into `dst`,
 * walking page boundaries so each pin stays within a single page. ───────── */
static inline int vs_pg_get(VecStore *vs, uint64_t off, uint32_t len,
                            void *dst) {
    uint8_t *out = (uint8_t *)dst;
    while (len > 0) {
        uint64_t tok;
        const void *src = pista_pager_pin(vs->pager, off, len, &tok);
        uint32_t got = len;
        if (!src) {
            /* len spans a page boundary — pin the in-page remainder only. */
            PistaPagerStats st;
            pista_pager_stats(vs->pager, &st);
            uint32_t in_page = (uint32_t)(off & (st.page_size - 1));
            got = st.page_size - in_page;
            if (got > len) got = len;
            src = pista_pager_pin(vs->pager, off, got, &tok);
            if (!src) return PISTADB_EIO;
        }
        memcpy(out, src, got);
        pista_pager_unpin(vs->pager, tok);
        out += got; off += got; len -= got;
    }
    return PISTADB_OK;
}

/* Like vs_pg_get but with a 64-bit length, for one-time loads of large
 * resident sections (centroids / posting lists / hash tables). */
static inline int vs_pg_read(VecStore *vs, uint64_t off, uint64_t len,
                             void *dst) {
    uint8_t *out = (uint8_t *)dst;
    while (len > 0) {
        uint32_t chunk = (len > (1u << 28)) ? (1u << 28) : (uint32_t)len;
        int r = vs_pg_get(vs, off, chunk, out);
        if (r != PISTADB_OK) return r;
        out += chunk; off += chunk; len -= chunk;
    }
    return PISTADB_OK;
}

/* Region offset of slot `slot`: explicit table for variable-stride formats
 * (HNSW/DiskANN), else the fixed-stride formula (LINEAR/IVF/LSH). */
static inline uint64_t vs_pg_slot_off(const VecStore *vs, int slot) {
    return vs->pg_off ? vs->pg_off[slot]
                      : vs->pg_base + (uint64_t)slot * vs->pg_stride;
}

static inline float *vs_pg_vec(VecStore *vs, int slot) {
    float *dst = vs->pg_vbuf + (size_t)vs->pg_vnext * (size_t)vs->dim;
    vs->pg_vnext = (vs->pg_vnext + 1) % VS_PG_VRING;
    vs_pg_get(vs, vs_pg_slot_off(vs, slot) + vs->pg_vec_rel,
              (uint32_t)((size_t)vs->dim * sizeof(float)), dst);
    return dst;
}

static inline char *vs_pg_label(VecStore *vs, int slot) {
    char *dst = vs->pg_lbuf[vs->pg_lnext];
    vs->pg_lnext = (vs->pg_lnext + 1) % VS_PG_LRING;
    vs_pg_get(vs, vs_pg_slot_off(vs, slot) + vs->pg_lbl_rel, 256, dst);
    return dst;
}

/* ── vs_open_paged ───────────────────────────────────────────────────────────
 * Shared setup for every *_load_paged: open the pager over the file's vector
 * region and allocate the staging rings.  The caller then fills the layout
 * fields — either (pg_base,pg_stride) for a fixed-stride format or pg_off[]
 * for a variable-stride one — plus pg_lbl_rel/pg_vec_rel.  On failure the
 * VecStore is left clean (pager==NULL) so vs_free is a safe no-op. */
static inline int vs_open_paged(VecStore *vs, const char *path,
                                uint64_t region_off, uint64_t region_len,
                                int dim, size_t cache_bytes) {
    if (dim <= 0) return PISTADB_EINVAL;
    vs->dim   = dim;
    vs->pager = pista_pager_open(path, region_off, region_len, 0, cache_bytes);
    if (!vs->pager) return PISTADB_EIO;
    vs->pg_vbuf = (float *)malloc((size_t)VS_PG_VRING * (size_t)dim
                                  * sizeof(float));
    vs->pg_lbuf = (char (*)[256])malloc((size_t)VS_PG_LRING * 256);
    if (!vs->pg_vbuf || !vs->pg_lbuf) {
        pista_pager_close(vs->pager);
        free(vs->pg_vbuf); free(vs->pg_lbuf);
        vs->pager = NULL; vs->pg_vbuf = NULL; vs->pg_lbuf = NULL;
        return PISTADB_ENOMEM;
    }
    return PISTADB_OK;
}

/* Total slot capacity covered by current chunks. */
#define VS_CAP(vs)  ((vs)->n_chunks << VS_CHUNK_BITS)

/* Pointer to the float vector at slot `slot` (dim > 0 only).
 * Paged mode stages the vector out of the page cache; resident mode indexes
 * straight into the owning chunk (unchanged, hot path untouched). */
#define VS_VEC(vs, slot) \
    ((vs)->pager ? vs_pg_vec((vs), (int)(slot)) : \
     ((vs)->vchunks[(int)((slot) >> VS_CHUNK_BITS)] + \
      (size_t)((unsigned)(slot) & (unsigned)VS_CHUNK_MASK) * (size_t)(vs)->dim))

/* char[256] label at slot `slot`.  Decays to char* in most contexts. */
#define VS_LABEL(vs, slot) \
    ((vs)->pager ? vs_pg_label((vs), (int)(slot)) : \
     ((vs)->lchunks[(int)((slot) >> VS_CHUNK_BITS)][(unsigned)(slot) & (unsigned)VS_CHUNK_MASK]))

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
    vs->pager    = NULL;
    vs->pg_base  = 0;
    vs->pg_stride= 0;
    vs->pg_off   = NULL;
    vs->pg_lbl_rel = vs->pg_vec_rel = 0;
    vs->pg_vbuf  = NULL;
    vs->pg_lbuf  = NULL;
    vs->pg_vnext = vs->pg_lnext = 0;
    return vs_ensure(vs, initial_cap > 0 ? initial_cap : 1);
}

/* ── vs_free ─────────────────────────────────────────────────────────────── */
/* Free all chunks (or tear down the pager) and reset the struct. */
static inline void vs_free(VecStore *vs) {
    if (vs->pager) {
        pista_pager_close(vs->pager);
        free(vs->pg_vbuf);
        free(vs->pg_lbuf);
        free(vs->pg_off);
        vs->pager   = NULL;
        vs->pg_vbuf = NULL;
        vs->pg_lbuf = NULL;
        vs->pg_off  = NULL;
        return;                 /* paged mode never allocated chunks */
    }
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
