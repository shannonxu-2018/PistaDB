/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - index_ivf_pq.h
 * IVF with Product Quantization for compressed storage and fast ADC search.
 *
 * Each vector residual (relative to its centroid) is split into M sub-vectors,
 * each encoded with a codebook of K_sub (= 2^nbits) codewords.
 */
#ifndef PISTADB_INDEX_IVF_PQ_H
#define PISTADB_INDEX_IVF_PQ_H

#include "pistadb_types.h"
#include "distance.h"
#include <stdint.h>

typedef struct {
    /* IVF coarse quantiser */
    float  *coarse_centroids;  /* [nlist × dim]                 */
    int     nlist;
    int     nprobe;

    /* PQ codebooks: [M × K_sub × sub_dim] */
    float  *codebooks;
    int     M;          /* number of sub-spaces              */
    int     K_sub;      /* codewords per sub-space (e.g. 256)*/
    int     sub_dim;    /* dim / M                           */
    int     nbits;      /* 8 → K_sub=256, 4 → K_sub=16      */

    /* Inverted lists (per coarse cluster) */
    /* Each posting: uint64 id + M bytes (PQ codes) */
    uint8_t  **pq_lists;    /* [nlist] → array of (8 + M) bytes each */
    int       *list_sizes;
    int       *list_caps;

    /* Original ids in insertion order (for label lookup) */
    uint64_t *all_ids;
    char    (*all_labels)[256];
    uint8_t  *all_deleted;
    int       n_vecs, vec_cap;

    int     dim;
    DistFn  dist_fn;
    int     trained;
} IVFPQIndex;

int  ivfpq_create(IVFPQIndex *idx, int dim, DistFn dist_fn,
                  int nlist, int nprobe, int pq_M, int nbits);
void ivfpq_free(IVFPQIndex *idx);

int  ivfpq_train(IVFPQIndex *idx, const float *vecs, int n_train, int max_iter);
int  ivfpq_insert(IVFPQIndex *idx, uint64_t id, const char *label, const float *vec);
int  ivfpq_delete(IVFPQIndex *idx, uint64_t id);
int  ivfpq_update(IVFPQIndex *idx, uint64_t id, const float *vec);
int  ivfpq_search(const IVFPQIndex *idx, const float *query, int k,
                  PistaDBResult *results);

int  ivfpq_save(const IVFPQIndex *idx, void **out_buf, size_t *out_size);
int  ivfpq_load(IVFPQIndex *idx, const void *buf, size_t size,
                int dim, DistFn dist_fn);

#endif /* PISTADB_INDEX_IVF_PQ_H */
