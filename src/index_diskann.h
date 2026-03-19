/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - index_diskann.h
 * Vamana / DiskANN graph-based index.
 *
 * Reference: Subramanya et al. (2019) "DiskANN: Fast Accurate Billion-point
 *            Nearest Neighbor Search on a Single Node."
 */
#ifndef PISTADB_INDEX_DISKANN_H
#define PISTADB_INDEX_DISKANN_H

#include "pistadb_types.h"
#include "vec_store.h"
#include "distance.h"
#include <stdint.h>

typedef struct {
    uint64_t  vec_id;
    int      *neighbors;     /* outgoing edges (node indices) */
    int       neighbor_cnt;
    int       neighbor_cap;
    int       deleted;
} DiskANNNode;

typedef struct {
    DiskANNNode *nodes;
    VecStore     vs;           /* chunked vector + label storage */
    int          n_nodes;
    int          node_cap;

    int          dim;
    DistFn       dist_fn;

    /* Vamana parameters */
    int          R;          /* max degree                */
    int          L;          /* search list size          */
    float        alpha;      /* pruning parameter ≥ 1.0   */

    int          medoid;     /* entry point (medoid node) */
} DiskANNIndex;

int  diskann_create(DiskANNIndex *idx, int dim, DistFn dist_fn,
                    int R, int L, float alpha);
void diskann_free(DiskANNIndex *idx);

/** Insert single vector. */
int  diskann_insert(DiskANNIndex *idx, uint64_t id, const char *label, const float *vec);
int  diskann_delete(DiskANNIndex *idx, uint64_t id);
int  diskann_update(DiskANNIndex *idx, uint64_t id, const float *vec);

/**
 * Build the Vamana graph over all already-inserted vectors.
 * Call this after a bulk insertion for best graph quality.
 */
int  diskann_build(DiskANNIndex *idx);

int  diskann_search(DiskANNIndex *idx, const float *query, int k,
                    PistaDBResult *results);

int  diskann_save(const DiskANNIndex *idx, void **out_buf, size_t *out_size);
int  diskann_load(DiskANNIndex *idx, const void *buf, size_t size,
                  int dim, DistFn dist_fn);

#endif /* PISTADB_INDEX_DISKANN_H */
