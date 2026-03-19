/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - index_hnsw.h
 * Hierarchical Navigable Small World graph index.
 *
 * Reference: Malkov & Yashunin (2018) "Efficient and robust approximate
 *            nearest neighbor search using Hierarchical Navigable Small
 *            World graphs."
 */
#ifndef PISTADB_INDEX_HNSW_H
#define PISTADB_INDEX_HNSW_H

#include "pistadb_types.h"
#include "vec_store.h"
#include "distance.h"
#include <stdint.h>

/* Maximum number of layers.  log₂(2^31) = 31, so 48 is very safe. */
#define HNSW_MAX_LAYERS 48

typedef struct HNSWNode {
    uint64_t  vec_id;          /* external vector id                 */
    int       level;           /* highest layer this node appears in */
    int     **neighbors;       /* neighbors[layer][0..cnt-1]         */
    int      *neighbor_cnt;    /* count per layer                    */
    int      *neighbor_cap;    /* capacity per layer                 */
} HNSWNode;

typedef struct {
    /* Graph nodes (parallel to VectorStore slots) */
    HNSWNode *nodes;
    int       n_nodes;
    int       node_cap;

    /* Floating vector data (owned by this index) */
    VecStore  vs;              /* chunked vector + label storage    */

    int       dim;
    DistFn    dist_fn;

    /* HNSW parameters */
    int       M;               /* max connections per layer (≥ 2)   */
    int       M_max0;          /* max connections at layer 0        */
    int       ef_construction;
    int       ef_search;
    float     mL;              /* level multiplier = 1/ln(M)        */

    /* Entry point */
    int       ep_node;         /* index into nodes[], -1 if empty   */
    int       max_layer;

    /* RNG for level generation */
    /* (use global PCG seeded once) */
} HNSWIndex;

int  hnsw_create(HNSWIndex *idx, int dim, DistFn dist_fn,
                 int M, int ef_construction, int ef_search);
void hnsw_free(HNSWIndex *idx);

int  hnsw_insert(HNSWIndex *idx, uint64_t id, const char *label, const float *vec);
/** Lazy deletion – marks the node so it is skipped in searches. */
int  hnsw_delete(HNSWIndex *idx, uint64_t id);
int  hnsw_update(HNSWIndex *idx, uint64_t id, const float *vec);

int  hnsw_search(HNSWIndex *idx, const float *query, int k, int ef,
                 PistaDBResult *results);

int  hnsw_save(const HNSWIndex *idx, void **out_buf, size_t *out_size);
int  hnsw_load(HNSWIndex *idx, const void *buf, size_t size,
               int dim, DistFn dist_fn);

#endif /* PISTADB_INDEX_HNSW_H */
