/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - index_ivf.h
 * Inverted File Index (IVF) using k-means clustering.
 */
#ifndef PISTADB_INDEX_IVF_H
#define PISTADB_INDEX_IVF_H

#include "pistadb_types.h"
#include "vec_store.h"
#include "distance.h"
#include <stdint.h>

/** One posting in an inverted list. */
typedef struct {
    uint64_t id;
    int      slot;    /* slot in the flat IVFIndex.vectors array */
} IVFPosting;

typedef struct {
    /* Centroids: nlist × dim */
    float    *centroids;
    int       nlist;

    /* Inverted lists: one per centroid */
    IVFPosting **lists;
    int         *list_sizes;
    int         *list_caps;

    /* Flat vector storage (all inserted vectors) */
    VecStore  vs;              /* chunked vector + label storage    */
    uint64_t *vec_ids;
    uint8_t  *vec_deleted;
    int       n_vecs;
    int       vec_cap;

    int       dim;
    DistFn    dist_fn;
    int       nprobe;        /* centroids to search at query time */
    int       trained;       /* 0 = not trained */
} IVFIndex;

/**
 * Create IVF index.
 * @param nlist  number of centroids (clusters)
 * @param nprobe number of clusters to probe at query time
 */
int  ivf_create(IVFIndex *idx, int dim, DistFn dist_fn, int nlist, int nprobe);
void ivf_free(IVFIndex *idx);

/**
 * Train the index on a set of training vectors.
 * Must be called before inserting.
 * @param vecs  [n_train × dim] flat array
 */
int  ivf_train(IVFIndex *idx, const float *vecs, int n_train, int max_iter);

/** Insert a single vector (index must be trained). */
int  ivf_insert(IVFIndex *idx, uint64_t id, const char *label, const float *vec);

int  ivf_delete(IVFIndex *idx, uint64_t id);
int  ivf_update(IVFIndex *idx, uint64_t id, const float *vec);

/** KNN search. Returns actual result count. */
int  ivf_search(const IVFIndex *idx, const float *query, int k,
                PistaDBResult *results);

int  ivf_save(const IVFIndex *idx, void **out_buf, size_t *out_size);
int  ivf_load(IVFIndex *idx, const void *buf, size_t size,
              int dim, DistFn dist_fn);

#endif /* PISTADB_INDEX_IVF_H */
