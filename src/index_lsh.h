/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - index_lsh.h
 * Locality-Sensitive Hashing index.
 *
 * Implements:
 *  - Random hyperplane hashing (sign-based) for cosine/L2
 *  - E2LSH (random projection + quantisation) for Euclidean
 *  - L hash tables, each with K hash functions
 *
 * For Hamming distance: bit-sampling LSH can be used (same hash, different
 * interpretation – we sample K dimensions and use their sign).
 */
#ifndef PISTADB_INDEX_LSH_H
#define PISTADB_INDEX_LSH_H

#include "pistadb_types.h"
#include "vec_store.h"
#include "distance.h"
#include <stdint.h>

/* Each hash table bucket stores internal slot indices (0..n_vecs-1),
 * not external ids, so search requires no id-to-slot lookup. */
typedef struct {
    int  *slots;
    int   size;
    int   cap;
} LSHBucket;

/* One hash table: 2^K buckets addressed by K-bit hash key. */
typedef struct {
    float    *proj;      /* [K × dim] projection matrix */
    float    *bias;      /* [K] random biases for E2LSH */
    LSHBucket *buckets;  /* [num_buckets] */
    int       num_buckets;
    int       K;         /* hash functions per table */
    int       is_e2lsh;  /* 0 = sign-based, 1 = E2LSH */
    float     w;         /* bucket width (E2LSH only)  */
} LSHTable;

typedef struct {
    LSHTable *tables;
    int       L;         /* number of tables */
    int       K;
    float     w;

    /* All vectors for exact re-ranking */
    VecStore  vs;              /* chunked vector + label storage */
    uint64_t *vec_ids;
    uint8_t  *vec_deleted;
    int       n_vecs;
    int       vec_cap;

    int       dim;
    DistFn    dist_fn;
    PistaDBMetric metric;
} LSHIndex;

int  lsh_create(LSHIndex *idx, int dim, DistFn dist_fn, PistaDBMetric metric,
                int L, int K, float w);
void lsh_free(LSHIndex *idx);

int  lsh_insert(LSHIndex *idx, uint64_t id, const char *label, const float *vec);
int  lsh_delete(LSHIndex *idx, uint64_t id);
int  lsh_update(LSHIndex *idx, uint64_t id, const float *vec);
int  lsh_search(const LSHIndex *idx, const float *query, int k,
                PistaDBResult *results);

int  lsh_save(const LSHIndex *idx, void **out_buf, size_t *out_size);
int  lsh_load(LSHIndex *idx, const void *buf, size_t size,
              int dim, DistFn dist_fn, PistaDBMetric metric);

#endif /* PISTADB_INDEX_LSH_H */
