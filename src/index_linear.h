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
#include "distance.h"
#include <stdint.h>

typedef struct {
    float    *vectors;      /* [cap × dim] flat array          */
    uint64_t *ids;          /* parallel id array               */
    char    (*labels)[256]; /* parallel label array            */
    uint8_t  *deleted;      /* 1 = logically deleted           */
    int       size;         /* current number of entries       */
    int       cap;          /* allocated capacity              */
    int       dim;
    DistFn    dist_fn;
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

#endif /* PISTADB_INDEX_LINEAR_H */
