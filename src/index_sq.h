/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - index_sq.h
 * Scalar Quantization (SQ8) index.
 *
 * Quantizes float32 vectors to uint8 for 4x memory savings and faster
 * distance computation. Per-dimension min/max stats map the float range
 * [min, max] to [0, 255].
 */
#ifndef PISTADB_INDEX_SQ_H
#define PISTADB_INDEX_SQ_H

#include "pistadb_types.h"
#include "vec_store.h"
#include "distance.h"
#include <stdint.h>

typedef struct {
    VecStore  vs;           /* labels only (dim=0 mode)            */
    uint8_t  *codes;        /* quantized vectors: size * dim bytes */
    uint64_t *ids;
    uint8_t  *deleted;
    float    *vmin;         /* per-dimension min (dim floats)      */
    float    *vmax;         /* per-dimension max (dim floats)      */
    int       size;
    int       cap;
    int       dim;
    DistFn    dist_fn;
} SQIndex;

int  sq_create(SQIndex *idx, int dim, DistFn dist_fn, int initial_cap);
void sq_free(SQIndex *idx);

int  sq_insert(SQIndex *idx, uint64_t id, const char *label, const float *vec);
int  sq_delete(SQIndex *idx, uint64_t id);
int  sq_update(SQIndex *idx, uint64_t id, const float *vec);

int  sq_search(const SQIndex *idx, const float *query, int k,
               PistaDBResult *results);

int  sq_save(const SQIndex *idx, void **out_buf, size_t *out_size);
int  sq_load(SQIndex *idx, const void *buf, size_t size,
             int dim, DistFn dist_fn);

#endif /* PISTADB_INDEX_SQ_H */
