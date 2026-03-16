/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - distance.h
 * Five distance / similarity metrics on float32 vectors.
 */
#ifndef PISTADB_DISTANCE_H
#define PISTADB_DISTANCE_H

#include "pistadb_types.h"

/**
 * All functions return a non-negative "distance" value.
 * For IP and COSINE, which are naturally similarities,
 * we return (1 - similarity) or (-dot) so that
 * "smaller is more similar" holds uniformly.
 */

/** Squared Euclidean distance (L2²). Cheaper than sqrt, monotone for ranking. */
float dist_l2sq(const float *a, const float *b, int dim);

/** Euclidean distance (L2). */
float dist_l2(const float *a, const float *b, int dim);

/** Cosine distance = 1 - cosine_similarity  ∈ [0, 2]. */
float dist_cosine(const float *a, const float *b, int dim);

/** Negative inner product (so smaller = more similar). */
float dist_ip(const float *a, const float *b, int dim);

/** Manhattan (L1) distance. */
float dist_l1(const float *a, const float *b, int dim);

/**
 * Hamming distance: number of differing elements
 * (treats each float as an independent dimension, counts non-equal).
 * For binary vectors stored as floats (0.0 / 1.0) this is exact.
 */
float dist_hamming(const float *a, const float *b, int dim);

/** Function pointer type matching all metrics. */
typedef float (*DistFn)(const float *, const float *, int);

/** Return the distance function for the given metric type. */
DistFn pistadb_get_dist_fn(PistaDBMetric metric);

/** Inner-product helper (raw dot product, not distance). */
float vec_dot(const float *a, const float *b, int dim);

/** L2 norm of a vector. */
float vec_norm(const float *a, int dim);

#endif /* PISTADB_DISTANCE_H */
