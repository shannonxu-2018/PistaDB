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

/* ── Batched (one query → many candidates) kernels ──────────────────────── */

/**
 * Compute the distance from a single query to N candidate vectors in one
 * call.  Candidates are passed as an array of N float pointers (each of
 * length `dim`); results are written to `out[0..n-1]` in order.
 *
 * Today these are thin wrappers that hoist the SIMD dispatch out of the
 * inner loop and issue a prefetch for the next candidate vector.  The
 * results are bit-identical to calling the per-pair functions one at a
 * time; the win is amortised dispatch + cache-line prefetch + slightly
 * better register scheduling because the kernel body is monomorphic.
 *
 * Used by Linear / IVF probe / LSH rerank / HNSW fan-out hot loops.
 */
void dist_many_to_one_l2sq    (const float *query, const float *const *vecs,
                                size_t n, int dim, float *out);
void dist_many_to_one_cosine  (const float *query, const float *const *vecs,
                                size_t n, int dim, float *out);
void dist_many_to_one_ip      (const float *query, const float *const *vecs,
                                size_t n, int dim, float *out);
void dist_many_to_one_l1      (const float *query, const float *const *vecs,
                                size_t n, int dim, float *out);
void dist_many_to_one_hamming (const float *query, const float *const *vecs,
                                size_t n, int dim, float *out);

/** Function-pointer type for the batch kernels above. */
typedef void (*BatchDistFn)(const float *query, const float *const *vecs,
                            size_t n, int dim, float *out);

/** Return the batch kernel matching the metric.  For METRIC_L2 this
 *  returns the L2² variant — callers ranking by L2 should sqrtf the
 *  finished top-K (matching pistadb_get_rank_dist_fn). */
BatchDistFn pistadb_get_batch_rank_dist_fn(PistaDBMetric metric);

/**
 * Return a *ranking*-equivalent distance function for the given metric.
 *
 * For METRIC_L2 this returns dist_l2sq (squared L2) — sqrt is monotone over
 * non-negative values, so heap comparisons / top-K orderings are identical
 * but per-call cost drops by one sqrtf.  Callers that surface distances to
 * the user must apply sqrtf to the final K results to recover Euclidean
 * units.  For all other metrics this returns the same function pointer as
 * pistadb_get_dist_fn().
 */
DistFn pistadb_get_rank_dist_fn(PistaDBMetric metric);

/** Inner-product helper (raw dot product, not distance). */
float vec_dot(const float *a, const float *b, int dim);

/** L2 norm of a vector. */
float vec_norm(const float *a, int dim);

#endif /* PISTADB_DISTANCE_H */
