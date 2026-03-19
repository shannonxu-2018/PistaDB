/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - index_scann.h
 * ScaNN: Scalable Nearest Neighbors with Anisotropic Vector Quantization.
 *
 * Reference: "Accelerating Large-Scale Inference with Anisotropic Vector
 * Quantization", Guo et al., ICML 2020 (Google Research).
 *
 * Architecture
 * ────────────
 * ScaNN is a two-phase ANN algorithm:
 *
 *   Phase 1 – Fast approximate scoring (ADC over PQ codes)
 *     • Partition the dataset into nlist coarse clusters (k-means++).
 *     • Compute residuals r = x − centroid for each vector x.
 *     • Apply anisotropic transform before PQ training:
 *
 *         r̃ = r + η · (r · x̂) · x̂
 *
 *       where x̂ = x / ‖x‖ is the direction of the original data vector and
 *       η ≥ 0 is the anisotropic penalty parameter.  Setting η = 0 recovers
 *       standard IVF-PQ.  For MIPS/cosine metrics, η > 0 penalises
 *       quantisation errors parallel to the query direction, improving
 *       inner-product recall.
 *     • Train PQ codebooks on the transformed residuals {r̃_i}.
 *     • At insert time: store both the PQ code (8-bit per subspace) and
 *       the raw float vector alongside each inverted-list entry.
 *
 *   Phase 2 – Exact reranking
 *     • After collecting the top rerank_k candidates by ADC distance,
 *       re-score them with the exact distance function using the stored
 *       raw vectors, and return the true top k.
 *
 * Memory layout of each inverted-list entry (bytes):
 *   [ uint64_t id (8) | uint8_t codes[pq_M] | float raw_vec[dim] ]
 */
#ifndef PISTADB_INDEX_SCANN_H
#define PISTADB_INDEX_SCANN_H

#include "pistadb_types.h"
#include "vec_store.h"
#include "distance.h"
#include <stdint.h>

typedef struct {
    /* ── Coarse IVF quantiser ─────────────────────────────────────────── */
    float   *centroids;        /* [nlist × dim]                           */
    int      nlist;
    int      nprobe;

    /* ── Anisotropic Product Quantization codebooks ───────────────────── */
    /* Trained on anisotropically-transformed residuals.                   */
    float   *codebooks;        /* [pq_M × K_sub × sub_dim]                */
    int      pq_M;
    int      K_sub;            /* 2^pq_bits (16 for 4-bit, 256 for 8-bit) */
    int      sub_dim;          /* dim / pq_M                              */
    int      pq_bits;

    /* ── Anisotropic penalty ──────────────────────────────────────────── */
    /* η = 0 → standard PQ; η > 0 → amplify parallel-component error.    */
    float    aq_eta;

    /* ── Per-partition inverted lists ────────────────────────────────── */
    /* Each entry layout (bytes):                                         */
    /*   [ uint64_t id | pq_M × uint8 codes | dim × float raw_vec ]      */
    uint8_t **lists;
    int      *list_sizes;      /* number of stored entries                */
    int      *list_caps;       /* allocated capacity per list             */

    /* ── Global id / label / deletion store ──────────────────────────── */
    uint64_t *all_ids;
    VecStore  vs;              /* label-only chunked store (dim=0) */
    uint8_t  *all_deleted;
    int       n_vecs, vec_cap;

    /* ── Config ───────────────────────────────────────────────────────── */
    int           dim;
    int           trained;
    int           rerank_k;    /* candidates to exact-rerank (≥ k)        */
    DistFn        dist_fn;
    PistaDBMetric metric;
} ScaNNIndex;

/**
 * Create a ScaNN index.
 * @param nlist     number of coarse IVF partitions
 * @param nprobe    partitions to probe at query time
 * @param pq_M      number of PQ sub-spaces (dim must be divisible by pq_M)
 * @param pq_bits   bits per sub-code (4 or 8)
 * @param rerank_k  number of candidates to exact-rerank (should be > k)
 * @param aq_eta    anisotropic penalty η (0.0 = standard PQ; 0.2 recommended
 *                  for cosine/IP metrics)
 */
int  scann_create(ScaNNIndex *idx, int dim, DistFn dist_fn, PistaDBMetric metric,
                  int nlist, int nprobe, int pq_M, int pq_bits,
                  int rerank_k, float aq_eta);
void scann_free(ScaNNIndex *idx);

/**
 * Train the index on a representative set of vectors.
 * Must be called before any inserts.
 * @param vecs      [n_train × dim] row-major float array
 * @param max_iter  maximum k-means iterations
 */
int  scann_train(ScaNNIndex *idx, const float *vecs, int n_train, int max_iter);

int  scann_insert(ScaNNIndex *idx, uint64_t id, const char *label, const float *vec);
int  scann_delete(ScaNNIndex *idx, uint64_t id);
int  scann_update(ScaNNIndex *idx, uint64_t id, const float *vec);

/**
 * Two-phase KNN search.
 * Phase 1: fast ADC over PQ codes → top rerank_k candidates.
 * Phase 2: exact re-scoring of candidates → top k results.
 * @return actual result count (≤ k), or 0 on error / empty index.
 */
int  scann_search(const ScaNNIndex *idx, const float *query, int k,
                  PistaDBResult *results);

int  scann_save(const ScaNNIndex *idx, void **out_buf, size_t *out_size);
int  scann_load(ScaNNIndex *idx, const void *buf, size_t size,
                int dim, DistFn dist_fn, PistaDBMetric metric);

#endif /* PISTADB_INDEX_SCANN_H */
