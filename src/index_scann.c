/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - index_scann.c
 * ScaNN: Scalable Nearest Neighbors with Anisotropic Vector Quantization.
 *
 * Reference: "Accelerating Large-Scale Inference with Anisotropic Vector
 * Quantization", Guo et al., ICML 2020.
 *
 * Key innovation over IVF-PQ
 * ─────────────────────────
 * Standard PQ minimises L2 reconstruction error:  min E[ ‖r − Q(r)‖² ]
 *
 * For MIPS/cosine retrieval we instead want to minimise the expected
 * inner-product error.  Since typical queries q are positively correlated
 * with data vectors x (they have high dot-products by design), errors
 * parallel to x̂ = x/‖x‖ hurt recall more than perpendicular errors.
 *
 * Anisotropic PQ training minimises:
 *
 *   L_APQ(r) = ‖r − Q(r)‖² + η · ( x̂ · (r − Q(r)) )²
 *
 * This is equivalent to training standard PQ on the transformed residuals:
 *
 *   r̃ = r + η · (r · x̂) · x̂          (parallel component amplified)
 *
 * At query time the query residual is transformed the same way using q̂:
 *
 *   q̃ = q_res + η · (q_res · q̂) · q̂
 *
 * The ADC lookup table is then built from q̃, and the approximate distance
 * is Σ_m lut[m][code_m] — a standard asymmetric distance computation.
 *
 * After collecting rerank_k approximate candidates, we re-score them with
 * the exact distance function using the raw float vectors stored in each
 * inverted-list entry, and return the true top k.
 */
#include "index_scann.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

/* ── K-means with k-means++ initialisation ──────────────────────────────── */

static int scann_kmeans(const float *data, int n, int dim, int K,
                        float *centroids, int max_iter, uint64_t seed) {
    PCG rng;
    pcg_seed(&rng, seed);

    /* k-means++ initialisation */
    int first = (int)(pcg_u32(&rng) % (uint32_t)n);
    memcpy(centroids, data + (size_t)first * dim,
           sizeof(float) * (size_t)dim);

    float *min_d = (float *)malloc(sizeof(float) * (size_t)n);
    if (!min_d) return PISTADB_ENOMEM;

    for (int k = 1; k < K; k++) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            float bd = FLT_MAX;
            for (int j = 0; j < k; j++) {
                float d = dist_l2sq(data     + (size_t)i * dim,
                                    centroids + (size_t)j * dim, dim);
                if (d < bd) bd = d;
            }
            min_d[i] = bd;
            sum += bd;
        }
        double t = pcg_f32(&rng) * sum;
        double cum = 0.0;
        int chosen = n - 1;
        for (int i = 0; i < n; i++) {
            cum += min_d[i];
            if (cum >= t) { chosen = i; break; }
        }
        memcpy(centroids + (size_t)k * dim,
               data + (size_t)chosen * dim,
               sizeof(float) * (size_t)dim);
    }
    free(min_d);

    int   *assign = (int   *)malloc(sizeof(int)   * (size_t)n);
    float *new_c  = (float *)calloc((size_t)(K * dim), sizeof(float));
    int   *cnt    = (int   *)calloc((size_t)K, sizeof(int));
    if (!assign || !new_c || !cnt) {
        free(assign); free(new_c); free(cnt);
        return PISTADB_ENOMEM;
    }
    memset(assign, -1, sizeof(int) * (size_t)n);

    for (int iter = 0; iter < max_iter; iter++) {
        int changed = 0;
        for (int i = 0; i < n; i++) {
            int nc = 0; float bd = FLT_MAX;
            for (int k = 0; k < K; k++) {
                float d = dist_l2sq(data      + (size_t)i * dim,
                                    centroids + (size_t)k * dim, dim);
                if (d < bd) { bd = d; nc = k; }
            }
            if (nc != assign[i]) changed++;
            assign[i] = nc;
        }
        if (!changed) break;

        memset(new_c, 0, sizeof(float) * (size_t)(K * dim));
        memset(cnt,   0, sizeof(int)   * (size_t)K);
        for (int i = 0; i < n; i++) {
            float       *c = new_c + (size_t)assign[i] * dim;
            const float *v = data  + (size_t)i          * dim;
            for (int d = 0; d < dim; d++) c[d] += v[d];
            cnt[assign[i]]++;
        }
        for (int k = 0; k < K; k++) {
            if (!cnt[k]) continue;
            float *c = new_c + (size_t)k * dim;
            float inv = 1.0f / (float)cnt[k];
            for (int d = 0; d < dim; d++) c[d] *= inv;
        }
        memcpy(centroids, new_c, sizeof(float) * (size_t)(K * dim));
    }
    free(assign); free(new_c); free(cnt);
    return PISTADB_OK;
}

/* ── Anisotropic residual transform ─────────────────────────────────────── */
/*
 * r̃ = r + η · (r · x̂) · x̂
 *
 * where x̂ = x / ‖x‖ is the direction of the ORIGINAL (pre-residual) data
 * vector x.  Amplifying the component of r parallel to x ensures that
 * k-means on {r̃} preferentially minimises inner-product reconstruction
 * error rather than raw L2 reconstruction error.
 *
 * The transform applied to both training residuals (using the data vector)
 * and query residuals (using the query vector) keeps training and inference
 * in the same anisotropically-weighted metric space, making ADC consistent.
 */
static void aq_transform(const float *r,    /* residual                  */
                          const float *x,    /* original vector (for x̂)  */
                          int          dim,
                          float        eta,
                          float       *out)  /* output r̃                 */ {
    if (eta == 0.0f) {
        memcpy(out, r, sizeof(float) * (size_t)dim);
        return;
    }
    float norm_x = vec_norm(x, dim);
    if (norm_x < 1e-9f) {
        memcpy(out, r, sizeof(float) * (size_t)dim);
        return;
    }
    float inv = 1.0f / norm_x;
    /* dot = r · x̂ */
    float dot = 0.0f;
    for (int d = 0; d < dim; d++) dot += r[d] * x[d] * inv;
    /* r̃ = r + η · dot · x̂  =  r + (η · dot · inv) · x */
    float scale = eta * dot * inv;
    for (int d = 0; d < dim; d++) out[d] = r[d] + scale * x[d];
}

/* ── PQ encode: find nearest codebook entry in each sub-space ───────────── */

static void pq_encode(const ScaNNIndex *idx,
                      const float *r_tilde,   /* transformed residual    */
                      uint8_t     *codes) {
    int sub_dim = idx->sub_dim;
    for (int m = 0; m < idx->pq_M; m++) {
        const float *sub = r_tilde + m * sub_dim;
        const float *cb  = idx->codebooks + (size_t)(m * idx->K_sub * sub_dim);
        int best = 0; float bd = FLT_MAX;
        for (int k = 0; k < idx->K_sub; k++) {
            float d = dist_l2sq(sub, cb + (size_t)k * sub_dim, sub_dim);
            if (d < bd) { bd = d; best = k; }
        }
        codes[m] = (uint8_t)best;
    }
}

/* ── Build ADC lookup table for a (transformed) query residual ───────────── */
/*
 * lut[m * K_sub + k] = ‖q̃_m − codebook[m][k]‖²
 * where q̃_m is sub-vector m of the anisotropically-transformed query residual.
 */
static void pq_build_lut(const ScaNNIndex *idx,
                          const float *q_tilde,
                          float       *lut) {
    int sub_dim = idx->sub_dim;
    for (int m = 0; m < idx->pq_M; m++) {
        const float *q_sub = q_tilde + m * sub_dim;
        const float *cb    = idx->codebooks + (size_t)(m * idx->K_sub * sub_dim);
        for (int k = 0; k < idx->K_sub; k++)
            lut[(size_t)m * idx->K_sub + k] =
                dist_l2sq(q_sub, cb + (size_t)k * sub_dim, sub_dim);
    }
}

/* ── ADC: accumulate approximate distance from LUT + codes ──────────────── */

static float pq_adc(const ScaNNIndex *idx,
                    const float      *lut,
                    const uint8_t    *codes) {
    float d = 0.0f;
    for (int m = 0; m < idx->pq_M; m++)
        d += lut[(size_t)m * idx->K_sub + codes[m]];
    return d;
}

/* ── Bytes per inverted-list entry ──────────────────────────────────────── */

static inline int entry_sz(const ScaNNIndex *idx) {
    /* 8 (id) + pq_M (codes) + dim*4 (raw floats for reranking) */
    return 8 + idx->pq_M + idx->dim * (int)sizeof(float);
}

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

int scann_create(ScaNNIndex *idx, int dim, DistFn dist_fn, PistaDBMetric metric,
                 int nlist, int nprobe, int pq_M, int pq_bits,
                 int rerank_k, float aq_eta) {
    memset(idx, 0, sizeof(*idx));
    if (dim % pq_M != 0)               return PISTADB_EINVAL;
    if (pq_bits != 4 && pq_bits != 8)  return PISTADB_EINVAL;
    if (nlist < 1 || pq_M < 1)         return PISTADB_EINVAL;

    idx->dim      = dim;
    idx->dist_fn  = dist_fn;
    idx->metric   = metric;
    idx->nlist    = nlist;
    idx->nprobe   = (nprobe < nlist) ? nprobe : nlist;
    idx->pq_M     = pq_M;
    idx->pq_bits  = pq_bits;
    idx->K_sub    = 1 << pq_bits;
    idx->sub_dim  = dim / pq_M;
    idx->aq_eta   = aq_eta;
    idx->rerank_k = (rerank_k > 0) ? rerank_k : 100;
    idx->vec_cap  = 64;

    idx->centroids  = (float    *)calloc((size_t)(nlist * dim), sizeof(float));
    idx->codebooks  = (float    *)calloc(
                          (size_t)(pq_M * idx->K_sub * idx->sub_dim),
                          sizeof(float));
    idx->lists      = (uint8_t **)calloc((size_t)nlist, sizeof(uint8_t *));
    idx->list_sizes = (int       *)calloc((size_t)nlist, sizeof(int));
    idx->list_caps  = (int       *)calloc((size_t)nlist, sizeof(int));
    idx->all_ids    = (uint64_t  *)malloc(sizeof(uint64_t) * (size_t)idx->vec_cap);
    idx->all_deleted= (uint8_t   *)calloc((size_t)idx->vec_cap, 1);

    if (!idx->centroids || !idx->codebooks || !idx->lists ||
        !idx->list_sizes || !idx->list_caps ||
        !idx->all_ids || !idx->all_deleted)
        return PISTADB_ENOMEM;
    if (vs_init(&idx->vs, 0, idx->vec_cap) != PISTADB_OK) return PISTADB_ENOMEM;
    return PISTADB_OK;
}

void scann_free(ScaNNIndex *idx) {
    free(idx->centroids);
    free(idx->codebooks);
    for (int c = 0; c < idx->nlist; c++) free(idx->lists[c]);
    free(idx->lists);
    free(idx->list_sizes);
    free(idx->list_caps);
    free(idx->all_ids);
    vs_free(&idx->vs);
    free(idx->all_deleted);
    memset(idx, 0, sizeof(*idx));
}

/* ── Training ────────────────────────────────────────────────────────────── */

int scann_train(ScaNNIndex *idx, const float *vecs, int n_train, int max_iter) {
    if (n_train < idx->nlist) return PISTADB_EINVAL;

    int dim   = idx->dim;
    int nlist = idx->nlist;
    int pq_M  = idx->pq_M;

    /* 1. Train coarse IVF quantiser on raw vectors */
    int r = scann_kmeans(vecs, n_train, dim, nlist,
                         idx->centroids, max_iter, 42ULL);
    if (r != PISTADB_OK) return r;

    /* 2. Compute residuals and apply anisotropic transform */
    float *residuals = (float *)malloc(sizeof(float) * (size_t)(n_train * dim));
    float *raw_r     = (float *)malloc(sizeof(float) * (size_t)dim);
    int   *assign    = (int   *)malloc(sizeof(int)   * (size_t)n_train);
    if (!residuals || !raw_r || !assign) {
        free(residuals); free(raw_r); free(assign);
        return PISTADB_ENOMEM;
    }

    for (int i = 0; i < n_train; i++) {
        const float *v  = vecs + (size_t)i * dim;

        /* Find nearest coarse centroid */
        int best = 0; float bd = FLT_MAX;
        for (int c = 0; c < nlist; c++) {
            float d = dist_l2sq(v, idx->centroids + (size_t)c * dim, dim);
            if (d < bd) { bd = d; best = c; }
        }
        assign[i] = best;

        /* raw_r = v − centroid */
        const float *cc = idx->centroids + (size_t)best * dim;
        for (int d = 0; d < dim; d++) raw_r[d] = v[d] - cc[d];

        /* r̃ = raw_r + η · (raw_r · x̂) · x̂  (x̂ comes from original v) */
        aq_transform(raw_r, v, dim, idx->aq_eta,
                     residuals + (size_t)i * dim);
    }
    free(raw_r);
    free(assign);

    /* 3. Train PQ codebooks on transformed residuals, one sub-space at a time */
    int sub_dim = idx->sub_dim;
    float *sub_data = (float *)malloc(sizeof(float) * (size_t)(n_train * sub_dim));
    if (!sub_data) { free(residuals); return PISTADB_ENOMEM; }

    for (int m = 0; m < pq_M; m++) {
        for (int i = 0; i < n_train; i++) {
            memcpy(sub_data + (size_t)i * sub_dim,
                   residuals + (size_t)i * dim + m * sub_dim,
                   sizeof(float) * (size_t)sub_dim);
        }
        float *cb = idx->codebooks + (size_t)(m * idx->K_sub * sub_dim);
        r = scann_kmeans(sub_data, n_train, sub_dim, idx->K_sub,
                         cb, max_iter, (uint64_t)(m + 43));
        if (r != PISTADB_OK) break;
    }

    free(sub_data);
    free(residuals);
    idx->trained = (r == PISTADB_OK) ? 1 : 0;
    return r;
}

/* ── Append an entry to an inverted list ─────────────────────────────────── */

static int list_push(ScaNNIndex *idx, int c,
                     uint64_t id, const uint8_t *codes, const float *raw_vec) {
    int esz = entry_sz(idx);
    if (idx->list_sizes[c] == idx->list_caps[c]) {
        int nc = idx->list_caps[c] * 2 + 8;
        uint8_t *nl = (uint8_t *)realloc(idx->lists[c], (size_t)(nc * esz));
        if (!nl) return PISTADB_ENOMEM;
        idx->lists[c]     = nl;
        idx->list_caps[c] = nc;
    }
    uint8_t *entry = idx->lists[c] + (size_t)idx->list_sizes[c] * esz;
    *(uint64_t *)entry = id;
    memcpy(entry + 8,                    codes,   (size_t)idx->pq_M);
    memcpy(entry + 8 + idx->pq_M, raw_vec, sizeof(float) * (size_t)idx->dim);
    idx->list_sizes[c]++;
    return PISTADB_OK;
}

/* ── Insert ──────────────────────────────────────────────────────────────── */

int scann_insert(ScaNNIndex *idx, uint64_t id, const char *label,
                 const float *vec) {
    if (!idx->trained) return PISTADB_ENOTRAINED;

    /* Grow global id / label store if needed */
    if (idx->n_vecs == idx->vec_cap) {
        int nc = idx->vec_cap * 2 + 8;
        int rs = vs_ensure(&idx->vs, nc);
        if (rs != PISTADB_OK) return rs;
        uint64_t *ni = (uint64_t *)realloc(idx->all_ids,     sizeof(uint64_t) * (size_t)nc);
        uint8_t  *nd = (uint8_t  *)realloc(idx->all_deleted, (size_t)nc);
        if (!ni || !nd) return PISTADB_ENOMEM;
        memset(nd + idx->vec_cap, 0, (size_t)(nc - idx->vec_cap));
        idx->all_ids     = ni;
        idx->all_deleted = nd;
        idx->vec_cap     = nc;
    }
    int slot = idx->n_vecs++;
    idx->all_ids[slot]     = id;
    idx->all_deleted[slot] = 0;
    if (label) strncpy(VS_LABEL(&idx->vs, slot), label, 255);
    else        VS_LABEL(&idx->vs, slot)[0] = '\0';
    VS_LABEL(&idx->vs, slot)[255] = '\0';

    /* Find nearest coarse centroid */
    int best = 0; float bd = FLT_MAX;
    for (int c = 0; c < idx->nlist; c++) {
        float d = dist_l2sq(vec,
                            idx->centroids + (size_t)c * idx->dim,
                            idx->dim);
        if (d < bd) { bd = d; best = c; }
    }

    /* Compute residual and apply anisotropic transform */
    float *raw_r   = (float *)malloc(sizeof(float) * (size_t)idx->dim);
    float *r_tilde = (float *)malloc(sizeof(float) * (size_t)idx->dim);
    if (!raw_r || !r_tilde) { free(raw_r); free(r_tilde); return PISTADB_ENOMEM; }

    const float *cc = idx->centroids + (size_t)best * idx->dim;
    for (int d = 0; d < idx->dim; d++) raw_r[d] = vec[d] - cc[d];
    aq_transform(raw_r, vec, idx->dim, idx->aq_eta, r_tilde);
    free(raw_r);

    /* PQ encode the transformed residual */
    uint8_t *codes = (uint8_t *)malloc((size_t)idx->pq_M);
    if (!codes) { free(r_tilde); return PISTADB_ENOMEM; }
    pq_encode(idx, r_tilde, codes);
    free(r_tilde);

    /* Store: code + raw vector (for Phase-2 reranking) */
    int r = list_push(idx, best, id, codes, vec);
    free(codes);
    return r;
}

/* ── Delete ──────────────────────────────────────────────────────────────── */

int scann_delete(ScaNNIndex *idx, uint64_t id) {
    for (int i = 0; i < idx->n_vecs; i++) {
        if (idx->all_ids[i] == id && !idx->all_deleted[i]) {
            idx->all_deleted[i] = 1;
            return PISTADB_OK;
        }
    }
    return PISTADB_ENOTFOUND;
}

/* ── Update ──────────────────────────────────────────────────────────────── */

int scann_update(ScaNNIndex *idx, uint64_t id, const float *vec) {
    scann_delete(idx, id);
    char lbl[256] = {0};
    for (int i = 0; i < idx->n_vecs; i++) {
        if (idx->all_ids[i] == id) {
            strncpy(lbl, VS_LABEL(&idx->vs, i), 255);
            break;
        }
    }
    return scann_insert(idx, id, lbl, vec);
}

/* ── Two-phase search ────────────────────────────────────────────────────── */

int scann_search(const ScaNNIndex *idx, const float *query, int k,
                 PistaDBResult *results) {
    if (!idx->trained || k <= 0) return 0;

    int dim      = idx->dim;
    int nlist    = idx->nlist;
    int nprobe   = (idx->nprobe < nlist) ? idx->nprobe : nlist;
    int rerank_k = (idx->rerank_k > k)   ? idx->rerank_k : k;

    /* ── Identify the nprobe nearest coarse centroids ─────────────────── */
    float *c_dists = (float *)malloc(sizeof(float) * (size_t)nlist);
    int   *c_probe = (int   *)malloc(sizeof(int)   * (size_t)nprobe);
    int   *c_order = (int   *)malloc(sizeof(int)   * (size_t)nlist);
    if (!c_dists || !c_probe || !c_order) {
        free(c_dists); free(c_probe); free(c_order);
        return 0;
    }
    for (int c = 0; c < nlist; c++) {
        c_dists[c] = dist_l2sq(query,
                               idx->centroids + (size_t)c * dim, dim);
        c_order[c] = c;
    }
    /* Partial selection-sort to extract nprobe smallest distances */
    for (int i = 0; i < nprobe; i++) {
        int best = i;
        for (int j = i + 1; j < nlist; j++)
            if (c_dists[c_order[j]] < c_dists[c_order[best]]) best = j;
        int t = c_order[i]; c_order[i] = c_order[best]; c_order[best] = t;
    }
    for (int i = 0; i < nprobe; i++) c_probe[i] = c_order[i];
    free(c_dists);
    free(c_order);

    /* ── Allocate per-query working buffers ───────────────────────────── */
    float *q_res   = (float *)malloc(sizeof(float) * (size_t)dim);
    float *q_tilde = (float *)malloc(sizeof(float) * (size_t)dim);
    float *lut     = (float *)malloc(
                         sizeof(float) * (size_t)(idx->pq_M * idx->K_sub));
    if (!q_res || !q_tilde || !lut) {
        free(c_probe); free(q_res); free(q_tilde); free(lut);
        return 0;
    }

    /* ── Phase 1: ADC approximate scoring ────────────────────────────── */
    /* max-heap: keep the rerank_k smallest approximate distances         */
    Heap cand_heap;
    if (heap_init(&cand_heap, rerank_k + 4, /*is_max=*/1) != PISTADB_OK) {
        free(c_probe); free(q_res); free(q_tilde); free(lut);
        return 0;
    }

    int esz = entry_sz(idx);
    for (int pi = 0; pi < nprobe; pi++) {
        int c = c_probe[pi];
        const float *cc = idx->centroids + (size_t)c * dim;

        /* Query residual w.r.t. this centroid */
        for (int d = 0; d < dim; d++) q_res[d] = query[d] - cc[d];

        /* Anisotropic transform of query residual (using query direction) */
        aq_transform(q_res, query, dim, idx->aq_eta, q_tilde);

        /* Pre-compute ADC lookup table for this centroid */
        pq_build_lut(idx, q_tilde, lut);

        /* Score every entry in this partition */
        const uint8_t *list = idx->lists[c];
        for (int j = 0; j < idx->list_sizes[c]; j++) {
            const uint8_t *entry = list + (size_t)j * esz;
            uint64_t vid = *(const uint64_t *)entry;

            /* Skip deleted vectors */
            int del = 0;
            for (int s = 0; s < idx->n_vecs; s++) {
                if (idx->all_ids[s] == vid) { del = idx->all_deleted[s]; break; }
            }
            if (del) continue;

            float approx_d = pq_adc(idx, lut, entry + 8);

            if (cand_heap.size < rerank_k)
                heap_push(&cand_heap, approx_d, vid);
            else if (approx_d < heap_top(&cand_heap).key) {
                heap_pop(&cand_heap);
                heap_push(&cand_heap, approx_d, vid);
            }
        }
    }
    free(q_res);
    free(q_tilde);
    free(lut);

    int n_cand = cand_heap.size;
    if (n_cand == 0) {
        heap_free(&cand_heap);
        free(c_probe);
        return 0;
    }

    /* Drain the heap to obtain candidate ids */
    uint64_t *cand_ids = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)n_cand);
    if (!cand_ids) {
        heap_free(&cand_heap);
        free(c_probe);
        return 0;
    }
    /* Pop in any order — we only need the set of ids */
    for (int i = 0; i < n_cand; i++)
        cand_ids[i] = heap_pop(&cand_heap).id;
    heap_free(&cand_heap);

    /* ── Phase 2: Exact reranking using stored raw vectors ───────────── */
    /* All Phase-1 candidates came from the nprobe lists, so we only need
     * to scan those same lists.  This avoids an O(n) full scan.          */
    Heap final_heap;
    if (heap_init(&final_heap, k + 4, /*is_max=*/1) != PISTADB_OK) {
        free(cand_ids);
        free(c_probe);
        return 0;
    }

    for (int pi = 0; pi < nprobe; pi++) {
        int c = c_probe[pi];
        const uint8_t *list = idx->lists[c];
        for (int j = 0; j < idx->list_sizes[c]; j++) {
            const uint8_t *entry = list + (size_t)j * esz;
            uint64_t vid = *(const uint64_t *)entry;

            /* Check if this entry is a Phase-1 candidate */
            int is_cand = 0;
            for (int ci = 0; ci < n_cand; ci++) {
                if (cand_ids[ci] == vid) { is_cand = 1; break; }
            }
            if (!is_cand) continue;

            /* Skip deleted */
            int del = 0;
            for (int s = 0; s < idx->n_vecs; s++) {
                if (idx->all_ids[s] == vid) { del = idx->all_deleted[s]; break; }
            }
            if (del) continue;

            /* Exact distance using stored raw float vector */
            const float *raw_vec =
                (const float *)(entry + 8 + idx->pq_M);
            float exact_d = idx->dist_fn(query, raw_vec, dim);

            if (final_heap.size < k)
                heap_push(&final_heap, exact_d, vid);
            else if (exact_d < heap_top(&final_heap).key) {
                heap_pop(&final_heap);
                heap_push(&final_heap, exact_d, vid);
            }
        }
    }
    free(cand_ids);
    free(c_probe);

    /* Extract results in ascending distance order */
    int cnt = final_heap.size;
    float    *ds = (float    *)malloc(sizeof(float)    * (size_t)(cnt + 1));
    uint64_t *is = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(cnt + 1));
    if (!ds || !is) {
        free(ds); free(is);
        heap_free(&final_heap);
        return 0;
    }
    for (int i = cnt - 1; i >= 0; i--) {
        HeapItem it = heap_pop(&final_heap);
        ds[i] = it.key;
        is[i] = it.id;
    }
    heap_free(&final_heap);

    /* Fill output results with labels */
    for (int i = 0; i < cnt; i++) {
        results[i].id       = is[i];
        results[i].distance = ds[i];
        results[i].label[0] = '\0';
        for (int j = 0; j < idx->n_vecs; j++) {
            if (idx->all_ids[j] == is[i] && !idx->all_deleted[j]) {
                strncpy(results[i].label, VS_LABEL(&idx->vs, j), 255);
                results[i].label[255] = '\0';
                break;
            }
        }
    }
    free(ds);
    free(is);
    return cnt;
}

/* ── Serialisation ───────────────────────────────────────────────────────── */

int scann_save(const ScaNNIndex *idx, void **out_buf, size_t *out_size) {
    int  esz    = entry_sz(idx);
    /* Header: 12 int32 fields + 1 float32 (aq_eta) + 1 int32 (metric) */
    size_t hdr_sz = sizeof(int32_t) * 13 + sizeof(float);
    size_t cc_sz  = sizeof(float) * (size_t)(idx->nlist * idx->dim);
    size_t cb_sz  = sizeof(float) *
                    (size_t)(idx->pq_M * idx->K_sub * idx->sub_dim);
    size_t id_sz  = (size_t)idx->n_vecs * (sizeof(uint64_t) + 1 + 256);
    size_t lst_sz = (size_t)idx->nlist * sizeof(int32_t);
    for (int c = 0; c < idx->nlist; c++)
        lst_sz += (size_t)idx->list_sizes[c] * esz;

    size_t total = hdr_sz + cc_sz + cb_sz + id_sz + lst_sz;
    uint8_t *buf = (uint8_t *)malloc(total);
    if (!buf) return PISTADB_ENOMEM;
    uint8_t *p = buf;

#define WI32(v) do { *(int32_t  *)p = (int32_t )(v); p += 4; } while (0)
#define WF32(v) do { *(float    *)p = (float   )(v); p += 4; } while (0)
#define WU64(v) do { *(uint64_t *)p = (uint64_t)(v); p += 8; } while (0)

    WI32(idx->nlist);     WI32(idx->dim);      WI32(idx->nprobe);
    WI32(idx->pq_M);      WI32(idx->K_sub);    WI32(idx->sub_dim);
    WI32(idx->pq_bits);   WI32(idx->n_vecs);   WI32(idx->rerank_k);
    WI32(idx->trained);   WI32((int)idx->metric);
    WF32(idx->aq_eta);
    WI32(0); /* reserved */

    memcpy(p, idx->centroids, cc_sz);  p += cc_sz;
    memcpy(p, idx->codebooks, cb_sz);  p += cb_sz;

    for (int i = 0; i < idx->n_vecs; i++) {
        WU64(idx->all_ids[i]);
        *p++ = idx->all_deleted[i];
        memcpy(p, VS_LABEL(&idx->vs, i), 256); p += 256;
    }
    for (int c = 0; c < idx->nlist; c++) {
        WI32(idx->list_sizes[c]);
        size_t bytes = (size_t)idx->list_sizes[c] * esz;
        memcpy(p, idx->lists[c], bytes);
        p += bytes;
    }

#undef WI32
#undef WF32
#undef WU64

    *out_buf  = buf;
    *out_size = (size_t)(p - buf);
    return PISTADB_OK;
}

int scann_load(ScaNNIndex *idx, const void *buf, size_t size,
               int dim, DistFn dist_fn, PistaDBMetric metric) {
    const uint8_t *p = (const uint8_t *)buf;
    (void)size;

#define RI32() (*(const int32_t  *)p); p += 4
#define RF32() (*(const float    *)p); p += 4
#define RU64() (*(const uint64_t *)p); p += 8

    int nlist    = RI32(); int file_dim = RI32(); int nprobe   = RI32();
    int pq_M     = RI32(); int K_sub    = RI32(); int sub_dim  = RI32();
    int pq_bits  = RI32(); int n_vecs   = RI32(); int rerank_k = RI32();
    int trained  = RI32(); int met_raw  = RI32();
    float aq_eta = RF32();
    (void)RI32(); /* reserved */
    (void)K_sub; (void)sub_dim;

    if (file_dim != dim) return PISTADB_ECORRUPT;

    PistaDBMetric stored_metric = (PistaDBMetric)met_raw;
    /* Use the stored metric if it differs (database overrides caller) */
    if (stored_metric != metric) metric = stored_metric;

    int r = scann_create(idx, dim, dist_fn, metric,
                         nlist, nprobe, pq_M, pq_bits, rerank_k, aq_eta);
    if (r != PISTADB_OK) return r;
    idx->trained = trained;

    size_t cc_sz = sizeof(float) * (size_t)(nlist * dim);
    size_t cb_sz = sizeof(float) *
                   (size_t)(pq_M * idx->K_sub * idx->sub_dim);
    memcpy(idx->centroids, p, cc_sz); p += cc_sz;
    memcpy(idx->codebooks, p, cb_sz); p += cb_sz;

    /* Grow global label store */
    {
        int rl = vs_ensure(&idx->vs, n_vecs);
        if (rl != PISTADB_OK) return rl;
        while (idx->vec_cap < n_vecs) {
            int nc = idx->vec_cap * 2 + 8;
            uint64_t *ni = (uint64_t *)realloc(idx->all_ids,     sizeof(uint64_t) * (size_t)nc);
            uint8_t  *nd = (uint8_t  *)realloc(idx->all_deleted, (size_t)nc);
            if (!ni || !nd) return PISTADB_ENOMEM;
            memset(nd + idx->vec_cap, 0, (size_t)(nc - idx->vec_cap));
            idx->all_ids = ni; idx->all_deleted = nd; idx->vec_cap = nc;
        }
    }
    for (int i = 0; i < n_vecs; i++) {
        idx->all_ids[i]     = RU64();
        idx->all_deleted[i] = *p++;
        memcpy(VS_LABEL(&idx->vs, i), p, 256); p += 256;
    }
    idx->n_vecs = n_vecs;

    int esz = entry_sz(idx);
    for (int c = 0; c < nlist; c++) {
        int ls = RI32();
        idx->lists[c]     = (uint8_t *)malloc((size_t)((ls + 1) * esz));
        idx->list_caps[c] = ls + 1;
        idx->list_sizes[c]= ls;
        if (!idx->lists[c]) return PISTADB_ENOMEM;
        memcpy(idx->lists[c], p, (size_t)ls * esz);
        p += (size_t)ls * esz;
    }

#undef RI32
#undef RF32
#undef RU64
    return PISTADB_OK;
}
