/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - index_ivf_pq.c
 * IVF + Product Quantization.
 */
#include "index_ivf_pq.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

/* ── Simple k-means for PQ sub-spaces ───────────────────────────────────── */

static int kmeans(const float *data, int n, int dim, int K, float *centroids,
                  int max_iter, uint64_t rng_seed) {
    PCG rng; pcg_seed(&rng, rng_seed);

    /* k-means++ init */
    int first = (int)(pcg_u32(&rng) % (uint32_t)n);
    memcpy(centroids, data + (size_t)first * dim, sizeof(float) * (size_t)dim);

    float *min_d = (float *)malloc(sizeof(float) * (size_t)n);
    if (!min_d) return PISTADB_ENOMEM;

    for (int k = 1; k < K; k++) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            float bd = FLT_MAX;
            for (int j = 0; j < k; j++) {
                float d = dist_l2sq(data + (size_t)i*dim, centroids + (size_t)j*dim, dim);
                if (d < bd) bd = d;
            }
            min_d[i] = bd; sum += bd;
        }
        double t = pcg_f32(&rng) * sum; double cum = 0.0;
        int chosen = n - 1;
        for (int i = 0; i < n; i++) { cum += min_d[i]; if (cum >= t) { chosen = i; break; } }
        memcpy(centroids + (size_t)k * dim, data + (size_t)chosen * dim, sizeof(float) * (size_t)dim);
    }
    free(min_d);

    int *assign = (int *)malloc(sizeof(int) * (size_t)n);
    float *new_c = (float *)calloc((size_t)(K * dim), sizeof(float));
    int   *cnt   = (int   *)calloc((size_t)K, sizeof(int));
    if (!assign || !new_c || !cnt) { free(assign); free(new_c); free(cnt); return PISTADB_ENOMEM; }

    for (int iter = 0; iter < max_iter; iter++) {
        int changed = 0;
        for (int i = 0; i < n; i++) {
            int nc = 0; float bd = FLT_MAX;
            for (int k = 0; k < K; k++) {
                float d = dist_l2sq(data + (size_t)i*dim, centroids + (size_t)k*dim, dim);
                if (d < bd) { bd = d; nc = k; }
            }
            if (nc != assign[i]) changed++;
            assign[i] = nc;
        }
        if (!changed) break;
        memset(new_c, 0, sizeof(float) * (size_t)(K * dim));
        memset(cnt,   0, sizeof(int)   * (size_t)K);
        for (int i = 0; i < n; i++) {
            float *c = new_c + (size_t)assign[i] * dim;
            const float *v = data + (size_t)i * dim;
            for (int d = 0; d < dim; d++) c[d] += v[d];
            cnt[assign[i]]++;
        }
        for (int k = 0; k < K; k++) {
            if (!cnt[k]) continue;
            float *c = new_c + (size_t)k * dim;
            for (int d = 0; d < dim; d++) c[d] /= cnt[k];
        }
        memcpy(centroids, new_c, sizeof(float) * (size_t)(K * dim));
    }
    free(assign); free(new_c); free(cnt);
    return PISTADB_OK;
}

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

int ivfpq_create(IVFPQIndex *idx, int dim, DistFn dist_fn,
                 int nlist, int nprobe, int pq_M, int nbits) {
    memset(idx, 0, sizeof(*idx));
    if (dim % pq_M != 0) return PISTADB_EINVAL;
    idx->dim     = dim;
    idx->dist_fn = dist_fn;
    idx->nlist   = nlist;
    idx->nprobe  = nprobe < nlist ? nprobe : nlist;
    idx->M       = pq_M;
    idx->nbits   = nbits;
    idx->K_sub   = 1 << nbits;
    idx->sub_dim = dim / pq_M;
    idx->vec_cap = 64;

    idx->coarse_centroids = (float *)calloc((size_t)(nlist * dim), sizeof(float));
    idx->codebooks = (float *)calloc((size_t)(pq_M * idx->K_sub * idx->sub_dim), sizeof(float));
    idx->pq_lists  = (uint8_t **)calloc((size_t)nlist, sizeof(uint8_t *));
    idx->list_sizes= (int       *)calloc((size_t)nlist, sizeof(int));
    idx->list_caps = (int       *)calloc((size_t)nlist, sizeof(int));
    idx->all_ids   = (uint64_t  *)malloc(sizeof(uint64_t)  * (size_t)idx->vec_cap);
    idx->all_labels= (char (*)[256])malloc(256 * (size_t)idx->vec_cap);
    idx->all_deleted=(uint8_t   *)calloc((size_t)idx->vec_cap, 1);

    if (!idx->coarse_centroids || !idx->codebooks || !idx->pq_lists ||
        !idx->list_sizes || !idx->list_caps ||
        !idx->all_ids || !idx->all_labels || !idx->all_deleted)
        return PISTADB_ENOMEM;
    return PISTADB_OK;
}

void ivfpq_free(IVFPQIndex *idx) {
    free(idx->coarse_centroids);
    free(idx->codebooks);
    for (int c = 0; c < idx->nlist; c++) free(idx->pq_lists[c]);
    free(idx->pq_lists); free(idx->list_sizes); free(idx->list_caps);
    free(idx->all_ids); free(idx->all_labels); free(idx->all_deleted);
    memset(idx, 0, sizeof(*idx));
}

/* ── Training ────────────────────────────────────────────────────────────── */

int ivfpq_train(IVFPQIndex *idx, const float *vecs, int n_train, int max_iter) {
    if (n_train < idx->nlist) return PISTADB_EINVAL;
    int dim = idx->dim, nlist = idx->nlist;

    /* 1. Train coarse quantiser (k-means on full vectors) */
    int r = kmeans(vecs, n_train, dim, nlist, idx->coarse_centroids, max_iter, 1);
    if (r != PISTADB_OK) return r;

    /* 2. Compute residuals and assign to coarse clusters */
    float *residuals = (float *)malloc(sizeof(float) * (size_t)(n_train * dim));
    int   *assign    = (int   *)malloc(sizeof(int)   * (size_t)n_train);
    if (!residuals || !assign) { free(residuals); free(assign); return PISTADB_ENOMEM; }

    for (int i = 0; i < n_train; i++) {
        /* Find nearest coarse centroid */
        int best = 0; float bd = FLT_MAX;
        for (int c = 0; c < nlist; c++) {
            float d = dist_l2sq(vecs + (size_t)i*dim,
                                idx->coarse_centroids + (size_t)c*dim, dim);
            if (d < bd) { bd = d; best = c; }
        }
        assign[i] = best;
        /* residual = vec - centroid */
        const float *v = vecs + (size_t)i * dim;
        const float *cc = idx->coarse_centroids + (size_t)best * dim;
        float *res = residuals + (size_t)i * dim;
        for (int d = 0; d < dim; d++) res[d] = v[d] - cc[d];
    }

    /* 3. Train PQ codebooks on residuals for each sub-space */
    int sub_dim = idx->sub_dim;
    float *sub_data = (float *)malloc(sizeof(float) * (size_t)(n_train * sub_dim));
    if (!sub_data) { free(residuals); free(assign); return PISTADB_ENOMEM; }

    for (int m = 0; m < idx->M; m++) {
        /* Gather sub-vectors for sub-space m */
        for (int i = 0; i < n_train; i++) {
            memcpy(sub_data + (size_t)i * sub_dim,
                   residuals + (size_t)i * dim + m * sub_dim,
                   sizeof(float) * (size_t)sub_dim);
        }
        float *cb = idx->codebooks + (size_t)(m * idx->K_sub * sub_dim);
        r = kmeans(sub_data, n_train, sub_dim, idx->K_sub, cb, max_iter, (uint64_t)(m + 2));
        if (r != PISTADB_OK) break;
    }

    free(sub_data); free(residuals); free(assign);
    idx->trained = (r == PISTADB_OK) ? 1 : 0;
    return r;
}

/* ── Encode a vector into PQ codes ──────────────────────────────────────── */

static void pq_encode(const IVFPQIndex *idx, const float *residual, uint8_t *codes) {
    int sub_dim = idx->sub_dim;
    for (int m = 0; m < idx->M; m++) {
        const float *sub = residual + m * sub_dim;
        const float *cb  = idx->codebooks + (size_t)(m * idx->K_sub * sub_dim);
        int best = 0; float bd = FLT_MAX;
        for (int k = 0; k < idx->K_sub; k++) {
            float d = dist_l2sq(sub, cb + (size_t)k * sub_dim, sub_dim);
            if (d < bd) { bd = d; best = k; }
        }
        codes[m] = (uint8_t)best;
    }
}

/* ── Asymmetric distance (query vs PQ-encoded vector) ───────────────────── */
/* Uses pre-computed lookup tables for speed */
static float pq_adc(const IVFPQIndex *idx, const float *lut, const uint8_t *codes) {
    /* lut: [M × K_sub] pre-computed distances from query sub-vectors to codebook */
    float d = 0.0f;
    for (int m = 0; m < idx->M; m++)
        d += lut[(size_t)m * idx->K_sub + codes[m]];
    return d;
}

/* Build lookup table for a query */
static void pq_build_lut(const IVFPQIndex *idx, const float *query_residual, float *lut) {
    int sub_dim = idx->sub_dim;
    for (int m = 0; m < idx->M; m++) {
        const float *q_sub = query_residual + m * sub_dim;
        const float *cb    = idx->codebooks + (size_t)(m * idx->K_sub * sub_dim);
        for (int k = 0; k < idx->K_sub; k++)
            lut[(size_t)m * idx->K_sub + k] = dist_l2sq(q_sub, cb + (size_t)k * sub_dim, sub_dim);
    }
}

/* ── Insert ──────────────────────────────────────────────────────────────── */

static int pq_list_push(IVFPQIndex *idx, int c, uint64_t id, const uint8_t *codes) {
    int entry_size = 8 + idx->M;
    if (idx->list_sizes[c] == idx->list_caps[c]) {
        int nc = idx->list_caps[c] * 2 + 8;
        uint8_t *nl = (uint8_t *)realloc(idx->pq_lists[c], (size_t)(nc * entry_size));
        if (!nl) return PISTADB_ENOMEM;
        idx->pq_lists[c] = nl; idx->list_caps[c] = nc;
    }
    int pos = idx->list_sizes[c];
    uint8_t *entry = idx->pq_lists[c] + (size_t)pos * entry_size;
    *(uint64_t *)entry = id;
    memcpy(entry + 8, codes, (size_t)idx->M);
    idx->list_sizes[c]++;
    return PISTADB_OK;
}

int ivfpq_insert(IVFPQIndex *idx, uint64_t id, const char *label, const float *vec) {
    if (!idx->trained) return PISTADB_ENOTRAINED;

    /* Grow label/id store */
    if (idx->n_vecs == idx->vec_cap) {
        int nc = idx->vec_cap * 2 + 8;
        uint64_t *ni = (uint64_t *)realloc(idx->all_ids,    sizeof(uint64_t) * (size_t)nc);
        char (*nl)[256] = (char (*)[256])realloc(idx->all_labels, 256 * (size_t)nc);
        uint8_t *nd = (uint8_t *)realloc(idx->all_deleted, (size_t)nc);
        if (!ni || !nl || !nd) return PISTADB_ENOMEM;
        memset(nd + idx->vec_cap, 0, (size_t)(nc - idx->vec_cap));
        idx->all_ids = ni; idx->all_labels = nl; idx->all_deleted = nd;
        idx->vec_cap = nc;
    }
    int slot = idx->n_vecs++;
    idx->all_ids[slot] = id;
    idx->all_deleted[slot] = 0;
    if (label) strncpy(idx->all_labels[slot], label, 255);
    else        idx->all_labels[slot][0] = '\0';
    idx->all_labels[slot][255] = '\0';

    /* Coarse assignment */
    int best_c = 0; float bd = FLT_MAX;
    for (int c = 0; c < idx->nlist; c++) {
        float d = dist_l2sq(vec, idx->coarse_centroids + (size_t)c * idx->dim, idx->dim);
        if (d < bd) { bd = d; best_c = c; }
    }
    /* Compute residual */
    float *res = (float *)malloc(sizeof(float) * (size_t)idx->dim);
    if (!res) return PISTADB_ENOMEM;
    const float *cc = idx->coarse_centroids + (size_t)best_c * idx->dim;
    for (int d = 0; d < idx->dim; d++) res[d] = vec[d] - cc[d];
    /* Encode */
    uint8_t *codes = (uint8_t *)malloc((size_t)idx->M);
    if (!codes) { free(res); return PISTADB_ENOMEM; }
    pq_encode(idx, res, codes);
    free(res);
    int r = pq_list_push(idx, best_c, id, codes);
    free(codes);
    return r;
}

int ivfpq_delete(IVFPQIndex *idx, uint64_t id) {
    for (int i = 0; i < idx->n_vecs; i++) {
        if (idx->all_ids[i] == id && !idx->all_deleted[i]) {
            idx->all_deleted[i] = 1;
            return PISTADB_OK;
        }
    }
    return PISTADB_ENOTFOUND;
}

int ivfpq_update(IVFPQIndex *idx, uint64_t id, const float *vec) {
    /* Mark old deleted, re-insert with same id */
    ivfpq_delete(idx, id);
    /* Find label */
    char lbl[256] = {0};
    for (int i = 0; i < idx->n_vecs; i++) {
        if (idx->all_ids[i] == id) { strncpy(lbl, idx->all_labels[i], 255); break; }
    }
    return ivfpq_insert(idx, id, lbl, vec);
}

/* ── Search ──────────────────────────────────────────────────────────────── */

int ivfpq_search(const IVFPQIndex *idx, const float *query, int k,
                 PistaDBResult *results) {
    if (!idx->trained) return 0;

    /* Find nprobe nearest coarse centroids */
    float *c_dists = (float *)malloc(sizeof(float) * (size_t)idx->nlist);
    int   *c_order = (int   *)malloc(sizeof(int)   * (size_t)idx->nlist);
    if (!c_dists || !c_order) { free(c_dists); free(c_order); return 0; }
    for (int c = 0; c < idx->nlist; c++) {
        c_dists[c] = dist_l2sq(query, idx->coarse_centroids + (size_t)c*idx->dim, idx->dim);
        c_order[c] = c;
    }
    for (int i = 0; i < idx->nprobe; i++) {
        int best = i;
        for (int j = i+1; j < idx->nlist; j++)
            if (c_dists[c_order[j]] < c_dists[c_order[best]]) best = j;
        int t = c_order[i]; c_order[i] = c_order[best]; c_order[best] = t;
    }

    /* Build per-centroid LUT (query residual relative to that centroid) */
    float *lut = (float *)malloc(sizeof(float) * (size_t)(idx->M * idx->K_sub));
    float *qres = (float *)malloc(sizeof(float) * (size_t)idx->dim);
    if (!lut || !qres) { free(c_dists); free(c_order); free(lut); free(qres); return 0; }

    Heap heap; heap_init(&heap, k + 4, 1);  /* max-heap */

    int entry_size = 8 + idx->M;
    for (int pi = 0; pi < idx->nprobe && pi < idx->nlist; pi++) {
        int c = c_order[pi];
        const float *cc = idx->coarse_centroids + (size_t)c * idx->dim;
        for (int d = 0; d < idx->dim; d++) qres[d] = query[d] - cc[d];
        pq_build_lut(idx, qres, lut);

        const uint8_t *list = idx->pq_lists[c];
        for (int j = 0; j < idx->list_sizes[c]; j++) {
            const uint8_t *entry = list + (size_t)j * entry_size;
            uint64_t vid = *(const uint64_t *)entry;
            /* Check deleted */
            int del = 0;
            for (int s = 0; s < idx->n_vecs; s++) {
                if (idx->all_ids[s] == vid) { del = idx->all_deleted[s]; break; }
            }
            if (del) continue;
            float d = pq_adc(idx, lut, entry + 8);
            if (heap.size < k) heap_push(&heap, d, vid);
            else if (d < heap_top(&heap).key) { heap_pop(&heap); heap_push(&heap, d, vid); }
        }
    }
    free(c_dists); free(c_order); free(lut); free(qres);

    int cnt = heap.size;
    float *ds = (float    *)malloc(sizeof(float)    * (size_t)cnt);
    uint64_t *is = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)cnt);
    for (int i = cnt - 1; i >= 0; i--) {
        HeapItem it = heap_pop(&heap);
        ds[i] = it.key; is[i] = it.id;
    }
    for (int i = 0; i < cnt; i++) {
        results[i].id = is[i]; results[i].distance = ds[i];
        results[i].label[0] = '\0';
        for (int j = 0; j < idx->n_vecs; j++) {
            if (idx->all_ids[j] == is[i] && !idx->all_deleted[j]) {
                strncpy(results[i].label, idx->all_labels[j], 255);
                results[i].label[255] = '\0'; break;
            }
        }
    }
    free(ds); free(is); heap_free(&heap);
    return cnt;
}

/* ── Serialization ───────────────────────────────────────────────────────── */

int ivfpq_save(const IVFPQIndex *idx, void **out_buf, size_t *out_size) {
    int entry_size = 8 + idx->M;
    size_t cb_sz = sizeof(float) * (size_t)(idx->M * idx->K_sub * idx->sub_dim);
    size_t cc_sz = sizeof(float) * (size_t)(idx->nlist * idx->dim);
    size_t hdr   = sizeof(int32_t) * 8;
    size_t ids_sz= (size_t)idx->n_vecs * (sizeof(uint64_t) + 1 + 256);
    size_t lst_sz= (size_t)idx->nlist  * sizeof(int32_t);
    for (int c = 0; c < idx->nlist; c++)
        lst_sz += (size_t)idx->list_sizes[c] * entry_size;

    size_t total = hdr + cc_sz + cb_sz + ids_sz + lst_sz;
    uint8_t *buf = (uint8_t *)malloc(total);
    if (!buf) return PISTADB_ENOMEM;
    uint8_t *p = buf;

#define WI32(v) do{*(int32_t*)p=(int32_t)(v);p+=4;}while(0)
#define WU64(v) do{*(uint64_t*)p=(uint64_t)(v);p+=8;}while(0)
    WI32(idx->nlist); WI32(idx->dim); WI32(idx->nprobe);
    WI32(idx->M); WI32(idx->K_sub); WI32(idx->sub_dim);
    WI32(idx->nbits); WI32(idx->n_vecs);
    memcpy(p, idx->coarse_centroids, cc_sz); p += cc_sz;
    memcpy(p, idx->codebooks, cb_sz);        p += cb_sz;
    for (int i = 0; i < idx->n_vecs; i++) {
        WU64(idx->all_ids[i]);
        *p++ = idx->all_deleted[i];
        memcpy(p, idx->all_labels[i], 256); p += 256;
    }
    for (int c = 0; c < idx->nlist; c++) {
        WI32(idx->list_sizes[c]);
        memcpy(p, idx->pq_lists[c], (size_t)idx->list_sizes[c] * entry_size);
        p += (size_t)idx->list_sizes[c] * entry_size;
    }
#undef WI32
#undef WU64
    *out_buf  = buf;
    *out_size = (size_t)(p - buf);
    return PISTADB_OK;
}

int ivfpq_load(IVFPQIndex *idx, const void *buf, size_t size, int dim, DistFn dist_fn) {
    const uint8_t *p = (const uint8_t *)buf;
#define RI32() (*(const int32_t*)p); p+=4
#define RU64() (*(const uint64_t*)p); p+=8
    int nlist   = RI32(); int file_dim = RI32(); int nprobe = RI32();
    int pq_M    = RI32(); int K_sub= RI32(); int sub_dim= RI32();
    int nbits   = RI32(); int n_vecs = RI32();
    (void)size; (void)K_sub; (void)sub_dim;
    if (file_dim != dim) return PISTADB_ECORRUPT;

    int r = ivfpq_create(idx, dim, dist_fn, nlist, nprobe, pq_M, nbits);
    if (r != PISTADB_OK) return r;
    idx->trained = 1;

    size_t cc_sz = sizeof(float) * (size_t)(nlist * dim);
    size_t cb_sz = sizeof(float) * (size_t)(pq_M * idx->K_sub * idx->sub_dim);
    memcpy(idx->coarse_centroids, p, cc_sz); p += cc_sz;
    memcpy(idx->codebooks, p, cb_sz);        p += cb_sz;

    /* Grow label store */
    while (idx->vec_cap < n_vecs) {
        int nc = idx->vec_cap * 2 + 8;
        uint64_t *ni = (uint64_t *)realloc(idx->all_ids, sizeof(uint64_t) * (size_t)nc);
        char (*nl)[256] = (char (*)[256])realloc(idx->all_labels, 256 * (size_t)nc);
        uint8_t *nd = (uint8_t *)realloc(idx->all_deleted, (size_t)nc);
        if (!ni||!nl||!nd) return PISTADB_ENOMEM;
        memset(nd + idx->vec_cap, 0, (size_t)(nc - idx->vec_cap));
        idx->all_ids=ni; idx->all_labels=nl; idx->all_deleted=nd; idx->vec_cap=nc;
    }
    for (int i = 0; i < n_vecs; i++) {
        idx->all_ids[i] = RU64();
        idx->all_deleted[i] = *p++;
        memcpy(idx->all_labels[i], p, 256); p += 256;
    }
    idx->n_vecs = n_vecs;

    int entry_size = 8 + pq_M;
    for (int c = 0; c < nlist; c++) {
        int ls = RI32();
        idx->pq_lists[c]    = (uint8_t *)malloc((size_t)(ls + 1) * entry_size);
        idx->list_caps[c]   = ls + 1;
        idx->list_sizes[c]  = ls;
        memcpy(idx->pq_lists[c], p, (size_t)ls * entry_size);
        p += (size_t)ls * entry_size;
    }
#undef RI32
#undef RU64
    return PISTADB_OK;
}
