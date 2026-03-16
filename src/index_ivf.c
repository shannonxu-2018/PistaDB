/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - index_ivf.c
 * Inverted File Index with k-means clustering.
 */
#include "index_ivf.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static int nearest_centroid(const IVFIndex *idx, const float *vec) {
    int best = 0;
    float best_d = FLT_MAX;
    for (int c = 0; c < idx->nlist; c++) {
        float d = dist_l2sq(vec, idx->centroids + (size_t)c * idx->dim, idx->dim);
        if (d < best_d) { best_d = d; best = c; }
    }
    return best;
}

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

int ivf_create(IVFIndex *idx, int dim, DistFn dist_fn, int nlist, int nprobe) {
    memset(idx, 0, sizeof(*idx));
    idx->dim      = dim;
    idx->dist_fn  = dist_fn;
    idx->nlist    = nlist;
    idx->nprobe   = nprobe < nlist ? nprobe : nlist;
    idx->trained  = 0;
    idx->vec_cap  = 64;
    idx->n_vecs   = 0;

    idx->centroids = (float *)calloc((size_t)(nlist * dim), sizeof(float));
    idx->lists     = (IVFPosting **)calloc((size_t)nlist, sizeof(IVFPosting *));
    idx->list_sizes= (int         *)calloc((size_t)nlist, sizeof(int));
    idx->list_caps = (int         *)calloc((size_t)nlist, sizeof(int));
    idx->vectors   = (float    *)malloc(sizeof(float)    * (size_t)(idx->vec_cap * dim));
    idx->vec_ids   = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)idx->vec_cap);
    idx->vec_labels= (char (*)[256])malloc(256 * (size_t)idx->vec_cap);
    idx->vec_deleted=(uint8_t  *)calloc((size_t)idx->vec_cap, 1);

    if (!idx->centroids || !idx->lists || !idx->list_sizes || !idx->list_caps ||
        !idx->vectors || !idx->vec_ids || !idx->vec_labels || !idx->vec_deleted)
        return PISTADB_ENOMEM;
    return PISTADB_OK;
}

void ivf_free(IVFIndex *idx) {
    free(idx->centroids);
    for (int c = 0; c < idx->nlist; c++) free(idx->lists[c]);
    free(idx->lists); free(idx->list_sizes); free(idx->list_caps);
    free(idx->vectors); free(idx->vec_ids); free(idx->vec_labels); free(idx->vec_deleted);
    memset(idx, 0, sizeof(*idx));
}

/* ── K-means training ────────────────────────────────────────────────────── */

int ivf_train(IVFIndex *idx, const float *vecs, int n_train, int max_iter) {
    if (n_train < idx->nlist) return PISTADB_EINVAL;

    PCG rng; pcg_seed(&rng, 1234);

    /* k-means++ initialisation */
    int dim = idx->dim, nlist = idx->nlist;
    float *C = idx->centroids;

    /* First centroid: random */
    int first = (int)(pcg_u32(&rng) % (uint32_t)n_train);
    memcpy(C, vecs + (size_t)first * dim, sizeof(float) * (size_t)dim);

    float *min_dists = (float *)malloc(sizeof(float) * (size_t)n_train);
    if (!min_dists) return PISTADB_ENOMEM;

    for (int c = 1; c < nlist; c++) {
        /* Compute squared distances to nearest centroid */
        double sum = 0.0;
        for (int i = 0; i < n_train; i++) {
            float best = FLT_MAX;
            for (int j = 0; j < c; j++) {
                float d = dist_l2sq(vecs + (size_t)i * dim, C + (size_t)j * dim, dim);
                if (d < best) best = d;
            }
            min_dists[i] = best;
            sum += best;
        }
        /* Sample proportionally */
        double target = pcg_f32(&rng) * sum;
        double cum = 0.0;
        int chosen = n_train - 1;
        for (int i = 0; i < n_train; i++) {
            cum += min_dists[i];
            if (cum >= target) { chosen = i; break; }
        }
        memcpy(C + (size_t)c * dim, vecs + (size_t)chosen * dim, sizeof(float) * (size_t)dim);
    }
    free(min_dists);

    /* Lloyd's algorithm */
    int *assign = (int *)malloc(sizeof(int) * (size_t)n_train);
    if (!assign) return PISTADB_ENOMEM;

    float *new_C = (float *)calloc((size_t)(nlist * dim), sizeof(float));
    int   *cnt   = (int   *)calloc((size_t)nlist, sizeof(int));
    if (!new_C || !cnt) { free(assign); free(new_C); free(cnt); return PISTADB_ENOMEM; }

    for (int iter = 0; iter < max_iter; iter++) {
        /* Assign */
        int changed = 0;
        for (int i = 0; i < n_train; i++) {
            int nc = 0;
            float bd = FLT_MAX;
            for (int c = 0; c < nlist; c++) {
                float d = dist_l2sq(vecs + (size_t)i * dim, C + (size_t)c * dim, dim);
                if (d < bd) { bd = d; nc = c; }
            }
            if (nc != assign[i]) changed++;
            assign[i] = nc;
        }
        if (changed == 0) break;

        /* Recompute centroids */
        memset(new_C, 0, sizeof(float) * (size_t)(nlist * dim));
        memset(cnt,   0, sizeof(int)   * (size_t)nlist);
        for (int i = 0; i < n_train; i++) {
            int c = assign[i];
            float *nc = new_C + (size_t)c * dim;
            const float *v = vecs + (size_t)i * dim;
            for (int d = 0; d < dim; d++) nc[d] += v[d];
            cnt[c]++;
        }
        for (int c = 0; c < nlist; c++) {
            if (cnt[c] == 0) continue;
            float *nc = new_C + (size_t)c * dim;
            for (int d = 0; d < dim; d++) nc[d] /= (float)cnt[c];
        }
        memcpy(C, new_C, sizeof(float) * (size_t)(nlist * dim));
    }

    free(assign); free(new_C); free(cnt);
    idx->trained = 1;
    return PISTADB_OK;
}

/* ── Insert ──────────────────────────────────────────────────────────────── */

static int vec_store_grow(IVFIndex *idx) {
    int nc = idx->vec_cap * 2 + 8;
    float    *nv = (float    *)realloc(idx->vectors,    sizeof(float)    * (size_t)(nc * idx->dim));
    uint64_t *ni = (uint64_t *)realloc(idx->vec_ids,    sizeof(uint64_t) * (size_t)nc);
    char     (*nl)[256] = (char (*)[256])realloc(idx->vec_labels, 256 * (size_t)nc);
    uint8_t  *nd = (uint8_t  *)realloc(idx->vec_deleted,(size_t)nc);
    if (!nv || !ni || !nl || !nd) return PISTADB_ENOMEM;
    memset(nd + idx->vec_cap, 0, (size_t)(nc - idx->vec_cap));
    idx->vectors = nv; idx->vec_ids = ni; idx->vec_labels = nl; idx->vec_deleted = nd;
    idx->vec_cap = nc;
    return PISTADB_OK;
}

static int list_push(IVFIndex *idx, int c, IVFPosting p) {
    if (idx->list_sizes[c] == idx->list_caps[c]) {
        int nc = idx->list_caps[c] * 2 + 8;
        IVFPosting *nl = (IVFPosting *)realloc(idx->lists[c], sizeof(IVFPosting) * (size_t)nc);
        if (!nl) return PISTADB_ENOMEM;
        idx->lists[c] = nl; idx->list_caps[c] = nc;
    }
    idx->lists[c][idx->list_sizes[c]++] = p;
    return PISTADB_OK;
}

int ivf_insert(IVFIndex *idx, uint64_t id, const char *label, const float *vec) {
    if (!idx->trained) return PISTADB_ENOTRAINED;
    if (idx->n_vecs == idx->vec_cap) {
        int r = vec_store_grow(idx);
        if (r != PISTADB_OK) return r;
    }
    int slot = idx->n_vecs++;
    idx->vec_ids[slot] = id;
    idx->vec_deleted[slot] = 0;
    memcpy(idx->vectors + (size_t)slot * idx->dim, vec, sizeof(float) * (size_t)idx->dim);
    if (label) strncpy(idx->vec_labels[slot], label, 255);
    else        idx->vec_labels[slot][0] = '\0';
    idx->vec_labels[slot][255] = '\0';

    int c = nearest_centroid(idx, vec);
    IVFPosting post = { id, slot };
    return list_push(idx, c, post);
}

int ivf_delete(IVFIndex *idx, uint64_t id) {
    for (int i = 0; i < idx->n_vecs; i++) {
        if (idx->vec_ids[i] == id && !idx->vec_deleted[i]) {
            idx->vec_deleted[i] = 1;
            return PISTADB_OK;
        }
    }
    return PISTADB_ENOTFOUND;
}

int ivf_update(IVFIndex *idx, uint64_t id, const float *vec) {
    for (int i = 0; i < idx->n_vecs; i++) {
        if (idx->vec_ids[i] == id && !idx->vec_deleted[i]) {
            memcpy(idx->vectors + (size_t)i * idx->dim, vec, sizeof(float) * (size_t)idx->dim);
            return PISTADB_OK;
        }
    }
    return PISTADB_ENOTFOUND;
}

/* ── Search ──────────────────────────────────────────────────────────────── */

int ivf_search(const IVFIndex *idx, const float *query, int k,
               PistaDBResult *results) {
    if (!idx->trained) return 0;

    /* Find nprobe nearest centroids */
    float *c_dists = (float *)malloc(sizeof(float) * (size_t)idx->nlist);
    int   *c_order = (int   *)malloc(sizeof(int)   * (size_t)idx->nlist);
    if (!c_dists || !c_order) { free(c_dists); free(c_order); return 0; }

    for (int c = 0; c < idx->nlist; c++) {
        c_dists[c] = dist_l2sq(query, idx->centroids + (size_t)c * idx->dim, idx->dim);
        c_order[c] = c;
    }
    /* Partial sort: find nprobe smallest */
    for (int i = 0; i < idx->nprobe && i < idx->nlist; i++) {
        int best = i;
        for (int j = i + 1; j < idx->nlist; j++)
            if (c_dists[c_order[j]] < c_dists[c_order[best]]) best = j;
        int tmp = c_order[i]; c_order[i] = c_order[best]; c_order[best] = tmp;
    }

    /* Scan selected centroids and collect k-best */
    /* Use a max-heap of size k */
    Heap heap; heap_init(&heap, k + 4, 1);
    char labels_buf[256];

    for (int pi = 0; pi < idx->nprobe && pi < idx->nlist; pi++) {
        int c = c_order[pi];
        for (int j = 0; j < idx->list_sizes[c]; j++) {
            IVFPosting p = idx->lists[c][j];
            if (idx->vec_deleted[p.slot]) continue;
            float d = idx->dist_fn(query,
                                   idx->vectors + (size_t)p.slot * idx->dim,
                                   idx->dim);
            if (heap.size < k) {
                heap_push(&heap, d, p.id);
            } else if (d < heap_top(&heap).key) {
                heap_pop(&heap);
                heap_push(&heap, d, p.id);
            }
        }
    }
    free(c_dists); free(c_order);

    int cnt = heap.size;
    /* Drain heap into results (ascending order) */
    float *ds = (float    *)malloc(sizeof(float)    * (size_t)cnt);
    uint64_t *is = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)cnt);
    for (int i = cnt - 1; i >= 0; i--) {
        HeapItem it = heap_pop(&heap);
        ds[i] = it.key; is[i] = it.id;
    }
    for (int i = 0; i < cnt; i++) {
        results[i].id       = is[i];
        results[i].distance = ds[i];
        /* Fetch label */
        results[i].label[0] = '\0';
        for (int j = 0; j < idx->n_vecs; j++) {
            if (idx->vec_ids[j] == is[i] && !idx->vec_deleted[j]) {
                strncpy(results[i].label, idx->vec_labels[j], 255);
                results[i].label[255] = '\0';
                break;
            }
        }
    }
    free(ds); free(is);
    heap_free(&heap);
    (void)labels_buf;
    return cnt;
}

/* ── Serialization ───────────────────────────────────────────────────────── */
/*
 * int32  nlist, dim, nprobe, trained, n_vecs
 * float  centroids[nlist × dim]
 * For each vector slot:
 *   uint64 id, uint8 deleted, char label[256], float vec[dim]
 * For each list c:
 *   int32 list_size
 *   For each posting: uint64 id, int32 slot
 */

int ivf_save(const IVFIndex *idx, void **out_buf, size_t *out_size) {
    size_t sz = sizeof(int32_t) * 5
              + sizeof(float) * (size_t)(idx->nlist * idx->dim)
              + (size_t)idx->n_vecs * (sizeof(uint64_t) + 1 + 256 + sizeof(float) * (size_t)idx->dim)
              + (size_t)idx->nlist  * sizeof(int32_t);
    for (int c = 0; c < idx->nlist; c++)
        sz += (size_t)idx->list_sizes[c] * (sizeof(uint64_t) + sizeof(int32_t));

    uint8_t *buf = (uint8_t *)malloc(sz);
    if (!buf) return PISTADB_ENOMEM;
    uint8_t *p = buf;

#define WI32(v) do { *(int32_t*)p=(int32_t)(v); p+=4; } while(0)
#define WU64(v) do { *(uint64_t*)p=(uint64_t)(v); p+=8; } while(0)
    WI32(idx->nlist); WI32(idx->dim); WI32(idx->nprobe);
    WI32(idx->trained); WI32(idx->n_vecs);
    memcpy(p, idx->centroids, sizeof(float) * (size_t)(idx->nlist * idx->dim));
    p += sizeof(float) * (size_t)(idx->nlist * idx->dim);
    for (int i = 0; i < idx->n_vecs; i++) {
        WU64(idx->vec_ids[i]);
        *p++ = idx->vec_deleted[i];
        memcpy(p, idx->vec_labels[i], 256); p += 256;
        memcpy(p, idx->vectors + (size_t)i * idx->dim, sizeof(float) * (size_t)idx->dim);
        p += sizeof(float) * (size_t)idx->dim;
    }
    for (int c = 0; c < idx->nlist; c++) {
        WI32(idx->list_sizes[c]);
        for (int j = 0; j < idx->list_sizes[c]; j++) {
            WU64(idx->lists[c][j].id);
            WI32(idx->lists[c][j].slot);
        }
    }
#undef WI32
#undef WU64
    *out_buf  = buf;
    *out_size = (size_t)(p - buf);
    return PISTADB_OK;
}

int ivf_load(IVFIndex *idx, const void *buf, size_t size, int dim, DistFn dist_fn) {
    const uint8_t *p = (const uint8_t *)buf;
    const uint8_t *end = p + size;

#define RI32() (*(const int32_t*)p); p+=4
#define RU64() (*(const uint64_t*)p); p+=8

    int nlist    = RI32();
    int file_dim     = RI32();
    int nprobe   = RI32();
    int trained  = RI32();
    int n_vecs   = RI32();
    if (file_dim != dim) return PISTADB_ECORRUPT;
    (void)end;

    int r = ivf_create(idx, dim, dist_fn, nlist, nprobe);
    if (r != PISTADB_OK) return r;
    idx->trained = trained;

    memcpy(idx->centroids, p, sizeof(float) * (size_t)(nlist * dim));
    p += sizeof(float) * (size_t)(nlist * dim);

    for (int i = 0; i < n_vecs; i++) {
        uint64_t id  = RU64();
        uint8_t  del = *p++;
        const char *lbl = (const char *)p; p += 256;
        const float *vec = (const float *)p; p += sizeof(float) * (size_t)dim;
        r = ivf_insert(idx, id, lbl, vec);
        if (r != PISTADB_OK) return r;
        idx->vec_deleted[idx->n_vecs - 1] = del;
    }
    /* Now overwrite posting lists (ivf_insert already built them, but re-read) */
    /* Reset lists */
    for (int c = 0; c < nlist; c++) { free(idx->lists[c]); idx->lists[c] = NULL; idx->list_sizes[c] = idx->list_caps[c] = 0; }
    for (int c = 0; c < nlist; c++) {
        int ls = RI32();
        idx->lists[c]      = (IVFPosting *)malloc(sizeof(IVFPosting) * (size_t)(ls + 1));
        idx->list_caps[c]  = ls + 1;
        idx->list_sizes[c] = ls;
        for (int j = 0; j < ls; j++) {
            idx->lists[c][j].id   = RU64();
            idx->lists[c][j].slot = RI32();
        }
    }
#undef RI32
#undef RU64
    return PISTADB_OK;
}
