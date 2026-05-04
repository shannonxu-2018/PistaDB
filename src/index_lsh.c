/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - index_lsh.c
 * LSH index implementation.
 */
#include "index_lsh.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ── Hash computation ────────────────────────────────────────────────────── */

/* Sign-based: hash bit i = sign(proj[i] · vec)  → 0 or 1 */
/* E2LSH:      hash bit i = floor((proj[i]·vec + bias[i]) / w) mod 2 */

static uint32_t lsh_hash_vec(const LSHTable *t, const float *vec, int dim) {
    /* FNV-1a hash over the K integer bucket indices.
     * For E2LSH: bucket_idx = floor((dot + bias) / w)  — distance-sensitive.
     * For sign-based (cosine/IP): bucket_idx = sign of dot — 0 or 1.
     * Using the full integer index (not just parity) preserves the
     * distance-sensitive collision probability p(d)^K per table. */
    uint32_t h = 2166136261u;  /* FNV-1a offset basis */
    for (int k = 0; k < t->K; k++) {
        const float *row = t->proj + (size_t)k * dim;
        float dot = 0.0f;
        for (int d = 0; d < dim; d++) dot += row[d] * vec[d];
        int32_t bucket_idx;
        if (t->is_e2lsh) {
            bucket_idx = (int32_t)floorf((dot + t->bias[k]) / t->w);
        } else {
            bucket_idx = (dot >= 0.0f) ? 1 : 0;
        }
        h ^= (uint32_t)bucket_idx;
        h *= 16777619u;  /* FNV prime */
    }
    return h % (uint32_t)t->num_buckets;
}

/* ── Table helpers ───────────────────────────────────────────────────────── */

static int table_init(LSHTable *t, int K, int dim, float w, PistaDBMetric metric,
                      PCG *rng) {
    t->K          = K;
    t->w          = w;
    t->is_e2lsh   = (metric == METRIC_L2 || metric == METRIC_L1) ? 1 : 0;
    t->num_buckets= 1 << (K > 16 ? 16 : K);  /* cap at 64K buckets */

    t->proj    = (float *)malloc(sizeof(float) * (size_t)(K * dim));
    t->bias    = (float *)malloc(sizeof(float) * (size_t)K);
    t->buckets = (LSHBucket *)calloc((size_t)t->num_buckets, sizeof(LSHBucket));
    if (!t->proj || !t->bias || !t->buckets) {
        /* Release whatever did succeed and reset bookkeeping so a later
         * table_free traverses a sane (zero-bucket) table without crashing. */
        free(t->proj); free(t->bias); free(t->buckets);
        t->proj = NULL; t->bias = NULL; t->buckets = NULL;
        t->num_buckets = 0;
        return PISTADB_ENOMEM;
    }

    /* Fill projection matrix with Gaussian random values */
    for (int i = 0; i < K * dim; i++) t->proj[i] = pcg_normal(rng);
    /* Normalise each row for sign-based hashing */
    if (!t->is_e2lsh) {
        for (int k = 0; k < K; k++) {
            float norm = 0.0f;
            float *row = t->proj + (size_t)k * dim;
            for (int d = 0; d < dim; d++) norm += row[d] * row[d];
            norm = sqrtf(norm) + 1e-12f;
            for (int d = 0; d < dim; d++) row[d] /= norm;
        }
    }
    for (int k = 0; k < K; k++) t->bias[k] = pcg_f32(rng) * w;
    return PISTADB_OK;
}

static void table_free(LSHTable *t) {
    free(t->proj); free(t->bias);
    if (t->buckets) {
        for (int b = 0; b < t->num_buckets; b++) free(t->buckets[b].slots);
        free(t->buckets);
    }
    t->proj = NULL; t->bias = NULL; t->buckets = NULL;
    t->num_buckets = 0;
}

/* Store internal slot index (not external id) — O(1) lookup during search. */
static int table_insert(LSHTable *t, const float *vec, int dim, int slot) {
    uint32_t h = lsh_hash_vec(t, vec, dim);
    LSHBucket *bkt = &t->buckets[h];
    if (bkt->size == bkt->cap) {
        int nc = bkt->cap * 2 + 4;
        int *nd = (int *)realloc(bkt->slots, sizeof(int) * (size_t)nc);
        if (!nd) return PISTADB_ENOMEM;
        bkt->slots = nd; bkt->cap = nc;
    }
    bkt->slots[bkt->size++] = slot;
    return PISTADB_OK;
}

/* ── Index lifecycle ─────────────────────────────────────────────────────── */

int lsh_create(LSHIndex *idx, int dim, DistFn dist_fn, PistaDBMetric metric,
               int L, int K, float w) {
    memset(idx, 0, sizeof(*idx));
    idx->dim      = dim;
    idx->dist_fn  = dist_fn;
    idx->metric   = metric;
    idx->L        = L;
    idx->K        = K;
    idx->w        = w;
    idx->vec_cap  = 64;

    idx->tables    = (LSHTable *)calloc((size_t)L, sizeof(LSHTable));
    idx->vec_ids   = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)idx->vec_cap);
    idx->vec_deleted=(uint8_t  *)calloc((size_t)idx->vec_cap, 1);
    if (!idx->tables || !idx->vec_ids || !idx->vec_deleted)
        return PISTADB_ENOMEM;
    if (vs_init(&idx->vs, dim, idx->vec_cap) != PISTADB_OK) return PISTADB_ENOMEM;

    PCG rng; pcg_seed(&rng, 42);
    for (int l = 0; l < L; l++) {
        int r = table_init(&idx->tables[l], K, dim, w, metric, &rng);
        if (r != PISTADB_OK) return r;
    }
    return PISTADB_OK;
}

void lsh_free(LSHIndex *idx) {
    for (int l = 0; l < idx->L; l++) table_free(&idx->tables[l]);
    free(idx->tables);
    vs_free(&idx->vs);
    free(idx->vec_ids); free(idx->vec_deleted);
    memset(idx, 0, sizeof(*idx));
}

/* ── Insert / Delete / Update ────────────────────────────────────────────── */

static int vec_store_grow_lsh(LSHIndex *idx) {
    int nc = idx->vec_cap * 2 + 8;
    int r = vs_ensure(&idx->vs, nc);
    if (r != PISTADB_OK) return r;
    uint64_t *ni = (uint64_t *)realloc(idx->vec_ids, sizeof(uint64_t) * (size_t)nc);
    if (!ni) return PISTADB_ENOMEM;
    idx->vec_ids = ni;
    uint8_t  *nd = (uint8_t  *)realloc(idx->vec_deleted, (size_t)nc);
    if (!nd) return PISTADB_ENOMEM;
    idx->vec_deleted = nd;
    memset(nd + idx->vec_cap, 0, (size_t)(nc - idx->vec_cap));
    idx->vec_cap = nc;
    return PISTADB_OK;
}

int lsh_insert(LSHIndex *idx, uint64_t id, const char *label, const float *vec) {
    if (idx->n_vecs == idx->vec_cap) {
        int r = vec_store_grow_lsh(idx); if (r) return r;
    }
    int slot = idx->n_vecs++;
    idx->vec_ids[slot] = id;
    idx->vec_deleted[slot] = 0;
    memcpy(VS_VEC(&idx->vs, slot), vec, sizeof(float) * (size_t)idx->dim);
    if (label) strncpy(VS_LABEL(&idx->vs, slot), label, 255);
    else        VS_LABEL(&idx->vs, slot)[0] = '\0';
    VS_LABEL(&idx->vs, slot)[255] = '\0';

    for (int l = 0; l < idx->L; l++) {
        int r = table_insert(&idx->tables[l], vec, idx->dim, slot);
        if (r != PISTADB_OK) return r;
    }
    return PISTADB_OK;
}

int lsh_delete(LSHIndex *idx, uint64_t id) {
    for (int i = 0; i < idx->n_vecs; i++) {
        if (idx->vec_ids[i] == id) { idx->vec_deleted[i] = 1; return PISTADB_OK; }
    }
    return PISTADB_ENOTFOUND;
}

int lsh_update(LSHIndex *idx, uint64_t id, const float *vec) {
    for (int i = 0; i < idx->n_vecs; i++) {
        if (idx->vec_ids[i] == id && !idx->vec_deleted[i]) {
            memcpy(VS_VEC(&idx->vs, i), vec, sizeof(float) * (size_t)idx->dim);
            /* Re-hash with slot index — may add duplicate but OK */
            for (int l = 0; l < idx->L; l++)
                table_insert(&idx->tables[l], vec, idx->dim, i);
            return PISTADB_OK;
        }
    }
    return PISTADB_ENOTFOUND;
}

/* ── Search ──────────────────────────────────────────────────────────────── */

int lsh_search(const LSHIndex *idx, const float *query, int k,
               PistaDBResult *results) {
    if (idx->n_vecs == 0) return 0;

    /* Bitset over slot indices for O(1) deduplication — no O(n²) linear scan. */
    uint8_t *visited = (uint8_t *)calloc((size_t)idx->n_vecs, 1);
    if (!visited) return 0;

    int max_cands = idx->L * (1 << (idx->K > 8 ? 8 : idx->K));
    if (max_cands < 256) max_cands = 256;
    if (max_cands > idx->n_vecs) max_cands = idx->n_vecs;

    int *cand_slots = (int *)malloc(sizeof(int) * (size_t)max_cands);
    int  n_cands = 0;
    if (!cand_slots) { free(visited); return 0; }

    /* Gather candidate slots — O(1) dedup via visited[]. */
    for (int l = 0; l < idx->L; l++) {
        uint32_t h = lsh_hash_vec(&idx->tables[l], query, idx->dim);
        const LSHBucket *bkt = &idx->tables[l].buckets[h];
        for (int j = 0; j < bkt->size && n_cands < max_cands; j++) {
            int s = bkt->slots[j];
            if (visited[s]) continue;
            visited[s] = 1;
            cand_slots[n_cands++] = s;
        }
    }
    free(visited);

    /* Exact re-ranking — O(1) per candidate (direct slot access, no id lookup). */
    Heap heap; heap_init(&heap, k + 4, 1);
    for (int ci = 0; ci < n_cands; ci++) {
        int s = cand_slots[ci];
        if (idx->vec_deleted[s]) continue;
        float d = idx->dist_fn(query, VS_VEC(&idx->vs, s), idx->dim);
        if (heap.size < k) heap_push(&heap, d, (uint64_t)s);
        else if (d < heap_top(&heap).key) { heap_pop(&heap); heap_push(&heap, d, (uint64_t)s); }
    }
    free(cand_slots);

    /* Drain max-heap in ascending order, fill results directly from slot. */
    int total = heap.size;
    HeapItem *items = (HeapItem *)malloc(sizeof(HeapItem) * (size_t)(total + 1));
    if (!items) { heap_free(&heap); return 0; }
    while (heap.size > 0) {
        HeapItem it = heap_pop(&heap);
        items[heap.size] = it;   /* after pop, heap.size is one less → ascending fill */
    }
    for (int i = 0; i < total; i++) {
        int s = (int)items[i].id;
        results[i].id       = idx->vec_ids[s];
        results[i].distance = items[i].key;
        strncpy(results[i].label, VS_LABEL(&idx->vs, s), 255);
        results[i].label[255] = '\0';
    }
    free(items);
    heap_free(&heap);
    return total;
}

/* ── Serialization ───────────────────────────────────────────────────────── */
/*
 * Header: int32 L, K, dim, n_vecs, metric; float w
 * Vector store: same as LinearIndex
 * For each table:
 *   float proj[K×dim], bias[K]
 *   int32 num_buckets
 *   For each bucket: int32 size, uint64 ids[size]
 */

int lsh_save(const LSHIndex *idx, void **out_buf, size_t *out_size) {
    size_t hdr  = sizeof(int32_t) * 5 + sizeof(float);
    size_t vstk = (size_t)idx->n_vecs * (sizeof(uint64_t) + 1 + 256 + sizeof(float) * (size_t)idx->dim);
    size_t tbls = 0;
    for (int l = 0; l < idx->L; l++) {
        const LSHTable *t = &idx->tables[l];
        tbls += sizeof(float) * (size_t)(t->K * idx->dim) + sizeof(float) * (size_t)t->K;
        tbls += sizeof(int32_t);
        for (int b = 0; b < t->num_buckets; b++)
            tbls += sizeof(int32_t) + sizeof(int32_t) * (size_t)t->buckets[b].size;
    }
    size_t total = hdr + sizeof(int32_t) + vstk + tbls;

    uint8_t *buf = (uint8_t *)malloc(total);
    if (!buf) return PISTADB_ENOMEM;
    uint8_t *p = buf;

#define WI32(v) do{*(int32_t*)p=(int32_t)(v);p+=4;}while(0)
#define WU64(v) do{*(uint64_t*)p=(uint64_t)(v);p+=8;}while(0)
#define WF32(v) do{*(float*)p=(float)(v);p+=4;}while(0)
    WI32(idx->L); WI32(idx->K); WI32(idx->dim); WI32(idx->n_vecs); WI32((int)idx->metric);
    WF32(idx->w);
    /* vector store */
    WI32(idx->n_vecs);
    for (int i = 0; i < idx->n_vecs; i++) {
        WU64(idx->vec_ids[i]);
        *p++ = idx->vec_deleted[i];
        memcpy(p, VS_LABEL(&idx->vs, i), 256); p += 256;
        memcpy(p, VS_VEC(&idx->vs, i), sizeof(float) * (size_t)idx->dim);
        p += sizeof(float) * (size_t)idx->dim;
    }
    /* tables */
    for (int l = 0; l < idx->L; l++) {
        const LSHTable *t = &idx->tables[l];
        memcpy(p, t->proj, sizeof(float) * (size_t)(t->K * idx->dim)); p += sizeof(float) * (size_t)(t->K * idx->dim);
        memcpy(p, t->bias, sizeof(float) * (size_t)t->K);              p += sizeof(float) * (size_t)t->K;
        WI32(t->num_buckets);
        for (int b = 0; b < t->num_buckets; b++) {
            WI32(t->buckets[b].size);
            for (int j = 0; j < t->buckets[b].size; j++) WI32(t->buckets[b].slots[j]);
        }
    }
#undef WI32
#undef WU64
#undef WF32
    *out_buf  = buf;
    *out_size = (size_t)(p - buf);
    return PISTADB_OK;
}

int lsh_load(LSHIndex *idx, const void *buf, size_t size,
             int dim, DistFn dist_fn, PistaDBMetric metric) {
    const uint8_t *p = (const uint8_t *)buf;
    const uint8_t *end = p + size;
#define RI32() (*(const int32_t*)p); p+=4
#define RU64() (*(const uint64_t*)p); p+=8
#define RF32() (*(const float*)p); p+=4
    if (size < (size_t)(sizeof(int32_t) * 5 + sizeof(float))) return PISTADB_ECORRUPT;
    int L = RI32(); int K = RI32(); int file_dim = RI32(); int n_vecs = RI32();
    int met = RI32(); float w = RF32();
    (void)met;
    if (file_dim != dim) return PISTADB_ECORRUPT;
    if (L <= 0 || L > 100000 || K <= 0 || K > 64 || n_vecs < 0)
        return PISTADB_ECORRUPT;

    int r = lsh_create(idx, dim, dist_fn, metric, L, K, w);
    if (r) return r;

    /* vector store */
    if ((size_t)(end - p) < 4) return PISTADB_ECORRUPT;
    int nvs = RI32(); (void)nvs;
    /* Per-entry bytes: id(8) + deleted(1) + label(256) + vec(dim*4). */
    const size_t vec_entry_sz = 8 + 1 + 256 + sizeof(float) * (size_t)dim;
    if ((size_t)n_vecs > (size_t)(end - p) / (vec_entry_sz ? vec_entry_sz : 1))
        return PISTADB_ECORRUPT;
    for (int i = 0; i < n_vecs; i++) {
        uint64_t id = RU64(); uint8_t del = *p++; const char *lbl = (const char *)p; p += 256;
        const float *vec = (const float *)p; p += sizeof(float) * (size_t)dim;
        r = lsh_insert(idx, id, lbl, vec);
        if (r) return r;
        idx->vec_deleted[idx->n_vecs - 1] = del;
    }
    /* re-load table projections and buckets — verify each section fits in
     * the remaining buffer before trusting it. */
    const size_t proj_bytes = sizeof(float) * (size_t)K * (size_t)dim;
    const size_t bias_bytes = sizeof(float) * (size_t)K;
    for (int l = 0; l < L; l++) {
        LSHTable *t = &idx->tables[l];
        if ((size_t)(end - p) < proj_bytes + bias_bytes + 4) return PISTADB_ECORRUPT;
        memcpy(t->proj, p, proj_bytes); p += proj_bytes;
        memcpy(t->bias, p, bias_bytes); p += bias_bytes;
        int nb = RI32();
        /* num_buckets must be positive and bounded — table_init originally
         * caps at 1<<16 = 65536, leave generous headroom for older files. */
        if (nb <= 0 || nb > (1 << 24)) return PISTADB_ECORRUPT;
        /* Re-allocate buckets if needed */
        if (nb != t->num_buckets) {
            for (int b = 0; b < t->num_buckets; b++) free(t->buckets[b].slots);
            free(t->buckets);
            t->buckets = (LSHBucket *)calloc((size_t)nb, sizeof(LSHBucket));
            if (!t->buckets) { t->num_buckets = 0; return PISTADB_ENOMEM; }
            t->num_buckets = nb;
        } else {
            for (int b = 0; b < nb; b++) { free(t->buckets[b].slots); t->buckets[b].slots=NULL; t->buckets[b].size=t->buckets[b].cap=0; }
        }
        for (int b = 0; b < nb; b++) {
            if ((size_t)(end - p) < 4) return PISTADB_ECORRUPT;
            int sz = RI32();
            if (sz < 0) return PISTADB_ECORRUPT;
            if ((size_t)(end - p) < (size_t)sz * 4) return PISTADB_ECORRUPT;
            if (sz > 0) {
                t->buckets[b].slots = (int *)malloc(sizeof(int) * (size_t)sz);
                if (!t->buckets[b].slots) return PISTADB_ENOMEM;
                t->buckets[b].size = t->buckets[b].cap = sz;
                for (int j = 0; j < sz; j++) { t->buckets[b].slots[j] = *(const int32_t*)p; p += 4; }
            }
        }
    }
#undef RI32
#undef RU64
#undef RF32
    return PISTADB_OK;
}
