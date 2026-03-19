/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - index_hnsw.c
 * HNSW graph index implementation.
 */
#include "index_hnsw.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ── PCG RNG (module-level) ─────────────────────────────────────────────── */
static PCG g_rng;
static int g_rng_init = 0;

static void ensure_rng(void) {
    if (!g_rng_init) { pcg_seed(&g_rng, 42); g_rng_init = 1; }
}

/* ── Random layer assignment ─────────────────────────────────────────────── */
static int random_level(float mL) {
    ensure_rng();
    float r = pcg_f32(&g_rng);
    if (r < 1e-9f) r = 1e-9f;
    int level = (int)(-logf(r) * mL);
    if (level >= HNSW_MAX_LAYERS) level = HNSW_MAX_LAYERS - 1;
    return level;
}

/* ── Node helpers ────────────────────────────────────────────────────────── */
static int node_init(HNSWNode *n, uint64_t id, int level, int M, int M_max0) {
    n->vec_id       = id;
    n->level        = level;
    n->neighbors    = (int **)calloc((size_t)(level + 1), sizeof(int *));
    n->neighbor_cnt = (int  *)calloc((size_t)(level + 1), sizeof(int));
    n->neighbor_cap = (int  *)calloc((size_t)(level + 1), sizeof(int));
    if (!n->neighbors || !n->neighbor_cnt || !n->neighbor_cap) return PISTADB_ENOMEM;

    for (int i = 0; i <= level; i++) {
        int cap = (i == 0) ? M_max0 * 2 : M * 2;
        n->neighbors[i] = (int *)malloc(sizeof(int) * (size_t)cap);
        if (!n->neighbors[i]) return PISTADB_ENOMEM;
        n->neighbor_cap[i] = cap;
        n->neighbor_cnt[i] = 0;
    }
    return PISTADB_OK;
}

static void node_free(HNSWNode *n) {
    if (!n->neighbors) return;
    for (int i = 0; i <= n->level; i++) free(n->neighbors[i]);
    free(n->neighbors);
    free(n->neighbor_cnt);
    free(n->neighbor_cap);
    n->neighbors = NULL;
}

/* Add neighbor to node at given layer (no duplicate check for speed). */
static int node_add_neighbor(HNSWNode *n, int layer, int nb_node) {
    if (n->neighbor_cnt[layer] >= n->neighbor_cap[layer]) {
        int nc = n->neighbor_cap[layer] * 2 + 4;
        int *nd = (int *)realloc(n->neighbors[layer], sizeof(int) * (size_t)nc);
        if (!nd) return PISTADB_ENOMEM;
        n->neighbors[layer] = nd;
        n->neighbor_cap[layer] = nc;
    }
    n->neighbors[layer][n->neighbor_cnt[layer]++] = nb_node;
    return PISTADB_OK;
}

/* ── Index lifecycle ─────────────────────────────────────────────────────── */

int hnsw_create(HNSWIndex *idx, int dim, DistFn dist_fn,
                int M, int ef_construction, int ef_search) {
    if (M < 2) M = 2;
    idx->dim            = dim;
    idx->dist_fn        = dist_fn;
    idx->M              = M;
    idx->M_max0         = M * 2;
    idx->ef_construction= ef_construction;
    idx->ef_search      = ef_search;
    idx->mL             = 1.0f / logf((float)M);
    idx->ep_node        = -1;
    idx->max_layer      = -1;
    idx->n_nodes        = 0;
    idx->node_cap       = 64;

    idx->nodes   = (HNSWNode *)calloc((size_t)idx->node_cap, sizeof(HNSWNode));
    if (!idx->nodes) return PISTADB_ENOMEM;
    if (vs_init(&idx->vs, dim, idx->node_cap) != PISTADB_OK) return PISTADB_ENOMEM;
    return PISTADB_OK;
}

void hnsw_free(HNSWIndex *idx) {
    for (int i = 0; i < idx->n_nodes; i++) node_free(&idx->nodes[i]);
    free(idx->nodes);
    vs_free(&idx->vs);
    idx->nodes = NULL;
    idx->n_nodes = idx->node_cap = 0;
}

static int hnsw_grow(HNSWIndex *idx) {
    int nc = idx->node_cap * 2 + 8;
    HNSWNode *nn = (HNSWNode *)realloc(idx->nodes, sizeof(HNSWNode) * (size_t)nc);
    if (!nn) return PISTADB_ENOMEM;
    idx->nodes = nn;  /* assign early so pointer is not lost on vs_ensure failure */
    int r = vs_ensure(&idx->vs, nc);
    if (r != PISTADB_OK) return r;
    memset(idx->nodes + idx->node_cap, 0, sizeof(HNSWNode) * (size_t)(nc - idx->node_cap));
    idx->node_cap = nc;
    return PISTADB_OK;
}

static inline float node_dist(const HNSWIndex *idx, int a, int b) {
    return idx->dist_fn(VS_VEC(&idx->vs, a), VS_VEC(&idx->vs, b), idx->dim);
}

static inline float query_dist(const HNSWIndex *idx, const float *q, int node) {
    return idx->dist_fn(q, VS_VEC(&idx->vs, node), idx->dim);
}

/* ── SEARCH-LAYER ────────────────────────────────────────────────────────── */
/*
 * W (results) is a max-heap of size ef.
 * C (candidates) is a min-heap.
 * visited is a bitset.
 */
static int search_layer(const HNSWIndex *idx, const float *query,
                         int ep, int ef, int layer,
                         Heap *W, Heap *C, Bitset *visited) {
    heap_clear(W);
    heap_clear(C);
    bitset_clear(visited);

    float d_ep = query_dist(idx, query, ep);
    heap_push(C, d_ep, (uint64_t)ep);   /* min-heap: nearest on top */
    heap_push(W, d_ep, (uint64_t)ep);   /* max-heap: furthest on top */
    bitset_set(visited, ep);

    while (C->size > 0) {
        HeapItem c = heap_pop(C);       /* nearest candidate */
        float w_far = heap_top(W).key;  /* furthest in results */

        if (c.key > w_far) break;       /* all remaining candidates are farther */

        int c_idx = (int)c.id;
        const HNSWNode *cn = &idx->nodes[c_idx];
        if (layer > cn->level) continue;

        for (int j = 0; j < cn->neighbor_cnt[layer]; j++) {
            int nb = cn->neighbors[layer][j];
            if (nb < 0 || nb >= idx->n_nodes) continue;
            if (bitset_test(visited, nb)) continue;
            bitset_set(visited, nb);

            float d_nb = query_dist(idx, query, nb);
            float w_far2 = heap_top(W).key;

            if (d_nb < w_far2 || W->size < ef) {
                heap_push(C, d_nb, (uint64_t)nb);
                heap_push(W, d_nb, (uint64_t)nb);
                if (W->size > ef) heap_pop(W);  /* remove furthest */
            }
        }
    }
    return PISTADB_OK;
}

/* ── SELECT-NEIGHBORS (simple: take M nearest from W) ───────────────────── */
static void select_neighbors(Heap *W, int M, int *out, int *out_cnt) {
    /* W is a max-heap. Collect all, sort, take M nearest. */
    /* Extract all items into out[] */
    *out_cnt = 0;
    /* Temp: collect from max-heap, then reverse-select M nearest */
    /* We'll use a small buffer on stack or heap */
    int total = W->size;
    float *keys = (float    *)malloc(sizeof(float)    * (size_t)total);
    int   *ids  = (int      *)malloc(sizeof(int)      * (size_t)total);
    if (!keys || !ids) { free(keys); free(ids); return; }

    /* Drain W into arrays */
    int n = 0;
    while (W->size > 0) {
        HeapItem it = heap_pop(W);
        keys[n] = it.key;
        ids[n]  = (int)it.id;
        n++;
    }
    /* Sort ascending (insertion sort – small arrays) */
    for (int i = 1; i < n; i++) {
        float ki = keys[i]; int ii = ids[i];
        int j = i - 1;
        while (j >= 0 && keys[j] > ki) { keys[j+1] = keys[j]; ids[j+1] = ids[j]; j--; }
        keys[j+1] = ki; ids[j+1] = ii;
    }
    /* Take M nearest */
    *out_cnt = (n < M) ? n : M;
    for (int i = 0; i < *out_cnt; i++) out[i] = ids[i];
    free(keys); free(ids);
}

/* ── INSERT ──────────────────────────────────────────────────────────────── */

int hnsw_insert(HNSWIndex *idx, uint64_t id, const char *label, const float *vec) {
    if (idx->n_nodes == idx->node_cap) {
        int r = hnsw_grow(idx);
        if (r != PISTADB_OK) return r;
    }

    int new_node = idx->n_nodes;
    int level    = random_level(idx->mL);

    int r = node_init(&idx->nodes[new_node], id, level, idx->M, idx->M_max0);
    if (r != PISTADB_OK) return r;
    memcpy(VS_VEC(&idx->vs, new_node), vec, sizeof(float) * (size_t)idx->dim);
    if (label) { strncpy(VS_LABEL(&idx->vs, new_node), label, 255); VS_LABEL(&idx->vs, new_node)[255] = '\0'; }
    else        VS_LABEL(&idx->vs, new_node)[0] = '\0';
    idx->n_nodes++;

    if (idx->ep_node == -1) {
        /* First node */
        idx->ep_node  = new_node;
        idx->max_layer = level;
        return PISTADB_OK;
    }

    /* Work buffers */
    int max_ef = (idx->ef_construction > idx->M_max0 * 4) ?
                  idx->ef_construction : idx->M_max0 * 4;
    Heap   W, C;
    Bitset visited;
    heap_init(&W, max_ef + 8, 1);  /* max-heap */
    heap_init(&C, max_ef + 8, 0);  /* min-heap */
    bitset_init(&visited, idx->node_cap + 8);

    int ep = idx->ep_node;
    int L  = idx->max_layer;

    /* Phase 1: traverse layers above new node's level (greedy, ef=1) */
    for (int lc = L; lc > level; lc--) {
        search_layer(idx, vec, ep, 1, lc, &W, &C, &visited);
        if (W.size > 0) ep = (int)heap_pop(&W).id;
        heap_clear(&W); heap_clear(&C);
    }

    /* Phase 2: from min(L, level) down to 0 – build connections */
    int tmp_nb[HNSW_MAX_LAYERS * 64];  /* generous scratch space */
    for (int lc = (L < level ? L : level); lc >= 0; lc--) {
        search_layer(idx, vec, ep, idx->ef_construction, lc, &W, &C, &visited);

        int M_lc = (lc == 0) ? idx->M_max0 : idx->M;
        int nb_cnt = 0;
        select_neighbors(&W, M_lc, tmp_nb, &nb_cnt);

        /* Connect new_node → neighbors */
        for (int i = 0; i < nb_cnt; i++) {
            node_add_neighbor(&idx->nodes[new_node], lc, tmp_nb[i]);
        }

        /* Connect neighbors → new_node (bidirectional), pruning if needed */
        for (int i = 0; i < nb_cnt; i++) {
            int nb = tmp_nb[i];
            HNSWNode *nbn = &idx->nodes[nb];
            if (lc > nbn->level) continue;
            node_add_neighbor(nbn, lc, new_node);

            /* Prune if over capacity */
            int max_conn = (lc == 0) ? idx->M_max0 : idx->M;
            if (nbn->neighbor_cnt[lc] > max_conn) {
                /* Simple pruning: rebuild neighbor list keeping M nearest */
                Heap tmp_h;
                heap_init(&tmp_h, nbn->neighbor_cnt[lc] + 2, 1);
                for (int j = 0; j < nbn->neighbor_cnt[lc]; j++) {
                    int nb2 = nbn->neighbors[lc][j];
                    float d = node_dist(idx, nb, nb2);
                    heap_push(&tmp_h, d, (uint64_t)nb2);
                }
                int pruned_cnt = 0;
                select_neighbors(&tmp_h, max_conn, nbn->neighbors[lc], &pruned_cnt);
                nbn->neighbor_cnt[lc] = pruned_cnt;
                heap_free(&tmp_h);
            }
        }

        /* ep for next layer = nearest from W (already popped) */
        if (nb_cnt > 0) ep = tmp_nb[0];
        heap_clear(&W); heap_clear(&C);
    }

    if (level > idx->max_layer) {
        idx->max_layer = level;
        idx->ep_node   = new_node;
    }

    heap_free(&W);
    heap_free(&C);
    bitset_free(&visited);
    return PISTADB_OK;
}

/* ── DELETE (lazy) ───────────────────────────────────────────────────────── */
/* We mark by setting vec_id to UINT64_MAX; skip in search */
int hnsw_delete(HNSWIndex *idx, uint64_t id) {
    for (int i = 0; i < idx->n_nodes; i++) {
        if (idx->nodes[i].vec_id == id) {
            idx->nodes[i].vec_id = UINT64_MAX;
            return PISTADB_OK;
        }
    }
    return PISTADB_ENOTFOUND;
}

int hnsw_update(HNSWIndex *idx, uint64_t id, const float *vec) {
    for (int i = 0; i < idx->n_nodes; i++) {
        if (idx->nodes[i].vec_id == id) {
            memcpy(VS_VEC(&idx->vs, i), vec, sizeof(float) * (size_t)idx->dim);
            return PISTADB_OK;
        }
    }
    return PISTADB_ENOTFOUND;
}

/* ── KNN SEARCH ──────────────────────────────────────────────────────────── */

int hnsw_search(HNSWIndex *idx, const float *query, int k, int ef,
                PistaDBResult *results) {
    if (idx->ep_node < 0) return 0;
    if (ef < k) ef = k;

    int max_ef = ef + 8;
    Heap   W, C;
    Bitset visited;
    heap_init(&W, max_ef + 8, 1);
    heap_init(&C, max_ef + 8, 0);
    bitset_init(&visited, idx->n_nodes + 8);

    int ep = idx->ep_node;
    int L  = idx->max_layer;

    for (int lc = L; lc > 0; lc--) {
        search_layer(idx, query, ep, 1, lc, &W, &C, &visited);
        if (W.size > 0) ep = (int)heap_top(&W).id;
        heap_clear(&W); heap_clear(&C); bitset_clear(&visited);
    }
    search_layer(idx, query, ep, ef, 0, &W, &C, &visited);

    /* Collect results (W is max-heap; drain to get ascending order) */
    int    cnt = 0;
    float *dist_buf = (float    *)malloc(sizeof(float)    * (size_t)W.size);
    int   *node_buf = (int      *)malloc(sizeof(int)      * (size_t)W.size);
    int    total    = W.size;
    while (W.size > 0) {
        HeapItem it = heap_pop(&W);
        dist_buf[W.size] = it.key;
        node_buf[W.size] = (int)it.id;
    }
    /* dist_buf / node_buf are now in ascending distance order (0..total-1) */
    for (int i = 0; i < total && cnt < k; i++) {
        int ni = node_buf[i];
        if (idx->nodes[ni].vec_id == UINT64_MAX) continue;  /* deleted */
        results[cnt].id       = idx->nodes[ni].vec_id;
        results[cnt].distance = dist_buf[i];
        strncpy(results[cnt].label, VS_LABEL(&idx->vs, ni), 255);
        results[cnt].label[255] = '\0';
        cnt++;
    }
    free(dist_buf); free(node_buf);
    heap_free(&W); heap_free(&C);
    bitset_free(&visited);
    return cnt;
}

/* ── Serialization ───────────────────────────────────────────────────────── */
/*
 * Header:
 *   int32 n_nodes, dim, M, M_max0, ef_construction, ef_search
 *   int32 ep_node, max_layer
 * For each node:
 *   uint64 vec_id
 *   int32  level
 *   float  vec[dim]
 *   For each layer 0..level:
 *     int32 neighbor_cnt
 *     int32 neighbors[neighbor_cnt]
 */

int hnsw_save(const HNSWIndex *idx, void **out_buf, size_t *out_size) {
    /* First pass: compute size */
    size_t sz = sizeof(int32_t) * 8;  /* header fields */
    for (int i = 0; i < idx->n_nodes; i++) {
        sz += sizeof(uint64_t) + sizeof(int32_t) + 256;  /* vec_id, level, label */
        sz += sizeof(float) * (size_t)idx->dim;
        for (int l = 0; l <= idx->nodes[i].level; l++)
            sz += sizeof(int32_t) * (size_t)(1 + idx->nodes[i].neighbor_cnt[l]);
    }

    uint8_t *buf = (uint8_t *)malloc(sz);
    if (!buf) return PISTADB_ENOMEM;
    uint8_t *p = buf;

#define WI32(v) do { *(int32_t*)p = (int32_t)(v); p += 4; } while(0)
#define WU64(v) do { *(uint64_t*)p = (uint64_t)(v); p += 8; } while(0)
#define WF32(v) do { *(float*)p = (float)(v); p += 4; } while(0)

    WI32(idx->n_nodes);
    WI32(idx->dim);
    WI32(idx->M);
    WI32(idx->M_max0);
    WI32(idx->ef_construction);
    WI32(idx->ef_search);
    WI32(idx->ep_node);
    WI32(idx->max_layer);

    for (int i = 0; i < idx->n_nodes; i++) {
        const HNSWNode *n = &idx->nodes[i];
        WU64(n->vec_id);
        WI32(n->level);
        memcpy(p, VS_LABEL(&idx->vs, i), 256); p += 256;
        const float *v = (const float *)VS_VEC(&idx->vs, i);
        for (int d = 0; d < idx->dim; d++) WF32(v[d]);
        for (int l = 0; l <= n->level; l++) {
            WI32(n->neighbor_cnt[l]);
            for (int j = 0; j < n->neighbor_cnt[l]; j++)
                WI32(n->neighbors[l][j]);
        }
    }
#undef WI32
#undef WU64
#undef WF32

    *out_buf  = buf;
    *out_size = (size_t)(p - buf);
    return PISTADB_OK;
}

int hnsw_load(HNSWIndex *idx, const void *buf, size_t size,
              int dim, DistFn dist_fn) {
    const uint8_t *p = (const uint8_t *)buf;
    (void)size;

/* MSVC-compatible read macros (comma expression, no GCC statement extensions) */
#define RI32() (p += 4, *(const int32_t *)(p - 4))
#define RU64() (p += 8, *(const uint64_t*)(p - 8))
#define RF32() (p += 4, *(const float    *)(p - 4))

    int n_nodes        = RI32();
    int file_dim       = RI32();
    int M              = RI32();
    int M_max0         = RI32();
    int ef_construction= RI32();
    int ef_search      = RI32();
    int ep_node        = RI32();
    int max_layer      = RI32();

    if (file_dim != dim) return PISTADB_ECORRUPT;

    int r = hnsw_create(idx, dim, dist_fn, M, ef_construction, ef_search);
    if (r != PISTADB_OK) return r;
    idx->M_max0   = M_max0;
    idx->ep_node  = ep_node;
    idx->max_layer= max_layer;

    /* Grow to needed capacity */
    while (idx->node_cap < n_nodes) hnsw_grow(idx);

    for (int i = 0; i < n_nodes; i++) {
        uint64_t vec_id = RU64();
        int      level  = RI32();

        r = node_init(&idx->nodes[i], vec_id, level, idx->M, idx->M_max0);
        if (r != PISTADB_OK) return r;
        idx->n_nodes++;

        memcpy(VS_LABEL(&idx->vs, i), p, 256); p += 256;
        float *v = VS_VEC(&idx->vs, i);
        for (int d = 0; d < idx->dim; d++) v[d] = RF32();

        for (int l = 0; l <= level; l++) {
            int cnt = RI32();
            idx->nodes[i].neighbor_cnt[l] = cnt;
            if (cnt > idx->nodes[i].neighbor_cap[l]) {
                int *nd = (int *)realloc(idx->nodes[i].neighbors[l], sizeof(int) * (size_t)cnt);
                if (!nd) return PISTADB_ENOMEM;
                idx->nodes[i].neighbors[l]    = nd;
                idx->nodes[i].neighbor_cap[l] = cnt;
            }
            for (int j = 0; j < cnt; j++)
                idx->nodes[i].neighbors[l][j] = RI32();
        }
    }
#undef RI32
#undef RU64
#undef RF32
    return PISTADB_OK;
}
