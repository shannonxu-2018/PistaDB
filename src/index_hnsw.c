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

/* Cross-platform read-prefetch for hot HNSW fan-out paths.  No-op when the
 * compiler does not expose an intrinsic — prefetches are advisory and never
 * affect correctness. */
#if defined(__GNUC__) || defined(__clang__)
#  define PISTA_PREFETCH(p) __builtin_prefetch((const void *)(p), 0, 1)
#elif defined(_MSC_VER)
#  include <xmmintrin.h>
#  define PISTA_PREFETCH(p) _mm_prefetch((const char *)(p), _MM_HINT_T1)
#else
#  define PISTA_PREFETCH(p) ((void)0)
#endif

/* ── Random layer assignment (per-instance RNG, thread-safe w/ external lock) ── */
static int random_level(PCG *rng, float mL) {
    float r = pcg_f32(rng);
    if (r < 1e-9f) r = 1e-9f;
    int level = (int)(-logf(r) * mL);
    if (level >= HNSW_MAX_LAYERS) level = HNSW_MAX_LAYERS - 1;
    return level;
}

/* ── Node helpers ────────────────────────────────────────────────────────── */
static void node_free(HNSWNode *n) {
    if (!n->neighbors) return;
    for (int i = 0; i <= n->level; i++) free(n->neighbors[i]);
    free(n->neighbors);
    free(n->neighbor_cnt);
    free(n->neighbor_cap);
    n->neighbors    = NULL;
    n->neighbor_cnt = NULL;
    n->neighbor_cap = NULL;
}

static int node_init(HNSWNode *n, uint64_t id, int level, int M, int M_max0) {
    n->vec_id       = id;
    n->level        = level;
    n->neighbors    = (int **)calloc((size_t)(level + 1), sizeof(int *));
    n->neighbor_cnt = (int  *)calloc((size_t)(level + 1), sizeof(int));
    n->neighbor_cap = (int  *)calloc((size_t)(level + 1), sizeof(int));
    if (!n->neighbors || !n->neighbor_cnt || !n->neighbor_cap) {
        node_free(n);
        return PISTADB_ENOMEM;
    }

    for (int i = 0; i <= level; i++) {
        int cap = (i == 0) ? M_max0 * 2 : M * 2;
        n->neighbors[i] = (int *)malloc(sizeof(int) * (size_t)cap);
        if (!n->neighbors[i]) {
            /* Roll back already-allocated layers so they don't leak when the
             * caller treats this node as never-initialised. */
            n->level = i - 1;
            node_free(n);
            n->level = level;  /* preserve nominal level for caller diagnostics */
            return PISTADB_ENOMEM;
        }
        n->neighbor_cap[i] = cap;
        n->neighbor_cnt[i] = 0;
    }
    return PISTADB_OK;
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

/* HNSW_MAX_M caps the per-layer fan-out so the tmp_nb scratch array used by
 * hnsw_insert (sized HNSW_MAX_LAYERS * 64 = 3072) cannot overflow.  M_max0 is
 * 2*M, and select_neighbors writes up to M_max0 entries; 1024 keeps M_max0 at
 * 2048, well within the buffer. */
#define HNSW_MAX_M 1024

int hnsw_create(HNSWIndex *idx, int dim, DistFn dist_fn,
                int M, int ef_construction, int ef_search) {
    if (M < 2)          M = 2;
    if (M > HNSW_MAX_M) M = HNSW_MAX_M;
    idx->dim            = dim;
    idx->dist_fn        = dist_fn;
    idx->batch_fn       = NULL;       /* set by pistadb.c after create / load */
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
    pcg_seed(&idx->rng, 42);
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
                         Heap *W, Heap *C, EpochSet *visited) {
    heap_clear(W);
    heap_clear(C);
    epochset_clear(visited);

    float d_ep = query_dist(idx, query, ep);
    heap_push(C, d_ep, (uint64_t)ep);   /* min-heap: nearest on top */
    heap_push(W, d_ep, (uint64_t)ep);   /* max-heap: furthest on top */
    epochset_set(visited, ep);

    BatchDistFn batch_fn = idx->batch_fn;

    while (C->size > 0) {
        HeapItem c = heap_pop(C);       /* nearest candidate */
        float w_far = heap_top(W).key;  /* furthest in results */

        if (c.key > w_far) break;       /* all remaining candidates are farther */

        int c_idx = (int)c.id;
        const HNSWNode *cn = &idx->nodes[c_idx];
        if (layer > cn->level) continue;

        const int n_nb = cn->neighbor_cnt[layer];

        /* Two-pass fan-out: first gather "live & unvisited" neighbors into
         * a stack buffer, then issue ONE batched distance call for them.
         * This hoists the SIMD dispatch out of the per-neighbor loop and
         * lets the kernel prefetch each next vector while computing the
         * current one.  The output ordering and heap maintenance are
         * unchanged — bit-identical to the per-pair fallback. */
        enum { FAN_BATCH = 64 };       /* M_max0 ≤ 64 by construction      */
        const float *fan_ptrs [FAN_BATCH];
        int          fan_nb   [FAN_BATCH];
        float        fan_dist [FAN_BATCH];

        int fan_n = 0;
        for (int j = 0; j < n_nb && fan_n < FAN_BATCH; j++) {
            int nb = cn->neighbors[layer][j];
            /* Prefetch the next neighbor while we triage this one. */
            if (j + 1 < n_nb) {
                int nb_next = cn->neighbors[layer][j + 1];
                if (nb_next >= 0 && nb_next < idx->n_nodes) {
                    PISTA_PREFETCH(VS_VEC(&idx->vs, nb_next));
                }
            }
            if (nb < 0 || nb >= idx->n_nodes) continue;
            if (epochset_test(visited, nb)) continue;
            epochset_set(visited, nb);
            /* Tombstoned node — same vec_id == UINT64_MAX sentinel as
             * hnsw_delete().  Skip the distance call and do not enqueue
             * into C/W: it would only be filtered at result-extraction
             * time anyway.  Side-effect: we no longer traverse through
             * deleted nodes, which is fine at the M values we use here. */
            if (idx->nodes[nb].vec_id == UINT64_MAX) continue;
            fan_ptrs[fan_n] = VS_VEC(&idx->vs, nb);
            fan_nb  [fan_n] = nb;
            fan_n++;
        }
        if (fan_n > 0) {
            if (batch_fn) {
                batch_fn(query, fan_ptrs, (size_t)fan_n,
                         idx->dim, fan_dist);
            } else {
                for (int t = 0; t < fan_n; t++)
                    fan_dist[t] = query_dist(idx, query, fan_nb[t]);
            }
            for (int t = 0; t < fan_n; t++) {
                float d_nb   = fan_dist[t];
                int   nb     = fan_nb[t];
                float w_far2 = heap_top(W).key;
                if (d_nb < w_far2 || W->size < ef) {
                    heap_push(C, d_nb, (uint64_t)nb);
                    heap_push(W, d_nb, (uint64_t)nb);
                    if (W->size > ef) heap_pop(W);  /* remove furthest */
                }
            }
        }
    }
    return PISTADB_OK;
}

/* ── SELECT-NEIGHBORS (simple: take M nearest from W) ───────────────────── */
static int select_neighbors(Heap *W, int M, int *out, int *out_cnt) {
    *out_cnt = 0;
    int total = W->size;
    if (total == 0) return PISTADB_OK;
    float *keys = (float    *)malloc(sizeof(float)    * (size_t)total);
    int   *ids  = (int      *)malloc(sizeof(int)      * (size_t)total);
    if (!keys || !ids) { free(keys); free(ids); return PISTADB_ENOMEM; }

    int n = 0;
    while (W->size > 0) {
        HeapItem it = heap_pop(W);
        keys[n] = it.key;
        ids[n]  = (int)it.id;
        n++;
    }
    /* Sort ascending (insertion sort - small arrays) */
    for (int i = 1; i < n; i++) {
        float ki = keys[i]; int ii = ids[i];
        int j = i - 1;
        while (j >= 0 && keys[j] > ki) { keys[j+1] = keys[j]; ids[j+1] = ids[j]; j--; }
        keys[j+1] = ki; ids[j+1] = ii;
    }
    *out_cnt = (n < M) ? n : M;
    for (int i = 0; i < *out_cnt; i++) out[i] = ids[i];
    free(keys); free(ids);
    return PISTADB_OK;
}

/* ── INSERT ──────────────────────────────────────────────────────────────── */

int hnsw_insert(HNSWIndex *idx, uint64_t id, const char *label, const float *vec) {
    /* UINT64_MAX is reserved as the lazy-deletion tombstone (see hnsw_delete);
     * accepting it as a real id would later be misread as a deleted node and
     * be skipped by every search. */
    if (id == UINT64_MAX) return PISTADB_EINVAL;
    if (idx->vs.pager) return PISTADB_EINVAL;   /* read-only in paged mode */
    if (idx->n_nodes == idx->node_cap) {
        int r = hnsw_grow(idx);
        if (r != PISTADB_OK) return r;
    }

    int new_node = idx->n_nodes;
    int level    = random_level(&idx->rng, idx->mL);

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

    /* Work buffers — same stack-fallback pattern as hnsw_search().  Insert
     * runs less frequently than search, so the win here is smaller, but
     * removing two malloc/free pairs per insert keeps batch insert latency
     * down. */
    int max_ef = (idx->ef_construction > idx->M_max0 * 4) ?
                  idx->ef_construction : idx->M_max0 * 4;
    int hcap_i = max_ef + 8;
    HeapItem  w_stack[256];
    HeapItem  c_stack[256];
    Heap   W, C;
    EpochSet visited;
    int W_ok = 0, C_ok = 0, V_ok = 0;
    if (hcap_i <= (int)(sizeof w_stack / sizeof w_stack[0])) {
        heap_init_with_buffer(&W, w_stack, hcap_i, 1); W_ok = 1;
        heap_init_with_buffer(&C, c_stack, hcap_i, 0); C_ok = 1;
    } else {
        if (heap_init(&W, hcap_i, 1) != PISTADB_OK) goto insert_oom;
        W_ok = 1;
        if (heap_init(&C, hcap_i, 0) != PISTADB_OK) goto insert_oom;
        C_ok = 1;
    }
    if (epochset_init(&visited, idx->node_cap + 8) != PISTADB_OK) goto insert_oom;
    V_ok = 1;

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
                /* Simple pruning: rebuild neighbor list keeping M nearest.
                 * tmp_h is sized to (n_existing_neighbors + 2) which is
                 * always ≤ M_max0 + a few — well within the 256-slot stack
                 * buffer. */
                int       tcap = nbn->neighbor_cnt[lc] + 2;
                HeapItem  tmp_stack[256];
                Heap tmp_h;
                if (tcap <= (int)(sizeof tmp_stack / sizeof tmp_stack[0])) {
                    heap_init_with_buffer(&tmp_h, tmp_stack, tcap, 1);
                } else {
                    heap_init(&tmp_h, tcap, 1);
                }
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
    epochset_free(&visited);
    return PISTADB_OK;

insert_oom:
    if (W_ok) heap_free(&W);
    if (C_ok) heap_free(&C);
    if (V_ok) epochset_free(&visited);
    /* The new node has been added to the graph already; leaving it without
     * lower-layer connections is accepted as a graceful-degradation outcome.
     * Surface ENOMEM so callers can react. */
    return PISTADB_ENOMEM;
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
    if (idx->vs.pager) return PISTADB_EINVAL;   /* read-only in paged mode */
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
    int hcap   = max_ef + 8;

    /* Stack-allocate the typical heap buffers and result-collection arrays.
     * Up to ef ≈ 248 and total ≈ 256 we never touch the heap allocator on the
     * search hot path.  heap_push's grow-on-overflow path transparently
     * promotes to a malloc'd buffer if the workload exceeds these caps. */
    HeapItem  w_stack[256];
    HeapItem  c_stack[256];
    float     dist_stack[256];
    int       node_stack[256];

    Heap   W, C;
    EpochSet visited;
    int W_ok = 0, C_ok = 0, V_ok = 0;

    if (hcap <= (int)(sizeof w_stack / sizeof w_stack[0])) {
        heap_init_with_buffer(&W, w_stack, hcap, 1); W_ok = 1;
        heap_init_with_buffer(&C, c_stack, hcap, 0); C_ok = 1;
    } else {
        if (heap_init(&W, hcap, 1) != PISTADB_OK) return 0;
        W_ok = 1;
        if (heap_init(&C, hcap, 0) != PISTADB_OK) { heap_free(&W); return 0; }
        C_ok = 1;
    }
    if (epochset_init(&visited, idx->n_nodes + 8) != PISTADB_OK) {
        if (W_ok) heap_free(&W);
        if (C_ok) heap_free(&C);
        return 0;
    }
    V_ok = 1;
    (void)V_ok;

    int ep = idx->ep_node;
    int L  = idx->max_layer;

    for (int lc = L; lc > 0; lc--) {
        search_layer(idx, query, ep, 1, lc, &W, &C, &visited);
        if (W.size > 0) ep = (int)heap_top(&W).id;
        heap_clear(&W); heap_clear(&C); epochset_clear(&visited);
    }
    search_layer(idx, query, ep, ef, 0, &W, &C, &visited);

    /* Collect results (W is max-heap; drain to get ascending order). */
    int   total = W.size;
    int   cnt   = 0;
    float *dist_buf = dist_stack;
    int   *node_buf = node_stack;
    int    bufs_owned = 0;
    if (total > (int)(sizeof dist_stack / sizeof dist_stack[0])) {
        dist_buf = (float *)malloc(sizeof(float) * (size_t)total);
        node_buf = (int   *)malloc(sizeof(int)   * (size_t)total);
        if (!dist_buf || !node_buf) {
            free(dist_buf); free(node_buf);
            heap_free(&W); heap_free(&C); epochset_free(&visited);
            return 0;
        }
        bufs_owned = 1;
    }
    if (total > 0) {
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
    }
    if (bufs_owned) { free(dist_buf); free(node_buf); }
    heap_free(&W); heap_free(&C);
    epochset_free(&visited);
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
        memcpy(p, v, sizeof(float) * (size_t)idx->dim); p += sizeof(float) * (size_t)idx->dim;
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
    const uint8_t *p   = (const uint8_t *)buf;
    const uint8_t *end = p + size;

/* Bounds-checked read macros: bail on truncated buffer rather than read OOB. */
#define NEED(n)  do { if ((size_t)(end - p) < (size_t)(n)) return PISTADB_ECORRUPT; } while (0)
#define RI32()   (p += 4, *(const int32_t *)(p - 4))
#define RU64()   (p += 8, *(const uint64_t*)(p - 8))
#define RF32()   (p += 4, *(const float    *)(p - 4))

    NEED(8 * 4);
    int n_nodes        = RI32();
    int file_dim       = RI32();
    int M              = RI32();
    int M_max0         = RI32();
    int ef_construction= RI32();
    int ef_search      = RI32();
    int ep_node        = RI32();
    int max_layer      = RI32();

    if (file_dim != dim) return PISTADB_ECORRUPT;
    /* Sanity-bound counts read from disk before using them for allocation.
     * M and M_max0 must respect the HNSW_MAX_M cap so the tmp_nb scratch
     * buffer in hnsw_insert cannot overflow when re-indexing. */
    if (n_nodes < 0 || n_nodes > 500000000 || M < 2 || M > HNSW_MAX_M ||
        M_max0 < 0 || M_max0 > 2 * HNSW_MAX_M ||
        ef_construction < 0 || ef_search < 0 ||
        max_layer >= HNSW_MAX_LAYERS) return PISTADB_ECORRUPT;
    if (ep_node < -1 || ep_node >= n_nodes) return PISTADB_ECORRUPT;

    int r = hnsw_create(idx, dim, dist_fn, M, ef_construction, ef_search);
    if (r != PISTADB_OK) return r;
    idx->M_max0   = M_max0;
    idx->ep_node  = ep_node;
    idx->max_layer= max_layer;

    /* Grow to needed capacity — bail on grow failure instead of crashing later. */
    while (idx->node_cap < n_nodes) {
        int gr = hnsw_grow(idx);
        if (gr != PISTADB_OK) return gr;
    }

    for (int i = 0; i < n_nodes; i++) {
        NEED(8 + 4 + 256);
        uint64_t vec_id = RU64();
        int      level  = RI32();
        if (level < 0 || level >= HNSW_MAX_LAYERS) return PISTADB_ECORRUPT;

        r = node_init(&idx->nodes[i], vec_id, level, idx->M, idx->M_max0);
        if (r != PISTADB_OK) return r;
        idx->n_nodes++;

        memcpy(VS_LABEL(&idx->vs, i), p, 256); p += 256;
        NEED((size_t)idx->dim * sizeof(float));
        float *v = VS_VEC(&idx->vs, i);
        memcpy(v, p, sizeof(float) * (size_t)idx->dim); p += sizeof(float) * (size_t)idx->dim;

        for (int l = 0; l <= level; l++) {
            NEED(4);
            int cnt = RI32();
            if (cnt < 0) return PISTADB_ECORRUPT;
            NEED((size_t)cnt * 4);
            idx->nodes[i].neighbor_cnt[l] = cnt;
            if (cnt > idx->nodes[i].neighbor_cap[l]) {
                int *nd = (int *)realloc(idx->nodes[i].neighbors[l], sizeof(int) * (size_t)cnt);
                if (!nd) return PISTADB_ENOMEM;
                idx->nodes[i].neighbors[l]    = nd;
                idx->nodes[i].neighbor_cap[l] = cnt;
            }
            for (int j = 0; j < cnt; j++) {
                int nb = RI32();
                /* Refuse silent corruption — an out-of-range neighbor index
                 * would either point at a different node post-load or trigger
                 * OOB reads during search.  Bail rather than mutate the data. */
                if (nb < 0 || nb >= n_nodes) return PISTADB_ECORRUPT;
                idx->nodes[i].neighbors[l][j] = nb;
            }
        }
    }
#undef NEED
#undef RI32
#undef RU64
#undef RF32
    return PISTADB_OK;
}

/* ── Paged load ──────────────────────────────────────────────────────────────
 * HNSW records are variable-size (per-layer neighbour lists), so a fixed
 * stride can't address them — we build a per-node offset table instead.  The
 * navigation graph (neighbours) is loaded resident (search must walk it);
 * each node's label+vector are paged via vs->pg_off.  Read-only.
 *
 * On-disk (hnsw_save): [8×int32 header] then per node
 *   [u64 vec_id][int32 level][char label[256]][float vec[dim]]
 *   [per layer 0..level: int32 cnt, int32 neighbors[cnt]]
 */
int hnsw_load_paged(HNSWIndex *idx, const char *path,
                    uint64_t vec_off, uint64_t vec_size,
                    int dim, DistFn dist_fn, size_t cache_bytes) {
    memset(idx, 0, sizeof(*idx));
    idx->dim = dim;
    idx->dist_fn = dist_fn;
    idx->batch_fn = NULL;                 /* set by pistadb.c after load */
    pcg_seed(&idx->rng, 42);

    int rc = vs_open_paged(&idx->vs, path, vec_off, vec_size, dim, cache_bytes);
    if (rc != PISTADB_OK) { memset(idx, 0, sizeof(*idx)); return rc; }
    rc = PISTADB_ECORRUPT;

    uint8_t h[32];
    if (vec_size < 32 || vs_pg_get(&idx->vs, 0, 32, h) != PISTADB_OK) goto fail;
    int32_t n_nodes, fdim, M, M_max0, ef_c, ef_s, ep_node, max_layer;
    memcpy(&n_nodes,&h[0], 4); memcpy(&fdim,    &h[4],  4);
    memcpy(&M,      &h[8], 4); memcpy(&M_max0,  &h[12], 4);
    memcpy(&ef_c,   &h[16],4); memcpy(&ef_s,    &h[20], 4);
    memcpy(&ep_node,&h[24],4); memcpy(&max_layer,&h[28],4);
    if (fdim != dim) goto fail;
    if (n_nodes < 0 || n_nodes > 500000000 || M < 2 || M > HNSW_MAX_M ||
        M_max0 < 0 || M_max0 > 2 * HNSW_MAX_M || ef_c < 0 || ef_s < 0 ||
        max_layer >= HNSW_MAX_LAYERS) goto fail;
    if (ep_node < -1 || ep_node >= n_nodes) goto fail;

    idx->M = M; idx->M_max0 = M_max0;
    idx->ef_construction = ef_c; idx->ef_search = ef_s;
    idx->mL = 1.0f / logf((float)M);
    idx->ep_node = ep_node; idx->max_layer = max_layer;
    idx->n_nodes = 0;
    idx->node_cap = (n_nodes > 0) ? n_nodes : 1;
    idx->nodes = (HNSWNode *)calloc((size_t)idx->node_cap, sizeof(HNSWNode));
    idx->vs.pg_off = (uint64_t *)malloc(sizeof(uint64_t) *
                                        (size_t)(n_nodes > 0 ? n_nodes : 1));
    if (!idx->nodes || !idx->vs.pg_off) { rc = PISTADB_ENOMEM; goto fail; }
    idx->vs.pg_lbl_rel = 8 + 4;            /* after vec_id + level         */
    idx->vs.pg_vec_rel = 8 + 4 + 256;      /* + label[256]                 */

    uint64_t cur = 32;
    for (int i = 0; i < n_nodes; i++) {
        uint64_t node_off = cur;
        uint8_t  nh[12];
        if (vs_pg_get(&idx->vs, cur, 12, nh) != PISTADB_OK) { rc = PISTADB_EIO; goto fail; }
        uint64_t vec_id; int32_t level;
        memcpy(&vec_id, nh,     8);
        memcpy(&level,  nh + 8, 4);
        cur += 12;
        if (level < 0 || level >= HNSW_MAX_LAYERS) goto fail;

        idx->vs.pg_off[i] = node_off;
        cur += 256;                         /* skip label  (paged)         */
        cur += (uint64_t)dim * sizeof(float); /* skip vector (paged)       */

        if (node_init(&idx->nodes[i], vec_id, level, idx->M, idx->M_max0)
                != PISTADB_OK) { rc = PISTADB_ENOMEM; goto fail; }
        idx->n_nodes++;                     /* so hnsw_free unwinds cleanly */

        for (int l = 0; l <= level; l++) {
            int32_t cnt;
            if (vs_pg_get(&idx->vs, cur, 4, &cnt) != PISTADB_OK) { rc = PISTADB_EIO; goto fail; }
            cur += 4;
            if (cnt < 0) goto fail;
            if (cnt > idx->nodes[i].neighbor_cap[l]) {
                int *nd = (int *)realloc(idx->nodes[i].neighbors[l],
                                         sizeof(int) * (size_t)cnt);
                if (!nd) { rc = PISTADB_ENOMEM; goto fail; }
                idx->nodes[i].neighbors[l]    = nd;
                idx->nodes[i].neighbor_cap[l] = cnt;
            }
            idx->nodes[i].neighbor_cnt[l] = cnt;
            if (cnt > 0 &&
                vs_pg_read(&idx->vs, cur, (uint64_t)cnt * 4,
                           idx->nodes[i].neighbors[l]) != PISTADB_OK) {
                rc = PISTADB_EIO; goto fail;
            }
            for (int j = 0; j < cnt; j++) {
                int nb = idx->nodes[i].neighbors[l][j];
                if (nb < 0 || nb >= n_nodes) goto fail;   /* OOB-read guard */
            }
            cur += (uint64_t)cnt * 4;
        }
    }
    return PISTADB_OK;

fail:
    hnsw_free(idx);                         /* node_free + free(nodes) + vs_free */
    memset(idx, 0, sizeof(*idx));
    return rc;
}
