/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - index_diskann.c
 * Vamana (DiskANN) graph index.
 *
 * Algorithm:
 *  1. Insert: add node to graph, greedy-search from medoid to find candidates,
 *             prune with alpha-RNG criterion to obtain R neighbors,
 *             add bidirectional edges.
 *  2. Build:  optionally call after bulk insert to apply a full Vamana pass.
 *  3. Search: greedy beam search (similar to HNSW layer-0 search).
 */
#include "index_diskann.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

int diskann_create(DiskANNIndex *idx, int dim, DistFn dist_fn,
                   int R, int L, float alpha) {
    memset(idx, 0, sizeof(*idx));
    idx->dim      = dim;
    idx->dist_fn  = dist_fn;
    idx->R        = R;
    idx->L        = (L > R) ? L : R;
    idx->alpha    = (alpha >= 1.0f) ? alpha : 1.2f;
    idx->n_nodes  = 0;
    idx->node_cap = 64;
    idx->medoid   = -1;

    idx->nodes   = (DiskANNNode *)calloc((size_t)idx->node_cap, sizeof(DiskANNNode));
    if (!idx->nodes) return PISTADB_ENOMEM;
    if (vs_init(&idx->vs, dim, idx->node_cap) != PISTADB_OK) return PISTADB_ENOMEM;
    return PISTADB_OK;
}

void diskann_free(DiskANNIndex *idx) {
    for (int i = 0; i < idx->n_nodes; i++) free(idx->nodes[i].neighbors);
    free(idx->nodes);
    vs_free(&idx->vs);
    memset(idx, 0, sizeof(*idx));
}

static int da_grow(DiskANNIndex *idx) {
    int nc = idx->node_cap * 2 + 8;
    DiskANNNode *nn = (DiskANNNode *)realloc(idx->nodes, sizeof(DiskANNNode) * (size_t)nc);
    if (!nn) return PISTADB_ENOMEM;
    idx->nodes = nn;
    int r = vs_ensure(&idx->vs, nc);
    if (r != PISTADB_OK) return r;
    memset(idx->nodes + idx->node_cap, 0, sizeof(DiskANNNode) * (size_t)(nc - idx->node_cap));
    idx->node_cap = nc;
    return PISTADB_OK;
}

static inline float node_dist_da(const DiskANNIndex *idx, int a, int b) {
    return idx->dist_fn(VS_VEC(&idx->vs, a), VS_VEC(&idx->vs, b), idx->dim);
}
static inline float query_dist_da(const DiskANNIndex *idx, const float *q, int n) {
    return idx->dist_fn(q, VS_VEC(&idx->vs, n), idx->dim);
}

/* ── Greedy search from medoid, return L nearest in 'result' heap ────────── */

static void greedy_search(const DiskANNIndex *idx, const float *query,
                           int start, int L,
                           Heap *result, Bitset *visited) {
    heap_clear(result);
    bitset_clear(visited);

    Heap cand; heap_init(&cand, L * 2 + 8, 0);  /* min-heap */

    float d0 = query_dist_da(idx, query, start);
    heap_push(&cand,   d0, (uint64_t)start);
    heap_push(result,  d0, (uint64_t)start);
    bitset_set(visited, start);

    while (cand.size > 0) {
        HeapItem c = heap_pop(&cand);
        /* Prune: if all remaining are farther than worst in result */
        if (result->size >= L && c.key > heap_top(result).key) break;

        int ci = (int)c.id;
        const DiskANNNode *cn = &idx->nodes[ci];
        for (int j = 0; j < cn->neighbor_cnt; j++) {
            int nb = cn->neighbors[j];
            if (nb < 0 || nb >= idx->n_nodes) continue;
            if (bitset_test(visited, nb)) continue;
            bitset_set(visited, nb);

            float d_nb = query_dist_da(idx, query, nb);
            if (result->size < L) {
                heap_push(result, d_nb, (uint64_t)nb);
                heap_push(&cand,  d_nb, (uint64_t)nb);
            } else if (d_nb < heap_top(result).key) {
                heap_pop(result);
                heap_push(result, d_nb, (uint64_t)nb);
                heap_push(&cand,  d_nb, (uint64_t)nb);
            }
        }
    }
    heap_free(&cand);
}

/* ── Insertion sort (ascending by dist) ─────────────────────────────────── */
/* Faster than bubble sort for small, partially-ordered arrays (R typically 16-64). */
static void sort_by_dist(float *ds, int *ns, int n) {
    for (int i = 1; i < n; i++) {
        float dk = ds[i]; int nk = ns[i];
        int j = i - 1;
        while (j >= 0 && ds[j] > dk) { ds[j+1] = ds[j]; ns[j+1] = ns[j]; j--; }
        ds[j+1] = dk; ns[j+1] = nk;
    }
}

/* ── Alpha-RNG pruning ───────────────────────────────────────────────────── */
/*
 * From candidate set (sorted ascending by distance to p),
 * select R neighbors using the Vamana criterion:
 *   accept candidate c if for all already-accepted neighbors p*:
 *     dist(p*, c) > alpha * dist(p, c)
 */
static void robust_prune(const DiskANNIndex *idx, int p,
                          float *cand_d, int *cand_n, int cand_cnt,
                          int R, float alpha,
                          int *out, int *out_cnt) {
    *out_cnt = 0;
    for (int i = 0; i < cand_cnt && *out_cnt < R; i++) {
        int ci = cand_n[i];
        if (ci == p) continue;
        /* Check RNG criterion */
        int dominated = 0;
        for (int j = 0; j < *out_cnt; j++) {
            float d_star_c = node_dist_da(idx, out[j], ci);
            if (alpha * d_star_c <= cand_d[i]) { dominated = 1; break; }
        }
        if (!dominated) { out[*out_cnt] = ci; (*out_cnt)++; }
    }
}

/* ── Add bidirectional edge ──────────────────────────────────────────────── */
static int node_add_nb(DiskANNNode *n, int nb) {
    for (int i = 0; i < n->neighbor_cnt; i++) if (n->neighbors[i] == nb) return PISTADB_OK;
    if (n->neighbor_cnt == n->neighbor_cap) {
        int nc = n->neighbor_cap * 2 + 8;
        int *nd = (int *)realloc(n->neighbors, sizeof(int) * (size_t)nc);
        if (!nd) return PISTADB_ENOMEM;
        n->neighbors = nd; n->neighbor_cap = nc;
    }
    n->neighbors[n->neighbor_cnt++] = nb;
    return PISTADB_OK;
}

static void node_set_neighbors(DiskANNNode *n, const int *nb, int cnt) {
    if (cnt > n->neighbor_cap) {
        int *new_nb = (int *)malloc(sizeof(int) * (size_t)cnt);
        if (!new_nb) {
            /* Allocation failed — keep the old buffer intact and the count
             * truthful (do not mutate neighbor_cnt to a value the buffer can't
             * back, which would lead to OOB reads in greedy_search). */
            return;
        }
        free(n->neighbors);
        n->neighbors    = new_nb;
        n->neighbor_cap = cnt;
    }
    n->neighbor_cnt = cnt;
    if (cnt > 0) memcpy(n->neighbors, nb, sizeof(int) * (size_t)cnt);
}

/* ── Insert ──────────────────────────────────────────────────────────────── */

int diskann_insert(DiskANNIndex *idx, uint64_t id, const char *label, const float *vec) {
    if (idx->n_nodes == idx->node_cap) {
        int r = da_grow(idx); if (r) return r;
    }
    int p = idx->n_nodes++;
    idx->nodes[p].vec_id = id;
    idx->nodes[p].deleted = 0;
    idx->nodes[p].neighbor_cnt = 0;
    idx->nodes[p].neighbor_cap = idx->R + 4;
    idx->nodes[p].neighbors = (int *)malloc(sizeof(int) * (size_t)(idx->R + 4));
    if (!idx->nodes[p].neighbors) return PISTADB_ENOMEM;
    memcpy(VS_VEC(&idx->vs, p), vec, sizeof(float) * (size_t)idx->dim);
    if (label) { strncpy(VS_LABEL(&idx->vs, p), label, 255); VS_LABEL(&idx->vs, p)[255] = '\0'; }
    else        VS_LABEL(&idx->vs, p)[0] = '\0';

    if (p == 0) { idx->medoid = 0; return PISTADB_OK; }

    /* Greedy search from medoid */
    Heap result; heap_init(&result, idx->L * 2 + 8, 1);
    Bitset visited; bitset_init(&visited, idx->n_nodes + 8);

    greedy_search(idx, vec, idx->medoid, idx->L, &result, &visited);

    /* Sort result ascending */
    int total = result.size;
    float *ds = (float *)malloc(sizeof(float) * (size_t)total);
    int   *ns = (int   *)malloc(sizeof(int)   * (size_t)total);
    for (int i = total - 1; i >= 0; i--) {
        HeapItem it = heap_pop(&result);
        ds[i] = it.key; ns[i] = (int)it.id;
    }

    /* Prune to R neighbors */
    int *chosen = (int *)malloc(sizeof(int) * (size_t)(idx->R + 4));
    int  chosen_cnt = 0;
    robust_prune(idx, p, ds, ns, total, idx->R, idx->alpha, chosen, &chosen_cnt);

    node_set_neighbors(&idx->nodes[p], chosen, chosen_cnt);

    /* Pre-allocate buffers for bidirectional pruning — reused across all chosen neighbors. */
    int    max_nc = idx->R + 1;   /* nc ≤ R+1 after node_add_nb on a list capped at R */
    float *nds    = (float *)malloc(sizeof(float) * (size_t)max_nc);
    int   *nns    = (int   *)malloc(sizeof(int)   * (size_t)max_nc);
    int   *pruned = (int   *)malloc(sizeof(int)   * (size_t)(idx->R + 4));

    /* Bidirectional: add p as neighbor of each chosen */
    for (int i = 0; i < chosen_cnt; i++) {
        int nb = chosen[i];
        node_add_nb(&idx->nodes[nb], p);
        /* Prune nb's list if oversized */
        if (nds && nns && pruned && idx->nodes[nb].neighbor_cnt > idx->R) {
            int nc = idx->nodes[nb].neighbor_cnt;
            for (int j = 0; j < nc; j++) {
                nns[j] = idx->nodes[nb].neighbors[j];
                nds[j] = node_dist_da(idx, nb, nns[j]);
            }
            sort_by_dist(nds, nns, nc);
            int new_cnt = 0;
            robust_prune(idx, nb, nds, nns, nc, idx->R, idx->alpha, pruned, &new_cnt);
            node_set_neighbors(&idx->nodes[nb], pruned, new_cnt);
        }
    }
    free(nds); free(nns); free(pruned);

    free(ds); free(ns); free(chosen);
    heap_free(&result);
    bitset_free(&visited);
    return PISTADB_OK;
}

int diskann_delete(DiskANNIndex *idx, uint64_t id) {
    for (int i = 0; i < idx->n_nodes; i++) {
        if (idx->nodes[i].vec_id == id) { idx->nodes[i].deleted = 1; return PISTADB_OK; }
    }
    return PISTADB_ENOTFOUND;
}

int diskann_update(DiskANNIndex *idx, uint64_t id, const float *vec) {
    for (int i = 0; i < idx->n_nodes; i++) {
        if (idx->nodes[i].vec_id == id) {
            memcpy(VS_VEC(&idx->vs, i), vec, sizeof(float) * (size_t)idx->dim);
            return PISTADB_OK;
        }
    }
    return PISTADB_ENOTFOUND;
}

/* ── Full Vamana build pass ──────────────────────────────────────────────── */

int diskann_build(DiskANNIndex *idx) {
    if (idx->n_nodes < 2) return PISTADB_OK;

    /* Find medoid: average vector, nearest node */
    float *avg = (float *)calloc((size_t)idx->dim, sizeof(float));
    if (!avg) return PISTADB_ENOMEM;
    int active = 0;
    for (int i = 0; i < idx->n_nodes; i++) {
        if (idx->nodes[i].deleted) continue;
        const float *v = (const float *)VS_VEC(&idx->vs, i);
        for (int d = 0; d < idx->dim; d++) avg[d] += v[d];
        active++;
    }
    if (active == 0) { free(avg); return PISTADB_OK; }
    for (int d = 0; d < idx->dim; d++) avg[d] /= (float)active;
    float bd = FLT_MAX; int medoid = 0;
    for (int i = 0; i < idx->n_nodes; i++) {
        if (idx->nodes[i].deleted) continue;
        float d = dist_l2sq(avg, VS_VEC(&idx->vs, i), idx->dim);
        if (d < bd) { bd = d; medoid = i; }
    }
    free(avg);
    idx->medoid = medoid;

    /* Random permutation */
    int *order = (int *)malloc(sizeof(int) * (size_t)idx->n_nodes);
    if (!order) return PISTADB_ENOMEM;
    for (int i = 0; i < idx->n_nodes; i++) order[i] = i;
    PCG rng; pcg_seed(&rng, 777);
    for (int i = idx->n_nodes - 1; i > 0; i--) {
        int j = (int)(pcg_u32(&rng) % (uint32_t)(i + 1));
        int t = order[i]; order[i] = order[j]; order[j] = t;
    }

    Heap result; heap_init(&result, idx->L * 2 + 8, 1);
    Bitset visited; bitset_init(&visited, idx->n_nodes + 8);

    for (int oi = 0; oi < idx->n_nodes; oi++) {
        int p = order[oi];
        if (idx->nodes[p].deleted) continue;

        const float *vec = (const float *)VS_VEC(&idx->vs, p);
        greedy_search(idx, vec, medoid, idx->L, &result, &visited);

        int total = result.size;
        float *ds = (float *)malloc(sizeof(float) * (size_t)total);
        int   *ns = (int   *)malloc(sizeof(int)   * (size_t)total);
        for (int i = total - 1; i >= 0; i--) {
            HeapItem it = heap_pop(&result);
            ds[i] = it.key; ns[i] = (int)it.id;
        }

        int *chosen = (int *)malloc(sizeof(int) * (size_t)(idx->R + 4));
        int  chosen_cnt = 0;
        robust_prune(idx, p, ds, ns, total, idx->R, idx->alpha, chosen, &chosen_cnt);
        node_set_neighbors(&idx->nodes[p], chosen, chosen_cnt);

        /* Pre-allocate pruning buffers once, reuse for all chosen neighbors. */
        int    b_max_nc = idx->R + 1;
        float *b_nds    = (float *)malloc(sizeof(float) * (size_t)b_max_nc);
        int   *b_nns    = (int   *)malloc(sizeof(int)   * (size_t)b_max_nc);
        int   *b_pruned = (int   *)malloc(sizeof(int)   * (size_t)(idx->R + 4));
        for (int i = 0; i < chosen_cnt; i++) {
            int nb = chosen[i];
            node_add_nb(&idx->nodes[nb], p);
            if (b_nds && b_nns && b_pruned && idx->nodes[nb].neighbor_cnt > idx->R) {
                int nc = idx->nodes[nb].neighbor_cnt;
                for (int j = 0; j < nc; j++) {
                    b_nns[j] = idx->nodes[nb].neighbors[j];
                    b_nds[j] = node_dist_da(idx, nb, b_nns[j]);
                }
                sort_by_dist(b_nds, b_nns, nc);
                int new_cnt = 0;
                robust_prune(idx, nb, b_nds, b_nns, nc, idx->R, idx->alpha, b_pruned, &new_cnt);
                node_set_neighbors(&idx->nodes[nb], b_pruned, new_cnt);
            }
        }
        free(b_nds); free(b_nns); free(b_pruned);
        free(ds); free(ns); free(chosen);
        heap_clear(&result); bitset_clear(&visited);
    }
    free(order);
    heap_free(&result);
    bitset_free(&visited);
    return PISTADB_OK;
}

/* ── Search ──────────────────────────────────────────────────────────────── */

int diskann_search(DiskANNIndex *idx, const float *query, int k,
                   PistaDBResult *results) {
    if (idx->n_nodes == 0 || idx->medoid < 0) return 0;

    int L = (idx->L > k) ? idx->L : k;
    Heap result; heap_init(&result, L * 2 + 8, 1);
    Bitset visited; bitset_init(&visited, idx->n_nodes + 8);

    greedy_search(idx, query, idx->medoid, L, &result, &visited);

    int total = result.size;
    int cnt   = 0;
    if (total > 0) {
        float *ds = (float *)malloc(sizeof(float) * (size_t)total);
        int   *ns = (int   *)malloc(sizeof(int)   * (size_t)total);
        if (!ds || !ns) {
            free(ds); free(ns);
            heap_free(&result); bitset_free(&visited);
            return 0;
        }
        /* Drain max-heap into ascending order. */
        for (int i = total - 1; i >= 0; i--) {
            HeapItem it = heap_pop(&result);
            ds[i] = it.key; ns[i] = (int)it.id;
        }
        for (int i = 0; i < total && cnt < k; i++) {
            if (idx->nodes[ns[i]].deleted) continue;
            results[cnt].id       = idx->nodes[ns[i]].vec_id;
            results[cnt].distance = ds[i];
            strncpy(results[cnt].label, VS_LABEL(&idx->vs, ns[i]), 255);
            results[cnt].label[255] = '\0';
            cnt++;
        }
        free(ds); free(ns);
    }
    heap_free(&result);
    bitset_free(&visited);
    return cnt;
}

/* ── Serialization ───────────────────────────────────────────────────────── */
/*
 * Header: int32 n_nodes, dim, R, L, medoid; float alpha
 * For each node:
 *   uint64 vec_id, int32 deleted, int32 neighbor_cnt
 *   float vec[dim]
 *   int32 neighbors[neighbor_cnt]
 */

int diskann_save(const DiskANNIndex *idx, void **out_buf, size_t *out_size) {
    size_t sz = sizeof(int32_t) * 5 + sizeof(float);
    for (int i = 0; i < idx->n_nodes; i++)
        sz += sizeof(uint64_t) + sizeof(int32_t) * 2 + 256
            + sizeof(float) * (size_t)idx->dim
            + sizeof(int32_t) * (size_t)idx->nodes[i].neighbor_cnt;

    uint8_t *buf = (uint8_t *)malloc(sz);
    if (!buf) return PISTADB_ENOMEM;
    uint8_t *p = buf;

#define WI32(v) do{*(int32_t*)p=(int32_t)(v);p+=4;}while(0)
#define WU64(v) do{*(uint64_t*)p=(uint64_t)(v);p+=8;}while(0)
#define WF32(v) do{*(float*)p=(float)(v);p+=4;}while(0)
    WI32(idx->n_nodes); WI32(idx->dim); WI32(idx->R); WI32(idx->L); WI32(idx->medoid);
    WF32(idx->alpha);
    for (int i = 0; i < idx->n_nodes; i++) {
        WU64(idx->nodes[i].vec_id);
        WI32(idx->nodes[i].deleted);
        WI32(idx->nodes[i].neighbor_cnt);
        memcpy(p, VS_LABEL(&idx->vs, i), 256); p += 256;
        memcpy(p, VS_VEC(&idx->vs, i), sizeof(float) * (size_t)idx->dim);
        p += sizeof(float) * (size_t)idx->dim;
        for (int j = 0; j < idx->nodes[i].neighbor_cnt; j++) WI32(idx->nodes[i].neighbors[j]);
    }
#undef WI32
#undef WU64
#undef WF32
    *out_buf  = buf;
    *out_size = (size_t)(p - buf);
    return PISTADB_OK;
}

int diskann_load(DiskANNIndex *idx, const void *buf, size_t size,
                 int dim, DistFn dist_fn) {
    /* Minimum header size: 5 int32 + 1 float = 24 bytes */
    if (size < sizeof(int32_t) * 5 + sizeof(float)) return PISTADB_ECORRUPT;
    const uint8_t *p   = (const uint8_t *)buf;
    const uint8_t *end = p + size;
#define NEED(n) do { if ((size_t)(end - p) < (size_t)(n)) return PISTADB_ECORRUPT; } while (0)
#define RI32()  (p += 4, *(const int32_t *)(p - 4))
#define RU64()  (p += 8, *(const uint64_t*)(p - 8))
#define RF32()  (p += 4, *(const float   *)(p - 4))
    int n_nodes = RI32(); int file_dim = RI32(); int R = RI32(); int L = RI32(); int medoid = RI32();
    float alpha = RF32();
    if (file_dim != dim) return PISTADB_ECORRUPT;
    if (n_nodes < 0 || R < 0 || L < 0) return PISTADB_ECORRUPT;
    if (medoid < -1 || medoid >= n_nodes) return PISTADB_ECORRUPT;

    int r = diskann_create(idx, dim, dist_fn, R, L, alpha);
    if (r) return r;
    idx->medoid = medoid;
    while (idx->node_cap < n_nodes) {
        int gr = da_grow(idx);
        if (gr != PISTADB_OK) return gr;
    }

    for (int i = 0; i < n_nodes; i++) {
        NEED(8 + 4 + 4 + 256 + (size_t)dim * sizeof(float));
        uint64_t vid = RU64();
        int del = RI32();
        int nc  = RI32();
        if (nc < 0) return PISTADB_ECORRUPT;
        idx->nodes[i].vec_id = vid;
        idx->nodes[i].deleted = del;
        idx->nodes[i].neighbor_cnt = nc;
        idx->nodes[i].neighbor_cap = nc + 4;
        idx->nodes[i].neighbors = (int *)malloc(sizeof(int) * (size_t)(nc + 4));
        if (!idx->nodes[i].neighbors) return PISTADB_ENOMEM;
        memcpy(VS_LABEL(&idx->vs, i), p, 256); p += 256;
        memcpy(VS_VEC(&idx->vs, i), p, sizeof(float) * (size_t)dim);
        p += sizeof(float) * (size_t)dim;
        NEED((size_t)nc * 4);
        for (int j = 0; j < nc; j++) {
            int nb = *(const int32_t*)p; p += 4;
            /* Refuse silent corruption — see hnsw_load for rationale. */
            if (nb < 0 || nb >= n_nodes) return PISTADB_ECORRUPT;
            idx->nodes[i].neighbors[j] = nb;
        }
        idx->n_nodes++;
    }
#undef NEED
#undef RI32
#undef RU64
#undef RF32
    return PISTADB_OK;
}
