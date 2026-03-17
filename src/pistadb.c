/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - pistadb.c
 * Main database implementation.
 */
#include "pistadb.h"
#include "storage.h"
#include "distance.h"
#include "index_linear.h"
#include "index_hnsw.h"
#include "index_ivf.h"
#include "index_ivf_pq.h"
#include "index_diskann.h"
#include "index_lsh.h"
#include "index_scann.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Internal state ──────────────────────────────────────────────────────── */

struct PistaDB {
    char            path[4096];
    int             dim;
    PistaDBMetric    metric;
    PistaDBIndexType index_type;
    PistaDBParams    params;
    DistFn          dist_fn;
    uint64_t        next_id;       /* auto-increment (unused if caller supplies id) */
    char            last_err[512];

    /* Exactly one of these is active */
    union {
        LinearIndex  linear;
        HNSWIndex    hnsw;
        IVFIndex     ivf;
        IVFPQIndex   ivfpq;
        DiskANNIndex diskann;
        LSHIndex     lsh;
        ScaNNIndex   scann;
    } idx;

    /* Total vector count (including deleted) */
    uint64_t n_total;
};

/* ── Error helper ────────────────────────────────────────────────────────── */

static void set_err(PistaDB *db, int code, const char *msg) {
    (void)code;
    if (msg) strncpy(db->last_err, msg, sizeof(db->last_err) - 1);
    else db->last_err[0] = '\0';
}

static const char *err_str(int code) {
    switch (code) {
        case PISTADB_OK:         return "OK";
        case PISTADB_ERR:        return "generic error";
        case PISTADB_ENOMEM:     return "out of memory";
        case PISTADB_EIO:        return "I/O error";
        case PISTADB_ENOTFOUND:  return "vector not found";
        case PISTADB_EEXIST:     return "vector id already exists";
        case PISTADB_EINVAL:     return "invalid argument";
        case PISTADB_ENOTRAINED: return "index not trained";
        case PISTADB_ECORRUPT:   return "file corrupted";
        case PISTADB_EVERSION:   return "incompatible file version";
        default:                return "unknown error";
    }
}

/* ── Create new empty index ──────────────────────────────────────────────── */

static int create_index(PistaDB *db) {
    DistFn fn  = pistadb_get_dist_fn(db->metric);
    PistaDBParams *p = &db->params;
    int r;

    switch (db->index_type) {
        case INDEX_LINEAR:
            r = linear_create(&db->idx.linear, db->dim, fn, 64);
            break;
        case INDEX_HNSW:
            r = hnsw_create(&db->idx.hnsw, db->dim, fn,
                            p->hnsw_M, p->hnsw_ef_construction, p->hnsw_ef_search);
            break;
        case INDEX_IVF:
            r = ivf_create(&db->idx.ivf, db->dim, fn,
                           p->ivf_nlist, p->ivf_nprobe);
            break;
        case INDEX_IVF_PQ:
            r = ivfpq_create(&db->idx.ivfpq, db->dim, fn,
                             p->ivf_nlist, p->ivf_nprobe,
                             p->pq_M, p->pq_nbits);
            break;
        case INDEX_DISKANN:
            r = diskann_create(&db->idx.diskann, db->dim, fn,
                               p->diskann_R, p->diskann_L, p->diskann_alpha);
            break;
        case INDEX_LSH:
            r = lsh_create(&db->idx.lsh, db->dim, fn, db->metric,
                           p->lsh_L, p->lsh_K, p->lsh_w);
            break;
        case INDEX_SCANN:
            r = scann_create(&db->idx.scann, db->dim, fn, db->metric,
                             p->scann_nlist, p->scann_nprobe,
                             p->scann_pq_M, p->scann_pq_bits,
                             p->scann_rerank_k, p->scann_aq_eta);
            break;
        default:
            r = PISTADB_EINVAL;
    }
    return r;
}

/* ── Load from file ──────────────────────────────────────────────────────── */

static int load_from_file(PistaDB *db) {
    PistaDBFileHeader hdr;
    int r = storage_read_header(db->path, &hdr);
    if (r != PISTADB_OK) return r;

    if ((int)hdr.dimension != db->dim)    return PISTADB_EINVAL;
    db->metric     = (PistaDBMetric)hdr.metric_type;
    db->index_type = (PistaDBIndexType)hdr.index_type;
    db->next_id    = hdr.next_id;
    db->n_total    = hdr.num_vectors;
    db->dist_fn    = pistadb_get_dist_fn(db->metric);

    void *vec_buf = NULL, *idx_buf = NULL;
    size_t vec_sz = 0, idx_sz = 0;
    r = storage_read_sections(db->path, &hdr, &vec_buf, &vec_sz, &idx_buf, &idx_sz);
    if (r != PISTADB_OK) return r;

    DistFn fn = db->dist_fn;
    switch (db->index_type) {
        case INDEX_LINEAR:
            r = linear_load(&db->idx.linear, vec_buf, vec_sz, db->dim, fn);
            break;
        case INDEX_HNSW:
            r = hnsw_load(&db->idx.hnsw, vec_buf, vec_sz, db->dim, fn);
            break;
        case INDEX_IVF:
            r = ivf_load(&db->idx.ivf, vec_buf, vec_sz, db->dim, fn);
            break;
        case INDEX_IVF_PQ:
            r = ivfpq_load(&db->idx.ivfpq, vec_buf, vec_sz, db->dim, fn);
            break;
        case INDEX_DISKANN:
            r = diskann_load(&db->idx.diskann, idx_buf, idx_sz, db->dim, fn);
            /* vec section not used for diskann (vectors embedded in nodes) */
            if (r != PISTADB_OK) { r = diskann_load(&db->idx.diskann, vec_buf, vec_sz, db->dim, fn); }
            break;
        case INDEX_LSH:
            r = lsh_load(&db->idx.lsh, vec_buf, vec_sz, db->dim, fn, db->metric);
            break;
        case INDEX_SCANN:
            r = scann_load(&db->idx.scann, vec_buf, vec_sz, db->dim, fn, db->metric);
            break;
        default:
            r = PISTADB_EINVAL;
    }
    free(vec_buf);
    free(idx_buf);
    return r;
}

/* ── pistadb_open ─────────────────────────────────────────────────────────── */

PistaDB *pistadb_open(const char *path, int dim,
                    PistaDBMetric metric, PistaDBIndexType index,
                    const PistaDBParams *params) {
    PistaDB *db = (PistaDB *)calloc(1, sizeof(PistaDB));
    if (!db) return NULL;

    strncpy(db->path, path, sizeof(db->path) - 1);
    db->dim        = dim;
    db->metric     = metric;
    db->index_type = index;
    db->params     = params ? *params : pistadb_default_params();
    db->dist_fn    = pistadb_get_dist_fn(metric);
    db->next_id    = 1;
    db->n_total    = 0;

    /* Try to load existing file first */
    FILE *check = fopen(path, "rb");
    if (check) {
        fclose(check);
        int r = load_from_file(db);
        if (r == PISTADB_OK) return db;
        /* Fall through: create fresh (file may be new/empty) */
    }

    /* Create fresh index */
    int r = create_index(db);
    if (r != PISTADB_OK) {
        set_err(db, r, err_str(r));
        free(db);
        return NULL;
    }
    return db;
}

void pistadb_close(PistaDB *db) {
    if (!db) return;
    switch (db->index_type) {
        case INDEX_LINEAR:  linear_free(&db->idx.linear);   break;
        case INDEX_HNSW:    hnsw_free(&db->idx.hnsw);       break;
        case INDEX_IVF:     ivf_free(&db->idx.ivf);         break;
        case INDEX_IVF_PQ:  ivfpq_free(&db->idx.ivfpq);     break;
        case INDEX_DISKANN: diskann_free(&db->idx.diskann);  break;
        case INDEX_LSH:     lsh_free(&db->idx.lsh);         break;
        case INDEX_SCANN:   scann_free(&db->idx.scann);     break;
    }
    free(db);
}

/* ── pistadb_save ─────────────────────────────────────────────────────────── */

int pistadb_save(PistaDB *db) {
    void *vec_buf = NULL, *idx_buf = NULL;
    size_t vec_sz = 0, idx_sz = 0;
    int r = PISTADB_OK;

    /* For most indices, the "vec section" holds everything; idx section is empty. */
    /* For HNSW, vec section holds serialised graph (includes vectors). */
    /* We use a tiny placeholder for the empty section. */
    uint8_t placeholder = 0;

    switch (db->index_type) {
        case INDEX_LINEAR:
            r = linear_save(&db->idx.linear, &vec_buf, &vec_sz);
            idx_buf = &placeholder; idx_sz = 1;
            break;
        case INDEX_HNSW:
            r = hnsw_save(&db->idx.hnsw, &vec_buf, &vec_sz);
            idx_buf = &placeholder; idx_sz = 1;
            break;
        case INDEX_IVF:
            r = ivf_save(&db->idx.ivf, &vec_buf, &vec_sz);
            idx_buf = &placeholder; idx_sz = 1;
            break;
        case INDEX_IVF_PQ:
            r = ivfpq_save(&db->idx.ivfpq, &vec_buf, &vec_sz);
            idx_buf = &placeholder; idx_sz = 1;
            break;
        case INDEX_DISKANN:
            r = diskann_save(&db->idx.diskann, &vec_buf, &vec_sz);
            idx_buf = &placeholder; idx_sz = 1;
            break;
        case INDEX_LSH:
            r = lsh_save(&db->idx.lsh, &vec_buf, &vec_sz);
            idx_buf = &placeholder; idx_sz = 1;
            break;
        case INDEX_SCANN:
            r = scann_save(&db->idx.scann, &vec_buf, &vec_sz);
            idx_buf = &placeholder; idx_sz = 1;
            break;
        default:
            return PISTADB_EINVAL;
    }
    if (r != PISTADB_OK) return r;

    r = storage_write(db->path,
                      db->metric, db->index_type,
                      (uint32_t)db->dim, db->n_total, db->next_id,
                      vec_buf, vec_sz,
                      idx_buf, idx_sz);
    if (vec_buf != &placeholder) free(vec_buf);
    return r;
}

/* ── CRUD ────────────────────────────────────────────────────────────────── */

int pistadb_insert(PistaDB *db, uint64_t id, const char *label, const float *vec) {
    int r;
    switch (db->index_type) {
        case INDEX_LINEAR:  r = linear_insert(&db->idx.linear, id, label, vec);  break;
        case INDEX_HNSW:    r = hnsw_insert(&db->idx.hnsw, id, label, vec);      break;
        case INDEX_IVF:     r = ivf_insert(&db->idx.ivf, id, label, vec);        break;
        case INDEX_IVF_PQ:  r = ivfpq_insert(&db->idx.ivfpq, id, label, vec);   break;
        case INDEX_DISKANN: r = diskann_insert(&db->idx.diskann, id, label, vec); break;
        case INDEX_LSH:     r = lsh_insert(&db->idx.lsh, id, label, vec);        break;
        case INDEX_SCANN:   r = scann_insert(&db->idx.scann, id, label, vec);   break;
        default:            r = PISTADB_EINVAL;
    }
    if (r == PISTADB_OK) { db->n_total++; if (id >= db->next_id) db->next_id = id + 1; }
    else set_err(db, r, err_str(r));
    return r;
}

int pistadb_delete(PistaDB *db, uint64_t id) {
    int r;
    switch (db->index_type) {
        case INDEX_LINEAR:  r = linear_delete(&db->idx.linear, id);   break;
        case INDEX_HNSW:    r = hnsw_delete(&db->idx.hnsw, id);       break;
        case INDEX_IVF:     r = ivf_delete(&db->idx.ivf, id);         break;
        case INDEX_IVF_PQ:  r = ivfpq_delete(&db->idx.ivfpq, id);    break;
        case INDEX_DISKANN: r = diskann_delete(&db->idx.diskann, id);  break;
        case INDEX_LSH:     r = lsh_delete(&db->idx.lsh, id);         break;
        case INDEX_SCANN:   r = scann_delete(&db->idx.scann, id);     break;
        default:            r = PISTADB_EINVAL;
    }
    if (r != PISTADB_OK) set_err(db, r, err_str(r));
    return r;
}

int pistadb_update(PistaDB *db, uint64_t id, const float *vec) {
    int r;
    switch (db->index_type) {
        case INDEX_LINEAR:  r = linear_update(&db->idx.linear, id, vec);   break;
        case INDEX_HNSW:    r = hnsw_update(&db->idx.hnsw, id, vec);       break;
        case INDEX_IVF:     r = ivf_update(&db->idx.ivf, id, vec);         break;
        case INDEX_IVF_PQ:  r = ivfpq_update(&db->idx.ivfpq, id, vec);    break;
        case INDEX_DISKANN: r = diskann_update(&db->idx.diskann, id, vec);  break;
        case INDEX_LSH:     r = lsh_update(&db->idx.lsh, id, vec);         break;
        case INDEX_SCANN:   r = scann_update(&db->idx.scann, id, vec);     break;
        default:            r = PISTADB_EINVAL;
    }
    if (r != PISTADB_OK) set_err(db, r, err_str(r));
    return r;
}

int pistadb_get(PistaDB *db, uint64_t id, float *out_vec, char *out_label) {
    /* Only LINEAR supports direct get; others fall back to it */
    if (db->index_type == INDEX_LINEAR) {
        int slot = linear_find_id(&db->idx.linear, id);
        if (slot < 0) return PISTADB_ENOTFOUND;
        if (out_vec) memcpy(out_vec, db->idx.linear.vectors + (size_t)slot * db->dim,
                            sizeof(float) * (size_t)db->dim);
        if (out_label) { strncpy(out_label, db->idx.linear.labels[slot], 255); out_label[255] = '\0'; }
        return PISTADB_OK;
    }
    /* For other indices, a generic "get" would require a linear scan of their internal stores.
       Implement the most common case: IVF and LSH store raw vectors. */
    if (db->index_type == INDEX_IVF) {
        IVFIndex *iv = &db->idx.ivf;
        for (int i = 0; i < iv->n_vecs; i++) {
            if (iv->vec_ids[i] == id && !iv->vec_deleted[i]) {
                if (out_vec) memcpy(out_vec, iv->vectors + (size_t)i * db->dim,
                                    sizeof(float) * (size_t)db->dim);
                if (out_label) { strncpy(out_label, iv->vec_labels[i], 255); out_label[255] = '\0'; }
                return PISTADB_OK;
            }
        }
    }
    if (db->index_type == INDEX_LSH) {
        LSHIndex *ls = &db->idx.lsh;
        for (int i = 0; i < ls->n_vecs; i++) {
            if (ls->vec_ids[i] == id && !ls->vec_deleted[i]) {
                if (out_vec) memcpy(out_vec, ls->vectors + (size_t)i * db->dim,
                                    sizeof(float) * (size_t)db->dim);
                if (out_label) { strncpy(out_label, ls->vec_labels[i], 255); out_label[255] = '\0'; }
                return PISTADB_OK;
            }
        }
    }
    if (db->index_type == INDEX_HNSW) {
        HNSWIndex *h = &db->idx.hnsw;
        for (int i = 0; i < h->n_nodes; i++) {
            if (h->nodes[i].vec_id == id && h->nodes[i].vec_id != UINT64_MAX) {
                if (out_vec) memcpy(out_vec, h->vectors + (size_t)i * db->dim,
                                    sizeof(float) * (size_t)db->dim);
                if (out_label) { strncpy(out_label, h->labels[i], 255); out_label[255] = '\0'; }
                return PISTADB_OK;
            }
        }
    }
    if (db->index_type == INDEX_DISKANN) {
        DiskANNIndex *d = &db->idx.diskann;
        for (int i = 0; i < d->n_nodes; i++) {
            if (d->nodes[i].vec_id == id && !d->nodes[i].deleted) {
                if (out_vec) memcpy(out_vec, d->vectors + (size_t)i * db->dim,
                                    sizeof(float) * (size_t)db->dim);
                if (out_label) { strncpy(out_label, d->labels[i], 255); out_label[255] = '\0'; }
                return PISTADB_OK;
            }
        }
    }
    return PISTADB_ENOTFOUND;
}

/* ── Search ──────────────────────────────────────────────────────────────── */

int pistadb_search(PistaDB *db, const float *query, int k,
                  PistaDBResult *results) {
    int r;
    switch (db->index_type) {
        case INDEX_LINEAR:
            r = linear_search(&db->idx.linear, query, k, results);
            break;
        case INDEX_HNSW:
            r = hnsw_search(&db->idx.hnsw, query, k,
                            db->params.hnsw_ef_search, results);
            break;
        case INDEX_IVF:
            r = ivf_search(&db->idx.ivf, query, k, results);
            break;
        case INDEX_IVF_PQ:
            r = ivfpq_search(&db->idx.ivfpq, query, k, results);
            break;
        case INDEX_DISKANN:
            r = diskann_search(&db->idx.diskann, query, k, results);
            break;
        case INDEX_LSH:
            r = lsh_search(&db->idx.lsh, query, k, results);
            break;
        case INDEX_SCANN:
            r = scann_search(&db->idx.scann, query, k, results);
            break;
        default:
            r = 0;
    }
    return r;
}

/* ── Index management ────────────────────────────────────────────────────── */

int pistadb_train(PistaDB *db) {
    int r = PISTADB_OK;
    switch (db->index_type) {
        case INDEX_IVF: {
            IVFIndex *iv = &db->idx.ivf;
            /* Train on currently stored vectors */
            if (iv->n_vecs == 0) return PISTADB_EINVAL;
            r = ivf_train(iv, iv->vectors, iv->n_vecs, 100);
            break;
        }
        case INDEX_IVF_PQ: {
            IVFPQIndex *pq = &db->idx.ivfpq;
            /* We don't have a flat vector store here before insert.
               Caller must call pistadb_train with training data via the Python API. */
            (void)pq;
            break;
        }
        case INDEX_DISKANN:
            r = diskann_build(&db->idx.diskann);
            break;
        case INDEX_SCANN:
            /* ScaNN must be trained via pistadb_train_on before inserting */
            break;
        default:
            break;  /* LINEAR, HNSW, LSH don't need explicit training */
    }
    if (r != PISTADB_OK) set_err(db, r, err_str(r));
    return r;
}

/* ── Metadata ────────────────────────────────────────────────────────────── */

int pistadb_count(PistaDB *db) {
    switch (db->index_type) {
        case INDEX_LINEAR: {
            int cnt = 0;
            for (int i = 0; i < db->idx.linear.size; i++)
                if (!db->idx.linear.deleted[i]) cnt++;
            return cnt;
        }
        case INDEX_IVF:    {
            int cnt = 0;
            for (int i = 0; i < db->idx.ivf.n_vecs; i++)
                if (!db->idx.ivf.vec_deleted[i]) cnt++;
            return cnt;
        }
        case INDEX_LSH:    {
            int cnt = 0;
            for (int i = 0; i < db->idx.lsh.n_vecs; i++)
                if (!db->idx.lsh.vec_deleted[i]) cnt++;
            return cnt;
        }
        case INDEX_HNSW: {
            int cnt = 0;
            for (int i = 0; i < db->idx.hnsw.n_nodes; i++)
                if (db->idx.hnsw.nodes[i].vec_id != UINT64_MAX) cnt++;
            return cnt;
        }
        case INDEX_DISKANN: {
            int cnt = 0;
            for (int i = 0; i < db->idx.diskann.n_nodes; i++)
                if (!db->idx.diskann.nodes[i].deleted) cnt++;
            return cnt;
        }
        case INDEX_SCANN: {
            int cnt = 0;
            for (int i = 0; i < db->idx.scann.n_vecs; i++)
                if (!db->idx.scann.all_deleted[i]) cnt++;
            return cnt;
        }
        default: return (int)db->n_total;
    }
}

int              pistadb_dim(PistaDB *db)        { return db->dim; }
PistaDBMetric     pistadb_metric(PistaDB *db)     { return db->metric; }
PistaDBIndexType  pistadb_index_type(PistaDB *db) { return db->index_type; }
const char      *pistadb_last_error(PistaDB *db) { return db->last_err; }
const char      *pistadb_version(void)          { return "1.0.0"; }

/* ── Exported C helpers for Python ctypes ────────────────────────────────── */
/* These thin wrappers expose a flat C ABI suitable for ctypes. */

/** Train IVF / IVF_PQ on external data (used from Python). */
int pistadb_train_on(PistaDB *db, const float *train_vecs, int n_train) {
    int r = PISTADB_OK;
    switch (db->index_type) {
        case INDEX_IVF:
            r = ivf_train(&db->idx.ivf, train_vecs, n_train, 100);
            break;
        case INDEX_IVF_PQ:
            r = ivfpq_train(&db->idx.ivfpq, train_vecs, n_train, 30);
            break;
        case INDEX_SCANN:
            r = scann_train(&db->idx.scann, train_vecs, n_train, 100);
            break;
        default:
            break;
    }
    return r;
}

/** Serialise a PistaDBResult array into a flat buffer for Python:
 *  [n × (uint64 id, float dist, char label[256])]
 *  Caller must free the returned buffer. */
void *pistadb_results_to_buf(const PistaDBResult *res, int n, int *out_size) {
    size_t entry = sizeof(uint64_t) + sizeof(float) + 256;
    uint8_t *buf = (uint8_t *)malloc(entry * (size_t)n);
    if (!buf) { *out_size = 0; return NULL; }
    for (int i = 0; i < n; i++) {
        uint8_t *p = buf + (size_t)i * entry;
        *(uint64_t *)p = res[i].id;        p += 8;
        *(float    *)p = res[i].distance;  p += 4;
        memcpy(p, res[i].label, 256);
    }
    *out_size = (int)(entry * (size_t)n);
    return buf;
}

void pistadb_free_buf(void *buf) { free(buf); }
