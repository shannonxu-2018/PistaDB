/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - index_linear.c
 * Brute-force linear scan.
 */
#include "index_linear.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>

#define INITIAL_CAP 64

int linear_create(LinearIndex *idx, int dim, DistFn dist_fn, int initial_cap) {
    if (initial_cap <= 0) initial_cap = INITIAL_CAP;
    idx->dim     = dim;
    idx->dist_fn = dist_fn;
    idx->size    = 0;
    idx->cap     = initial_cap;

    idx->vectors = (float    *)malloc(sizeof(float)    * (size_t)(initial_cap * dim));
    idx->ids     = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)initial_cap);
    idx->labels  = (char (*)[256])malloc(256 * (size_t)initial_cap);
    idx->deleted = (uint8_t  *)calloc((size_t)initial_cap, 1);

    if (!idx->vectors || !idx->ids || !idx->labels || !idx->deleted) {
        free(idx->vectors); free(idx->ids); free(idx->labels); free(idx->deleted);
        return PISTADB_ENOMEM;
    }
    return PISTADB_OK;
}

void linear_free(LinearIndex *idx) {
    free(idx->vectors); free(idx->ids); free(idx->labels); free(idx->deleted);
    idx->vectors = NULL; idx->ids = NULL; idx->labels = NULL; idx->deleted = NULL;
    idx->size = idx->cap = 0;
}

static int linear_grow(LinearIndex *idx) {
    int nc = idx->cap * 2 + 8;
    float    *nv = (float    *)realloc(idx->vectors, sizeof(float)    * (size_t)(nc * idx->dim));
    uint64_t *ni = (uint64_t *)realloc(idx->ids,     sizeof(uint64_t) * (size_t)nc);
    char     (*nl)[256] = (char (*)[256])realloc(idx->labels, 256 * (size_t)nc);
    uint8_t  *nd = (uint8_t  *)realloc(idx->deleted, (size_t)nc);
    if (!nv || !ni || !nl || !nd) return PISTADB_ENOMEM;
    memset(nd + idx->cap, 0, (size_t)(nc - idx->cap));
    idx->vectors = nv; idx->ids = ni; idx->labels = nl; idx->deleted = nd;
    idx->cap = nc;
    return PISTADB_OK;
}

int linear_find_id(const LinearIndex *idx, uint64_t id) {
    for (int i = 0; i < idx->size; i++)
        if (!idx->deleted[i] && idx->ids[i] == id) return i;
    return -1;
}

int linear_insert(LinearIndex *idx, uint64_t id, const char *label, const float *vec) {
    if (idx->size == idx->cap) {
        int r = linear_grow(idx);
        if (r != PISTADB_OK) return r;
    }
    int slot = idx->size++;
    idx->ids[slot] = id;
    idx->deleted[slot] = 0;
    memcpy(idx->vectors + (size_t)slot * idx->dim, vec, sizeof(float) * (size_t)idx->dim);
    if (label) strncpy(idx->labels[slot], label, 255);
    else        idx->labels[slot][0] = '\0';
    idx->labels[slot][255] = '\0';
    return PISTADB_OK;
}

int linear_delete(LinearIndex *idx, uint64_t id) {
    int slot = linear_find_id(idx, id);
    if (slot < 0) return PISTADB_ENOTFOUND;
    idx->deleted[slot] = 1;
    return PISTADB_OK;
}

int linear_update(LinearIndex *idx, uint64_t id, const float *vec) {
    int slot = linear_find_id(idx, id);
    if (slot < 0) return PISTADB_ENOTFOUND;
    memcpy(idx->vectors + (size_t)slot * idx->dim, vec, sizeof(float) * (size_t)idx->dim);
    return PISTADB_OK;
}

/* Insertion sort into a fixed-size result buffer (max-heap by distance). */
static void result_insert(PistaDBResult *res, int *cnt, int k,
                           uint64_t id, float dist, const char *label) {
    if (*cnt < k) {
        res[*cnt].id       = id;
        res[*cnt].distance = dist;
        if (label) strncpy(res[*cnt].label, label, 255);
        else        res[*cnt].label[0] = '\0';
        res[*cnt].label[255] = '\0';
        (*cnt)++;
        /* bubble-up largest to front for easy eviction */
        for (int i = *cnt - 1; i > 0 && res[i].distance > res[i-1].distance; i--) {
            PistaDBResult tmp = res[i]; res[i] = res[i-1]; res[i-1] = tmp;
        }
    } else if (dist < res[0].distance) {
        res[0].id       = id;
        res[0].distance = dist;
        if (label) strncpy(res[0].label, label, 255);
        else        res[0].label[0] = '\0';
        res[0].label[255] = '\0';
        /* sift down */
        int pos = 0;
        for (;;) {
            int worst = pos;
            if (pos + 1 < *cnt && res[pos+1].distance > res[worst].distance) worst = pos + 1;
            if (worst == pos) break;
            PistaDBResult tmp = res[pos]; res[pos] = res[worst]; res[worst] = tmp;
            pos = worst;
        }
    }
}

int linear_search(const LinearIndex *idx, const float *query, int k,
                  PistaDBResult *results) {
    int cnt = 0;
    for (int i = 0; i < idx->size; i++) {
        if (idx->deleted[i]) continue;
        float d = idx->dist_fn(query, idx->vectors + (size_t)i * idx->dim, idx->dim);
        result_insert(results, &cnt, k, idx->ids[i], d, idx->labels[i]);
    }
    /* sort ascending by distance */
    for (int i = 0; i < cnt - 1; i++) {
        for (int j = i + 1; j < cnt; j++) {
            if (results[j].distance < results[i].distance) {
                PistaDBResult tmp = results[i]; results[i] = results[j]; results[j] = tmp;
            }
        }
    }
    return cnt;
}

/* ── Serialization ───────────────────────────────────────────────────────── */
/*
 * Layout:
 *   int32  size
 *   int32  dim
 *   For each entry:
 *     uint64 id
 *     uint8  deleted
 *     char   label[256]
 *     float  vec[dim]
 */

int linear_save(const LinearIndex *idx, void **out_buf, size_t *out_size) {
    size_t entry = sizeof(uint64_t) + 1 + 256 + sizeof(float) * (size_t)idx->dim;
    size_t total = sizeof(int32_t) * 2 + (size_t)idx->size * entry;
    uint8_t *buf = (uint8_t *)malloc(total);
    if (!buf) return PISTADB_ENOMEM;

    uint8_t *p = buf;
    *(int32_t *)p = (int32_t)idx->size; p += 4;
    *(int32_t *)p = (int32_t)idx->dim;  p += 4;

    for (int i = 0; i < idx->size; i++) {
        *(uint64_t *)p = idx->ids[i];    p += 8;
        *p++ = idx->deleted[i];
        memcpy(p, idx->labels[i], 256);  p += 256;
        memcpy(p, idx->vectors + (size_t)i * idx->dim, sizeof(float) * (size_t)idx->dim);
        p += sizeof(float) * (size_t)idx->dim;
    }
    *out_buf  = buf;
    *out_size = total;
    return PISTADB_OK;
}

int linear_load(LinearIndex *idx, const void *buf, size_t size,
                int dim, DistFn dist_fn) {
    const uint8_t *p = (const uint8_t *)buf;
    if (size < 8) return PISTADB_ECORRUPT;

    int32_t count = *(const int32_t *)p; p += 4;
    int32_t fdim  = *(const int32_t *)p; p += 4;
    if (fdim != dim) return PISTADB_ECORRUPT;

    int r = linear_create(idx, dim, dist_fn, count + 8);
    if (r != PISTADB_OK) return r;

    size_t entry = sizeof(uint64_t) + 1 + 256 + sizeof(float) * (size_t)dim;
    if (size < 8 + (size_t)count * entry) return PISTADB_ECORRUPT;

    for (int i = 0; i < count; i++) {
        uint64_t id  = *(const uint64_t *)p; p += 8;
        uint8_t  del = *p++;
        const char *label = (const char *)p; p += 256;
        const float *vec  = (const float *)p; p += sizeof(float) * (size_t)dim;

        r = linear_insert(idx, id, label, vec);
        if (r != PISTADB_OK) return r;
        idx->deleted[idx->size - 1] = del;
    }
    return PISTADB_OK;
}
