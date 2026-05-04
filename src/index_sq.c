/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - index_sq.c
 * Scalar Quantization (SQ8) index implementation.
 *
 * Stores vectors as uint8 codes with per-dimension min/max ranges.
 * On insert: updates running min/max, quantizes to uint8.
 * On search: quantizes query, computes L2 on uint8 (or dequantizes for other metrics).
 * On get: dequantizes back to float32.
 * On save: stores uint8 codes (4x compression vs float32).
 */
#include "index_sq.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#define SQ_INITIAL_CAP 64

/* ── Quantize / dequantize helpers ──────────────────────────────────────── */

static inline uint8_t sq_quantize_val(float val, float vmin, float vmax) {
    float range = vmax - vmin;
    if (range < 1e-10f) return 128;
    float norm = (val - vmin) / range;
    if (norm < 0.0f) norm = 0.0f;
    if (norm > 1.0f) norm = 1.0f;
    return (uint8_t)(norm * 255.0f + 0.5f);
}

static inline float sq_dequantize_val(uint8_t code, float vmin, float vmax) {
    return vmin + (float)code * (vmax - vmin) / 255.0f;
}

static void sq_quantize_vec(const float *vec, uint8_t *out,
                            const float *vmin, const float *vmax, int dim) {
    for (int d = 0; d < dim; d++)
        out[d] = sq_quantize_val(vec[d], vmin[d], vmax[d]);
}

static void sq_dequantize_vec(const uint8_t *codes, float *out,
                              const float *vmin, const float *vmax, int dim) {
    for (int d = 0; d < dim; d++)
        out[d] = sq_dequantize_val(codes[d], vmin[d], vmax[d]);
}

/* ── Update min/max stats with a new vector ─────────────────────────────── */
/* Returns 1 if the per-dimension range was widened by this vector, 0 if it
 * fits inside the existing bounds.  Callers use this to decide whether
 * previously-stored codes must be re-quantised against the new range. */
static int sq_update_stats(SQIndex *idx, const float *vec) {
    int changed = 0;
    for (int d = 0; d < idx->dim; d++) {
        if (vec[d] < idx->vmin[d]) { idx->vmin[d] = vec[d]; changed = 1; }
        if (vec[d] > idx->vmax[d]) { idx->vmax[d] = vec[d]; changed = 1; }
    }
    return changed;
}

/* ── Re-quantize all stored vectors with the current min/max ────────────── */
/* The naïve scheme stores 8-bit codes referenced to per-dim [vmin,vmax].
 * When the range widens, codes produced earlier are scaled wrong (they
 * compress the tighter old range into 0..255).  Without a true raw-vector
 * snapshot we can only approximate the original floats by dequantising
 * with the OLD range, then re-quantising with the NEW range — reasonable
 * but not lossless.  Caller must pass the previous range arrays. */
static void sq_requantize_all(SQIndex *idx,
                              const float *old_vmin, const float *old_vmax) {
    for (int i = 0; i < idx->size; i++) {
        if (idx->deleted[i]) continue;
        uint8_t *codes = idx->codes + (size_t)i * (size_t)idx->dim;
        for (int d = 0; d < idx->dim; d++) {
            /* Decode with the old range */
            float val = sq_dequantize_val(codes[d], old_vmin[d], old_vmax[d]);
            /* Encode with the new range */
            codes[d] = sq_quantize_val(val, idx->vmin[d], idx->vmax[d]);
        }
    }
}

/* ── Lifecycle ──────────────────────────────────────────────────────────── */

int sq_create(SQIndex *idx, int dim, DistFn dist_fn, int initial_cap) {
    if (initial_cap <= 0) initial_cap = SQ_INITIAL_CAP;
    idx->dim     = dim;
    idx->dist_fn = dist_fn;
    idx->size    = 0;
    idx->cap     = initial_cap;

    idx->codes   = (uint8_t  *)malloc(sizeof(uint8_t) * (size_t)initial_cap * (size_t)dim);
    idx->ids     = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)initial_cap);
    idx->deleted = (uint8_t  *)calloc((size_t)initial_cap, 1);
    idx->vmin    = (float    *)malloc(sizeof(float) * (size_t)dim);
    idx->vmax    = (float    *)malloc(sizeof(float) * (size_t)dim);

    if (!idx->codes || !idx->ids || !idx->deleted || !idx->vmin || !idx->vmax) {
        free(idx->codes); free(idx->ids); free(idx->deleted);
        free(idx->vmin); free(idx->vmax);
        return PISTADB_ENOMEM;
    }

    /* Initialize min/max to extreme values */
    for (int d = 0; d < dim; d++) {
        idx->vmin[d] =  FLT_MAX;
        idx->vmax[d] = -FLT_MAX;
    }

    if (vs_init(&idx->vs, 0, initial_cap) != PISTADB_OK) return PISTADB_ENOMEM;
    return PISTADB_OK;
}

void sq_free(SQIndex *idx) {
    vs_free(&idx->vs);
    free(idx->codes); free(idx->ids); free(idx->deleted);
    free(idx->vmin); free(idx->vmax);
    idx->codes = NULL; idx->ids = NULL; idx->deleted = NULL;
    idx->vmin = NULL; idx->vmax = NULL;
    idx->size = idx->cap = 0;
}

static int sq_grow(SQIndex *idx) {
    int nc = idx->cap * 2 + 8;
    int r = vs_ensure(&idx->vs, nc);
    if (r != PISTADB_OK) return r;

    /* Commit each realloc result individually so a later failure does not
     * dangle the not-yet-assigned pointer. */
    uint8_t *new_codes = (uint8_t *)realloc(
        idx->codes, sizeof(uint8_t) * (size_t)nc * (size_t)idx->dim);
    if (!new_codes) return PISTADB_ENOMEM;
    idx->codes = new_codes;

    uint64_t *new_ids = (uint64_t *)realloc(idx->ids, sizeof(uint64_t) * (size_t)nc);
    if (!new_ids) return PISTADB_ENOMEM;
    idx->ids = new_ids;

    uint8_t *new_del = (uint8_t *)realloc(idx->deleted, (size_t)nc);
    if (!new_del) return PISTADB_ENOMEM;
    idx->deleted = new_del;

    memset(new_del + idx->cap, 0, (size_t)(nc - idx->cap));
    idx->cap = nc;
    return PISTADB_OK;
}

/* ── CRUD ───────────────────────────────────────────────────────────────── */

int sq_insert(SQIndex *idx, uint64_t id, const char *label, const float *vec) {
    if (idx->size == idx->cap) {
        int r = sq_grow(idx);
        if (r != PISTADB_OK) return r;
    }

    /* Cheap pre-check: only snapshot + requantise when the new vector actually
     * widens the per-dimension range.  In the steady state most inserts fall
     * inside the bounds, so this avoids 2*dim*4 bytes of malloc per call. */
    int will_widen = 0;
    if (idx->size > 0) {
        for (int d = 0; d < idx->dim; d++) {
            if (vec[d] < idx->vmin[d] || vec[d] > idx->vmax[d]) { will_widen = 1; break; }
        }
    }

    if (will_widen) {
        float *old_vmin = (float *)malloc(sizeof(float) * (size_t)idx->dim);
        float *old_vmax = (float *)malloc(sizeof(float) * (size_t)idx->dim);
        if (!old_vmin || !old_vmax) { free(old_vmin); free(old_vmax); return PISTADB_ENOMEM; }
        memcpy(old_vmin, idx->vmin, sizeof(float) * (size_t)idx->dim);
        memcpy(old_vmax, idx->vmax, sizeof(float) * (size_t)idx->dim);
        sq_update_stats(idx, vec);
        sq_requantize_all(idx, old_vmin, old_vmax);
        free(old_vmin); free(old_vmax);
    } else {
        /* No requantise needed — but still update stats so the very first
         * insert (size == 0) initialises vmin/vmax from the vector itself. */
        sq_update_stats(idx, vec);
    }

    int slot = idx->size++;
    idx->ids[slot] = id;
    idx->deleted[slot] = 0;

    /* Quantize and store */
    sq_quantize_vec(vec, idx->codes + (size_t)slot * (size_t)idx->dim,
                    idx->vmin, idx->vmax, idx->dim);

    /* Store label */
    if (label) strncpy(VS_LABEL(&idx->vs, slot), label, 255);
    else        VS_LABEL(&idx->vs, slot)[0] = '\0';
    VS_LABEL(&idx->vs, slot)[255] = '\0';

    return PISTADB_OK;
}

int sq_delete(SQIndex *idx, uint64_t id) {
    for (int i = 0; i < idx->size; i++) {
        if (!idx->deleted[i] && idx->ids[i] == id) {
            idx->deleted[i] = 1;
            return PISTADB_OK;
        }
    }
    return PISTADB_ENOTFOUND;
}

int sq_update(SQIndex *idx, uint64_t id, const float *vec) {
    for (int i = 0; i < idx->size; i++) {
        if (!idx->deleted[i] && idx->ids[i] == id) {
            /* Same cheap pre-check as sq_insert: only snapshot + requantise
             * when the replacement vector actually widens the bounds. */
            int will_widen = 0;
            for (int d = 0; d < idx->dim; d++) {
                if (vec[d] < idx->vmin[d] || vec[d] > idx->vmax[d]) { will_widen = 1; break; }
            }
            if (will_widen) {
                float *old_vmin = (float *)malloc(sizeof(float) * (size_t)idx->dim);
                float *old_vmax = (float *)malloc(sizeof(float) * (size_t)idx->dim);
                if (!old_vmin || !old_vmax) { free(old_vmin); free(old_vmax); return PISTADB_ENOMEM; }
                memcpy(old_vmin, idx->vmin, sizeof(float) * (size_t)idx->dim);
                memcpy(old_vmax, idx->vmax, sizeof(float) * (size_t)idx->dim);
                sq_update_stats(idx, vec);
                sq_requantize_all(idx, old_vmin, old_vmax);
                free(old_vmin); free(old_vmax);
            }

            sq_quantize_vec(vec, idx->codes + (size_t)i * (size_t)idx->dim,
                            idx->vmin, idx->vmax, idx->dim);
            return PISTADB_OK;
        }
    }
    return PISTADB_ENOTFOUND;
}

/* ── Search ─────────────────────────────────────────────────────────────── */

static void sq_result_insert(PistaDBResult *res, int *cnt, int k,
                              uint64_t id, float dist, const char *label) {
    if (*cnt < k) {
        res[*cnt].id       = id;
        res[*cnt].distance = dist;
        if (label) strncpy(res[*cnt].label, label, 255);
        else        res[*cnt].label[0] = '\0';
        res[*cnt].label[255] = '\0';
        (*cnt)++;
        for (int i = *cnt - 1; i > 0 && res[i].distance > res[i-1].distance; i--) {
            PistaDBResult tmp = res[i]; res[i] = res[i-1]; res[i-1] = tmp;
        }
    } else if (dist < res[0].distance) {
        res[0].id       = id;
        res[0].distance = dist;
        if (label) strncpy(res[0].label, label, 255);
        else        res[0].label[0] = '\0';
        res[0].label[255] = '\0';
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

/* qsort comparator: ascending by distance. */
static int sq_cmp_result(const void *a, const void *b) {
    float da = ((const PistaDBResult *)a)->distance;
    float db = ((const PistaDBResult *)b)->distance;
    return (da > db) - (da < db);
}

int sq_search(const SQIndex *idx, const float *query, int k,
              PistaDBResult *results) {
    if (idx->size == 0 || k <= 0) return 0;

    /* Fast path for the L2 family: ranking on integer codes is monotone with
     * the true L2² distance only when every dimension shares the same scale,
     * which is not the case for per-dim min/max.  Be safe and dequantise into
     * a scratch buffer, then use the distance function the caller asked for.
     * 4-byte temp per query — negligible.  */
    float *cand = (float *)malloc(sizeof(float) * (size_t)idx->dim);
    if (!cand) return 0;

    DistFn dfn = idx->dist_fn ? idx->dist_fn : dist_l2;
    int cnt = 0;
    for (int i = 0; i < idx->size; i++) {
        if (idx->deleted[i]) continue;
        const uint8_t *codes = idx->codes + (size_t)i * (size_t)idx->dim;
        sq_dequantize_vec(codes, cand, idx->vmin, idx->vmax, idx->dim);
        float d = dfn(query, cand, idx->dim);
        sq_result_insert(results, &cnt, k, idx->ids[i], d, VS_LABEL(&idx->vs, i));
    }
    free(cand);

    /* O(k log k) final sort; the partial-result heap above only kept the top
     * k as a max-heap-like array, so we still need to order ascending. */
    if (cnt > 1) qsort(results, (size_t)cnt, sizeof(PistaDBResult), sq_cmp_result);
    return cnt;
}

/* ── Serialization ──────────────────────────────────────────────────────── */
/*
 * Layout:
 *   int32  size
 *   int32  dim
 *   float  vmin[dim]
 *   float  vmax[dim]
 *   For each entry:
 *     uint64 id
 *     uint8  deleted
 *     char   label[256]
 *     uint8  codes[dim]   (NOT float32 - 4x compression)
 */

int sq_save(const SQIndex *idx, void **out_buf, size_t *out_size) {
    size_t hdr_sz  = sizeof(int32_t) * 2 + sizeof(float) * (size_t)idx->dim * 2;
    size_t entry   = sizeof(uint64_t) + 1 + 256 + (size_t)idx->dim;
    size_t total   = hdr_sz + (size_t)idx->size * entry;

    uint8_t *buf = (uint8_t *)malloc(total);
    if (!buf) return PISTADB_ENOMEM;
    uint8_t *p = buf;

    *(int32_t *)p = (int32_t)idx->size; p += 4;
    *(int32_t *)p = (int32_t)idx->dim;  p += 4;

    memcpy(p, idx->vmin, sizeof(float) * (size_t)idx->dim);
    p += sizeof(float) * (size_t)idx->dim;
    memcpy(p, idx->vmax, sizeof(float) * (size_t)idx->dim);
    p += sizeof(float) * (size_t)idx->dim;

    for (int i = 0; i < idx->size; i++) {
        *(uint64_t *)p = idx->ids[i]; p += 8;
        *p++ = idx->deleted[i];
        memcpy(p, VS_LABEL(&idx->vs, i), 256); p += 256;
        memcpy(p, idx->codes + (size_t)i * (size_t)idx->dim, (size_t)idx->dim);
        p += (size_t)idx->dim;
    }

    *out_buf  = buf;
    *out_size = total;
    return PISTADB_OK;
}

int sq_load(SQIndex *idx, const void *buf, size_t size,
            int dim, DistFn dist_fn) {
    const uint8_t *p = (const uint8_t *)buf;
    if (size < 8) return PISTADB_ECORRUPT;

    int32_t count = *(const int32_t *)p; p += 4;
    int32_t fdim  = *(const int32_t *)p; p += 4;
    if (fdim != dim) return PISTADB_ECORRUPT;
    if (count < 0) return PISTADB_ECORRUPT;

    size_t hdr_sz = sizeof(int32_t) * 2 + sizeof(float) * (size_t)dim * 2;
    if (size < hdr_sz) return PISTADB_ECORRUPT;
    size_t entry  = sizeof(uint64_t) + 1 + 256 + (size_t)dim;
    if ((size_t)count > (size - hdr_sz) / (entry ? entry : 1)) return PISTADB_ECORRUPT;
    if (size < hdr_sz + (size_t)count * entry) return PISTADB_ECORRUPT;

    int r = sq_create(idx, dim, dist_fn, count + 8);
    if (r != PISTADB_OK) return r;

    memcpy(idx->vmin, p, sizeof(float) * (size_t)dim);
    p += sizeof(float) * (size_t)dim;
    memcpy(idx->vmax, p, sizeof(float) * (size_t)dim);
    p += sizeof(float) * (size_t)dim;

    for (int i = 0; i < count; i++) {
        uint64_t id  = *(const uint64_t *)p; p += 8;
        uint8_t  del = *p++;
        const char *label = (const char *)p; p += 256;
        const uint8_t *codes = p; p += (size_t)dim;

        if (idx->size == idx->cap) {
            r = sq_grow(idx);
            if (r != PISTADB_OK) return r;
        }

        int slot = idx->size++;
        idx->ids[slot] = id;
        idx->deleted[slot] = del;
        memcpy(idx->codes + (size_t)slot * (size_t)dim, codes, (size_t)dim);
        if (label) strncpy(VS_LABEL(&idx->vs, slot), label, 255);
        VS_LABEL(&idx->vs, slot)[255] = '\0';
    }

    return PISTADB_OK;
}
