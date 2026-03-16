/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - distance.c
 * Distance / similarity metric implementations.
 */
#include "distance.h"
#include <math.h>
#include <float.h>

/* ── helpers ─────────────────────────────────────────────────────────────── */

float vec_dot(const float *a, const float *b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; i++) s += a[i] * b[i];
    return s;
}

float vec_norm(const float *a, int dim) {
    return sqrtf(vec_dot(a, a, dim));
}

/* ── L2 squared ──────────────────────────────────────────────────────────── */

float dist_l2sq(const float *a, const float *b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

/* ── L2 ──────────────────────────────────────────────────────────────────── */

float dist_l2(const float *a, const float *b, int dim) {
    return sqrtf(dist_l2sq(a, b, dim));
}

/* ── Cosine distance ─────────────────────────────────────────────────────── */

float dist_cosine(const float *a, const float *b, int dim) {
    float dot  = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    if (denom < FLT_EPSILON) return 1.0f;  /* treat zero vector as max dist */
    float sim = dot / denom;
    /* clamp to [-1,1] due to floating-point drift */
    if (sim >  1.0f) sim =  1.0f;
    if (sim < -1.0f) sim = -1.0f;
    return 1.0f - sim;  /* distance in [0, 2] */
}

/* ── Inner product (as distance) ─────────────────────────────────────────── */

float dist_ip(const float *a, const float *b, int dim) {
    /* Return negative so that maximum IP → minimum distance */
    return -vec_dot(a, b, dim);
}

/* ── L1 (Manhattan) ──────────────────────────────────────────────────────── */

float dist_l1(const float *a, const float *b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        s += (d >= 0.0f) ? d : -d;
    }
    return s;
}

/* ── Hamming ─────────────────────────────────────────────────────────────── */

float dist_hamming(const float *a, const float *b, int dim) {
    int count = 0;
    for (int i = 0; i < dim; i++) {
        if (a[i] != b[i]) count++;
    }
    return (float)count;
}

/* ── dispatch ────────────────────────────────────────────────────────────── */

DistFn pistadb_get_dist_fn(PistaDBMetric metric) {
    switch (metric) {
        case METRIC_L2:      return dist_l2;
        case METRIC_COSINE:  return dist_cosine;
        case METRIC_IP:      return dist_ip;
        case METRIC_L1:      return dist_l1;
        case METRIC_HAMMING: return dist_hamming;
        default:             return dist_l2;
    }
}
