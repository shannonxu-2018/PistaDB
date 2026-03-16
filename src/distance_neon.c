/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - distance_neon.c
 * ARM NEON accelerated distance kernels (AArch64 / Apple Silicon).
 *
 * AArch64 always has NEON; no compile flag is required.
 * On ARMv7 with NEON, pass -mfpu=neon.
 *
 * Each Q-register (128-bit) holds 4 × float32.
 * AArch64 has 32 V registers, so we maintain 4 independent accumulators
 * to saturate the FMA pipeline and hide 4-cycle latency.
 *
 * All kernels:
 *   - Process 16 elements per main loop (4 × float32x4_t)
 *   - Handle tail in a scalar loop
 *   - Use vmlaq_f32 (multiply-accumulate) on ARMv7/A32
 *   - Use vfmaq_f32 (FMA) on AArch64 when available
 */

#include "distance_simd.h"
#include <arm_neon.h>
#include <math.h>
#include <float.h>

/* ── horizontal sum: float32x4_t → float ────────────────────────────────── */

static inline float hsum_f32x4(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);   /* single instruction on AArch64 */
#else
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    float32x2_t s  = vadd_f32(lo, hi);
    return vget_lane_f32(vpadd_f32(s, s), 0);
#endif
}

/* ── vec_dot ─────────────────────────────────────────────────────────────── */

float vec_dot_neon(const float *a, const float *b, int dim) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i <= dim - 16; i += 16) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(a + i),      vld1q_f32(b + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(a + i +  4), vld1q_f32(b + i +  4));
        acc2 = vmlaq_f32(acc2, vld1q_f32(a + i +  8), vld1q_f32(b + i +  8));
        acc3 = vmlaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i <= dim - 4; i += 4) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float s = hsum_f32x4(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < dim; i++) s += a[i] * b[i];
    return s;
}

/* ── vec_norm ────────────────────────────────────────────────────────────── */

float vec_norm_neon(const float *a, int dim) {
    return sqrtf(vec_dot_neon(a, a, dim));
}

/* ── dist_l2sq ───────────────────────────────────────────────────────────── */

float dist_l2sq_neon(const float *a, const float *b, int dim) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i <= dim - 16; i += 16) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a + i),      vld1q_f32(b + i));
        float32x4_t d1 = vsubq_f32(vld1q_f32(a + i +  4), vld1q_f32(b + i +  4));
        float32x4_t d2 = vsubq_f32(vld1q_f32(a + i +  8), vld1q_f32(b + i +  8));
        float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        acc0 = vmlaq_f32(acc0, d0, d0);
        acc1 = vmlaq_f32(acc1, d1, d1);
        acc2 = vmlaq_f32(acc2, d2, d2);
        acc3 = vmlaq_f32(acc3, d3, d3);
    }
    for (; i <= dim - 4; i += 4) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        acc0 = vmlaq_f32(acc0, d0, d0);
    }
    float s = hsum_f32x4(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < dim; i++) { float d = a[i] - b[i]; s += d * d; }
    return s;
}

/* ── dist_l2 ─────────────────────────────────────────────────────────────── */

float dist_l2_neon(const float *a, const float *b, int dim) {
    return sqrtf(dist_l2sq_neon(a, b, dim));
}

/* ── dist_cosine ─────────────────────────────────────────────────────────── */

float dist_cosine_neon(const float *a, const float *b, int dim) {
    float32x4_t vdot0 = vdupq_n_f32(0.0f);
    float32x4_t vdot1 = vdupq_n_f32(0.0f);
    float32x4_t vna0  = vdupq_n_f32(0.0f);
    float32x4_t vna1  = vdupq_n_f32(0.0f);
    float32x4_t vnb0  = vdupq_n_f32(0.0f);
    float32x4_t vnb1  = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i <= dim - 8; i += 8) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        vdot0 = vmlaq_f32(vdot0, a0, b0);
        vdot1 = vmlaq_f32(vdot1, a1, b1);
        vna0  = vmlaq_f32(vna0,  a0, a0);
        vna1  = vmlaq_f32(vna1,  a1, a1);
        vnb0  = vmlaq_f32(vnb0,  b0, b0);
        vnb1  = vmlaq_f32(vnb1,  b1, b1);
    }
    for (; i <= dim - 4; i += 4) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        vdot0 = vmlaq_f32(vdot0, a0, b0);
        vna0  = vmlaq_f32(vna0,  a0, a0);
        vnb0  = vmlaq_f32(vnb0,  b0, b0);
    }
    float dot = hsum_f32x4(vaddq_f32(vdot0, vdot1));
    float na  = hsum_f32x4(vaddq_f32(vna0,  vna1));
    float nb  = hsum_f32x4(vaddq_f32(vnb0,  vnb1));

    for (; i < dim; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }

    float denom = sqrtf(na) * sqrtf(nb);
    if (denom < FLT_EPSILON) return 1.0f;
    float sim = dot / denom;
    if (sim >  1.0f) sim =  1.0f;
    if (sim < -1.0f) sim = -1.0f;
    return 1.0f - sim;
}

/* ── dist_ip ─────────────────────────────────────────────────────────────── */

float dist_ip_neon(const float *a, const float *b, int dim) {
    return -vec_dot_neon(a, b, dim);
}

/* ── dist_l1 ─────────────────────────────────────────────────────────────── */

float dist_l1_neon(const float *a, const float *b, int dim) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i <= dim - 16; i += 16) {
        acc0 = vaddq_f32(acc0, vabsq_f32(vsubq_f32(vld1q_f32(a + i),      vld1q_f32(b + i))));
        acc1 = vaddq_f32(acc1, vabsq_f32(vsubq_f32(vld1q_f32(a + i +  4), vld1q_f32(b + i +  4))));
        acc2 = vaddq_f32(acc2, vabsq_f32(vsubq_f32(vld1q_f32(a + i +  8), vld1q_f32(b + i +  8))));
        acc3 = vaddq_f32(acc3, vabsq_f32(vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12))));
    }
    for (; i <= dim - 4; i += 4) {
        acc0 = vaddq_f32(acc0, vabsq_f32(vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i))));
    }
    float s = hsum_f32x4(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < dim; i++) { float d = a[i] - b[i]; s += (d >= 0.0f) ? d : -d; }
    return s;
}

/* ── dist_hamming ────────────────────────────────────────────────────────── */

float dist_hamming_neon(const float *a, const float *b, int dim) {
    /* vceqq_f32 returns 0xFFFFFFFF for equal lanes, 0 for unequal.
     * Cast to uint32x4_t, right-shift by 31 to get 1 for equal / 0 for not.
     * Accumulate unequal count. */
    uint32x4_t acc0 = vdupq_n_u32(0);
    uint32x4_t acc1 = vdupq_n_u32(0);

    int i = 0;
    for (; i <= dim - 8; i += 8) {
        uint32x4_t eq0 = vceqq_f32(vld1q_f32(a + i),     vld1q_f32(b + i));
        uint32x4_t eq1 = vceqq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        /* equal → 1; shift right 31 to normalise */
        acc0 = vaddq_u32(acc0, vshrq_n_u32(eq0, 31));
        acc1 = vaddq_u32(acc1, vshrq_n_u32(eq1, 31));
    }
    for (; i <= dim - 4; i += 4) {
        uint32x4_t eq0 = vceqq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        acc0 = vaddq_u32(acc0, vshrq_n_u32(eq0, 31));
    }

    /* acc holds equal counts; unequal = dim - equal */
    uint32x4_t total = vaddq_u32(acc0, acc1);
#if defined(__aarch64__)
    uint32_t eq_count = vaddvq_u32(total);
#else
    uint32x2_t s = vadd_u32(vget_low_u32(total), vget_high_u32(total));
    s = vpadd_u32(s, s);
    uint32_t eq_count = vget_lane_u32(s, 0);
#endif

    /* add tail scalar equal count */
    for (; i < dim; i++) {
        if (a[i] == b[i]) eq_count++;
    }

    return (float)(dim - (int)eq_count);
}
