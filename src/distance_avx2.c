/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - distance_avx2.c
 * AVX2 + FMA accelerated distance kernels for x86-64.
 *
 * Compile with: -mavx2 -mfma  (GCC/Clang)
 *               /arch:AVX2    (MSVC)
 *
 * Each YMM register holds 8 × float32.
 * FMA fuses multiply-add into a single instruction, halving rounding error
 * and improving throughput on Haswell+ / Zen+ micro-architectures.
 *
 * All kernels:
 *   - Process 8 elements per iteration (one YMM)
 *   - Unroll ×2 to hide FMA latency (two independent accumulators)
 *   - Handle the tail (dim % 16) with a scalar loop
 *   - Never read past the end of the input arrays
 */

#include "distance_simd.h"
#include <immintrin.h>   /* AVX2 + FMA intrinsics */
#include <math.h>
#include <float.h>

/* Portable popcount: MSVC uses __popcnt (from <intrin.h> via immintrin.h);
 * GCC/Clang use __builtin_popcount. */
#if defined(_MSC_VER)
#  include <intrin.h>
#  define PISTADB_POPCOUNT(x) __popcnt((unsigned int)(x))
#else
#  define PISTADB_POPCOUNT(x) __builtin_popcount((unsigned int)(x))
#endif

/* ── horizontal sum of a __m256 (8 lanes → 1 scalar) ───────────────────── */

static inline float hsum256(__m256 v) {
    /* reduce 8 → 4 */
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    /* reduce 4 → 2 → 1 */
    __m128 shuf = _mm_movehdup_ps(lo);     /* [1,1,3,3] */
    __m128 sums = _mm_add_ps(lo, shuf);   /* [0+1, _, 2+3, _] */
    shuf = _mm_movehl_ps(shuf, sums);     /* [2+3, _, ?, ?] */
    sums = _mm_add_ss(sums, shuf);        /* [0+1+2+3, ...] */
    return _mm_cvtss_f32(sums);
}

/* ── vec_dot ─────────────────────────────────────────────────────────────── */

float vec_dot_avx2(const float *a, const float *b, int dim) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int i = 0;
    for (; i <= dim - 16; i += 16) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
    }
    for (; i <= dim - 8; i += 8) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
    }
    float s = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < dim; i++) s += a[i] * b[i];
    return s;
}

/* ── vec_norm ────────────────────────────────────────────────────────────── */

float vec_norm_avx2(const float *a, int dim) {
    return sqrtf(vec_dot_avx2(a, a, dim));
}

/* ── dist_l2sq ───────────────────────────────────────────────────────────── */

float dist_l2sq_avx2(const float *a, const float *b, int dim) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int i = 0;
    for (; i <= dim - 16; i += 16) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    }
    for (; i <= dim - 8; i += 8) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    }
    float s = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < dim; i++) { float d = a[i] - b[i]; s += d * d; }
    return s;
}

/* ── dist_l2 ─────────────────────────────────────────────────────────────── */

float dist_l2_avx2(const float *a, const float *b, int dim) {
    return sqrtf(dist_l2sq_avx2(a, b, dim));
}

/* ── dist_cosine ─────────────────────────────────────────────────────────── */

float dist_cosine_avx2(const float *a, const float *b, int dim) {
    __m256 vdot0 = _mm256_setzero_ps();
    __m256 vdot1 = _mm256_setzero_ps();
    __m256 vna0  = _mm256_setzero_ps();
    __m256 vna1  = _mm256_setzero_ps();
    __m256 vnb0  = _mm256_setzero_ps();
    __m256 vnb1  = _mm256_setzero_ps();

    int i = 0;
    for (; i <= dim - 16; i += 16) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        vdot0 = _mm256_fmadd_ps(a0, b0, vdot0);
        vdot1 = _mm256_fmadd_ps(a1, b1, vdot1);
        vna0  = _mm256_fmadd_ps(a0, a0, vna0);
        vna1  = _mm256_fmadd_ps(a1, a1, vna1);
        vnb0  = _mm256_fmadd_ps(b0, b0, vnb0);
        vnb1  = _mm256_fmadd_ps(b1, b1, vnb1);
    }
    for (; i <= dim - 8; i += 8) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        vdot0 = _mm256_fmadd_ps(a0, b0, vdot0);
        vna0  = _mm256_fmadd_ps(a0, a0, vna0);
        vnb0  = _mm256_fmadd_ps(b0, b0, vnb0);
    }
    float dot = hsum256(_mm256_add_ps(vdot0, vdot1));
    float na  = hsum256(_mm256_add_ps(vna0,  vna1));
    float nb  = hsum256(_mm256_add_ps(vnb0,  vnb1));

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

float dist_ip_avx2(const float *a, const float *b, int dim) {
    return -vec_dot_avx2(a, b, dim);
}

/* ── dist_l1 ─────────────────────────────────────────────────────────────── */

float dist_l1_avx2(const float *a, const float *b, int dim) {
    /* abs mask: clear the sign bit of each lane */
    const __m256 sign_mask = _mm256_castsi256_ps(
        _mm256_set1_epi32(0x7FFFFFFF));

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    int i = 0;
    for (; i <= dim - 16; i += 16) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        acc0 = _mm256_add_ps(acc0, _mm256_and_ps(d0, sign_mask));
        acc1 = _mm256_add_ps(acc1, _mm256_and_ps(d1, sign_mask));
    }
    for (; i <= dim - 8; i += 8) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_add_ps(acc0, _mm256_and_ps(d0, sign_mask));
    }
    float s = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < dim; i++) { float d = a[i] - b[i]; s += (d >= 0.0f) ? d : -d; }
    return s;
}

/* ── dist_hamming ────────────────────────────────────────────────────────── */

float dist_hamming_avx2(const float *a, const float *b, int dim) {
    /* Compare lane-by-lane: _mm256_cmp_ps gives 0xFFFFFFFF on equal.
     * We want to count *unequal*, so negate the mask.
     * movemask extracts MSB of each lane → popcount gives count per 8. */
    int count = 0;
    int i = 0;

    for (; i <= dim - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        /* equal lanes → bit = 1; unequal → bit = 0 */
        int eq_mask = _mm256_movemask_ps(_mm256_cmp_ps(va, vb, _CMP_EQ_OQ));
        /* invert lower 8 bits, count set bits = unequal lanes */
        count += (int)PISTADB_POPCOUNT((~eq_mask) & 0xFF);
    }
    for (; i < dim; i++) {
        if (a[i] != b[i]) count++;
    }
    return (float)count;
}
