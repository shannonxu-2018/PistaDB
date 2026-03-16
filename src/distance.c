/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - distance.c
 * Distance / similarity metric implementations with runtime SIMD dispatch.
 *
 * At first use, pistadb_get_dist_fn() detects the best available kernel
 * and patches the active function pointers via a one-time initialisation:
 *
 *   Priority (x86-64): AVX2+FMA → scalar
 *   Priority (ARM)   : NEON     → scalar
 *   Priority (other) : scalar
 *
 * Scalar fallbacks are always available and never removed.
 */
#include "distance.h"
#include "distance_simd.h"
#include <math.h>
#include <float.h>

/* MSVC: __cpuid, _xgetbv live in <intrin.h> */
#if defined(_MSC_VER)
#  include <intrin.h>
#endif

/* ── Scalar implementations ──────────────────────────────────────────────── */

static float scalar_vec_dot(const float *a, const float *b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; i++) s += a[i] * b[i];
    return s;
}

static float scalar_vec_norm(const float *a, int dim) {
    return sqrtf(scalar_vec_dot(a, a, dim));
}

static float scalar_dist_l2sq(const float *a, const float *b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

static float scalar_dist_l2(const float *a, const float *b, int dim) {
    return sqrtf(scalar_dist_l2sq(a, b, dim));
}

static float scalar_dist_cosine(const float *a, const float *b, int dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < dim; i++) {
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

static float scalar_dist_ip(const float *a, const float *b, int dim) {
    return -scalar_vec_dot(a, b, dim);
}

static float scalar_dist_l1(const float *a, const float *b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        s += (d >= 0.0f) ? d : -d;
    }
    return s;
}

static float scalar_dist_hamming(const float *a, const float *b, int dim) {
    int count = 0;
    for (int i = 0; i < dim; i++) {
        if (a[i] != b[i]) count++;
    }
    return (float)count;
}

/* ── Active kernel pointers (set by simd_init, default = scalar) ─────────── */

typedef float (*VecDotFn)  (const float *, const float *, int);
typedef float (*VecNormFn) (const float *,                int);

static VecDotFn   active_vec_dot    = scalar_vec_dot;
static VecNormFn  active_vec_norm   = scalar_vec_norm;
static DistFn     active_dist_l2sq  = scalar_dist_l2sq;
static DistFn     active_dist_l2    = scalar_dist_l2;
static DistFn     active_dist_cosine = scalar_dist_cosine;
static DistFn     active_dist_ip    = scalar_dist_ip;
static DistFn     active_dist_l1    = scalar_dist_l1;
static DistFn     active_dist_hamming = scalar_dist_hamming;

/* ── SIMD detection body (called exactly once, any platform) ─────────────── */
/*
 * PISTADB_HAS_AVX2 and PISTADB_HAS_NEON are defined by CMake only when the
 * corresponding SIMD source file (distance_avx2.c / distance_neon.c) is
 * actually compiled into the same build.  Guarding the symbol references with
 * these macros prevents undefined-reference linker errors on toolchains that
 * do not support AVX2/NEON.
 */

static void simd_detect(void)
{
#if defined(PISTADB_HAS_AVX2)
#  if defined(__GNUC__) || defined(__clang__)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        active_vec_dot      = vec_dot_avx2;
        active_vec_norm     = vec_norm_avx2;
        active_dist_l2sq    = dist_l2sq_avx2;
        active_dist_l2      = dist_l2_avx2;
        active_dist_cosine  = dist_cosine_avx2;
        active_dist_ip      = dist_ip_avx2;
        active_dist_l1      = dist_l1_avx2;
        active_dist_hamming = dist_hamming_avx2;
    }
#  elif defined(_MSC_VER)
    /* MSVC: __cpuid + _xgetbv (both from <intrin.h>, included at top). */
    int regs[4] = {0};
    __cpuid(regs, 7);
    int has_avx2 = (regs[1] >> 5) & 1;     /* CPUID leaf 7, EBX bit 5 */
    __cpuid(regs, 1);
    int xsave_supported = (regs[2] >> 27) & 1;  /* ECX bit 27 */
    int os_ymm = 0;
    if (xsave_supported) {
        unsigned long long xcr0 = _xgetbv(0);
        os_ymm = (xcr0 & 6) == 6;          /* bits 1+2: SSE+YMM state saved */
    }
    if (has_avx2 && os_ymm) {
        active_vec_dot      = vec_dot_avx2;
        active_vec_norm     = vec_norm_avx2;
        active_dist_l2sq    = dist_l2sq_avx2;
        active_dist_l2      = dist_l2_avx2;
        active_dist_cosine  = dist_cosine_avx2;
        active_dist_ip      = dist_ip_avx2;
        active_dist_l1      = dist_l1_avx2;
        active_dist_hamming = dist_hamming_avx2;
    }
#  endif /* _MSC_VER */

#elif defined(PISTADB_HAS_NEON)
    /* AArch64 always has NEON; ARMv7 is gated by __ARM_NEON at compile time. */
    active_vec_dot      = vec_dot_neon;
    active_vec_norm     = vec_norm_neon;
    active_dist_l2sq    = dist_l2sq_neon;
    active_dist_l2      = dist_l2_neon;
    active_dist_cosine  = dist_cosine_neon;
    active_dist_ip      = dist_ip_neon;
    active_dist_l1      = dist_l1_neon;
    active_dist_hamming = dist_hamming_neon;
#endif /* PISTADB_HAS_NEON */
    /* scalar fallback: already set as the default, nothing to do */
}

/* ── Thread-safe one-time wrapper ────────────────────────────────────────── */

#if defined(_WIN32)
/* WIN32_LEAN_AND_MEAN already pulled in by distance_simd.h or windows.h */
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
static INIT_ONCE   g_simd_once = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK simd_init_cb(PINIT_ONCE o, PVOID p, PVOID *c)
    { (void)o; (void)p; (void)c; simd_detect(); return TRUE; }
static void simd_init(void)
    { InitOnceExecuteOnce(&g_simd_once, simd_init_cb, NULL, NULL); }
#else
#  include <pthread.h>
static pthread_once_t g_simd_once = PTHREAD_ONCE_INIT;
static void simd_init(void) { pthread_once(&g_simd_once, simd_detect); }
#endif

/* ── Public API (thin dispatch wrappers) ─────────────────────────────────── */

float vec_dot(const float *a, const float *b, int dim) {
    simd_init();
    return active_vec_dot(a, b, dim);
}

float vec_norm(const float *a, int dim) {
    simd_init();
    return active_vec_norm(a, dim);
}

float dist_l2sq(const float *a, const float *b, int dim) {
    simd_init();
    return active_dist_l2sq(a, b, dim);
}

float dist_l2(const float *a, const float *b, int dim) {
    simd_init();
    return active_dist_l2(a, b, dim);
}

float dist_cosine(const float *a, const float *b, int dim) {
    simd_init();
    return active_dist_cosine(a, b, dim);
}

float dist_ip(const float *a, const float *b, int dim) {
    simd_init();
    return active_dist_ip(a, b, dim);
}

float dist_l1(const float *a, const float *b, int dim) {
    simd_init();
    return active_dist_l1(a, b, dim);
}

float dist_hamming(const float *a, const float *b, int dim) {
    simd_init();
    return active_dist_hamming(a, b, dim);
}

/* ── dispatch ────────────────────────────────────────────────────────────── */

DistFn pistadb_get_dist_fn(PistaDBMetric metric) {
    simd_init();
    switch (metric) {
        case METRIC_L2:      return active_dist_l2;
        case METRIC_COSINE:  return active_dist_cosine;
        case METRIC_IP:      return active_dist_ip;
        case METRIC_L1:      return active_dist_l1;
        case METRIC_HAMMING: return active_dist_hamming;
        default:             return active_dist_l2;
    }
}
