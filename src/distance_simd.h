/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - distance_simd.h
 * Internal declarations for SIMD-accelerated distance kernels.
 *
 * These symbols are NOT part of the public API.
 * Include this header only from distance_avx2.c, distance_neon.c,
 * and distance.c (for the dispatch table).
 */
#ifndef PISTADB_DISTANCE_SIMD_H
#define PISTADB_DISTANCE_SIMD_H

/* ── AVX2 kernels (x86-64 with AVX2 + FMA) ─────────────────────────────── */

float vec_dot_avx2  (const float *a, const float *b, int dim);
float vec_norm_avx2 (const float *a,                 int dim);
float dist_l2sq_avx2(const float *a, const float *b, int dim);
float dist_l2_avx2  (const float *a, const float *b, int dim);
float dist_cosine_avx2(const float *a, const float *b, int dim);
float dist_ip_avx2  (const float *a, const float *b, int dim);
float dist_l1_avx2  (const float *a, const float *b, int dim);
float dist_hamming_avx2(const float *a, const float *b, int dim);

/* ── NEON kernels (AArch64 / Apple Silicon / ARMv7+NEON) ────────────────── */

float vec_dot_neon  (const float *a, const float *b, int dim);
float vec_norm_neon (const float *a,                 int dim);
float dist_l2sq_neon(const float *a, const float *b, int dim);
float dist_l2_neon  (const float *a, const float *b, int dim);
float dist_cosine_neon(const float *a, const float *b, int dim);
float dist_ip_neon  (const float *a, const float *b, int dim);
float dist_l1_neon  (const float *a, const float *b, int dim);
float dist_hamming_neon(const float *a, const float *b, int dim);

#endif /* PISTADB_DISTANCE_SIMD_H */
