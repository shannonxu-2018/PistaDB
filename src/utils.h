/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - utils.h
 * Priority queues (min-heap / max-heap), dynamic arrays, RNG.
 */
#ifndef PISTADB_UTILS_H
#define PISTADB_UTILS_H

#include <stdint.h>
#include <stddef.h>

/* ══════════════════════════════════════════════════════════════════════════
 * Priority Queue  (binary heap)
 * HeapItem.key is the priority (float), .id is the payload (uint64_t).
 * MinHeap: smallest key at top.  MaxHeap: largest key at top.
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float    key;
    uint64_t id;
} HeapItem;

typedef struct {
    HeapItem *data;
    int       size;
    int       cap;
    int       is_max;   /* 0 = min-heap, 1 = max-heap */
} Heap;

/** Initialise heap.  is_max: 0 = min, 1 = max. */
int  heap_init(Heap *h, int capacity, int is_max);

/** Free heap memory. */
void heap_free(Heap *h);

/** Insert item; returns PISTADB_OK or PISTADB_ENOMEM. */
int  heap_push(Heap *h, float key, uint64_t id);

/** Peek top item without removing. */
HeapItem heap_top(const Heap *h);

/** Remove and return top item. */
HeapItem heap_pop(Heap *h);

/** Clear all items (keep allocation). */
void heap_clear(Heap *h);

/* ══════════════════════════════════════════════════════════════════════════
 * Dynamic Array  (uint64_t elements)
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint64_t *data;
    int       size;
    int       cap;
} U64Array;

int  u64arr_init(U64Array *a, int cap);
void u64arr_free(U64Array *a);
int  u64arr_push(U64Array *a, uint64_t val);

/* ══════════════════════════════════════════════════════════════════════════
 * Simple PCG random  (thread-local state via pointer)
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct { uint64_t state; uint64_t inc; } PCG;

void  pcg_seed(PCG *rng, uint64_t seed);
/** Uniform uint32 */
uint32_t pcg_u32(PCG *rng);
/** Uniform float in [0,1) */
float pcg_f32(PCG *rng);
/** Standard normal via Box-Muller */
float pcg_normal(PCG *rng);

/* ══════════════════════════════════════════════════════════════════════════
 * Visited bitset  (for HNSW search)
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint8_t *bits;
    int      n_bytes;
    int      n_bits;
} Bitset;

int  bitset_init(Bitset *bs, int n_bits);
void bitset_free(Bitset *bs);
void bitset_clear(Bitset *bs);
void bitset_set(Bitset *bs, int idx);
int  bitset_test(const Bitset *bs, int idx);

/* ══════════════════════════════════════════════════════════════════════════
 * CRC32  (for file header integrity)
 * ══════════════════════════════════════════════════════════════════════════ */
uint32_t crc32_compute(const void *data, size_t len);

#endif /* PISTADB_UTILS_H */
