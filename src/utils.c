/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - utils.c
 */
#include "utils.h"
#include "pistadb_types.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ══════════════════════════════════════════════════════════════════════════
 * Heap
 * ══════════════════════════════════════════════════════════════════════════ */

int heap_init(Heap *h, int capacity, int is_max) {
    h->data = (HeapItem *)malloc(sizeof(HeapItem) * (size_t)capacity);
    if (!h->data) return PISTADB_ENOMEM;
    h->size   = 0;
    h->cap    = capacity;
    h->is_max = is_max;
    return PISTADB_OK;
}

void heap_free(Heap *h) {
    free(h->data);
    h->data = NULL;
    h->size = 0;
    h->cap  = 0;
}

void heap_clear(Heap *h) { h->size = 0; }

static inline int heap_cmp(const Heap *h, int i, int j) {
    /* Returns non-zero if data[i] should be above data[j] in the heap. */
    if (h->is_max) return h->data[i].key > h->data[j].key;
    else           return h->data[i].key < h->data[j].key;
}

static inline void heap_swap(Heap *h, int i, int j) {
    HeapItem tmp = h->data[i];
    h->data[i] = h->data[j];
    h->data[j] = tmp;
}

static void sift_up(Heap *h, int idx) {
    while (idx > 0) {
        int parent = (idx - 1) / 2;
        if (heap_cmp(h, idx, parent)) {
            heap_swap(h, idx, parent);
            idx = parent;
        } else break;
    }
}

static void sift_down(Heap *h, int idx) {
    int n = h->size;
    for (;;) {
        int best = idx;
        int l = 2 * idx + 1, r = 2 * idx + 2;
        if (l < n && heap_cmp(h, l, best)) best = l;
        if (r < n && heap_cmp(h, r, best)) best = r;
        if (best == idx) break;
        heap_swap(h, idx, best);
        idx = best;
    }
}

int heap_push(Heap *h, float key, uint64_t id) {
    if (h->size == h->cap) {
        int newcap = h->cap * 2 + 8;
        HeapItem *nd = (HeapItem *)realloc(h->data, sizeof(HeapItem) * (size_t)newcap);
        if (!nd) return PISTADB_ENOMEM;
        h->data = nd;
        h->cap  = newcap;
    }
    h->data[h->size].key = key;
    h->data[h->size].id  = id;
    sift_up(h, h->size);
    h->size++;
    return PISTADB_OK;
}

HeapItem heap_top(const Heap *h) { return h->data[0]; }

HeapItem heap_pop(Heap *h) {
    HeapItem top = h->data[0];
    h->size--;
    if (h->size > 0) {
        h->data[0] = h->data[h->size];
        sift_down(h, 0);
    }
    return top;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Dynamic uint64 array
 * ══════════════════════════════════════════════════════════════════════════ */

int u64arr_init(U64Array *a, int cap) {
    a->data = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)cap);
    if (!a->data) return PISTADB_ENOMEM;
    a->size = 0;
    a->cap  = cap;
    return PISTADB_OK;
}

void u64arr_free(U64Array *a) { free(a->data); a->data = NULL; a->size = a->cap = 0; }

int u64arr_push(U64Array *a, uint64_t val) {
    if (a->size == a->cap) {
        int nc = a->cap * 2 + 8;
        uint64_t *nd = (uint64_t *)realloc(a->data, sizeof(uint64_t) * (size_t)nc);
        if (!nd) return PISTADB_ENOMEM;
        a->data = nd;
        a->cap  = nc;
    }
    a->data[a->size++] = val;
    return PISTADB_OK;
}

/* ══════════════════════════════════════════════════════════════════════════
 * PCG32 random
 * ══════════════════════════════════════════════════════════════════════════ */

void pcg_seed(PCG *rng, uint64_t seed) {
    rng->state = 0ULL;
    rng->inc   = (seed << 1u) | 1u;
    pcg_u32(rng);
    rng->state += seed;
    pcg_u32(rng);
}

uint32_t pcg_u32(PCG *rng) {
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
}

float pcg_f32(PCG *rng) {
    return (float)(pcg_u32(rng) >> 8) / (float)(1 << 24);
}

float pcg_normal(PCG *rng) {
    /* Box-Muller */
    float u1 = pcg_f32(rng) + 1e-10f;
    float u2 = pcg_f32(rng);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

/* ══════════════════════════════════════════════════════════════════════════
 * Bitset
 * ══════════════════════════════════════════════════════════════════════════ */

int bitset_init(Bitset *bs, int n_bits) {
    int nb = (n_bits + 7) / 8;
    bs->bits   = (uint8_t *)calloc((size_t)nb, 1);
    if (!bs->bits) return PISTADB_ENOMEM;
    bs->n_bytes = nb;
    bs->n_bits  = n_bits;
    return PISTADB_OK;
}

void bitset_free(Bitset *bs)  { free(bs->bits); bs->bits = NULL; }
void bitset_clear(Bitset *bs) { memset(bs->bits, 0, (size_t)bs->n_bytes); }
void bitset_set(Bitset *bs, int idx)       { bs->bits[idx >> 3] |=  (uint8_t)(1u << (idx & 7)); }
int  bitset_test(const Bitset *bs, int idx){ return (bs->bits[idx >> 3] >> (idx & 7)) & 1; }

/* ══════════════════════════════════════════════════════════════════════════
 * CRC32 (IEEE 802.3 polynomial, no-table version)
 * ══════════════════════════════════════════════════════════════════════════ */

uint32_t crc32_compute(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= p[i];
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0xEDB88320u & ((crc & 1u) ? 0xFFFFFFFFu : 0u));
    }
    return crc ^ 0xFFFFFFFFu;
}
