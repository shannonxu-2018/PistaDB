/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - pistadb_cache.c
 * Embedding cache implementation (FNV-1a hash map + LRU doubly-linked list)
 */

#include "pistadb_cache.h"
#include "pistadb_types.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

/* ── Platform threading (same abstraction as pistadb_batch.c) ────────────── */

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
   typedef CRITICAL_SECTION  pdb_mutex_t;
#  define pdb_mutex_init(m)    InitializeCriticalSection(m)
#  define pdb_mutex_destroy(m) DeleteCriticalSection(m)
#  define pdb_mutex_lock(m)    EnterCriticalSection(m)
#  define pdb_mutex_unlock(m)  LeaveCriticalSection(m)
#else
#  include <pthread.h>
   typedef pthread_mutex_t pdb_mutex_t;
#  define pdb_mutex_init(m)    pthread_mutex_init(m, NULL)
#  define pdb_mutex_destroy(m) pthread_mutex_destroy(m)
#  define pdb_mutex_lock(m)    pthread_mutex_lock(m)
#  define pdb_mutex_unlock(m)  pthread_mutex_unlock(m)
#endif

/* ── File format constants ───────────────────────────────────────────────── */

#define PCC_MAGIC        "PCCH"
#define PCC_VER_MAJOR    1
#define PCC_VER_MINOR    0

/* Sanity cap on key length to protect against corrupt/malicious files.
 * 64 KiB is generous for any real embedding key. */
#define PCC_MAX_TEXT_LEN (64u * 1024u)

/* ── Hash map defaults ───────────────────────────────────────────────────── */

#define CACHE_INIT_BUCKETS  256u     /* must be power of 2 */
#define CACHE_LOAD_FACTOR   0.75

/* ── Internal data structures ────────────────────────────────────────────── */

typedef struct CacheEntry {
    uint64_t          hash;       /* FNV-1a 64-bit hash of text              */
    char             *text;       /* heap-allocated, null-terminated          */
    float            *vec;        /* heap-allocated float[dim]                */
    struct CacheEntry *hash_next; /* separate-chaining within bucket          */
    struct CacheEntry *lru_prev;  /* doubly-linked LRU list: prev (toward LRU)*/
    struct CacheEntry *lru_next;  /* doubly-linked LRU list: next (toward MRU)*/
} CacheEntry;

struct PistaDBCache {
    /* Settings */
    char       *path;
    int         dim;
    int         max_entries;

    /* Hash table */
    CacheEntry **buckets;
    uint32_t    n_buckets;   /* always power of 2 */
    int         n_live;

    /* LRU doubly-linked list
     * lru_head = MRU (most recently used)
     * lru_tail = LRU (least recently used, evicted first)
     * Traversal direction: head --[lru_next]--> tail  (MRU → LRU)
     *                      tail --[lru_prev]--> head  (LRU → MRU)
     */
    CacheEntry *lru_head;
    CacheEntry *lru_tail;

    /* Stats */
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;

    /* Thread safety */
    pdb_mutex_t mu;
};

/* ── FNV-1a 64-bit ───────────────────────────────────────────────────────── */

static uint64_t fnv1a_64(const char *s)
{
    uint64_t h = UINT64_C(14695981039346656037);
    for (; *s; s++) {
        h ^= (uint8_t)*s;
        h *= UINT64_C(1099511628211);
    }
    return h;
}

/* ── LRU list helpers ────────────────────────────────────────────────────── */

/* Unlink entry from LRU list (entry MUST be in the list). */
static void lru_unlink(PistaDBCache *c, CacheEntry *e)
{
    if (e->lru_prev) e->lru_prev->lru_next = e->lru_next;
    else             c->lru_head = e->lru_next;

    if (e->lru_next) e->lru_next->lru_prev = e->lru_prev;
    else             c->lru_tail = e->lru_prev;

    e->lru_prev = e->lru_next = NULL;
}

/* Insert entry at the head (MRU position). */
static void lru_push_head(PistaDBCache *c, CacheEntry *e)
{
    e->lru_prev = NULL;
    e->lru_next = c->lru_head;
    if (c->lru_head) c->lru_head->lru_prev = e;
    c->lru_head = e;
    if (!c->lru_tail) c->lru_tail = e;
}

/* Move an existing entry to the head (promote to MRU). */
static void lru_touch(PistaDBCache *c, CacheEntry *e)
{
    if (c->lru_head == e) return; /* already MRU */
    lru_unlink(c, e);
    lru_push_head(c, e);
}

/* ── Hash table helpers ──────────────────────────────────────────────────── */

static CacheEntry **bucket_for(PistaDBCache *c, uint64_t hash)
{
    return &c->buckets[hash & (c->n_buckets - 1)];
}

/* Find entry in hash chain; returns pointer-to-pointer for easy removal.
 * The returned pointer points to the slot that holds the matching entry,
 * enabling O(1) removal via *pp = (*pp)->hash_next. */
static CacheEntry **chain_find(CacheEntry **slot, uint64_t hash, const char *text)
{
    while (*slot) {
        if ((*slot)->hash == hash && strcmp((*slot)->text, text) == 0)
            return slot;
        slot = &(*slot)->hash_next;
    }
    return NULL;
}

/* Remove entry from its hash chain (does NOT free or unlink LRU). */
static void chain_remove(PistaDBCache *c, CacheEntry *e)
{
    CacheEntry **slot = bucket_for(c, e->hash);
    while (*slot && *slot != e)
        slot = &(*slot)->hash_next;
    if (*slot == e)
        *slot = e->hash_next;
    e->hash_next = NULL;
}

/* Rehash to new_cap buckets (power of 2). Returns 1 on success, 0 on OOM. */
static int cache_rehash(PistaDBCache *c, uint32_t new_cap)
{
    CacheEntry **nb = (CacheEntry **)calloc(new_cap, sizeof(CacheEntry *));
    if (!nb) return 0;

    for (uint32_t i = 0; i < c->n_buckets; i++) {
        CacheEntry *e = c->buckets[i];
        while (e) {
            CacheEntry *next = e->hash_next;
            uint32_t idx = (uint32_t)(e->hash & (new_cap - 1));
            e->hash_next = nb[idx];
            nb[idx] = e;
            e = next;
        }
    }
    free(c->buckets);
    c->buckets   = nb;
    c->n_buckets = new_cap;
    return 1;
}

/* ── Entry alloc / free ──────────────────────────────────────────────────── */

static CacheEntry *entry_alloc(uint64_t hash, const char *text,
                                const float *vec, int dim)
{
    CacheEntry *e = (CacheEntry *)calloc(1, sizeof(CacheEntry));
    if (!e) return NULL;

    size_t tlen = strlen(text) + 1;
    e->text = (char *)malloc(tlen);
    e->vec  = (float *)malloc((size_t)dim * sizeof(float));
    if (!e->text || !e->vec) {
        free(e->text); free(e->vec); free(e);
        return NULL;
    }
    memcpy(e->text, text, tlen);
    memcpy(e->vec, vec, (size_t)dim * sizeof(float));
    e->hash = hash;
    return e;
}

static void entry_free(CacheEntry *e)
{
    if (!e) return;
    free(e->text);
    free(e->vec);
    free(e);
}

/* ── Evict the LRU entry (caller holds lock, cache must be non-empty). ───── */

static void evict_lru(PistaDBCache *c)
{
    CacheEntry *e = c->lru_tail;
    if (!e) return;
    lru_unlink(c, e);
    chain_remove(c, e);
    entry_free(e);
    c->n_live--;
    c->evictions++;
}

/* ── File I/O helpers (explicit little-endian, no alignment assumptions) ─── */

/* Write helpers: accumulate error state in *ok rather than silently ignoring. */
static void write_u32_le(FILE *f, uint32_t v, int *ok)
{
    if (!*ok) return;
    uint8_t buf[4] = {
        (uint8_t)(v),
        (uint8_t)(v >>  8),
        (uint8_t)(v >> 16),
        (uint8_t)(v >> 24)
    };
    if (fwrite(buf, 1, 4, f) != 4) *ok = 0;
}

static void write_u64_le(FILE *f, uint64_t v, int *ok)
{
    if (!*ok) return;
    uint8_t buf[8] = {
        (uint8_t)(v),
        (uint8_t)(v >>  8),
        (uint8_t)(v >> 16),
        (uint8_t)(v >> 24),
        (uint8_t)(v >> 32),
        (uint8_t)(v >> 40),
        (uint8_t)(v >> 48),
        (uint8_t)(v >> 56)
    };
    if (fwrite(buf, 1, 8, f) != 8) *ok = 0;
}

static void write_bytes(FILE *f, const void *buf, size_t n, int *ok)
{
    if (!*ok) return;
    if (fwrite(buf, 1, n, f) != n) *ok = 0;
}

static int read_u32_le(FILE *f, uint32_t *out)
{
    uint8_t buf[4];
    if (fread(buf, 1, 4, f) != 4) return 0;
    *out = (uint32_t)buf[0]
         | ((uint32_t)buf[1] <<  8)
         | ((uint32_t)buf[2] << 16)
         | ((uint32_t)buf[3] << 24);
    return 1;
}

static int read_u64_le(FILE *f, uint64_t *out)
{
    uint8_t buf[8];
    if (fread(buf, 1, 8, f) != 8) return 0;
    *out = (uint64_t)buf[0]
         | ((uint64_t)buf[1] <<  8)
         | ((uint64_t)buf[2] << 16)
         | ((uint64_t)buf[3] << 24)
         | ((uint64_t)buf[4] << 32)
         | ((uint64_t)buf[5] << 40)
         | ((uint64_t)buf[6] << 48)
         | ((uint64_t)buf[7] << 56);
    return 1;
}

/* ── Load entries from an existing .pcc file ─────────────────────────────── */

static void cache_load(PistaDBCache *c, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return;

    /* ── Header ── */
    char magic[4];
    if (fread(magic, 1, 4, f) != 4)            goto done;
    if (memcmp(magic, PCC_MAGIC, 4) != 0)       goto done;

    uint8_t ver[2];
    if (fread(ver, 1, 2, f) != 2)              goto done;
    if (ver[0] != PCC_VER_MAJOR)               goto done; /* major version mismatch */

    uint32_t file_dim;
    if (!read_u32_le(f, &file_dim))            goto done;
    if ((int)file_dim != c->dim)               goto done; /* dimension mismatch */

    uint32_t file_max;
    if (!read_u32_le(f, &file_max))            goto done;

    uint64_t n_entries, hits, misses, evictions;
    if (!read_u64_le(f, &n_entries))           goto done;
    if (!read_u64_le(f, &hits))                goto done;
    if (!read_u64_le(f, &misses))              goto done;
    if (!read_u64_le(f, &evictions))           goto done;

    /* skip reserved[18] */
    if (fseek(f, 18, SEEK_CUR) != 0)          goto done;

    /* Restore cumulative stats from the persisted file. */
    c->hits      = hits;
    c->misses    = misses;
    c->evictions = evictions;

    /* ── Entries (stored LRU-first → reload in same order via put()) ── */
    char  *tbuf     = NULL;
    size_t tbuf_cap = 0;

    for (uint64_t i = 0; i < n_entries; i++) {
        /* Stop loading once the cache is full.  This avoids the fragile
         * evict_lru() + evictions-- pattern and preserves stats integrity. */
        if (c->max_entries > 0 && c->n_live >= c->max_entries)
            break;

        uint32_t tlen;
        if (!read_u32_le(f, &tlen))            break;

        /* Guard against corrupt files: zero-length or absurdly large keys. */
        if (tlen == 0 || tlen > PCC_MAX_TEXT_LEN) break;

        /* Grow text buffer only when necessary (reuse across iterations). */
        if ((size_t)tlen > tbuf_cap) {
            char *nb = (char *)realloc(tbuf, tlen);
            if (!nb)                           break;
            tbuf     = nb;
            tbuf_cap = tlen;
        }
        if (fread(tbuf, 1, tlen, f) != tlen)  break;
        tbuf[tlen - 1] = '\0'; /* enforce null-termination */

        size_t vbytes = (size_t)c->dim * sizeof(float);
        float *vec = (float *)malloc(vbytes);
        if (!vec)                              break;
        if (fread(vec, 1, vbytes, f) != vbytes) { free(vec); break; }

        uint64_t hash = fnv1a_64(tbuf);
        CacheEntry **slot  = bucket_for(c, hash);
        CacheEntry **found = chain_find(slot, hash, tbuf);

        if (found) {
            /* Duplicate in file — update vector, promote to MRU. */
            memcpy((*found)->vec, vec, vbytes);
            lru_touch(c, *found);
            free(vec);
        } else {
            /* Grow hash table if load factor exceeded. */
            if (c->n_live >= (int)(c->n_buckets * CACHE_LOAD_FACTOR)) {
                if (!cache_rehash(c, c->n_buckets * 2)) { free(vec); break; }
                slot = bucket_for(c, hash); /* bucket array may have moved */
            }

            CacheEntry *e = (CacheEntry *)calloc(1, sizeof(CacheEntry));
            if (!e) { free(vec); break; }
            e->text = (char *)malloc(tlen);
            if (!e->text) { free(e); free(vec); break; }
            memcpy(e->text, tbuf, tlen);
            e->vec  = vec;
            e->hash = hash;

            e->hash_next = *slot;
            *slot = e;
            lru_push_head(c, e);
            c->n_live++;
        }
    }
    free(tbuf);

done:
    fclose(f);
}

/* ── Public API ──────────────────────────────────────────────────────────── */

PistaDBCache *pistadb_cache_open(const char *path, int dim, int max_entries)
{
    if (dim <= 0) return NULL;

    PistaDBCache *c = (PistaDBCache *)calloc(1, sizeof(PistaDBCache));
    if (!c) return NULL;

    if (path) {
        c->path = (char *)malloc(strlen(path) + 1);
        if (!c->path) { free(c); return NULL; }
        strcpy(c->path, path);
    }
    c->dim         = dim;
    c->max_entries = max_entries;
    c->n_buckets   = CACHE_INIT_BUCKETS;
    c->buckets     = (CacheEntry **)calloc(c->n_buckets, sizeof(CacheEntry *));
    if (!c->buckets) { free(c->path); free(c); return NULL; }

    pdb_mutex_init(&c->mu);

    if (path) cache_load(c, path);

    return c;
}

int pistadb_cache_save(PistaDBCache *c)
{
    if (!c || !c->path) return PISTADB_OK; /* no-op for in-memory-only caches */

    pdb_mutex_lock(&c->mu);

    FILE *f = fopen(c->path, "wb");
    if (!f) { pdb_mutex_unlock(&c->mu); return PISTADB_EIO; }

    int ok = 1; /* tracks all write errors in one place */

    /* ── Header (64 bytes total) ──
     * 4  magic
     * 2  ver[major, minor]
     * 4  dim
     * 4  max_entries
     * 8  n_entries
     * 8  hits
     * 8  misses
     * 8  evictions
     * 18 reserved (zeros)
     * ─────────────────────────────
     * 64 bytes total
     */
    write_bytes(f, PCC_MAGIC, 4, &ok);
    uint8_t ver[2] = { PCC_VER_MAJOR, PCC_VER_MINOR };
    write_bytes(f, ver, 2, &ok);
    write_u32_le(f, (uint32_t)c->dim,          &ok);
    write_u32_le(f, (uint32_t)c->max_entries,  &ok);
    write_u64_le(f, (uint64_t)c->n_live,        &ok);
    write_u64_le(f, c->hits,                    &ok);
    write_u64_le(f, c->misses,                  &ok);
    write_u64_le(f, c->evictions,               &ok);
    static const uint8_t zeros[18] = {0};
    write_bytes(f, zeros, 18, &ok);

    /* ── Entries: walk from tail (LRU) toward head (MRU) via lru_prev.
     * Storing LRU-first means a sequential reload via lru_push_head()
     * recreates the exact same MRU ordering. */
    size_t vbytes = (size_t)c->dim * sizeof(float);
    for (CacheEntry *e = c->lru_tail; e && ok; e = e->lru_prev) {
        size_t tlen = strlen(e->text) + 1;
        write_u32_le(f, (uint32_t)tlen, &ok);
        write_bytes(f, e->text, tlen,   &ok);
        write_bytes(f, e->vec,  vbytes, &ok);
    }

    if (ok) ok = (fflush(f) == 0);
    fclose(f);

    pdb_mutex_unlock(&c->mu);
    return ok ? PISTADB_OK : PISTADB_EIO;
}

void pistadb_cache_close(PistaDBCache *c)
{
    if (!c) return;
    pistadb_cache_clear(c);   /* frees all entries (acquires + releases mutex) */
    free(c->buckets);
    free(c->path);
    pdb_mutex_destroy(&c->mu);
    free(c);
}

int pistadb_cache_get(PistaDBCache *c, const char *text, float *out_vec)
{
    if (!c || !text || !out_vec) return 0;

    pdb_mutex_lock(&c->mu);

    uint64_t    hash  = fnv1a_64(text);
    CacheEntry **slot  = bucket_for(c, hash);
    CacheEntry **found = chain_find(slot, hash, text);
    int hit = 0;

    if (found) {
        memcpy(out_vec, (*found)->vec, (size_t)c->dim * sizeof(float));
        lru_touch(c, *found);
        c->hits++;
        hit = 1;
    } else {
        c->misses++;
    }

    pdb_mutex_unlock(&c->mu);
    return hit;
}

int pistadb_cache_put(PistaDBCache *c, const char *text, const float *vec)
{
    if (!c || !text || !vec) return PISTADB_EINVAL;

    pdb_mutex_lock(&c->mu);

    uint64_t    hash  = fnv1a_64(text);
    CacheEntry **slot  = bucket_for(c, hash);
    CacheEntry **found = chain_find(slot, hash, text);

    if (found) {
        /* Key exists: update vector, promote to MRU. */
        memcpy((*found)->vec, vec, (size_t)c->dim * sizeof(float));
        lru_touch(c, *found);
        pdb_mutex_unlock(&c->mu);
        return PISTADB_OK;
    }

    /* Grow hash table before inserting if load factor exceeded. */
    if (c->n_live >= (int)(c->n_buckets * CACHE_LOAD_FACTOR)) {
        if (!cache_rehash(c, c->n_buckets * 2)) {
            pdb_mutex_unlock(&c->mu);
            return PISTADB_ENOMEM;
        }
        slot = bucket_for(c, hash); /* bucket array may have moved */
    }

    /* Evict LRU if at capacity. */
    if (c->max_entries > 0 && c->n_live >= c->max_entries)
        evict_lru(c);

    CacheEntry *e = entry_alloc(hash, text, vec, c->dim);
    if (!e) {
        pdb_mutex_unlock(&c->mu);
        return PISTADB_ENOMEM;
    }

    e->hash_next = *slot;
    *slot = e;
    lru_push_head(c, e);
    c->n_live++;

    pdb_mutex_unlock(&c->mu);
    return PISTADB_OK;
}

int pistadb_cache_contains(PistaDBCache *c, const char *text)
{
    if (!c || !text) return 0;

    pdb_mutex_lock(&c->mu);
    uint64_t    hash  = fnv1a_64(text);
    CacheEntry **slot  = bucket_for(c, hash);
    int present = (chain_find(slot, hash, text) != NULL);
    pdb_mutex_unlock(&c->mu);
    return present;
}

int pistadb_cache_evict_key(PistaDBCache *c, const char *text)
{
    if (!c || !text) return 0;

    pdb_mutex_lock(&c->mu);
    uint64_t    hash  = fnv1a_64(text);
    CacheEntry **slot  = bucket_for(c, hash);
    CacheEntry **found = chain_find(slot, hash, text);
    int removed = 0;

    if (found) {
        CacheEntry *e = *found;
        *found = e->hash_next;  /* unlink from hash chain */
        lru_unlink(c, e);
        entry_free(e);
        c->n_live--;
        removed = 1;
    }

    pdb_mutex_unlock(&c->mu);
    return removed;
}

void pistadb_cache_clear(PistaDBCache *c)
{
    if (!c) return;

    pdb_mutex_lock(&c->mu);

    /* Walk LRU list and free every entry. */
    CacheEntry *e = c->lru_head;
    while (e) {
        CacheEntry *next = e->lru_next;
        entry_free(e);
        e = next;
    }
    c->lru_head = c->lru_tail = NULL;

    /* Zero the bucket array (pointers are now dangling — must clear). */
    memset(c->buckets, 0, c->n_buckets * sizeof(CacheEntry *));
    c->n_live = 0;

    pdb_mutex_unlock(&c->mu);
}

void pistadb_cache_stats(PistaDBCache *c, PistaDBCacheStats *out)
{
    if (!c || !out) return;

    pdb_mutex_lock(&c->mu);
    out->hits        = c->hits;
    out->misses      = c->misses;
    out->evictions   = c->evictions;
    out->count       = c->n_live;
    out->max_entries = c->max_entries;
    pdb_mutex_unlock(&c->mu);
}

int pistadb_cache_count(PistaDBCache *c)
{
    if (!c) return 0;
    pdb_mutex_lock(&c->mu);
    int n = c->n_live;
    pdb_mutex_unlock(&c->mu);
    return n;
}
