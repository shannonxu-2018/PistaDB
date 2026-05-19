/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - pager.c
 * SQLite-style read pager: fixed pages + bounded LRU page cache.
 */
#ifndef _WIN32
#  define _FILE_OFFSET_BITS 64
#endif
#include "pager.h"
#include "pistadb_types.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* 64-bit fseek (fseek(long) silently truncates >2 GB offsets on Windows). */
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
   typedef CRITICAL_SECTION pdb_pg_mutex_t;
#  define pg_mutex_init(m)    InitializeCriticalSection(m)
#  define pg_mutex_destroy(m) DeleteCriticalSection(m)
#  define pg_mutex_lock(m)    EnterCriticalSection(m)
#  define pg_mutex_unlock(m)  LeaveCriticalSection(m)
#  define PG_FSEEK64(f,o,w)   _fseeki64((f),(__int64)(o),(w))
#else
#  include <pthread.h>
   typedef pthread_mutex_t pdb_pg_mutex_t;
#  define pg_mutex_init(m)    pthread_mutex_init((m), NULL)
#  define pg_mutex_destroy(m) pthread_mutex_destroy(m)
#  define pg_mutex_lock(m)    pthread_mutex_lock(m)
#  define pg_mutex_unlock(m)  pthread_mutex_unlock(m)
#  define PG_FSEEK64(f,o,w)   fseeko((f),(off_t)(o),(w))
#endif

#define PG_DEFAULT_PAGE_SIZE  (64u * 1024u)
#define PG_DEFAULT_CACHE      (64u * 1024u * 1024u)
#define PG_MIN_PAGE_SIZE      4096u
#define PG_MIN_FRAMES         16          /* fits a search's transient pins */

typedef struct {
    uint64_t page_no;   /* region page index held when valid                 */
    int      valid;     /* 1 = data[] backs page_no                          */
    int      pin;       /* pin count; 0 ⇒ on the LRU list and evictable      */
    int      lru_prev;  /* LRU dll (only valid, unpinned frames) / free stack*/
    int      lru_next;
    int      h_next;    /* hash chain                                        */
} Frame;

struct PistaPager {
    FILE          *f;
    pdb_pg_mutex_t mu;
    uint64_t       region_off;
    uint64_t       region_len;
    uint32_t       page_size;
    uint32_t       page_mask;   /* page_size - 1                             */
    int            page_shift;  /* log2(page_size)                           */

    int       n_frames;
    uint8_t  *data;             /* n_frames * page_size                      */
    Frame    *fr;               /* n_frames                                  */
    int      *htab;             /* hmask+1 buckets → frame idx or -1         */
    int       hmask;

    int       lru_head;         /* MRU                                       */
    int       lru_tail;         /* LRU — eviction victim                     */
    int       free_top;         /* free-frame stack (linked via lru_next)    */

    uint64_t  hits, misses, evictions, fetches;
    int       n_resident;
};

/* ── Hash: Knuth multiplicative on the region page index ─────────────────── */
static inline int pg_bucket(const PistaPager *pg, uint64_t page_no) {
    return (int)(((uint32_t)page_no * 2654435761u) & (uint32_t)pg->hmask);
}

static int pg_hash_find(const PistaPager *pg, uint64_t page_no) {
    int i = pg->htab[pg_bucket(pg, page_no)];
    while (i != -1) {
        if (pg->fr[i].valid && pg->fr[i].page_no == page_no) return i;
        i = pg->fr[i].h_next;
    }
    return -1;
}

static void pg_hash_insert(PistaPager *pg, int idx) {
    int b = pg_bucket(pg, pg->fr[idx].page_no);
    pg->fr[idx].h_next = pg->htab[b];
    pg->htab[b] = idx;
}

static void pg_hash_remove(PistaPager *pg, int idx) {
    int *slot = &pg->htab[pg_bucket(pg, pg->fr[idx].page_no)];
    while (*slot != -1 && *slot != idx) slot = &pg->fr[*slot].h_next;
    if (*slot == idx) *slot = pg->fr[idx].h_next;
    pg->fr[idx].h_next = -1;
}

/* ── LRU doubly-linked list (valid, unpinned frames only) ────────────────── */
static void pg_lru_unlink(PistaPager *pg, int i) {
    int p = pg->fr[i].lru_prev, n = pg->fr[i].lru_next;
    if (p != -1) pg->fr[p].lru_next = n; else pg->lru_head = n;
    if (n != -1) pg->fr[n].lru_prev = p; else pg->lru_tail = p;
    pg->fr[i].lru_prev = pg->fr[i].lru_next = -1;
}

static void pg_lru_push_head(PistaPager *pg, int i) {
    pg->fr[i].lru_prev = -1;
    pg->fr[i].lru_next = pg->lru_head;
    if (pg->lru_head != -1) pg->fr[pg->lru_head].lru_prev = i;
    pg->lru_head = i;
    if (pg->lru_tail == -1) pg->lru_tail = i;
}

/* Obtain a usable frame: pop the free stack, else evict the LRU tail.
 * Returns a frame idx with valid==0, or -1 if every frame is pinned. */
static int pg_alloc_frame(PistaPager *pg) {
    if (pg->free_top != -1) {
        int i = pg->free_top;
        pg->free_top = pg->fr[i].lru_next;
        pg->fr[i].lru_prev = pg->fr[i].lru_next = -1;
        return i;
    }
    int v = pg->lru_tail;            /* tail is by construction unpinned */
    if (v == -1) return -1;          /* working set exceeds cache        */
    pg_lru_unlink(pg, v);
    pg_hash_remove(pg, v);
    pg->fr[v].valid = 0;
    pg->n_resident--;
    pg->evictions++;
    return v;
}

/* ── Open / close ────────────────────────────────────────────────────────── */

PistaPager *pista_pager_open(const char *path,
                             uint64_t region_off, uint64_t region_len,
                             uint32_t page_size, size_t cache_bytes) {
    if (!path) return NULL;
    if (page_size == 0) page_size = PG_DEFAULT_PAGE_SIZE;
    if (page_size < PG_MIN_PAGE_SIZE ||
        (page_size & (page_size - 1)) != 0) return NULL;   /* not pow2 */
    if (cache_bytes == 0) cache_bytes = PG_DEFAULT_CACHE;

    PistaPager *pg = (PistaPager *)calloc(1, sizeof(*pg));
    if (!pg) return NULL;

    pg->f = fopen(path, "rb");
    if (!pg->f) { free(pg); return NULL; }

    pg->region_off = region_off;
    pg->region_len = region_len;
    pg->page_size  = page_size;
    pg->page_mask  = page_size - 1;
    pg->page_shift = 0;
    while (((uint32_t)1 << pg->page_shift) != page_size) pg->page_shift++;

    /* Frame count: cache budget in whole pages, but never more pages than
     * the region actually has, and never fewer than the small floor so a
     * single search's transient pin set always fits. */
    uint64_t total_pages = (region_len + page_size - 1) / page_size;
    if (total_pages == 0) total_pages = 1;
    size_t want = cache_bytes / page_size;
    if (want < PG_MIN_FRAMES) want = PG_MIN_FRAMES;
    if ((uint64_t)want > total_pages) want = (size_t)total_pages;
    pg->n_frames = (int)want;

    int hsz = 16;
    while (hsz < pg->n_frames * 2) hsz <<= 1;
    pg->hmask = hsz - 1;

    pg->data = (uint8_t *)malloc((size_t)pg->n_frames * page_size);
    pg->fr   = (Frame   *)malloc((size_t)pg->n_frames * sizeof(Frame));
    pg->htab = (int     *)malloc((size_t)hsz * sizeof(int));
    if (!pg->data || !pg->fr || !pg->htab) {
        free(pg->data); free(pg->fr); free(pg->htab);
        fclose(pg->f); free(pg);
        return NULL;
    }
    for (int i = 0; i < hsz; i++) pg->htab[i] = -1;
    for (int i = 0; i < pg->n_frames; i++) {
        pg->fr[i].valid = 0;
        pg->fr[i].pin   = 0;
        pg->fr[i].h_next = -1;
        pg->fr[i].lru_prev = -1;
        pg->fr[i].lru_next = (i + 1 < pg->n_frames) ? i + 1 : -1; /* free stack */
    }
    pg->free_top = 0;
    pg->lru_head = pg->lru_tail = -1;

    pg_mutex_init(&pg->mu);
    return pg;
}

void pista_pager_close(PistaPager *pg) {
    if (!pg) return;
    pg_mutex_destroy(&pg->mu);
    if (pg->f) fclose(pg->f);
    free(pg->data);
    free(pg->fr);
    free(pg->htab);
    free(pg);
}

/* ── Pin / unpin ─────────────────────────────────────────────────────────── */

const void *pista_pager_pin(PistaPager *pg, uint64_t off, uint32_t span,
                            uint64_t *tok) {
    if (!pg || span == 0) return NULL;
    /* Range + single-page checks (overflow-safe). */
    if (off > pg->region_len || (uint64_t)span > pg->region_len - off)
        return NULL;
    uint32_t in_page = (uint32_t)(off & pg->page_mask);
    if ((uint64_t)in_page + span > pg->page_size) return NULL; /* straddles */
    uint64_t page_no = off >> pg->page_shift;

    pg_mutex_lock(&pg->mu);

    int f = pg_hash_find(pg, page_no);
    if (f != -1) {
        if (pg->fr[f].pin == 0) pg_lru_unlink(pg, f);  /* leave evictable set */
        pg->fr[f].pin++;
        pg->hits++;
        const void *ptr = pg->data + (size_t)f * pg->page_size + in_page;
        if (tok) *tok = (uint64_t)f;
        pg_mutex_unlock(&pg->mu);
        return ptr;
    }

    /* Miss: get a frame and read the page from the file. */
    f = pg_alloc_frame(pg);
    if (f == -1) { pg_mutex_unlock(&pg->mu); return NULL; }

    uint64_t base   = page_no << pg->page_shift;       /* offset within region */
    uint64_t avail  = pg->region_len - base;
    uint32_t nbytes = (avail < pg->page_size) ? (uint32_t)avail : pg->page_size;

    int io_ok = (PG_FSEEK64(pg->f, pg->region_off + base, SEEK_SET) == 0) &&
                (fread(pg->data + (size_t)f * pg->page_size, 1, nbytes, pg->f)
                     == nbytes);
    if (!io_ok) {
        /* Return the frame to the free stack and fail the pin. */
        pg->fr[f].lru_next = pg->free_top;
        pg->fr[f].lru_prev = -1;
        pg->free_top = f;
        pg_mutex_unlock(&pg->mu);
        return NULL;
    }
    if (nbytes < pg->page_size)                         /* zero short tail */
        memset(pg->data + (size_t)f * pg->page_size + nbytes, 0,
               pg->page_size - nbytes);

    pg->fr[f].page_no  = page_no;
    pg->fr[f].valid    = 1;
    pg->fr[f].pin      = 1;
    pg->fr[f].lru_prev = pg->fr[f].lru_next = -1;
    pg_hash_insert(pg, f);
    pg->n_resident++;
    pg->misses++;
    pg->fetches++;

    const void *ptr = pg->data + (size_t)f * pg->page_size + in_page;
    if (tok) *tok = (uint64_t)f;
    pg_mutex_unlock(&pg->mu);
    return ptr;
}

void pista_pager_unpin(PistaPager *pg, uint64_t tok) {
    if (!pg) return;
    int f = (int)tok;
    pg_mutex_lock(&pg->mu);
    if (f >= 0 && f < pg->n_frames && pg->fr[f].valid && pg->fr[f].pin > 0) {
        if (--pg->fr[f].pin == 0) pg_lru_push_head(pg, f); /* MRU, evictable */
    }
    pg_mutex_unlock(&pg->mu);
}

void pista_pager_stats(PistaPager *pg, PistaPagerStats *out) {
    if (!pg || !out) return;
    pg_mutex_lock(&pg->mu);
    out->hits       = pg->hits;
    out->misses     = pg->misses;
    out->evictions  = pg->evictions;
    out->fetches    = pg->fetches;
    out->n_frames   = pg->n_frames;
    out->n_resident = pg->n_resident;
    out->page_size  = pg->page_size;
    pg_mutex_unlock(&pg->mu);
}
