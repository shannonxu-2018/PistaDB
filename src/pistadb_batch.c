/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - pistadb_batch.c
 * Multi-threaded batch insert implementation.
 *
 * Architecture
 * ────────────
 *
 *                 ┌─────────────────────┐
 *  Producer 0 ──▶ │                     │
 *  Producer 1 ──▶ │   Ring-buffer queue │──▶ Worker 0 ─┐
 *  Producer N ──▶ │   (bounded MPMC)    │──▶ Worker 1 ─┤──▶ pistadb_insert()
 *                 └─────────────────────┘──▶ Worker M ─┘    (serialised by
 *                                                             db_mutex)
 *
 * Multiple producer threads (callers of pistadb_batch_push) feed a bounded
 * ring-buffer queue.  Worker threads consume from the queue.  All calls to
 * pistadb_insert() are serialized by a single mutex (db_mutex), because the
 * underlying index data structures are not thread-safe.
 *
 * Performance model
 * ─────────────────
 * The primary throughput gain is pipeline parallelism: embedding generation
 * (CPU-bound, fully parallel) overlaps with sequential index writes.
 * A single worker thread suffices when the index is the bottleneck; more
 * workers help when each insert has a large lock-free preamble (e.g., IVF
 * centroid lookup or HNSW graph search) relative to the locked write portion.
 *
 * Platform threading
 * ──────────────────
 * Uses Win32 CRITICAL_SECTION + CONDITION_VARIABLE on Windows,
 * and pthreads on every other platform (Linux, macOS, BSDs, Android, iOS).
 * No external dependencies.
 */

#include "pistadb_batch.h"
#include <stdlib.h>
#include <string.h>

/* ── Platform thread / mutex / cond abstraction ──────────────────────────── */

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>

typedef HANDLE             pdb_thread_t;
typedef CRITICAL_SECTION   pdb_mutex_t;
typedef CONDITION_VARIABLE pdb_cond_t;

static void pdb_mutex_init   (pdb_mutex_t *m) { InitializeCriticalSection(m); }
static void pdb_mutex_lock   (pdb_mutex_t *m) { EnterCriticalSection(m); }
static void pdb_mutex_unlock (pdb_mutex_t *m) { LeaveCriticalSection(m); }
static void pdb_mutex_destroy(pdb_mutex_t *m) { DeleteCriticalSection(m); }

static void pdb_cond_init     (pdb_cond_t *c) { InitializeConditionVariable(c); }
static void pdb_cond_destroy  (pdb_cond_t *c) { (void)c; /* no-op on Win32 */ }
static void pdb_cond_wait     (pdb_cond_t *c, pdb_mutex_t *m)
    { SleepConditionVariableCS(c, m, INFINITE); }
static void pdb_cond_signal   (pdb_cond_t *c) { WakeConditionVariable(c); }
static void pdb_cond_broadcast(pdb_cond_t *c) { WakeAllConditionVariable(c); }

static int hw_concurrency(void) {
    SYSTEM_INFO si; GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
}

static DWORD WINAPI win32_worker_entry(LPVOID arg);

static int  pdb_thread_start(pdb_thread_t *t, void *arg) {
    *t = CreateThread(NULL, 0, win32_worker_entry, arg, 0, NULL);
    return (*t == NULL) ? -1 : 0;
}
static void pdb_thread_join(pdb_thread_t t) {
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
}

#else  /* POSIX */
#  include <pthread.h>
#  if defined(__linux__)
#    include <unistd.h>
#  elif defined(__APPLE__)
#    include <sys/sysctl.h>
#  endif

typedef pthread_t        pdb_thread_t;
typedef pthread_mutex_t  pdb_mutex_t;
typedef pthread_cond_t   pdb_cond_t;

static void pdb_mutex_init   (pdb_mutex_t *m) { pthread_mutex_init(m, NULL); }
static void pdb_mutex_lock   (pdb_mutex_t *m) { pthread_mutex_lock(m); }
static void pdb_mutex_unlock (pdb_mutex_t *m) { pthread_mutex_unlock(m); }
static void pdb_mutex_destroy(pdb_mutex_t *m) { pthread_mutex_destroy(m); }

static void pdb_cond_init     (pdb_cond_t *c) { pthread_cond_init(c, NULL); }
static void pdb_cond_destroy  (pdb_cond_t *c) { pthread_cond_destroy(c); }
static void pdb_cond_wait     (pdb_cond_t *c, pdb_mutex_t *m)
    { pthread_cond_wait(c, m); }
static void pdb_cond_signal   (pdb_cond_t *c) { pthread_cond_signal(c); }
static void pdb_cond_broadcast(pdb_cond_t *c) { pthread_cond_broadcast(c); }

static int hw_concurrency(void) {
#  if defined(_SC_NPROCESSORS_ONLN)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
#  elif defined(__APPLE__)
    int n = 1; size_t len = sizeof(n);
    sysctlbyname("hw.logicalcpu", &n, &len, NULL, 0);
    return n;
#  else
    return 1;
#  endif
}

static void *posix_worker_entry(void *arg);

static int  pdb_thread_start(pdb_thread_t *t, void *arg) {
    return pthread_create(t, NULL, posix_worker_entry, arg);
}
static void pdb_thread_join(pdb_thread_t t) { pthread_join(t, NULL); }

#endif  /* POSIX */

/* ── Constants ────────────────────────────────────────────────────────────── */

#define BATCH_DEFAULT_CAP   4096
#define BATCH_MAX_THREADS     32

/* ── Work queue item ─────────────────────────────────────────────────────── */

typedef struct {
    uint64_t  id;
    char      label[256];   /* empty string → no label */
    float    *vec;          /* heap-allocated copy; freed by the worker */
} BatchItem;

/* ── PistaDBBatch struct ──────────────────────────────────────────────────── */

struct PistaDBBatch {
    PistaDB  *db;
    int       dim;
    int       n_threads;

    /* ── Ring-buffer work queue ───────────────────────────────────────────── */
    BatchItem **ring;       /* circular buffer of (BatchItem *) */
    int         cap;        /* capacity  (power of 2 not required) */
    int         head;       /* consumer: next pop index            */
    int         tail;       /* producer: next push index           */
    int         count;      /* items waiting in ring               */
    int         in_flight;  /* items dequeued but insert not done  */

    pdb_mutex_t q_mu;
    pdb_cond_t  q_not_empty;  /* workers wait here when queue is empty */
    pdb_cond_t  q_not_full;   /* producers wait here when queue is full */
    pdb_cond_t  q_all_done;   /* flush waiter wakes when count+in_flight == 0 */

    /* ── Index write serializer ───────────────────────────────────────────── */
    pdb_mutex_t db_mu;        /* gates all pistadb_insert() calls */

    /* ── Control ──────────────────────────────────────────────────────────── */
    int  shutdown;            /* set to 1 to drain and exit workers */
    int  error_count;         /* cumulative insert failures (all time) */
    int  flush_errors;        /* errors since last pistadb_batch_flush() */

    pdb_thread_t threads[BATCH_MAX_THREADS];
};

/* ── Worker logic ─────────────────────────────────────────────────────────── */

static void worker_body(PistaDBBatch *b) {
    for (;;) {
        /* ── Wait for work ───────────────────────────────────────────────── */
        pdb_mutex_lock(&b->q_mu);
        while (b->count == 0 && !b->shutdown)
            pdb_cond_wait(&b->q_not_empty, &b->q_mu);

        if (b->count == 0) {
            /* Queue is empty and shutdown was requested — exit. */
            pdb_mutex_unlock(&b->q_mu);
            return;
        }

        /* ── Dequeue one item ────────────────────────────────────────────── */
        BatchItem *item = b->ring[b->head];
        b->head = (b->head + 1) % b->cap;
        b->count--;
        b->in_flight++;
        pdb_mutex_unlock(&b->q_mu);

        /* Unblock any producer that was waiting for a free slot. */
        pdb_cond_signal(&b->q_not_full);

        /* ── Insert (serialized across all workers) ──────────────────────── */
        const char *lbl = (item->label[0] != '\0') ? item->label : NULL;
        pdb_mutex_lock(&b->db_mu);
        int rc = pistadb_insert(b->db, item->id, lbl, item->vec);
        pdb_mutex_unlock(&b->db_mu);

        free(item->vec);
        free(item);

        /* ── Update counters and signal flush waiter ─────────────────────── */
        pdb_mutex_lock(&b->q_mu);
        b->in_flight--;
        if (rc != PISTADB_OK) {
            b->error_count++;
            b->flush_errors++;
        }
        /* Broadcast when queue is fully drained (including in-flight). */
        if (b->count == 0 && b->in_flight == 0)
            pdb_cond_broadcast(&b->q_all_done);
        pdb_mutex_unlock(&b->q_mu);
    }
}

/* Platform-specific entry point wrappers */
#if defined(_WIN32)
static DWORD WINAPI win32_worker_entry(LPVOID arg) {
    worker_body((PistaDBBatch *)arg); return 0;
}
#else
static void *posix_worker_entry(void *arg) {
    worker_body((PistaDBBatch *)arg); return NULL;
}
#endif

/* ── pistadb_batch_create ─────────────────────────────────────────────────── */

PistaDBBatch *pistadb_batch_create(PistaDB *db, int n_threads, int queue_cap) {
    if (!db) return NULL;

    if (n_threads <= 0) n_threads = hw_concurrency();
    if (n_threads  > BATCH_MAX_THREADS) n_threads = BATCH_MAX_THREADS;
    if (queue_cap  <= 0) queue_cap = BATCH_DEFAULT_CAP;

    PistaDBBatch *b = (PistaDBBatch *)calloc(1, sizeof(*b));
    if (!b) return NULL;

    b->ring = (BatchItem **)malloc((size_t)queue_cap * sizeof(BatchItem *));
    if (!b->ring) { free(b); return NULL; }

    b->db       = db;
    b->dim      = pistadb_dim(db);
    b->n_threads = n_threads;
    b->cap      = queue_cap;

    pdb_mutex_init(&b->q_mu);
    pdb_cond_init(&b->q_not_empty);
    pdb_cond_init(&b->q_not_full);
    pdb_cond_init(&b->q_all_done);
    pdb_mutex_init(&b->db_mu);

    /* Start worker threads */
    int started = 0;
    for (int i = 0; i < n_threads; i++) {
        if (pdb_thread_start(&b->threads[i], b) != 0) break;
        started++;
    }

    if (started == 0) {
        /* Could not start any thread — clean up */
        pdb_cond_destroy(&b->q_not_empty);
        pdb_cond_destroy(&b->q_not_full);
        pdb_cond_destroy(&b->q_all_done);
        pdb_mutex_destroy(&b->q_mu);
        pdb_mutex_destroy(&b->db_mu);
        free(b->ring);
        free(b);
        return NULL;
    }

    /* Adjust to the actual number of threads started */
    b->n_threads = started;
    return b;
}

/* ── pistadb_batch_push ──────────────────────────────────────────────────── */

int pistadb_batch_push(PistaDBBatch *b,
                       uint64_t      id,
                       const char   *label,
                       const float  *vec) {
    if (!b || !vec) return PISTADB_EINVAL;

    /* Allocate and populate the work item outside the lock. */
    BatchItem *item = (BatchItem *)malloc(sizeof(*item));
    if (!item) return PISTADB_ENOMEM;

    item->vec = (float *)malloc((size_t)b->dim * sizeof(float));
    if (!item->vec) { free(item); return PISTADB_ENOMEM; }

    item->id = id;
    memcpy(item->vec, vec, (size_t)b->dim * sizeof(float));

    if (label && label[0]) {
        strncpy(item->label, label, 255);
        item->label[255] = '\0';
    } else {
        item->label[0] = '\0';
    }

    /* Enqueue — block if the ring buffer is full (back-pressure). */
    pdb_mutex_lock(&b->q_mu);
    while (b->count >= b->cap && !b->shutdown)
        pdb_cond_wait(&b->q_not_full, &b->q_mu);

    if (b->shutdown) {
        pdb_mutex_unlock(&b->q_mu);
        free(item->vec);
        free(item);
        return PISTADB_ERR;
    }

    b->ring[b->tail] = item;
    b->tail = (b->tail + 1) % b->cap;
    b->count++;
    pdb_mutex_unlock(&b->q_mu);

    /* Wake one worker. */
    pdb_cond_signal(&b->q_not_empty);
    return PISTADB_OK;
}

/* ── pistadb_batch_flush ─────────────────────────────────────────────────── */

int pistadb_batch_flush(PistaDBBatch *b) {
    if (!b) return 0;

    pdb_mutex_lock(&b->q_mu);
    /* Wait until the queue is empty AND no items are being processed. */
    while (b->count > 0 || b->in_flight > 0)
        pdb_cond_wait(&b->q_all_done, &b->q_mu);

    int errs = b->flush_errors;
    b->flush_errors = 0;
    pdb_mutex_unlock(&b->q_mu);
    return errs;
}

/* ── pistadb_batch_error_count ────────────────────────────────────────────── */

int pistadb_batch_error_count(PistaDBBatch *b) {
    if (!b) return 0;
    pdb_mutex_lock(&b->q_mu);
    int n = b->error_count;
    pdb_mutex_unlock(&b->q_mu);
    return n;
}

/* ── pistadb_batch_destroy ────────────────────────────────────────────────── */

void pistadb_batch_destroy(PistaDBBatch *b) {
    if (!b) return;

    /* Drain any remaining items first. */
    pistadb_batch_flush(b);

    /* Signal all workers to exit once the queue stays empty. */
    pdb_mutex_lock(&b->q_mu);
    b->shutdown = 1;
    pdb_cond_broadcast(&b->q_not_empty);
    pdb_mutex_unlock(&b->q_mu);

    for (int i = 0; i < b->n_threads; i++)
        pdb_thread_join(b->threads[i]);

    /* Free any items that might have been push()ed after flush but before
       shutdown (shouldn't happen with correct usage, but be safe). */
    while (b->count > 0) {
        BatchItem *item = b->ring[b->head];
        b->head = (b->head + 1) % b->cap;
        b->count--;
        free(item->vec);
        free(item);
    }

    pdb_cond_destroy(&b->q_not_empty);
    pdb_cond_destroy(&b->q_not_full);
    pdb_cond_destroy(&b->q_all_done);
    pdb_mutex_destroy(&b->q_mu);
    pdb_mutex_destroy(&b->db_mu);
    free(b->ring);
    free(b);
}

/* ── pistadb_batch_insert (convenience) ───────────────────────────────────── */

int pistadb_batch_insert(PistaDB            *db,
                         const uint64_t     *ids,
                         const char * const *labels,
                         const float        *vecs,
                         int                 n,
                         int                 n_threads) {
    if (!db || !ids || !vecs || n <= 0) return 0;

    /* Cap the queue to n so pistadb_batch_flush() wakes as soon as possible. */
    int cap = (n < BATCH_DEFAULT_CAP) ? n : BATCH_DEFAULT_CAP;

    PistaDBBatch *b = pistadb_batch_create(db, n_threads, cap);
    if (!b) return n;   /* could not create context — treat all as failed */

    const int dim = b->dim;
    for (int i = 0; i < n; i++) {
        const char  *lbl = (labels && labels[i]) ? labels[i] : NULL;
        const float *vec = vecs + (size_t)i * (size_t)dim;
        /* push() blocks if the queue is full — natural back-pressure. */
        pistadb_batch_push(b, ids[i], lbl, vec);
    }

    int errors = pistadb_batch_flush(b);
    pistadb_batch_destroy(b);
    return errors;
}
