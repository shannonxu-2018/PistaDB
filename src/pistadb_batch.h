/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - pistadb_batch.h
 * Multi-threaded batch insert API for high-throughput embedding pipelines.
 *
 * Two usage patterns are provided:
 *
 * ── Pattern A: streaming (online pipeline) ────────────────────────────────
 *
 *   PistaDBBatch *b = pistadb_batch_create(db, 0, 0); // auto threads/cap
 *
 *   // Any number of producer threads may call push() concurrently:
 *   pistadb_batch_push(b, id, label, vec);
 *
 *   // Block until all pushed items are indexed:
 *   int errors = pistadb_batch_flush(b);
 *   pistadb_batch_destroy(b);
 *
 * ── Pattern B: offline bulk (all vectors available upfront) ───────────────
 *
 *   // Blocking – returns when all n inserts are complete:
 *   int errors = pistadb_batch_insert(db, ids, labels, vecs, n, 0);
 *
 * ── Thread safety ─────────────────────────────────────────────────────────
 *
 *   pistadb_batch_push()  is thread-safe – call from any number of threads.
 *   All other batch functions must be called from a single "owner" thread.
 *
 *   Index writes are serialized internally; the underlying PistaDB index does
 *   NOT need to be thread-safe itself.  Do not mix pistadb_insert() / other
 *   direct calls to `db` while a batch context is active on the same handle.
 */
#ifndef PISTADB_BATCH_H
#define PISTADB_BATCH_H

#include "pistadb.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque handle ────────────────────────────────────────────────────────── */

/** Opaque batch-insert context owning a thread pool and work queue. */
typedef struct PistaDBBatch PistaDBBatch;

/* ── Streaming API ────────────────────────────────────────────────────────── */

/**
 * Create a batch-insert context.
 *
 * @param db        Target database.  Must remain open until pistadb_batch_destroy().
 * @param n_threads Worker thread count.  0 → auto (hardware_concurrency, max 32).
 * @param queue_cap Work-queue capacity (items in-flight at once).
 *                  0 → default (4096).
 *                  pistadb_batch_push() blocks when the queue is full.
 * @return          Batch context, or NULL on allocation / thread-create failure.
 */
PistaDBBatch *pistadb_batch_create(PistaDB *db, int n_threads, int queue_cap);

/**
 * Push one item onto the work queue.
 *
 * Thread-safe.  Any number of producer threads may call this simultaneously.
 * Blocks (with back-pressure) if the queue is at capacity.
 * The vector data is copied internally; the caller may free/reuse `vec`
 * immediately after this function returns.
 *
 * @return PISTADB_OK on success, PISTADB_ENOMEM on allocation failure,
 *         PISTADB_ERR if the batch has been shut down.
 */
int pistadb_batch_push(PistaDBBatch *b,
                       uint64_t      id,
                       const char   *label,  /* may be NULL */
                       const float  *vec);

/**
 * Wait until every previously-pushed item has been inserted.
 *
 * Resets the per-flush error counter.
 *
 * @return Number of failed inserts since the last flush (0 on full success).
 */
int pistadb_batch_flush(PistaDBBatch *b);

/**
 * Total number of insert errors accumulated since pistadb_batch_create().
 * (Not reset by pistadb_batch_flush.)
 */
int pistadb_batch_error_count(PistaDBBatch *b);

/**
 * Flush remaining items, shut down worker threads, and free all memory.
 * Safe to call even if pistadb_batch_flush() was not called first.
 */
void pistadb_batch_destroy(PistaDBBatch *b);

/* ── Convenience: offline bulk insert ────────────────────────────────────── */

/**
 * Insert an array of n vectors using a temporary thread pool.  Blocking.
 *
 * Internally creates a PistaDBBatch, pushes all items, flushes, and destroys
 * the context before returning.
 *
 * @param db        Target database.
 * @param ids       Array of n unique ids.
 * @param labels    Array of n label strings, or NULL for no labels.
 *                  Individual elements may also be NULL.
 * @param vecs      Row-major float array of shape [n × dim(db)].
 * @param n         Number of vectors to insert.
 * @param n_threads Worker count.  0 → auto.
 * @return          Number of failed inserts (0 on full success).
 */
int pistadb_batch_insert(PistaDB            *db,
                         const uint64_t     *ids,
                         const char * const *labels,
                         const float        *vecs,
                         int                 n,
                         int                 n_threads);

#ifdef __cplusplus
}
#endif

#endif /* PISTADB_BATCH_H */
