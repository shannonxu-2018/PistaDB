/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - pistadb_txn.h
 * Lightweight transaction API.
 *
 * Semantics
 * ─────────
 * A transaction buffers INSERT, DELETE, and UPDATE operations without
 * modifying the database until pistadb_txn_commit() is called.
 *
 * Commit uses a two-phase approach:
 *
 *   Phase 1 – Validation
 *     Check for structural errors within the staged op list (e.g. duplicate
 *     INSERT ids). If validation fails the database is untouched and a
 *     negative error code is returned.
 *
 *   Phase 2 – Apply
 *     Execute operations in staging order. If an individual operation fails
 *     (e.g. out-of-memory or id-not-found), all already-applied operations
 *     are rolled back using internally-saved undo snapshots:
 *
 *       • Undo INSERT  → pistadb_delete(id)
 *       • Undo DELETE  → pistadb_insert(id, original_label, original_vec)
 *       • Undo UPDATE  → pistadb_update(id, original_vec)
 *
 * Undo data availability
 * ──────────────────────
 * For DELETE and UPDATE the original vector is snapshotted at staging time
 * via pistadb_get(). This works for LINEAR, HNSW, IVF, DiskANN, and LSH.
 * For IVF_PQ (which does not store raw vectors) and ScaNN the snapshot will
 * not be available; if rollback is needed for such an operation,
 * PISTADB_ETXN_PARTIAL is returned to signal that the rollback was incomplete.
 *
 * Thread safety
 * ─────────────
 * A PistaDBTxn handle is NOT thread-safe. Use one transaction per thread,
 * or protect concurrent access externally. Concurrent transactions on the
 * same database are not supported.
 */
#ifndef PISTADB_TXN_H
#define PISTADB_TXN_H

#include "pistadb.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Additional error codes ──────────────────────────────────────────────── */

/** Commit partially succeeded; some rollback steps could not be completed. */
#define PISTADB_ETXN_PARTIAL  -10

/* ── Operation type constants ────────────────────────────────────────────── */

#define PISTADB_TXN_INSERT  0
#define PISTADB_TXN_DELETE  1
#define PISTADB_TXN_UPDATE  2

/* ── Opaque transaction handle ───────────────────────────────────────────── */

typedef struct PistaDBTxn PistaDBTxn;

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

/**
 * Begin a new transaction on `db`.
 *
 * @param db  Open database handle.  Must remain valid until pistadb_txn_free().
 * @return    Transaction handle, or NULL on allocation failure.
 */
PistaDBTxn *pistadb_txn_begin(PistaDB *db);

/**
 * Validate and apply all staged operations atomically.
 *
 * @return  PISTADB_OK on full success.
 *          A negative error code (other than PISTADB_ETXN_PARTIAL) if
 *            validation fails — no operations are applied.
 *          PISTADB_ERR if apply partially failed and a complete rollback
 *            succeeded — the database is restored to its pre-commit state.
 *          PISTADB_ETXN_PARTIAL if apply partially failed and rollback was
 *            incomplete (some operations lacking undo data were not reversed).
 *
 * After this call the staged operation list is cleared regardless of outcome,
 * so the handle may be reused or freed.
 */
int pistadb_txn_commit(PistaDBTxn *txn);

/**
 * Discard all staged operations without touching the database.
 * The handle remains valid and may be reused for a new transaction.
 */
void pistadb_txn_rollback(PistaDBTxn *txn);

/**
 * Release all resources.  Implies rollback if not yet committed.
 */
void pistadb_txn_free(PistaDBTxn *txn);

/* ── Staging operations ──────────────────────────────────────────────────── */

/**
 * Stage an insert.
 * `vec` is copied; the caller may free it immediately after this returns.
 *
 * @return  PISTADB_OK, PISTADB_ENOMEM, or PISTADB_EINVAL.
 */
int pistadb_txn_insert(PistaDBTxn *txn, uint64_t id, const char *label,
                       const float *vec);

/**
 * Stage a delete.
 * Attempts to snapshot the original vector+label for undo purposes.
 *
 * @return  PISTADB_OK on success (id may or may not exist yet — checked at
 *          commit time), PISTADB_ENOMEM on allocation failure.
 */
int pistadb_txn_delete(PistaDBTxn *txn, uint64_t id);

/**
 * Stage an update (replace the vector data for the given id).
 * Attempts to snapshot the original vector for undo purposes.
 * `vec` is copied; the caller may free it immediately.
 *
 * @return  PISTADB_OK, PISTADB_ENOMEM, or PISTADB_EINVAL.
 */
int pistadb_txn_update(PistaDBTxn *txn, uint64_t id, const float *vec);

/* ── Introspection ───────────────────────────────────────────────────────── */

/** Number of staged (uncommitted) operations. */
int         pistadb_txn_op_count(PistaDBTxn *txn);

/** Last error message.  Never NULL. */
const char *pistadb_txn_last_error(PistaDBTxn *txn);

#ifdef __cplusplus
}
#endif

#endif /* PISTADB_TXN_H */
