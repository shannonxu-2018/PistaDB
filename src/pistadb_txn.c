/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - pistadb_txn.c
 * Transaction implementation.
 *
 * Each transaction holds a dynamic array of TxnOp descriptors. Staging
 * functions populate this array without touching the database. Commit()
 * validates and then applies the ops, keeping undo data to reverse any
 * already-applied steps if a later step fails.
 */

#include "pistadb_txn.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Internal operation descriptor ──────────────────────────────────────── */

typedef struct {
    int      type;          /* PISTADB_TXN_{INSERT,DELETE,UPDATE}            */
    uint64_t id;
    char     label[256];    /* used for INSERT (and as undo label for DELETE) */
    float   *vec;           /* heap copy of new vector; NULL for staged DELETE */

    /* Undo data — captured at staging time via pistadb_get().
       For INSERT: undo = DELETE(id) — no vec needed (has_undo = 1 always).
       For DELETE: undo = INSERT(id, undo_label, undo_vec) — needs snapshot.
       For UPDATE: undo = UPDATE(id, undo_vec)              — needs snapshot.  */
    int      has_undo;      /* 1 = undo_vec (and undo_label for DELETE) valid */
    float   *undo_vec;      /* original vector before this operation          */
    char     undo_label[256];
} TxnOp;

/* ── Opaque struct ───────────────────────────────────────────────────────── */

struct PistaDBTxn {
    PistaDB *db;
    int      dim;
    TxnOp   *ops;
    int      n_ops;
    int      cap_ops;
    char     last_err[512];
};

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static void txn_set_err(PistaDBTxn *txn, const char *msg)
{
    strncpy(txn->last_err, msg ? msg : "", sizeof(txn->last_err) - 1);
    txn->last_err[sizeof(txn->last_err) - 1] = '\0';
}

static int txn_grow(PistaDBTxn *txn)
{
    int new_cap = (txn->cap_ops == 0) ? 8 : txn->cap_ops * 2;
    TxnOp *p = (TxnOp *)realloc(txn->ops, (size_t)new_cap * sizeof(TxnOp));
    if (!p) return PISTADB_ENOMEM;
    txn->ops     = p;
    txn->cap_ops = new_cap;
    return PISTADB_OK;
}

/* Free heap data inside ops[0..n-1] (does NOT free the ops array itself). */
static void free_op_data(TxnOp *ops, int n)
{
    for (int i = 0; i < n; i++) {
        free(ops[i].vec);
        free(ops[i].undo_vec);
        ops[i].vec      = NULL;
        ops[i].undo_vec = NULL;
    }
}

/* ── pistadb_txn_begin ───────────────────────────────────────────────────── */

PistaDBTxn *pistadb_txn_begin(PistaDB *db)
{
    if (!db) return NULL;
    PistaDBTxn *txn = (PistaDBTxn *)calloc(1, sizeof(*txn));
    if (!txn) return NULL;
    txn->db  = db;
    txn->dim = pistadb_dim(db);
    return txn;
}

/* ── pistadb_txn_rollback ────────────────────────────────────────────────── */

void pistadb_txn_rollback(PistaDBTxn *txn)
{
    if (!txn) return;
    free_op_data(txn->ops, txn->n_ops);
    free(txn->ops);
    txn->ops     = NULL;
    txn->n_ops   = 0;
    txn->cap_ops = 0;
}

/* ── pistadb_txn_free ────────────────────────────────────────────────────── */

void pistadb_txn_free(PistaDBTxn *txn)
{
    if (!txn) return;
    pistadb_txn_rollback(txn);
    free(txn);
}

/* ── pistadb_txn_insert ──────────────────────────────────────────────────── */

int pistadb_txn_insert(PistaDBTxn *txn, uint64_t id, const char *label,
                       const float *vec)
{
    if (!txn || !vec) return PISTADB_EINVAL;
    if (txn->n_ops >= txn->cap_ops) {
        int r = txn_grow(txn);
        if (r != PISTADB_OK) { txn_set_err(txn, "out of memory"); return r; }
    }

    float *vcopy = (float *)malloc((size_t)txn->dim * sizeof(float));
    if (!vcopy) { txn_set_err(txn, "out of memory"); return PISTADB_ENOMEM; }
    memcpy(vcopy, vec, (size_t)txn->dim * sizeof(float));

    TxnOp *op = &txn->ops[txn->n_ops++];
    memset(op, 0, sizeof(*op));
    op->type     = PISTADB_TXN_INSERT;
    op->id       = id;
    op->vec      = vcopy;
    op->has_undo = 1;   /* undo = DELETE — no vec snapshot needed */
    if (label && label[0]) {
        strncpy(op->label, label, 255);
        op->label[255] = '\0';
    }
    return PISTADB_OK;
}

/* ── pistadb_txn_delete ──────────────────────────────────────────────────── */

int pistadb_txn_delete(PistaDBTxn *txn, uint64_t id)
{
    if (!txn) return PISTADB_EINVAL;
    if (txn->n_ops >= txn->cap_ops) {
        int r = txn_grow(txn);
        if (r != PISTADB_OK) { txn_set_err(txn, "out of memory"); return r; }
    }

    /* Attempt to snapshot the current state for undo.
       pistadb_get() may return PISTADB_ENOTFOUND for index types that do not
       store raw vectors (e.g. IVF_PQ) — we stage the op anyway and mark the
       undo as unavailable.                                                   */
    float *snap      = (float *)malloc((size_t)txn->dim * sizeof(float));
    int    has_snap  = 0;
    char   snap_lbl[256] = {0};

    if (snap) {
        int r = pistadb_get(txn->db, id, snap, snap_lbl);
        if (r == PISTADB_OK) has_snap = 1;
    }

    TxnOp *op = &txn->ops[txn->n_ops++];
    memset(op, 0, sizeof(*op));
    op->type     = PISTADB_TXN_DELETE;
    op->id       = id;
    op->vec      = NULL;
    op->has_undo = has_snap;
    op->undo_vec = has_snap ? snap : NULL;
    if (has_snap) {
        strncpy(op->undo_label, snap_lbl, 255);
        op->undo_label[255] = '\0';
    } else {
        free(snap);   /* allocation succeeded but get() failed */
    }
    return PISTADB_OK;
}

/* ── pistadb_txn_update ──────────────────────────────────────────────────── */

int pistadb_txn_update(PistaDBTxn *txn, uint64_t id, const float *vec)
{
    if (!txn || !vec) return PISTADB_EINVAL;
    if (txn->n_ops >= txn->cap_ops) {
        int r = txn_grow(txn);
        if (r != PISTADB_OK) { txn_set_err(txn, "out of memory"); return r; }
    }

    float *vcopy = (float *)malloc((size_t)txn->dim * sizeof(float));
    if (!vcopy) { txn_set_err(txn, "out of memory"); return PISTADB_ENOMEM; }
    memcpy(vcopy, vec, (size_t)txn->dim * sizeof(float));

    /* Snapshot original for undo. */
    float *snap     = (float *)malloc((size_t)txn->dim * sizeof(float));
    int    has_snap = 0;
    if (snap) {
        int r = pistadb_get(txn->db, id, snap, NULL);
        if (r == PISTADB_OK) has_snap = 1;
    }

    TxnOp *op = &txn->ops[txn->n_ops++];
    memset(op, 0, sizeof(*op));
    op->type     = PISTADB_TXN_UPDATE;
    op->id       = id;
    op->vec      = vcopy;
    op->has_undo = has_snap;
    op->undo_vec = has_snap ? snap : NULL;
    if (!has_snap) free(snap);
    return PISTADB_OK;
}

/* ── pistadb_txn_op_count / last_error ──────────────────────────────────── */

int pistadb_txn_op_count(PistaDBTxn *txn)
{
    return txn ? txn->n_ops : 0;
}

const char *pistadb_txn_last_error(PistaDBTxn *txn)
{
    return (txn && txn->last_err[0]) ? txn->last_err : "";
}

/* ── pistadb_txn_commit ──────────────────────────────────────────────────── */

int pistadb_txn_commit(PistaDBTxn *txn)
{
    if (!txn) return PISTADB_EINVAL;
    if (txn->n_ops == 0) return PISTADB_OK;

    PistaDB *db = txn->db;

    /* ── Phase 1: structural validation ─────────────────────────────────── */

    for (int i = 0; i < txn->n_ops; i++) {
        TxnOp *op = &txn->ops[i];
        if (op->type == PISTADB_TXN_INSERT) {
            /* Disallow duplicate INSERT ids within the same transaction. */
            for (int j = 0; j < i; j++) {
                if (txn->ops[j].type == PISTADB_TXN_INSERT &&
                    txn->ops[j].id   == op->id) {
                    char buf[256];
                    snprintf(buf, sizeof(buf),
                             "duplicate INSERT id %llu in transaction",
                             (unsigned long long)op->id);
                    txn_set_err(txn, buf);
                    return PISTADB_EEXIST;
                }
            }
        }
    }

    /* ── Phase 2: apply ──────────────────────────────────────────────────── */

    int applied   = 0;
    int final_rc  = PISTADB_OK;

    for (int i = 0; i < txn->n_ops; i++) {
        TxnOp      *op  = &txn->ops[i];
        const char *lbl = (op->label[0] != '\0') ? op->label : NULL;
        int         r;

        switch (op->type) {
            case PISTADB_TXN_INSERT:
                r = pistadb_insert(db, op->id, lbl, op->vec);
                break;
            case PISTADB_TXN_DELETE:
                r = pistadb_delete(db, op->id);
                break;
            case PISTADB_TXN_UPDATE:
                r = pistadb_update(db, op->id, op->vec);
                break;
            default:
                r = PISTADB_EINVAL;
        }

        if (r != PISTADB_OK) {
            /* Build an error message. */
            char buf[512];
            snprintf(buf, sizeof(buf),
                     "op[%d] type=%d id=%llu failed (code=%d): %s",
                     i, op->type, (unsigned long long)op->id, r,
                     pistadb_last_error(db));
            txn_set_err(txn, buf);

            /* ── Rollback applied ops in reverse order ─────────────────── */
            int partial = 0;
            for (int j = applied - 1; j >= 0; j--) {
                TxnOp *prev = &txn->ops[j];
                int    ur;
                switch (prev->type) {
                    case PISTADB_TXN_INSERT:
                        /* Undo: delete the just-inserted vector. */
                        ur = pistadb_delete(db, prev->id);
                        break;
                    case PISTADB_TXN_DELETE:
                        if (prev->has_undo) {
                            const char *ulbl = prev->undo_label[0]
                                               ? prev->undo_label : NULL;
                            ur = pistadb_insert(db, prev->id, ulbl,
                                                prev->undo_vec);
                        } else {
                            ur = PISTADB_ERR;
                            partial = 1;
                        }
                        break;
                    case PISTADB_TXN_UPDATE:
                        if (prev->has_undo) {
                            ur = pistadb_update(db, prev->id, prev->undo_vec);
                        } else {
                            ur = PISTADB_ERR;
                            partial = 1;
                        }
                        break;
                    default:
                        ur = PISTADB_ERR;
                        partial = 1;
                        break;
                }
                if (ur != PISTADB_OK) partial = 1;
            }
            final_rc = partial ? PISTADB_ETXN_PARTIAL : PISTADB_ERR;
            break;  /* stop applying further ops */
        }
        applied++;
    }

    /* ── Clear staged ops (handle reusable after commit) ─────────────────── */
    free_op_data(txn->ops, txn->n_ops);
    free(txn->ops);
    txn->ops     = NULL;
    txn->n_ops   = 0;
    txn->cap_ops = 0;

    return final_rc;
}
