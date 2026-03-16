/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB - pistadb.h
 * Public C API – the only header users need to include.
 *
 * Usage:
 *   PistaDBParams p = pistadb_default_params();
 *   p.hnsw_M = 32;
 *   PistaDB *db = pistadb_open("mydb.pst", 128, METRIC_L2, INDEX_HNSW, &p);
 *   pistadb_insert(db, 1, "dog", vec);
 *   PistaDBResult results[10];
 *   int n = pistadb_search(db, query, 10, results);
 *   pistadb_save(db);
 *   pistadb_close(db);
 */
#ifndef PISTADB_H
#define PISTADB_H

#include "pistadb_types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque database handle */
typedef struct PistaDB PistaDB;

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

/**
 * Open or create a database file.
 *
 * @param path    File path.  If the file exists and is a valid PistaDB file it
 *                is loaded;  otherwise a new database is created.
 * @param dim     Vector dimension (must match existing file if loading).
 * @param metric  Distance metric.
 * @param index   Index algorithm.
 * @param params  Index parameters.  Pass NULL for defaults.
 * @return        Handle, or NULL on failure.  Call pistadb_last_error() for info.
 */
PistaDB *pistadb_open(const char *path, int dim,
                    PistaDBMetric metric, PistaDBIndexType index,
                    const PistaDBParams *params);

/** Close and free (does NOT auto-save; call pistadb_save() first). */
void pistadb_close(PistaDB *db);

/** Persist the database to its file. */
int  pistadb_save(PistaDB *db);

/* ── CRUD ────────────────────────────────────────────────────────────────── */

/**
 * Insert a vector.
 * @param id     User-supplied id (must be unique within this database).
 * @param label  Optional human-readable label (< 256 bytes).  May be NULL.
 * @param vec    Float array of length dim.
 */
int pistadb_insert(PistaDB *db, uint64_t id, const char *label, const float *vec);

/** Delete vector by id.  Logically deleted; space reclaimed on next save/rebuild. */
int pistadb_delete(PistaDB *db, uint64_t id);

/** Replace the vector data for the given id. */
int pistadb_update(PistaDB *db, uint64_t id, const float *vec);

/**
 * Get the raw vector for a given id.
 * @param out_vec  Float array of length dim, written by this function.
 * @param out_label  256-byte buffer for the label (may be NULL).
 */
int pistadb_get(PistaDB *db, uint64_t id, float *out_vec, char *out_label);

/* ── Search ──────────────────────────────────────────────────────────────── */

/**
 * K-nearest-neighbour search.
 * @param query    Float array of length dim.
 * @param k        Number of results requested.
 * @param results  Output array of at least k PistaDBResult elements.
 * @return         Actual number of results (≤ k), or negative on error.
 */
int pistadb_search(PistaDB *db, const float *query, int k,
                  PistaDBResult *results);

/* ── Index management ────────────────────────────────────────────────────── */

/**
 * Train the index on currently inserted vectors.
 * Required for IVF, IVF_PQ before calling insert.
 * Optional for HNSW/DiskANN (triggers a rebuild pass).
 */
int pistadb_train(PistaDB *db);

/** Number of active (non-deleted) vectors. */
int pistadb_count(PistaDB *db);

/* ── Metadata ────────────────────────────────────────────────────────────── */

int            pistadb_dim(PistaDB *db);
PistaDBMetric   pistadb_metric(PistaDB *db);
PistaDBIndexType pistadb_index_type(PistaDB *db);

/** Human-readable error message for the last operation.  Never NULL. */
const char *pistadb_last_error(PistaDB *db);

/* ── Version ─────────────────────────────────────────────────────────────── */
const char *pistadb_version(void);  /* e.g. "1.0.0" */

#ifdef __cplusplus
}
#endif

#endif /* PISTADB_H */
