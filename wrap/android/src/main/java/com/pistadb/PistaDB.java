/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
package com.pistadb;

import java.io.Closeable;

/**
 * PistaDB – embedded vector database for Android.
 *
 * <p>Wraps the native C library through JNI.  Each instance holds an opaque
 * handle to a {@code PistaDB*} allocated on the native heap; call
 * {@link #close()} (or use try-with-resources) to free it.
 *
 * <h3>Quick-start (Java)</h3>
 * <pre>{@code
 * try (PistaDB db = new PistaDB(path, 384)) {
 *     db.insert(1L, embedding, "my document");
 *     SearchResult[] hits = db.search(queryVec, 5);
 *     db.save();
 * }
 * }</pre>
 *
 * <h3>Quick-start (Kotlin DSL)</h3>
 * <pre>{@code
 * val db = pistaDB(path, dim = 384) {
 *     metric    = Metric.COSINE
 *     indexType = IndexType.HNSW
 *     params.hnswEfSearch = 100
 * }
 * }</pre>
 *
 * <p><b>Thread safety:</b> all public methods are {@code synchronized}.
 * Concurrent access from multiple threads is safe but serialised.
 */
public class PistaDB implements Closeable {

    static {
        System.loadLibrary("pistadb_jni");
    }

    /* ── Native method declarations ──────────────────────────────────────── */

    private static native long   nativeOpen(String path, int dim,
                                            int metric, int indexType,
                                            PistaDBParams params);
    private static native void   nativeClose(long handle);
    private static native void   nativeSave(long handle);
    private static native void   nativeInsert(long handle, long id,
                                              String label, float[] vec);
    private static native void   nativeDelete(long handle, long id);
    private static native void   nativeUpdate(long handle, long id, float[] vec);
    private static native VectorEntry    nativeGet(long handle, long id);
    private static native SearchResult[] nativeSearch(long handle,
                                                      float[] query, int k);
    private static native void   nativeTrain(long handle);
    private static native int    nativeCount(long handle);
    private static native int    nativeDim(long handle);
    private static native int    nativeMetric(long handle);
    private static native int    nativeIndexType(long handle);
    private static native String nativeLastError(long handle);
    private static native String nativeVersion();

    /* ── State ───────────────────────────────────────────────────────────── */

    private volatile long nativeHandle;

    /* ── Construction ────────────────────────────────────────────────────── */

    /**
     * Opens or creates a database with default parameters and HNSW index.
     *
     * @param path File path for the {@code .pst} file.
     * @param dim  Vector dimensionality.
     */
    public PistaDB(String path, int dim) {
        this(path, dim, Metric.L2, IndexType.HNSW, null);
    }

    /**
     * Opens or creates a database.
     *
     * @param path      File path for the {@code .pst} file.
     * @param dim       Vector dimensionality.
     * @param metric    Distance metric.
     * @param indexType Index algorithm.
     * @param params    Tuning parameters; pass {@code null} for defaults.
     * @throws PistaDBException if the file cannot be opened or created.
     */
    public PistaDB(String path, int dim, Metric metric,
                   IndexType indexType, PistaDBParams params) {
        if (path == null)   throw new IllegalArgumentException("path must not be null");
        if (dim <= 0)       throw new IllegalArgumentException("dim must be > 0");
        if (metric == null) metric = Metric.L2;
        if (indexType == null) indexType = IndexType.HNSW;

        nativeHandle = nativeOpen(path, dim,
                                  metric.getValue(), indexType.getValue(),
                                  params != null ? params : PistaDBParams.defaults());
        if (nativeHandle == 0) {
            throw new PistaDBException("Failed to open database at: " + path);
        }
    }

    /* ── Lifecycle ───────────────────────────────────────────────────────── */

    /**
     * Saves the database to disk.  Does NOT close the handle.
     *
     * @throws PistaDBException on I/O error.
     */
    public synchronized void save() {
        checkOpen();
        nativeSave(nativeHandle);
    }

    /**
     * Saves the database, then frees the native handle.
     * Safe to call multiple times.
     */
    @Override
    public synchronized void close() {
        if (nativeHandle != 0) {
            try {
                nativeSave(nativeHandle);
            } catch (PistaDBException ignored) {
                // best-effort save; always free
            } finally {
                nativeClose(nativeHandle);
                nativeHandle = 0;
            }
        }
    }

    /* ── CRUD ────────────────────────────────────────────────────────────── */

    /**
     * Inserts a vector with an optional label.
     *
     * @param id    Unique identifier (user-managed).
     * @param vec   Float array of length {@link #getDim()}.
     * @param label Optional human-readable label (max 255 bytes); may be null.
     * @throws PistaDBException on duplicate id or dimension mismatch.
     */
    public synchronized void insert(long id, float[] vec, String label) {
        checkOpen();
        nativeInsert(nativeHandle, id, label != null ? label : "", vec);
    }

    /** Inserts a vector without a label. */
    public synchronized void insert(long id, float[] vec) {
        insert(id, vec, "");
    }

    /**
     * Soft-deletes a vector.  Space is reclaimed on the next {@link #save()}.
     *
     * @throws PistaDBException if the id is not found.
     */
    public synchronized void delete(long id) {
        checkOpen();
        nativeDelete(nativeHandle, id);
    }

    /**
     * Replaces the vector data for an existing id.
     *
     * @param vec Float array of length {@link #getDim()}.
     * @throws PistaDBException if the id is not found or dimensions mismatch.
     */
    public synchronized void update(long id, float[] vec) {
        checkOpen();
        nativeUpdate(nativeHandle, id, vec);
    }

    /**
     * Retrieves a stored vector and its label.
     *
     * @return {@link VectorEntry} containing the float array and label.
     * @throws PistaDBException if the id is not found.
     */
    public synchronized VectorEntry get(long id) {
        checkOpen();
        return nativeGet(nativeHandle, id);
    }

    /* ── Batch helpers ───────────────────────────────────────────────────── */

    /**
     * Inserts multiple vectors in one call.
     *
     * @param ids    Array of unique identifiers.
     * @param vecs   Array of float arrays, each of length {@link #getDim()}.
     * @param labels Optional array of labels (may be null; any element may be null).
     */
    public synchronized void insertBatch(long[] ids, float[][] vecs, String[] labels) {
        checkOpen();
        for (int i = 0; i < ids.length; i++) {
            String lbl = (labels != null && i < labels.length) ? labels[i] : "";
            nativeInsert(nativeHandle, ids[i], lbl != null ? lbl : "", vecs[i]);
        }
    }

    /** Inserts multiple vectors without labels. */
    public synchronized void insertBatch(long[] ids, float[][] vecs) {
        insertBatch(ids, vecs, null);
    }

    /* ── Search ──────────────────────────────────────────────────────────── */

    /**
     * K-nearest-neighbour search.
     *
     * @param query Float array of length {@link #getDim()}.
     * @param k     Number of results to return.
     * @return Array of up to {@code k} results ordered by ascending distance.
     * @throws PistaDBException on error (e.g. index not trained).
     */
    public synchronized SearchResult[] search(float[] query, int k) {
        checkOpen();
        return nativeSearch(nativeHandle, query, k);
    }

    /** Searches with k = 10. */
    public synchronized SearchResult[] search(float[] query) {
        return search(query, 10);
    }

    /* ── Index management ────────────────────────────────────────────────── */

    /**
     * Trains the index on currently inserted vectors.
     * <p>Required for {@link IndexType#IVF}, {@link IndexType#IVF_PQ},
     * and {@link IndexType#SCANN} before performing inserts.
     * Optional for {@link IndexType#HNSW} and {@link IndexType#DISKANN}
     * (triggers a rebuild pass).
     */
    public synchronized void train() {
        checkOpen();
        nativeTrain(nativeHandle);
    }

    /* ── Metadata ────────────────────────────────────────────────────────── */

    /** Returns the number of active (non-deleted) vectors. */
    public synchronized int getCount() {
        checkOpen();
        return nativeCount(nativeHandle);
    }

    /** Returns the vector dimensionality. */
    public synchronized int getDim() {
        checkOpen();
        return nativeDim(nativeHandle);
    }

    /** Returns the distance metric. */
    public synchronized Metric getMetric() {
        checkOpen();
        return Metric.fromValue(nativeMetric(nativeHandle));
    }

    /** Returns the index algorithm. */
    public synchronized IndexType getIndexType() {
        checkOpen();
        return IndexType.fromValue(nativeIndexType(nativeHandle));
    }

    /**
     * Returns the human-readable description of the last error.
     * Useful for debugging after a caught {@link PistaDBException}.
     */
    public synchronized String getLastError() {
        if (nativeHandle == 0) return "database is closed";
        return nativeLastError(nativeHandle);
    }

    /** Returns the PistaDB library version string (e.g. {@code "1.0.0"}). */
    public static String getVersion() {
        return nativeVersion();
    }

    /** Returns {@code true} if the database is open (handle not closed). */
    public boolean isOpen() {
        return nativeHandle != 0;
    }

    /* ── Internal ────────────────────────────────────────────────────────── */

    private void checkOpen() {
        if (nativeHandle == 0) {
            throw new PistaDBException("Database is closed");
        }
    }
}
