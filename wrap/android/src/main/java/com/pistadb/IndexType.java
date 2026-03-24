/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
package com.pistadb;

/**
 * Index algorithm used internally to organise and search vectors.
 * The integer value matches the C PistaDBIndexType enum.
 */
public enum IndexType {

    /** Brute-force linear scan – exact results, no training needed. */
    LINEAR(0),

    /**
     * Hierarchical Navigable Small World graphs.
     * Best speed/recall tradeoff for most use-cases (default).
     */
    HNSW(1),

    /**
     * Inverted File Index with k-means clustering.
     * Good for large datasets; {@link PistaDB#train()} required before inserts.
     */
    IVF(2),

    /**
     * IVF + Product Quantization – memory-efficient compression.
     * {@link PistaDB#train()} required before inserts.
     */
    IVF_PQ(3),

    /**
     * Vamana / DiskANN graphs – optimised for billion-scale datasets.
     */
    DISKANN(4),

    /** Locality-Sensitive Hashing – ultra-low memory footprint. */
    LSH(5),

    /**
     * ScaNN: Anisotropic Vector Quantization (Google ICML 2020).
     * Highest recall on cosine / IP metrics.
     * {@link PistaDB#train()} required before inserts.
     */
    SCANN(6);

    final int value;

    IndexType(int value) {
        this.value = value;
    }

    /** Returns the integer value sent to the native layer. */
    public int getValue() {
        return value;
    }

    /** Reconstructs an IndexType from its native integer value. */
    public static IndexType fromValue(int value) {
        for (IndexType t : values()) {
            if (t.value == value) return t;
        }
        throw new IllegalArgumentException("Unknown index type value: " + value);
    }
}
