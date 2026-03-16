/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
package com.pistadb;

/**
 * Distance metric used for vector comparisons.
 * The integer value matches the C PistaDBMetric enum.
 */
public enum Metric {

    /** Euclidean distance – general purpose, image and multimodal embeddings. */
    L2(0),

    /** Cosine similarity (stored as 1 – similarity) – text embeddings. */
    COSINE(1),

    /** Inner product (stored as negative dot product) – normalized vectors. */
    IP(2),

    /** Manhattan / L1 distance – sparse vectors. */
    L1(3),

    /** Hamming distance – binary embeddings, deduplication. */
    HAMMING(4);

    final int value;

    Metric(int value) {
        this.value = value;
    }

    /** Returns the integer value sent to the native layer. */
    public int getValue() {
        return value;
    }

    /** Reconstructs a Metric from its native integer value. */
    public static Metric fromValue(int value) {
        for (Metric m : values()) {
            if (m.value == value) return m;
        }
        throw new IllegalArgumentException("Unknown metric value: " + value);
    }
}
