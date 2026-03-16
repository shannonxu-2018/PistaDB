/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
package com.pistadb;

/**
 * A single k-NN search result returned by {@link PistaDB#search}.
 *
 * <p>Results are ordered by ascending distance (nearest first).
 */
public final class SearchResult {

    /** The vector's unique identifier. */
    public final long id;

    /**
     * Distance from the query vector.
     * Interpretation depends on the metric:
     * <ul>
     *   <li>L2 / L1 / Hamming – lower is closer</li>
     *   <li>COSINE – {@code 1 - cosine_similarity}, lower is closer</li>
     *   <li>IP – negative dot product, lower is closer</li>
     * </ul>
     */
    public final float distance;

    /** Human-readable label supplied at insert time (empty string if none). */
    public final String label;

    /** Called by the JNI bridge. */
    public SearchResult(long id, float distance, String label) {
        this.id       = id;
        this.distance = distance;
        this.label    = label != null ? label : "";
    }

    @Override
    public String toString() {
        return "SearchResult{id=" + id
               + ", distance=" + distance
               + ", label='" + label + "'}";
    }
}
