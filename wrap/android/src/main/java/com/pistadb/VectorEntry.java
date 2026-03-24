/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
package com.pistadb;

/**
 * A stored vector with its label, returned by {@link PistaDB#get}.
 */
public final class VectorEntry {

    /** The raw float vector (length == {@link PistaDB#getDim()}). */
    public final float[] vector;

    /** Human-readable label supplied at insert time (empty string if none). */
    public final String label;

    /** Called by the JNI bridge. */
    public VectorEntry(float[] vector, String label) {
        this.vector = vector;
        this.label  = label != null ? label : "";
    }

    @Override
    public String toString() {
        return "VectorEntry{dim=" + vector.length + ", label='" + label + "'}";
    }
}
