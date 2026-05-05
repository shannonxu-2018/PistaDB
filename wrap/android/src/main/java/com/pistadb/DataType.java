/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
package com.pistadb;

/**
 * Per-field data type used by {@link FieldSchema}.
 *
 * <p>Mirrors {@code pymilvus.DataType}.  The integer values match the wire
 * representation written to the JSON sidecar so collections created from
 * Python and Android remain interchangeable.
 */
public enum DataType {

    /** Boolean. */
    BOOL(1, "BOOL"),
    /** 8-bit signed integer. */
    INT8(2, "INT8"),
    /** 16-bit signed integer. */
    INT16(3, "INT16"),
    /** 32-bit signed integer. */
    INT32(4, "INT32"),
    /** 64-bit signed integer (used for the primary key). */
    INT64(5, "INT64"),
    /** 32-bit IEEE-754 float. */
    FLOAT(10, "FLOAT"),
    /** 64-bit IEEE-754 float. */
    DOUBLE(11, "DOUBLE"),
    /** Variable-length UTF-8 string with optional max byte length. */
    VARCHAR(21, "VARCHAR"),
    /** Arbitrary JSON value, stored verbatim in the sidecar. */
    JSON(23, "JSON"),
    /** Float vector — must declare a positive {@code dim}. */
    FLOAT_VECTOR(101, "FLOAT_VECTOR");

    final int    value;
    final String wire;

    DataType(int value, String wire) {
        this.value = value;
        this.wire  = wire;
    }

    public int    getValue() { return value; }
    public String getWire()  { return wire;  }

    /** True if this is one of the integer types. */
    public boolean isInt() {
        return this == INT8 || this == INT16 || this == INT32 || this == INT64;
    }

    /** True if this is one of the floating-point scalar types. */
    public boolean isFloat() {
        return this == FLOAT || this == DOUBLE;
    }

    /** Parse a wire-format string back to a DataType. */
    public static DataType fromWire(String s) {
        for (DataType d : values()) if (d.wire.equals(s)) return d;
        throw new IllegalArgumentException("Unknown DataType: " + s);
    }
}
