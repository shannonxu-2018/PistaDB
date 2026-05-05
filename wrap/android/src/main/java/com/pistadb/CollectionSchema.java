/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
package com.pistadb;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Ordered list of {@link FieldSchema} with a description.
 *
 * <p>A valid schema must have exactly one primary key (INT64), exactly one
 * FLOAT_VECTOR field, and unique field names — checked in the constructor.
 */
public final class CollectionSchema {

    private final List<FieldSchema> fields;
    private final String            description;
    private final FieldSchema       primary;
    private final FieldSchema       vector;

    public CollectionSchema(List<FieldSchema> fields, String description) {
        if (fields == null) throw new IllegalArgumentException("fields must not be null");

        this.fields      = Collections.unmodifiableList(new ArrayList<>(fields));
        this.description = description != null ? description : "";

        Set<String> seen     = new HashSet<>();
        FieldSchema primary  = null;
        FieldSchema vector   = null;
        for (FieldSchema f : this.fields) {
            if (!seen.add(f.getName()))
                throw new IllegalArgumentException("duplicate field name '" + f.getName() + "'");
            if (f.isPrimary()) {
                if (primary != null)
                    throw new IllegalArgumentException("schema must have exactly one primary key");
                primary = f;
            }
            if (f.getDType() == DataType.FLOAT_VECTOR) {
                if (vector != null)
                    throw new IllegalArgumentException("schema must have exactly one FLOAT_VECTOR field");
                vector = f;
            }
        }
        if (primary == null) throw new IllegalArgumentException("schema must have a primary key");
        if (vector  == null) throw new IllegalArgumentException("schema must have a FLOAT_VECTOR field");

        this.primary = primary;
        this.vector  = vector;
    }

    /** Convenience varargs constructor. */
    public CollectionSchema(String description, FieldSchema... fields) {
        this(Arrays.asList(fields), description);
    }

    public List<FieldSchema> getFields()      { return fields; }
    public String            getDescription() { return description; }
    public FieldSchema       getPrimary()     { return primary; }
    public FieldSchema       getVector()      { return vector; }

    /** Non-primary, non-vector fields, in declaration order. */
    public List<FieldSchema> getScalarFields() {
        List<FieldSchema> out = new ArrayList<>();
        for (FieldSchema f : fields)
            if (!f.isPrimary() && f.getDType() != DataType.FLOAT_VECTOR)
                out.add(f);
        return out;
    }

    /** Look up a field by name; throws if absent. */
    public FieldSchema field(String name) {
        for (FieldSchema f : fields) if (f.getName().equals(name)) return f;
        throw new IllegalArgumentException("no field named '" + name + "'");
    }
}
