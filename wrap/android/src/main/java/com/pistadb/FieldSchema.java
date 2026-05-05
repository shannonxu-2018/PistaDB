/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
package com.pistadb;

/**
 * Description of a single field in a {@link CollectionSchema}.
 *
 * <p>Mirrors {@code pymilvus.FieldSchema}.  Use the {@link Builder} for
 * readable construction:
 *
 * <pre>{@code
 * FieldSchema id = new FieldSchema.Builder("lc_id", DataType.INT64)
 *     .primary(true).autoId(true).build();
 *
 * FieldSchema vec = new FieldSchema.Builder("lc_vector", DataType.FLOAT_VECTOR)
 *     .dim(1536).description("OpenAI embedding").build();
 * }</pre>
 */
public final class FieldSchema {

    private final String   name;
    private final DataType dtype;
    private final boolean  isPrimary;
    private final boolean  autoId;
    private final Integer  maxLength;   // null = unspecified
    private final Integer  dim;         // null = unspecified
    private final String   description;

    private FieldSchema(Builder b) {
        this.name        = b.name;
        this.dtype       = b.dtype;
        this.isPrimary   = b.isPrimary;
        this.autoId      = b.autoId;
        this.maxLength   = b.maxLength;
        this.dim         = b.dim;
        this.description = b.description != null ? b.description : "";
        validate();
    }

    private void validate() {
        if (name == null || name.isEmpty())
            throw new IllegalArgumentException("FieldSchema: name must not be empty");
        if (dtype == DataType.FLOAT_VECTOR && (dim == null || dim <= 0))
            throw new IllegalArgumentException(
                "FLOAT_VECTOR field '" + name + "' requires a positive dim");
        if (dtype == DataType.VARCHAR && maxLength != null && maxLength <= 0)
            throw new IllegalArgumentException(
                "VARCHAR field '" + name + "': maxLength must be positive");
        if (isPrimary && dtype != DataType.INT64)
            throw new IllegalArgumentException(
                "Primary key '" + name + "' must be INT64");
        if (autoId && !isPrimary)
            throw new IllegalArgumentException(
                "autoId only valid on the primary field (got '" + name + "')");
    }

    public String   getName()        { return name; }
    public DataType getDType()       { return dtype; }
    public boolean  isPrimary()      { return isPrimary; }
    public boolean  isAutoId()       { return autoId; }
    public Integer  getMaxLength()   { return maxLength; }
    public Integer  getDim()         { return dim; }
    public String   getDescription() { return description; }

    @Override
    public String toString() {
        return "FieldSchema{" + name + ", " + dtype +
               (isPrimary ? ", primary" : "") +
               (autoId    ? ", autoId"  : "") +
               (dim != null       ? ", dim="       + dim       : "") +
               (maxLength != null ? ", maxLength=" + maxLength : "") + "}";
    }

    /** Fluent builder for {@link FieldSchema}. */
    public static final class Builder {
        private final String   name;
        private final DataType dtype;
        private boolean  isPrimary   = false;
        private boolean  autoId      = false;
        private Integer  maxLength;
        private Integer  dim;
        private String   description = "";

        public Builder(String name, DataType dtype) {
            this.name  = name;
            this.dtype = dtype;
        }

        public Builder primary(boolean v)     { this.isPrimary = v; return this; }
        public Builder autoId(boolean v)      { this.autoId    = v; return this; }
        public Builder maxLength(int v)       { this.maxLength = v; return this; }
        public Builder dim(int v)             { this.dim       = v; return this; }
        public Builder description(String v)  { this.description = v; return this; }

        public FieldSchema build() { return new FieldSchema(this); }
    }
}
