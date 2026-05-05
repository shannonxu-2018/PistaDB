/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
package com.pistadb;

import java.util.Collections;
import java.util.Map;

/** A single search result row enriched with the projected scalar fields. */
public final class Hit {

    private final long              id;
    private final float             distance;
    private final Map<String, Object> fields;

    public Hit(long id, float distance, Map<String, Object> fields) {
        this.id       = id;
        this.distance = distance;
        this.fields   = fields == null
                ? Collections.<String, Object>emptyMap()
                : Collections.unmodifiableMap(fields);
    }

    public long                getId()       { return id; }
    public float               getDistance() { return distance; }
    public Map<String, Object> getFields()   { return fields; }

    /** Get a field by name, or {@code null} if not projected. */
    public Object get(String name) { return fields.get(name); }

    @Override
    public String toString() {
        return "Hit{id=" + id + ", distance=" + distance + ", fields=" + fields + "}";
    }
}
