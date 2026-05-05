/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
package com.pistadb;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.Closeable;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * A schema-backed collection on top of {@link PistaDB} plus a JSON sidecar.
 *
 * <p>The vector field is stored in the underlying {@code .pst} file; all other
 * scalar fields are kept in {@code <path>.meta.json} keyed by the primary id.
 *
 * <h3>Quick start</h3>
 * <pre>{@code
 * List<FieldSchema> fields = Arrays.asList(
 *     new FieldSchema.Builder("lc_id",     DataType.INT64).primary(true).autoId(true).build(),
 *     new FieldSchema.Builder("lc_section",DataType.VARCHAR).maxLength(100).build(),
 *     new FieldSchema.Builder("lc_vector", DataType.FLOAT_VECTOR).dim(1536).build());
 *
 * Collection.Options opt = new Collection.Options()
 *     .baseDir(getCacheDir().getAbsolutePath())
 *     .metric(Metric.COSINE)
 *     .index(IndexType.HNSW);
 *
 * try (com.pistadb.Collection coll =
 *          com.pistadb.Collection.create("common_text", fields, "Common text search", opt)) {
 *     Map<String, Object> row = new HashMap<>();
 *     row.put("lc_section", "common");
 *     row.put("lc_vector",  embedding);   // float[]
 *     coll.insert(Collections.singletonList(row));
 *
 *     List<Hit> hits = coll.search(query, 5, null);
 *     coll.flush();
 * }
 * }</pre>
 */
public final class Collection implements Closeable {

    private static final int SIDECAR_VERSION = 1;

    private final String           name;
    private final String           path;
    private final File             metaFile;
    private final CollectionSchema schema;
    private final PistaDB          db;
    private final Metric           metric;
    private final IndexType        indexType;
    private final Map<Long, Map<String, Object>> rows;
    private long                   nextId;

    private Collection(String name,
                       String path,
                       CollectionSchema schema,
                       PistaDB db,
                       Metric metric,
                       IndexType indexType,
                       Map<Long, Map<String, Object>> rows,
                       long nextId)
    {
        this.name      = name;
        this.path      = path;
        this.metaFile  = new File(path + ".meta.json");
        this.schema    = schema;
        this.db        = db;
        this.metric    = metric;
        this.indexType = indexType;
        this.rows      = rows;
        this.nextId    = nextId;
    }

    // ── Factories ─────────────────────────────────────────────────────────

    /**
     * Create a new collection.  Fails if the .pst or .meta.json already exist
     * unless {@code opt.overwrite} is true.
     */
    public static Collection create(String name,
                                    List<FieldSchema> fields,
                                    String description,
                                    Options opt)
    {
        if (opt == null) opt = new Options();
        CollectionSchema schema = new CollectionSchema(fields, description);
        String pstPath  = opt.resolvePath(name);
        File   metaFile = new File(pstPath + ".meta.json");
        File   pstFile  = new File(pstPath);

        if (opt.overwrite) {
            //noinspection ResultOfMethodCallIgnored
            pstFile.delete();
            //noinspection ResultOfMethodCallIgnored
            metaFile.delete();
        } else {
            if (pstFile.exists())  throw new PistaDBException(pstPath + " already exists");
            if (metaFile.exists()) throw new PistaDBException(metaFile + " already exists");
        }

        File parent = pstFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs())
            throw new PistaDBException("cannot create directory " + parent);

        PistaDB db = new PistaDB(
                pstPath, schema.getVector().getDim(),
                opt.metric, opt.indexType, opt.params);

        Collection coll = new Collection(
                name, pstPath, schema, db, opt.metric, opt.indexType,
                new HashMap<Long, Map<String, Object>>(), 1L);
        coll.saveSidecar();
        return coll;
    }

    /** Convenience varargs form. */
    public static Collection create(String name, String description, Options opt, FieldSchema... fields) {
        return create(name, Arrays.asList(fields), description, opt);
    }

    /** Re-open an existing collection from disk. */
    public static Collection load(String name, Options opt) {
        if (opt == null) opt = new Options();
        String pstPath  = opt.resolvePath(name);
        File   metaFile = new File(pstPath + ".meta.json");
        if (!metaFile.exists())
            throw new PistaDBException("sidecar not found: " + metaFile);

        try {
            String body = new String(Files.readAllBytes(metaFile.toPath()), StandardCharsets.UTF_8);
            JSONObject sc = new JSONObject(body);

            JSONArray  fa = sc.getJSONArray("fields");
            List<FieldSchema> fields = new ArrayList<>(fa.length());
            for (int i = 0; i < fa.length(); i++) {
                JSONObject o  = fa.getJSONObject(i);
                FieldSchema.Builder b = new FieldSchema.Builder(
                        o.getString("name"), DataType.fromWire(o.getString("dtype")));
                if (o.optBoolean("is_primary", false)) b.primary(true);
                if (o.optBoolean("auto_id", false))    b.autoId(true);
                if (o.has("max_length"))               b.maxLength(o.getInt("max_length"));
                if (o.has("dim"))                      b.dim(o.getInt("dim"));
                if (o.has("description"))              b.description(o.getString("description"));
                fields.add(b.build());
            }
            CollectionSchema schema = new CollectionSchema(fields, sc.optString("description", ""));

            Metric    metric = parseMetric(sc.getString("metric"));
            IndexType idx    = parseIndex (sc.getString("index"));
            long      nextId = sc.optLong("next_id", 1L);
            String    saved  = sc.optString("name", name);

            Map<Long, Map<String, Object>> rows = new HashMap<>();
            JSONObject jrows = sc.optJSONObject("rows");
            if (jrows != null) {
                for (java.util.Iterator<String> it = jrows.keys(); it.hasNext(); ) {
                    String k = it.next();
                    long   id = Long.parseLong(k);
                    JSONObject ro = jrows.getJSONObject(k);
                    Map<String, Object> row = new LinkedHashMap<>();
                    for (java.util.Iterator<String> ck = ro.keys(); ck.hasNext(); ) {
                        String col = ck.next();
                        row.put(col, jsonToJava(ro.opt(col)));
                    }
                    rows.put(id, row);
                }
            }

            PistaDB db = new PistaDB(
                    pstPath, schema.getVector().getDim(),
                    metric, idx, opt.params);

            return new Collection(saved, pstPath, schema, db, metric, idx, rows, Math.max(nextId, 1L));
        } catch (IOException | JSONException e) {
            throw new PistaDBException("cannot load collection: " + e.getMessage(), e);
        }
    }

    // ── Properties ─────────────────────────────────────────────────────────

    public String           getName()         { return name; }
    public String           getPath()         { return path; }
    public CollectionSchema getSchema()       { return schema; }
    public PistaDB          getDatabase()     { return db; }
    public int              getNumEntities()  { return db.getCount(); }

    // ── Lifecycle ──────────────────────────────────────────────────────────

    /** Persist both the .pst file and the JSON sidecar. */
    public synchronized void flush() {
        db.save();
        saveSidecar();
    }

    /** Alias for {@link #flush()}. */
    public void save() { flush(); }

    @Override
    public synchronized void close() {
        db.close();
    }

    // ── Insert ─────────────────────────────────────────────────────────────

    /**
     * Insert one or more rows.  Each row is a {@code Map<String,Object>} keyed
     * by field name; the vector field accepts a {@code float[]} or a numeric
     * {@code List/Iterable}.  Returns the assigned primary ids in order.
     */
    public synchronized List<Long> insert(List<? extends Map<String, ?>> rowsToInsert) {
        FieldSchema pk      = schema.getPrimary();
        FieldSchema vec     = schema.getVector();
        List<FieldSchema> scalars = schema.getScalarFields();
        Set<String> known   = new HashSet<>();
        for (FieldSchema f : schema.getFields()) known.add(f.getName());

        List<Long> out = new ArrayList<>(rowsToInsert.size());
        for (Map<String, ?> row : rowsToInsert) {
            for (String k : row.keySet())
                if (!known.contains(k))
                    throw new PistaDBException("unknown field '" + k + "'");

            long id;
            if (pk.isAutoId()) {
                Object v = row.get(pk.getName());
                if (v != null)
                    throw new PistaDBException(
                            "autoId enabled on '" + pk.getName() + "' — do not supply it");
                id = nextId++;
            } else {
                Object v = row.get(pk.getName());
                if (v == null)
                    throw new PistaDBException("missing primary key '" + pk.getName() + "'");
                id = coerceLong(v);
                if (rows.containsKey(id))
                    throw new PistaDBException("duplicate primary id=" + id);
                if (id >= nextId) nextId = id + 1;
            }
            if (id <= 0) throw new PistaDBException("primary key must be > 0");

            Object vraw = row.get(vec.getName());
            if (vraw == null) throw new PistaDBException("missing vector field '" + vec.getName() + "'");
            float[] v = coerceFloatArray(vraw, vec.getDim());

            Map<String, Object> scalarVals = new LinkedHashMap<>();
            for (FieldSchema f : scalars) {
                Object raw = row.get(f.getName());
                scalarVals.put(f.getName(), raw == null ? null : coerceScalar(raw, f));
            }

            db.insert(id, v, "");
            rows.put(id, scalarVals);
            out.add(id);
        }
        return out;
    }

    // ── Delete / Get ───────────────────────────────────────────────────────

    /** Delete one row by primary id; returns true if it existed. */
    public synchronized boolean delete(long id) {
        try {
            db.delete(id);
            rows.remove(id);
            return true;
        } catch (PistaDBException e) {
            return false;
        }
    }

    /** Delete multiple ids; returns the number actually removed. */
    public synchronized int delete(java.util.Collection<Long> ids) {
        int removed = 0;
        for (Long id : ids) if (delete(id)) removed++;
        return removed;
    }

    /** Get the full row (all fields, including vector) by primary id. */
    public synchronized Map<String, Object> get(long id) {
        Map<String, Object> meta = rows.get(id);
        if (meta == null) throw new PistaDBException("id=" + id + " not found");
        VectorEntry entry = db.get(id);
        Map<String, Object> out = new LinkedHashMap<>(meta);
        out.put(schema.getPrimary().getName(), id);
        out.put(schema.getVector().getName(), entry.vector);
        return out;
    }

    // ── Search ─────────────────────────────────────────────────────────────

    /**
     * k-NN search.  Pass {@code outputFields=null} to project all scalar fields,
     * or a list of names to project only those.
     */
    public synchronized List<Hit> search(float[] query, int k, List<String> outputFields) {
        FieldSchema vec = schema.getVector();
        if (query.length != vec.getDim())
            throw new PistaDBException("query length " + query.length + " != dim " + vec.getDim());

        List<String> want;
        if (outputFields == null) {
            want = new ArrayList<>();
            for (FieldSchema f : schema.getScalarFields()) want.add(f.getName());
        } else {
            for (String n : outputFields)
                if (!n.equals(schema.getPrimary().getName())
                        && !n.equals(vec.getName()))
                    schema.field(n);            // throws if unknown
            want = outputFields;
        }

        SearchResult[] raw = db.search(query, k);
        List<Hit> hits = new ArrayList<>(raw.length);
        String pkName  = schema.getPrimary().getName();
        String vecName = vec.getName();

        for (SearchResult r : raw) {
            Map<String, Object> meta = rows.get(r.id);
            Map<String, Object> fields = new LinkedHashMap<>();
            for (String n : want) {
                if (n.equals(pkName)) {
                    fields.put(n, r.id);
                } else if (n.equals(vecName)) {
                    try {
                        fields.put(n, db.get(r.id).vector);
                    } catch (PistaDBException ignored) {
                        fields.put(n, null);
                    }
                } else {
                    fields.put(n, meta != null ? meta.get(n) : null);
                }
            }
            hits.add(new Hit(r.id, r.distance, fields));
        }
        return hits;
    }

    // ── Sidecar I/O ────────────────────────────────────────────────────────

    private void saveSidecar() {
        try {
            JSONObject sc = new JSONObject();
            sc.put("version", SIDECAR_VERSION);
            sc.put("name", name);
            sc.put("description", schema.getDescription());
            sc.put("metric", metric.name());
            sc.put("index",  indexType.name());
            sc.put("next_id", nextId);

            JSONArray fa = new JSONArray();
            for (FieldSchema f : schema.getFields()) {
                JSONObject o = new JSONObject();
                o.put("name", f.getName());
                o.put("dtype", f.getDType().getWire());
                o.put("is_primary", f.isPrimary());
                o.put("auto_id", f.isAutoId());
                if (f.getMaxLength() != null) o.put("max_length", f.getMaxLength());
                if (f.getDim() != null)       o.put("dim",        f.getDim());
                if (f.getDescription() != null && !f.getDescription().isEmpty())
                    o.put("description", f.getDescription());
                fa.put(o);
            }
            sc.put("fields", fa);

            // Stable order for diff-friendly output.
            TreeMap<Long, Map<String, Object>> sortedRows = new TreeMap<>(rows);
            JSONObject jrows = new JSONObject();
            for (Map.Entry<Long, Map<String, Object>> e : sortedRows.entrySet()) {
                JSONObject r = new JSONObject();
                for (Map.Entry<String, Object> col : e.getValue().entrySet())
                    r.put(col.getKey(), col.getValue() == null ? JSONObject.NULL : col.getValue());
                jrows.put(String.valueOf(e.getKey()), r);
            }
            sc.put("rows", jrows);

            // Atomic write: temp file + rename.
            File tmp = new File(metaFile.getPath() + ".tmp");
            try (FileOutputStream fos = new FileOutputStream(tmp)) {
                fos.write(sc.toString(2).getBytes(StandardCharsets.UTF_8));
            }
            if (metaFile.exists() && !metaFile.delete())
                throw new PistaDBException("cannot replace " + metaFile);
            if (!tmp.renameTo(metaFile))
                throw new PistaDBException("rename failed: " + tmp + " -> " + metaFile);
        } catch (IOException | JSONException e) {
            throw new PistaDBException("cannot save sidecar: " + e.getMessage(), e);
        }
    }

    // ── Coercion helpers ───────────────────────────────────────────────────

    private static long coerceLong(Object v) {
        if (v instanceof Long)      return (Long) v;
        if (v instanceof Integer)   return ((Integer) v).longValue();
        if (v instanceof Number)    return ((Number) v).longValue();
        if (v instanceof String)    return Long.parseLong((String) v);
        throw new PistaDBException("cannot coerce " + v.getClass().getName() + " to long");
    }

    private static double coerceDouble(Object v) {
        if (v instanceof Number)    return ((Number) v).doubleValue();
        if (v instanceof String)    return Double.parseDouble((String) v);
        throw new PistaDBException("cannot coerce " + v.getClass().getName() + " to double");
    }

    private static float[] coerceFloatArray(Object v, int dim) {
        if (v instanceof float[]) {
            float[] a = (float[]) v;
            if (a.length != dim) throw new PistaDBException("vector length " + a.length + " != dim " + dim);
            return a;
        }
        if (v instanceof double[]) {
            double[] a = (double[]) v;
            if (a.length != dim) throw new PistaDBException("vector length " + a.length + " != dim " + dim);
            float[] out = new float[dim];
            for (int i = 0; i < dim; i++) out[i] = (float) a[i];
            return out;
        }
        if (v instanceof Number[]) {
            Number[] a = (Number[]) v;
            if (a.length != dim) throw new PistaDBException("vector length " + a.length + " != dim " + dim);
            float[] out = new float[dim];
            for (int i = 0; i < dim; i++) out[i] = a[i].floatValue();
            return out;
        }
        if (v instanceof Iterable) {
            List<Float> tmp = new ArrayList<>(dim);
            for (Object e : (Iterable<?>) v) tmp.add((float) coerceDouble(e));
            if (tmp.size() != dim) throw new PistaDBException("vector length " + tmp.size() + " != dim " + dim);
            float[] out = new float[dim];
            for (int i = 0; i < dim; i++) out[i] = tmp.get(i);
            return out;
        }
        if (v instanceof JSONArray) {
            JSONArray a = (JSONArray) v;
            if (a.length() != dim)
                throw new PistaDBException("vector length " + a.length() + " != dim " + dim);
            float[] out = new float[dim];
            for (int i = 0; i < dim; i++) out[i] = (float) a.optDouble(i);
            return out;
        }
        throw new PistaDBException("cannot coerce " + v.getClass().getName() + " to float[]");
    }

    private static Object coerceScalar(Object v, FieldSchema f) {
        switch (f.getDType()) {
            case BOOL:
                if (v instanceof Boolean) return v;
                throw new PistaDBException("field " + f.getName() + ": expected Boolean");
            case VARCHAR:
                String s = (v instanceof String) ? (String) v : v.toString();
                if (f.getMaxLength() != null
                        && s.getBytes(StandardCharsets.UTF_8).length > f.getMaxLength())
                    throw new PistaDBException(
                            "field " + f.getName() + ": exceeds maxLength=" + f.getMaxLength());
                return s;
            case JSON:
                return v;
            case FLOAT_VECTOR:
                throw new PistaDBException("vector cannot appear as scalar");
            default:
                if (f.getDType().isInt())   return coerceLong(v);
                if (f.getDType().isFloat()) return coerceDouble(v);
                throw new PistaDBException("unsupported dtype " + f.getDType());
        }
    }

    private static Object jsonToJava(Object v) {
        if (v == null || v == JSONObject.NULL) return null;
        if (v instanceof JSONArray) {
            JSONArray a = (JSONArray) v;
            List<Object> out = new ArrayList<>(a.length());
            for (int i = 0; i < a.length(); i++) out.add(jsonToJava(a.opt(i)));
            return out;
        }
        if (v instanceof JSONObject) {
            JSONObject o = (JSONObject) v;
            Map<String, Object> out = new LinkedHashMap<>();
            for (java.util.Iterator<String> it = o.keys(); it.hasNext(); ) {
                String k = it.next();
                out.put(k, jsonToJava(o.opt(k)));
            }
            return out;
        }
        return v;
    }

    private static Metric parseMetric(String s) {
        if (s == null) return Metric.L2;
        switch (s) {
            case "L2":      return Metric.L2;
            case "Cosine":  case "COSINE":  return Metric.COSINE;
            case "IP":      case "InnerProduct": return Metric.IP;
            case "L1":      return Metric.L1;
            case "Hamming": case "HAMMING": return Metric.HAMMING;
            default: throw new PistaDBException("unknown metric: " + s);
        }
    }

    private static IndexType parseIndex(String s) {
        if (s == null) return IndexType.HNSW;
        switch (s) {
            case "Linear":  case "LINEAR":  return IndexType.LINEAR;
            case "HNSW":    return IndexType.HNSW;
            case "IVF":     return IndexType.IVF;
            case "IVF_PQ":  return IndexType.IVF_PQ;
            case "DiskANN": case "DISKANN": return IndexType.DISKANN;
            case "LSH":     return IndexType.LSH;
            case "ScaNN":   case "SCANN":   return IndexType.SCANN;
            case "SQ":      return IndexType.SQ;
            default: throw new PistaDBException("unknown index type: " + s);
        }
    }

    // ── Options ────────────────────────────────────────────────────────────

    /** Configuration for {@link Collection#create} / {@link Collection#load}. */
    public static final class Options {
        Metric        metric    = Metric.L2;
        IndexType     indexType = IndexType.HNSW;
        PistaDBParams params;
        String        baseDir;
        String        path;
        boolean       overwrite = false;

        public Options metric(Metric m)      { this.metric = m;     return this; }
        public Options index(IndexType t)    { this.indexType = t;  return this; }
        public Options params(PistaDBParams p){ this.params = p;    return this; }
        public Options baseDir(String d)     { this.baseDir = d;    return this; }
        public Options path(String p)        { this.path = p;       return this; }
        public Options overwrite(boolean v)  { this.overwrite = v;  return this; }

        String resolvePath(String name) {
            if (path != null) return path;
            if (baseDir == null || baseDir.isEmpty()) return name + ".pst";
            String sep = baseDir.endsWith(File.separator) ? "" : File.separator;
            return baseDir + sep + name + ".pst";
        }
    }
}
