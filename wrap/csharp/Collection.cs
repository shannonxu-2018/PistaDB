/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB C# Binding — Collection.cs
 * Milvus-style schema-backed wrapper around <see cref="PistaDatabase"/>.
 *
 * Quick start:
 *   var fields = new[] {
 *       new FieldSchema("lc_id",     DataType.Int64, isPrimary: true, autoId: true),
 *       new FieldSchema("lc_section",DataType.VarChar, maxLength: 100),
 *       new FieldSchema("lc_vector", DataType.FloatVector, dim: 1536),
 *   };
 *   var coll = Collection.Create("common_text", fields, "Common text search",
 *       metric: Metric.Cosine, indexType: IndexType.HNSW, baseDir: "./db");
 *   var ids = coll.Insert(new[] {
 *       new Dictionary<string, object?> {
 *           ["lc_section"] = "common", ["lc_vector"] = new float[1536],
 *       },
 *   });
 *   var hits = coll.Search(query, k: 10);
 *   coll.Flush();
 *   coll.Dispose();
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace PistaDB
{
    /// <summary>
    /// A named collection backed by a <see cref="PistaDatabase"/> plus a JSON
    /// sidecar (<c>&lt;path&gt;.meta.json</c>) for non-vector scalar fields.
    /// </summary>
    public sealed class Collection : IDisposable
    {
        private const int SidecarVersion = 1;

        public string           Name       { get; }
        public CollectionSchema Schema     { get; }
        public string           Path       { get; }

        private readonly PistaDatabase _db;
        private readonly string        _meta;
        private readonly Metric        _metric;
        private readonly IndexType     _index;
        private readonly Dictionary<ulong, Dictionary<string, object?>> _rows;
        private ulong _nextId;

        private Collection(
            string name,
            string path,
            CollectionSchema schema,
            PistaDatabase db,
            Metric metric,
            IndexType index,
            Dictionary<ulong, Dictionary<string, object?>> rows,
            ulong nextId)
        {
            Name    = name;
            Path    = path;
            Schema  = schema;
            _db     = db;
            _meta   = path + ".meta.json";
            _metric = metric;
            _index  = index;
            _rows   = rows;
            _nextId = nextId;
        }

        // ── Properties ────────────────────────────────────────────────────────

        /// <summary>Underlying <see cref="PistaDatabase"/> handle (advanced use).</summary>
        public PistaDatabase Database => _db;

        /// <summary>Number of active (non-deleted) rows.</summary>
        public int NumEntities => _db.Count;

        // ── Factory ───────────────────────────────────────────────────────────

        /// <summary>Create a new collection.</summary>
        /// <param name="name">Used as the file stem for <c>&lt;name&gt;.pst</c>.</param>
        /// <param name="fields">Field schemas (must contain exactly one primary INT64 and one FLOAT_VECTOR field).</param>
        /// <param name="description">Free-text description.</param>
        /// <param name="metric">Distance metric. Default L2.</param>
        /// <param name="indexType">Index algorithm. Default HNSW.</param>
        /// <param name="params">Optional index tuning params.</param>
        /// <param name="baseDir">Directory in which to create the file. Defaults to CWD.</param>
        /// <param name="path">Explicit path to the .pst file (overrides baseDir+name).</param>
        /// <param name="overwrite">If true, replace any existing files; otherwise fails.</param>
        public static Collection Create(
            string name,
            IEnumerable<FieldSchema> fields,
            string description = "",
            Metric metric = Metric.L2,
            IndexType indexType = IndexType.HNSW,
            PistaDBParams? @params = null,
            string? baseDir = null,
            string? path = null,
            bool overwrite = false)
        {
            var schema   = new CollectionSchema(fields, description);
            var pstPath  = ResolvePath(name, baseDir, path);
            var metaPath = pstPath + ".meta.json";

            if (overwrite)
            {
                if (File.Exists(pstPath))  File.Delete(pstPath);
                if (File.Exists(metaPath)) File.Delete(metaPath);
            }
            else
            {
                if (File.Exists(pstPath))  throw new IOException($"{pstPath} already exists");
                if (File.Exists(metaPath)) throw new IOException($"{metaPath} already exists");
            }

            var dir = System.IO.Path.GetDirectoryName(pstPath);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir!);

            var db = PistaDatabase.Open(pstPath, schema.Vector.Dim ?? 0, metric, indexType, @params);
            var coll = new Collection(
                name, pstPath, schema, db, metric, indexType,
                new Dictionary<ulong, Dictionary<string, object?>>(), 1UL);
            coll.SaveSidecar();
            return coll;
        }

        /// <summary>Re-open an existing collection from disk.</summary>
        public static Collection Load(
            string name,
            string? baseDir = null,
            string? path = null,
            PistaDBParams? @params = null)
        {
            var pstPath  = ResolvePath(name, baseDir, path);
            var metaPath = pstPath + ".meta.json";
            if (!File.Exists(metaPath))
                throw new FileNotFoundException(
                    $"sidecar not found: {metaPath} — collection was not created via Collection.Create",
                    metaPath);

            var json = File.ReadAllText(metaPath);
            var sc   = JsonSerializer.Deserialize<JsonElement>(json);

            // Schema
            var fields = new List<FieldSchema>();
            foreach (var el in sc.GetProperty("fields").EnumerateArray())
            {
                var dt        = DataTypeExt.ParseWire(el.GetProperty("dtype").GetString()!);
                var fname     = el.GetProperty("name").GetString()!;
                var isPrimary = el.TryGetProperty("is_primary", out var p) && p.GetBoolean();
                var autoId    = el.TryGetProperty("auto_id",    out var a) && a.GetBoolean();
                int? maxLen   = el.TryGetProperty("max_length", out var m) && m.ValueKind == JsonValueKind.Number ? m.GetInt32() : (int?)null;
                int? dim      = el.TryGetProperty("dim",        out var d) && d.ValueKind == JsonValueKind.Number ? d.GetInt32() : (int?)null;
                var desc      = el.TryGetProperty("description", out var dd) ? dd.GetString() ?? "" : "";
                fields.Add(new FieldSchema(fname, dt, isPrimary, autoId, maxLen, dim, desc));
            }
            var schema = new CollectionSchema(fields,
                sc.TryGetProperty("description", out var dEl) ? (dEl.GetString() ?? "") : "");

            var metric  = ParseMetric(sc.GetProperty("metric").GetString()!);
            var index   = ParseIndex(sc.GetProperty("index").GetString()!);
            var nextId  = sc.TryGetProperty("next_id", out var nEl) && nEl.ValueKind == JsonValueKind.Number
                            ? nEl.GetUInt64() : 1UL;
            var savedNm = sc.TryGetProperty("name", out var nm) ? (nm.GetString() ?? name) : name;

            var rows = new Dictionary<ulong, Dictionary<string, object?>>();
            if (sc.TryGetProperty("rows", out var rowsEl) && rowsEl.ValueKind == JsonValueKind.Object)
            {
                foreach (var prop in rowsEl.EnumerateObject())
                {
                    if (!ulong.TryParse(prop.Name, out var rid))
                        throw new PistaDBException($"bad row key: {prop.Name}");
                    var entry = new Dictionary<string, object?>();
                    foreach (var col in prop.Value.EnumerateObject())
                        entry[col.Name] = JsonElementToObject(col.Value);
                    rows[rid] = entry;
                }
            }

            var db = PistaDatabase.Open(pstPath, schema.Vector.Dim ?? 0, metric, index, @params);
            return new Collection(savedNm, pstPath, schema, db, metric, index, rows, Math.Max(nextId, 1UL));
        }

        // ── Lifecycle ─────────────────────────────────────────────────────────

        /// <summary>Persist both the .pst and the JSON sidecar.</summary>
        public void Flush()
        {
            _db.Save();
            SaveSidecar();
        }

        /// <summary>Alias for <see cref="Flush"/>.</summary>
        public void Save() => Flush();

        public void Dispose() => _db.Dispose();

        // ── Insert ────────────────────────────────────────────────────────────

        /// <summary>
        /// Insert one or more rows.  Each row is a dict keyed by field name;
        /// the vector field accepts <c>float[]</c> or <c>double[]</c> or any
        /// <see cref="System.Collections.IEnumerable"/> of numbers.
        /// Returns the list of assigned primary ids (auto-generated when
        /// AutoId is enabled).
        /// </summary>
        public IReadOnlyList<ulong> Insert(IEnumerable<IDictionary<string, object?>> rows)
        {
            if (rows == null) throw new ArgumentNullException(nameof(rows));
            var pk      = Schema.Primary;
            var vec     = Schema.Vector;
            var scalars = Schema.ScalarFields().ToList();
            var known   = new HashSet<string>(Schema.Fields.Select(f => f.Name));

            var ids = new List<ulong>();
            foreach (var row in rows)
            {
                foreach (var k in row.Keys)
                    if (!known.Contains(k))
                        throw new ArgumentException($"unknown field '{k}'");

                ulong id;
                if (pk.AutoId)
                {
                    if (row.TryGetValue(pk.Name, out var pv) && pv != null)
                        throw new ArgumentException(
                            $"AutoId enabled on '{pk.Name}' — do not supply it");
                    id = _nextId++;
                }
                else
                {
                    if (!row.TryGetValue(pk.Name, out var pv) || pv == null)
                        throw new ArgumentException($"missing primary key '{pk.Name}'");
                    id = CoerceUInt64(pv);
                    if (_rows.ContainsKey(id))
                        throw new ArgumentException($"duplicate primary id={id}");
                    if (id >= _nextId) _nextId = id + 1;
                }
                if (id == 0) throw new ArgumentException("primary key must be > 0");

                if (!row.TryGetValue(vec.Name, out var vraw) || vraw == null)
                    throw new ArgumentException($"missing vector field '{vec.Name}'");
                var v = CoerceFloatArray(vraw, vec.Dim ?? 0);

                var scalarVals = new Dictionary<string, object?>();
                foreach (var f in scalars)
                {
                    row.TryGetValue(f.Name, out var rv);
                    scalarVals[f.Name] = rv == null ? null : CoerceScalar(rv, f);
                }

                _db.Insert(id, v);
                _rows[id] = scalarVals;
                ids.Add(id);
            }
            return ids;
        }

        // ── Delete / Get ──────────────────────────────────────────────────────

        /// <summary>Delete rows by primary id; missing ids are skipped silently.</summary>
        public int Delete(IEnumerable<ulong> ids)
        {
            int removed = 0;
            foreach (var id in ids)
            {
                try
                {
                    _db.Delete(id);
                    _rows.Remove(id);
                    removed++;
                }
                catch (PistaDBException) { /* ignore missing */ }
            }
            return removed;
        }

        /// <summary>Get the full row (all fields, including vector) by primary id.</summary>
        public IDictionary<string, object?> Get(ulong id)
        {
            if (!_rows.TryGetValue(id, out var meta))
                throw new KeyNotFoundException($"id={id} not found");
            var entry = _db.Get(id);
            var output = new Dictionary<string, object?>(meta);
            output[Schema.Primary.Name] = id;
            output[Schema.Vector.Name]  = entry.Vector;
            return output;
        }

        // ── Search ────────────────────────────────────────────────────────────

        /// <summary>
        /// k-NN search.  Pass <c>outputFields=null</c> to project all scalar fields,
        /// or a list of names to project only those.
        /// </summary>
        public IReadOnlyList<Hit> Search(float[] query, int k, IEnumerable<string>? outputFields = null)
        {
            if (query == null) throw new ArgumentNullException(nameof(query));
            var vec = Schema.Vector;
            if (query.Length != (vec.Dim ?? 0))
                throw new ArgumentException(
                    $"query length {query.Length} != Dim {vec.Dim}");

            List<string> want;
            if (outputFields == null)
            {
                want = Schema.ScalarFields().Select(f => f.Name).ToList();
            }
            else
            {
                want = outputFields.ToList();
                foreach (var n in want)
                    if (n != Schema.Primary.Name && n != vec.Name)
                        Schema.Field(n);   // throws if unknown
            }

            var raw = _db.Search(query, k);
            var hits = new List<Hit>(raw.Count);
            foreach (var r in raw)
            {
                _rows.TryGetValue(r.Id, out var meta);
                var fields = new Dictionary<string, object?>();
                foreach (var n in want)
                {
                    if (n == Schema.Primary.Name)
                        fields[n] = r.Id;
                    else if (n == vec.Name)
                        fields[n] = _db.Get(r.Id).Vector;
                    else
                        fields[n] = meta != null && meta.TryGetValue(n, out var mv) ? mv : null;
                }
                hits.Add(new Hit(r.Id, r.Distance, fields));
            }
            return hits;
        }

        // ── Sidecar I/O ───────────────────────────────────────────────────────

        private void SaveSidecar()
        {
            using var ms = new MemoryStream();
            using (var w = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = true }))
            {
                w.WriteStartObject();
                w.WriteNumber("version", SidecarVersion);
                w.WriteString("name", Name);
                w.WriteString("description", Schema.Description);
                w.WriteString("metric", _metric.ToString());
                w.WriteString("index",  _index.ToString());
                w.WriteNumber("next_id", _nextId);

                w.WriteStartArray("fields");
                foreach (var f in Schema.Fields)
                {
                    w.WriteStartObject();
                    w.WriteString("name", f.Name);
                    w.WriteString("dtype", f.DType.Wire());
                    w.WriteBoolean("is_primary", f.IsPrimary);
                    w.WriteBoolean("auto_id", f.AutoId);
                    if (f.MaxLength.HasValue) w.WriteNumber("max_length", f.MaxLength.Value);
                    if (f.Dim.HasValue)       w.WriteNumber("dim",        f.Dim.Value);
                    if (!string.IsNullOrEmpty(f.Description))
                        w.WriteString("description", f.Description);
                    w.WriteEndObject();
                }
                w.WriteEndArray();

                w.WriteStartObject("rows");
                foreach (var kv in _rows.OrderBy(p => p.Key))
                {
                    w.WriteStartObject(kv.Key.ToString());
                    foreach (var col in kv.Value)
                        WriteJsonValue(w, col.Key, col.Value);
                    w.WriteEndObject();
                }
                w.WriteEndObject();

                w.WriteEndObject();
            }
            var tmp = _meta + ".tmp";
            File.WriteAllBytes(tmp, ms.ToArray());
            if (File.Exists(_meta)) File.Delete(_meta);
            File.Move(tmp, _meta);
        }

        // ── Private helpers ──────────────────────────────────────────────────

        private static string ResolvePath(string name, string? baseDir, string? path)
        {
            if (path != null) return path;
            return string.IsNullOrEmpty(baseDir)
                ? name + ".pst"
                : System.IO.Path.Combine(baseDir!, name + ".pst");
        }

        private static Metric ParseMetric(string s) => s switch
        {
            "L2"           => Metric.L2,
            "Cosine"       => Metric.Cosine,
            "IP"           => Metric.IP,
            "InnerProduct" => Metric.IP,
            "L1"           => Metric.L1,
            "Hamming"      => Metric.Hamming,
            _ => throw new PistaDBException($"unknown metric: {s}"),
        };

        private static IndexType ParseIndex(string s) => s switch
        {
            "Linear"  => IndexType.Linear,
            "HNSW"    => IndexType.HNSW,
            "IVF"     => IndexType.IVF,
            "IVF_PQ"  => IndexType.IVF_PQ,
            "DiskANN" => IndexType.DiskANN,
            "LSH"     => IndexType.LSH,
            "ScaNN"   => IndexType.ScaNN,
            _ => throw new PistaDBException($"unknown index type: {s}"),
        };

        private static ulong CoerceUInt64(object v) => v switch
        {
            ulong u  => u,
            long l   => l < 0 ? throw new ArgumentException("negative") : (ulong)l,
            int i    => i < 0 ? throw new ArgumentException("negative") : (ulong)i,
            uint ui  => ui,
            short sh => sh < 0 ? throw new ArgumentException("negative") : (ulong)sh,
            float f  => (ulong)f,
            double d => (ulong)d,
            string s => ulong.Parse(s),
            _        => throw new ArgumentException($"cannot coerce {v.GetType()} to ulong"),
        };

        private static long CoerceInt64(object v) => v switch
        {
            long l   => l,
            int i    => i,
            ulong u  => (long)u,
            uint ui  => ui,
            short sh => sh,
            byte b   => b,
            float f  => (long)f,
            double d => (long)d,
            string s => long.Parse(s),
            _        => throw new ArgumentException($"cannot coerce {v.GetType()} to long"),
        };

        private static double CoerceDouble(object v) => v switch
        {
            double d => d,
            float f  => f,
            long l   => l,
            int i    => i,
            ulong u  => u,
            string s => double.Parse(s, System.Globalization.CultureInfo.InvariantCulture),
            _        => throw new ArgumentException($"cannot coerce {v.GetType()} to double"),
        };

        private static float[] CoerceFloatArray(object v, int dim)
        {
            switch (v)
            {
                case float[] f when f.Length == dim:
                    return f;
                case double[] d when d.Length == dim:
                    var fa = new float[dim];
                    for (int i = 0; i < dim; i++) fa[i] = (float)d[i];
                    return fa;
                case System.Collections.IEnumerable seq:
                    var list = new List<float>(dim);
                    foreach (var item in seq) list.Add((float)CoerceDouble(item!));
                    if (list.Count != dim)
                        throw new ArgumentException($"vector length {list.Count} != Dim {dim}");
                    return list.ToArray();
                default:
                    throw new ArgumentException($"cannot coerce {v.GetType()} to float[]");
            }
        }

        private static object? CoerceScalar(object v, FieldSchema f)
        {
            if (f.DType == DataType.Bool)
            {
                if (v is bool b) return b;
                throw new ArgumentException($"field {f.Name}: expected bool, got {v.GetType()}");
            }
            if (f.DType == DataType.VarChar)
            {
                var s = v as string ?? v.ToString() ?? "";
                if (f.MaxLength.HasValue &&
                    System.Text.Encoding.UTF8.GetByteCount(s) > f.MaxLength.Value)
                    throw new ArgumentException(
                        $"field {f.Name}: value exceeds MaxLength={f.MaxLength}");
                return s;
            }
            if (f.DType == DataType.Json)        return v;
            if (f.DType.IsInt())                 return CoerceInt64(v);
            if (f.DType.IsFloat())               return CoerceDouble(v);
            if (f.DType == DataType.FloatVector)
                throw new ArgumentException("vector cannot appear as scalar");
            throw new ArgumentException($"unsupported dtype {f.DType}");
        }

        private static void WriteJsonValue(Utf8JsonWriter w, string name, object? v)
        {
            switch (v)
            {
                case null:                w.WriteNull(name); break;
                case bool b:              w.WriteBoolean(name, b); break;
                case string s:            w.WriteString(name, s); break;
                case long l:              w.WriteNumber(name, l); break;
                case int i:               w.WriteNumber(name, i); break;
                case ulong ul:            w.WriteNumber(name, ul); break;
                case uint ui:             w.WriteNumber(name, ui); break;
                case short sh:            w.WriteNumber(name, sh); break;
                case byte by:             w.WriteNumber(name, by); break;
                case double d:            w.WriteNumber(name, d); break;
                case float f:             w.WriteNumber(name, f); break;
                default:
                    // Fall back to JSON.NET-style serialisation for arbitrary objects.
                    w.WritePropertyName(name);
                    JsonSerializer.Serialize(w, v);
                    break;
            }
        }

        private static object? JsonElementToObject(JsonElement el) => el.ValueKind switch
        {
            JsonValueKind.Null      => null,
            JsonValueKind.True      => (object)true,
            JsonValueKind.False     => (object)false,
            JsonValueKind.String    => el.GetString(),
            JsonValueKind.Number    => el.TryGetInt64(out var l) ? (object)l : el.GetDouble(),
            JsonValueKind.Array     => el.EnumerateArray().Select(JsonElementToObject).ToArray(),
            JsonValueKind.Object    => el.EnumerateObject()
                                          .ToDictionary(p => p.Name, p => JsonElementToObject(p.Value)),
            _                       => null,
        };
    }
}
