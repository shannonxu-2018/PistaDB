/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB C# Binding — Schema.cs
 * Milvus-style FieldSchema / CollectionSchema / DataType / Hit definitions.
 */

using System;
using System.Collections.Generic;

namespace PistaDB
{
    // ── DataType ──────────────────────────────────────────────────────────────

    /// <summary>Per-field data type (mirrors <c>pymilvus.DataType</c>).</summary>
    public enum DataType
    {
        Bool         = 1,
        Int8         = 2,
        Int16        = 3,
        Int32        = 4,
        Int64        = 5,
        Float        = 10,
        Double       = 11,
        VarChar      = 21,
        Json         = 23,
        FloatVector  = 101,
    }

    internal static class DataTypeExt
    {
        public static bool IsInt(this DataType d) =>
            d == DataType.Int8 || d == DataType.Int16 ||
            d == DataType.Int32 || d == DataType.Int64;

        public static bool IsFloat(this DataType d) =>
            d == DataType.Float || d == DataType.Double;

        public static string Wire(this DataType d) => d switch
        {
            DataType.Bool        => "BOOL",
            DataType.Int8        => "INT8",
            DataType.Int16       => "INT16",
            DataType.Int32       => "INT32",
            DataType.Int64       => "INT64",
            DataType.Float       => "FLOAT",
            DataType.Double      => "DOUBLE",
            DataType.VarChar     => "VARCHAR",
            DataType.Json        => "JSON",
            DataType.FloatVector => "FLOAT_VECTOR",
            _ => throw new ArgumentOutOfRangeException(nameof(d)),
        };

        public static DataType ParseWire(string s) => s switch
        {
            "BOOL"         => DataType.Bool,
            "INT8"         => DataType.Int8,
            "INT16"        => DataType.Int16,
            "INT32"        => DataType.Int32,
            "INT64"        => DataType.Int64,
            "FLOAT"        => DataType.Float,
            "DOUBLE"       => DataType.Double,
            "VARCHAR"      => DataType.VarChar,
            "JSON"         => DataType.Json,
            "FLOAT_VECTOR" => DataType.FloatVector,
            _ => throw new PistaDBException($"unknown DataType: {s}"),
        };
    }

    // ── FieldSchema ───────────────────────────────────────────────────────────

    /// <summary>
    /// Description of a single field in a <see cref="CollectionSchema"/>.
    /// Mirrors <c>pymilvus.FieldSchema</c>.
    /// </summary>
    public sealed class FieldSchema
    {
        public string   Name        { get; }
        public DataType DType       { get; }
        public bool     IsPrimary   { get; }
        public bool     AutoId      { get; }
        public int?     MaxLength   { get; }   // VARCHAR only
        public int?     Dim         { get; }   // FLOAT_VECTOR only
        public string   Description { get; }

        public FieldSchema(
            string name,
            DataType dtype,
            bool isPrimary    = false,
            bool autoId       = false,
            int? maxLength    = null,
            int? dim          = null,
            string description = "")
        {
            if (string.IsNullOrEmpty(name))
                throw new ArgumentException("field name must not be empty", nameof(name));

            Name        = name;
            DType       = dtype;
            IsPrimary   = isPrimary;
            AutoId      = autoId;
            MaxLength   = maxLength;
            Dim         = dim;
            Description = description ?? "";

            Validate();
        }

        private void Validate()
        {
            if (DType == DataType.FloatVector && (Dim == null || Dim <= 0))
                throw new ArgumentException(
                    $"FLOAT_VECTOR field '{Name}' requires a positive Dim");
            if (DType == DataType.VarChar && MaxLength != null && MaxLength <= 0)
                throw new ArgumentException(
                    $"VARCHAR field '{Name}': MaxLength must be positive");
            if (IsPrimary && DType != DataType.Int64)
                throw new ArgumentException(
                    $"primary key field '{Name}' must be INT64");
            if (AutoId && !IsPrimary)
                throw new ArgumentException(
                    $"AutoId is only valid on the primary field (got it on '{Name}')");
        }
    }

    // ── CollectionSchema ──────────────────────────────────────────────────────

    /// <summary>Ordered list of <see cref="FieldSchema"/> with a description.</summary>
    public sealed class CollectionSchema
    {
        public IReadOnlyList<FieldSchema> Fields      { get; }
        public string                     Description { get; }

        public FieldSchema Primary { get; }
        public FieldSchema Vector  { get; }

        public CollectionSchema(IEnumerable<FieldSchema> fields, string description = "")
        {
            if (fields == null) throw new ArgumentNullException(nameof(fields));
            var list = new List<FieldSchema>(fields);

            var seen = new HashSet<string>();
            FieldSchema? primary = null;
            FieldSchema? vector  = null;
            foreach (var f in list)
            {
                if (!seen.Add(f.Name))
                    throw new ArgumentException($"duplicate field name '{f.Name}'");
                if (f.IsPrimary)
                {
                    if (primary != null)
                        throw new ArgumentException("schema must have exactly one primary key");
                    primary = f;
                }
                if (f.DType == DataType.FloatVector)
                {
                    if (vector != null)
                        throw new ArgumentException("schema must have exactly one FLOAT_VECTOR field");
                    vector = f;
                }
            }
            if (primary == null) throw new ArgumentException("schema must have a primary key");
            if (vector  == null) throw new ArgumentException("schema must have a FLOAT_VECTOR field");

            Fields      = list;
            Description = description ?? "";
            Primary     = primary;
            Vector      = vector;
        }

        /// <summary>Non-primary, non-vector fields in declaration order.</summary>
        public IEnumerable<FieldSchema> ScalarFields()
        {
            foreach (var f in Fields)
                if (!f.IsPrimary && f.DType != DataType.FloatVector)
                    yield return f;
        }

        public FieldSchema Field(string name)
        {
            foreach (var f in Fields)
                if (f.Name == name) return f;
            throw new KeyNotFoundException($"no field named '{name}'");
        }
    }

    // ── Hit ────────────────────────────────────────────────────────────────────

    /// <summary>One row returned by <see cref="Collection.Search"/>.</summary>
    public sealed class Hit
    {
        public ulong                       Id       { get; }
        public float                       Distance { get; }
        public IReadOnlyDictionary<string, object?> Fields { get; }

        public Hit(ulong id, float distance, IReadOnlyDictionary<string, object?> fields)
        {
            Id       = id;
            Distance = distance;
            Fields   = fields;
        }

        /// <summary>Get a projected field; returns <c>null</c> if absent.</summary>
        public object? Get(string name) =>
            Fields.TryGetValue(name, out var v) ? v : null;
    }
}
