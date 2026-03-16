/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB C# Binding — PistaDBTypes.cs
 * Public types: enums, result types, parameters, exception.
 */

using System;

namespace PistaDB
{
    // ── Distance metrics ──────────────────────────────────────────────────────

    /// <summary>Vector distance metric used by the database.</summary>
    public enum Metric
    {
        /// <summary>Euclidean distance (L2). Default; good for most embeddings.</summary>
        L2      = 0,
        /// <summary>Cosine similarity expressed as distance. Ideal for text embeddings.</summary>
        Cosine  = 1,
        /// <summary>Inner product (stored as negative). Good for dot-product similarity.</summary>
        IP      = 2,
        /// <summary>Manhattan distance (L1).</summary>
        L1      = 3,
        /// <summary>Hamming distance. Suitable for binary or integer vectors.</summary>
        Hamming = 4,
    }

    // ── Index algorithms ──────────────────────────────────────────────────────

    /// <summary>ANN index algorithm to use.</summary>
    public enum IndexType
    {
        /// <summary>Brute-force exact scan. Perfect accuracy, O(n) query.</summary>
        Linear  = 0,
        /// <summary>HNSW — fast approximate search. Recommended for RAG workloads.</summary>
        HNSW    = 1,
        /// <summary>Inverted File Index. Requires <see cref="PistaDatabase.Train"/> before insert.</summary>
        IVF     = 2,
        /// <summary>IVF + Product Quantization. Low memory, lossy. Requires Train.</summary>
        IVF_PQ  = 3,
        /// <summary>DiskANN / Vamana. Scales to billion-vector datasets.</summary>
        DiskANN = 4,
        /// <summary>Locality-Sensitive Hashing. Ultra-low memory footprint.</summary>
        LSH     = 5,
        /// <summary>ScaNN (Anisotropic Vector Quantization). Two-phase reranking.</summary>
        ScaNN   = 6,
    }

    // ── Result types ──────────────────────────────────────────────────────────

    /// <summary>A single KNN search result.</summary>
    public sealed class SearchResult
    {
        /// <summary>User-supplied vector id.</summary>
        public ulong Id { get; }
        /// <summary>Distance from the query (lower is more similar for L2/L1/Hamming/Cosine).</summary>
        public float Distance { get; }
        /// <summary>Optional human-readable label associated with the vector.</summary>
        public string Label { get; }

        internal SearchResult(ulong id, float distance, string label)
        {
            Id = id;
            Distance = distance;
            Label = label;
        }

        public override string ToString() =>
            $"SearchResult(Id={Id}, Distance={Distance:F4}, Label=\"{Label}\")";
    }

    /// <summary>A stored vector entry returned by <see cref="PistaDatabase.Get"/>.</summary>
    public sealed class VectorEntry
    {
        /// <summary>The raw float vector.</summary>
        public float[] Vector { get; }
        /// <summary>Optional label stored alongside the vector.</summary>
        public string Label { get; }

        internal VectorEntry(float[] vector, string label)
        {
            Vector = vector;
            Label  = label;
        }
    }

    // ── Parameters ────────────────────────────────────────────────────────────

    /// <summary>
    /// Index parameters passed when opening a database.
    /// Unused parameters for the selected <see cref="IndexType"/> are ignored.
    /// </summary>
    public sealed class PistaDBParams
    {
        // ── HNSW ──────────────────────────────────────────────────────────────

        /// <summary>Max bi-directional connections per layer. Higher = better recall, more memory. Default 16.</summary>
        public int   HnswM              { get; set; } = 16;
        /// <summary>Build-time search width. Higher = better graph quality, slower build. Default 200.</summary>
        public int   HnswEfConstruction { get; set; } = 200;
        /// <summary>Query-time search width. Higher = better recall, slower query. Default 50.</summary>
        public int   HnswEfSearch       { get; set; } = 50;

        // ── IVF / IVF_PQ ──────────────────────────────────────────────────────

        /// <summary>Number of IVF centroids (clusters). Default 128.</summary>
        public int   IvfNList  { get; set; } = 128;
        /// <summary>Number of centroids to search at query time. Default 8.</summary>
        public int   IvfNProbe { get; set; } = 8;
        /// <summary>Number of PQ sub-spaces. Default 8.</summary>
        public int   PqM       { get; set; } = 8;
        /// <summary>Bits per PQ sub-code (4 or 8). Default 8.</summary>
        public int   PqNBits   { get; set; } = 8;

        // ── DiskANN / Vamana ──────────────────────────────────────────────────

        /// <summary>Max graph out-degree. Default 32.</summary>
        public int   DiskannR     { get; set; } = 32;
        /// <summary>Build-time search list size. Default 100.</summary>
        public int   DiskannL     { get; set; } = 100;
        /// <summary>Pruning parameter (≥ 1.0). Default 1.2.</summary>
        public float DiskannAlpha { get; set; } = 1.2f;

        // ── LSH ───────────────────────────────────────────────────────────────

        /// <summary>Number of hash tables. Default 10.</summary>
        public int   LshL { get; set; } = 10;
        /// <summary>Hash functions per table. Default 8.</summary>
        public int   LshK { get; set; } = 8;
        /// <summary>Bucket width (E2LSH). Default 10.0.</summary>
        public float LshW { get; set; } = 10.0f;

        // ── ScaNN ─────────────────────────────────────────────────────────────

        /// <summary>Coarse IVF partitions. Default 128.</summary>
        public int   ScannNList   { get; set; } = 128;
        /// <summary>Partitions to probe during search. Default 32.</summary>
        public int   ScannNProbe  { get; set; } = 32;
        /// <summary>PQ sub-spaces. Default 8.</summary>
        public int   ScannPqM     { get; set; } = 8;
        /// <summary>Bits per PQ sub-code (4 or 8). Default 8.</summary>
        public int   ScannPqBits  { get; set; } = 8;
        /// <summary>Candidates to rerank with exact distances. Default 100.</summary>
        public int   ScannRerankK { get; set; } = 100;
        /// <summary>Anisotropic penalty η. Default 0.2.</summary>
        public float ScannAqEta   { get; set; } = 0.2f;

        // ── Presets ───────────────────────────────────────────────────────────

        /// <summary>High-recall preset: large M and ef values for HNSW.</summary>
        public static PistaDBParams HighRecall => new PistaDBParams
        {
            HnswM              = 32,
            HnswEfConstruction = 400,
            HnswEfSearch       = 200,
        };

        /// <summary>Low-latency preset: smaller ef values for HNSW.</summary>
        public static PistaDBParams LowLatency => new PistaDBParams
        {
            HnswM              = 16,
            HnswEfConstruction = 100,
            HnswEfSearch       = 20,
        };
    }

    // ── Exception ─────────────────────────────────────────────────────────────

    /// <summary>Thrown when a PistaDB native operation fails.</summary>
    public class PistaDBException : Exception
    {
        public PistaDBException(string message) : base(message) { }
        public PistaDBException(string message, Exception inner) : base(message, inner) { }
    }
}
