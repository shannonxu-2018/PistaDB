/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB C# Binding — PistaDatabase.cs
 * Main managed wrapper over the native C library via P/Invoke.
 *
 * Usage:
 *   var db = PistaDatabase.Open("my.pst", dim: 128, metric: Metric.Cosine);
 *   db.Insert(1, embedding, label: "hello");
 *   var results = db.Search(query, k: 5);
 *   db.Save();
 *   db.Dispose();
 *
 * Thread-safety: all public members are protected by an internal lock.
 * Note: Dispose does NOT auto-save. Call Save() before disposing.
 */

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using PistaDB.Native;

namespace PistaDB
{
    /// <summary>
    /// Embedded vector database backed by a single <c>.pst</c> file.
    /// Implements <see cref="IDisposable"/> — use with <c>using</c> or call
    /// <see cref="Close"/> explicitly. <b>Does not auto-save on dispose.</b>
    /// </summary>
    public sealed class PistaDatabase : IDisposable
    {
        private IntPtr _handle;
        private readonly object _lock = new object();
        private bool _disposed;

        private PistaDatabase(IntPtr handle) => _handle = handle;

        // ── Factory ───────────────────────────────────────────────────────────

        /// <summary>
        /// Open an existing <c>.pst</c> database or create a new one.
        /// </summary>
        /// <param name="path">File path. Created if it does not exist.</param>
        /// <param name="dim">Vector dimension (must match file if loading).</param>
        /// <param name="metric">Distance metric. Default: L2.</param>
        /// <param name="indexType">Index algorithm. Default: HNSW.</param>
        /// <param name="params">Index parameters. Pass <c>null</c> for library defaults.</param>
        /// <exception cref="PistaDBException">Thrown if the database cannot be opened.</exception>
        public static PistaDatabase Open(
            string path,
            int dim,
            Metric metric = Metric.L2,
            IndexType indexType = IndexType.HNSW,
            PistaDBParams? @params = null)
        {
            if (string.IsNullOrEmpty(path)) throw new ArgumentNullException(nameof(path));
            if (dim <= 0)                   throw new ArgumentOutOfRangeException(nameof(dim));

            IntPtr handle;
            if (@params == null)
            {
                handle = NativeMethods.OpenDefaults(path, dim, (int)metric, (int)indexType, IntPtr.Zero);
            }
            else
            {
                var np = ToNative(@params);
                handle = NativeMethods.Open(path, dim, (int)metric, (int)indexType, ref np);
            }

            if (handle == IntPtr.Zero)
                throw new PistaDBException("pistadb_open returned null — check path permissions and parameters.");

            return new PistaDatabase(handle);
        }

        // ── Lifecycle ─────────────────────────────────────────────────────────

        /// <summary>Persist the in-memory state to the <c>.pst</c> file.</summary>
        /// <exception cref="PistaDBException">Thrown on I/O failure.</exception>
        public void Save()
        {
            lock (_lock)
            {
                ThrowIfDisposed();
                if (NativeMethods.Save(_handle) != 0) ThrowLastError();
            }
        }

        /// <summary>
        /// Close the native database handle and release all resources.
        /// Safe to call multiple times. Auto-called by <see cref="Dispose"/>.
        /// </summary>
        public void Close()
        {
            lock (_lock)
            {
                if (!_disposed && _handle != IntPtr.Zero)
                {
                    NativeMethods.Close(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        /// <inheritdoc />
        public void Dispose() => Close();

        // ── CRUD ──────────────────────────────────────────────────────────────

        /// <summary>
        /// Insert a new vector.
        /// </summary>
        /// <param name="id">User-supplied unique identifier.</param>
        /// <param name="vec">Float array of length <see cref="Dim"/>.</param>
        /// <param name="label">Optional human-readable label (max 255 bytes).</param>
        /// <exception cref="PistaDBException">Thrown if id already exists or on error.</exception>
        public void Insert(ulong id, float[] vec, string? label = null)
        {
            if (vec == null) throw new ArgumentNullException(nameof(vec));
            lock (_lock)
            {
                ThrowIfDisposed();
                if (NativeMethods.Insert(_handle, id, label, vec) != 0) ThrowLastError();
            }
        }

        /// <summary>Logically delete a vector. Space is reclaimed on the next Save/rebuild.</summary>
        /// <exception cref="PistaDBException">Thrown if id is not found or on error.</exception>
        public void Delete(ulong id)
        {
            lock (_lock)
            {
                ThrowIfDisposed();
                if (NativeMethods.Delete(_handle, id) != 0) ThrowLastError();
            }
        }

        /// <summary>Replace the vector data for an existing id.</summary>
        /// <exception cref="PistaDBException">Thrown if id is not found or on error.</exception>
        public void Update(ulong id, float[] vec)
        {
            if (vec == null) throw new ArgumentNullException(nameof(vec));
            lock (_lock)
            {
                ThrowIfDisposed();
                if (NativeMethods.Update(_handle, id, vec) != 0) ThrowLastError();
            }
        }

        /// <summary>Retrieve the raw vector and label for a given id.</summary>
        /// <exception cref="PistaDBException">Thrown if id is not found or on error.</exception>
        public VectorEntry Get(ulong id)
        {
            lock (_lock)
            {
                ThrowIfDisposed();
                int dim = NativeMethods.Dim(_handle);
                var vec       = new float[dim];
                var labelBuf  = new byte[256];
                if (NativeMethods.Get(_handle, id, vec, labelBuf) != 0) ThrowLastError();
                return new VectorEntry(vec, NativeMethods.LabelBytesToString(labelBuf));
            }
        }

        // ── Search ────────────────────────────────────────────────────────────

        /// <summary>
        /// K-nearest-neighbour search.
        /// </summary>
        /// <param name="query">Query vector of length <see cref="Dim"/>.</param>
        /// <param name="k">Number of results requested.</param>
        /// <returns>Up to <paramref name="k"/> results ordered by ascending distance.</returns>
        /// <exception cref="PistaDBException">Thrown on index error.</exception>
        public IReadOnlyList<SearchResult> Search(float[] query, int k)
        {
            if (query == null) throw new ArgumentNullException(nameof(query));
            if (k <= 0)        throw new ArgumentOutOfRangeException(nameof(k));
            lock (_lock)
            {
                ThrowIfDisposed();
                var buf = new NativeResult[k];
                int n = NativeMethods.Search(_handle, query, k, buf);
                if (n < 0) ThrowLastError();
                var results = new SearchResult[n];
                for (int i = 0; i < n; i++)
                    results[i] = new SearchResult(buf[i].id, buf[i].distance, buf[i].label ?? string.Empty);
                return results;
            }
        }

        // ── Index management ──────────────────────────────────────────────────

        /// <summary>
        /// Train the index on currently inserted vectors.
        /// Required before <see cref="Insert"/> for <see cref="IndexType.IVF"/> and
        /// <see cref="IndexType.IVF_PQ"/>. Optional for HNSW/DiskANN (triggers rebuild).
        /// </summary>
        /// <exception cref="PistaDBException">Thrown on training failure.</exception>
        public void Train()
        {
            lock (_lock)
            {
                ThrowIfDisposed();
                if (NativeMethods.Train(_handle) != 0) ThrowLastError();
            }
        }

        // ── Properties ────────────────────────────────────────────────────────

        /// <summary>Number of active (non-deleted) vectors.</summary>
        public int Count
        {
            get { lock (_lock) { ThrowIfDisposed(); return NativeMethods.Count(_handle); } }
        }

        /// <summary>Vector dimension this database was opened with.</summary>
        public int Dim
        {
            get { lock (_lock) { ThrowIfDisposed(); return NativeMethods.Dim(_handle); } }
        }

        /// <summary>Distance metric in use.</summary>
        public Metric Metric
        {
            get { lock (_lock) { ThrowIfDisposed(); return (Metric)NativeMethods.Metric(_handle); } }
        }

        /// <summary>Index algorithm in use.</summary>
        public IndexType IndexType
        {
            get { lock (_lock) { ThrowIfDisposed(); return (IndexType)NativeMethods.IndexType(_handle); } }
        }

        /// <summary>Human-readable description of the last native error.</summary>
        public string LastError
        {
            get { lock (_lock) { return NativeMethods.PtrToString(NativeMethods.LastError(_handle)); } }
        }

        /// <summary>Native library version string (e.g. "1.0.0").</summary>
        public static string Version => NativeMethods.PtrToString(NativeMethods.Version());

        // ── Async wrappers ────────────────────────────────────────────────────
        // All async methods run the blocking native call on the thread pool.

        /// <inheritdoc cref="Insert"/>
        public Task InsertAsync(ulong id, float[] vec, string? label = null, CancellationToken ct = default) =>
            Task.Run(() => Insert(id, vec, label), ct);

        /// <inheritdoc cref="Delete"/>
        public Task DeleteAsync(ulong id, CancellationToken ct = default) =>
            Task.Run(() => Delete(id), ct);

        /// <inheritdoc cref="Update"/>
        public Task UpdateAsync(ulong id, float[] vec, CancellationToken ct = default) =>
            Task.Run(() => Update(id, vec), ct);

        /// <inheritdoc cref="Get"/>
        public Task<VectorEntry> GetAsync(ulong id, CancellationToken ct = default) =>
            Task.Run(() => Get(id), ct);

        /// <inheritdoc cref="Search"/>
        public Task<IReadOnlyList<SearchResult>> SearchAsync(float[] query, int k, CancellationToken ct = default) =>
            Task.Run(() => Search(query, k), ct);

        /// <inheritdoc cref="Train"/>
        public Task TrainAsync(CancellationToken ct = default) =>
            Task.Run(() => Train(), ct);

        /// <inheritdoc cref="Save"/>
        public Task SaveAsync(CancellationToken ct = default) =>
            Task.Run(() => Save(), ct);

        // ── Private helpers ───────────────────────────────────────────────────

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(PistaDatabase));
        }

        private void ThrowLastError()
        {
            string msg = NativeMethods.PtrToString(NativeMethods.LastError(_handle));
            throw new PistaDBException(string.IsNullOrEmpty(msg) ? "Unknown PistaDB error" : msg);
        }

        private static NativeParams ToNative(PistaDBParams p) => new NativeParams
        {
            hnsw_M               = p.HnswM,
            hnsw_ef_construction = p.HnswEfConstruction,
            hnsw_ef_search       = p.HnswEfSearch,
            ivf_nlist            = p.IvfNList,
            ivf_nprobe           = p.IvfNProbe,
            pq_M                 = p.PqM,
            pq_nbits             = p.PqNBits,
            diskann_R            = p.DiskannR,
            diskann_L            = p.DiskannL,
            diskann_alpha        = p.DiskannAlpha,
            lsh_L                = p.LshL,
            lsh_K                = p.LshK,
            lsh_w                = p.LshW,
            scann_nlist          = p.ScannNList,
            scann_nprobe         = p.ScannNProbe,
            scann_pq_M           = p.ScannPqM,
            scann_pq_bits        = p.ScannPqBits,
            scann_rerank_k       = p.ScannRerankK,
            scann_aq_eta         = p.ScannAqEta,
        };
    }
}
