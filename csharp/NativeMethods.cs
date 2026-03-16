/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * PistaDB C# Binding — NativeMethods.cs
 * P/Invoke declarations and native struct definitions.
 */

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace PistaDB.Native
{
    /// <summary>
    /// Mirrors C struct PistaDBParams exactly (Sequential layout).
    /// Field order must match pistadb_types.h.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeParams
    {
        // HNSW
        public int   hnsw_M;
        public int   hnsw_ef_construction;
        public int   hnsw_ef_search;
        // IVF / IVF_PQ
        public int   ivf_nlist;
        public int   ivf_nprobe;
        public int   pq_M;
        public int   pq_nbits;
        // DiskANN / Vamana
        public int   diskann_R;
        public int   diskann_L;
        public float diskann_alpha;
        // LSH
        public int   lsh_L;
        public int   lsh_K;
        public float lsh_w;
        // ScaNN
        public int   scann_nlist;
        public int   scann_nprobe;
        public int   scann_pq_M;
        public int   scann_pq_bits;
        public int   scann_rerank_k;
        public float scann_aq_eta;
    }

    /// <summary>
    /// Mirrors C struct PistaDBResult exactly.
    /// label is a fixed 256-byte ANSI/UTF-8 buffer.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    internal struct NativeResult
    {
        public ulong id;
        public float distance;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
        public string label;
    }

    internal static class NativeMethods
    {
        // Library name — .NET runtime resolves platform suffix automatically:
        //   Windows : pistadb.dll
        //   Linux   : libpistadb.so
        //   macOS   : libpistadb.dylib
        private const string Lib = "pistadb";

        // ── Lifecycle ─────────────────────────────────────────────────────────

        // Two overloads of pistadb_open: one with params struct, one accepting null.
        [DllImport(Lib, EntryPoint = "pistadb_open", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr Open(
            [MarshalAs(UnmanagedType.LPStr)] string path,
            int dim, int metric, int indexType,
            ref NativeParams p);

        [DllImport(Lib, EntryPoint = "pistadb_open", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr OpenDefaults(
            [MarshalAs(UnmanagedType.LPStr)] string path,
            int dim, int metric, int indexType,
            IntPtr nullParams);   // pass IntPtr.Zero

        [DllImport(Lib, EntryPoint = "pistadb_close", CallingConvention = CallingConvention.Cdecl)]
        public static extern void Close(IntPtr db);

        [DllImport(Lib, EntryPoint = "pistadb_save", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Save(IntPtr db);

        // ── CRUD ──────────────────────────────────────────────────────────────

        [DllImport(Lib, EntryPoint = "pistadb_insert", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Insert(
            IntPtr db, ulong id,
            [MarshalAs(UnmanagedType.LPStr)] string? label,
            [In] float[] vec);

        [DllImport(Lib, EntryPoint = "pistadb_delete", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Delete(IntPtr db, ulong id);

        [DllImport(Lib, EntryPoint = "pistadb_update", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Update(IntPtr db, ulong id, [In] float[] vec);

        // out_label is a caller-supplied 256-byte buffer; pass byte[256], may be null.
        [DllImport(Lib, EntryPoint = "pistadb_get", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Get(
            IntPtr db, ulong id,
            [Out] float[] outVec,
            [Out] byte[] outLabel);

        // ── Search ────────────────────────────────────────────────────────────

        [DllImport(Lib, EntryPoint = "pistadb_search", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Search(
            IntPtr db,
            [In] float[] query,
            int k,
            [Out] NativeResult[] results);

        // ── Index management ──────────────────────────────────────────────────

        [DllImport(Lib, EntryPoint = "pistadb_train", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Train(IntPtr db);

        // ── Metadata ──────────────────────────────────────────────────────────

        [DllImport(Lib, EntryPoint = "pistadb_count", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Count(IntPtr db);

        [DllImport(Lib, EntryPoint = "pistadb_dim", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Dim(IntPtr db);

        [DllImport(Lib, EntryPoint = "pistadb_metric", CallingConvention = CallingConvention.Cdecl)]
        public static extern int Metric(IntPtr db);

        [DllImport(Lib, EntryPoint = "pistadb_index_type", CallingConvention = CallingConvention.Cdecl)]
        public static extern int IndexType(IntPtr db);

        [DllImport(Lib, EntryPoint = "pistadb_last_error", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr LastError(IntPtr db);

        // ── Version ───────────────────────────────────────────────────────────

        [DllImport(Lib, EntryPoint = "pistadb_version", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr Version();

        // ── Helpers ───────────────────────────────────────────────────────────

        internal static string PtrToString(IntPtr ptr) =>
            ptr == IntPtr.Zero ? string.Empty : Marshal.PtrToStringAnsi(ptr) ?? string.Empty;

        /// <summary>Convert a null-terminated UTF-8 byte buffer to a managed string.</summary>
        internal static string LabelBytesToString(byte[] bytes)
        {
            int len = Array.IndexOf(bytes, (byte)0);
            if (len < 0) len = bytes.Length;
            return Encoding.UTF8.GetString(bytes, 0, len);
        }
    }
}
