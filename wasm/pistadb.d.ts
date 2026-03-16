/**
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 *
 * pistadb.d.ts — TypeScript declarations for the PistaDB WebAssembly binding
 *
 * Copy this file alongside pistadb.js / pistadb.wasm in your project.
 */

// ── Enums ─────────────────────────────────────────────────────────────────────

/** Vector distance metric. */
export declare const enum Metric {
  L2      = 0,  /** Euclidean distance. Default. */
  Cosine  = 1,  /** Cosine similarity (as distance). Ideal for text embeddings. */
  IP      = 2,  /** Inner product (stored negative). For dot-product similarity. */
  L1      = 3,  /** Manhattan distance. */
  Hamming = 4,  /** Hamming distance. For binary/integer vectors. */
}

/** ANN index algorithm. */
export declare const enum IndexType {
  Linear  = 0,  /** Brute-force exact scan. */
  HNSW    = 1,  /** Hierarchical NSW. Recommended for RAG. */
  IVF     = 2,  /** Inverted File Index. Requires train() before insert. */
  IVF_PQ  = 3,  /** IVF + Product Quantization. Low memory, lossy. */
  DiskANN = 4,  /** Vamana / DiskANN. Scales to billion vectors. */
  LSH     = 5,  /** Locality-Sensitive Hashing. Ultra-low memory. */
  ScaNN   = 6,  /** Anisotropic Vector Quantization. Best recall. */
}

// ── Parameter object ──────────────────────────────────────────────────────────

/**
 * Index tuning parameters.
 * All fields are optional — omitted fields use the library defaults.
 * Pass `null` or `undefined` to use all defaults.
 */
export interface PistaDBParams {
  // HNSW
  hnsw_m?:               number;  // default 16
  hnsw_ef_construction?: number;  // default 200
  hnsw_ef_search?:       number;  // default 50
  // IVF / IVF_PQ
  ivf_nlist?:  number;  // default 128
  ivf_nprobe?: number;  // default 8
  pq_m?:       number;  // default 8
  pq_nbits?:   number;  // default 8 (4 or 8)
  // DiskANN / Vamana
  diskann_r?:     number;  // default 32
  diskann_l?:     number;  // default 100
  diskann_alpha?: number;  // default 1.2
  // LSH
  lsh_l?: number;  // default 10
  lsh_k?: number;  // default 8
  lsh_w?: number;  // default 10.0
  // ScaNN
  scann_nlist?:    number;  // default 128
  scann_nprobe?:   number;  // default 32
  scann_pq_m?:     number;  // default 8
  scann_pq_bits?:  number;  // default 8
  scann_rerank_k?: number;  // default 100
  scann_aq_eta?:   number;  // default 0.2
}

// ── Result types ──────────────────────────────────────────────────────────────

/** A single KNN search result. */
export interface SearchResult {
  /** User-supplied vector id (JS number, safe for id < 2^53). */
  id: number;
  /** Distance from the query. Lower is more similar for L2/L1/Hamming/Cosine. */
  distance: number;
  /** Optional human-readable label. Empty string if no label was set. */
  label: string;
}

/** A stored vector entry returned by Database.get(). */
export interface VectorEntry {
  /** The raw float vector as a new, detached Float32Array. */
  vector: Float32Array;
  /** Optional label. Empty string if no label was set. */
  label: string;
}

// ── Database class ────────────────────────────────────────────────────────────

export interface Database {
  /**
   * Persist the database to its .pst file.
   * @throws Error on I/O failure.
   */
  save(): void;

  /**
   * Insert a vector.
   * @param id    Unique numeric id (safe for id < 2^53).
   * @param vec   Float32Array of length `dim`.
   * @param label Optional human-readable label. Pass "" for none.
   * @throws Error on duplicate id or other error.
   */
  insert(id: number, vec: Float32Array, label: string): void;

  /**
   * Logically delete a vector by id.
   * @note Named 'remove' — 'delete' is reserved in JavaScript.
   * @throws Error if id is not found.
   */
  remove(id: number): void;

  /**
   * Replace the vector data for an existing id.
   * @throws Error if id is not found.
   */
  update(id: number, vec: Float32Array): void;

  /**
   * Retrieve the stored vector and label for a given id.
   * @throws Error if id is not found.
   */
  get(id: number): VectorEntry;

  /**
   * K-nearest-neighbour search.
   * @param query Float32Array of length `dim`.
   * @param k     Number of results requested.
   * @returns     Up to k results ordered by ascending distance.
   * @throws Error on index error.
   */
  search(query: Float32Array, k: number): SearchResult[];

  /**
   * Train the index on currently inserted vectors.
   * Required before insert() for IndexType.IVF and IndexType.IVF_PQ.
   * @throws Error on training failure.
   */
  train(): void;

  /** Number of active (non-deleted) vectors. */
  count(): number;

  /** Vector dimension this database was opened with. */
  dim(): number;

  /** Distance metric in use (one of the Metric enum values). */
  metric(): Metric;

  /** Index algorithm in use (one of the IndexType enum values). */
  indexType(): IndexType;

  /** Human-readable description of the last native error. */
  lastError(): string;

  /**
   * Free the underlying C++ object.
   * **Must be called** when done to prevent memory leaks.
   * After calling delete(), do not call any other methods on this object.
   */
  delete(): void;
}

// ── Module interface ──────────────────────────────────────────────────────────

/** Emscripten virtual filesystem (exposed via EXPORTED_RUNTIME_METHODS). */
export interface EmscriptenFS {
  mkdir(path: string): void;
  mount(fs: object, opts: object, mountpoint: string): void;
  syncfs(populate: boolean, callback: (err: Error | null) => void): void;
  writeFile(path: string, data: Uint8Array | string): void;
  readFile(path: string): Uint8Array;
  unlink(path: string): void;
  // ... full Emscripten FS API
}

/** The initialised PistaDB WebAssembly module. */
export interface PistaDBModule {
  /**
   * Open or create a .pst database.
   *
   * @param path       File path on the Emscripten virtual filesystem.
   *                   In browser: in-memory (MEMFS) by default.
   *                   In Node.js: real filesystem path.
   * @param dim        Vector dimension.
   * @param metric     Distance metric (use M.Metric.* constants).
   * @param indexType  Index algorithm (use M.IndexType.* constants).
   * @param params     Optional parameter overrides, or null for defaults.
   *
   * @example
   * const db = new M.Database('data.pst', 128, M.Metric.Cosine, M.IndexType.HNSW, null);
   * // ...
   * db.delete();  // free C++ memory when done
   */
  Database: new (
    path: string,
    dim: number,
    metric: Metric,
    indexType: IndexType,
    params: PistaDBParams | null | undefined,
  ) => Database;

  /** Distance metric constants. */
  Metric: {
    L2:      Metric.L2;
    Cosine:  Metric.Cosine;
    IP:      Metric.IP;
    L1:      Metric.L1;
    Hamming: Metric.Hamming;
  };

  /** Index algorithm constants. */
  IndexType: {
    Linear:  IndexType.Linear;
    HNSW:    IndexType.HNSW;
    IVF:     IndexType.IVF;
    IVF_PQ:  IndexType.IVF_PQ;
    DiskANN: IndexType.DiskANN;
    LSH:     IndexType.LSH;
    ScaNN:   IndexType.ScaNN;
  };

  /** Emscripten virtual filesystem. Use for IDBFS mounting in browser. */
  FS: EmscriptenFS;

  /** IndexedDB filesystem (for persistent browser storage). */
  IDBFS: object;

  /** Native library version string (e.g. "1.0.0"). */
  Database_version: () => string;
}

// ── Factory function ──────────────────────────────────────────────────────────

/**
 * Initialise the PistaDB WebAssembly module.
 * Returns a Promise that resolves once the .wasm binary has been loaded.
 *
 * @example
 * import PistaDB from './pistadb.js';
 * const M = await PistaDB();
 * const db = new M.Database('knowledge.pst', 384, M.Metric.Cosine, M.IndexType.HNSW, null);
 */
declare function PistaDB(options?: object): Promise<PistaDBModule>;
export default PistaDB;
