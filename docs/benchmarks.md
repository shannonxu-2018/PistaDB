# Benchmarks & Internals

---

## Large-Scale CRUD Benchmark

Benchmarked on **9,000,000 records** · `dim=128` · L2 metric · Windows AMD64 · Python 3.13 · 69 GB RAM.

Full benchmark script: [`benchmarks/benchmark_10m.py`](../benchmarks/benchmark_10m.py)

---

### IVF Index — Complete CRUD at 9 M Records

Configuration: `nlist=500`, `nprobe=50` (10% of clusters probed per query), 10 K training vectors.

| Operation | Records | Time | Throughput |
|-----------|---------|------|-----------|
| **INSERT** (post-train) | 9,000,000 | 2m 26s | **61,494 vec/s** |
| **SEARCH** k=10 | 200 queries | — | p50=**222 ms** · p95=245 ms · p99=258 ms · QPS=4 |
| **UPDATE** | 100,000 | 4m 35s | **363 ops/s** |
| **DELETE** | 100,000 | 6m 10s | **270 ops/s** |

**INSERT milestone breakdown:**

| Cumulative records | Elapsed | Throughput |
|--------------------|---------|-----------|
| 1,000,000 | 16.67 s | 59,989 vec/s |
| 2,000,000 | 33.14 s | 60,355 vec/s |
| 3,000,000 | 49.85 s | 60,176 vec/s |
| 4,000,000 | 1m 06s | 60,120 vec/s |
| 5,000,000 | 1m 23s | 59,737 vec/s |
| 6,000,000 | 1m 40s | 59,938 vec/s |
| 7,000,000 | 1m 56s | 60,016 vec/s |
| 8,000,000 | 2m 12s | 60,387 vec/s |
| 9,000,000 | 2m 26s | 61,494 vec/s |

> IVF insert is O(1) — each vector is simply assigned to its nearest centroid, so throughput is nearly flat across all scales. Search latency at `nprobe=50` reflects scanning 50 clusters × ~18 K vectors each (≈ 900 K distance computations per query at 9 M scale). UPDATE and DELETE perform an O(N) id-scan; workloads with heavy mutations benefit more from HNSW which uses lazy deletion without a full-table scan.

---

### HNSW Index — INSERT Throughput at Scale

Configuration: `M=8`, `ef_construction=50`, `ef_search=32` · dim=128 · L2.

| Cumulative records | Elapsed | Throughput |
|--------------------|---------|-----------|
| 1,000,000 | 2m 54s | 5,741 vec/s |
| 2,000,000 | 7m 20s | 4,540 vec/s |
| 3,000,000 | 11m 27s | 4,362 vec/s |
| 4,000,000 | 15m 49s | 4,212 vec/s |
| 5,000,000 | 21m 54s | 3,805 vec/s |
| 6,000,000 | 32m 02s | 3,122 vec/s |
| 7,000,000 | 44m 00s | 2,651 vec/s |
| 8,000,000 | 56m 03s | 2,378 vec/s |
| 9,000,000 | 1h 08m | 2,196 vec/s |

HNSW insert cost grows logarithmically with N (each insert traverses the graph with O(M·log N) distance computations), which explains the gradual throughput decline. In return, HNSW delivers **sub-millisecond approximate k-NN search** — independent of dataset size.

### Verified at 10 M Vectors

HNSW successfully inserted **10,000,000 vectors** (`dim=128`, `M=8`, `ef_construction=50`) on Windows AMD64 with no OOM error, enabled by the VecStore chunked storage layer.

| Cumulative | Elapsed | Throughput |
|---|---|---|
| 500,000 | 94s | 5,314 vec/s |
| 1,000,000 | 3m 47s | 4,404 vec/s |
| 5,000,000 | 24m 53s | 3,348 vec/s |
| **10,000,000** | **1h 22m** | **2,030 vec/s** |

---

## SIMD Distance Kernels

All five distance functions are backed by hand-written SIMD kernels, selected automatically at **runtime** with zero configuration:

| Architecture | ISA | Main loop | Typical speedup |
|---|---|---|---|
| x86-64 (Haswell+) | **AVX2 + FMA** | 16 floats/iter, dual-accumulator unroll, fused multiply-add | 4–8× vs. scalar |
| ARM / Apple Silicon | **NEON** | 16 floats/iter (4× `float32x4_t`), 4-accumulator unroll | 3–5× vs. scalar |
| Any other | Scalar | Standard C11 loop | baseline |

**Runtime dispatch** — at first call, the CPU is probed once (via `__builtin_cpu_supports` on GCC/Clang, or `__cpuid` + `_xgetbv` on MSVC). The active function pointers are patched in-place; all subsequent calls jump directly to the best implementation with no branching.

| Kernel | AVX2 technique | NEON technique |
|---|---|---|
| `vec_dot` / `dist_ip` | `_mm256_fmadd_ps` × 2 YMM | `vmlaq_f32` × 4 Q-regs |
| `dist_l2sq` / `dist_l2` | `_mm256_sub_ps` → `_mm256_fmadd_ps` | `vsubq_f32` → `vmlaq_f32` |
| `dist_cosine` | Single-pass 3-accumulator (dot, ‖a‖², ‖b‖²) | 6 Q-reg accumulators |
| `dist_l1` | Sign-bit mask (`AND 0x7FFFFFFF`) | `vabsq_f32` → `vaddq_f32` |
| `dist_hamming` | `_mm256_cmp_ps` + `movemask` + `popcount` | `vceqq_f32` + `vshrq_n_u32` + `vaddvq_u32` |

---

## File Format

```
PistaDB File Format v1.0
───────────────────────────────────────────
[Header]  128 bytes (fixed)
  magic[4]           = "PSDB"
  version_major[2]   = 1
  version_minor[2]   = 0
  flags[4]           = 0  (reserved for future use)
  dimension[4]       = vector dimensionality
  metric_type[2]     = distance metric enum
  index_type[2]      = index algorithm enum
  num_vectors[8]     = total vectors (including soft-deleted)
  next_id[8]         = next auto-increment id hint
  vec_offset[8]      = byte offset of data block
  vec_size[8]        = data block size in bytes
  idx_offset[8]      = byte offset of auxiliary index data
  idx_size[8]        = auxiliary index data size
  reserved[54]       = 0
  header_crc[4]      = CRC32 over header bytes [0..123]

[Vector + Index Data]  at vec_offset
  Per-index binary serialization (self-describing, versioned)
```

Every index type serializes itself into this block — graph edges, cluster centroids, PQ codebooks, projection matrices, and raw float vectors all in one place. The file is fully self-contained: no sidecar files, no registry entries, no environment variables needed at load time.

---

## Project Structure

```
PistaDB/
├── src/                          # C99 core — all platforms share this
│   ├── pistadb_types.h           # Shared types, error codes, default params
│   ├── pistadb.h / .c            # Primary database API
│   ├── distance.h / .c           # Runtime dispatch + scalar fallbacks for 5 metrics
│   ├── distance_avx2.c           # AVX2+FMA kernels (compiled with -mavx2 -mfma)
│   ├── distance_neon.c           # ARM NEON kernels (AArch64 built-in)
│   ├── pistadb_batch.h / .c      # Multi-threaded batch insert (thread pool + ring queue)
│   ├── pistadb_cache.h / .c      # Embedding cache (FNV-1a hash map + LRU list + .pcc file)
│   ├── pistadb_txn.h / .c        # Transaction API (atomic multi-op groups, undo-on-failure)
│   ├── index_linear.*            # Exact brute-force scan
│   ├── index_hnsw.*              # HNSW (Malkov & Yashunin, 2018)
│   ├── index_ivf.*               # IVF with k-means clustering
│   ├── index_ivf_pq.*            # IVF + Product Quantization
│   ├── index_diskann.*           # Vamana / DiskANN (Subramanya et al., 2019)
│   ├── index_lsh.*               # E2LSH + sign-based LSH
│   └── index_scann.*             # ScaNN: Anisotropic Vector Quantization (Guo et al., ICML 2020)
├── python/                       # Python binding (pure ctypes, no Cython)
├── go/                           # Go binding (CGO), module: pistadb.io/go
├── android/                      # Android JNI bridge + Java + Kotlin API
├── ios/                          # iOS/macOS ObjC + Swift SPM package
├── wasm/                         # WebAssembly binding (Emscripten + Embind)
├── cpp/                          # C++ header-only wrapper (C++17)
├── rust/                         # Rust FFI crate (no external deps)
├── csharp/                       # C# P/Invoke binding (.NET Standard 2.0)
├── tests/
│   └── test_pistadb.py           # 109-test pytest suite
├── examples/
│   └── example.py                # 12 end-to-end usage scenarios
├── benchmarks/
│   └── benchmark_10m.py          # Large-scale CRUD benchmark script
├── Package.swift                 # SPM manifest
├── CMakeLists.txt
├── build.sh                      # Linux / macOS build script
└── build.bat                     # Windows MSVC build script
```
