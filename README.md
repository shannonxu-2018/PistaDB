<div align="center">

[English](./README.md) · [中文](./README_CN.md)

<h1>🌰 PistaDB</h1>

<p><strong>The embedded vector database for LLM-native applications.</strong><br>
RAG-ready · Zero dependencies · Single-file storage · MIT Licensed</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C99](https://img.shields.io/badge/C-99-blue.svg)](https://en.wikipedia.org/wiki/C99)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776ab.svg)]()
[![Android](https://img.shields.io/badge/Android-API%2021%2B-3ddc84.svg)]()
[![iOS](https://img.shields.io/badge/iOS-13%2B-lightgrey.svg)]()
[![.NET](https://img.shields.io/badge/.NET-Standard%202.0%2B-512bd4.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20Android%20%7C%20iOS-lightgrey.svg)]()
[![Tests](https://img.shields.io/badge/tests-48%2F48%20passing-brightgreen.svg)]()

</div>

---

> **Every LLM application eventually needs a vector database.**
> For retrieval-augmented generation (RAG), semantic search, agent memory, or embedding caches —
> the standard answer is a cloud service or a containerized cluster. PistaDB disagrees.
>
> Small, dense, and full of value — like the nut it's named after —
> **PistaDB gives you a production-grade vector store in a single `.pst` file and a C library with zero dependencies.**
> Ship it inside a desktop app. Bundle it in an edge device. Drop it next to your Python script.
> No Docker. No API keys. No data leaving the machine.

---

## ✨ Why PistaDB?

| | PistaDB | Cloud / Server Vector DB |
|---|---|---|
| Deployment | Copy a `.dll` / `.so` | Docker, Kubernetes, cloud subscriptions |
| Storage | One `.pst` file | Separate data + WAL + config + sidecar files |
| Privacy | **All data stays local** | Embeddings sent over the network |
| Memory | Configurable, minimal | GBs of JVM / runtime overhead |
| Dependencies | **None** (pure C99) | Dozens of packages |
| Latency | **Sub-millisecond** on a laptop | Network round-trips |
| Cost | Free forever (MIT) | Per-query or per-vector pricing |

PistaDB is purpose-built for **local RAG pipelines, offline AI agents, privacy-sensitive applications, edge inference, and anywhere shipping a full vector database cluster is impractical** — which, honestly, is most places.

---

## 🤖 Built for the LLM Stack

Modern LLM applications share a common pattern: convert text (or images, audio, code) into embedding vectors, store them, and retrieve the most relevant ones at inference time to give the model precise context. PistaDB is the storage layer for exactly that workflow.

### RAG in 20 Lines

```python
from pistadb import PistaDB, Metric, Index
from sentence_transformers import SentenceTransformer  # or any embedding model

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim embeddings

# ── Index your knowledge base ──────────────────────────────────────────────
docs = [
    "PistaDB stores vectors locally with zero dependencies.",
    "RAG improves LLM output quality by injecting retrieved context.",
    "HNSW delivers sub-millisecond approximate nearest-neighbor search.",
]
embeddings = model.encode(docs, normalize_embeddings=True).astype("float32")

with PistaDB("knowledge.pst", dim=384, metric=Metric.COSINE, index=Index.HNSW) as db:
    for i, (doc, vec) in enumerate(zip(docs, embeddings)):
        db.insert(i + 1, vec, label=doc)
    db.save()

# ── Retrieve context at inference time ─────────────────────────────────────
question = "How does PistaDB help with retrieval-augmented generation?"
q_vec = model.encode([question], normalize_embeddings=True)[0].astype("float32")

with PistaDB("knowledge.pst", dim=384) as db:
    results = db.search(q_vec, k=3)

context = "\n".join(r.label for r in results)
# → pass `context` to OpenAI / Claude / a local model as part of the prompt
```

### Persistent Agent Memory

LLM agents need memory that survives between sessions. PistaDB stores episodic and semantic memory as embedding vectors — queryable in microseconds, persisted to a single file you control.

```python
# Store a new observation from the agent's session
memory_db = PistaDB("agent_memory.pst", dim=1536, metric=Metric.COSINE, index=Index.HNSW)
vec = openai_embed("User prefers concise technical explanations.")
memory_db.insert(next_id, vec, label="User prefers concise technical explanations.")
memory_db.save()

# Recall relevant memories before the next LLM call
recall_vec = openai_embed(current_user_message)
memories = memory_db.search(recall_vec, k=5)
context = "\n".join(m.label for m in memories)
```

### Private Document Search

PistaDB runs entirely on-device. Embed your documents with a local model (Ollama, llama.cpp), store them in PistaDB, and search without a single byte leaving the machine.

```python
# Offline pipeline — nothing touches the network
from pistadb import build_from_array

vecs   = local_model.encode(my_documents)           # local embedding model
labels = [doc[:255] for doc in my_documents]        # label = document excerpt

db = build_from_array("private_docs.pst", vecs, labels=labels,
                       metric=Metric.COSINE, index=Index.HNSW)
db.save()
```

---

## 🚀 Features

### 7 Production-Ready Index Algorithms

| Index | Algorithm | Best For |
|-------|-----------|----------|
| `LINEAR` | Brute-force exact scan | Ground truth, small embedding sets |
| `HNSW` | Hierarchical Navigable Small World | **Recommended for RAG** — best speed/recall tradeoff |
| `IVF` | Inverted File Index (k-means) | Large knowledge bases with a training budget |
| `IVF_PQ` | IVF + Product Quantization | Memory-constrained deployments |
| `DISKANN` | Vamana graph (DiskANN) | Billion-scale embedding collections |
| `LSH` | Locality-Sensitive Hashing | Ultra-low memory footprint |
| `SCANN` | Anisotropic Vector Quantization (Google ScaNN) | Maximum recall on MIPS / cosine workloads |

### 5 Distance Metrics — All LLM Embedding Models Covered

| Metric | LLM / Embedding Use Case |
|--------|--------------------------|
| `COSINE` | **Text embeddings** — OpenAI `text-embedding-3`, Cohere, `sentence-transformers`, BGE, GTE |
| `IP` | Inner product — embeddings already L2-normalised (same result as cosine, faster) |
| `L2` | Image / multimodal embeddings (CLIP, ImageBind) |
| `L1` | Sparse feature vectors, BM25-style hybrid retrieval |
| `HAMMING` | Binary embeddings, hash-based deduplication |

### SIMD-Accelerated Distance Kernels

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

The scalar fallback is always compiled in and gives identical numerical results — correctness is never traded for speed. The SIMD files (`distance_avx2.c`, `distance_neon.c`) are compiled with their own ISA flags (`-mavx2 -mfma` / no extra flag needed on AArch64) and linked into the same shared library.

### Multi-Threaded Batch Insert

For high-throughput embedding pipelines, PistaDB provides a **thread-pool + ring-buffer** batch insert API that decouples vector generation from index writes:

```
Producer thread 0 ──▶ ┌──────────────────┐
Producer thread 1 ──▶ │  Ring-buffer     │──▶ Worker 0 ─┐
Producer thread N ──▶ │  work queue      │──▶ Worker 1 ─┤──▶ pistadb_insert()
                       │  (bounded MPMC)  │──▶ Worker M ─┘    (serialised)
                       └──────────────────┘
```

**Streaming API** — for online pipelines where embeddings arrive continuously:

```c
#include "pistadb_batch.h"

// Create a batch context: 4 workers, default queue capacity (4096)
PistaDBBatch *batch = pistadb_batch_create(db, 4, 0);

// Any number of producer threads may call push() concurrently.
// The call blocks only when the queue is full (back-pressure).
// vec is copied internally — caller may free it immediately.
pistadb_batch_push(batch, id, label, vec);   // thread-safe

// Wait for all queued items to finish
int errors = pistadb_batch_flush(batch);     // 0 on full success

pistadb_batch_destroy(batch);   // flush + shutdown workers + free memory
```

**Offline bulk API** — single blocking call for pre-loaded arrays:

```c
// ids[n], labels[n] (may be NULL), vecs[n × dim]
// 0 workers → auto-detect hardware_concurrency
int errors = pistadb_batch_insert(db, ids, labels, vecs, n, /*n_threads=*/0);
```

**Performance model:**

| Scenario | Benefit |
|---|---|
| Embedding generation + indexing pipeline | Overlap compute (parallel embeds) with sequential index writes |
| Multi-process embedding servers | Multiple producer threads push concurrently; one worker drains |
| HNSW / DiskANN | Worker does graph-search (read-heavy) while next item is being prepared |
| IVF / ScaNN | Centroid lookup (read-only) overlaps across items in queue |

**Thread safety:** `pistadb_batch_push()` is safe to call from any number of threads simultaneously. All index writes are serialized internally — the underlying `PistaDB` handle does not need external locking while a batch context is active on it.

**Platform:** Win32 `CRITICAL_SECTION` + `CONDITION_VARIABLE` on Windows; `pthread_mutex_t` + `pthread_cond_t` on Linux / macOS / Android / iOS. No external dependencies.

### Embedding Cache — Automatic Input Deduplication

Embedding APIs (OpenAI, Cohere, local models) are expensive. When the same text appears more than once — repeated queries, corpus deduplication, cached document chunks — re-encoding wastes time and money. The embedding cache eliminates redundant calls automatically.

```
┌──────────────────────────────────────────────────────┐
│                   Your Application                   │
│                                                      │
│   text ──► CachedEmbedder ──► EmbeddingCache ──► hit │ ──► float32[]
│                    │               (LRU map)          │
│                    │ miss                             │
│                    ▼                                  │
│              embed_fn(text)   ← model call skipped   │
│              (OpenAI / local)    on cache hit         │
└──────────────────────────────────────────────────────┘
```

**C API** — `src/pistadb_cache.h`:

```c
// Open (or reload) a persistent cache
PistaDBCache *cache = pistadb_cache_open("embed.pcc", /*dim=*/384, /*max=*/100000);

float vec[384];
if (!pistadb_cache_get(cache, text, vec)) {
    my_model_encode(text, vec);           // ← only called on a miss
    pistadb_cache_put(cache, text, vec);  // copy stored internally
}
// use vec …

pistadb_cache_save(cache);   // flush to embed.pcc (survives restarts)
pistadb_cache_close(cache);
```

**Python API** — `EmbeddingCache` + `CachedEmbedder`:

```python
from pistadb import EmbeddingCache, CachedEmbedder

cache    = EmbeddingCache("embed.pcc", dim=384, max_entries=100_000)
embedder = CachedEmbedder(openai_encode, cache, autosave_every=500)

vec  = embedder("What is RAG?")      # calls OpenAI only on first access
vecs = embedder.embed_batch(texts)   # np.ndarray (n, 384)

print(cache.stats())
# CacheStats(hits=4821, misses=179, evictions=0, count=179, hit_rate=96.4%)

cache.close()
```

**Design details:**

| Property | Value |
|---|---|
| Hash function | FNV-1a 64-bit, separate chaining |
| Eviction policy | LRU (doubly-linked list, O(1) promote / evict) |
| Rehash threshold | 75 % load factor, doubles bucket count |
| Persistence format | `.pcc` binary — 64-byte header + variable-length entries |
| Thread safety | Single internal mutex (all public functions protected) |
| Byte order | Little-endian (all platforms) |

The `.pcc` file stores entries in LRU-to-MRU order so a reload via sequential `put()` calls reconstructs the exact same LRU ordering.

### Single-File Storage — Ships With Your App

- **One `.pst` file** contains everything: header, index graph, and raw vectors
- 128-byte fixed header with **CRC32 integrity check**
- Fully forward-compatible binary format (versioned for future extensions)
- Atomic save: buffer-then-flush — no partial writes, no corruption

---

## 🌍 Language & Platform Support

PistaDB's core is written in pure C99. Every language binding wraps this same library through a thin, zero-copy layer — no reimplementation, no divergence in behaviour. The same `.pst` database file is portable across all supported platforms.

### Supported Languages

| Language | Binding mechanism | API style | Where to find it |
|---|---|---|---|
| **C / C++** | Direct `#include` | Native C API, zero overhead | `src/pistadb.h` |
| **Python** | `ctypes` (no Cython) | Pythonic, NumPy-compatible | `python/` |
| **Java** | JNI | `AutoCloseable`, Builder, `synchronized` | `android/src/main/java/` |
| **Kotlin** | JNI + extension functions | DSL builder, coroutines, operator overloads | `android/src/main/kotlin/` |
| **Objective-C** | Direct C interop | Cocoa conventions, `NSError`, `NSLock` | `ios/Sources/PistaDBObjC/` |
| **Swift** | ObjC bridge | `throws`, `async/await`, trailing-closure DSL | `ios/Sources/PistaDB/` |
| **C#** | P/Invoke | `IDisposable`, async/await, thread-safe | `csharp/` |
| **Rust** | FFI (`extern "C"`) | `Send + Sync`, `Drop`, `Result<T, Error>` | `rust/` |
| **C++** | Direct include | RAII, move-only, `std::mutex`, header-only | `cpp/pistadb.hpp` |
| **WASM** | Emscripten / Embind | ESM module, `Float32Array`, TypeScript types | `wasm/` |

### Supported Platforms

| Platform | Toolchain | Library output | ABI targets |
|---|---|---|---|
| **Windows** | MSVC | `pistadb.dll` | x86_64 |
| **Linux** | GCC | `libpistadb.so` | x86_64, aarch64 |
| **macOS** | Clang | `libpistadb.dylib` | x86_64, arm64 |
| **Android** | NDK (Clang) | `libpistadb_jni.so` | arm64-v8a, armeabi-v7a, x86_64, x86 |
| **iOS / macOS** | Xcode / SPM | Static library | arm64, arm64-Simulator, x86_64-Simulator |
| **WASM** | Emscripten | `.wasm` | — *(planned)* |

---

## 📱 Android Integration (Java & Kotlin)

The `android/` module is a self-contained Android library. Add it to your project once and get the full vector-database API in both Java and Kotlin, backed by the native C library through JNI.

### Gradle setup

```groovy
// settings.gradle
include ':android'
project(':android').projectDir = new File('../PistaDB/android')

// app/build.gradle
dependencies {
    implementation project(':android')
}
```

### Java

```java
import com.pistadb.*;

// Open or create — try-with-resources ensures save + close
try (PistaDB db = new PistaDB(
        context.getFilesDir() + "/knowledge.pst",
        384, Metric.COSINE, IndexType.HNSW, null)) {

    // Insert
    db.insert(1L, embeddingVector, "My first document");

    // Search — returns SearchResult[] ordered by ascending distance
    SearchResult[] hits = db.search(queryVector, /* k= */ 5);
    for (SearchResult r : hits) {
        Log.d("RAG", r.label + "  d=" + r.distance);
    }

    db.save();
}
```

Fine-tune the index via the builder:

```java
PistaDBParams params = PistaDBParams.builder()
        .hnswM(32)
        .hnswEfSearch(100)
        .build();

PistaDB db = new PistaDB(path, 768, Metric.COSINE, IndexType.HNSW, params);
```

### Kotlin — DSL builder & coroutines

```kotlin
import com.pistadb.*

// DSL builder — reads like configuration, not constructor soup
val db = pistaDB(path, dim = 384) {
    metric    = Metric.COSINE
    indexType = IndexType.HNSW
    params { hnswEfSearch = 100 }
}

// All blocking operations have suspend counterparts — safe on MainScope
lifecycleScope.launch {
    db.insertAsync(id = 1L, vector = embedding, label = "doc")

    val results: List<SearchResult> = db.searchAsync(queryVec, k = 10)
    results.forEach { Log.d("RAG", "${it.label}  d=${it.distance}") }

    db.saveAsync()
}

// Convenience extensions
val ids: List<Long> = db.searchIds(queryVec, k = 5)   // just the ids
db += (42L to VectorEntry(vec, "quick insert"))        // += operator
```

---

## 🍎 iOS / macOS Integration (Swift & Objective-C)

The `ios/` directory provides an SPM-compatible package with two layers: a thin Objective-C wrapper over the C API and an idiomatic Swift API on top.

### Swift Package Manager

Add the repository (or local path) in Xcode → *File → Add Packages*, or directly in `Package.swift`:

```swift
// In your app's Package.swift
dependencies: [
    .package(path: "../PistaDB")   // local checkout
],
targets: [
    .target(name: "MyApp", dependencies: [
        .product(name: "PistaDB", package: "PistaDB")   // Swift API
        // or "PistaDBObjC" for the Objective-C layer only
    ])
]
```

### Swift

```swift
import PistaDB

// DSL builder
let db = try pistaDB(path: dbPath, dim: 384) {
    $0.metric    = .cosine
    $0.indexType = .hnsw
    $0.params    { $0.hnswEfSearch = 100 }
}
defer { db.close() }

// Insert
try db.insert(id: 1, vector: embeddingVector, label: "My document")

// Search — runs on Dispatchers.IO, UI thread never blocks
let results = try await db.search(queryVector, k: 10)
results.forEach { print("\($0.label)  d=\($0.distance)") }

// Scoped use — auto-saves and closes
try withDatabase(path: dbPath, dim: 384, metric: .cosine) { db in
    try db.insertBatch(entries)
}   // ← saved & closed here
```

High-recall and low-latency parameter presets are built in:

```swift
var params = PistaDBParams.highRecall   // M=32, efConstruction=400, efSearch=200
// or
var params = PistaDBParams.lowLatency   // M=16, efConstruction=100, efSearch=20
```

### Objective-C

```objc
#import <PistaDBObjC/PistaDBObjC.h>

NSError *error;

// Open (or create) a database
PSTDatabase *db = [PSTDatabase databaseWithPath:dbPath
                                            dim:384
                                         metric:PSTMetricCosine
                                      indexType:PSTIndexTypeHNSW
                                         params:nil
                                          error:&error];

// Insert
float vec[384] = { /* ... */ };
[db insertId:1 floatArray:vec count:384 label:@"My document" error:&error];

// k-NN search
NSArray<PSTSearchResult *> *hits =
    [db searchFloatArray:queryVec count:384 k:5 error:&error];
for (PSTSearchResult *r in hits) {
    NSLog(@"%@  d=%.4f", r.label, r.distance);
}

[db save:&error];
[db close];
```

---

## 🔷 .NET / C# Integration

The `csharp/` directory is a .NET Standard 2.0 class library that wraps the native C library via P/Invoke. It targets **.NET Standard 2.0**, so it works on .NET Core 2.0+, .NET 5/6/7/8, .NET Framework 4.6.1+, Unity, Xamarin, and MAUI.

### Project Setup

Add the `csharp/` project as a reference, or copy the three source files into your solution:

```xml
<!-- In your .csproj -->
<ItemGroup>
  <ProjectReference Include="../PistaDB/csharp/PistaDB.csproj" />
</ItemGroup>
```

Ensure `pistadb.dll` (Windows), `libpistadb.so` (Linux), or `libpistadb.dylib` (macOS) is on the library search path (e.g. next to your executable, or in a directory listed in `PATH` / `LD_LIBRARY_PATH`).

### Basic Usage

```csharp
using PistaDB;

// Open or create a database (IDisposable — use `using`)
using var db = PistaDatabase.Open("knowledge.pst", dim: 384,
                                   metric: Metric.Cosine,
                                   indexType: IndexType.HNSW);

// Insert a vector with an optional label
float[] embedding = GetEmbedding("PistaDB stores vectors locally.");
db.Insert(id: 1, vec: embedding, label: "PistaDB stores vectors locally.");

// K-nearest-neighbour search
float[] query = GetEmbedding("How does PistaDB help with RAG?");
IReadOnlyList<SearchResult> results = db.Search(query, k: 5);

foreach (var r in results)
    Console.WriteLine($"id={r.Id}  dist={r.Distance:F4}  label={r.Label}");

db.Save();   // Dispose does NOT auto-save; call Save() explicitly
```

### Custom Parameters

```csharp
var p = new PistaDBParams
{
    HnswM              = 32,
    HnswEfConstruction = 400,
    HnswEfSearch       = 100,
};

// Or use a built-in preset
var p = PistaDBParams.HighRecall;   // M=32, efConstruction=400, efSearch=200
var p = PistaDBParams.LowLatency;  // M=16, efConstruction=100, efSearch=20

using var db = PistaDatabase.Open("mydb.pst", dim: 1536,
                                   metric: Metric.Cosine,
                                   indexType: IndexType.HNSW,
                                   @params: p);
```

### Async / Await

All blocking operations have `Task`-based wrappers that run on the thread pool — safe to call from UI threads or ASP.NET Core controllers:

```csharp
await db.InsertAsync(id: 1, vec: embedding, label: "doc");

IReadOnlyList<SearchResult> results = await db.SearchAsync(query, k: 10);

await db.SaveAsync();
```

### IVF / IVF_PQ (Train First)

```csharp
using var db = PistaDatabase.Open("large.pst", dim: 1536,
                                   indexType: IndexType.IVF,
                                   @params: new PistaDBParams { IvfNList = 256, IvfNProbe = 16 });

db.Train();   // build centroids before inserting

foreach (var (id, vec, label) in corpus)
    db.Insert(id, vec, label);

db.Save();
```

### Library Version

```csharp
Console.WriteLine(PistaDatabase.Version);   // e.g. "1.0.0"
```

---

## 🌐 WebAssembly Integration

PistaDB compiles to a self-contained `.wasm` + `.js` module pair using **Emscripten + Embind**. Run the full vector database in any modern browser or Node.js — no server, no network, no native binary.

### Build

```bash
# Activate Emscripten SDK first
source /path/to/emsdk/emsdk_env.sh

# Build from the wasm/ directory
cd wasm
bash build.sh                  # Release build (default)
bash build.sh Debug            # Debug build with source maps

# Output: wasm/build/pistadb.js + wasm/build/pistadb.wasm
```

### Usage — Browser (ESM)

```javascript
import PistaDB from './pistadb.js';

const M = await PistaDB();

// Open or create a database (in-memory MEMFS by default)
const db = new M.Database('knowledge.pst', 384, M.Metric.Cosine, M.IndexType.HNSW, null);

// Insert — pass a Float32Array
const embedding = new Float32Array(384).fill(0.1);
db.insert(1, embedding, 'My first document');

// Search — returns an array of { id, distance, label }
const results = db.search(embedding, 5);
for (const r of results)
    console.log(`id=${r.id}  dist=${r.distance.toFixed(4)}  label=${r.label}`);

db.save();
db.delete();   // free C++ memory — always call when done
```

### Usage — Node.js (CommonJS)

```javascript
const PistaDB = require('./pistadb.js');

const M = await PistaDB();
const db = new M.Database('/tmp/knowledge.pst', 384, M.Metric.Cosine, M.IndexType.HNSW, null);

// In Node.js, the Emscripten NODEFS layer maps paths to the real filesystem.
db.insert(1, Float32Array.from({length: 384}, () => Math.random()), 'doc-1');
const results = db.search(new Float32Array(384).fill(0.5), 5);

db.save();
db.delete();
```

### Persistent Storage in the Browser (IDBFS)

By default, files live in the in-memory MEMFS and are lost on page reload. Mount **IndexedDB Filesystem** for persistence:

```javascript
const M = await PistaDB();

// Mount IndexedDB filesystem at /idb
M.FS.mkdir('/idb');
M.FS.mount(M.IDBFS, {}, '/idb');

// Populate from IndexedDB (true = sync IDB → MEMFS)
await new Promise((res, rej) =>
    M.FS.syncfs(true, err => err ? rej(err) : res()));

const db = new M.Database('/idb/knowledge.pst', 384, M.Metric.Cosine, M.IndexType.HNSW, null);

// ... use db ...

db.save();

// Flush MEMFS → IndexedDB (false = sync MEMFS → IDB)
await new Promise((res, rej) =>
    M.FS.syncfs(false, err => err ? rej(err) : res()));

db.delete();
```

### Custom Parameters

```javascript
// Pass a partial params object — omitted fields use library defaults
const params = {
    hnsw_m:               32,
    hnsw_ef_construction: 400,
    hnsw_ef_search:       100,
};
const db = new M.Database('my.pst', 1536, M.Metric.Cosine, M.IndexType.HNSW, params);
```

### Removing Vectors

```javascript
// 'delete' is reserved in JavaScript — use 'remove' to delete a vector by id
db.remove(42);
```

### TypeScript

TypeScript declarations are provided in `wasm/pistadb.d.ts`. Copy it alongside the generated files:

```typescript
import PistaDB, { type PistaDBModule, type SearchResult } from './pistadb.js';

const M: PistaDBModule = await PistaDB();
const db = new M.Database('data.pst', 128, M.Metric.L2, M.IndexType.HNSW, null);

const results: SearchResult[] = db.search(new Float32Array(128), 5);
db.delete();
```

### Serving the WASM File

Browsers require both files to be served from the **same HTTP origin** with the correct MIME type:

```nginx
# nginx — serve .wasm with the correct MIME type
location ~* \.wasm$ {
    add_header Content-Type application/wasm;
}
```

Or with any static file server:

```bash
npx serve wasm/build       # serves pistadb.js and pistadb.wasm
```

---

## ➕ C++ Integration

`cpp/pistadb.hpp` is a **single-file, header-only** C++17 wrapper. Drop it into your project and `#include` it — no build step, no generated code.

### CMake Setup

The cleanest way is to use the provided `cpp/CMakeLists.txt` as an interface target:

```cmake
add_subdirectory(PistaDB)        # builds pistadb shared library
add_subdirectory(PistaDB/cpp)    # registers pistadb_cpp INTERFACE target

target_link_libraries(my_app PRIVATE pistadb_cpp)
# ↑ this automatically adds pistadb.hpp and the C headers to your include path
```

Alternatively, add the paths manually and link the library yourself:

```cmake
target_include_directories(my_app PRIVATE PistaDB/cpp PistaDB/src)
target_link_libraries(my_app PRIVATE pistadb)
```

### Basic Usage

```cpp
#include "pistadb.hpp"
#include <iostream>

int main() {
    // RAII — destructor calls pistadb_close() automatically
    pistadb::Database db("knowledge.pst", 384,
                         pistadb::Metric::Cosine,
                         pistadb::IndexType::HNSW);

    // Insert a vector with an optional label
    std::vector<float> embedding(384, 0.1f);
    db.insert(1, embedding, "My first document");

    // K-nearest-neighbour search
    auto results = db.search(embedding, 5);
    for (const auto& r : results)
        std::cout << "id=" << r.id
                  << " dist=" << r.distance
                  << " label=" << r.label << '\n';

    db.save();   // destructor does NOT auto-save — call explicitly
}
```

### Custom Parameters

```cpp
// Fine-tune with designated initializers (C++20) or field assignment (C++17)
pistadb::Params p;
p.hnsw_m               = 32;
p.hnsw_ef_construction = 400;
p.hnsw_ef_search       = 100;

// Or use a built-in preset
auto p = pistadb::Params::high_recall();   // M=32, efConstruction=400, efSearch=200
auto p = pistadb::Params::low_latency();  // M=16, efConstruction=100, efSearch=20

pistadb::Database db("mydb.pst", 1536,
                     pistadb::Metric::Cosine,
                     pistadb::IndexType::HNSW, &p);
```

### Removing Vectors

```cpp
// 'delete' is a C++ keyword — the method is named 'remove'
db.remove(42);
```

### IVF / IVF_PQ (Train First)

```cpp
pistadb::Params p;
p.ivf_nlist  = 256;
p.ivf_nprobe = 16;

pistadb::Database db("large.pst", 1536,
                     pistadb::Metric::Cosine,
                     pistadb::IndexType::IVF, &p);
db.train();   // build centroids before inserting

for (auto& [id, vec, label] : corpus)
    db.insert(id, vec, label);

db.save();
```

### Exception Handling

```cpp
try {
    pistadb::Database db("my.pst", 128);
    db.insert(1, vec, "hello");
    db.save();
} catch (const pistadb::Exception& e) {
    std::cerr << "PistaDB error: " << e.what() << '\n';
}
```

### Raw Pointer Overloads

All vector-taking methods have a raw `const float*` overload to avoid copying:

```cpp
const float* raw = embedding_buffer;
db.insert(1, raw, "doc");
auto results = db.search(raw, 5);
db.update(1, raw);
```

### Library Version

```cpp
std::cout << pistadb::Database::version() << '\n';   // e.g. "1.0.0"
```

---

## 🦀 Rust Integration

The `rust/` directory is a standard Cargo crate (`pistadb`, no external dependencies) that wraps the native C library via `extern "C"` FFI.

### Cargo Setup

Add the crate to your `Cargo.toml` (local path reference):

```toml
[dependencies]
pistadb = { path = "../PistaDB/rust" }
```

Set the library search path before building:

```bash
# Windows
set PISTADB_LIB_DIR=..\PistaDB\build\Release
cargo build

# Linux / macOS
PISTADB_LIB_DIR=../PistaDB/build cargo build
```

The `build.rs` also searches `../build`, `../build/Release`, and `../build/Debug` automatically relative to the crate root.

### Basic Usage

```rust
use pistadb::{Database, Metric, IndexType};

fn main() -> Result<(), pistadb::Error> {
    // Open or create a database
    let db = Database::open("knowledge.pst", 384, Metric::Cosine, IndexType::HNSW, None)?;

    // Insert a vector with an optional label
    let embedding = vec![0.1_f32; 384];
    db.insert(1, &embedding, Some("My first document"))?;

    // K-nearest-neighbour search
    let results = db.search(&embedding, 5)?;
    for r in &results {
        println!("id={} dist={:.4} label={:?}", r.id, r.distance, r.label);
    }

    db.save()?;
    // db dropped here — pistadb_close() called automatically
    Ok(())
}
```

### Custom Parameters

```rust
use pistadb::Params;

// Fine-tune with struct update syntax
let p = Params { hnsw_m: 32, hnsw_ef_search: 100, ..Params::default() };

// Or use built-in presets
let p = Params::high_recall();   // M=32, efConstruction=400, efSearch=200
let p = Params::low_latency();   // M=16, efConstruction=100, efSearch=20

let db = Database::open("mydb.pst", 1536, Metric::Cosine, IndexType::HNSW, Some(&p))?;
```

### Multi-threaded Usage

`Database` is `Send + Sync`. Wrap in `Arc` to share across threads:

```rust
use std::sync::Arc;
use pistadb::{Database, Metric, IndexType};

let db = Arc::new(Database::open("db.pst", 128, Metric::L2, IndexType::HNSW, None)?);

let handles: Vec<_> = (0..4).map(|i| {
    let db = Arc::clone(&db);
    std::thread::spawn(move || {
        db.insert(i, &vec![i as f32; 128], Some(&format!("vec-{i}"))).unwrap();
    })
}).collect();

for h in handles { h.join().unwrap(); }
db.save()?;
```

### IVF / IVF_PQ (Train First)

```rust
use pistadb::{Database, Metric, IndexType, Params};

let p = Params { ivf_nlist: 256, ivf_nprobe: 16, ..Params::default() };
let db = Database::open("large.pst", 1536, Metric::Cosine, IndexType::IVF, Some(&p))?;

db.train()?;   // build centroids before inserting

for (id, vec) in corpus.iter().enumerate() {
    db.insert(id as u64 + 1, vec, None)?;
}
db.save()?;
```

---

## 📦 Installation

### 1. Build the C Library

**Windows (MSVC):**
```bat
build.bat Release
```

**Linux / macOS (GCC / Clang):**
```bash
bash build.sh Release
```

Produces `pistadb.dll` (Windows) or `libpistadb.so` (Linux / macOS) with **zero external dependencies**.

### 2. Install the Python Binding

```bash
pip install -e python/
```

No Rust compiler. No CMake for the Python step. No surprises.

### 3. Android Integration

Open `android/` as a library module in Android Studio, or declare it in `settings.gradle`:

```groovy
include ':android'
project(':android').projectDir = new File('<path-to-PistaDB>/android')
```

The NDK build is handled automatically by `android/CMakeLists.txt`. Ensure NDK `26.x` is installed and `ndkVersion` in `android/build.gradle` matches.

### 4. WASM Integration

```bash
source /path/to/emsdk/emsdk_env.sh
cd wasm && bash build.sh
# → wasm/build/pistadb.js + pistadb.wasm
```

Serve both files from the same HTTP origin, or use directly in Node.js.

### 5. C++ Integration

Add `cpp/` and `src/` to your include path, then link against the native library:

```cmake
add_subdirectory(PistaDB)
add_subdirectory(PistaDB/cpp)
target_link_libraries(my_app PRIVATE pistadb_cpp)
```

Or manually (GCC/Clang):

```bash
g++ -std=c++17 -IPistaDB/cpp -IPistaDB/src main.cpp -Lbuild -lpistadb -o my_app
```

### 6. Rust Integration

Set `PISTADB_LIB_DIR` to the directory containing the compiled native library, then build:

```bash
cd rust
PISTADB_LIB_DIR=../build cargo build --release
```

### 7. C# / .NET Integration

Add the `csharp/` project as a reference and ensure the native library is on the search path:

```bash
# Windows: copy pistadb.dll next to your executable
copy build\Release\pistadb.dll MyApp\bin\Debug\net8.0\

# Linux: set LD_LIBRARY_PATH or copy libpistadb.so
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
```

### 8. iOS / macOS Integration (Swift Package Manager)

In Xcode: **File → Add Package Dependencies** → point to this repository (or local checkout).
Or add to your own `Package.swift`:

```swift
.package(path: "../PistaDB")
```

The `Package.swift` at the project root declares three targets — `CPistaDB` (C core), `PistaDBObjC`, and `PistaDB` (Swift) — wired together automatically by SPM.

---

## ⚡ Quick Start

### Basic Usage

```python
import numpy as np
from pistadb import PistaDB, Metric, Index, Params

# One file, everything inside — ready to ship alongside your app
params = Params(hnsw_M=16, hnsw_ef_construction=200, hnsw_ef_search=50)
db = PistaDB("mydb.pst", dim=1536, metric=Metric.COSINE, index=Index.HNSW, params=params)

# Insert — label stores the original text chunk for easy retrieval
vec = np.random.rand(1536).astype("float32")
db.insert(1, vec, label="chunk_0001")

# Search — returns ranked results with id, distance, and label
query = np.random.rand(1536).astype("float32")
results = db.search(query, k=10)
for r in results:
    print(f"id={r.id}  dist={r.distance:.4f}  label={r.label!r}")

# Persist to disk
db.save()
db.close()

# Reload — index fully restored, ready to query in milliseconds
db2 = PistaDB("mydb.pst", dim=1536)
```

### Context Manager

```python
with PistaDB("docs.pst", dim=768, metric=Metric.COSINE) as db:
    db.insert(1, vec, label="document excerpt")
    results = db.search(query, k=5)
    db.save()
# Automatically closed on exit
```

### Batch Build from NumPy Array

```python
from pistadb import build_from_array

# Typical pattern: embed your corpus, then hand the array to PistaDB
vecs   = embed_model.encode(corpus).astype("float32")   # shape (n, dim)
labels = [chunk[:255] for chunk in corpus]

db = build_from_array("corpus.pst", vecs, labels=labels,
                       metric=Metric.COSINE, index=Index.HNSW)
db.save()
```

### IVF / IVF_PQ (Train First)

```python
# Good choice for large embedding collections (100k+ vectors)
db = PistaDB("large_kb.pst", dim=1536, index=Index.IVF,
             params=Params(ivf_nlist=256, ivf_nprobe=16))

db.train(representative_vecs)   # build cluster centroids on a sample

for i, v in enumerate(all_embeddings):
    db.insert(i + 1, v, label=doc_chunks[i])

db.save()
```

### DiskANN / Vamana Graph Index

```python
db = PistaDB("graph.pst", dim=1536, index=Index.DISKANN,
             params=Params(diskann_R=32, diskann_L=100, diskann_alpha=1.2))

for i, v in enumerate(vectors):
    db.insert(i + 1, v)

db.train()       # optional: trigger a full Vamana graph rebuild
results = db.search(query, k=10)
```

### ScaNN — Anisotropic Vector Quantization

ScaNN (Scalable Nearest Neighbors, Google Research ICML 2020) is the highest-recall index in PistaDB. It extends IVF-PQ with an **anisotropic quantization transform** that amplifies the quantization error component parallel to the original data vector direction — the component that matters most for inner-product and cosine recall.

**Two-phase search:**
1. **Phase 1** — Fast ADC scoring over PQ codes: query residuals are approximated via precomputed lookup tables, giving O(nprobe × nlist_size) scoring with no float multiplies.
2. **Phase 2** — Exact reranking: the top `rerank_k` candidates are re-scored with the raw stored float vectors, recovering near-perfect recall from the compressed candidates.

```python
from pistadb import PistaDB, Metric, Index, Params

# ScaNN requires a training pass before inserts (like IVF / IVF_PQ)
params = Params(
    scann_nlist    = 256,   # coarse IVF partitions
    scann_nprobe   = 32,    # partitions to probe at query time
    scann_pq_M     = 16,    # PQ sub-spaces (dim must be divisible by scann_pq_M)
    scann_pq_bits  = 8,     # bits per sub-code (4 or 8)
    scann_rerank_k = 200,   # candidates to exact-rerank (should be > k)
    scann_aq_eta   = 0.2,   # anisotropic penalty η (0 = standard PQ)
)

db = PistaDB("scann.pst", dim=1536, metric=Metric.COSINE,
             index=Index.SCANN, params=params)

db.train(representative_vecs)   # build centroids + PQ codebooks on a sample

for i, v in enumerate(all_embeddings):
    db.insert(i + 1, v, label=doc_chunks[i])

results = db.search(query, k=10)
db.save()
```

**When to use ScaNN:**
- You need the best recall@K on cosine / inner-product metrics (OpenAI, Cohere, sentence-transformers embeddings)
- Your dataset is large enough to benefit from IVF partitioning (50k+ vectors)
- You can afford slightly more memory than IVF-PQ (raw vectors stored alongside PQ codes for reranking)

**Key parameter guidance:**

| Parameter | Effect | Recommended starting point |
|-----------|--------|---------------------------|
| `scann_nlist` | Coarse partition count | `sqrt(n_vectors)` |
| `scann_nprobe` | Partitions searched per query | 10–15% of `nlist` |
| `scann_pq_M` | Compression ratio (`dim / pq_M` floats per sub-space) | `dim / 4` to `dim / 8` |
| `scann_rerank_k` | Rerank candidates (higher = better recall, slower) | 5–20× the query `k` |
| `scann_aq_eta` | Anisotropic penalty (0 = standard PQ) | `0.2` for cosine/IP; `0.0` for L2 |

---

## 🧪 Running Tests

```bash
# Windows
set PISTADB_LIB_DIR=build\Release
pytest tests\ -v

# Linux / macOS
PISTADB_LIB_DIR=build pytest tests/ -v
```

**48 / 48 tests passing** — recall benchmarks, roundtrip persistence, corrupt-file rejection, metric correctness, and ScaNN two-phase search.

---

## 📊 Running Examples

```bash
# Windows
set PISTADB_LIB_DIR=build\Release
python examples/example.py

# Linux / macOS
PISTADB_LIB_DIR=build python examples/example.py
```

The example script walks through all 9 scenarios: per-metric demos, all 7 index types (including ScaNN), batch build, DiskANN rebuild, persistence roundtrip, and delete/update operations.

---

## 🗂️ File Format

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

Every index type (HNSW, IVF, DiskANN, LSH, ScaNN, …) serializes itself into this block — graph edges, cluster centroids, PQ codebooks, projection matrices, and raw float vectors all in one place. The file is fully self-contained: no sidecar files, no registry entries, no environment variables needed at load time.

---

## 🏗️ Project Structure

```
PistaDB/
├── src/                          # C99 core — all platforms share this
│   ├── pistadb_types.h           # Shared types, error codes, default params
│   ├── pistadb.h / .c            # Primary database API
│   ├── distance.h / .c           # Runtime dispatch + scalar fallbacks for 5 metrics
│   ├── distance_simd.h           # Internal declarations for AVX2 / NEON kernels
│   ├── distance_avx2.c           # AVX2+FMA kernels (compiled with -mavx2 -mfma)
│   ├── distance_neon.c           # ARM NEON kernels (AArch64 built-in)
│   ├── pistadb_batch.h / .c      # Multi-threaded batch insert (thread pool + ring queue)
│   ├── pistadb_cache.h / .c      # Embedding cache (FNV-1a hash map + LRU list + .pcc file)
│   ├── utils.h / .c              # Binary heap, PCG32 RNG, bitset
│   ├── storage.h / .c            # File I/O, header CRC32
│   ├── index_linear.*            # Exact brute-force scan
│   ├── index_hnsw.*              # HNSW (Malkov & Yashunin, 2018)
│   ├── index_ivf.*               # IVF with k-means clustering
│   ├── index_ivf_pq.*            # IVF + Product Quantization
│   ├── index_diskann.*           # Vamana / DiskANN (Subramanya et al., 2019)
│   ├── index_lsh.*               # E2LSH + sign-based LSH
│   └── index_scann.*             # ScaNN: Anisotropic Vector Quantization (Guo et al., ICML 2020)
├── python/                       # Python binding
│   ├── pistadb/__init__.py       # Pure-ctypes wrapper (no Cython, no cffi)
│   └── setup.py
├── android/                      # Android integration layer
│   ├── CMakeLists.txt            # NDK build (compiles C core + JNI bridge)
│   ├── build.gradle              # Android library module (minSdk 21)
│   ├── proguard-rules.pro        # Keeps JNI-reflected fields
│   └── src/main/
│       ├── cpp/pistadb_jni.c     # JNI bridge (15 native methods)
│       ├── java/com/pistadb/     # Java API
│       │   ├── PistaDB.java      # Main class (AutoCloseable, synchronized)
│       │   ├── Metric.java       # Distance metric enum
│       │   ├── IndexType.java    # Index algorithm enum
│       │   ├── PistaDBParams.java# All 19 tuning params + fluent Builder
│       │   ├── SearchResult.java # id + distance + label
│       │   ├── VectorEntry.java  # float[] vector + label
│       │   └── PistaDBException.java
│       └── kotlin/com/pistadb/  # Kotlin extensions
│           ├── PistaDBExtensions.kt  # DSL builder, operator overloads
│           └── PistaDBCoroutines.kt  # suspend wrappers (Dispatchers.IO)
├── ios/                          # iOS / macOS integration layer
│   └── Sources/
│       ├── PistaDBObjC/          # Objective-C wrapper
│       │   ├── include/          # Public headers (PST prefix)
│       │   │   ├── PistaDBObjC.h # Umbrella header
│       │   │   ├── PSTDatabase.h # Main class
│       │   │   ├── PSTMetric.h   # Distance metric enum
│       │   │   ├── PSTIndexType.h# Index algorithm enum
│       │   │   ├── PSTParams.h   # Tuning parameters (NSCopying)
│       │   │   ├── PSTSearchResult.h
│       │   │   ├── PSTVectorEntry.h
│       │   │   └── PSTError.h    # Error domain + NS_ENUM codes
│       │   └── PSTDatabase.m     # Implementation (NSLock thread safety)
│       └── PistaDB/              # Swift API
│           ├── PistaDB.swift     # Main class (throws, Closeable)
│           ├── PistaDBTypes.swift# Metric/IndexType/PistaDBParams/PistaDBError
│           ├── PistaDBAsync.swift# async/await extensions (iOS 13+)
│           └── PistaDBExtensions.swift # pistaDB() DSL, subscript, presets
├── wasm/                         # WebAssembly binding (Emscripten + Embind)
│   ├── pistadb_wasm.cpp          # Embind binding — Database class + enum registration
│   ├── CMakeLists.txt            # emcmake build (compiles C core + binding → .js + .wasm)
│   ├── build.sh                  # Build script (requires emsdk active in shell)
│   └── pistadb.d.ts              # TypeScript declarations
├── cpp/                          # C++ binding (header-only, C++17)
│   ├── pistadb.hpp               # Single-file wrapper — the only file you need
│   └── CMakeLists.txt            # INTERFACE library target for CMake projects
├── rust/                         # Rust binding (Cargo crate, no external deps)
│   ├── Cargo.toml                # Package definition (edition 2021)
│   ├── build.rs                  # Links native library, respects PISTADB_LIB_DIR
│   └── src/
│       ├── lib.rs                # Public API: Database struct, re-exports, doc tests
│       ├── ffi.rs                # Raw extern "C" declarations + repr(C) structs
│       └── types.rs              # Metric/IndexType enums, SearchResult, VectorEntry, Params, Error
├── csharp/                       # C# / .NET binding (.NET Standard 2.0)
│   ├── PistaDB.csproj            # Project file (netstandard2.0)
│   ├── NativeMethods.cs          # P/Invoke declarations + native structs
│   ├── PistaDBTypes.cs           # Metric/IndexType enums, SearchResult, PistaDBParams, exception
│   └── PistaDatabase.cs          # Main class (IDisposable, thread-safe, async wrappers)
├── tests/
│   └── test_pistadb.py           # 48-test pytest suite
├── examples/
│   └── example.py                # 8 end-to-end usage scenarios
├── Package.swift                 # SPM manifest (CPistaDB → PistaDBObjC → PistaDB)
├── CMakeLists.txt
├── build.sh                      # Linux / macOS build script
└── build.bat                     # Windows MSVC build script
```

---

## 🗺️ Roadmap

**Core engine**
- [x] SIMD-accelerated distance kernels (AVX2 / NEON) for faster embedding comparisons
- [ ] Filtered search with metadata predicates (filter by source, date, tag before ANN)
- [x] Multi-threaded batch insert for high-throughput embedding pipelines

**LLM & RAG ecosystem**
- [ ] LangChain and LlamaIndex integration (drop-in vectorstore)
- [ ] First-class support for OpenAI, Cohere, and sentence-transformers embedding dimensions
- [ ] Hybrid search: dense vector + sparse BM25 re-ranking in a single query
- [x] Embedding cache layer — deduplicate identical inputs automatically

**Portability**
- [x] Android bindings — Java + Kotlin via JNI (`android/`)
- [x] iOS / macOS bindings — Objective-C + Swift via SPM (`ios/`, `Package.swift`)
- [x] C# / .NET binding — P/Invoke, `IDisposable`, async/await (`csharp/`)
- [x] Rust binding — FFI, `Send + Sync`, `Drop`, `Result<T, Error>` (`rust/`)
- [x] C++ binding — header-only C++17, RAII, move-only, `std::mutex` (`cpp/pistadb.hpp`)
- [x] WASM build — Emscripten + Embind, ESM module, `Float32Array`, TypeScript types (`wasm/`)
- [ ] WASM build — run full RAG pipelines in the browser
- [ ] HTTP microserver mode (optional, single binary, for multi-process access)

---

## 🤝 Contributing

Pull requests are warmly welcomed. Whether it's a new index algorithm, a language binding, a performance improvement, an LLM integration, or documentation — every contribution makes PistaDB better for the whole community.

1. Fork the repository
2. Create your feature branch (`git checkout -b feat/langchain-integration`)
3. Commit your changes
4. Open a Pull Request

Please ensure all 48 tests continue to pass before submitting.

---

<div align="center">
<strong>Built in C99 · C++ · WASM · Python · Java · Kotlin · Swift · Objective-C · C# · Rust · Runs anywhere · Keeps your data private</strong><br>
<em>The best infrastructure for an LLM app is the kind you never have to think about.</em>
</div>
