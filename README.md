<div align="center">

[English](./README.md) · [中文](./README_CN.md)

<h1>🌰 PistaDB</h1>

<p><strong>The embedded vector database for LLM-native applications.</strong><br>
RAG-ready · Zero dependencies · Single-file storage · MIT Licensed</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C99](https://img.shields.io/badge/C-99-blue.svg)](https://en.wikipedia.org/wiki/C99)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776ab.svg)]()
[![Go](https://img.shields.io/badge/Go-1.21%2B-00add8.svg)]()
[![Android](https://img.shields.io/badge/Android-API%2021%2B-3ddc84.svg)]()
[![iOS](https://img.shields.io/badge/iOS-13%2B-lightgrey.svg)]()
[![.NET](https://img.shields.io/badge/.NET-Standard%202.0%2B-512bd4.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20Android%20%7C%20iOS-lightgrey.svg)]()
[![Tests](https://img.shields.io/badge/tests-109%2F109%20passing-brightgreen.svg)]()

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

## Why PistaDB?

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

## Key Features

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

### Production Feature Set

- **SIMD-accelerated** distance kernels — AVX2+FMA on x86-64, NEON on ARM, runtime-dispatched (4–8× scalar)
- **VecStore chunked storage** — no scale ceiling; verified at 10 M vectors (HNSW) and 9 M full CRUD (IVF)
- **Transactions** — ACID-style atomic multi-op groups with full undo-on-failure rollback
- **Multi-threaded batch insert** — thread-pool + ring-buffer API for high-throughput embedding pipelines
- **Embedding cache** — persistent LRU cache (`.pcc`) that eliminates redundant model calls
- **Single-file storage** — CRC32-verified `.pst` format; atomic save, no partial writes
- **9 language bindings** — C, C++, Python, Go, Java, Kotlin, Swift, Objective-C, C#, Rust, WASM
- **109 / 109 tests passing** across all features and platforms

---

## Language & Platform Support

| Language | Binding mechanism | Where to find it |
|---|---|---|
| **C / C++** | Direct `#include` | `src/pistadb.h` / `cpp/pistadb.hpp` |
| **Python** | `ctypes` (no Cython) | `python/` |
| **Go** | CGO | `go/` |
| **Java** | JNI | `android/src/main/java/` |
| **Kotlin** | JNI + extension functions | `android/src/main/kotlin/` |
| **Objective-C** | Direct C interop | `ios/Sources/PistaDBObjC/` |
| **Swift** | ObjC bridge | `ios/Sources/PistaDB/` |
| **C#** | P/Invoke | `csharp/` |
| **Rust** | FFI (`extern "C"`) | `rust/` |
| **WASM** | Emscripten / Embind | `wasm/` |

| Platform | Library output | ABI targets |
|---|---|---|
| **Windows** | `pistadb.dll` | x86_64 |
| **Linux** | `libpistadb.so` | x86_64, aarch64 |
| **macOS** | `libpistadb.dylib` | x86_64, arm64 |
| **Android** | `libpistadb_jni.so` | arm64-v8a, armeabi-v7a, x86_64, x86 |
| **iOS / macOS** | Static library (SPM) | arm64, arm64-Simulator, x86_64-Simulator |
| **WASM** | `.wasm` | — *(planned)* |

---

## Installation

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

### 4. iOS / macOS Integration (Swift Package Manager)

In Xcode: **File → Add Package Dependencies** → point to this repository (or local checkout).
Or add to your own `Package.swift`:

```swift
.package(path: "../PistaDB")
```

The `Package.swift` at the project root declares three targets — `CPistaDB` (C core), `PistaDBObjC`, and `PistaDB` (Swift) — wired together automatically by SPM.

### 5. WASM Integration

```bash
source /path/to/emsdk/emsdk_env.sh
cd wasm && bash build.sh
# → wasm/build/pistadb.js + pistadb.wasm
```

Serve both files from the same HTTP origin, or use directly in Node.js.

### 6. C++ Integration

```cmake
add_subdirectory(PistaDB)
add_subdirectory(PistaDB/cpp)
target_link_libraries(my_app PRIVATE pistadb_cpp)
```

### 7. Go Integration

```go
// go.mod
replace pistadb.io/go => ../PistaDB/go
```

```bash
export CGO_LDFLAGS="-L../PistaDB/build -lpistadb"
go get pistadb.io/go/pistadb
go build ./...
```

### 8. Rust Integration

```bash
cd rust
PISTADB_LIB_DIR=../build cargo build --release
```

### 9. C# / .NET Integration

```xml
<!-- In your .csproj -->
<ItemGroup>
  <ProjectReference Include="../PistaDB/csharp/PistaDB.csproj" />
</ItemGroup>
```

```bash
# Windows: copy pistadb.dll next to your executable
copy build\Release\pistadb.dll MyApp\bin\Debug\net8.0\

# Linux: set LD_LIBRARY_PATH or copy libpistadb.so
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
```

---

## Quick Start

```python
import numpy as np
from pistadb import PistaDB, Metric, Index, Params

params = Params(hnsw_M=16, hnsw_ef_construction=200, hnsw_ef_search=50)
db = PistaDB("mydb.pst", dim=1536, metric=Metric.COSINE, index=Index.HNSW, params=params)

vec = np.random.rand(1536).astype("float32")
db.insert(1, vec, label="chunk_0001")

query = np.random.rand(1536).astype("float32")
results = db.search(query, k=10)
for r in results:
    print(f"id={r.id}  dist={r.distance:.4f}  label={r.label!r}")

db.save()
db.close()
```

```python
# Context manager — auto-closed on exit
with PistaDB("docs.pst", dim=768, metric=Metric.COSINE) as db:
    db.insert(1, vec, label="document excerpt")
    results = db.search(query, k=5)
    db.save()
```

For more examples — RAG pipelines, agent memory, advanced indexes, transactions, batch insert, embedding cache, and per-language integration guides — see the docs below.

---

## Running Tests

```bash
# Windows
set PISTADB_LIB_DIR=build\Release
pytest tests\ -v

# Linux / macOS
PISTADB_LIB_DIR=build pytest tests/ -v
```

**109 / 109 tests passing** — recall benchmarks, roundtrip persistence, corrupt-file rejection, metric correctness, ScaNN two-phase search, and transaction atomicity / rollback.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/examples.md](docs/examples.md) | RAG pipelines, agent memory, all index types, transactions, batch insert, embedding cache |
| [docs/language-bindings.md](docs/language-bindings.md) | Android, iOS/macOS, .NET, WASM, C++, Rust, Go — full integration guides |
| [docs/benchmarks.md](docs/benchmarks.md) | Large-scale CRUD benchmarks, SIMD details, file format, project structure |

---

## Roadmap

- [ ] Filtered search with metadata predicates (filter by source, date, tag before ANN)
- [ ] LangChain and LlamaIndex integration (drop-in vectorstore)
- [ ] Hybrid search: dense vector + sparse BM25 re-ranking in a single query
- [ ] Full in-browser RAG pipeline via WASM (IDBFS persistence, SharedArrayBuffer workers)
- [ ] HTTP microserver mode (optional, single binary, for multi-process access)

---

## Contributing

Pull requests are warmly welcomed. Whether it's a new index algorithm, a language binding, a performance improvement, an LLM integration, or documentation — every contribution makes PistaDB better for the whole community.

1. Fork the repository
2. Create your feature branch (`git checkout -b feat/langchain-integration`)
3. Commit your changes
4. Open a Pull Request

Please ensure all 109 tests continue to pass before submitting.

---

<div align="center">
<strong>Built in C99 · C++ · WASM · Python · Go · Java · Kotlin · Swift · Objective-C · C# · Rust · Runs anywhere · Keeps your data private</strong><br>
<em>The best infrastructure for an LLM app is the kind you never have to think about.</em>
</div>
