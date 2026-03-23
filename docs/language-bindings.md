# Language Bindings

PistaDB's core is written in pure C99. Every binding wraps this same library through a thin, zero-copy layer — no reimplementation, no behavioural divergence. The same `.pst` file is portable across all platforms.

---

## Android (Java & Kotlin)

The `android/` module is a self-contained Android library. Add it to your project once and get the full vector-database API in both Java and Kotlin, backed by the native C library through JNI.

### Gradle Setup

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

### Kotlin — DSL Builder & Coroutines

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

## iOS / macOS (Swift & Objective-C)

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

## .NET / C#

The `csharp/` directory is a .NET Standard 2.0 class library wrapping the native C library via P/Invoke. Supports .NET Core 2.0+, .NET 5/6/7/8, .NET Framework 4.6.1+, Unity, Xamarin, and MAUI.

### Project Setup

```xml
<!-- In your .csproj -->
<ItemGroup>
  <ProjectReference Include="../PistaDB/csharp/PistaDB.csproj" />
</ItemGroup>
```

Ensure `pistadb.dll` (Windows), `libpistadb.so` (Linux), or `libpistadb.dylib` (macOS) is on the library search path.

### Basic Usage

```csharp
using PistaDB;

using var db = PistaDatabase.Open("knowledge.pst", dim: 384,
                                   metric: Metric.Cosine,
                                   indexType: IndexType.HNSW);

float[] embedding = GetEmbedding("PistaDB stores vectors locally.");
db.Insert(id: 1, vec: embedding, label: "PistaDB stores vectors locally.");

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

---

## WebAssembly

PistaDB compiles to a self-contained `.wasm` + `.js` module pair using **Emscripten + Embind**. Run the full vector database in any modern browser or Node.js — no server, no network, no native binary.

### Build

```bash
# Activate Emscripten SDK first
source /path/to/emsdk/emsdk_env.sh

cd wasm
bash build.sh                  # Release build (default)
bash build.sh Debug            # Debug build with source maps

# Output: wasm/build/pistadb.js + wasm/build/pistadb.wasm
```

### Browser (ESM)

```javascript
import PistaDB from './pistadb.js';

const M = await PistaDB();
const db = new M.Database('knowledge.pst', 384, M.Metric.Cosine, M.IndexType.HNSW, null);

const embedding = new Float32Array(384).fill(0.1);
db.insert(1, embedding, 'My first document');

const results = db.search(embedding, 5);
for (const r of results)
    console.log(`id=${r.id}  dist=${r.distance.toFixed(4)}  label=${r.label}`);

db.save();
db.delete();   // free C++ memory — always call when done
```

### Node.js (CommonJS)

```javascript
const PistaDB = require('./pistadb.js');

const M = await PistaDB();
const db = new M.Database('/tmp/knowledge.pst', 384, M.Metric.Cosine, M.IndexType.HNSW, null);

db.insert(1, Float32Array.from({length: 384}, () => Math.random()), 'doc-1');
const results = db.search(new Float32Array(384).fill(0.5), 5);

db.save();
db.delete();
```

### Persistent Storage (IDBFS)

By default, files live in the in-memory MEMFS and are lost on page reload. Mount **IndexedDB Filesystem** for persistence:

```javascript
const M = await PistaDB();

M.FS.mkdir('/idb');
M.FS.mount(M.IDBFS, {}, '/idb');

// Populate from IndexedDB (true = sync IDB → MEMFS)
await new Promise((res, rej) =>
    M.FS.syncfs(true, err => err ? rej(err) : res()));

const db = new M.Database('/idb/knowledge.pst', 384, M.Metric.Cosine, M.IndexType.HNSW, null);

// ... use db ...

db.save();

// Flush MEMFS → IndexedDB
await new Promise((res, rej) =>
    M.FS.syncfs(false, err => err ? rej(err) : res()));

db.delete();
```

### TypeScript

```typescript
import PistaDB, { type PistaDBModule, type SearchResult } from './pistadb.js';

const M: PistaDBModule = await PistaDB();
const db = new M.Database('data.pst', 128, M.Metric.L2, M.IndexType.HNSW, null);

const results: SearchResult[] = db.search(new Float32Array(128), 5);
db.delete();
```

TypeScript declarations are provided in `wasm/pistadb.d.ts`.

### Serving the WASM File

```nginx
# nginx — serve .wasm with the correct MIME type
location ~* \.wasm$ {
    add_header Content-Type application/wasm;
}
```

---

## C++

`cpp/pistadb.hpp` is a **single-file, header-only** C++17 wrapper. Drop it into your project and `#include` it — no build step, no generated code.

### CMake Setup

```cmake
add_subdirectory(PistaDB)        # builds pistadb shared library
add_subdirectory(PistaDB/cpp)    # registers pistadb_cpp INTERFACE target

target_link_libraries(my_app PRIVATE pistadb_cpp)
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

    std::vector<float> embedding(384, 0.1f);
    db.insert(1, embedding, "My first document");

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

### Raw Pointer Overloads

```cpp
const float* raw = embedding_buffer;
db.insert(1, raw, "doc");
auto results = db.search(raw, 5);
db.update(1, raw);
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

---

## Rust

The `rust/` directory is a standard Cargo crate (`pistadb`, no external dependencies) that wraps the native C library via `extern "C"` FFI.

### Cargo Setup

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

### Basic Usage

```rust
use pistadb::{Database, Metric, IndexType};

fn main() -> Result<(), pistadb::Error> {
    let db = Database::open("knowledge.pst", 384, Metric::Cosine, IndexType::HNSW, None)?;

    let embedding = vec![0.1_f32; 384];
    db.insert(1, &embedding, Some("My first document"))?;

    let results = db.search(&embedding, 5)?;
    for r in &results {
        println!("id={} dist={:.4} label={:?}", r.id, r.distance, r.label);
    }

    db.save()?;
    Ok(())
}
```

### Custom Parameters

```rust
use pistadb::Params;

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

## Go

The `go/` directory is a standard Go module (`pistadb.io/go`) that wraps the native C library via **CGO**. It provides idiomatic Go types for all three APIs: database, batch insert, and embedding cache.

### Module Setup

```go
// go.mod
module myapp

go 1.21

require pistadb.io/go v0.0.0

replace pistadb.io/go => ../PistaDB/go
```

Build the C library first, then build normally:

```bash
# Build C library (from PistaDB root)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel

# Build your Go app
go build ./...
```

### Basic Usage

```go
import "pistadb.io/go/pistadb"

db, err := pistadb.Open("knowledge.pst", 384,
    pistadb.MetricCosine, pistadb.IndexHNSW, nil)
if err != nil {
    log.Fatal(err)
}
defer db.Close()

if err := db.Insert(1, "My first document", embedding); err != nil {
    log.Fatal(err)
}

results, err := db.Search(queryVec, 10)
if err != nil {
    log.Fatal(err)
}
for _, r := range results {
    fmt.Printf("id=%d  dist=%.4f  label=%q\n", r.ID, r.Distance, r.Label)
}

db.Save()
```

### Custom Parameters

```go
p := pistadb.DefaultParams()
p.HNSWM = 32
p.HNSWEfSearch = 100

db, err := pistadb.Open("mydb.pst", 1536,
    pistadb.MetricCosine, pistadb.IndexHNSW, &p)
```

### IVF / IVF_PQ (Train First)

```go
p := pistadb.DefaultParams()
p.IVFNList  = 256
p.IVFNProbe = 16

db, _ := pistadb.Open("large.pst", 1536,
    pistadb.MetricCosine, pistadb.IndexIVF, &p)

db.Train()   // build centroids before inserting

for i, vec := range corpus {
    db.Insert(uint64(i+1), "", vec)
}
db.Save()
```

### Batch Insert

```go
// Streaming API — multiple goroutines push concurrently
batch, _ := pistadb.NewBatch(db, 0, 0)   // 0 threads = auto, 0 cap = default 4096
defer batch.Destroy()

batch.Push(id, label, vec)

errors := batch.Flush()   // wait for all items, reset per-flush counter

// Offline bulk API — blocking convenience wrapper
failures, err := pistadb.BatchInsert(db, ids, labels, vecs, 0)
```

### Embedding Cache

```go
cache, _ := pistadb.OpenCache("embed.pcc", 384, 100_000)
defer cache.Close()

vec, ok := cache.Get(text)
if !ok {
    vec = myModel.Encode(text)   // only called on a miss
    cache.Put(text, vec)
}

cache.Save()

stats := cache.Stats()
fmt.Printf("hits=%d  misses=%d  evictions=%d\n",
    stats.Hits, stats.Misses, stats.Evictions)
```

### Running Tests

```bash
cd go
go test ./pistadb/ -v
```
