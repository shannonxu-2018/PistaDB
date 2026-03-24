# 语言绑定集成指南

PistaDB 的核心由纯 C99 编写。所有语言绑定都通过薄薄的零拷贝层封装同一个 C 库——无重复实现，无行为差异。同一个 `.pst` 数据库文件可在所有支持的平台之间无缝移植。

---

## Android（Java & Kotlin）

`android/` 是一个独立的 Android Library 模块。将其加入项目后，即可在 Java 和 Kotlin 中使用完整的向量数据库 API，底层通过 JNI 调用原生 C 库。

### Gradle 接入

```groovy
// settings.gradle
include ':android'
project(':android').projectDir = new File('../PistaDB/wrap/android')

// app/build.gradle
dependencies {
    implementation project(':android')
}
```

### Java

```java
import com.pistadb.*;

// try-with-resources 自动完成 save + close
try (PistaDB db = new PistaDB(
        context.getFilesDir() + "/knowledge.pst",
        384, Metric.COSINE, IndexType.HNSW, null)) {

    // 插入向量
    db.insert(1L, embeddingVector, "我的第一篇文档");

    // 搜索——返回按距离升序排列的 SearchResult[]
    SearchResult[] hits = db.search(queryVector, /* k= */ 5);
    for (SearchResult r : hits) {
        Log.d("RAG", r.label + "  d=" + r.distance);
    }

    db.save();
}
```

通过 Builder 精细调整索引参数：

```java
PistaDBParams params = PistaDBParams.builder()
        .hnswM(32)
        .hnswEfSearch(100)
        .build();

PistaDB db = new PistaDB(path, 768, Metric.COSINE, IndexType.HNSW, params);
```

### Kotlin — DSL 构建器与协程

```kotlin
import com.pistadb.*

// DSL 构建器——配置式语法，告别构造函数嵌套
val db = pistaDB(path, dim = 384) {
    metric    = Metric.COSINE
    indexType = IndexType.HNSW
    params { hnswEfSearch = 100 }
}

// 所有阻塞操作均有 suspend 版本——可在 MainScope 安全调用
lifecycleScope.launch {
    db.insertAsync(id = 1L, vector = embedding, label = "doc")

    val results: List<SearchResult> = db.searchAsync(queryVec, k = 10)
    results.forEach { Log.d("RAG", "${it.label}  d=${it.distance}") }

    db.saveAsync()
}

// 便捷扩展
val ids: List<Long> = db.searchIds(queryVec, k = 5)   // 仅返回 id 列表
db += (42L to VectorEntry(vec, "快速插入"))             // += 运算符
```

---

## iOS / macOS（Swift & Objective-C）

`ios/` 目录提供一个兼容 SPM 的包，分为两层：基于 C API 的 Objective-C 薄封装，以及其上的惯用 Swift API。

### Swift Package Manager

在 Xcode 中选择 *File → Add Packages*，指向本仓库或本地路径；或直接修改 `Package.swift`：

```swift
// 在你的 Package.swift 中
dependencies: [
    .package(path: "../PistaDB")   // 本地路径
],
targets: [
    .target(name: "MyApp", dependencies: [
        .product(name: "PistaDB", package: "PistaDB")   // Swift API
        // 或 "PistaDBObjC" 仅使用 Objective-C 层
    ])
]
```

### Swift

```swift
import PistaDB

// DSL 构建器
let db = try pistaDB(path: dbPath, dim: 384) {
    $0.metric    = .cosine
    $0.indexType = .hnsw
    $0.params    { $0.hnswEfSearch = 100 }
}
defer { db.close() }

// 插入向量
try db.insert(id: 1, vector: embeddingVector, label: "我的文档")

// 搜索——在 Dispatchers.IO 上运行，UI 线程不阻塞
let results = try await db.search(queryVector, k: 10)
results.forEach { print("\($0.label)  d=\($0.distance)") }

// 作用域用法——自动保存并关闭
try withDatabase(path: dbPath, dim: 384, metric: .cosine) { db in
    try db.insertBatch(entries)
}   // ← 此处已保存并关闭
```

内置高召回率和低延迟参数预设：

```swift
var params = PistaDBParams.highRecall   // M=32, efConstruction=400, efSearch=200
// 或
var params = PistaDBParams.lowLatency   // M=16, efConstruction=100, efSearch=20
```

### Objective-C

```objc
#import <PistaDBObjC/PistaDBObjC.h>

NSError *error;

// 打开（或创建）数据库
PSTDatabase *db = [PSTDatabase databaseWithPath:dbPath
                                            dim:384
                                         metric:PSTMetricCosine
                                      indexType:PSTIndexTypeHNSW
                                         params:nil
                                          error:&error];

// 插入向量
float vec[384] = { /* ... */ };
[db insertId:1 floatArray:vec count:384 label:@"我的文档" error:&error];

// k-NN 搜索
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

`csharp/` 目录是一个 .NET Standard 2.0 类库，通过 P/Invoke 封装原生 C 库。支持 .NET Core 2.0+、.NET 5/6/7/8、.NET Framework 4.6.1+、Unity、Xamarin 和 MAUI。

### 项目接入

```xml
<!-- 在你的 .csproj 中 -->
<ItemGroup>
  <ProjectReference Include="../PistaDB/csharp/PistaDB.csproj" />
</ItemGroup>
```

确保 `pistadb.dll`（Windows）、`libpistadb.so`（Linux）或 `libpistadb.dylib`（macOS）在库搜索路径中。

### 基础用法

```csharp
using PistaDB;

using var db = PistaDatabase.Open("knowledge.pst", dim: 384,
                                   metric: Metric.Cosine,
                                   indexType: IndexType.HNSW);

float[] embedding = GetEmbedding("PistaDB 在本地存储向量。");
db.Insert(id: 1, vec: embedding, label: "PistaDB 在本地存储向量。");

float[] query = GetEmbedding("PistaDB 如何助力 RAG？");
IReadOnlyList<SearchResult> results = db.Search(query, k: 5);

foreach (var r in results)
    Console.WriteLine($"id={r.Id}  dist={r.Distance:F4}  label={r.Label}");

db.Save();   // Dispose 不会自动保存，需显式调用 Save()
```

### 自定义参数

```csharp
var p = new PistaDBParams
{
    HnswM              = 32,
    HnswEfConstruction = 400,
    HnswEfSearch       = 100,
};

// 或使用内置预设
var p = PistaDBParams.HighRecall;   // M=32, efConstruction=400, efSearch=200
var p = PistaDBParams.LowLatency;  // M=16, efConstruction=100, efSearch=20

using var db = PistaDatabase.Open("mydb.pst", dim: 1536,
                                   metric: Metric.Cosine,
                                   indexType: IndexType.HNSW,
                                   @params: p);
```

### 异步 / Await

所有阻塞操作均有基于 `Task` 的包装版本，在线程池上运行——可从 UI 线程或 ASP.NET Core 控制器安全调用：

```csharp
await db.InsertAsync(id: 1, vec: embedding, label: "doc");
IReadOnlyList<SearchResult> results = await db.SearchAsync(query, k: 10);
await db.SaveAsync();
```

### IVF / IVF_PQ（需先训练）

```csharp
using var db = PistaDatabase.Open("large.pst", dim: 1536,
                                   indexType: IndexType.IVF,
                                   @params: new PistaDBParams { IvfNList = 256, IvfNProbe = 16 });

db.Train();   // 插入前先构建聚类质心

foreach (var (id, vec, label) in corpus)
    db.Insert(id, vec, label);

db.Save();
```

---

## WebAssembly

PistaDB 可编译为 `.wasm` + `.js` 模块对（使用 Emscripten + Embind），在任何现代浏览器或 Node.js 中运行完整向量数据库——无需服务器、无需网络、无需原生二进制文件。

### 构建

```bash
# 先激活 Emscripten SDK
source /path/to/emsdk/emsdk_env.sh

cd wasm
bash build.sh                  # Release 构建（默认）
bash build.sh Debug            # Debug 构建（含 source maps）

# 产物：wasm/build/pistadb.js + wasm/build/pistadb.wasm
```

### 浏览器（ESM）

```javascript
import PistaDB from './pistadb.js';

const M = await PistaDB();
const db = new M.Database('knowledge.pst', 384, M.Metric.Cosine, M.IndexType.HNSW, null);

const embedding = new Float32Array(384).fill(0.1);
db.insert(1, embedding, '我的第一篇文档');

const results = db.search(embedding, 5);
for (const r of results)
    console.log(`id=${r.id}  dist=${r.distance.toFixed(4)}  label=${r.label}`);

db.save();
db.delete();   // 释放 C++ 内存——使用完毕后必须调用
```

### Node.js（CommonJS）

```javascript
const PistaDB = require('./pistadb.js');

const M = await PistaDB();
const db = new M.Database('/tmp/knowledge.pst', 384, M.Metric.Cosine, M.IndexType.HNSW, null);

db.insert(1, Float32Array.from({length: 384}, () => Math.random()), 'doc-1');
const results = db.search(new Float32Array(384).fill(0.5), 5);

db.save();
db.delete();
```

### 浏览器持久化（IDBFS）

默认情况下，文件存储在内存 MEMFS 中，页面刷新后丢失。挂载 **IndexedDB 文件系统**实现持久化：

```javascript
const M = await PistaDB();

M.FS.mkdir('/idb');
M.FS.mount(M.IDBFS, {}, '/idb');

// 从 IndexedDB 同步到 MEMFS（true = IDB → MEMFS）
await new Promise((res, rej) =>
    M.FS.syncfs(true, err => err ? rej(err) : res()));

const db = new M.Database('/idb/knowledge.pst', 384, M.Metric.Cosine, M.IndexType.HNSW, null);

// ... 使用 db ...

db.save();

// 将 MEMFS 同步回 IndexedDB（false = MEMFS → IDB）
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

TypeScript 声明文件位于 `wasm/pistadb.d.ts`。

### 服务 WASM 文件

浏览器要求两个文件必须从**同一 HTTP 源**以正确的 MIME 类型提供：

```nginx
# nginx — 为 .wasm 设置正确的 MIME 类型
location ~* \.wasm$ {
    add_header Content-Type application/wasm;
}
```

---

## C++

`cpp/pistadb.hpp` 是一个**单文件纯头文件** C++17 封装。将其加入项目并 `#include` 即可——无需构建步骤，无需代码生成。

### CMake 接入

```cmake
add_subdirectory(PistaDB)        # 构建 pistadb 共享库
add_subdirectory(PistaDB/wrap/cpp)    # 注册 pistadb_cpp INTERFACE 目标

target_link_libraries(my_app PRIVATE pistadb_cpp)
# ↑ 自动将 pistadb.hpp 和 C 头文件加入 include 路径
```

### 基础用法

```cpp
#include "pistadb.hpp"
#include <iostream>

int main() {
    // RAII — 析构函数自动调用 pistadb_close()
    pistadb::Database db("knowledge.pst", 384,
                         pistadb::Metric::Cosine,
                         pistadb::IndexType::HNSW);

    std::vector<float> embedding(384, 0.1f);
    db.insert(1, embedding, "我的第一篇文档");

    auto results = db.search(embedding, 5);
    for (const auto& r : results)
        std::cout << "id=" << r.id
                  << " dist=" << r.distance
                  << " label=" << r.label << '\n';

    db.save();   // 析构函数不会自动保存——需显式调用
}
```

### 自定义参数

```cpp
pistadb::Params p;
p.hnsw_m               = 32;
p.hnsw_ef_construction = 400;
p.hnsw_ef_search       = 100;

// 或使用内置预设
auto p = pistadb::Params::high_recall();   // M=32, efConstruction=400, efSearch=200
auto p = pistadb::Params::low_latency();  // M=16, efConstruction=100, efSearch=20

pistadb::Database db("mydb.pst", 1536,
                     pistadb::Metric::Cosine,
                     pistadb::IndexType::HNSW, &p);
```

### IVF / IVF_PQ（需先训练）

```cpp
pistadb::Params p;
p.ivf_nlist  = 256;
p.ivf_nprobe = 16;

pistadb::Database db("large.pst", 1536,
                     pistadb::Metric::Cosine,
                     pistadb::IndexType::IVF, &p);
db.train();   // 插入前先构建聚类质心

for (auto& [id, vec, label] : corpus)
    db.insert(id, vec, label);

db.save();
```

### 异常处理

```cpp
try {
    pistadb::Database db("my.pst", 128);
    db.insert(1, vec, "hello");
    db.save();
} catch (const pistadb::Exception& e) {
    std::cerr << "PistaDB 错误: " << e.what() << '\n';
}
```

---

## Rust

`rust/` 目录是一个标准 Cargo crate（`pistadb`，无外部依赖），通过 `extern "C"` FFI 封装原生 C 库。

### Cargo 接入

```toml
[dependencies]
pistadb = { path = "../PistaDB/wrap/rust" }
```

构建前设置库搜索路径：

```bash
# Windows
set PISTADB_LIB_DIR=..\PistaDB\build\Release
cargo build

# Linux / macOS
PISTADB_LIB_DIR=../PistaDB/build cargo build
```

### 基础用法

```rust
use pistadb::{Database, Metric, IndexType};

fn main() -> Result<(), pistadb::Error> {
    let db = Database::open("knowledge.pst", 384, Metric::Cosine, IndexType::HNSW, None)?;

    let embedding = vec![0.1_f32; 384];
    db.insert(1, &embedding, Some("我的第一篇文档"))?;

    let results = db.search(&embedding, 5)?;
    for r in &results {
        println!("id={} dist={:.4} label={:?}", r.id, r.distance, r.label);
    }

    db.save()?;
    Ok(())
}
```

### 自定义参数

```rust
use pistadb::Params;

let p = Params { hnsw_m: 32, hnsw_ef_search: 100, ..Params::default() };

// 或使用内置预设
let p = Params::high_recall();   // M=32, efConstruction=400, efSearch=200
let p = Params::low_latency();   // M=16, efConstruction=100, efSearch=20

let db = Database::open("mydb.pst", 1536, Metric::Cosine, IndexType::HNSW, Some(&p))?;
```

### 多线程使用

`Database` 实现了 `Send + Sync`，用 `Arc` 包装即可在线程间共享：

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

### IVF / IVF_PQ（需先训练）

```rust
use pistadb::{Database, Metric, IndexType, Params};

let p = Params { ivf_nlist: 256, ivf_nprobe: 16, ..Params::default() };
let db = Database::open("large.pst", 1536, Metric::Cosine, IndexType::IVF, Some(&p))?;

db.train()?;   // 插入前先构建聚类质心

for (id, vec) in corpus.iter().enumerate() {
    db.insert(id as u64 + 1, vec, None)?;
}
db.save()?;
```

---

## Go

`go/` 目录是一个标准 Go 模块（`pistadb.io/go`），通过 **CGO** 封装原生 C 库，为数据库、批量插入和嵌入缓存三套 API 提供惯用 Go 类型。

### 模块接入

```go
// go.mod
module myapp

go 1.21

require pistadb.io/go v0.0.0

replace pistadb.io/go => ../PistaDB/wrap/go
```

先构建 C 库，再正常构建：

```bash
# 构建 C 库（在 PistaDB 根目录执行）
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel

# 构建你的 Go 应用
go build ./...
```

### 基础用法

```go
import "pistadb.io/go/pistadb"

db, err := pistadb.Open("knowledge.pst", 384,
    pistadb.MetricCosine, pistadb.IndexHNSW, nil)
if err != nil {
    log.Fatal(err)
}
defer db.Close()

if err := db.Insert(1, "我的第一篇文档", embedding); err != nil {
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

### 自定义参数

```go
p := pistadb.DefaultParams()
p.HNSWM = 32
p.HNSWEfSearch = 100

db, err := pistadb.Open("mydb.pst", 1536,
    pistadb.MetricCosine, pistadb.IndexHNSW, &p)
```

### IVF / IVF_PQ（需先训练）

```go
p := pistadb.DefaultParams()
p.IVFNList  = 256
p.IVFNProbe = 16

db, _ := pistadb.Open("large.pst", 1536,
    pistadb.MetricCosine, pistadb.IndexIVF, &p)

db.Train()   // 插入前先构建聚类质心

for i, vec := range corpus {
    db.Insert(uint64(i+1), "", vec)
}
db.Save()
```

### 批量插入

```go
// 流式 API — 多个 goroutine 并发 push
batch, _ := pistadb.NewBatch(db, 0, 0)   // 0 线程=自动，0 容量=默认 4096
defer batch.Destroy()

batch.Push(id, label, vec)

errors := batch.Flush()   // 等待所有项目完成，重置计数器

// 离线批量 API — 阻塞的便捷封装
failures, err := pistadb.BatchInsert(db, ids, labels, vecs, 0)
```

### 嵌入缓存

```go
cache, _ := pistadb.OpenCache("embed.pcc", 384, 100_000)
defer cache.Close()

vec, ok := cache.Get(text)
if !ok {
    vec = myModel.Encode(text)   // 仅在未命中时调用
    cache.Put(text, vec)
}

cache.Save()

stats := cache.Stats()
fmt.Printf("hits=%d  misses=%d  evictions=%d\n",
    stats.Hits, stats.Misses, stats.Evictions)
```

### 运行测试

```bash
cd go
go test ./pistadb/ -v
```
