<div align="center">

[English](./README.md) · [中文](./README_CN.md)

<h1>🌰 PistaDB</h1>

<p><strong>专为 LLM 原生应用打造的嵌入式向量数据库。</strong><br>
RAG 就绪 · 零外部依赖 · 单文件存储 · MIT 开源协议</p>

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

> **每一个 LLM 应用，最终都需要一个向量数据库。**
> 无论是检索增强生成（RAG）、语义搜索、智能体记忆，还是向量嵌入缓存——
> 通常的答案是云服务或容器化集群。PistaDB 给出了不同的答案。
>
> 小巧、密集、充满价值——就像它名字来源的那颗坚果——
> **PistaDB 将生产级向量存储浓缩进一个 `.pst` 文件和一个零依赖的 C 库。**
> 内嵌进桌面应用，部署到边缘设备，或直接和你的 Python 脚本放在一起。
> 无需 Docker，无需 API 密钥，数据永不离开本机。

---

## ✨ 为什么选择 PistaDB？

| | PistaDB | 云端 / 服务端向量数据库 |
|---|---|---|
| 部署方式 | 复制一个 `.dll` / `.so` | Docker、Kubernetes、云订阅 |
| 存储结构 | 单个 `.pst` 文件 | 数据文件 + WAL + 配置 + 附属文件 |
| 数据隐私 | **全部数据留在本地** | 向量嵌入通过网络传输 |
| 内存占用 | 可配置，极小 | JVM / 运行时动辄数 GB |
| 外部依赖 | **无**（纯 C99） | 数十个第三方包 |
| 查询延迟 | 笔记本上**亚毫秒级** | 网络往返延迟 |
| 使用成本 | 永久免费（MIT） | 按查询次数或向量数量计费 |

PistaDB 专为**本地 RAG 管道、离线 AI 智能体、隐私敏感应用、边缘推理，以及一切不适合部署完整向量集群的场景**而生——坦白说，大多数场景都是这样。

---

## 🤖 为 LLM 技术栈而生

现代 LLM 应用有一个共同的模式：将文本（或图片、音频、代码）转换为向量嵌入，存储起来，并在推理时检索最相关的内容，为模型提供精准的上下文。PistaDB 正是这套工作流中的存储层。

### 20 行实现 RAG

```python
from pistadb import PistaDB, Metric, Index
from sentence_transformers import SentenceTransformer  # 或任意嵌入模型

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 维向量

# ── 索引知识库 ────────────────────────────────────────────────────────────
docs = [
    "PistaDB 在本地存储向量，零外部依赖。",
    "RAG 通过注入检索到的上下文，显著提升 LLM 输出质量。",
    "HNSW 在大规模数据上实现亚毫秒级近似最近邻搜索。",
]
embeddings = model.encode(docs, normalize_embeddings=True).astype("float32")

with PistaDB("knowledge.pst", dim=384, metric=Metric.COSINE, index=Index.HNSW) as db:
    for i, (doc, vec) in enumerate(zip(docs, embeddings)):
        db.insert(i + 1, vec, label=doc)
    db.save()

# ── 推理时检索上下文 ───────────────────────────────────────────────────────
question = "PistaDB 如何助力检索增强生成？"
q_vec = model.encode([question], normalize_embeddings=True)[0].astype("float32")

with PistaDB("knowledge.pst", dim=384) as db:
    results = db.search(q_vec, k=3)

context = "\n".join(r.label for r in results)
# → 将 context 作为 prompt 的一部分传给 OpenAI / Claude / 本地模型
```

### 智能体持久记忆

LLM 智能体需要能跨会话保持的记忆。PistaDB 将情景记忆和语义记忆存储为向量——微秒级即可查询，持久化到你完全掌控的单个文件中。

```python
# 存储智能体本次会话产生的新记忆
memory_db = PistaDB("agent_memory.pst", dim=1536, metric=Metric.COSINE, index=Index.HNSW)
vec = openai_embed("用户偏好简洁的技术性表达。")
memory_db.insert(next_id, vec, label="用户偏好简洁的技术性表达。")
memory_db.save()

# 在下次 LLM 调用前，召回相关记忆
recall_vec = openai_embed(current_user_message)
memories = memory_db.search(recall_vec, k=5)
context = "\n".join(m.label for m in memories)
```

### 私有文档本地搜索

PistaDB 完全在设备本地运行。用本地模型（Ollama、llama.cpp）为文档生成向量，存入 PistaDB，实现搜索全程零网络请求。

```python
# 完全离线的流水线——没有任何数据触达网络
from pistadb import build_from_array

vecs   = local_model.encode(my_documents)           # 本地嵌入模型
labels = [doc[:255] for doc in my_documents]        # label 存储文档片段

db = build_from_array("private_docs.pst", vecs, labels=labels,
                       metric=Metric.COSINE, index=Index.HNSW)
db.save()
```

---

## 🚀 功能特性

### 7 种生产级索引算法

| 索引类型 | 算法 | 适用场景 |
|----------|------|----------|
| `LINEAR` | 暴力精确扫描 | 基准测试、小规模嵌入集 |
| `HNSW` | 分层可导航小世界图 | **RAG 首选** — 速度与召回率最佳平衡 |
| `IVF` | 倒排文件索引（k-means 聚类） | 有训练预算的大型知识库 |
| `IVF_PQ` | IVF + 乘积量化 | 内存受限的部署环境 |
| `DISKANN` | Vamana 图（DiskANN） | 十亿级向量集合 |
| `LSH` | 局部敏感哈希 | 极低内存占用场景 |
| `SCANN` | 各向异性向量量化（Google ScaNN） | MIPS / 余弦场景下的极致召回率 |

### 5 种距离度量——覆盖所有主流 LLM 嵌入模型

| 度量方式 | LLM / 嵌入模型适用场景 |
|----------|------------------------|
| `COSINE` | **文本嵌入**——OpenAI `text-embedding-3`、Cohere、`sentence-transformers`、BGE、GTE |
| `IP` | 内积——向量已 L2 归一化时与余弦等价，速度更快 |
| `L2` | 图像 / 多模态嵌入（CLIP、ImageBind） |
| `L1` | 稀疏特征向量、BM25 式混合检索 |
| `HAMMING` | 二值嵌入、哈希去重 |

### SIMD 加速距离计算核

全部 5 种距离函数均配备了手写 SIMD 内核，**运行时自动选择**，无需任何配置：

| 架构 | ISA | 主循环策略 | 典型加速比 |
|---|---|---|---|
| x86-64（Haswell+） | **AVX2 + FMA** | 每次迭代 16 floats，双累加器展开，融合乘加 | 标量的 4–8× |
| ARM / Apple Silicon | **NEON** | 每次迭代 16 floats（4× `float32x4_t`），4 累加器展开 | 标量的 3–5× |
| 其他架构 | 标量 | 标准 C11 循环 | 基准线 |

**运行时探测**——首次调用时对 CPU 进行一次探测（GCC/Clang 使用 `__builtin_cpu_supports`，MSVC 使用 `__cpuid` + `_xgetbv`），随即将函数指针原地替换为最优实现，此后所有调用直接跳转，无分支开销。

| 计算核 | AVX2 技术 | NEON 技术 |
|---|---|---|
| `vec_dot` / `dist_ip` | `_mm256_fmadd_ps` × 2 YMM | `vmlaq_f32` × 4 Q-寄存器 |
| `dist_l2sq` / `dist_l2` | `_mm256_sub_ps` → `_mm256_fmadd_ps` | `vsubq_f32` → `vmlaq_f32` |
| `dist_cosine` | 单遍三路累加器（dot、‖a‖²、‖b‖²） | 6 个 Q-寄存器累加器 |
| `dist_l1` | 符号位掩码（`AND 0x7FFFFFFF`）取绝对值 | `vabsq_f32` → `vaddq_f32` |
| `dist_hamming` | `_mm256_cmp_ps` + `movemask` + `popcount` | `vceqq_f32` + `vshrq_n_u32` + `vaddvq_u32` |

标量回退路径始终参与编译，数值结果与原始实现完全一致——速度提升不以正确性为代价。SIMD 文件（`distance_avx2.c`、`distance_neon.c`）各自以独立的 ISA 编译选项（`-mavx2 -mfma` / AArch64 无需额外标志）编译，链接进同一个共享库。

### 多线程批量插入

针对高吞吐量向量嵌入管道，PistaDB 提供了一套基于**线程池 + 有界环形缓冲队列**的批量插入 API，将向量生成与索引写入解耦：

```
生产者线程 0 ──▶ ┌──────────────────┐
生产者线程 1 ──▶ │  环形缓冲队列     │──▶ 工作线程 0 ─┐
生产者线程 N ──▶ │  (有界 MPMC)     │──▶ 工作线程 1 ─┤──▶ pistadb_insert()
                 └──────────────────┘──▶ 工作线程 M ─┘    （已串行化）
```

**流式 API** — 适合向量嵌入持续到达的在线管道：

```c
#include "pistadb_batch.h"

// 创建批量上下文：4 个工作线程，默认队列容量（4096）
PistaDBBatch *batch = pistadb_batch_create(db, 4, 0);

// 任意数量的生产者线程可同时调用 push()，线程安全。
// 队列已满时阻塞（背压机制）；vec 内部拷贝，调用方可立即释放。
pistadb_batch_push(batch, id, label, vec);   // 线程安全

// 等待所有已入队项目完成插入
int errors = pistadb_batch_flush(batch);     // 全部成功返回 0

pistadb_batch_destroy(batch);   // flush + 关闭工作线程 + 释放内存
```

**离线批量 API** — 数据已全部就绪时的单次阻塞调用：

```c
// ids[n]、labels[n]（可为 NULL）、vecs[n × dim]
// n_threads=0 → 自动探测 hardware_concurrency
int errors = pistadb_batch_insert(db, ids, labels, vecs, n, /*n_threads=*/0);
```

**性能模型：**

| 场景 | 收益 |
|---|---|
| 嵌入生成 + 索引写入流水线 | 并行嵌入计算与串行索引写入重叠 |
| 多生产者嵌入服务 | 多线程并发 push，单工作线程消费 |
| HNSW / DiskANN | 工作线程做图搜索（读密集）期间，下一项已就绪 |
| IVF / ScaNN | 队列中多项的质心查询（只读）可并发预取 |

**线程安全：** `pistadb_batch_push()` 可从任意线程安全调用。所有索引写入由内部锁串行化——批量上下文活跃期间，外部无需对 `PistaDB` 句柄加锁。

**平台实现：** Windows 使用 `CRITICAL_SECTION` + `CONDITION_VARIABLE`；Linux / macOS / Android / iOS 使用 `pthread_mutex_t` + `pthread_cond_t`。零外部依赖。

### 嵌入缓存层——自动对相同输入去重

嵌入 API（OpenAI、Cohere、本地模型）调用成本高昂。当同一段文本多次出现时——重复查询、语料去重、文档分块缓存——重复编码既浪费时间又消耗金钱。嵌入缓存层可自动消除冗余调用。

```
┌──────────────────────────────────────────────────────┐
│                      你的应用                        │
│                                                      │
│   文本 ──► CachedEmbedder ──► EmbeddingCache ──► 命中 │ ──► float32[]
│                    │               (LRU 映射)         │
│                    │ 未命中                           │
│                    ▼                                  │
│              embed_fn(text)    ← 缓存命中时           │
│           （OpenAI / 本地模型）    跳过模型调用        │
└──────────────────────────────────────────────────────┘
```

**C API** — `src/pistadb_cache.h`：

```c
// 打开（或加载）持久化缓存
PistaDBCache *cache = pistadb_cache_open("embed.pcc", /*dim=*/384, /*max=*/100000);

float vec[384];
if (!pistadb_cache_get(cache, text, vec)) {
    my_model_encode(text, vec);           // ← 仅在未命中时调用
    pistadb_cache_put(cache, text, vec);  // 内部拷贝，调用方可立即释放
}
// 使用 vec …

pistadb_cache_save(cache);   // 持久化到 embed.pcc（重启后依然有效）
pistadb_cache_close(cache);
```

**Python API** — `EmbeddingCache` + `CachedEmbedder`：

```python
from pistadb import EmbeddingCache, CachedEmbedder

cache    = EmbeddingCache("embed.pcc", dim=384, max_entries=100_000)
embedder = CachedEmbedder(openai_encode, cache, autosave_every=500)

vec  = embedder("什么是 RAG？")         # 首次访问时才调用 OpenAI
vecs = embedder.embed_batch(texts)      # np.ndarray (n, 384)

print(cache.stats())
# CacheStats(hits=4821, misses=179, evictions=0, count=179, hit_rate=96.4%)

cache.close()
```

**设计细节：**

| 属性 | 值 |
|---|---|
| 哈希函数 | FNV-1a 64 位，独立链表法 |
| 淘汰策略 | LRU（双向链表，O(1) 提升 / 淘汰） |
| 扩容阈值 | 负载因子 75%，桶数量翻倍 |
| 持久化格式 | `.pcc` 二进制——64 字节文件头 + 变长条目 |
| 线程安全 | 单内部互斥锁（所有公共函数均受保护） |
| 字节序 | 小端序（全平台一致） |

`.pcc` 文件按 LRU→MRU 顺序存储条目，重新加载时通过顺序 `put()` 调用即可精确还原 LRU 排序。

### 事务——原子性多操作批次

PistaDB 支持 **ACID 风格的事务**，可将任意 INSERT、DELETE、UPDATE 操作组合为一个原子执行单元。提交时若任何操作失败，已成功执行的操作将自动回滚。

**C API** — `src/pistadb_txn.h`：

```c
#include "pistadb_txn.h"

PistaDBTxn *txn = pistadb_txn_begin(db);

pistadb_txn_insert(txn, 101, "doc_a", vec_a);
pistadb_txn_insert(txn, 102, "doc_b", vec_b);
pistadb_txn_delete(txn, 55);              // 暂存时捕获撤销快照
pistadb_txn_update(txn, 77, vec_updated);

int rc = pistadb_txn_commit(txn);
if (rc == PISTADB_OK) {
    /* 全部操作已应用 */
} else if (rc == PISTADB_ETXN_PARTIAL) {
    /* 提交失败且回滚不完整（如 IVF-PQ 索引缺少原始向量） */
    fprintf(stderr, "partial failure: %s\n", pistadb_txn_last_error(txn));
} else {
    /* 提交失败，已完整回滚 */
    fprintf(stderr, "rolled back: %s\n", pistadb_txn_last_error(txn));
}

pistadb_txn_free(txn);
```

手动回滚：

```c
pistadb_txn_rollback(txn);   /* 丢弃所有暂存操作，句柄仍可复用 */
pistadb_txn_free(txn);
```

**Python API** — `Transaction` + 上下文管理器：

```python
from pistadb import PistaDB, Transaction

with PistaDB("mydb.pst", dim=128) as db:
    # 上下文管理器形式——正常退出时提交，发生异常时回滚
    with db.begin_transaction() as txn:
        txn.insert(101, vec_a, label="doc_a")
        txn.insert(102, vec_b, label="doc_b")
        txn.delete(55)
        txn.update(77, vec_updated)
    # 此处完成提交

    # 手动形式
    txn = db.begin_transaction()
    txn.insert(200, vec_c, label="doc_c")
    try:
        txn.commit()
    except RuntimeError as e:
        if getattr(e, "partial", False):
            print("部分回滚失败:", e)
        else:
            print("已回滚:", e)
    finally:
        txn.free()
```

**原子性模型：**

| 阶段 | 说明 |
|---|---|
| 暂存 | 操作在本地校验（检测重复 INSERT id） |
| 提交阶段一 | 结构校验（检查暂存 INSERT 中无重复 id） |
| 提交阶段二 | 操作依次应用；撤销快照在暂存时已捕获 |
| 回滚 | 操作 `i` 失败时，按逆序撤销 `i-1 … 0` |

**关于 IVF-PQ / ScaNN 的注意事项：** 这两种索引类型仅存储 PQ 码，不保留原始向量。针对此类索引的 DELETE 或 UPDATE 操作无法捕获撤销向量。若提交失败且需要对 PQ 操作执行撤销，函数将返回 `PISTADB_ETXN_PARTIAL = -10`，表示数据库处于部分应用状态，无法自动完整恢复。

### 单文件存储——与应用一同发布

- **一个 `.pst` 文件**存储所有内容：文件头、索引图、原始向量数据
- 128 字节固定文件头，内含 **CRC32 完整性校验**
- 完全向前兼容的二进制格式（版本化，保留字段供未来扩展）
- 原子写入：先写入缓冲区，一次性 flush——无部分写入，无数据损坏

---

## 🌍 多语言 & 多平台支持

PistaDB 的核心由纯 C99 编写。所有语言绑定都通过薄薄的零拷贝层封装同一个 C 库——无重复实现，无行为差异。同一个 `.pst` 数据库文件可在所有支持的平台之间无缝移植。

### 语言支持

| 语言 | 绑定机制 | API 风格 | 文件位置 |
|---|---|---|---|
| **C / C++** | 直接 `#include` | 原生 C API，零开销 | `src/pistadb.h` |
| **Python** | `ctypes`（无 Cython） | Pythonic，兼容 NumPy | `python/` |
| **Go** | CGO | 惯用 Go 类型，GC 终结器，批量 + 缓存 API | `go/` |
| **Java** | JNI | `AutoCloseable`、Builder 模式、`synchronized` | `android/src/main/java/` |
| **Kotlin** | JNI + 扩展函数 | DSL 构建器、协程、运算符重载 | `android/src/main/kotlin/` |
| **Objective-C** | 直接 C 互操作 | Cocoa 惯例，`NSError`，`NSLock` | `ios/Sources/PistaDBObjC/` |
| **Swift** | ObjC 桥接 | `throws`、`async/await`、尾随闭包 DSL | `ios/Sources/PistaDB/` |
| **C#** | P/Invoke | `IDisposable`、async/await、线程安全 | `csharp/` |
| **Rust** | FFI (`extern "C"`) | `Send + Sync`、`Drop`、`Result<T, Error>` | `rust/` |
| **C++** | 直接 include | RAII、只可移动、`std::mutex`、仅头文件 | `cpp/pistadb.hpp` |
| **WASM** | Emscripten / Embind | ESM 模块、`Float32Array`、TypeScript 类型 | `wasm/` |

### 平台支持

| 平台 | 工具链 | 产物 | ABI 目标 |
|---|---|---|---|
| **Windows** | MSVC | `pistadb.dll` | x86_64 |
| **Linux** | GCC | `libpistadb.so` | x86_64、aarch64 |
| **macOS** | Clang | `libpistadb.dylib` | x86_64、arm64 |
| **Android** | NDK (Clang) | `libpistadb_jni.so` | arm64-v8a、armeabi-v7a、x86_64、x86 |
| **iOS / macOS** | Xcode / SPM | 静态库 | arm64、arm64-Simulator、x86_64-Simulator |
| **WASM** | Emscripten | `.wasm` | — *(规划中)* |

---

## 📱 Android 接入（Java & Kotlin）

`android/` 是一个独立的 Android Library 模块。将其加入项目后，即可在 Java 和 Kotlin 中使用完整的向量数据库 API，底层通过 JNI 调用原生 C 库。

### Gradle 接入

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

### Kotlin — DSL 构建器 & 协程

```kotlin
import com.pistadb.*

// DSL 构建器——配置式写法，清晰易读
val db = pistaDB(path, dim = 384) {
    metric    = Metric.COSINE
    indexType = IndexType.HNSW
    params { hnswEfSearch = 100 }
}

// 所有阻塞操作都有 suspend 对应版本，在 MainScope 中调用完全安全
lifecycleScope.launch {
    db.insertAsync(id = 1L, vector = embedding, label = "文档")

    val results: List<SearchResult> = db.searchAsync(queryVec, k = 10)
    results.forEach { Log.d("RAG", "${it.label}  d=${it.distance}") }

    db.saveAsync()
}

// 便捷扩展
val ids: List<Long> = db.searchIds(queryVec, k = 5)   // 只返回 id 列表
db += (42L to VectorEntry(vec, "快速插入"))             // += 运算符
```

---

## 🍎 iOS / macOS 接入（Swift & Objective-C）

`ios/` 目录提供 SPM 兼容的软件包，分两层：直接调用 C API 的 Objective-C 封装层，以及其上的 Swift 惯用 API。

### Swift Package Manager 接入

在 Xcode 中选择 **File → Add Package Dependencies**，指向本仓库或本地路径；或在自己的 `Package.swift` 中添加：

```swift
// 在应用的 Package.swift 中
dependencies: [
    .package(path: "../PistaDB")   // 本地检出
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

// 插入
try db.insert(id: 1, vector: embeddingVector, label: "我的文档")

// 搜索——在后台线程执行，主线程永不阻塞
let results = try await db.search(queryVector, k: 10)
results.forEach { print("\($0.label)  d=\($0.distance)") }

// 作用域用法——自动保存并关闭
try withDatabase(path: dbPath, dim: 384, metric: .cosine) { db in
    try db.insertBatch(entries)
}   // ← 此处自动保存并关闭
```

内置高召回率与低延迟预设参数：

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

## 🔷 .NET / C# 接入

`csharp/` 目录是一个 .NET Standard 2.0 类库，通过 P/Invoke 封装原生 C 库。目标框架为 **.NET Standard 2.0**，兼容 .NET Core 2.0+、.NET 5/6/7/8、.NET Framework 4.6.1+、Unity、Xamarin 和 MAUI。

### 项目引用

将 `csharp/` 作为项目引用，或直接将三个源文件复制到你的解决方案中：

```xml
<!-- 在 .csproj 中 -->
<ItemGroup>
  <ProjectReference Include="../PistaDB/csharp/PistaDB.csproj" />
</ItemGroup>
```

确保 `pistadb.dll`（Windows）、`libpistadb.so`（Linux）或 `libpistadb.dylib`（macOS）在库搜索路径中（例如放在可执行文件旁边，或加入 `PATH` / `LD_LIBRARY_PATH`）。

### 基础用法

```csharp
using PistaDB;

// 打开或创建数据库（IDisposable — 推荐使用 using）
using var db = PistaDatabase.Open("knowledge.pst", dim: 384,
                                   metric: Metric.Cosine,
                                   indexType: IndexType.HNSW);

// 插入向量，附带可选标签
float[] embedding = GetEmbedding("PistaDB 在本地存储向量，零外部依赖。");
db.Insert(id: 1, vec: embedding, label: "PistaDB 在本地存储向量，零外部依赖。");

// K 近邻搜索
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

所有阻塞操作都有基于 `Task` 的封装，在线程池中运行——可在 UI 线程或 ASP.NET Core 控制器中安全调用：

```csharp
await db.InsertAsync(id: 1, vec: embedding, label: "文档");

IReadOnlyList<SearchResult> results = await db.SearchAsync(query, k: 10);

await db.SaveAsync();
```

### IVF / IVF_PQ（需先训练）

```csharp
using var db = PistaDatabase.Open("large.pst", dim: 1536,
                                   indexType: IndexType.IVF,
                                   @params: new PistaDBParams { IvfNList = 256, IvfNProbe = 16 });

db.Train();   // 插入前先构建聚类中心

foreach (var (id, vec, label) in corpus)
    db.Insert(id, vec, label);

db.Save();
```

---

## 🌐 WebAssembly 接入

PistaDB 通过 **Emscripten + Embind** 编译为自包含的 `.wasm` + `.js` 模块对。可在任意现代浏览器或 Node.js 中运行完整向量数据库——无需服务器、无需网络、无需原生二进制文件。

### 构建

```bash
# 首先激活 Emscripten SDK
source /path/to/emsdk/emsdk_env.sh

cd wasm
bash build.sh          # Release 构建（默认）

# 输出：wasm/build/pistadb.js + wasm/build/pistadb.wasm
```

### 浏览器使用（ESM）

```javascript
import PistaDB from './pistadb.js';

const M = await PistaDB();

// 打开或创建数据库（默认使用内存中的 MEMFS）
const db = new M.Database('knowledge.pst', 384, M.Metric.Cosine, M.IndexType.HNSW, null);

// 插入——传入 Float32Array
const embedding = new Float32Array(384).fill(0.1);
db.insert(1, embedding, '我的第一篇文档');

// 搜索——返回 { id, distance, label } 对象数组
const results = db.search(embedding, 5);
for (const r of results)
    console.log(`id=${r.id}  dist=${r.distance.toFixed(4)}  label=${r.label}`);

db.save();
db.delete();   // 释放 C++ 内存——使用完毕后必须调用
```

### Node.js 使用

```javascript
const PistaDB = require('./pistadb.js');

const M = await PistaDB();
// 在 Node.js 中，Emscripten 的 NODEFS 层会直接映射到真实文件系统路径
const db = new M.Database('/tmp/knowledge.pst', 384, M.Metric.Cosine, M.IndexType.HNSW, null);
db.insert(1, Float32Array.from({length: 384}, () => Math.random()), 'doc-1');
const results = db.search(new Float32Array(384).fill(0.5), 5);
db.save();
db.delete();
```

### 浏览器持久化存储（IDBFS）

默认情况下，文件存于内存中（MEMFS），页面刷新后丢失。挂载 **IndexedDB 文件系统**实现持久化：

```javascript
const M = await PistaDB();

// 挂载 IndexedDB 文件系统到 /idb
M.FS.mkdir('/idb');
M.FS.mount(M.IDBFS, {}, '/idb');

// 从 IndexedDB 同步到 MEMFS（true = IDB → MEMFS）
await new Promise((res, rej) =>
    M.FS.syncfs(true, err => err ? rej(err) : res()));

const db = new M.Database('/idb/knowledge.pst', 384, M.Metric.Cosine, M.IndexType.HNSW, null);

// ... 使用 db ...

db.save();

// 从 MEMFS 同步回 IndexedDB（false = MEMFS → IDB）
await new Promise((res, rej) =>
    M.FS.syncfs(false, err => err ? rej(err) : res()));

db.delete();
```

### TypeScript 类型支持

TypeScript 声明文件位于 `wasm/pistadb.d.ts`，将其与生成的文件一同部署即可：

```typescript
import PistaDB, { type PistaDBModule, type SearchResult } from './pistadb.js';

const M: PistaDBModule = await PistaDB();
const db = new M.Database('data.pst', 128, M.Metric.L2, M.IndexType.HNSW, null);
const results: SearchResult[] = db.search(new Float32Array(128), 5);
db.delete();
```

---

## ➕ C++ 接入

`cpp/pistadb.hpp` 是**单文件纯头文件** C++17 封装。直接复制到项目中并 `#include` 即用，无需编译步骤，无需代码生成。

### CMake 配置

推荐使用提供的 `cpp/CMakeLists.txt` 注册 INTERFACE target：

```cmake
add_subdirectory(PistaDB)        # 编译 pistadb 共享库
add_subdirectory(PistaDB/cpp)    # 注册 pistadb_cpp INTERFACE target

target_link_libraries(my_app PRIVATE pistadb_cpp)
# ↑ 自动将 pistadb.hpp 和 C 头文件加入 include 路径
```

或手动指定（GCC/Clang）：

```bash
g++ -std=c++17 -IPistaDB/cpp -IPistaDB/src main.cpp -Lbuild -lpistadb -o my_app
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

    // 插入向量，附带可选标签
    std::vector<float> embedding(384, 0.1f);
    db.insert(1, embedding, "我的第一篇文档");

    // K 近邻搜索
    auto results = db.search(embedding, 5);
    for (const auto& r : results)
        std::cout << "id=" << r.id
                  << " dist=" << r.distance
                  << " label=" << r.label << '\n';

    db.save();   // 析构函数不会自动保存，需显式调用
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

### 删除向量

```cpp
// 'delete' 是 C++ 关键字，此方法命名为 'remove'
db.remove(42);
```

### 异常处理

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

## 🦀 Rust 接入

`rust/` 目录是一个标准 Cargo crate（包名 `pistadb`，零外部依赖），通过 `extern "C"` FFI 封装原生 C 库。

### Cargo 配置

在你的 `Cargo.toml` 中添加本地路径依赖：

```toml
[dependencies]
pistadb = { path = "../PistaDB/rust" }
```

构建前设置库搜索路径：

```bash
# Windows
set PISTADB_LIB_DIR=..\PistaDB\build\Release
cargo build

# Linux / macOS
PISTADB_LIB_DIR=../PistaDB/build cargo build
```

`build.rs` 也会自动在 crate 根目录的 `../build`、`../build/Release`、`../build/Debug` 中搜索。

### 基础用法

```rust
use pistadb::{Database, Metric, IndexType};

fn main() -> Result<(), pistadb::Error> {
    // 打开或创建数据库
    let db = Database::open("knowledge.pst", 384, Metric::Cosine, IndexType::HNSW, None)?;

    // 插入向量，附带可选标签
    let embedding = vec![0.1_f32; 384];
    db.insert(1, &embedding, Some("我的第一篇文档"))?;

    // K 近邻搜索
    let results = db.search(&embedding, 5)?;
    for r in &results {
        println!("id={} dist={:.4} label={:?}", r.id, r.distance, r.label);
    }

    db.save()?;
    // db 在此被 drop，自动调用 pistadb_close()
    Ok(())
}
```

### 自定义参数

```rust
use pistadb::Params;

// 结构体更新语法精细调参
let p = Params { hnsw_m: 32, hnsw_ef_search: 100, ..Params::default() };

// 或使用内置预设
let p = Params::high_recall();   // M=32, efConstruction=400, efSearch=200
let p = Params::low_latency();   // M=16, efConstruction=100, efSearch=20

let db = Database::open("mydb.pst", 1536, Metric::Cosine, IndexType::HNSW, Some(&p))?;
```

### 多线程使用

`Database` 实现了 `Send + Sync`，用 `Arc` 包装后即可跨线程共享：

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

---

## 🐹 Go 接入

`go/` 目录是一个标准 Go 模块（`pistadb.io/go`），通过 **CGO** 封装原生 C 库，提供了面向数据库、批量插入和嵌入缓存三个 API 的惯用 Go 类型。

### 模块配置

在你的 `go.mod` 中添加本地 replace 指令：

```go
// go.mod
module myapp

go 1.21

require pistadb.io/go v0.0.0

replace pistadb.io/go => ../PistaDB/go
```

先编译 C 库，然后正常 build：

```bash
# 从 PistaDB 根目录编译 C 库
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel

# 编译 Go 应用
go build ./...
```

如需指定自定义构建目录，设置 `CGO_LDFLAGS`：

```bash
export CGO_LDFLAGS="-L/自定义路径 -lpistadb"
go build ./...
```

### 基本用法

```go
import "pistadb.io/go/pistadb"

// 打开或创建数据库
db, err := pistadb.Open("knowledge.pst", 384,
    pistadb.MetricCosine, pistadb.IndexHNSW, nil)
if err != nil {
    log.Fatal(err)
}
defer db.Close()

// 插入向量
db.Insert(1, "我的第一篇文档", embedding)

// K 近邻检索
results, err := db.Search(queryVec, 10)
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

### 批量插入

```go
// 流式 API — 多 goroutine 并发 Push
batch, _ := pistadb.NewBatch(db, 0, 0)   // 0 = 自动检测线程数和队列容量
defer batch.Destroy()

batch.Push(id, label, vec)     // 线程安全，队列满时自动背压
errors := batch.Flush()        // 等待所有任务完成

// 离线批量 API — 阻塞式便捷封装
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
fmt.Printf("命中=%d  未命中=%d  淘汰=%d\n",
    stats.Hits, stats.Misses, stats.Evictions)
```

### 运行测试

```bash
cd go
go test ./pistadb/ -v
```

---

## 📦 安装

### 1. 编译 C 库

**Windows（MSVC）：**
```bat
build.bat Release
```

**Linux / macOS（GCC / Clang）：**
```bash
bash build.sh Release
```

编译产物为 `pistadb.dll`（Windows）或 `libpistadb.so`（Linux / macOS），**零外部依赖**。

### 2. 安装 Python 绑定

```bash
pip install -e python/
```

就这些。无需 Rust 编译器，Python 安装步骤也不依赖 CMake，没有任何隐藏惊喜。

### 3. Android 接入

在 Android Studio 中将 `android/` 作为 Library 模块引入，或在 `settings.gradle` 中声明：

```groovy
include ':android'
project(':android').projectDir = new File('<PistaDB路径>/android')
```

NDK 构建由 `android/CMakeLists.txt` 全程管理。请确保已安装 NDK `26.x`，并将 `android/build.gradle` 中的 `ndkVersion` 与本地版本对应。

### 4. WASM 接入

```bash
source /path/to/emsdk/emsdk_env.sh
cd wasm && bash build.sh
# → wasm/build/pistadb.js + pistadb.wasm
```

将两个文件部署到同一 HTTP 源，或在 Node.js 中直接使用。

### 5. C++ 接入

将 `cpp/` 和 `src/` 加入 include 路径，链接原生库：

```cmake
add_subdirectory(PistaDB)
add_subdirectory(PistaDB/cpp)
target_link_libraries(my_app PRIVATE pistadb_cpp)
```

### 6. Go 接入

在 `go.mod` 中添加 replace 指令，然后正常构建：

```go
replace pistadb.io/go => ../PistaDB/go
```

```bash
export CGO_LDFLAGS="-L../PistaDB/build -lpistadb"
go get pistadb.io/go/pistadb
go build ./...
```

### 7. Rust 接入

设置 `PISTADB_LIB_DIR` 指向编译好的原生库目录，然后执行构建：

```bash
cd rust
PISTADB_LIB_DIR=../build cargo build --release
```

### 8. C# / .NET 接入

将 `csharp/` 加入项目引用，并确保原生库在搜索路径中：

```bash
# Windows：将 pistadb.dll 复制到可执行文件旁
copy build\Release\pistadb.dll MyApp\bin\Debug\net8.0\

# Linux：设置 LD_LIBRARY_PATH 或复制 libpistadb.so
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
```

### 9. iOS / macOS 接入（Swift Package Manager）

在 Xcode 中选择 **File → Add Package Dependencies**，指向本仓库或本地路径；或在自己的 `Package.swift` 中添加：

```swift
.package(path: "../PistaDB")
```

项目根目录的 `Package.swift` 已声明三个 SPM Target——`CPistaDB`（C 核心）、`PistaDBObjC`、`PistaDB`（Swift），SPM 会自动处理依赖链。

---

## ⚡ 快速上手

### 基础用法

```python
import numpy as np
from pistadb import PistaDB, Metric, Index, Params

# 一个文件，所有内容都在里面——可以随应用一同分发
params = Params(hnsw_M=16, hnsw_ef_construction=200, hnsw_ef_search=50)
db = PistaDB("mydb.pst", dim=1536, metric=Metric.COSINE, index=Index.HNSW, params=params)

# 插入向量——label 存储原始文本片段，方便检索时直接取用
vec = np.random.rand(1536).astype("float32")
db.insert(1, vec, label="chunk_0001")

# 搜索——返回按距离排序的结果，包含 id、距离和标签
query = np.random.rand(1536).astype("float32")
results = db.search(query, k=10)
for r in results:
    print(f"id={r.id}  dist={r.distance:.4f}  label={r.label!r}")

# 持久化到磁盘
db.save()
db.close()

# 重新加载——索引完整恢复，数毫秒内可用
db2 = PistaDB("mydb.pst", dim=1536)
```

### 上下文管理器

```python
with PistaDB("docs.pst", dim=768, metric=Metric.COSINE) as db:
    db.insert(1, vec, label="文档片段")
    results = db.search(query, k=5)
    db.save()
# 退出时自动关闭
```

### 从 NumPy 数组批量构建

```python
from pistadb import build_from_array

# 典型流程：对语料库生成向量嵌入，再交给 PistaDB 完成构建
vecs   = embed_model.encode(corpus).astype("float32")   # shape (n, dim)
labels = [chunk[:255] for chunk in corpus]

db = build_from_array("corpus.pst", vecs, labels=labels,
                       metric=Metric.COSINE, index=Index.HNSW)
db.save()
```

### IVF / IVF_PQ（需先训练）

```python
# 适合大型嵌入集合（10 万条向量以上）
db = PistaDB("large_kb.pst", dim=1536, index=Index.IVF,
             params=Params(ivf_nlist=256, ivf_nprobe=16))

db.train(representative_vecs)   # 在样本上构建聚类中心

for i, v in enumerate(all_embeddings):
    db.insert(i + 1, v, label=doc_chunks[i])

db.save()
```

### DiskANN / Vamana 图索引

```python
db = PistaDB("graph.pst", dim=1536, index=Index.DISKANN,
             params=Params(diskann_R=32, diskann_L=100, diskann_alpha=1.2))

for i, v in enumerate(vectors):
    db.insert(i + 1, v)

db.train()       # 可选：触发完整 Vamana 图重建以获得最佳质量
results = db.search(query, k=10)
```

### ScaNN — 各向异性向量量化

ScaNN（Scalable Nearest Neighbors，Google Research，ICML 2020）是 PistaDB 中召回率最高的索引。它在 IVF-PQ 基础上引入**各向异性量化变换**——优先压制平行于原始向量方向上的量化误差，而这正是影响内积和余弦召回率的关键误差分量。

**两阶段搜索：**
1. **第一阶段** — 基于 PQ 编码的快速 ADC 打分：通过预计算的查询残差查找表完成近似距离打分，无需浮点乘法，速度极快。
2. **第二阶段** — 精确重排序：对 top `rerank_k` 个候选向量用存储的原始浮点向量进行精确重打分，从压缩候选集中恢复接近完美的召回率。

```python
from pistadb import PistaDB, Metric, Index, Params

# ScaNN 和 IVF / IVF_PQ 一样，插入前必须先训练
params = Params(
    scann_nlist    = 256,   # 粗量化 IVF 分区数
    scann_nprobe   = 32,    # 查询时探测的分区数
    scann_pq_M     = 16,    # PQ 子空间数（dim 必须能被 scann_pq_M 整除）
    scann_pq_bits  = 8,     # 每个子码的位数（4 或 8）
    scann_rerank_k = 200,   # 精确重排序的候选数量（应大于查询的 k）
    scann_aq_eta   = 0.2,   # 各向异性惩罚系数 η（0 = 标准 PQ）
)

db = PistaDB("scann.pst", dim=1536, metric=Metric.COSINE,
             index=Index.SCANN, params=params)

db.train(representative_vecs)   # 在样本上构建聚类中心和 PQ 码本

for i, v in enumerate(all_embeddings):
    db.insert(i + 1, v, label=doc_chunks[i])

results = db.search(query, k=10)
db.save()
```

**推荐使用场景：**
- 在余弦 / 内积度量下追求最高召回率（OpenAI、Cohere、sentence-transformers 等文本嵌入）
- 数据集足够大，能从 IVF 分区中受益（5 万条向量以上）
- 可以接受比 IVF-PQ 稍高的内存开销（每条倒排列表条目同时存储 PQ 编码和原始浮点向量用于重排序）

**关键参数说明：**

| 参数 | 作用 | 推荐起始值 |
|------|------|-----------|
| `scann_nlist` | 粗量化分区数 | `sqrt(n_vectors)` |
| `scann_nprobe` | 每次查询探测的分区数 | `nlist` 的 10–15% |
| `scann_pq_M` | 压缩率（每个子空间 `dim/pq_M` 个浮点） | `dim/4` 到 `dim/8` |
| `scann_rerank_k` | 重排序候选数量（越大召回率越高，速度越慢） | 查询 `k` 的 5–20 倍 |
| `scann_aq_eta` | 各向异性惩罚（0 = 标准 PQ） | 余弦 / IP 用 `0.2`；L2 用 `0.0` |

### 事务

```python
from pistadb import PistaDB, Metric, Index
import numpy as np

dim = 128
rng = np.random.default_rng(42)

with PistaDB("mydb.pst", dim=dim, metric=Metric.COSINE, index=Index.HNSW) as db:
    # 初始化数据
    for i in range(1, 6):
        db.insert(i, rng.random(dim).astype("float32"), label=f"doc_{i}")

    # 原子批次：全部成功或全部回滚
    with db.begin_transaction() as txn:
        txn.insert(10, rng.random(dim).astype("float32"), label="new_doc")
        txn.delete(3)                                        # 删除旧条目
        txn.update(1, rng.random(dim).astype("float32"))    # 替换向量
    # 三个操作在此处同时生效；提交前均不可见

    # 发生异常时自动回滚
    try:
        with db.begin_transaction() as txn:
            txn.insert(20, rng.random(dim).astype("float32"), label="maybe")
            raise ValueError("出错了")
    except ValueError:
        pass  # 事务已回滚——id 20 从未被插入
```

---

## 🧪 运行测试

```bash
# Windows
set PISTADB_LIB_DIR=build\Release
pytest tests\ -v

# Linux / macOS
PISTADB_LIB_DIR=build pytest tests/ -v
```

**109 / 109 测试全部通过**，覆盖召回率基准测试、持久化往返测试、损坏文件拒绝加载、各距离度量正确性验证、ScaNN 两阶段搜索专项测试，以及事务原子性与回滚测试。

---

## 📊 运行示例

```bash
# Windows
set PISTADB_LIB_DIR=build\Release
python examples/example.py

# Linux / macOS
PISTADB_LIB_DIR=build python examples/example.py
```

示例脚本演示了 12 个完整场景：各距离度量演示、全部 7 种索引类型（含 ScaNN）、批量构建、DiskANN 图重建、持久化往返验证、删除与更新操作、多线程批量插入，以及事务原子性与回滚。

---

## 🗂️ 文件格式

```
PistaDB 文件格式 v1.0
───────────────────────────────────────────
[文件头]  128 字节（固定）
  magic[4]           = "PSDB"
  version_major[2]   = 1
  version_minor[2]   = 0
  flags[4]           = 0  （保留，供未来使用）
  dimension[4]       = 向量维度
  metric_type[2]     = 距离度量枚举值
  index_type[2]      = 索引算法枚举值
  num_vectors[8]     = 向量总数（含软删除）
  next_id[8]         = 自增 ID 计数器
  vec_offset[8]      = 数据块字节偏移
  vec_size[8]        = 数据块字节大小
  idx_offset[8]      = 辅助索引数据偏移
  idx_size[8]        = 辅助索引数据大小
  reserved[54]       = 0
  header_crc[4]      = CRC32（文件头字节 [0..123]）

[向量 + 索引数据]  位于 vec_offset
  各索引自描述二进制序列化（版本化）
```

所有支持的索引类型（HNSW、IVF、DiskANN、LSH、ScaNN 等）均将自身序列化至此数据块——图边、聚类中心、PQ 码本、投影矩阵和原始浮点向量全部集中在一处。文件完全自包含：无附属文件，无注册表条目，加载时无需任何环境变量。

---

## 🏗️ 项目结构

```
PistaDB/
├── src/                          # C99 核心——所有平台共用
│   ├── pistadb_types.h           # 公共类型、错误码、默认参数
│   ├── pistadb.h / .c            # 主数据库 API
│   ├── distance.h / .c           # 运行时 SIMD 探测与调度 + 标量回退
│   ├── distance_simd.h           # AVX2 / NEON 内核内部声明
│   ├── distance_avx2.c           # AVX2+FMA 内核（编译选项：-mavx2 -mfma）
│   ├── distance_neon.c           # ARM NEON 内核（AArch64 内置，无需额外标志）
│   ├── pistadb_batch.h / .c      # 多线程批量插入（线程池 + 环形缓冲队列）
│   ├── pistadb_cache.h / .c      # 嵌入缓存（FNV-1a 哈希表 + LRU 链表 + .pcc 文件）
│   ├── pistadb_txn.h / .c        # 事务 API（原子多操作批次，失败自动回滚）
│   ├── utils.h / .c              # 二叉堆、PCG32 随机数生成器、位图
│   ├── storage.h / .c            # 文件 I/O、文件头 CRC32
│   ├── index_linear.*            # 精确暴力扫描
│   ├── index_hnsw.*              # HNSW（Malkov & Yashunin，2018）
│   ├── index_ivf.*               # IVF + k-means 聚类
│   ├── index_ivf_pq.*            # IVF + 乘积量化
│   ├── index_diskann.*           # Vamana / DiskANN（Subramanya 等，2019）
│   ├── index_lsh.*               # E2LSH + 符号哈希
│   └── index_scann.*             # ScaNN：各向异性向量量化（Guo 等，ICML 2020）
├── python/                       # Python 绑定
│   ├── pistadb/__init__.py       # 纯 ctypes 封装（无 Cython，无 cffi）
│   └── setup.py
├── go/                           # Go 绑定（CGO）
│   ├── go.mod                    # 模块：pistadb.io/go（Go 1.21）
│   └── pistadb/
│       ├── pistadb.go            # Database：Open、Insert、Search、Get……
│       ├── batch.go              # Batch：NewBatch、Push、Flush、BatchInsert
│       ├── cache.go              # Cache：OpenCache、Get、Put、Save、Stats
│       └── pistadb_test.go       # 集成测试（go test ./pistadb/）
├── android/                      # Android 接入层
│   ├── CMakeLists.txt            # NDK 构建（编译 C 核心 + JNI 桥接）
│   ├── build.gradle              # Android Library 模块（minSdk 21）
│   ├── proguard-rules.pro        # 保留 JNI 反射字段
│   └── src/main/
│       ├── cpp/pistadb_jni.c     # JNI 桥接层（15 个 native 方法）
│       ├── java/com/pistadb/     # Java API
│       │   ├── PistaDB.java      # 主类（AutoCloseable，synchronized）
│       │   ├── Metric.java       # 距离度量枚举
│       │   ├── IndexType.java    # 索引算法枚举
│       │   ├── PistaDBParams.java# 19 个调参项 + fluent Builder
│       │   ├── SearchResult.java # id + distance + label
│       │   ├── VectorEntry.java  # float[] vector + label
│       │   └── PistaDBException.java
│       └── kotlin/com/pistadb/ # Kotlin 扩展
│           ├── PistaDBExtensions.kt  # DSL 构建器、运算符重载
│           └── PistaDBCoroutines.kt  # suspend 封装（Dispatchers.IO）
├── ios/                          # iOS / macOS 接入层
│   └── Sources/
│       ├── PistaDBObjC/          # Objective-C 封装层
│       │   ├── include/          # 公开头文件（PST 前缀）
│       │   │   ├── PistaDBObjC.h # 伞形头文件
│       │   │   ├── PSTDatabase.h # 主类
│       │   │   ├── PSTMetric.h   # 距离度量枚举
│       │   │   ├── PSTIndexType.h# 索引算法枚举
│       │   │   ├── PSTParams.h   # 调参对象（NSCopying）
│       │   │   ├── PSTSearchResult.h
│       │   │   ├── PSTVectorEntry.h
│       │   │   └── PSTError.h    # 错误域 + NS_ENUM 错误码
│       │   └── PSTDatabase.m     # 实现（NSLock 线程安全）
│       └── PistaDB/              # Swift API
│           ├── PistaDB.swift     # 主类（throws，Closeable）
│           ├── PistaDBTypes.swift# Metric/IndexType/PistaDBParams/PistaDBError
│           ├── PistaDBAsync.swift# async/await 扩展（iOS 13+）
│           └── PistaDBExtensions.swift # pistaDB() DSL、subscript、预设参数
├── wasm/                         # WebAssembly 绑定（Emscripten + Embind）
│   ├── pistadb_wasm.cpp          # Embind 绑定——Database 类 + 枚举注册
│   ├── CMakeLists.txt            # emcmake 构建（C 核心 + 绑定 → .js + .wasm）
│   ├── build.sh                  # 构建脚本（需在 shell 中激活 emsdk）
│   └── pistadb.d.ts              # TypeScript 声明文件
├── cpp/                          # C++ 绑定（纯头文件，C++17）
│   ├── pistadb.hpp               # 单文件封装——唯一需要的文件
│   └── CMakeLists.txt            # CMake INTERFACE library target
├── rust/                         # Rust 绑定（Cargo crate，零外部依赖）
│   ├── Cargo.toml                # 包定义（edition 2021）
│   ├── build.rs                  # 链接原生库，支持 PISTADB_LIB_DIR 环境变量
│   └── src/
│       ├── lib.rs                # 公开 API：Database 结构体、re-exports、文档测试
│       ├── ffi.rs                # 原始 extern "C" 声明 + repr(C) 结构体
│       └── types.rs              # Metric/IndexType 枚举、SearchResult、VectorEntry、Params、Error
├── csharp/                       # C# / .NET 绑定（.NET Standard 2.0）
│   ├── PistaDB.csproj            # 项目文件（netstandard2.0）
│   ├── NativeMethods.cs          # P/Invoke 声明 + 原生结构体
│   ├── PistaDBTypes.cs           # Metric/IndexType 枚举、SearchResult、PistaDBParams、异常
│   └── PistaDatabase.cs          # 主类（IDisposable、线程安全、异步封装）
├── tests/
│   └── test_pistadb.py           # 109 个 pytest 测试用例
├── examples/
│   └── example.py                # 12 个端到端使用示例
├── Package.swift                 # SPM 清单（CPistaDB → PistaDBObjC → PistaDB）
├── CMakeLists.txt
├── build.sh                      # Linux / macOS 构建脚本
└── build.bat                     # Windows MSVC 构建脚本
```

---

## 🗺️ 路线图

**核心引擎**
- [x] SIMD 加速距离计算核（AVX2 / NEON），加速向量嵌入比较
- [ ] 带元数据谓词的过滤搜索（按来源、日期、标签先过滤再做 ANN）
- [x] 多线程批量插入，支持高吞吐量向量嵌入管道

**LLM 与 RAG 生态**
- [ ] LangChain 和 LlamaIndex 集成（开箱即用的 vectorstore 适配器）
- [ ] 原生支持 OpenAI、Cohere、sentence-transformers 常见嵌入维度的预设配置
- [ ] 混合搜索：稠密向量 + 稀疏 BM25 重排序，单次查询完成
- [x] 嵌入缓存层——自动对相同输入去重，避免重复调用嵌入 API
- [x] 事务支持——原子性多操作批次，失败时自动回滚

**可移植性**
- [x] Android 绑定——通过 JNI 提供 Java + Kotlin API（`android/`）
- [x] iOS / macOS 绑定——通过 SPM 提供 Objective-C + Swift API（`ios/`、`Package.swift`）
- [x] C# / .NET 绑定——P/Invoke、`IDisposable`、async/await（`csharp/`）
- [x] Rust 绑定——FFI、`Send + Sync`、`Drop`、`Result<T, Error>`（`rust/`）
- [x] C++ 绑定——纯头文件 C++17、RAII、只可移动、`std::mutex`（`cpp/pistadb.hpp`）
- [x] Go 绑定——CGO、惯用 Go 类型、GC 终结器、批量插入 + 嵌入缓存 API（`go/`）
- [x] WASM 构建——Emscripten + Embind、ESM 模块、`Float32Array`、TypeScript 类型（`wasm/`）
- [ ] WASM 构建——在浏览器中运行完整 RAG 管道
- [ ] HTTP 微服务模式（可选，单一二进制文件，支持多进程共享访问）

---

## 🤝 参与贡献

欢迎提交 Pull Request。无论是新的索引算法、语言绑定、性能优化、LLM 集成还是文档改进——每一份贡献都让 PistaDB 对整个社区更有价值。

1. Fork 本仓库
2. 创建功能分支（`git checkout -b feat/langchain-integration`）
3. 提交你的修改
4. 发起 Pull Request

提交前请确保 109 个测试用例全部通过。

---

<div align="center">
<strong>C99 编写 · C++ · WASM · Python · Go · Java · Kotlin · Swift · Objective-C · C# · Rust · 随处运行 · 数据永不离机</strong><br>
<em>LLM 应用最好的基础设施，是那种让你完全不需要费心去管的基础设施。</em>
</div>
