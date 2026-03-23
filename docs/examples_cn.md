# 使用示例

---

## LLM 与 RAG 集成

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

## 快速入门示例

### 基础用法

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

# 重新加载——索引完整恢复，毫秒级即可查询
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

vecs   = embed_model.encode(corpus).astype("float32")   # shape (n, dim)
labels = [chunk[:255] for chunk in corpus]

db = build_from_array("corpus.pst", vecs, labels=labels,
                       metric=Metric.COSINE, index=Index.HNSW)
db.save()
```

---

## 高级索引示例

### IVF / IVF_PQ（大规模向量集合）

适合 10 万条以上的向量集合。插入前需要先执行训练。

```python
db = PistaDB("large_kb.pst", dim=1536, index=Index.IVF,
             params=Params(ivf_nlist=256, ivf_nprobe=16))

db.train(representative_vecs)   # 在样本数据上构建聚类质心

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

db.train()       # 可选：触发完整 Vamana 图重建
results = db.search(query, k=10)
```

### ScaNN — 各向异性向量量化

ScaNN（Google Research，ICML 2020）是 PistaDB 中召回率最高的索引。它在 IVF-PQ 基础上引入**各向异性量化变换**，重点压缩与原始向量方向平行的量化误差分量——该分量对内积和余弦召回率影响最大。

**两阶段搜索：**
1. **阶段一** — 基于 PQ 码的快速 ADC 打分：通过预计算查找表近似查询残差。
2. **阶段二** — 精确重排序：对 top `rerank_k` 候选项用原始 float 向量重新打分，从压缩候选中恢复接近完美的召回率。

```python
from pistadb import PistaDB, Metric, Index, Params

params = Params(
    scann_nlist    = 256,   # 粗粒度 IVF 分区数
    scann_nprobe   = 32,    # 查询时探测的分区数
    scann_pq_M     = 16,    # PQ 子空间数（dim 须能被 scann_pq_M 整除）
    scann_pq_bits  = 8,     # 每个子码的位数（4 或 8）
    scann_rerank_k = 200,   # 精确重排候选数（应大于 k）
    scann_aq_eta   = 0.2,   # 各向异性惩罚系数 η（0 = 标准 PQ）
)

db = PistaDB("scann.pst", dim=1536, metric=Metric.COSINE,
             index=Index.SCANN, params=params)

db.train(representative_vecs)   # 在样本上构建质心和 PQ 码本

for i, v in enumerate(all_embeddings):
    db.insert(i + 1, v, label=doc_chunks[i])

results = db.search(query, k=10)
db.save()
```

**关键参数说明：**

| 参数 | 作用 | 推荐起点 |
|------|------|----------|
| `scann_nlist` | 粗粒度分区数 | `sqrt(n_vectors)` |
| `scann_nprobe` | 每次查询探测的分区数 | `nlist` 的 10–15% |
| `scann_pq_M` | 压缩比（每个子空间 `dim/pq_M` 个 float） | `dim/4` 到 `dim/8` |
| `scann_rerank_k` | 重排候选数（越大召回率越高，速度越慢） | 查询 `k` 的 5–20 倍 |
| `scann_aq_eta` | 各向异性惩罚（0 = 标准 PQ） | 余弦/IP 场景用 `0.2`；L2 场景用 `0.0` |

---

## 事务

### Python — 上下文管理器

```python
from pistadb import PistaDB, Metric, Index
import numpy as np

dim = 128
rng = np.random.default_rng(42)

with PistaDB("mydb.pst", dim=dim, metric=Metric.COSINE, index=Index.HNSW) as db:
    for i in range(1, 6):
        db.insert(i, rng.random(dim).astype("float32"), label=f"doc_{i}")

    # 原子批次：全部成功或全部回滚
    with db.begin_transaction() as txn:
        txn.insert(10, rng.random(dim).astype("float32"), label="new_doc")
        txn.delete(3)
        txn.update(1, rng.random(dim).astype("float32"))
    # 三个操作现在全部可见；提交前均不可见

    # 异常触发自动回滚
    try:
        with db.begin_transaction() as txn:
            txn.insert(20, rng.random(dim).astype("float32"), label="maybe")
            raise ValueError("出现错误")
    except ValueError:
        pass  # 事务已回滚——id 20 从未被插入
```

### C API

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

**原子性模型：**

| 阶段 | 说明 |
|---|---|
| 暂存 | 操作在本地校验（检测重复 INSERT id） |
| 提交阶段一 | 结构校验（检查暂存 INSERT 中无重复 id） |
| 提交阶段二 | 操作依次应用；撤销快照在暂存时已捕获 |
| 回滚 | 操作 `i` 失败时，按逆序撤销 `i-1 … 0` |

> **关于 IVF-PQ / ScaNN：** 这两种索引类型仅存储 PQ 码，不保留原始向量。针对此类索引的 DELETE 或 UPDATE 操作无法捕获撤销向量。若提交失败且需执行 PQ 操作的撤销，函数将返回 `PISTADB_ETXN_PARTIAL = -10`，表示数据库处于部分应用状态，无法自动完整恢复。

---

## 多线程批量插入

针对高吞吐量向量嵌入管道，PistaDB 提供了一套基于**线程池 + 有界环形缓冲队列**的批量插入 API，将向量生成与索引写入解耦。

```
生产者线程 0 ──▶ ┌──────────────────┐
生产者线程 1 ──▶ │  环形缓冲队列     │──▶ 工作线程 0 ─┐
生产者线程 N ──▶ │  (有界 MPMC)     │──▶ 工作线程 1 ─┤──▶ pistadb_insert()
                 └──────────────────┘──▶ 工作线程 M ─┘    （已串行化）
```

### 流式 API（C）

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

### 离线批量 API（C）

```c
// ids[n]、labels[n]（可为 NULL）、vecs[n × dim]
// n_threads=0 → 自动探测 hardware_concurrency
int errors = pistadb_batch_insert(db, ids, labels, vecs, n, /*n_threads=*/0);
```

---

## 嵌入缓存

嵌入 API（OpenAI、Cohere、本地模型）调用成本高昂。当同一段文本多次出现时，嵌入缓存可自动消除冗余调用。

### C API

```c
PistaDBCache *cache = pistadb_cache_open("embed.pcc", /*dim=*/384, /*max=*/100000);

float vec[384];
if (!pistadb_cache_get(cache, text, vec)) {
    my_model_encode(text, vec);           // ← 仅在未命中时调用
    pistadb_cache_put(cache, text, vec);  // 内部拷贝，调用方可立即释放
}

pistadb_cache_save(cache);
pistadb_cache_close(cache);
```

### Python API

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
