# Usage Examples

---

## LLM & RAG Integration

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

## Quick Start Examples

### Basic Usage

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

vecs   = embed_model.encode(corpus).astype("float32")   # shape (n, dim)
labels = [chunk[:255] for chunk in corpus]

db = build_from_array("corpus.pst", vecs, labels=labels,
                       metric=Metric.COSINE, index=Index.HNSW)
db.save()
```

---

## Advanced Index Examples

### IVF / IVF_PQ (Large Collections)

Good choice for large embedding collections (100k+ vectors). Requires a training pass before inserts.

```python
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
1. **Phase 1** — Fast ADC scoring over PQ codes: query residuals are approximated via precomputed lookup tables.
2. **Phase 2** — Exact reranking: the top `rerank_k` candidates are re-scored with the raw stored float vectors, recovering near-perfect recall from compressed candidates.

```python
from pistadb import PistaDB, Metric, Index, Params

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

**Key parameter guidance:**

| Parameter | Effect | Recommended starting point |
|-----------|--------|---------------------------|
| `scann_nlist` | Coarse partition count | `sqrt(n_vectors)` |
| `scann_nprobe` | Partitions searched per query | 10–15% of `nlist` |
| `scann_pq_M` | Compression ratio (`dim / pq_M` floats per sub-space) | `dim / 4` to `dim / 8` |
| `scann_rerank_k` | Rerank candidates (higher = better recall, slower) | 5–20× the query `k` |
| `scann_aq_eta` | Anisotropic penalty (0 = standard PQ) | `0.2` for cosine/IP; `0.0` for L2 |

---

## Transactions

### Python — Context Manager

```python
from pistadb import PistaDB, Metric, Index
import numpy as np

dim = 128
rng = np.random.default_rng(42)

with PistaDB("mydb.pst", dim=dim, metric=Metric.COSINE, index=Index.HNSW) as db:
    for i in range(1, 6):
        db.insert(i, rng.random(dim).astype("float32"), label=f"doc_{i}")

    # Atomic batch: all-or-nothing
    with db.begin_transaction() as txn:
        txn.insert(10, rng.random(dim).astype("float32"), label="new_doc")
        txn.delete(3)
        txn.update(1, rng.random(dim).astype("float32"))
    # All three operations are now visible; none were visible before commit

    # Automatic rollback on exception
    try:
        with db.begin_transaction() as txn:
            txn.insert(20, rng.random(dim).astype("float32"), label="maybe")
            raise ValueError("something went wrong")
    except ValueError:
        pass  # txn rolled back — id 20 was never inserted
```

### C API

```c
#include "pistadb_txn.h"

PistaDBTxn *txn = pistadb_txn_begin(db);

pistadb_txn_insert(txn, 101, "doc_a", vec_a);
pistadb_txn_insert(txn, 102, "doc_b", vec_b);
pistadb_txn_delete(txn, 55);
pistadb_txn_update(txn, 77, vec_updated);

int rc = pistadb_txn_commit(txn);
if (rc == PISTADB_OK) {
    /* all operations applied */
} else if (rc == PISTADB_ETXN_PARTIAL) {
    /* commit failed AND rollback could not fully undo (e.g. IVF-PQ) */
    fprintf(stderr, "partial failure: %s\n", pistadb_txn_last_error(txn));
} else {
    /* commit failed, full rollback succeeded */
    fprintf(stderr, "rolled back: %s\n", pistadb_txn_last_error(txn));
}

pistadb_txn_free(txn);
```

**Atomicity model:**

| Phase | What happens |
|---|---|
| Staging | Operations validated locally (duplicate INSERT id check) |
| Commit phase 1 | Structural validation (no duplicate ids across staged inserts) |
| Commit phase 2 | Operations applied sequentially; undo snapshots captured at staging time |
| Rollback | On failure at index `i`, ops `i-1 … 0` are undone in reverse order |

> **Note on IVF-PQ / ScaNN:** These index types do not store raw vectors (only PQ codes). A staged DELETE or UPDATE captures no undo vector. If commit fails and a PQ-only undo is required, the function returns `PISTADB_ETXN_PARTIAL = -10`, indicating the database is in a partially-applied state that cannot be fully reversed automatically.

---

## Multi-Threaded Batch Insert

For high-throughput embedding pipelines, PistaDB provides a **thread-pool + ring-buffer** batch insert API that decouples vector generation from index writes.

```
Producer thread 0 ──▶ ┌──────────────────┐
Producer thread 1 ──▶ │  Ring-buffer     │──▶ Worker 0 ─┐
Producer thread N ──▶ │  work queue      │──▶ Worker 1 ─┤──▶ pistadb_insert()
                       │  (bounded MPMC)  │──▶ Worker M ─┘    (serialised)
                       └──────────────────┘
```

### Streaming API (C)

```c
#include "pistadb_batch.h"

// Create a batch context: 4 workers, default queue capacity (4096)
PistaDBBatch *batch = pistadb_batch_create(db, 4, 0);

// Any number of producer threads may call push() concurrently.
// The call blocks only when the queue is full (back-pressure).
pistadb_batch_push(batch, id, label, vec);   // thread-safe

// Wait for all queued items to finish
int errors = pistadb_batch_flush(batch);     // 0 on full success

pistadb_batch_destroy(batch);   // flush + shutdown workers + free memory
```

### Offline Bulk API (C)

```c
// ids[n], labels[n] (may be NULL), vecs[n × dim]
// 0 workers → auto-detect hardware_concurrency
int errors = pistadb_batch_insert(db, ids, labels, vecs, n, /*n_threads=*/0);
```

---

## Embedding Cache

Embedding APIs (OpenAI, Cohere, local models) are expensive. When the same text appears more than once, the embedding cache eliminates redundant calls automatically.

### C API

```c
PistaDBCache *cache = pistadb_cache_open("embed.pcc", /*dim=*/384, /*max=*/100000);

float vec[384];
if (!pistadb_cache_get(cache, text, vec)) {
    my_model_encode(text, vec);           // ← only called on a miss
    pistadb_cache_put(cache, text, vec);
}

pistadb_cache_save(cache);
pistadb_cache_close(cache);
```

### Python API

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
| Rehash threshold | 75% load factor, doubles bucket count |
| Persistence format | `.pcc` binary — 64-byte header + variable-length entries |
| Thread safety | Single internal mutex (all public functions protected) |
