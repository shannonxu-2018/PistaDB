---
name: pistadb
description: >
  Use PistaDB — an embedded vector database (single .pst file, native lib +
  Python ctypes wrapper) — for KNN / semantic search / RAG / dedup /
  recommendation features. Trigger whenever the task involves vector
  similarity search, embeddings storage, nearest-neighbour lookup, or an
  on-disk ANN index in a Python project that vendors PistaDB.
---

# PistaDB usage skill

PistaDB is an embedded vector DB: one native library (`.dll`/`.so`/`.dylib`)
plus a pure-Python `ctypes` wrapper package `pistadb`. Storage is a single
`.pst` file. Zero third-party runtime deps (numpy is the only Python dep).

## 0. Setup (assume already done; verify, don't reinvent)

The wrapper finds the native library via env var `PISTADB_LIB_DIR` (a folder
containing `pistadb.dll`/`libpistadb.so`), or the lib sitting inside the
`pistadb/` package folder. If `from pistadb import PistaDB` raises
`OSError: ... shared library not found`, the deployment is incomplete — see
the project's `DEPLOY.md`; do **not** try to build it from this skill.

```python
from pistadb import (
    PistaDB, Metric, Index, Params, SearchResult,
    build_from_array, insert_batch,
    EmbeddingCache, CachedEmbedder, CacheStats,
)
```

## 1. Core API (this is the whole surface — do not invent methods)

```python
db = PistaDB(path, dim, metric=Metric.L2, index=Index.HNSW, params=None)
#   path: str path to the .pst file (created if missing; loaded if it exists)
#   dim:  vector dimension (fixed for the life of the file)
#   Opening an EXISTING file restores its metric/index from the file;
#   `dim` must match the stored dim.

db.insert(id: int, vector, label: str = "")   # id = unique uint64 (>0)
db.delete(id)                                  # logical delete
db.update(id, vector)                          # replace vector for id
vec, label = db.get(id)                        # -> (np.float32[dim], str)
hits = db.search(query, k=10)                  # -> List[SearchResult], asc by distance, len<=k
db.train(sample_vectors)                       # REQUIRED before insert for IVF/IVF_PQ/SCANN
db.save()                                      # persist (atomic). close() does NOT auto-save!
db.close()                                     # free handle

db.count        # int: active (non-deleted) vectors
db.dim          # int
db.metric       # Metric
db.index_type   # Index
db.last_error   # str (last C error message)
PistaDB.version()  # staticmethod -> "1.0.0"

# SearchResult: .id (int), .distance (float), .label (str)
# Always use as a context manager so close() runs:
with PistaDB("data.pst", dim=384, index=Index.HNSW) as db:
    ...
    db.save()        # <-- explicit; required before the `with` block ends
```

Vectors are coerced to contiguous `float32` and must have length == `dim`
(wrong length raises `ValueError`). Pass numpy arrays.

### Bulk build
```python
db = build_from_array(path, vectors,                 # vectors: np.ndarray (n, dim)
                       ids=None, labels=None,         # ids default to 1..n
                       metric=Metric.L2, index=Index.HNSW,
                       params=None, train_first=False)# train_first=True for IVF/IVF_PQ
# Returns an OPEN db — you still must db.save() then db.close().
insert_batch(db, ids, vectors, labels=None)          # convenience loop on an open db
```

## 2. Choosing index + metric

Metric (`Metric.*`): `L2` (Euclidean), `COSINE` (= 1 − cosine sim),
`IP` (= −dot product; smaller = more similar), `L1`, `HAMMING`.
For text embeddings: use `COSINE`, or L2-normalize vectors and use `IP`.

| Index | Train? | Use it when |
|---|---|---|
| `HNSW` | no | **Default.** Best recall/speed, in-RAM graph. Start here. |
| `LINEAR` | no | Exact brute force. Small data / ground-truth / <50k vectors. |
| `IVF` | **yes** | Large N, faster than linear, approximate. Needs `train()`. |
| `IVF_PQ` | **yes** | Huge N, memory-compressed (PQ codes). Lower recall. `get()` unsupported. |
| `SCANN` | **yes** | Compressed + reranked; best approximate cosine/IP. |
| `SQ` | no | uint8-quantized, ~4× smaller than float, near-exact. |
| `DISKANN` | no | Graph index for very large datasets. |
| `LSH` | no | Sublinear, tunable recall via `Params(lsh_L, lsh_K)`. |

Tuning is via `Params(...)` (a dataclass; all fields optional, sane defaults).
Common knobs: `Params(hnsw_M=16, hnsw_ef_construction=200, hnsw_ef_search=50)`,
`Params(ivf_nlist=128, ivf_nprobe=8)`. Pass it as `PistaDB(..., params=Params(...))`.

## 3. Rules & gotchas (these cause real bugs — follow them)

1. **`close()` does NOT save.** Always `db.save()` before the handle goes away,
   or data inserted since open is lost. Use a `with` block + explicit `save()`.
2. **Training is mandatory for IVF / IVF_PQ / SCANN** and must happen **before**
   any `insert()` (`db.train(sample)` where `sample` is a representative
   `np.ndarray (m, dim)`). Insert before train returns an error → `RuntimeError`.
   Not needed for HNSW/LINEAR/DISKANN/LSH/SQ.
3. **ids** are `uint64`, must be unique, and `2**64-1` is reserved (don't use it).
4. **`get(id)` raises for `IVF_PQ`** (only PQ codes stored, no raw vector).
   Works for LINEAR/IVF/LSH/HNSW/DISKANN/SQ/SCANN.
5. `search` returns at most `k`, sorted ascending by `distance`. For `L2` the
   distance is Euclidean; for `IP`/`COSINE` smaller still = closer.
6. `delete` is logical; deleted ids vanish from `search`/`count`; space is
   reclaimed on the next `save()`/rebuild.
7. **One `PistaDB` handle is single-threaded.** Serialize calls or use one
   handle per thread (separate file or read-only paged copies).
8. Errors surface as Python exceptions (`RuntimeError`/`IOError`/`ValueError`);
   `db.last_error` has the C-side message. Wrap user-facing ops in try/except.

## 4. Paged mode — open a huge .pst with bounded RAM

For large prebuilt indexes you only query, open in SQLite-style paged mode so
resident memory is capped (independent of file size) instead of loading the
whole file:

```python
import os
os.environ["PISTADB_PAGED"] = "1"
os.environ["PISTADB_PAGE_CACHE_BYTES"] = str(64 * 1024 * 1024)  # optional, default 64MB
db = PistaDB("huge.pst", dim=768)   # set env BEFORE opening
hits = db.search(q, k=10)           # search/get/delete/save work
```

- Works for `LINEAR/IVF/LSH/HNSW/DISKANN`; reads existing `.pst` unchanged.
- **Read-only**: `insert()`/`update()` raise (`RuntimeError`, EINVAL). To
  modify, open normally (env unset). `SQ/IVF_PQ/SCANN` ignore the flag
  (already compact).
- Single-threaded per handle in paged mode.

## 5. Transactions (atomic multi-op)

```python
with db.begin_transaction() as txn:
    txn.insert(1, v1, label="a")
    txn.update(2, v2)
    txn.delete(3)
# clean exit -> committed atomically; exception -> rolled back and re-raised
```
Validation (e.g. duplicate insert ids) runs before any change. Note: on
`IVF_PQ`, delete/update undo may be unavailable (raises with partial state).

## 6. Embedding cache (memoize expensive model calls)

```python
cache = EmbeddingCache("emb.pcc", dim=384, max_entries=100_000)
embed = CachedEmbedder(model.encode, cache, autosave_every=500)  # model.encode(str)->vec
vec = embed("some text")          # model called only on cache miss
print(cache.stats().hit_rate)
cache.save(); cache.close()
```

## 7. Complete pattern: semantic search / RAG ingestion + query

```python
import numpy as np
from pistadb import PistaDB, Metric, Index

DIM = 384  # match your embedding model's output

def embed(texts):                     # plug in your real model here
    return np.asarray(model.encode(texts), dtype=np.float32)

# ── Ingest ───────────────────────────────────────────────────────────────
docs = [{"id": 1, "text": "..."}, {"id": 2, "text": "..."}]
with PistaDB("corpus.pst", dim=DIM, metric=Metric.COSINE,
             index=Index.HNSW) as db:
    vecs = embed([d["text"] for d in docs])
    for d, v in zip(docs, vecs):
        db.insert(d["id"], v, label=d["text"][:255])   # label = snippet
    db.save()

# ── Query (later / another process) ──────────────────────────────────────
with PistaDB("corpus.pst", dim=DIM) as db:             # metric/index from file
    q = embed(["user question"])[0]
    for r in db.search(q, k=5):
        print(r.id, round(r.distance, 4), r.label)     # label = retrieved text
```

For RAG, store the chunk text in `label` (≤255 bytes) for short chunks, or
store only the id and keep full chunk text in your own store keyed by id.

## 8. When unsure

Verify behaviour with `db.count`, `db.index_type`, `db.last_error`, and a
tiny round-trip (insert → save → reopen → search) before building on top.
Do not assume APIs beyond §1; this is the complete supported surface.
