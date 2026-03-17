#!/usr/bin/env python3
"""
PistaDB Usage Examples
=====================
Demonstrates all major features:
  - All 5 distance metrics
  - All 7 index algorithms (including ScaNN)
  - CRUD operations
  - Persistence (save / load)
  - ScaNN: anisotropic vector quantization, two-phase search
  - Embedding cache: LRU persistent cache, CachedEmbedder
  - Multi-threaded batch insert: streaming API + offline bulk API
  - Performance benchmarking

Requirements:
    pip install numpy
    # Build C library first, then:
    pip install -e python/

Run:
    python examples/example.py
"""
import sys
import os
import time
import threading
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pistadb import (
    PistaDB, Metric, Index, Params, build_from_array,
    EmbeddingCache, CachedEmbedder, Transaction,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DIM    = 64
N_VECS = 1000
K      = 10
SEED   = 42
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "example_dbs")
os.makedirs(DB_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: pretty-print results
# ──────────────────────────────────────────────────────────────────────────────

def print_results(results, title="Search results"):
    print(f"\n  {title}:")
    for i, r in enumerate(results[:5]):
        print(f"    [{i+1}] id={r.id:4d}  dist={r.distance:.5f}  label={r.label!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Example 1: Quick-start with HNSW + L2
# ──────────────────────────────────────────────────────────────────────────────

def example_quickstart():
    print("\n" + "="*60)
    print("Example 1: Quick-start (HNSW + L2)")
    print("="*60)

    path = os.path.join(DB_DIR, "quickstart.pst")

    # Create database
    params = Params(hnsw_M=16, hnsw_ef_construction=200, hnsw_ef_search=50)
    db = PistaDB(path, dim=DIM, metric=Metric.L2, index=Index.HNSW, params=params)
    print(f"  Created: {db}")

    # Generate and insert vectors
    vecs = rng.random((N_VECS, DIM), dtype=np.float32)
    t0 = time.perf_counter()
    for i, v in enumerate(vecs):
        db.insert(i + 1, v, label=f"item_{i}")
    t_insert = time.perf_counter() - t0
    print(f"  Inserted {N_VECS} vectors in {t_insert*1000:.1f} ms "
          f"({N_VECS/t_insert:.0f} vec/s)")

    # Search
    query = rng.random(DIM, dtype=np.float32)
    t0 = time.perf_counter()
    results = db.search(query, k=K)
    t_search = time.perf_counter() - t0
    print(f"  Search (k={K}) in {t_search*1000:.3f} ms")
    print_results(results)

    # Save and reload
    db.save()
    db.close()
    print(f"\n  Saved to {path}")

    # Reload
    db2 = PistaDB(path, dim=DIM)
    print(f"  Reloaded: {db2}")
    results2 = db2.search(query, k=K)
    print(f"  Top-1 after reload: id={results2[0].id}  "
          f"(was {results[0].id}) – match={results[0].id == results2[0].id}")
    db2.close()


# ──────────────────────────────────────────────────────────────────────────────
# Example 2: All distance metrics
# ──────────────────────────────────────────────────────────────────────────────

def example_metrics():
    print("\n" + "="*60)
    print("Example 2: All distance metrics (LINEAR index)")
    print("="*60)

    vecs = rng.random((200, DIM), dtype=np.float32)

    for metric in Metric:
        path = os.path.join(DB_DIR, f"metric_{metric.name}.pst")
        with PistaDB(path, dim=DIM, metric=metric, index=Index.LINEAR) as db:
            for i, v in enumerate(vecs):
                db.insert(i + 1, v)
            q = rng.random(DIM, dtype=np.float32)
            res = db.search(q, k=3)
            print(f"  {metric.name:8s}  top-1: id={res[0].id:4d}  "
                  f"dist={res[0].distance:.5f}")


# ──────────────────────────────────────────────────────────────────────────────
# Example 3: All index algorithms
# ──────────────────────────────────────────────────────────────────────────────

def example_all_indexes():
    print("\n" + "="*60)
    print("Example 3: All index algorithms")
    print("="*60)

    n      = 500
    dim    = 32
    vecs   = rng.random((n, dim), dtype=np.float32)
    query  = rng.random(dim, dtype=np.float32)

    index_configs = [
        (Index.LINEAR,  Params(), False),
        (Index.HNSW,    Params(hnsw_M=12, hnsw_ef_construction=80, hnsw_ef_search=40), False),
        (Index.IVF,     Params(ivf_nlist=20, ivf_nprobe=4), True),
        (Index.IVF_PQ,  Params(ivf_nlist=10, ivf_nprobe=3, pq_M=4, pq_nbits=4), True),
        (Index.DISKANN, Params(diskann_R=16, diskann_L=50, diskann_alpha=1.2), False),
        (Index.LSH,     Params(lsh_L=8, lsh_K=6, lsh_w=4.0), False),
    ]

    # Ground truth from linear scan
    dists_gt = np.sqrt(((vecs - query) ** 2).sum(axis=1))
    gt_ids   = set(np.argsort(dists_gt)[:K] + 1)

    print(f"  {'Index':10s} {'Build':>8s} {'Search':>8s} {'Recall@10':>10s}")
    print("  " + "-" * 45)

    for index, params, needs_train in index_configs:
        path = os.path.join(DB_DIR, f"idx_{index.name}.pst")
        db   = PistaDB(path, dim=dim, metric=Metric.L2, index=index, params=params)

        if needs_train:
            db.train(vecs)

        t0 = time.perf_counter()
        for i, v in enumerate(vecs):
            db.insert(i + 1, v)
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        res = db.search(query, k=K)
        t_search = time.perf_counter() - t0

        found_ids = {r.id for r in res}
        recall    = len(found_ids & gt_ids) / K

        print(f"  {index.name:10s} {t_build*1000:7.1f}ms {t_search*1000:7.2f}ms "
              f"  {recall:8.1%}")
        db.close()


# ──────────────────────────────────────────────────────────────────────────────
# Example 4: CRUD operations
# ──────────────────────────────────────────────────────────────────────────────

def example_crud():
    print("\n" + "="*60)
    print("Example 4: CRUD operations")
    print("="*60)

    path = os.path.join(DB_DIR, "crud.pst")
    db   = PistaDB(path, dim=DIM, index=Index.LINEAR)

    # Insert
    animals = {1: "cat", 2: "dog", 3: "fox", 4: "bear"}
    vecs = rng.random((4, DIM), dtype=np.float32)
    for id_, name in animals.items():
        db.insert(id_, vecs[id_ - 1], label=name)
    print(f"  Inserted: {list(animals.values())}")
    print(f"  Count: {db.count}")

    # Search near 'dog'
    res = db.search(vecs[1], k=4)
    print(f"  Nearest to 'dog': {[r.label for r in res]}")

    # Update 'fox' → push it far away
    db.update(3, np.ones(DIM, dtype=np.float32) * 99)
    res = db.search(vecs[1], k=4)
    print(f"  After moving 'fox': {[r.label for r in res[:3]]}")

    # Delete 'bear'
    db.delete(4)
    print(f"  After deleting 'bear': count={db.count}")
    res = db.search(vecs[1], k=4)
    print(f"  Results (no bear): {[r.label for r in res]}")

    # Get by id
    vec, label = db.get(1)
    print(f"  get(id=1) → label={label!r}, vec[:3]={vec[:3]}")

    db.close()


# ──────────────────────────────────────────────────────────────────────────────
# Example 5: Cosine similarity (semantic search demo)
# ──────────────────────────────────────────────────────────────────────────────

def example_cosine_semantic():
    print("\n" + "="*60)
    print("Example 5: Cosine similarity – semantic search (simulated)")
    print("="*60)

    # Simulate 3 clusters of "topic vectors"
    n_per_topic = 100
    topics = ["science", "sports", "music"]
    topic_vecs = [rng.random(DIM, dtype=np.float32) for _ in topics]

    path = os.path.join(DB_DIR, "cosine.pst")
    db   = PistaDB(path, dim=DIM, metric=Metric.COSINE, index=Index.HNSW)

    vec_id = 1
    for ti, (name, base) in enumerate(zip(topics, topic_vecs)):
        for j in range(n_per_topic):
            v = base + rng.random(DIM, dtype=np.float32) * 0.2
            v = (v / np.linalg.norm(v)).astype(np.float32)
            db.insert(vec_id, v, label=f"{name}_{j}")
            vec_id += 1

    # Query with a vector close to "sports"
    query = topic_vecs[1] + rng.random(DIM, dtype=np.float32) * 0.1
    query = (query / np.linalg.norm(query)).astype(np.float32)

    res = db.search(query, k=10)
    topic_counts = {t: 0 for t in topics}
    for r in res:
        for t in topics:
            if r.label.startswith(t):
                topic_counts[t] += 1
    print(f"  Query close to 'sports':")
    for t, c in topic_counts.items():
        print(f"    {t:10s}: {c}/10 results")
    db.close()


# ──────────────────────────────────────────────────────────────────────────────
# Example 6: IVF with separate train / insert workflow
# ──────────────────────────────────────────────────────────────────────────────

def example_ivf():
    print("\n" + "="*60)
    print("Example 6: IVF – train then insert workflow")
    print("="*60)

    n_train = 2000
    n_index = 500
    dim     = 64

    train_vecs = rng.random((n_train, dim), dtype=np.float32)
    index_vecs = rng.random((n_index, dim), dtype=np.float32)
    query      = rng.random(dim, dtype=np.float32)

    params = Params(ivf_nlist=50, ivf_nprobe=8)
    path   = os.path.join(DB_DIR, "ivf_workflow.pst")

    db = PistaDB(path, dim=dim, metric=Metric.L2, index=Index.IVF, params=params)

    # Step 1: train on representative data
    t0 = time.perf_counter()
    db.train(train_vecs)
    print(f"  Training ({n_train} vecs): {(time.perf_counter()-t0)*1000:.1f} ms")

    # Step 2: insert actual data
    t0 = time.perf_counter()
    for i, v in enumerate(index_vecs):
        db.insert(i + 1, v, label=f"doc_{i}")
    print(f"  Inserting ({n_index} vecs): {(time.perf_counter()-t0)*1000:.1f} ms")

    # Step 3: search
    t0 = time.perf_counter()
    res = db.search(query, k=10)
    print(f"  Search: {(time.perf_counter()-t0)*1000:.3f} ms")
    print_results(res, "IVF top-5")

    db.save()
    db.close()


# ──────────────────────────────────────────────────────────────────────────────
# Example 7: Build from NumPy array (batch API)
# ──────────────────────────────────────────────────────────────────────────────

def example_batch_api():
    print("\n" + "="*60)
    print("Example 7: Batch build from NumPy array")
    print("="*60)

    vecs   = rng.random((500, DIM), dtype=np.float32)
    labels = [f"item_{i}" for i in range(500)]
    path   = os.path.join(DB_DIR, "batch.pst")

    db = build_from_array(path, vecs, labels=labels,
                           metric=Metric.L2, index=Index.HNSW)
    print(f"  {db}")
    q   = rng.random(DIM, dtype=np.float32)
    res = db.search(q, k=5)
    print_results(res)
    db.save()
    db.close()


# ──────────────────────────────────────────────────────────────────────────────
# Example 8: DiskANN graph build
# ──────────────────────────────────────────────────────────────────────────────

def example_diskann():
    print("\n" + "="*60)
    print("Example 8: DiskANN (Vamana) graph index")
    print("="*60)

    n   = 300
    dim = 48
    vecs = rng.random((n, dim), dtype=np.float32)
    q    = rng.random(dim, dtype=np.float32)

    params = Params(diskann_R=24, diskann_L=60, diskann_alpha=1.2)
    path   = os.path.join(DB_DIR, "diskann.pst")

    with PistaDB(path, dim=dim, metric=Metric.L2,
                index=Index.DISKANN, params=params) as db:
        t0 = time.perf_counter()
        for i, v in enumerate(vecs):
            db.insert(i + 1, v)
        print(f"  Online insert ({n} vecs): {(time.perf_counter()-t0)*1000:.1f} ms")

        # Optional: trigger a full Vamana rebuild pass
        t0 = time.perf_counter()
        db.train()  # calls diskann_build
        print(f"  Graph rebuild: {(time.perf_counter()-t0)*1000:.1f} ms")

        res = db.search(q, k=5)
        print_results(res, "DiskANN top-5")
        db.save()


# ──────────────────────────────────────────────────────────────────────────────
# Example 9: ScaNN – Anisotropic Vector Quantization
# ──────────────────────────────────────────────────────────────────────────────

def example_scann():
    print("\n" + "="*60)
    print("Example 9: ScaNN – Anisotropic Vector Quantization")
    print("="*60)
    print("  Two-phase search: fast ADC scoring → exact reranking")

    # ScaNN shines on cosine/IP workloads with large datasets.
    # dim must be divisible by scann_pq_M.
    # Train on the same vectors that will be indexed (same distribution)
    # so that centroids and PQ codebooks are well-calibrated.
    n_index = 500
    dim     = 32   # 32 / 8 = 4 floats per PQ sub-space

    raw_vecs   = rng.random((n_index, dim), dtype=np.float32)
    # L2-normalise → cosine workload (simulates real embedding outputs)
    norms      = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    index_vecs = (raw_vecs / norms).astype(np.float32)
    query      = rng.random(dim, dtype=np.float32)
    query      = (query / np.linalg.norm(query)).astype(np.float32)

    # Ground-truth top-K (exact cosine, i.e. highest dot product)
    dots   = index_vecs @ query
    gt_ids = set(np.argsort(-dots)[:K] + 1)

    # ── Parameter sweep: nprobe → recall vs. latency ──────────────────────
    print(f"\n  {'nprobe':>8s}  {'rerank_k':>9s}  {'latency':>10s}  {'recall@10':>10s}")
    print("  " + "-" * 48)
    for nprobe in (2, 4, 8, 16):
        rerank_k = max(K * 5, nprobe * 8)
        params   = Params(
            scann_nlist    = 16,        # coarse IVF partitions
            scann_nprobe   = nprobe,    # partitions probed per query
            scann_pq_M     = 8,         # PQ sub-spaces
            scann_pq_bits  = 8,         # 8-bit sub-codes
            scann_rerank_k = rerank_k,  # exact reranking candidates
            scann_aq_eta   = 0.2,       # anisotropic penalty η
        )
        path_s = os.path.join(DB_DIR, f"scann_np{nprobe}.pst")
        db_s   = PistaDB(path_s, dim=dim, metric=Metric.COSINE,
                         index=Index.SCANN, params=params)
        # Train on the same normalised corpus
        db_s.train(index_vecs)
        for i, v in enumerate(index_vecs):
            db_s.insert(i + 1, v, label=f"doc_{i}")
        t0  = time.perf_counter()
        res = db_s.search(query, k=K)
        lat = (time.perf_counter() - t0) * 1000
        rec = len({r.id for r in res} & gt_ids) / K
        db_s.close()
        print(f"  {nprobe:>8d}  {rerank_k:>9d}  {lat:>9.3f}ms  {rec:>9.1%}")

    # ── Full example with best params ─────────────────────────────────────
    print(f"\n  Full run (nprobe=16, rerank_k=100):")
    params = Params(scann_nlist=16, scann_nprobe=16,
                    scann_pq_M=8, scann_pq_bits=8,
                    scann_rerank_k=100, scann_aq_eta=0.2)
    path = os.path.join(DB_DIR, "scann.pst")
    db   = PistaDB(path, dim=dim, metric=Metric.COSINE,
                   index=Index.SCANN, params=params)

    t0 = time.perf_counter()
    db.train(index_vecs)
    print(f"  Training ({n_index} vecs): {(time.perf_counter()-t0)*1000:.1f} ms")

    t0 = time.perf_counter()
    for i, v in enumerate(index_vecs):
        db.insert(i + 1, v, label=f"doc_{i}")
    t_insert = time.perf_counter() - t0
    print(f"  Insert  ({n_index} vecs): {t_insert*1000:.1f} ms  "
          f"({n_index/t_insert:.0f} vec/s)")

    t0      = time.perf_counter()
    results = db.search(query, k=K)
    t_search = time.perf_counter() - t0
    recall   = len({r.id for r in results} & gt_ids) / K
    print(f"  Search  (k={K}): {t_search*1000:.3f} ms  |  recall@{K}={recall:.1%}")
    print_results(results, "ScaNN top-5 (cosine)")

    db.save()
    db.close()


# ──────────────────────────────────────────────────────────────────────────────
# Example 10: Embedding cache
# ──────────────────────────────────────────────────────────────────────────────

def example_embedding_cache():
    print("\n" + "="*60)
    print("Example 10: Embedding Cache (LRU + persistent .pcc file)")
    print("="*60)
    print("  Avoids re-encoding identical texts; survives restarts.")

    cache_path = os.path.join(DB_DIR, "embed_cache.pcc")
    dim        = DIM

    # ── Simulate an expensive embedding model ─────────────────────────────
    call_count = [0]
    def fake_embed(text: str) -> np.ndarray:
        """Deterministic 'embedding' based on text hash; 1 ms simulated cost."""
        call_count[0] += 1
        seed = hash(text) & 0xFFFF_FFFF
        v    = np.random.default_rng(seed).random(dim, dtype=np.float32)
        time.sleep(0.001)   # simulate network / GPU latency
        return v

    # ── Part A: basic get / put ────────────────────────────────────────────
    print("\n  [A] Basic get / put")
    cache = EmbeddingCache(cache_path, dim=dim, max_entries=1000)

    texts = [
        "PistaDB stores vectors locally with zero dependencies.",
        "HNSW delivers sub-millisecond approximate nearest-neighbour search.",
        "RAG improves LLM output by injecting retrieved context.",
        "ScaNN uses anisotropic quantization for high-recall cosine search.",
        "The embedding cache deduplicates repeated model calls automatically.",
    ]

    # First pass – all misses, model is called
    t0 = time.perf_counter()
    for t in texts:
        vec = cache.get(t)
        if vec is None:
            vec = fake_embed(t)
            cache.put(t, vec)
    t_first = time.perf_counter() - t0
    s = cache.stats()
    print(f"  First pass  (all misses): {t_first*1000:.1f} ms  "
          f"| hits={s.hits}  misses={s.misses}  model_calls={call_count[0]}")

    # Second pass – all hits, model is NOT called
    call_before = call_count[0]
    t0 = time.perf_counter()
    for t in texts:
        vec = cache.get(t)
        assert vec is not None, f"Expected a cache hit for: {t!r}"
    t_second = time.perf_counter() - t0
    s = cache.stats()
    print(f"  Second pass (all hits) : {t_second*1000:.1f} ms  "
          f"| hits={s.hits}  misses={s.misses}  model_calls={call_count[0]-call_before}")
    print(f"  Speedup: {t_first/t_second:.0f}x  |  hit_rate={s.hit_rate:.1%}")

    # ── Part B: persistence across restarts ───────────────────────────────
    print("\n  [B] Persistence – save, re-open, verify hits")
    cache.save()
    cache.close()
    print(f"  Saved {len(texts)} entries to {cache_path}")

    cache2 = EmbeddingCache(cache_path, dim=dim, max_entries=1000)
    hit_count = sum(1 for t in texts if cache2.get(t) is not None)
    print(f"  After reload: {hit_count}/{len(texts)} entries recovered")
    cache2.close()

    # ── Part C: LRU eviction ──────────────────────────────────────────────
    print("\n  [C] LRU eviction (capacity=5)")
    cap = 5
    small_cache = EmbeddingCache(
        os.path.join(DB_DIR, "lru_demo.pcc"), dim=dim, max_entries=cap)

    print(f"  Inserting 8 entries into a cache of capacity {cap}:")
    for i in range(8):
        v = np.random.default_rng(i).random(dim, dtype=np.float32)
        small_cache.put(f"entry_{i}", v)
        s = small_cache.stats()
        print(f"    put entry_{i}  → count={s.count}  evictions={s.evictions}")

    # Access entry_7 (MRU) and entry_6 to make them hot
    small_cache.get("entry_7")
    small_cache.get("entry_6")
    # Insert one more – LRU entries (3, 4, 5) have already been evicted
    small_cache.put("entry_8", np.ones(dim, dtype=np.float32))
    print(f"  entry_7 still in cache: {small_cache.contains('entry_7')}")
    print(f"  entry_0 still in cache: {small_cache.contains('entry_0')}")
    small_cache.close()

    # ── Part D: CachedEmbedder high-level wrapper ─────────────────────────
    print("\n  [D] CachedEmbedder – transparent cache wrapper")
    call_count[0] = 0
    fresh_cache = EmbeddingCache(
        os.path.join(DB_DIR, "embedder.pcc"), dim=dim, max_entries=500)
    embedder = CachedEmbedder(fake_embed, fresh_cache, autosave_every=3)

    corpus = [
        "neural information retrieval",
        "dense passage retrieval with BERT",
        "approximate nearest neighbour algorithms",
        "neural information retrieval",     # duplicate – should hit cache
        "product quantization for fast ANN",
        "dense passage retrieval with BERT", # duplicate
    ]

    print(f"  Encoding {len(corpus)} texts ({len(set(corpus))} unique):")
    t0 = time.perf_counter()
    vecs = embedder.embed_batch(corpus)
    t_total = time.perf_counter() - t0
    s = fresh_cache.stats()
    print(f"    model_calls={call_count[0]}  hits={s.hits}  "
          f"misses={s.misses}  hit_rate={s.hit_rate:.1%}")
    print(f"    batch shape: {vecs.shape}  time: {t_total*1000:.1f} ms")
    print(f"    autosave triggered: {os.path.exists(embedder.cache._path)}")

    # Build a searchable database from the unique embeddings
    db_path   = os.path.join(DB_DIR, "cached_rag.pst")
    unique    = list(set(corpus))
    u_vecs    = np.stack([embedder(t) for t in unique])
    with PistaDB(db_path, dim=dim, metric=Metric.COSINE,
                 index=Index.HNSW) as db:
        for i, (t, v) in enumerate(zip(unique, u_vecs)):
            db.insert(i + 1, v, label=t[:64])
        q_vec = embedder("information retrieval methods")
        res   = db.search(q_vec, k=3)
    print(f"\n  RAG query: 'information retrieval methods'")
    print_results(res, "Cosine top-3 from cached embeddings")

    fresh_cache.close()


# ──────────────────────────────────────────────────────────────────────────────
# Example 11: Multi-threaded batch insert
# ──────────────────────────────────────────────────────────────────────────────

def example_multithreaded_batch():
    print("\n" + "="*60)
    print("Example 11: Multi-threaded Batch Insert")
    print("="*60)
    print("  Thread pool + bounded queue; producers run concurrently with indexing.")

    # We compare three strategies for inserting N_BATCH vectors:
    #   (a) single-threaded sequential insert
    #   (b) offline bulk API   – pistadb_batch_insert (auto thread-count)
    #   (c) streaming API      – multiple Python threads push concurrently

    N_BATCH = 2000
    dim     = DIM
    vecs    = rng.random((N_BATCH, dim), dtype=np.float32)
    ids     = list(range(1, N_BATCH + 1))

    # ── (a) Baseline: single-threaded sequential ──────────────────────────
    print(f"\n  [A] Single-threaded sequential insert ({N_BATCH} vecs)")
    path_a = os.path.join(DB_DIR, "batch_sequential.pst")
    db_a   = PistaDB(path_a, dim=dim, metric=Metric.L2, index=Index.HNSW)
    t0     = time.perf_counter()
    for i, v in enumerate(vecs):
        db_a.insert(ids[i], v, label=f"v{i}")
    t_seq = time.perf_counter() - t0
    print(f"    Inserted {N_BATCH} vectors in {t_seq*1000:.1f} ms  "
          f"({N_BATCH/t_seq:.0f} vec/s)")
    db_a.save()
    db_a.close()

    # ── (b) Offline bulk: build_from_array (uses single-pass insert) ──────
    print(f"\n  [B] build_from_array bulk helper ({N_BATCH} vecs)")
    path_b  = os.path.join(DB_DIR, "batch_bulk.pst")
    labels  = [f"v{i}" for i in range(N_BATCH)]
    t0      = time.perf_counter()
    db_b    = build_from_array(path_b, vecs, ids=ids, labels=labels,
                                metric=Metric.L2, index=Index.HNSW)
    t_bulk  = time.perf_counter() - t0
    print(f"    Inserted {N_BATCH} vectors in {t_bulk*1000:.1f} ms  "
          f"({N_BATCH/t_bulk:.0f} vec/s)  count={db_b.count}")
    db_b.save()
    db_b.close()

    # ── (c) Streaming: multiple Python producer threads ───────────────────
    # PistaDB inserts are not thread-safe; the typical pattern is to have
    # worker threads do the expensive CPU/IO work (e.g., embedding) and then
    # serialize the actual insert with a lock.
    n_cpus = os.cpu_count() or 4
    print(f"\n  [C] Streaming API – {n_cpus} producer threads ({N_BATCH} vecs)")
    path_c      = os.path.join(DB_DIR, "batch_streaming.pst")
    db_c        = PistaDB(path_c, dim=dim, metric=Metric.L2, index=Index.HNSW)
    insert_lock = threading.Lock()

    # Chunk the work across producer threads
    n_producers = min(n_cpus, 8)
    chunk_size  = N_BATCH // n_producers
    push_times  = []

    def producer(start: int, end: int):
        t_prod = time.perf_counter()
        for i in range(start, end):
            # Simulate per-item work (e.g., embedding) done in parallel,
            # then acquire the lock only for the actual insert.
            vec_i = vecs[i]          # parallel "work"
            with insert_lock:
                db_c.insert(ids[i], vec_i, label=f"v{i}")
        push_times.append(time.perf_counter() - t_prod)

    t0      = time.perf_counter()
    threads = []
    for p in range(n_producers):
        s = p * chunk_size
        e = s + chunk_size if p < n_producers - 1 else N_BATCH
        t = threading.Thread(target=producer, args=(s, e))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    t_stream = time.perf_counter() - t0

    print(f"    {n_producers} producer threads finished in {t_stream*1000:.1f} ms  "
          f"({N_BATCH/t_stream:.0f} vec/s)  count={db_c.count}")
    print(f"    Per-producer wall time: "
          f"min={min(push_times)*1000:.1f}ms  "
          f"max={max(push_times)*1000:.1f}ms")

    db_c.save()
    db_c.close()

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n  {'Strategy':30s} {'Time':>10s} {'vec/s':>10s}")
    print("  " + "-" * 55)
    for name, t in [("(A) Sequential",      t_seq),
                    ("(B) build_from_array", t_bulk),
                    ("(C) Streaming threads", t_stream)]:
        print(f"  {name:30s} {t*1000:>9.1f}ms {N_BATCH/t:>9.0f}")

    # ── (d) Combine batch insert with the embedding cache ─────────────────
    print(f"\n  [D] Cache-assisted pipeline: deduplicate → batch-insert")
    docs = [
        "vector databases enable semantic search",
        "HNSW is the fastest graph-based ANN index",
        "vector databases enable semantic search",   # duplicate
        "ScaNN uses product quantization and reranking",
        "HNSW is the fastest graph-based ANN index", # duplicate
        "RAG combines retrieval with generation",
        "embedding caches reduce API costs significantly",
        "RAG combines retrieval with generation",    # duplicate
    ]

    call_counter = [0]
    def cheap_embed(text):
        call_counter[0] += 1
        seed = hash(text) & 0xFFFF_FFFF
        return np.random.default_rng(seed).random(dim, dtype=np.float32)

    dedup_cache = EmbeddingCache(
        os.path.join(DB_DIR, "dedup.pcc"), dim=dim)
    embedder    = CachedEmbedder(cheap_embed, dedup_cache)

    # Embed with deduplication
    unique_texts, unique_vecs, unique_ids = [], [], []
    seen = {}
    for doc in docs:
        if doc not in seen:
            v = embedder(doc)
            seen[doc] = len(unique_texts) + 1
            unique_texts.append(doc)
            unique_vecs.append(v)
            unique_ids.append(seen[doc])

    print(f"    {len(docs)} docs → {len(unique_texts)} unique  "
          f"(model_calls={call_counter[0]}  "
          f"hits={dedup_cache.stats().hits})")

    # Batch-insert only the unique vectors
    path_d = os.path.join(DB_DIR, "dedup_index.pst")
    db_d   = build_from_array(
        path_d,
        np.stack(unique_vecs),
        ids    = unique_ids,
        labels = [t[:64] for t in unique_texts],
        metric = Metric.COSINE,
        index  = Index.HNSW,
    )
    q_text = "semantic search with dense vectors"
    q_vec  = embedder(q_text)
    res    = db_d.search(q_vec, k=3)
    print(f"\n    Query: {q_text!r}")
    print_results(res, "Top-3 from dedup index")
    db_d.save()
    db_d.close()
    dedup_cache.close()


# ──────────────────────────────────────────────────────────────────────────────
# Example 12: Transactions
# ──────────────────────────────────────────────────────────────────────────────

def example_transactions():
    print("=" * 60)
    print("Example 12: Transactions")
    print("=" * 60)
    print("  Atomic groups of INSERT / DELETE / UPDATE operations.\n")

    dim  = 32
    path = os.path.join(DB_DIR, "txn_demo.pst")
    if os.path.exists(path):
        os.remove(path)
    db   = PistaDB(path, dim=dim, metric=Metric.L2, index=Index.LINEAR)
    rng2 = np.random.default_rng(7)

    def rand_vec():
        return rng2.random(dim, dtype=np.float64).astype(np.float32)

    # ── (A) Basic atomic batch insert ─────────────────────────────────────
    print("  [A] Atomic batch insert (context-manager form)")
    t0 = time.perf_counter()
    with db.begin_transaction() as txn:
        for i in range(1, 101):
            txn.insert(i, rand_vec(), label=f"doc_{i}")
    t_txn = time.perf_counter() - t0
    print(f"    Committed 100 inserts atomically in {t_txn*1000:.1f} ms")
    print(f"    count={db.count}")

    # ── (B) Rollback on exception ──────────────────────────────────────────
    print("\n  [B] Rollback on exception — count stays unchanged")
    count_before = db.count
    try:
        with db.begin_transaction() as txn:
            txn.insert(200, rand_vec(), label="new_doc")
            txn.insert(201, rand_vec(), label="new_doc_2")
            raise ValueError("simulated downstream failure")
    except ValueError:
        pass
    count_after = db.count
    print(f"    Before: {count_before}  After: {count_after}  "
          f"({'unchanged' if count_before == count_after else 'CHANGED (bug!)'})")

    # ── (C) Mixed: insert + delete + update atomically ────────────────────
    print("\n  [C] Mixed transaction: insert 5, delete id=1, update id=2")
    v_new = rand_vec()
    with db.begin_transaction() as txn:
        for i in range(101, 106):            # insert 5 more docs
            txn.insert(i, rand_vec(), label=f"doc_{i}")
        txn.delete(1)                         # remove id=1
        txn.update(2, v_new)                  # replace id=2 vector

    got_vec, _ = db.get(2)
    print(f"    count={db.count}  (was 100, +5 inserts -1 delete = 104)")
    print(f"    id=2 vector updated: {np.allclose(got_vec, v_new, atol=1e-5)}")
    with_id1 = True
    try:
        db.get(1)
    except RuntimeError:
        with_id1 = False
    print(f"    id=1 deleted: {not with_id1}")

    # ── (D) Failed commit with automatic rollback ─────────────────────────
    print("\n  [D] Commit failure with rollback")
    count_before = db.count
    try:
        txn = db.begin_transaction()
        txn.insert(999, rand_vec(), label="will_be_undone")   # succeeds in apply
        txn.delete(9999)                                       # fails: id doesn't exist
        txn.commit()
    except RuntimeError as exc:
        print(f"    Commit failed as expected: {exc}")
        txn.free()
    count_after = db.count
    print(f"    id=999 rolled back: count {count_before} -> {count_after} "
          f"({'OK' if count_before == count_after else 'rollback incomplete'})")

    # ── (E) Manual form + reuse ────────────────────────────────────────────
    print("\n  [E] Manual commit/rollback + handle reuse")
    txn = db.begin_transaction()
    txn.insert(500, rand_vec(), label="first_use")
    print(f"    Staged {txn.op_count} op(s)")
    txn.commit()
    print(f"    After first commit: count={db.count}")
    # Reuse the same handle
    txn.insert(501, rand_vec(), label="second_use")
    txn.commit()
    txn.free()
    print(f"    After second commit: count={db.count}")

    # ── (F) Data integrity demo: atomic document index update ─────────────
    print("\n  [F] Data integrity: atomic 'swap' of two document embeddings")
    v_a = rand_vec()
    v_b = rand_vec()
    db.insert(1000, v_a, label="doc_A")
    db.insert(1001, v_b, label="doc_B")

    with db.begin_transaction() as txn:
        txn.update(1000, v_b)   # swap A <- B's vector
        txn.update(1001, v_a)   # swap B <- A's vector

    got_1000, _ = db.get(1000)
    got_1001, _ = db.get(1001)
    print(f"    Swap verified: 1000 now has B's vec={np.allclose(got_1000, v_b, atol=1e-5)}"
          f"  1001 now has A's vec={np.allclose(got_1001, v_a, atol=1e-5)}")

    db.save()
    db.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("PistaDB", PistaDB.version())
    print(f"Database files will be written to: {os.path.abspath(DB_DIR)}\n")

    example_quickstart()
    example_metrics()
    example_all_indexes()
    example_crud()
    example_cosine_semantic()
    example_ivf()
    example_batch_api()
    example_diskann()
    example_scann()
    example_embedding_cache()
    example_multithreaded_batch()
    example_transactions()

    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
