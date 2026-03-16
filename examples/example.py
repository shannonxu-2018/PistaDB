#!/usr/bin/env python3
"""
PistaDB Usage Examples
=====================
Demonstrates all major features:
  - All 5 distance metrics
  - All 6 index algorithms
  - CRUD operations
  - Persistence (save / load)
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
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pistadb import PistaDB, Metric, Index, Params, build_from_array

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

    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
