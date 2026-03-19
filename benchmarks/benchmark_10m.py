#!/usr/bin/env python3
"""
PistaDB 千万级数据 CRUD 基准测试
===================================
Tests INSERT / SEARCH / UPDATE / DELETE at scales up to 9,000,000 records.

Background
----------
Two indices are included:

  HNSW (M=8, ef_c=50, ef_s=32)
      Best recall for approximate nearest-neighbor workloads.
      All insert milestones from 1 M to 9 M measured.
      The 10 M mark triggers an internal capacity doubling from 9.4 M → 18.8 M,
      requiring a single contiguous ~10 GB realloc() that Windows heap cannot
      satisfy when the process already has hundreds of thousands of smaller
      live allocations (graph neighbor-list nodes).

  IVF (nlist=500, nprobe=50)
      Inverted-file index; fast insert (just centroid assignment) and
      competitive recall with appropriate nprobe.
      Complete CRUD benchmark at 9 M records.

Configuration
-------------
  Dimension   : 128  (typical sentence-embedding size, e.g. all-MiniLM-L6-v2)
  Metric      : L2
  HNSW params : M=8, ef_construction=50, ef_search=32
  IVF  params : nlist=500, nprobe=50, training sample=10 K vectors
  Insert batch: 500 K vectors / iteration  (≈ 256 MB RAM per batch)
  Search      : 200 random queries, k=10
  Update      : 100 000 random records
  Delete      : 100 000 random records

Run
---
    python benchmarks/benchmark_10m.py
"""

import sys
import os
import time
import gc
import platform
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pistadb import PistaDB, Metric, Index, Params

# ── Configuration ─────────────────────────────────────────────────────────────
TOTAL     = 9_000_000    # stays within IVF capacity (9,437,176); no realloc OOM
DIM       = 128
BATCH     = 500_000
K         = 10
N_SEARCH  = 200
N_UPDATE  = 100_000
N_DELETE  = 100_000
N_TRAIN   = 10_000       # 20× nlist; fast k-means (~20 s)
SEED      = 42

DB_DIR  = os.path.join(os.path.dirname(__file__), "..", "example_dbs")
DB_PATH = os.path.join(DB_DIR, "bench_9m_ivf.pst")
os.makedirs(DB_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────

def hr(char="─", n=64):
    print(char * n, flush=True)

def fmt_n(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)

def fmt_sec(s):
    if s >= 3600:
        return f"{int(s)//3600}h {(int(s)%3600)//60}m"
    if s >= 60:
        return f"{int(s)//60}m {int(s)%60}s"
    return f"{s:.2f}s"

def sys_ram():
    try:
        import ctypes
        class _MEM(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullExtendedVirtual", ctypes.c_ulonglong)]
        m = _MEM(); m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 1e9, m.ullAvailPhys / 1e9
    except Exception:
        return None, None

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    total_phys, avail_phys = sys_ram()
    mem_note = (f"{total_phys:.0f} GB total / {avail_phys:.0f} GB free"
                if total_phys else "unknown")

    print(flush=True)
    hr("=")
    print("  PistaDB 千万级数据 CRUD 基准测试 — IVF Index", flush=True)
    hr("=")
    print(f"  Records    : {fmt_n(TOTAL)} ({TOTAL:,})", flush=True)
    print(f"  Dimension  : {DIM}", flush=True)
    print(f"  Index      : IVF  (nlist=500  nprobe=50)", flush=True)
    print(f"  Metric     : L2", flush=True)
    print(f"  Platform   : {platform.system()} {platform.machine()}", flush=True)
    print(f"  RAM        : {mem_note}", flush=True)
    print(f"  Python     : {platform.python_version()}", flush=True)
    hr("=")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[Train] Building IVF cluster centroids on {fmt_n(N_TRAIN)} samples...",
          flush=True)
    hr()

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    params = Params(ivf_nlist=500, ivf_nprobe=50)
    db = PistaDB(DB_PATH, dim=DIM, metric=Metric.L2, index=Index.IVF, params=params)

    train_vecs = rng.random((N_TRAIN, DIM), dtype=np.float32)
    t0 = time.perf_counter()
    db.train(train_vecs)
    t_train = time.perf_counter() - t0
    del train_vecs; gc.collect()
    print(f"  Training complete in {fmt_sec(t_train)}", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1: INSERT
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[1/4] INSERT — {fmt_n(TOTAL)} vectors (dim={DIM}, IVF)", flush=True)
    hr()

    milestone_rows = []
    t_insert_wall = time.perf_counter()

    for batch_start in range(0, TOTAL, BATCH):
        batch_end = min(batch_start + BATCH, TOTAL)
        n = batch_end - batch_start

        vecs = rng.random((n, DIM), dtype=np.float32)
        for i in range(n):
            db.insert(batch_start + i + 1, vecs[i])
        del vecs; gc.collect()

        n_inserted = batch_end
        elapsed    = time.perf_counter() - t_insert_wall
        throughput = n_inserted / elapsed

        if n_inserted % 1_000_000 == 0:
            print(f"  {fmt_n(n_inserted):>4s} inserted  |  {fmt_sec(elapsed):>8s} elapsed  |"
                  f"  {throughput:>9,.0f} vec/s", flush=True)
            milestone_rows.append((n_inserted, elapsed, throughput))

    t_insert_total  = time.perf_counter() - t_insert_wall
    avg_insert_tput = TOTAL / t_insert_total

    print(f"\n  Total time    : {fmt_sec(t_insert_total)}", flush=True)
    print(f"  Avg throughput: {avg_insert_tput:,.0f} vec/s", flush=True)
    print(f"  Final count   : {db.count:,}", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 2: SEARCH
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[2/4] SEARCH — {N_SEARCH} queries (k={K}) on {fmt_n(TOTAL)} records",
          flush=True)
    hr()

    queries   = rng.random((N_SEARCH, DIM), dtype=np.float32)
    latencies = []

    for q in queries[:2]:          # warm-up
        db.search(q, k=K)

    for q in queries:
        t0 = time.perf_counter()
        _ = db.search(q, k=K)
        latencies.append((time.perf_counter() - t0) * 1_000)

    lat = np.array(latencies)
    p50 = np.percentile(lat, 50)
    p95 = np.percentile(lat, 95)
    p99 = np.percentile(lat, 99)
    lat_avg = lat.mean()
    qps = 1_000 / lat_avg

    print(f"  Queries  : {N_SEARCH}", flush=True)
    print(f"  p50      : {p50:.3f} ms", flush=True)
    print(f"  p95      : {p95:.3f} ms", flush=True)
    print(f"  p99      : {p99:.3f} ms", flush=True)
    print(f"  avg      : {lat_avg:.3f} ms", flush=True)
    print(f"  min/max  : {lat.min():.3f} ms / {lat.max():.3f} ms", flush=True)
    print(f"  QPS      : {qps:,.0f}", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 3: UPDATE
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[3/4] UPDATE — {fmt_n(N_UPDATE)} random records", flush=True)
    hr()

    update_ids = rng.integers(1, TOTAL + 1, size=N_UPDATE)
    new_vecs   = rng.random((N_UPDATE, DIM), dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(N_UPDATE):
        db.update(int(update_ids[i]), new_vecs[i])
    t_update    = time.perf_counter() - t0
    update_tput = N_UPDATE / t_update

    print(f"  Updated   : {N_UPDATE:,} records", flush=True)
    print(f"  Time      : {fmt_sec(t_update)}", flush=True)
    print(f"  Throughput: {update_tput:,.0f} ops/s", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 4: DELETE
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[4/4] DELETE — {fmt_n(N_DELETE)} random records", flush=True)
    hr()

    delete_ids = list(set(int(x) for x in
                          rng.integers(1, TOTAL + 1, size=N_DELETE * 2)))[:N_DELETE]
    actual_del = len(delete_ids)

    t0 = time.perf_counter()
    for did in delete_ids:
        db.delete(did)
    t_delete    = time.perf_counter() - t0
    delete_tput = actual_del / t_delete

    print(f"  Deleted   : {actual_del:,} records", flush=True)
    print(f"  Time      : {fmt_sec(t_delete)}", flush=True)
    print(f"  Throughput: {delete_tput:,.0f} ops/s", flush=True)
    print(f"  Remaining : {db.count:,} vectors", flush=True)

    db.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────────
    print(flush=True)
    hr("=")
    print("  BENCHMARK SUMMARY", flush=True)
    hr("=")
    print(f"  Config  : dim={DIM}  index=IVF(nlist=500, nprobe=50)  metric=L2", flush=True)
    print(f"  Platform: {platform.system()} {platform.machine()}  RAM {mem_note}", flush=True)
    print(flush=True)
    print(f"  {'Operation':<12} {'Records':>12} {'Time':>10} {'Throughput':>16}", flush=True)
    hr("-", 58)
    print(f"  {'INSERT':<12} {TOTAL:>12,} {fmt_sec(t_insert_total):>10} "
          f"{avg_insert_tput:>14,.0f} vec/s", flush=True)
    print(f"  {'SEARCH':<12} {'200 queries':>12} {'—':>10} "
          f"  p50={p50:.2f} ms  QPS={qps:,.0f}", flush=True)
    print(f"  {'UPDATE':<12} {N_UPDATE:>12,} {fmt_sec(t_update):>10} "
          f"{update_tput:>14,.0f} ops/s", flush=True)
    print(f"  {'DELETE':<12} {actual_del:>12,} {fmt_sec(t_delete):>10} "
          f"{delete_tput:>14,.0f} ops/s", flush=True)
    hr("=")

    print("\n  INSERT milestones:", flush=True)
    print(f"  {'Vectors':>10} {'Elapsed':>10} {'vec/s':>10}", flush=True)
    hr("-", 36)
    for n, el, tp in milestone_rows:
        print(f"  {fmt_n(n):>10} {fmt_sec(el):>10} {tp:>10,.0f}", flush=True)

if __name__ == "__main__":
    main()
