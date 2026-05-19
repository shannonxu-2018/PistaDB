"""
SQLite-style paging demo / verification — all paged index types.

For each of LINEAR / IVF / LSH / HNSW / DISKANN it builds a db once, then
opens it in two fresh subprocesses — resident (default) and paged
(PISTADB_PAGED=1, small cache) — and checks:
  * search results are bit-identical between the two modes
  * paged resident memory is well below resident-mode memory
  * inserts are rejected in paged (read-only) mode

Usage:  python scripts/paging_demo.py            (all indices)
        python scripts/paging_demo.py LINEAR     (one)
        python scripts/paging_demo.py search <IDX> <PATH>   (worker)
"""
import os, sys, json, subprocess, tempfile, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "wrap", "python"))
os.environ.setdefault("PISTADB_LIB_DIR", os.path.join(HERE, "..", "build", "Release"))

N, DIM, NQ, K = 25_000, 64, 20, 10
CACHE = 4 * 1024 * 1024


def working_set_mb():
    import ctypes, ctypes.wintypes as w
    class PMC(ctypes.Structure):
        _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    GetPMI = getattr(ctypes.WinDLL("kernel32"), "K32GetProcessMemoryInfo", None)
    if GetPMI is None:
        GetPMI = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    GetPMI.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), w.DWORD]
    GetPMI.restype  = w.BOOL
    c = PMC(); c.cb = ctypes.sizeof(c)
    if not GetPMI(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return c.WorkingSetSize / 1048576.0, c.PeakWorkingSetSize / 1048576.0


def queries():
    return np.random.default_rng(2024).random((NQ, DIM), dtype=np.float32)


def _index(name):
    from pistadb import Index
    return getattr(Index, name)


def run_search(idx_name, path):
    from pistadb import PistaDB
    db = PistaDB(path, dim=DIM)
    digest = hashlib.sha256()
    for q in queries():
        for r in db.search(q, k=K):
            digest.update(f"{r.id}:{r.distance:.4f};".encode())
    cnt = db.count
    ws, peak = working_set_mb()
    mut = "n/a"
    if os.environ.get("PISTADB_PAGED"):
        try:
            db.insert(10**9, queries()[0]); mut = "NOT rejected (BUG)"
        except Exception:
            mut = "insert rejected"
    db.close()
    print(json.dumps({"rss": round(ws, 1), "peak": round(peak, 1),
                       "count": cnt, "hash": digest.hexdigest()[:16],
                       "mut": mut}))


def build(idx_name, path):
    from pistadb import build_from_array
    rng  = np.random.default_rng(7)
    vecs = rng.random((N, DIM), dtype=np.float32)
    train = idx_name in ("IVF", "IVF_PQ")
    db = build_from_array(path, vecs, index=_index(idx_name),
                          train_first=train)
    db.save(); db.close()
    return os.path.getsize(path) / 1048576.0


def one_index(idx_name):
    path = os.path.join(tempfile.gettempdir(), f"paging_{idx_name}.pst")
    if os.path.exists(path):
        os.remove(path)
    fsz = build(idx_name, path)

    def child(env_extra):
        e = dict(os.environ); e.update(env_extra)
        out = subprocess.check_output(
            [sys.executable, __file__, "search", idx_name, path], env=e)
        return json.loads(out.decode().strip().splitlines()[-1])

    res = child({})
    pag = child({"PISTADB_PAGED": "1",
                 "PISTADB_PAGE_CACHE_BYTES": str(CACHE)})
    os.remove(path)

    same = res["hash"] == pag["hash"] and res["count"] == pag["count"]
    mem  = pag["rss"] < res["rss"] - 3            # clearly lower
    ok   = same and mem and pag["mut"] == "insert rejected"
    print(f"{idx_name:<8}{fsz:7.0f}MB  resident RSS {res['rss']:6.1f} "
          f"peak {res['peak']:6.1f} | paged RSS {pag['rss']:6.1f} "
          f"peak {pag['peak']:6.1f} | identical={same} ro={pag['mut']} "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "search":
        run_search(sys.argv[2], sys.argv[3]); return

    only = sys.argv[1] if len(sys.argv) > 1 else None
    indices = [only] if only else ["LINEAR", "IVF", "LSH", "HNSW", "DISKANN"]
    print(f"N={N} DIM={DIM} cache={CACHE//1024//1024}MB\n")
    results = [one_index(ix) for ix in indices]
    print("\n" + ("ALL PASS" if all(results) else "SOME FAILED"))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
