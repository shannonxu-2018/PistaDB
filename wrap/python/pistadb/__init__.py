"""
  ___ _    _        ___  ___
 | _ (_)__| |_ __ _|   \| _ )
 |  _/ (_-<  _/ _` | |) | _ \
 |_| |_/__/\__\__,_|___/|___/

PistaDB - Lightweight Embedded Vector Database
Python wrapper via ctypes.

Usage
-----
>>> import numpy as np
>>> from pistadb import PistaDB, Metric, Index

>>> db = PistaDB("mydb.pst", dim=128, metric=Metric.L2, index=Index.HNSW)
>>> db.insert(1, np.random.rand(128).astype("float32"), label="dog")
>>> results = db.search(np.random.rand(128).astype("float32"), k=5)
>>> db.save()
>>> db.close()

Context-manager form:
>>> with PistaDB("mydb.pst", dim=128) as db:
...     db.insert(1, vec)
...     results = db.search(query, k=10)
"""

from __future__ import annotations

import ctypes
import os
import sys
import platform
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Union
import struct

import numpy as np

# ── Locate the shared library ─────────────────────────────────────────────────

def _find_lib() -> ctypes.CDLL:
    """Search for pistadb shared library in common locations."""
    system = platform.system()
    if system == "Windows":
        names = ["pistadb.dll", "libpistadb.dll"]
    elif system == "Darwin":
        names = ["libpistadb.dylib", "libpistadb.1.dylib"]
    else:
        names = ["libpistadb.so", "libpistadb.so.1"]

    # Search order: env var → adjacent build dirs → package dir
    search_dirs = []
    if "PISTADB_LIB_DIR" in os.environ:
        search_dirs.append(os.environ["PISTADB_LIB_DIR"])

    pkg_dir = Path(__file__).parent
    search_dirs += [
        str(pkg_dir),
        str(pkg_dir.parent.parent / "build"),
        str(pkg_dir.parent.parent / "build" / "Release"),
        str(pkg_dir.parent.parent / "build" / "Debug"),
        str(pkg_dir.parent.parent / "build" / "RelWithDebInfo"),
        "/usr/local/lib",
        "/usr/lib",
    ]

    for d in search_dirs:
        for name in names:
            candidate = os.path.join(d, name)
            if os.path.isfile(candidate):
                try:
                    return ctypes.CDLL(candidate)
                except OSError:
                    continue

    raise OSError(
        "PistaDB shared library not found. "
        "Build with CMake first (cmake -B build && cmake --build build), "
        "or set the PISTADB_LIB_DIR environment variable."
    )


_lib: Optional[ctypes.CDLL] = None

def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _find_lib()
        _setup_argtypes(_lib)
    return _lib


def _setup_argtypes(lib: ctypes.CDLL) -> None:
    """Declare C function signatures for type safety."""
    c_void_p = ctypes.c_void_p
    c_int    = ctypes.c_int
    c_uint64 = ctypes.c_uint64
    c_float  = ctypes.c_float
    c_char_p = ctypes.c_char_p
    c_size_p = ctypes.POINTER(ctypes.c_size_t)

    # PistaDBParams struct (must match C layout exactly)
    # We pass it as a byte buffer.

    lib.pistadb_open.restype  = c_void_p
    lib.pistadb_open.argtypes = [c_char_p, c_int, c_int, c_int, c_void_p]

    lib.pistadb_close.restype  = None
    lib.pistadb_close.argtypes = [c_void_p]

    lib.pistadb_save.restype  = c_int
    lib.pistadb_save.argtypes = [c_void_p]

    lib.pistadb_insert.restype  = c_int
    lib.pistadb_insert.argtypes = [c_void_p, c_uint64, c_char_p,
                                   ctypes.POINTER(c_float)]

    lib.pistadb_delete.restype  = c_int
    lib.pistadb_delete.argtypes = [c_void_p, c_uint64]

    lib.pistadb_update.restype  = c_int
    lib.pistadb_update.argtypes = [c_void_p, c_uint64, ctypes.POINTER(c_float)]

    lib.pistadb_get.restype  = c_int
    lib.pistadb_get.argtypes = [c_void_p, c_uint64,
                                ctypes.POINTER(c_float), c_char_p]

    lib.pistadb_search.restype  = c_int
    lib.pistadb_search.argtypes = [c_void_p, ctypes.POINTER(c_float),
                                   c_int, c_void_p]

    lib.pistadb_train.restype  = c_int
    lib.pistadb_train.argtypes = [c_void_p]

    lib.pistadb_train_on.restype  = c_int
    lib.pistadb_train_on.argtypes = [c_void_p, ctypes.POINTER(c_float), c_int]

    lib.pistadb_count.restype  = c_int
    lib.pistadb_count.argtypes = [c_void_p]

    lib.pistadb_dim.restype  = c_int
    lib.pistadb_dim.argtypes = [c_void_p]

    lib.pistadb_metric.restype  = c_int
    lib.pistadb_metric.argtypes = [c_void_p]

    lib.pistadb_index_type.restype  = c_int
    lib.pistadb_index_type.argtypes = [c_void_p]

    lib.pistadb_last_error.restype  = c_char_p
    lib.pistadb_last_error.argtypes = [c_void_p]

    lib.pistadb_version.restype  = c_char_p
    lib.pistadb_version.argtypes = []

    lib.pistadb_free_buf.restype  = None
    lib.pistadb_free_buf.argtypes = [c_void_p]

    # ── Embedding cache ───────────────────────────────────────────────────
    lib.pistadb_cache_open.restype  = c_void_p
    lib.pistadb_cache_open.argtypes = [c_char_p, c_int, c_int]

    lib.pistadb_cache_save.restype  = c_int
    lib.pistadb_cache_save.argtypes = [c_void_p]

    lib.pistadb_cache_close.restype  = None
    lib.pistadb_cache_close.argtypes = [c_void_p]

    lib.pistadb_cache_get.restype  = c_int
    lib.pistadb_cache_get.argtypes = [c_void_p, c_char_p,
                                      ctypes.POINTER(c_float)]

    lib.pistadb_cache_put.restype  = c_int
    lib.pistadb_cache_put.argtypes = [c_void_p, c_char_p,
                                      ctypes.POINTER(c_float)]

    lib.pistadb_cache_contains.restype  = c_int
    lib.pistadb_cache_contains.argtypes = [c_void_p, c_char_p]

    lib.pistadb_cache_evict_key.restype  = c_int
    lib.pistadb_cache_evict_key.argtypes = [c_void_p, c_char_p]

    lib.pistadb_cache_clear.restype  = None
    lib.pistadb_cache_clear.argtypes = [c_void_p]

    lib.pistadb_cache_count.restype  = c_int
    lib.pistadb_cache_count.argtypes = [c_void_p]

    lib.pistadb_cache_stats.restype  = None
    lib.pistadb_cache_stats.argtypes = [c_void_p, c_void_p]

    # ── Transactions ──────────────────────────────────────────────────────
    lib.pistadb_txn_begin.restype  = c_void_p
    lib.pistadb_txn_begin.argtypes = [c_void_p]

    lib.pistadb_txn_insert.restype  = c_int
    lib.pistadb_txn_insert.argtypes = [c_void_p, c_uint64, c_char_p,
                                       ctypes.POINTER(c_float)]

    lib.pistadb_txn_delete.restype  = c_int
    lib.pistadb_txn_delete.argtypes = [c_void_p, c_uint64]

    lib.pistadb_txn_update.restype  = c_int
    lib.pistadb_txn_update.argtypes = [c_void_p, c_uint64,
                                       ctypes.POINTER(c_float)]

    lib.pistadb_txn_commit.restype  = c_int
    lib.pistadb_txn_commit.argtypes = [c_void_p]

    lib.pistadb_txn_rollback.restype  = None
    lib.pistadb_txn_rollback.argtypes = [c_void_p]

    lib.pistadb_txn_free.restype  = None
    lib.pistadb_txn_free.argtypes = [c_void_p]

    lib.pistadb_txn_op_count.restype  = c_int
    lib.pistadb_txn_op_count.argtypes = [c_void_p]

    lib.pistadb_txn_last_error.restype  = c_char_p
    lib.pistadb_txn_last_error.argtypes = [c_void_p]


# ── Python enumerations ────────────────────────────────────────────────────────

class Metric(IntEnum):
    L2      = 0
    COSINE  = 1
    IP      = 2
    L1      = 3
    HAMMING = 4


class Index(IntEnum):
    LINEAR  = 0
    HNSW    = 1
    IVF     = 2
    IVF_PQ  = 3
    DISKANN = 4
    LSH     = 5
    SCANN   = 6
    SQ      = 7


# ── PistaDBParams (mirrors C struct) ───────────────────────────────────────────

class _CParams(ctypes.Structure):
    """ctypes mirror of PistaDBParams.  Must match C struct field order exactly."""
    _fields_ = [
        # HNSW
        ("hnsw_M",               ctypes.c_int),
        ("hnsw_ef_construction", ctypes.c_int),
        ("hnsw_ef_search",       ctypes.c_int),
        # IVF / IVF_PQ
        ("ivf_nlist",   ctypes.c_int),
        ("ivf_nprobe",  ctypes.c_int),
        ("pq_M",        ctypes.c_int),
        ("pq_nbits",    ctypes.c_int),
        # DiskANN
        ("diskann_R",     ctypes.c_int),
        ("diskann_L",     ctypes.c_int),
        ("diskann_alpha", ctypes.c_float),
        # LSH
        ("lsh_L", ctypes.c_int),
        ("lsh_K", ctypes.c_int),
        ("lsh_w", ctypes.c_float),
        # ScaNN
        ("scann_nlist",    ctypes.c_int),
        ("scann_nprobe",   ctypes.c_int),
        ("scann_pq_M",     ctypes.c_int),
        ("scann_pq_bits",  ctypes.c_int),
        ("scann_rerank_k", ctypes.c_int),
        ("scann_aq_eta",   ctypes.c_float),
    ]


@dataclass
class Params:
    """Index tuning parameters (Python-friendly)."""
    # HNSW
    hnsw_M:               int   = 16
    hnsw_ef_construction: int   = 200
    hnsw_ef_search:       int   = 50
    # IVF / IVF_PQ
    ivf_nlist:  int = 128
    ivf_nprobe: int = 8
    pq_M:       int = 8
    pq_nbits:   int = 8
    # DiskANN
    diskann_R:     int   = 32
    diskann_L:     int   = 100
    diskann_alpha: float = 1.2
    # LSH
    lsh_L: int   = 10
    lsh_K: int   = 8
    lsh_w: float = 10.0
    # ScaNN
    scann_nlist:    int   = 128
    scann_nprobe:   int   = 32
    scann_pq_M:     int   = 8
    scann_pq_bits:  int   = 8
    scann_rerank_k: int   = 100
    scann_aq_eta:   float = 0.2

    def _to_c(self) -> _CParams:
        return _CParams(
            self.hnsw_M, self.hnsw_ef_construction, self.hnsw_ef_search,
            self.ivf_nlist, self.ivf_nprobe, self.pq_M, self.pq_nbits,
            self.diskann_R, self.diskann_L, self.diskann_alpha,
            self.lsh_L, self.lsh_K, self.lsh_w,
            self.scann_nlist, self.scann_nprobe, self.scann_pq_M,
            self.scann_pq_bits, self.scann_rerank_k, self.scann_aq_eta,
        )


# ── Search result ──────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    id:       int
    distance: float
    label:    str = ""

    def __repr__(self):
        return f"SearchResult(id={self.id}, distance={self.distance:.6f}, label={self.label!r})"


# ── C result struct ────────────────────────────────────────────────────────────

class _CResult(ctypes.Structure):
    _fields_ = [
        ("id",       ctypes.c_uint64),
        ("distance", ctypes.c_float),
        ("label",    ctypes.c_char * 256),
    ]


# ── Main PistaDB class ──────────────────────────────────────────────────────────

class PistaDB:
    """
    Embedded vector database handle.

    Parameters
    ----------
    path : str
        Database file path (.pst).  Created if it does not exist.
    dim : int
        Vector dimension.  Must match if loading an existing file.
    metric : Metric
        Distance metric (default: Metric.L2).
    index : Index
        Index algorithm (default: Index.HNSW).
    params : Params | None
        Index tuning parameters.  None → use defaults.
    """

    def __init__(
        self,
        path: Union[str, Path],
        dim: int,
        metric: Metric = Metric.L2,
        index: Index   = Index.HNSW,
        params: Optional[Params] = None,
    ):
        self._lib  = _get_lib()
        self._path = str(path)

        c_params = (params or Params())._to_c()
        handle = self._lib.pistadb_open(
            self._path.encode(),
            ctypes.c_int(dim),
            ctypes.c_int(int(metric)),
            ctypes.c_int(int(index)),
            ctypes.byref(c_params),
        )
        if not handle:
            raise RuntimeError("pistadb_open failed – check that the library is built.")
        self._handle = handle
        self._dim    = dim

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "PistaDB":
        return self

    def __exit__(self, *args):
        self.close()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the database to disk."""
        r = self._lib.pistadb_save(self._handle)
        if r != 0:
            raise IOError(f"pistadb_save failed with code {r}: {self.last_error}")

    def close(self) -> None:
        """Free all resources (does NOT auto-save)."""
        if hasattr(self, "_handle") and self._handle:
            self._lib.pistadb_close(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    # ── CRUD ───────────────────────────────────────────────────────────────

    def insert(
        self,
        id: int,
        vector: np.ndarray,
        label: str = "",
    ) -> None:
        """
        Insert a vector.

        Parameters
        ----------
        id : int
            Unique integer identifier.
        vector : np.ndarray
            1-D float32 array of length dim.
        label : str
            Optional human-readable tag (< 256 bytes).
        """
        vec = self._check_vec(vector)
        r = self._lib.pistadb_insert(
            self._handle,
            ctypes.c_uint64(id),
            label.encode()[:255] if label else b"",
            vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        self._raise_if(r, "insert")

    def delete(self, id: int) -> None:
        """Logically delete a vector by id."""
        r = self._lib.pistadb_delete(self._handle, ctypes.c_uint64(id))
        self._raise_if(r, "delete")

    def update(self, id: int, vector: np.ndarray) -> None:
        """Replace the vector data for the given id."""
        vec = self._check_vec(vector)
        r = self._lib.pistadb_update(
            self._handle,
            ctypes.c_uint64(id),
            vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        self._raise_if(r, "update")

    def get(self, id: int) -> tuple:
        """
        Retrieve a vector and label by id.

        Returns
        -------
        (np.ndarray, str)  – (vector, label)
        """
        out_vec = np.zeros(self._dim, dtype=np.float32)
        out_lbl = ctypes.create_string_buffer(256)
        r = self._lib.pistadb_get(
            self._handle,
            ctypes.c_uint64(id),
            out_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_lbl,
        )
        self._raise_if(r, "get")
        return out_vec, out_lbl.value.decode(errors="replace")

    # ── Search ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> List[SearchResult]:
        """
        K-nearest-neighbour search.

        Parameters
        ----------
        query : np.ndarray
            Query vector (float32, length dim).
        k : int
            Number of results.

        Returns
        -------
        List[SearchResult]  sorted ascending by distance.
        """
        q = self._check_vec(query)
        ResultArray = _CResult * k
        res_buf = ResultArray()
        count = self._lib.pistadb_search(
            self._handle,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(k),
            res_buf,
        )
        return [
            SearchResult(
                id       = res_buf[i].id,
                distance = res_buf[i].distance,
                label    = res_buf[i].label.decode(errors="replace").rstrip("\x00"),
            )
            for i in range(max(count, 0))
        ]

    # ── Index management ───────────────────────────────────────────────────

    def train(self, training_vectors: Optional[np.ndarray] = None) -> None:
        """
        Train the index.

        For IVF / IVF_PQ, call this with a representative sample of vectors
        **before** inserting data:

            db.train(train_vecs)   # train on external data
            db.insert(...)         # then insert

        For HNSW / DiskANN, call with no argument after bulk insert to trigger
        a graph rebuild pass.
        """
        if training_vectors is not None:
            vecs = np.ascontiguousarray(training_vectors, dtype=np.float32)
            n = vecs.shape[0]
            r = self._lib.pistadb_train_on(
                self._handle,
                vecs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(n),
            )
        else:
            r = self._lib.pistadb_train_on(self._handle, None, ctypes.c_int(0))
            # Fall back to pistadb_train (DiskANN rebuild)
            if hasattr(self._lib, "pistadb_train"):
                r = self._lib.pistadb_train(self._handle)
        self._raise_if(r, "train")

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """Number of active (non-deleted) vectors."""
        return self._lib.pistadb_count(self._handle)

    @property
    def dim(self) -> int:
        return self._lib.pistadb_dim(self._handle)

    @property
    def metric(self) -> Metric:
        return Metric(self._lib.pistadb_metric(self._handle))

    @property
    def index_type(self) -> Index:
        return Index(self._lib.pistadb_index_type(self._handle))

    @property
    def last_error(self) -> str:
        s = self._lib.pistadb_last_error(self._handle)
        return s.decode() if s else ""

    @staticmethod
    def version() -> str:
        lib = _get_lib()
        return lib.pistadb_version().decode()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _check_vec(self, v: np.ndarray) -> np.ndarray:
        v = np.ascontiguousarray(v, dtype=np.float32).ravel()
        if len(v) != self._dim:
            raise ValueError(f"Expected vector of dim {self._dim}, got {len(v)}")
        return v

    def _raise_if(self, code: int, op: str) -> None:
        if code != 0:
            raise RuntimeError(
                f"pistadb_{op} failed (code={code}): {self.last_error}"
            )

    def __repr__(self):
        return (
            f"PistaDB(path={self._path!r}, dim={self._dim}, "
            f"metric={self.metric.name}, index={self.index_type.name}, "
            f"count={self.count})"
        )

    # ── Transactions ───────────────────────────────────────────────────────

    def begin_transaction(self) -> "Transaction":
        """
        Begin a new transaction on this database.

        Returns a :class:`Transaction` handle.  Use as a context manager for
        automatic commit/rollback, or call :meth:`Transaction.commit` and
        :meth:`Transaction.rollback` manually.

        Example
        -------
        >>> with db.begin_transaction() as txn:
        ...     txn.insert(1, vec1, label="a")
        ...     txn.insert(2, vec2, label="b")
        ...     # auto-commits on clean exit; rolls back on exception
        """
        handle = self._lib.pistadb_txn_begin(self._handle)
        if not handle:
            raise MemoryError("pistadb_txn_begin: allocation failed")
        return Transaction(self._lib, handle, self._dim)


# ── Convenience batch helpers ─────────────────────────────────────────────────

def insert_batch(db: PistaDB, ids, vectors: np.ndarray, labels=None) -> None:
    """Insert a batch of vectors efficiently."""
    n = len(ids)
    for i in range(n):
        lbl = labels[i] if labels is not None else ""
        db.insert(int(ids[i]), vectors[i], label=str(lbl))


def build_from_array(
    path: Union[str, Path],
    vectors: np.ndarray,
    ids=None,
    labels=None,
    metric: Metric = Metric.L2,
    index:  Index  = Index.HNSW,
    params: Optional[Params] = None,
    train_first: bool = False,
) -> PistaDB:
    """
    Convenience: create a new database and bulk-insert an array of vectors.

    Parameters
    ----------
    vectors : np.ndarray  shape (n, dim)
    ids     : array-like of int, or None (uses 1..n)
    labels  : array-like of str, or None
    train_first : for IVF/IVF_PQ, train on vectors before inserting
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n, dim  = vectors.shape
    if ids is None:
        ids = list(range(1, n + 1))

    db = PistaDB(path, dim=dim, metric=metric, index=index, params=params)

    if train_first:
        db.train(vectors)

    for i in range(n):
        lbl = str(labels[i]) if labels is not None else ""
        db.insert(int(ids[i]), vectors[i], label=lbl)

    return db


# ── Embedding cache ───────────────────────────────────────────────────────────

@dataclass
class CacheStats:
    """Snapshot of EmbeddingCache statistics."""
    hits:        int
    misses:      int
    evictions:   int
    count:       int
    max_entries: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __repr__(self):
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"evictions={self.evictions}, count={self.count}, "
            f"hit_rate={self.hit_rate:.1%})"
        )


class _CCacheStats(ctypes.Structure):
    """ctypes mirror of PistaDBCacheStats."""
    _fields_ = [
        ("hits",        ctypes.c_uint64),
        ("misses",      ctypes.c_uint64),
        ("evictions",   ctypes.c_uint64),
        ("count",       ctypes.c_int),
        ("max_entries", ctypes.c_int),
    ]


class EmbeddingCache:
    """
    Persistent LRU embedding cache: text string → float32 vector.

    Wraps the C ``pistadb_cache_*`` API.  Survives process restarts via a
    ``.pcc`` binary file.  Thread-safe (mutex held inside the C layer).

    Parameters
    ----------
    path : str | None
        File path for persistence (``*.pcc``).  Pass ``None`` for an
        in-memory-only cache that is never saved to disk.
    dim : int
        Embedding vector dimension.
    max_entries : int
        Capacity limit.  When full the least-recently-used entry is evicted.
        0 = unlimited.

    Example
    -------
    >>> cache = EmbeddingCache("embed.pcc", dim=384, max_entries=100_000)
    >>> vec = cache.get("hello world")
    >>> if vec is None:
    ...     vec = my_model.encode("hello world")
    ...     cache.put("hello world", vec)
    >>> cache.save()
    >>> cache.close()
    """

    def __init__(
        self,
        path: Optional[Union[str, Path]],
        dim: int,
        max_entries: int = 0,
    ):
        self._lib = _get_lib()
        self._dim  = dim
        self._path = str(path) if path is not None else None

        handle = self._lib.pistadb_cache_open(
            self._path.encode() if self._path else None,
            ctypes.c_int(dim),
            ctypes.c_int(max_entries),
        )
        if not handle:
            raise MemoryError("pistadb_cache_open: allocation failed")
        self._handle = handle

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *args):
        self.close()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the cache to its .pcc file."""
        r = self._lib.pistadb_cache_save(self._handle)
        if r != 0:
            raise IOError(f"pistadb_cache_save failed (code={r})")

    def close(self) -> None:
        """Free all resources.  Does NOT auto-save."""
        if hasattr(self, "_handle") and self._handle:
            self._lib.pistadb_cache_close(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    # ── Lookup / store ─────────────────────────────────────────────────────

    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Look up the cached embedding for ``text``.

        Returns a float32 numpy array on a hit, or ``None`` on a miss.
        The returned array is a copy — safe to modify.
        """
        out = np.empty(self._dim, dtype=np.float32)
        hit = self._lib.pistadb_cache_get(
            self._handle,
            text.encode(),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out if hit else None

    def put(self, text: str, vec: np.ndarray) -> None:
        """
        Store an embedding in the cache.

        ``vec`` is copied — caller may free it immediately.
        """
        v = np.ascontiguousarray(vec, dtype=np.float32).ravel()
        if len(v) != self._dim:
            raise ValueError(f"Expected vector of dim {self._dim}, got {len(v)}")
        r = self._lib.pistadb_cache_put(
            self._handle,
            text.encode(),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if r != 0:
            raise MemoryError(f"pistadb_cache_put failed (code={r})")

    def contains(self, text: str) -> bool:
        """Return ``True`` if ``text`` is cached (does not touch LRU order)."""
        return bool(self._lib.pistadb_cache_contains(self._handle, text.encode()))

    def evict(self, text: str) -> bool:
        """Remove a specific entry.  Returns ``True`` if the entry existed."""
        return bool(self._lib.pistadb_cache_evict_key(self._handle, text.encode()))

    def clear(self) -> None:
        """Remove all entries (keeps file path and settings)."""
        self._lib.pistadb_cache_clear(self._handle)

    # ── Metadata ───────────────────────────────────────────────────────────

    def stats(self) -> CacheStats:
        """Return a snapshot of cache statistics."""
        s = _CCacheStats()
        self._lib.pistadb_cache_stats(self._handle, ctypes.byref(s))
        return CacheStats(
            hits=s.hits,
            misses=s.misses,
            evictions=s.evictions,
            count=s.count,
            max_entries=s.max_entries,
        )

    @property
    def count(self) -> int:
        """Number of entries currently in the cache."""
        return self._lib.pistadb_cache_count(self._handle)

    def __len__(self) -> int:
        return self.count

    def __contains__(self, text: str) -> bool:
        return self.contains(text)

    def __repr__(self):
        s = self.stats()
        return (
            f"EmbeddingCache(path={self._path!r}, dim={self._dim}, "
            f"count={s.count}, hit_rate={s.hit_rate:.1%})"
        )


class CachedEmbedder:
    """
    Transparent caching wrapper around any embedding callable.

    Checks the cache before calling the model; stores the result on a miss.
    Optionally auto-saves the cache every ``autosave_every`` new entries.

    Parameters
    ----------
    embed_fn : callable
        ``embed_fn(text: str) -> np.ndarray`` — the expensive model call.
    cache : EmbeddingCache
        The cache to consult / populate.
    autosave_every : int
        Flush to disk after this many new embeddings.  0 = never.

    Example
    -------
    >>> cache   = EmbeddingCache("embed.pcc", dim=384, max_entries=100_000)
    >>> embedder = CachedEmbedder(my_model.encode, cache, autosave_every=500)
    >>> vec = embedder("hello world")   # hits model only on first call
    >>> cache.close()
    """

    def __init__(
        self,
        embed_fn,
        cache: EmbeddingCache,
        autosave_every: int = 0,
    ):
        self._fn             = embed_fn
        self._cache          = cache
        self._autosave_every = autosave_every
        self._since_save     = 0

    def __call__(self, text: str) -> np.ndarray:
        vec = self._cache.get(text)
        if vec is not None:
            return vec
        vec = np.ascontiguousarray(self._fn(text), dtype=np.float32).ravel()
        self._cache.put(text, vec)
        if self._autosave_every > 0:
            self._since_save += 1
            if self._since_save >= self._autosave_every:
                self._cache.save()
                self._since_save = 0
        return vec

    def embed_batch(self, texts) -> np.ndarray:
        """
        Encode a list of texts, using the cache for any already-seen strings.

        Returns
        -------
        np.ndarray  shape (len(texts), dim), dtype float32
        """
        results = []
        for t in texts:
            results.append(self(t))
        return np.stack(results)

    @property
    def cache(self) -> EmbeddingCache:
        return self._cache


# ── Transaction ───────────────────────────────────────────────────────────────

#: Return code from :func:`pistadb_txn_commit` when the commit partially
#: succeeded and rollback of some operations was not possible.
TXN_PARTIAL = -10


class Transaction:
    """
    Atomic group of INSERT / DELETE / UPDATE operations on a :class:`PistaDB`.

    Do not instantiate directly — use :meth:`PistaDB.begin_transaction`.

    Commit semantics
    ----------------
    1. **Validation**: structural checks (e.g. duplicate INSERT ids) are
       performed before any change is applied.  On failure the database is
       untouched and a :class:`RuntimeError` is raised.

    2. **Apply**: operations execute in staging order.  If any individual
       operation fails, all previously applied ops are rolled back using
       internally-saved undo snapshots.  If rollback is complete a
       :class:`RuntimeError` is raised; if rollback is incomplete (e.g. for
       IVF_PQ / ScaNN where raw vectors are not directly retrievable) a
       :class:`RuntimeError` with ``partial=True`` is raised.

    Undo availability
    -----------------
    * INSERT → undo = DELETE (always available)
    * DELETE / UPDATE on LINEAR, HNSW, IVF, DiskANN, LSH → undo available
    * DELETE / UPDATE on IVF_PQ → undo unavailable (PQ codes, not raw vectors)

    Context-manager usage (recommended)
    ------------------------------------
    ::

        with db.begin_transaction() as txn:
            txn.insert(1, vec1, label="dog")
            txn.delete(42)
        # commits on clean exit; rolls back (and re-raises) on exception

    Manual usage
    ------------
    ::

        txn = db.begin_transaction()
        try:
            txn.insert(1, vec1)
            txn.commit()
        except Exception:
            txn.rollback()
        finally:
            txn.free()
    """

    def __init__(self, lib: ctypes.CDLL, handle: int, dim: int):
        self._lib    = lib
        self._handle = handle
        self._dim    = dim

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "Transaction":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        self.free()
        return False  # do not suppress exceptions

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def commit(self) -> None:
        """
        Validate and apply all staged operations atomically.

        Raises
        ------
        RuntimeError
            If validation fails (no changes applied) or if apply fails and
            rollback completes successfully.
        RuntimeError (with ``partial=True`` attribute)
            If apply fails and rollback is incomplete.
        """
        r = self._lib.pistadb_txn_commit(self._handle)
        if r != 0:
            err = self._last_error
            exc = RuntimeError(
                f"pistadb_txn_commit failed (code={r}): {err}"
            )
            exc.partial = (r == TXN_PARTIAL)
            raise exc

    def rollback(self) -> None:
        """Discard all staged operations without touching the database."""
        self._lib.pistadb_txn_rollback(self._handle)

    def free(self) -> None:
        """Release all resources.  Implies rollback if not yet committed."""
        if self._handle:
            self._lib.pistadb_txn_free(self._handle)
            self._handle = None

    def __del__(self):
        self.free()

    # ── Staging operations ─────────────────────────────────────────────────

    def insert(self, id: int, vector: np.ndarray, label: str = "") -> None:
        """Stage an INSERT.  The vector is copied; caller may reuse it."""
        vec = self._check_vec(vector)
        r = self._lib.pistadb_txn_insert(
            self._handle,
            ctypes.c_uint64(id),
            label.encode()[:255] if label else b"",
            vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        self._raise_if(r, "txn_insert")

    def delete(self, id: int) -> None:
        """Stage a DELETE."""
        r = self._lib.pistadb_txn_delete(self._handle, ctypes.c_uint64(id))
        self._raise_if(r, "txn_delete")

    def update(self, id: int, vector: np.ndarray) -> None:
        """Stage an UPDATE.  The vector is copied; caller may reuse it."""
        vec = self._check_vec(vector)
        r = self._lib.pistadb_txn_update(
            self._handle,
            ctypes.c_uint64(id),
            vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        self._raise_if(r, "txn_update")

    # ── Introspection ──────────────────────────────────────────────────────

    @property
    def op_count(self) -> int:
        """Number of staged (uncommitted) operations."""
        return self._lib.pistadb_txn_op_count(self._handle)

    @property
    def _last_error(self) -> str:
        s = self._lib.pistadb_txn_last_error(self._handle)
        return s.decode() if s else ""

    def __repr__(self):
        return f"Transaction(op_count={self.op_count})"

    # ── Internal helpers ───────────────────────────────────────────────────

    def _check_vec(self, v: np.ndarray) -> np.ndarray:
        v = np.ascontiguousarray(v, dtype=np.float32).ravel()
        if len(v) != self._dim:
            raise ValueError(f"Expected vector of dim {self._dim}, got {len(v)}")
        return v

    def _raise_if(self, code: int, op: str) -> None:
        if code != 0:
            raise RuntimeError(
                f"pistadb_{op} failed (code={code}): {self._last_error}"
            )


__all__ = [
    "PistaDB",
    "Metric",
    "Index",
    "Params",
    "SearchResult",
    "insert_batch",
    "build_from_array",
    "EmbeddingCache",
    "CachedEmbedder",
    "CacheStats",
    "Transaction",
    "TXN_PARTIAL",
]
