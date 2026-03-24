"""
Low-level driver that talks to pistadb.dll / libpistadb.so directly via ctypes.
Shuckr uses this instead of depending on the pistadb Python package,
so it only needs the compiled shared library.
"""

from __future__ import annotations

import ctypes
import os
import platform
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ── Enums ─────────────────────────────────────────────────────────────────────

class Metric(IntEnum):
    L2      = 0
    COSINE  = 1
    IP      = 2
    L1      = 3
    HAMMING = 4

    @classmethod
    def names(cls):
        return [m.name for m in cls]


class IndexType(IntEnum):
    LINEAR  = 0
    HNSW    = 1
    IVF     = 2
    IVF_PQ  = 3
    DISKANN = 4
    LSH     = 5
    SCANN   = 6

    @classmethod
    def names(cls):
        return [m.name for m in cls]


# ── C structs ─────────────────────────────────────────────────────────────────

class _CParams(ctypes.Structure):
    _fields_ = [
        ("hnsw_M", ctypes.c_int), ("hnsw_ef_construction", ctypes.c_int),
        ("hnsw_ef_search", ctypes.c_int),
        ("ivf_nlist", ctypes.c_int), ("ivf_nprobe", ctypes.c_int),
        ("pq_M", ctypes.c_int), ("pq_nbits", ctypes.c_int),
        ("diskann_R", ctypes.c_int), ("diskann_L", ctypes.c_int),
        ("diskann_alpha", ctypes.c_float),
        ("lsh_L", ctypes.c_int), ("lsh_K", ctypes.c_int),
        ("lsh_w", ctypes.c_float),
        ("scann_nlist", ctypes.c_int), ("scann_nprobe", ctypes.c_int),
        ("scann_pq_M", ctypes.c_int), ("scann_pq_bits", ctypes.c_int),
        ("scann_rerank_k", ctypes.c_int), ("scann_aq_eta", ctypes.c_float),
    ]


class _CResult(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("distance", ctypes.c_float),
        ("label", ctypes.c_char * 256),
    ]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class VectorRecord:
    id: int
    label: str
    vector: np.ndarray


@dataclass
class SearchResult:
    id: int
    distance: float
    label: str


# ── .pst header reader (standalone, no library needed) ────────────────────────

PST_HEADER_SIZE = 128
PST_MAGIC = b"PSDB"

@dataclass
class PstHeader:
    magic: bytes
    version_major: int
    version_minor: int
    flags: int
    dimension: int
    metric_type: int
    index_type: int
    num_vectors: int

    @property
    def metric_name(self) -> str:
        try:
            return Metric(self.metric_type).name
        except ValueError:
            return f"UNKNOWN({self.metric_type})"

    @property
    def index_name(self) -> str:
        try:
            return IndexType(self.index_type).name
        except ValueError:
            return f"UNKNOWN({self.index_type})"


def read_pst_header(path: str) -> Optional[PstHeader]:
    """Read the 128-byte header of a .pst file without loading the library."""
    try:
        with open(path, "rb") as f:
            data = f.read(PST_HEADER_SIZE)
        if len(data) < PST_HEADER_SIZE or data[:4] != PST_MAGIC:
            return None
        magic = data[0:4]
        ver_major, ver_minor = struct.unpack_from("<HH", data, 4)
        flags, = struct.unpack_from("<I", data, 8)
        dim, = struct.unpack_from("<I", data, 12)
        metric_type, index_type = struct.unpack_from("<HH", data, 16)
        num_vectors, = struct.unpack_from("<Q", data, 20)
        return PstHeader(magic, ver_major, ver_minor, flags, dim,
                         metric_type, index_type, num_vectors)
    except (OSError, struct.error):
        return None


# ── Library loader ────────────────────────────────────────────────────────────

def _find_lib(extra_dirs: Optional[List[str]] = None) -> ctypes.CDLL:
    system = platform.system()
    if system == "Windows":
        names = ["pistadb.dll", "libpistadb.dll"]
    elif system == "Darwin":
        names = ["libpistadb.dylib"]
    else:
        names = ["libpistadb.so"]

    search_dirs: list[str] = []
    if extra_dirs:
        search_dirs.extend(extra_dirs)
    if "PISTADB_LIB_DIR" in os.environ:
        search_dirs.append(os.environ["PISTADB_LIB_DIR"])

    here = Path(__file__).parent
    search_dirs += [
        str(here),
        str(here.parent / "build" / "Release"),
        str(here.parent / "build" / "Debug"),
        str(here.parent / "build"),
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
        "pistadb shared library not found. "
        "Set PISTADB_LIB_DIR or place pistadb.dll next to this script."
    )


def _setup(lib: ctypes.CDLL):
    V = ctypes.c_void_p
    I = ctypes.c_int
    U = ctypes.c_uint64
    F = ctypes.c_float
    S = ctypes.c_char_p
    PF = ctypes.POINTER(F)

    def sig(name, res, args):
        fn = getattr(lib, name)
        fn.restype = res
        fn.argtypes = args

    sig("pistadb_open",       V, [S, I, I, I, V])
    sig("pistadb_close",      None, [V])
    sig("pistadb_save",       I, [V])
    sig("pistadb_insert",     I, [V, U, S, PF])
    sig("pistadb_delete",     I, [V, U])
    sig("pistadb_update",     I, [V, U, PF])
    sig("pistadb_get",        I, [V, U, PF, S])
    sig("pistadb_search",     I, [V, PF, I, V])
    sig("pistadb_count",      I, [V])
    sig("pistadb_dim",        I, [V])
    sig("pistadb_metric",     I, [V])
    sig("pistadb_index_type", I, [V])
    sig("pistadb_last_error", S, [V])
    sig("pistadb_version",    S, [])
    sig("pistadb_train",      I, [V])
    sig("pistadb_train_on",   I, [V, PF, I])


# ── High-level database handle ────────────────────────────────────────────────

class Database:
    """High-level wrapper around a single pistadb handle."""

    def __init__(self, lib: ctypes.CDLL, handle, path: str, dim: int):
        self._lib = lib
        self._handle = handle
        self.path = path
        self._dim = dim

    # -- properties ---
    @property
    def dim(self) -> int:
        return self._lib.pistadb_dim(self._handle)

    @property
    def count(self) -> int:
        return self._lib.pistadb_count(self._handle)

    @property
    def metric(self) -> Metric:
        return Metric(self._lib.pistadb_metric(self._handle))

    @property
    def index_type(self) -> IndexType:
        return IndexType(self._lib.pistadb_index_type(self._handle))

    @property
    def last_error(self) -> str:
        e = self._lib.pistadb_last_error(self._handle)
        return e.decode(errors="replace") if e else ""

    # -- CRUD ---
    def insert(self, vid: int, vector: np.ndarray, label: str = "") -> None:
        vec = np.ascontiguousarray(vector, dtype=np.float32)
        r = self._lib.pistadb_insert(
            self._handle, ctypes.c_uint64(vid),
            label.encode()[:255] if label else b"",
            vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if r != 0:
            raise RuntimeError(f"insert failed: {self.last_error}")

    def delete(self, vid: int) -> None:
        r = self._lib.pistadb_delete(self._handle, ctypes.c_uint64(vid))
        if r != 0:
            raise RuntimeError(f"delete failed: {self.last_error}")

    def update(self, vid: int, vector: np.ndarray) -> None:
        vec = np.ascontiguousarray(vector, dtype=np.float32)
        r = self._lib.pistadb_update(
            self._handle, ctypes.c_uint64(vid),
            vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if r != 0:
            raise RuntimeError(f"update failed: {self.last_error}")

    def get(self, vid: int) -> Tuple[np.ndarray, str]:
        out_vec = np.zeros(self._dim, dtype=np.float32)
        out_lbl = ctypes.create_string_buffer(256)
        r = self._lib.pistadb_get(
            self._handle, ctypes.c_uint64(vid),
            out_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_lbl,
        )
        if r != 0:
            raise RuntimeError(f"get failed: {self.last_error}")
        return out_vec, out_lbl.value.decode(errors="replace")

    def search(self, query: np.ndarray, k: int = 10) -> List[SearchResult]:
        q = np.ascontiguousarray(query, dtype=np.float32)
        ResultArray = _CResult * k
        buf = ResultArray()
        n = self._lib.pistadb_search(
            self._handle,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(k), buf,
        )
        return [
            SearchResult(buf[i].id, buf[i].distance,
                         buf[i].label.decode(errors="replace").rstrip("\x00"))
            for i in range(max(n, 0))
        ]

    def save(self) -> None:
        r = self._lib.pistadb_save(self._handle)
        if r != 0:
            raise RuntimeError(f"save failed: {self.last_error}")

    def close(self):
        if self._handle:
            self._lib.pistadb_close(self._handle)
            self._handle = None


# ── Driver (manages the library and database instances) ───────────────────────

class Driver:
    """Singleton-ish driver that loads the native library once."""

    def __init__(self, lib_dir: Optional[str] = None):
        extra = [lib_dir] if lib_dir else []
        self._lib = _find_lib(extra)
        _setup(self._lib)

    @property
    def version(self) -> str:
        v = self._lib.pistadb_version()
        return v.decode() if v else "unknown"

    def open(self, path: str, dim: int,
             metric: Metric = Metric.L2,
             index_type: IndexType = IndexType.HNSW) -> Database:
        params = _CParams()
        # defaults
        params.hnsw_M = 16
        params.hnsw_ef_construction = 200
        params.hnsw_ef_search = 50
        params.ivf_nlist = 128
        params.ivf_nprobe = 8
        params.pq_M = 8
        params.pq_nbits = 8
        params.diskann_R = 32
        params.diskann_L = 100
        params.diskann_alpha = 1.2
        params.lsh_L = 10
        params.lsh_K = 8
        params.lsh_w = 10.0
        params.scann_nlist = 128
        params.scann_nprobe = 32
        params.scann_pq_M = 8
        params.scann_pq_bits = 8
        params.scann_rerank_k = 100
        params.scann_aq_eta = 0.2

        handle = self._lib.pistadb_open(
            path.encode(), ctypes.c_int(dim),
            ctypes.c_int(int(metric)), ctypes.c_int(int(index_type)),
            ctypes.byref(params),
        )
        if not handle:
            raise RuntimeError("pistadb_open failed")
        return Database(self._lib, handle, path, dim)

    def create(self, path: str, dim: int,
               metric: Metric = Metric.L2,
               index_type: IndexType = IndexType.HNSW) -> Database:
        return self.open(path, dim, metric, index_type)
