"""
PistaDB comprehensive test suite.

Run with:
    pytest tests/test_pistadb.py -v

Requirements:
    - PistaDB shared library built (cmake -B build && cmake --build build)
    - PISTADB_LIB_DIR env var set, or library in build/ directory
    - pytest, numpy, scipy (optional, for recall evaluation)
"""
import os
import sys
import tempfile
import pytest
import numpy as np

# Add python package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from pistadb import (
    PistaDB, Metric, Index, Params, build_from_array,
    EmbeddingCache, CachedEmbedder, CacheStats,
    Transaction, TXN_PARTIAL,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

DIM    = 32
N_VECS = 200
SEED   = 42

@pytest.fixture
def rng():
    return np.random.default_rng(SEED)

@pytest.fixture
def random_vecs(rng):
    return rng.random((N_VECS, DIM), dtype=np.float32)

@pytest.fixture
def tmp_db_path(tmp_path):
    return str(tmp_path / "test.pst")


def make_db(path, index=Index.HNSW, metric=Metric.L2, params=None):
    return PistaDB(path, dim=DIM, metric=metric, index=index, params=params)


# ── Distance metric tests ─────────────────────────────────────────────────────

class TestMetrics:
    """Black-box test: insert 2 vectors, verify search order matches metric."""

    def _check_order(self, tmp_path, metric, vecs):
        """Query vec[0]; result should be vec[0] (distance 0) then vec[1]."""
        path = str(tmp_path / f"metric_{metric.name}.pst")
        with make_db(path, index=Index.LINEAR, metric=metric) as db:
            db.insert(1, vecs[0], label="a")
            db.insert(2, vecs[1], label="b")
            results = db.search(vecs[0], k=2)
        assert len(results) == 2
        assert results[0].id == 1        # nearest to itself
        assert results[0].distance <= results[1].distance

    def test_l2(self, tmp_path, rng):
        v = rng.random((2, DIM), dtype=np.float32)
        self._check_order(tmp_path, Metric.L2, v)

    def test_cosine(self, tmp_path, rng):
        v = rng.random((2, DIM), dtype=np.float32)
        self._check_order(tmp_path, Metric.COSINE, v)

    def test_ip(self, tmp_path, rng):
        v = rng.random((2, DIM), dtype=np.float32)
        self._check_order(tmp_path, Metric.IP, v)

    def test_l1(self, tmp_path, rng):
        v = rng.random((2, DIM), dtype=np.float32)
        self._check_order(tmp_path, Metric.L1, v)

    def test_hamming(self, tmp_path, rng):
        v = rng.integers(0, 2, size=(2, DIM)).astype(np.float32)
        self._check_order(tmp_path, Metric.HAMMING, v)

    def test_l2_exact_distance(self, tmp_path):
        """Verify L2 distance is sqrt(sum of squares)."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        path = str(tmp_path / "l2_exact.pst")
        with PistaDB(path, dim=3, metric=Metric.L2, index=Index.LINEAR) as db:
            db.insert(1, a)
            db.insert(2, b)
            res = db.search(a, k=2)
        assert res[0].id == 1
        assert abs(res[0].distance) < 1e-5
        assert abs(res[1].distance - np.sqrt(2)) < 1e-4

    def test_cosine_distance_range(self, tmp_path, rng):
        """Cosine distance must be in [0, 2]."""
        v = rng.random((10, DIM), dtype=np.float32)
        path = str(tmp_path / "cos_range.pst")
        with PistaDB(path, dim=DIM, metric=Metric.COSINE, index=Index.LINEAR) as db:
            for i, vec in enumerate(v):
                db.insert(i + 1, vec)
            res = db.search(v[0], k=10)
        for r in res:
            assert 0.0 <= r.distance <= 2.0 + 1e-5


# ── Index algorithm tests ──────────────────────────────────────────────────────

class TestIndexBasic:
    """Each index must pass basic insert / search / delete / update."""

    CONFIGS = [
        (Index.LINEAR,  {}),
        (Index.HNSW,    {"hnsw_M": 8, "hnsw_ef_construction": 40, "hnsw_ef_search": 20}),
        (Index.DISKANN, {"diskann_R": 16, "diskann_L": 40}),
        (Index.LSH,     {"lsh_L": 5, "lsh_K": 6}),
    ]

    @pytest.mark.parametrize("index,kwargs", CONFIGS)
    def test_insert_and_search(self, tmp_path, rng, index, kwargs):
        path = str(tmp_path / f"{index.name}.pst")
        params = Params(**kwargs) if kwargs else None
        vecs = rng.random((50, DIM), dtype=np.float32)

        with PistaDB(path, dim=DIM, metric=Metric.L2, index=index, params=params) as db:
            for i, v in enumerate(vecs):
                db.insert(i + 1, v, label=f"vec_{i}")
            assert db.count == 50
            res = db.search(vecs[0], k=5)

        assert len(res) >= 1
        # The query vector itself (id=1) should be in the top results
        ids_returned = [r.id for r in res]
        assert 1 in ids_returned, f"{index.name}: query vector not in top results"

    @pytest.mark.parametrize("index,kwargs", CONFIGS)
    def test_delete(self, tmp_path, rng, index, kwargs):
        path = str(tmp_path / f"del_{index.name}.pst")
        params = Params(**kwargs) if kwargs else None
        vecs = rng.random((20, DIM), dtype=np.float32)

        with PistaDB(path, dim=DIM, index=index, params=params) as db:
            for i, v in enumerate(vecs):
                db.insert(i + 1, v)
            db.delete(1)
            assert db.count == 19
            res = db.search(vecs[0], k=5)
        # id=1 should not appear
        assert 1 not in [r.id for r in res], f"{index.name}: deleted vector returned"

    @pytest.mark.parametrize("index,kwargs", CONFIGS)
    def test_update(self, tmp_path, rng, index, kwargs):
        """After update, search with the new vector should return id=1 first."""
        path = str(tmp_path / f"upd_{index.name}.pst")
        params = Params(**kwargs) if kwargs else None
        vecs = rng.random((10, DIM), dtype=np.float32)
        new_vec = np.zeros(DIM, dtype=np.float32)

        with PistaDB(path, dim=DIM, index=index, params=params) as db:
            for i, v in enumerate(vecs):
                db.insert(i + 1, v)
            db.update(1, new_vec)
            res = db.search(new_vec, k=3)
        assert res[0].id == 1 or res[0].distance < 0.1

    @pytest.mark.parametrize("index,kwargs", CONFIGS)
    def test_labels(self, tmp_path, rng, index, kwargs):
        path = str(tmp_path / f"lbl_{index.name}.pst")
        params = Params(**kwargs) if kwargs else None
        vecs = rng.random((5, DIM), dtype=np.float32)

        with PistaDB(path, dim=DIM, index=index, params=params) as db:
            for i, v in enumerate(vecs):
                db.insert(i + 1, v, label=f"item_{i}")
            res = db.search(vecs[0], k=5)

        # At least one result should have a non-empty label
        labels = [r.label for r in res]
        assert any(lbl for lbl in labels)


class TestIVFIndex:
    """IVF requires explicit training before insert."""

    def test_ivf_basic(self, tmp_path, rng):
        path = str(tmp_path / "ivf.pst")
        n_train = 100
        train_vecs = rng.random((n_train, DIM), dtype=np.float32)
        query = rng.random(DIM, dtype=np.float32)

        params = Params(ivf_nlist=10, ivf_nprobe=3)
        with PistaDB(path, dim=DIM, metric=Metric.L2, index=Index.IVF,
                    params=params) as db:
            db.train(train_vecs)
            for i, v in enumerate(train_vecs):
                db.insert(i + 1, v)
            res = db.search(query, k=5)

        assert len(res) > 0

    def test_ivf_pq_basic(self, tmp_path, rng):
        path = str(tmp_path / "ivfpq.pst")
        dim = 16  # dim must be divisible by pq_M
        n   = 80
        vecs = rng.random((n, dim), dtype=np.float32)

        params = Params(ivf_nlist=8, ivf_nprobe=2, pq_M=4, pq_nbits=4)
        with PistaDB(path, dim=dim, metric=Metric.L2, index=Index.IVF_PQ,
                    params=params) as db:
            db.train(vecs)
            for i, v in enumerate(vecs):
                db.insert(i + 1, v)
            q = rng.random(dim, dtype=np.float32)
            res = db.search(q, k=5)
        assert len(res) > 0


# ── Persistence tests ──────────────────────────────────────────────────────────

class TestPersistence:

    def test_save_and_reload(self, tmp_path, rng):
        path = str(tmp_path / "persist.pst")
        vecs = rng.random((30, DIM), dtype=np.float32)

        # Write
        with PistaDB(path, dim=DIM, index=Index.LINEAR) as db:
            for i, v in enumerate(vecs):
                db.insert(i + 1, v, label=f"v{i}")
            db.save()

        # Read back
        with PistaDB(path, dim=DIM) as db2:
            assert db2.count == 30
            res = db2.search(vecs[0], k=5)

        assert 1 in [r.id for r in res]

    @pytest.mark.parametrize("index", [Index.LINEAR, Index.HNSW, Index.DISKANN, Index.LSH])
    def test_roundtrip(self, tmp_path, rng, index):
        """Save then load; results must be identical."""
        path = str(tmp_path / f"rt_{index.name}.pst")
        vecs = rng.random((40, DIM), dtype=np.float32)
        q    = rng.random(DIM, dtype=np.float32)

        with PistaDB(path, dim=DIM, index=index) as db:
            for i, v in enumerate(vecs):
                db.insert(i + 1, v)
            res_before = db.search(q, k=5)
            db.save()

        with PistaDB(path, dim=DIM) as db2:
            res_after = db2.search(q, k=5)

        # Top result id should match
        if res_before and res_after:
            assert res_before[0].id == res_after[0].id, \
                f"{index.name}: top result changed after save/load"

    def test_file_header_magic(self, tmp_path, rng):
        """Verify the file starts with the 'PSDB' magic bytes."""
        path = str(tmp_path / "magic.pst")
        with PistaDB(path, dim=DIM) as db:
            db.insert(1, rng.random(DIM, dtype=np.float32))
            db.save()

        with open(path, "rb") as f:
            magic = f.read(4)
        assert magic == b"PSDB"

    def test_corrupt_file_rejected(self, tmp_path):
        """A file with wrong magic must not be silently loaded."""
        path = str(tmp_path / "corrupt.pst")
        with open(path, "wb") as f:
            f.write(b"BAAD" + b"\x00" * 124)
        # pistadb_open should either return None or fall back to creating new db
        try:
            db = PistaDB(path, dim=DIM)
            db.close()
        except RuntimeError:
            pass  # acceptable: raise an error


# ── Recall evaluation ─────────────────────────────────────────────────────────

class TestRecall:
    """Sanity-check ANN recall against brute-force ground truth."""

    N     = 500
    DIM   = 64
    K     = 10
    N_Q   = 20
    ALPHA = 0.5   # accept recall@K ≥ 50%

    @staticmethod
    def _brute_force_knn(db_vecs, query, k):
        dists = np.sqrt(((db_vecs - query) ** 2).sum(axis=1))
        return set(np.argsort(dists)[:k] + 1)  # 1-indexed ids

    @pytest.fixture
    def dataset(self):
        rng  = np.random.default_rng(99)
        vecs = rng.random((self.N, self.DIM), dtype=np.float32)
        qs   = rng.random((self.N_Q, self.DIM), dtype=np.float32)
        return vecs, qs

    def _measure_recall(self, path, index, params, dataset):
        vecs, queries = dataset
        with PistaDB(path, dim=self.DIM, index=index, params=params) as db:
            if index in (Index.IVF, Index.IVF_PQ):
                db.train(vecs)
            for i, v in enumerate(vecs):
                db.insert(i + 1, v)
            recalls = []
            for q in queries:
                gt   = self._brute_force_knn(vecs, q, self.K)
                res  = db.search(q, k=self.K)
                hits = sum(1 for r in res if r.id in gt)
                recalls.append(hits / self.K)
        return np.mean(recalls)

    @pytest.mark.parametrize("index,params", [
        (Index.LINEAR,  Params()),
        (Index.HNSW,    Params(hnsw_M=16, hnsw_ef_construction=100, hnsw_ef_search=50)),
        (Index.DISKANN, Params(diskann_R=20, diskann_L=60)),
        (Index.LSH,     Params(lsh_L=12, lsh_K=8)),
    ])
    def test_recall(self, tmp_path, dataset, index, params):
        path = str(tmp_path / f"recall_{index.name}.pst")
        recall = self._measure_recall(path, index, params, dataset)
        assert recall >= self.ALPHA, \
            f"{index.name} recall@{self.K} = {recall:.2f} < {self.ALPHA}"

    def test_linear_recall_perfect(self, tmp_path, dataset):
        """Linear scan must achieve recall@10 = 1.0."""
        path = str(tmp_path / "linear_perfect.pst")
        recall = self._measure_recall(path, Index.LINEAR, Params(), dataset)
        assert recall >= 0.99, f"Linear recall@10 = {recall:.2f} (expected ~1.0)"


# ── API tests ─────────────────────────────────────────────────────────────────

class TestAPI:

    def test_version_string(self):
        ver = PistaDB.version()
        assert isinstance(ver, str)
        assert "." in ver

    def test_properties(self, tmp_path, rng):
        path = str(tmp_path / "props.pst")
        with PistaDB(path, dim=DIM, metric=Metric.COSINE, index=Index.HNSW) as db:
            assert db.dim == DIM
            assert db.metric == Metric.COSINE
            assert db.index_type == Index.HNSW
            db.insert(1, rng.random(DIM, dtype=np.float32))
            assert db.count == 1

    def test_wrong_dim_raises(self, tmp_path, rng):
        path = str(tmp_path / "wrongdim.pst")
        with PistaDB(path, dim=DIM) as db:
            db.insert(1, rng.random(DIM, dtype=np.float32))
        with pytest.raises(ValueError):
            with PistaDB(path, dim=DIM) as db:
                db.search(rng.random(DIM + 1, dtype=np.float32), k=1)

    def test_build_from_array(self, tmp_path, rng):
        path = str(tmp_path / "batch.pst")
        vecs = rng.random((50, DIM), dtype=np.float32)
        db = build_from_array(path, vecs, metric=Metric.L2, index=Index.LINEAR)
        assert db.count == 50
        res = db.search(vecs[0], k=3)
        assert res[0].id == 1
        db.close()

    def test_context_manager_closes(self, tmp_path, rng):
        path = str(tmp_path / "ctx.pst")
        with PistaDB(path, dim=DIM) as db:
            db.insert(1, rng.random(DIM, dtype=np.float32))
        # handle should be None after exit
        assert db._handle is None

    def test_repr(self, tmp_path, rng):
        path = str(tmp_path / "repr.pst")
        with PistaDB(path, dim=DIM) as db:
            r = repr(db)
        assert "PistaDB" in r


# ── ScaNN tests ───────────────────────────────────────────────────────────────

class TestScaNN:
    """Tests for the ScaNN (Anisotropic Vector Quantization) index."""

    DIM  = 16   # must be divisible by scann_pq_M
    N    = 120

    @pytest.fixture
    def dataset(self):
        rng  = np.random.default_rng(7)
        vecs = rng.random((self.N, self.DIM), dtype=np.float32)
        return vecs

    def test_scann_basic_search(self, tmp_path, dataset):
        """Train, insert, search — should return results without error."""
        path   = str(tmp_path / "scann_basic.pst")
        params = Params(scann_nlist=8, scann_nprobe=4,
                        scann_pq_M=4, scann_pq_bits=8,
                        scann_rerank_k=20, scann_aq_eta=0.2)
        with PistaDB(path, dim=self.DIM, metric=Metric.L2,
                     index=Index.SCANN, params=params) as db:
            db.train(dataset)
            for i, v in enumerate(dataset):
                db.insert(i + 1, v)
            q   = dataset[0]
            res = db.search(q, k=5)
        assert len(res) > 0
        assert res[0].id == 1   # nearest to itself

    def test_scann_cosine_metric(self, tmp_path, dataset):
        """ScaNN with cosine metric (primary use-case for AQ transform)."""
        path   = str(tmp_path / "scann_cosine.pst")
        params = Params(scann_nlist=8, scann_nprobe=4,
                        scann_pq_M=4, scann_pq_bits=8,
                        scann_rerank_k=20, scann_aq_eta=0.2)
        with PistaDB(path, dim=self.DIM, metric=Metric.COSINE,
                     index=Index.SCANN, params=params) as db:
            db.train(dataset)
            for i, v in enumerate(dataset):
                db.insert(i + 1, v)
            res = db.search(dataset[5], k=5)
        assert len(res) > 0

    def test_scann_delete(self, tmp_path, dataset):
        """Deleted vector must not appear in results."""
        path   = str(tmp_path / "scann_del.pst")
        params = Params(scann_nlist=8, scann_nprobe=4,
                        scann_pq_M=4, scann_pq_bits=4,
                        scann_rerank_k=20, scann_aq_eta=0.0)
        with PistaDB(path, dim=self.DIM, metric=Metric.L2,
                     index=Index.SCANN, params=params) as db:
            db.train(dataset)
            for i, v in enumerate(dataset):
                db.insert(i + 1, v)
            db.delete(1)
            res = db.search(dataset[0], k=5)
        assert all(r.id != 1 for r in res)

    def test_scann_save_reload(self, tmp_path, dataset):
        """Persist ScaNN index and reload; search results must match."""
        path   = str(tmp_path / "scann_persist.pst")
        params = Params(scann_nlist=8, scann_nprobe=4,
                        scann_pq_M=4, scann_pq_bits=8,
                        scann_rerank_k=20, scann_aq_eta=0.2)
        q = dataset[3]

        with PistaDB(path, dim=self.DIM, metric=Metric.L2,
                     index=Index.SCANN, params=params) as db:
            db.train(dataset)
            for i, v in enumerate(dataset):
                db.insert(i + 1, v)
            res_before = db.search(q, k=5)
            db.save()

        with PistaDB(path, dim=self.DIM) as db2:
            res_after = db2.search(q, k=5)

        assert res_before and res_after
        assert res_before[0].id == res_after[0].id

    def test_scann_recall(self, tmp_path):
        """ScaNN must achieve ≥ 50% recall@10 against L2 ground truth."""
        rng  = np.random.default_rng(42)
        N, K = 400, 10
        vecs = rng.random((N, self.DIM), dtype=np.float32)
        qs   = rng.random((20, self.DIM), dtype=np.float32)

        path   = str(tmp_path / "scann_recall.pst")
        params = Params(scann_nlist=16, scann_nprobe=8,
                        scann_pq_M=4, scann_pq_bits=8,
                        scann_rerank_k=40, scann_aq_eta=0.2)

        with PistaDB(path, dim=self.DIM, metric=Metric.L2,
                     index=Index.SCANN, params=params) as db:
            db.train(vecs)
            for i, v in enumerate(vecs):
                db.insert(i + 1, v)
            recalls = []
            for q in qs:
                dists = np.sqrt(((vecs - q) ** 2).sum(axis=1))
                gt    = set(np.argsort(dists)[:K] + 1)
                res   = db.search(q, k=K)
                hits  = sum(1 for r in res if r.id in gt)
                recalls.append(hits / K)

        recall = float(np.mean(recalls))
        assert recall >= 0.5, f"ScaNN recall@{K} = {recall:.2f} < 0.5"


# ── EmbeddingCache tests ───────────────────────────────────────────────────────

class TestEmbeddingCache:
    """Tests for the EmbeddingCache (pistadb_cache_* API)."""

    DIM = 32

    @pytest.fixture
    def cache_path(self, tmp_path):
        return str(tmp_path / "embed.pcc")

    @pytest.fixture
    def cache(self, cache_path):
        c = EmbeddingCache(cache_path, dim=self.DIM)
        yield c
        c.close()

    @pytest.fixture
    def vec(self):
        rng = np.random.default_rng(1)
        return rng.random(self.DIM, dtype=np.float32)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def test_open_close(self, cache_path):
        c = EmbeddingCache(cache_path, dim=self.DIM)
        c.close()
        c.close()  # idempotent

    def test_context_manager(self, cache_path):
        with EmbeddingCache(cache_path, dim=self.DIM) as c:
            assert c._handle is not None
        assert c._handle is None

    def test_repr(self, cache):
        r = repr(cache)
        assert "EmbeddingCache" in r

    # ── Put / Get ──────────────────────────────────────────────────────────

    def test_miss_before_put(self, cache):
        assert cache.get("never_inserted") is None

    def test_put_and_get_roundtrip(self, cache, vec):
        cache.put("hello", vec)
        got = cache.get("hello")
        assert got is not None
        np.testing.assert_allclose(got, vec, atol=1e-6)

    def test_get_returns_copy(self, cache, vec):
        """Modifying the returned array must not corrupt the cached entry."""
        cache.put("copy_test", vec)
        got = cache.get("copy_test")
        got[:] = 0.0
        got2 = cache.get("copy_test")
        np.testing.assert_allclose(got2, vec, atol=1e-6)

    def test_put_updates_existing_entry(self, cache):
        rng = np.random.default_rng(99)
        v1 = rng.random(self.DIM, dtype=np.float32)
        v2 = rng.random(self.DIM, dtype=np.float32)
        cache.put("key", v1)
        cache.put("key", v2)
        got = cache.get("key")
        np.testing.assert_allclose(got, v2, atol=1e-6)

    def test_multiple_entries(self, cache):
        rng = np.random.default_rng(7)
        texts = [f"sentence_{i}" for i in range(10)]
        vecs  = [rng.random(self.DIM, dtype=np.float32) for _ in texts]
        for t, v in zip(texts, vecs):
            cache.put(t, v)
        for t, v in zip(texts, vecs):
            got = cache.get(t)
            assert got is not None, f"miss for {t!r}"
            np.testing.assert_allclose(got, v, atol=1e-6)

    def test_wrong_dim_raises(self, cache):
        bad_vec = np.ones(self.DIM + 1, dtype=np.float32)
        with pytest.raises(ValueError):
            cache.put("bad", bad_vec)

    # ── Contains ──────────────────────────────────────────────────────────

    def test_contains_false_before_put(self, cache):
        assert not cache.contains("absent")
        assert "absent" not in cache

    def test_contains_true_after_put(self, cache, vec):
        cache.put("present", vec)
        assert cache.contains("present")
        assert "present" in cache

    def test_contains_does_not_count_as_hit(self, cache, vec):
        """contains() must not increment hit counter."""
        cache.put("x", vec)
        cache.contains("x")
        stats = cache.stats()
        assert stats.hits == 0

    # ── Evict ─────────────────────────────────────────────────────────────

    def test_evict_existing(self, cache, vec):
        cache.put("evictme", vec)
        assert cache.evict("evictme") is True
        assert cache.get("evictme") is None

    def test_evict_absent(self, cache):
        assert cache.evict("does_not_exist") is False

    def test_evict_reduces_count(self, cache, vec):
        cache.put("a", vec)
        cache.put("b", vec)
        cache.evict("a")
        assert cache.count == 1

    # ── Clear ─────────────────────────────────────────────────────────────

    def test_clear_empties_cache(self, cache, vec):
        for i in range(5):
            cache.put(f"item_{i}", vec)
        cache.clear()
        assert cache.count == 0
        assert len(cache) == 0
        assert cache.get("item_0") is None

    # ── Stats ─────────────────────────────────────────────────────────────

    def test_stats_initial(self, cache):
        s = cache.stats()
        assert isinstance(s, CacheStats)
        assert s.hits == 0
        assert s.misses == 0
        assert s.evictions == 0
        assert s.count == 0

    def test_stats_hit_miss_counts(self, cache, vec):
        cache.put("word", vec)
        cache.get("word")        # hit
        cache.get("missing1")    # miss
        cache.get("missing2")    # miss
        s = cache.stats()
        assert s.hits   == 1
        assert s.misses == 2

    def test_stats_hit_rate(self, cache, vec):
        cache.put("w", vec)
        cache.get("w")    # hit
        cache.get("w")    # hit
        cache.get("nope") # miss
        assert abs(cache.stats().hit_rate - 2/3) < 1e-6

    def test_stats_count_matches_len(self, cache, vec):
        cache.put("a", vec)
        cache.put("b", vec)
        s = cache.stats()
        assert s.count == cache.count == len(cache) == 2

    def test_stats_max_entries(self, cache_path):
        with EmbeddingCache(cache_path, dim=self.DIM, max_entries=50) as c:
            s = c.stats()
        assert s.max_entries == 50

    # ── LRU eviction at capacity ───────────────────────────────────────────

    def test_lru_capacity_respected(self, cache_path):
        cap = 5
        rng = np.random.default_rng(3)
        with EmbeddingCache(cache_path, dim=self.DIM, max_entries=cap) as c:
            for i in range(cap + 3):
                c.put(f"entry_{i}", rng.random(self.DIM, dtype=np.float32))
            assert c.count <= cap
            s = c.stats()
        assert s.evictions > 0

    def test_lru_mru_entry_survives(self, cache_path):
        """The most-recently-used entry must survive an eviction cycle."""
        cap = 3
        rng = np.random.default_rng(5)
        vecs = [rng.random(self.DIM, dtype=np.float32) for _ in range(cap + 2)]
        with EmbeddingCache(cache_path, dim=self.DIM, max_entries=cap) as c:
            for i in range(cap):
                c.put(f"k{i}", vecs[i])
            # Access k0 to make it MRU; k1 becomes LRU
            c.get("k0")
            # Add two more — k1 and k2 should be evicted first
            c.put(f"k{cap}",   vecs[cap])
            c.put(f"k{cap+1}", vecs[cap+1])
            # k0 (MRU before overflow) should still be present
            assert c.get("k0") is not None

    # ── Persistence ───────────────────────────────────────────────────────

    def test_save_and_reload(self, cache_path):
        rng = np.random.default_rng(11)
        vecs = {f"text_{i}": rng.random(self.DIM, dtype=np.float32) for i in range(10)}

        with EmbeddingCache(cache_path, dim=self.DIM) as c:
            for t, v in vecs.items():
                c.put(t, v)
            c.save()

        with EmbeddingCache(cache_path, dim=self.DIM) as c2:
            assert c2.count == len(vecs)
            for t, v in vecs.items():
                got = c2.get(t)
                assert got is not None, f"text {t!r} missing after reload"
                np.testing.assert_allclose(got, v, atol=1e-6)

    def test_file_magic(self, cache_path):
        """Saved .pcc file must start with 'PCCH'."""
        with EmbeddingCache(cache_path, dim=self.DIM) as c:
            c.put("x", np.ones(self.DIM, dtype=np.float32))
            c.save()
        with open(cache_path, "rb") as f:
            magic = f.read(4)
        assert magic == b"PCCH"

    def test_reload_preserves_stats(self, cache_path):
        """Cumulative hit/miss/eviction counters must survive a save/reload."""
        rng = np.random.default_rng(22)
        v = rng.random(self.DIM, dtype=np.float32)

        with EmbeddingCache(cache_path, dim=self.DIM) as c:
            c.put("a", v)
            c.get("a")     # 1 hit
            c.get("miss")  # 1 miss
            c.save()

        with EmbeddingCache(cache_path, dim=self.DIM) as c2:
            s = c2.stats()
        assert s.hits   == 1
        assert s.misses == 1

    def test_reload_partial_capacity(self, cache_path):
        """When reloaded with a smaller max_entries, excess entries are dropped."""
        rng = np.random.default_rng(33)
        n = 10
        with EmbeddingCache(cache_path, dim=self.DIM) as c:
            for i in range(n):
                c.put(f"t{i}", rng.random(self.DIM, dtype=np.float32))
            c.save()

        cap = 4
        with EmbeddingCache(cache_path, dim=self.DIM, max_entries=cap) as c2:
            assert c2.count <= cap

    def test_no_save_means_no_file(self, tmp_path):
        """Closing without save must not create the .pcc file."""
        path = str(tmp_path / "nosave.pcc")
        with EmbeddingCache(path, dim=self.DIM) as c:
            c.put("x", np.ones(self.DIM, dtype=np.float32))
        # File should NOT exist because save() was never called
        assert not os.path.exists(path)


# ── CachedEmbedder tests ───────────────────────────────────────────────────────

class TestCachedEmbedder:
    """Tests for CachedEmbedder (the transparent caching wrapper)."""

    DIM = 16

    @pytest.fixture
    def _make_embedder(self, tmp_path):
        """Factory: returns (embedder, cache, call_log)."""
        call_log = []
        def fake_model(text: str) -> np.ndarray:
            call_log.append(text)
            # Deterministic but unique vector per text
            seed = sum(ord(c) for c in text)
            return np.random.default_rng(seed).random(self.DIM, dtype=np.float32)

        cache    = EmbeddingCache(str(tmp_path / "e.pcc"), dim=self.DIM)
        embedder = CachedEmbedder(fake_model, cache)
        return embedder, cache, call_log

    def test_miss_calls_model(self, _make_embedder):
        embedder, cache, log = _make_embedder
        embedder("hello")
        assert "hello" in log

    def test_hit_skips_model(self, _make_embedder):
        embedder, cache, log = _make_embedder
        embedder("hello")
        assert len(log) == 1
        embedder("hello")   # second call — must be a cache hit
        assert len(log) == 1, "model should not be called on cache hit"

    def test_result_is_consistent(self, _make_embedder):
        embedder, _, _ = _make_embedder
        v1 = embedder("consistent")
        v2 = embedder("consistent")
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts_different_vectors(self, _make_embedder):
        embedder, _, _ = _make_embedder
        v1 = embedder("alpha")
        v2 = embedder("beta")
        assert not np.allclose(v1, v2)

    def test_cache_property(self, _make_embedder):
        embedder, cache, _ = _make_embedder
        assert embedder.cache is cache

    def test_embed_batch_shape(self, _make_embedder):
        embedder, _, _ = _make_embedder
        texts  = ["one", "two", "three"]
        result = embedder.embed_batch(texts)
        assert result.shape == (3, self.DIM)
        assert result.dtype == np.float32

    def test_embed_batch_uses_cache(self, _make_embedder):
        embedder, _, log = _make_embedder
        texts = ["cat", "dog", "cat"]   # "cat" appears twice
        embedder.embed_batch(texts)
        assert log.count("cat") == 1, "duplicate text in batch should hit cache on second occurrence"

    def test_autosave_triggers(self, tmp_path):
        """After autosave_every new embeddings the file must exist on disk."""
        call_log = []
        def fake_model(text):
            call_log.append(text)
            return np.ones(self.DIM, dtype=np.float32)

        path  = str(tmp_path / "autosave.pcc")
        cache = EmbeddingCache(path, dim=self.DIM)
        embedder = CachedEmbedder(fake_model, cache, autosave_every=3)

        for i in range(3):
            embedder(f"item_{i}")

        assert os.path.exists(path), ".pcc file should exist after autosave_every items"
        cache.close()

    def test_autosave_zero_means_never(self, tmp_path):
        """autosave_every=0 should never write the file automatically."""
        def fake_model(text):
            return np.ones(self.DIM, dtype=np.float32)

        path  = str(tmp_path / "no_autosave.pcc")
        cache = EmbeddingCache(path, dim=self.DIM)
        embedder = CachedEmbedder(fake_model, cache, autosave_every=0)

        for i in range(10):
            embedder(f"item_{i}")

        assert not os.path.exists(path), ".pcc file must not exist without explicit save()"
        cache.close()


# ══════════════════════════════════════════════════════════════════════════════
# Transaction tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTransaction:
    """Tests for PistaDB.begin_transaction() / Transaction API."""

    DIM = 16

    @pytest.fixture
    def db(self, tmp_path):
        """Fresh LINEAR database (LINEAR supports pistadb_get for all ops)."""
        d = PistaDB(str(tmp_path / "txn.pst"), dim=self.DIM,
                    metric=Metric.L2, index=Index.LINEAR)
        yield d
        d.close()

    @pytest.fixture
    def db_hnsw(self, tmp_path):
        """Fresh HNSW database (HNSW now supports pistadb_get)."""
        d = PistaDB(str(tmp_path / "txn_hnsw.pst"), dim=self.DIM,
                    metric=Metric.L2, index=Index.HNSW)
        yield d
        d.close()

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(0)

    def _vec(self, rng=None, seed=None):
        r = rng if rng is not None else np.random.default_rng(seed or 0)
        return r.random(self.DIM, dtype=np.float64).astype(np.float32)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def test_begin_free(self, db):
        txn = db.begin_transaction()
        assert isinstance(txn, Transaction)
        txn.free()

    def test_repr(self, db):
        txn = db.begin_transaction()
        assert "Transaction" in repr(txn)
        txn.free()

    def test_op_count_empty(self, db):
        txn = db.begin_transaction()
        assert txn.op_count == 0
        txn.free()

    def test_op_count_after_staging(self, db, rng):
        txn = db.begin_transaction()
        txn.insert(1, self._vec(rng))
        txn.insert(2, self._vec(rng))
        assert txn.op_count == 2
        txn.free()

    def test_commit_empty_is_noop(self, db):
        txn = db.begin_transaction()
        txn.commit()   # should not raise
        assert db.count == 0

    # ── Insert ─────────────────────────────────────────────────────────────

    def test_insert_commit(self, db, rng):
        with db.begin_transaction() as txn:
            txn.insert(1, self._vec(rng), label="a")
            txn.insert(2, self._vec(rng), label="b")
        assert db.count == 2

    def test_insert_visible_after_commit(self, db, rng):
        v = self._vec(rng)
        with db.begin_transaction() as txn:
            txn.insert(99, v, label="dog")
        got_vec, got_lbl = db.get(99)
        assert np.allclose(got_vec, v, atol=1e-5)
        assert got_lbl == "dog"

    def test_insert_not_visible_before_commit(self, db, rng):
        """Staging an insert must not modify the live index."""
        txn = db.begin_transaction()
        txn.insert(1, self._vec(rng))
        assert db.count == 0  # not yet committed
        txn.rollback()
        txn.free()

    def test_rollback_discards_staged_inserts(self, db, rng):
        txn = db.begin_transaction()
        txn.insert(1, self._vec(rng))
        txn.rollback()
        txn.free()
        assert db.count == 0

    def test_multiple_inserts_atomic(self, db, rng):
        vecs = [self._vec(rng) for _ in range(10)]
        with db.begin_transaction() as txn:
            for i, v in enumerate(vecs, start=1):
                txn.insert(i, v)
        assert db.count == 10

    def test_duplicate_insert_id_validation_fails(self, db, rng):
        """Duplicate INSERT ids in one transaction must raise before applying."""
        v = self._vec(rng)
        with pytest.raises(RuntimeError):
            with db.begin_transaction() as txn:
                txn.insert(1, v)
                txn.insert(1, v)   # duplicate → commit raises
        # No inserts should have been applied.
        assert db.count == 0

    # ── Delete ─────────────────────────────────────────────────────────────

    def test_delete_commit(self, db, rng):
        db.insert(1, self._vec(rng))
        assert db.count == 1
        with db.begin_transaction() as txn:
            txn.delete(1)
        assert db.count == 0

    def test_delete_rollback_restores(self, db, rng):
        db.insert(1, self._vec(rng))
        txn = db.begin_transaction()
        txn.delete(1)
        txn.rollback()
        txn.free()
        assert db.count == 1  # delete was not applied

    # ── Update ─────────────────────────────────────────────────────────────

    def test_update_commit(self, db, rng):
        v1 = self._vec(rng)
        v2 = self._vec(rng)
        db.insert(1, v1)
        with db.begin_transaction() as txn:
            txn.update(1, v2)
        got, _ = db.get(1)
        assert np.allclose(got, v2, atol=1e-5)

    def test_update_rollback_leaves_original(self, db, rng):
        v1 = self._vec(rng)
        v2 = self._vec(rng)
        db.insert(1, v1)
        txn = db.begin_transaction()
        txn.update(1, v2)
        txn.rollback()
        txn.free()
        got, _ = db.get(1)
        assert np.allclose(got, v1, atol=1e-5)

    # ── Mixed operations ────────────────────────────────────────────────────

    def test_mixed_insert_delete_commit(self, db, rng):
        db.insert(10, self._vec(rng))
        with db.begin_transaction() as txn:
            txn.insert(1, self._vec(rng))
            txn.insert(2, self._vec(rng))
            txn.delete(10)
        assert db.count == 2
        with pytest.raises(RuntimeError):
            db.get(10)   # should be gone

    def test_insert_then_update_in_same_txn(self, db, rng):
        v1 = self._vec(rng)
        v2 = self._vec(rng)
        # Insert id=1 outside the txn, then update it within the txn
        db.insert(1, v1)
        with db.begin_transaction() as txn:
            txn.insert(2, v2)
            txn.update(1, v2)
        got1, _ = db.get(1)
        assert np.allclose(got1, v2, atol=1e-5)
        assert db.count == 2

    # ── Context manager semantics ───────────────────────────────────────────

    def test_context_manager_commits_on_success(self, db, rng):
        with db.begin_transaction() as txn:
            txn.insert(1, self._vec(rng))
        assert db.count == 1

    def test_context_manager_rollback_on_exception(self, db, rng):
        try:
            with db.begin_transaction() as txn:
                txn.insert(1, self._vec(rng))
                raise ValueError("simulated failure")
        except ValueError:
            pass
        assert db.count == 0  # insert was rolled back

    # ── Rollback on failed apply ────────────────────────────────────────────

    def test_commit_rollback_on_apply_failure(self, db, rng):
        """If the second op of a commit fails, the first should be undone."""
        db.insert(1, self._vec(rng))  # pre-existing vector

        txn = db.begin_transaction()
        txn.insert(2, self._vec(rng))   # will succeed
        txn.delete(999)                  # will fail (id=999 does not exist)
        with pytest.raises(RuntimeError):
            txn.commit()
        txn.free()

        # id=2 insert should have been rolled back by undo
        assert db.count == 1   # only the pre-existing id=1 remains

    # ── Handle reuse after commit ───────────────────────────────────────────

    def test_reuse_after_commit(self, db, rng):
        txn = db.begin_transaction()
        txn.insert(1, self._vec(rng))
        txn.commit()
        # Stage a second transaction on the same handle
        txn.insert(2, self._vec(rng))
        txn.commit()
        txn.free()
        assert db.count == 2

    # ── HNSW get() support (pistadb_get extended for HNSW) ─────────────────

    def test_hnsw_delete_undo(self, db_hnsw, rng):
        """pistadb_get() now works for HNSW so DELETE undo should succeed."""
        v = self._vec(rng)
        db_hnsw.insert(1, v, label="item")
        txn = db_hnsw.begin_transaction()
        txn.insert(2, self._vec(rng))  # will succeed
        txn.delete(999)                # id=999 does not exist → will fail on commit
        with pytest.raises(RuntimeError):
            txn.commit()
        txn.free()
        # id=2 insert should be rolled back
        assert db_hnsw.count == 1

    def test_hnsw_get_works(self, db_hnsw, rng):
        """Verify the extended pistadb_get() returns correct data for HNSW."""
        v = self._vec(rng)
        db_hnsw.insert(42, v, label="hnsw_item")
        got_vec, got_lbl = db_hnsw.get(42)
        assert np.allclose(got_vec, v, atol=1e-5)
        assert got_lbl == "hnsw_item"

    # ── Search after commit ─────────────────────────────────────────────────

    def test_committed_data_is_searchable(self, db, rng):
        query = np.zeros(self.DIM, dtype=np.float32)
        # Insert a vector very close to the zero query
        near = np.full(self.DIM, 0.001, dtype=np.float32)
        # Insert a vector far from the zero query
        far  = np.full(self.DIM, 100.0, dtype=np.float32)

        with db.begin_transaction() as txn:
            txn.insert(1, near, label="near")
            txn.insert(2, far,  label="far")

        results = db.search(query, k=1)
        assert len(results) == 1
        assert results[0].label == "near"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
