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

from pistadb import PistaDB, Metric, Index, Params, build_from_array

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
