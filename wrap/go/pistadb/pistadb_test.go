package pistadb_test

import (
	"math"
	"math/rand"
	"testing"

	pistadb "pistadb.io/go/pistadb"
)

const testDim = 64

func randVec(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()*2 - 1
	}
	return v
}

func vecClose(a, b []float32, tol float32) bool {
	for i := range a {
		if float32(math.Abs(float64(a[i]-b[i]))) > tol {
			return false
		}
	}
	return true
}

// ── Database tests ────────────────────────────────────────────────────────────

func TestOpenClose(t *testing.T) {
	path := t.TempDir() + "/test.pst"

	db, err := pistadb.Open(path, testDim, pistadb.MetricL2, pistadb.IndexLinear, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	db.Close()
	db.Close() // second Close must be a no-op
}

func TestDefaultParams(t *testing.T) {
	p := pistadb.DefaultParams()
	if p.HNSWM != 16 {
		t.Errorf("HNSWM: want 16, got %d", p.HNSWM)
	}
	if p.HNSWEfConstruction != 200 {
		t.Errorf("HNSWEfConstruction: want 200, got %d", p.HNSWEfConstruction)
	}
	if p.HNSWEfSearch != 50 {
		t.Errorf("HNSWEfSearch: want 50, got %d", p.HNSWEfSearch)
	}
	if p.IVFNList != 128 {
		t.Errorf("IVFNList: want 128, got %d", p.IVFNList)
	}
}

func TestInsertSearch(t *testing.T) {
	path := t.TempDir() + "/test.pst"

	db, err := pistadb.Open(path, testDim, pistadb.MetricL2, pistadb.IndexLinear, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	const n = 100
	vecs := make([][]float32, n)
	for i := 0; i < n; i++ {
		vecs[i] = randVec(testDim)
		if err := db.Insert(uint64(i+1), "label", vecs[i]); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}

	if db.Count() != n {
		t.Errorf("Count: want %d, got %d", n, db.Count())
	}

	results, err := db.Search(vecs[0], 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("Search returned no results")
	}
	// The nearest neighbour of vecs[0] must be itself (id=1, distance≈0).
	if results[0].ID != 1 {
		t.Errorf("nearest: want id=1, got id=%d (dist=%.6f)", results[0].ID, results[0].Distance)
	}
	if results[0].Distance > 1e-4 {
		t.Errorf("self-distance: want ~0, got %f", results[0].Distance)
	}
}

func TestInsertDuplicateID(t *testing.T) {
	path := t.TempDir() + "/test.pst"
	db, err := pistadb.Open(path, testDim, pistadb.MetricL2, pistadb.IndexLinear, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	vec := randVec(testDim)
	if err := db.Insert(1, "", vec); err != nil {
		t.Fatalf("first Insert: %v", err)
	}
	if err := db.Insert(1, "", vec); err == nil {
		t.Error("second Insert with same id: expected error, got nil")
	}
}

func TestGet(t *testing.T) {
	path := t.TempDir() + "/test.pst"
	db, err := pistadb.Open(path, testDim, pistadb.MetricL2, pistadb.IndexLinear, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	vec := randVec(testDim)
	if err := db.Insert(7, "cat", vec); err != nil {
		t.Fatalf("Insert: %v", err)
	}

	got, label, err := db.Get(7)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if label != "cat" {
		t.Errorf("label: want %q, got %q", "cat", label)
	}
	if !vecClose(vec, got, 1e-6) {
		t.Error("Get returned different vector")
	}

	_, _, err = db.Get(999)
	if err == nil {
		t.Error("Get non-existent: expected error, got nil")
	}
}

func TestDeleteUpdate(t *testing.T) {
	path := t.TempDir() + "/test.pst"
	db, err := pistadb.Open(path, testDim, pistadb.MetricL2, pistadb.IndexLinear, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	vec := randVec(testDim)
	if err := db.Insert(1, "dog", vec); err != nil {
		t.Fatal(err)
	}

	vec2 := randVec(testDim)
	if err := db.Update(1, vec2); err != nil {
		t.Fatalf("Update: %v", err)
	}
	got, _, err := db.Get(1)
	if err != nil {
		t.Fatalf("Get after Update: %v", err)
	}
	if !vecClose(vec2, got, 1e-6) {
		t.Error("Get after Update returned wrong vector")
	}

	if err := db.Delete(1); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, _, err := db.Get(1); err == nil {
		t.Error("Get after Delete: expected error, got nil")
	}

	// Update on non-existent id should return an error.
	if err := db.Update(999, vec); err == nil {
		t.Error("Update non-existent: expected error, got nil")
	}
}

func TestSaveLoad(t *testing.T) {
	path := t.TempDir() + "/persist.pst"

	vec := randVec(testDim)

	// Write and save.
	{
		db, err := pistadb.Open(path, testDim, pistadb.MetricL2, pistadb.IndexLinear, nil)
		if err != nil {
			t.Fatal(err)
		}
		if err := db.Insert(42, "persisted", vec); err != nil {
			t.Fatal(err)
		}
		if err := db.Save(); err != nil {
			t.Fatalf("Save: %v", err)
		}
		db.Close()
	}

	// Re-open and verify.
	{
		db, err := pistadb.Open(path, testDim, pistadb.MetricL2, pistadb.IndexLinear, nil)
		if err != nil {
			t.Fatalf("reload Open: %v", err)
		}
		defer db.Close()

		if db.Count() != 1 {
			t.Fatalf("Count after reload: want 1, got %d", db.Count())
		}
		got, label, err := db.Get(42)
		if err != nil {
			t.Fatalf("Get after reload: %v", err)
		}
		if label != "persisted" {
			t.Errorf("label: want %q, got %q", "persisted", label)
		}
		if !vecClose(vec, got, 1e-6) {
			t.Error("Get after reload returned different vector")
		}
	}
}

func TestMetadata(t *testing.T) {
	path := t.TempDir() + "/meta.pst"
	db, err := pistadb.Open(path, testDim, pistadb.MetricCosine, pistadb.IndexHNSW, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	if db.Dim() != testDim {
		t.Errorf("Dim: want %d, got %d", testDim, db.Dim())
	}
	if db.Metric() != pistadb.MetricCosine {
		t.Errorf("Metric: want Cosine, got %s", db.Metric())
	}
	if db.IndexType() != pistadb.IndexHNSW {
		t.Errorf("IndexType: want HNSW, got %s", db.IndexType())
	}
}

func TestVersion(t *testing.T) {
	v := pistadb.Version()
	if v == "" {
		t.Error("Version returned empty string")
	}
	t.Logf("PistaDB version: %s", v)
}

// ── Batch tests ───────────────────────────────────────────────────────────────

func TestBatchInsert(t *testing.T) {
	path := t.TempDir() + "/batch.pst"
	db, err := pistadb.Open(path, testDim, pistadb.MetricL2, pistadb.IndexLinear, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	const n = 200
	ids := make([]uint64, n)
	labels := make([]string, n)
	vecs := make([][]float32, n)
	for i := 0; i < n; i++ {
		ids[i] = uint64(i + 1)
		labels[i] = ""
		vecs[i] = randVec(testDim)
	}

	failures, err := pistadb.BatchInsert(db, ids, labels, vecs, 0)
	if err != nil {
		t.Fatalf("BatchInsert: %v", err)
	}
	if failures != 0 {
		t.Errorf("BatchInsert: %d failures", failures)
	}
	if db.Count() != n {
		t.Errorf("Count: want %d, got %d", n, db.Count())
	}
}

func TestNewBatchStreamingFlush(t *testing.T) {
	path := t.TempDir() + "/batch2.pst"
	db, err := pistadb.Open(path, testDim, pistadb.MetricL2, pistadb.IndexLinear, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	b, err := pistadb.NewBatch(db, 2, 0)
	if err != nil {
		t.Fatalf("NewBatch: %v", err)
	}
	defer b.Destroy()

	const n = 50
	for i := 0; i < n; i++ {
		if err := b.Push(uint64(i+1), "", randVec(testDim)); err != nil {
			t.Fatalf("Push %d: %v", i, err)
		}
	}

	if failed := b.Flush(); failed != 0 {
		t.Errorf("Flush: %d failures", failed)
	}
	if db.Count() != n {
		t.Errorf("Count after flush: want %d, got %d", n, db.Count())
	}
	if b.ErrorCount() != 0 {
		t.Errorf("ErrorCount: want 0, got %d", b.ErrorCount())
	}
}

// ── Cache tests ───────────────────────────────────────────────────────────────

func TestCacheGetPut(t *testing.T) {
	path := t.TempDir() + "/embed.pcc"

	c, err := pistadb.OpenCache(path, testDim, 100)
	if err != nil {
		t.Fatalf("OpenCache: %v", err)
	}
	defer c.Close()

	if c.Contains("hello") {
		t.Error("Contains before Put: expected false")
	}

	vec := randVec(testDim)
	if err := c.Put("hello", vec); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if !c.Contains("hello") {
		t.Error("Contains after Put: expected true")
	}
	if c.Count() != 1 {
		t.Errorf("Count: want 1, got %d", c.Count())
	}

	got, ok := c.Get("hello")
	if !ok {
		t.Fatal("Get: expected hit")
	}
	if !vecClose(vec, got, 1e-6) {
		t.Error("Get returned different vector")
	}

	_, ok2 := c.Get("world")
	if ok2 {
		t.Error("Get on missing key: expected miss")
	}

	stats := c.Stats()
	if stats.Hits != 1 {
		t.Errorf("Hits: want 1, got %d", stats.Hits)
	}
	if stats.Misses != 1 {
		t.Errorf("Misses: want 1, got %d", stats.Misses)
	}
}

func TestCacheEvict(t *testing.T) {
	path := t.TempDir() + "/evict.pcc"
	c, err := pistadb.OpenCache(path, testDim, 100)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	if err := c.Put("key", randVec(testDim)); err != nil {
		t.Fatal(err)
	}
	if !c.Evict("key") {
		t.Error("Evict existing: expected true")
	}
	if c.Evict("key") {
		t.Error("Evict missing: expected false")
	}
	if c.Count() != 0 {
		t.Errorf("Count after evict: want 0, got %d", c.Count())
	}
}

func TestCacheLRUCapacity(t *testing.T) {
	const cap = 5
	path := t.TempDir() + "/lru.pcc"
	c, err := pistadb.OpenCache(path, testDim, cap)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	for i := 0; i < cap+3; i++ {
		key := string(rune('a' + i))
		if err := c.Put(key, randVec(testDim)); err != nil {
			t.Fatalf("Put %q: %v", key, err)
		}
	}
	if c.Count() > cap {
		t.Errorf("Count exceeds cap: got %d, max %d", c.Count(), cap)
	}
	stats := c.Stats()
	if stats.Evictions == 0 {
		t.Error("expected at least one LRU eviction")
	}
}

func TestCacheSavePersistence(t *testing.T) {
	path := t.TempDir() + "/persist.pcc"
	vec := randVec(testDim)

	// Write and save.
	{
		c, err := pistadb.OpenCache(path, testDim, 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := c.Put("sentence", vec); err != nil {
			t.Fatal(err)
		}
		if err := c.Save(); err != nil {
			t.Fatalf("Save: %v", err)
		}
		c.Close()
	}

	// Reload and verify.
	{
		c, err := pistadb.OpenCache(path, testDim, 0)
		if err != nil {
			t.Fatalf("reload OpenCache: %v", err)
		}
		defer c.Close()

		got, ok := c.Get("sentence")
		if !ok {
			t.Fatal("Get after reload: expected hit")
		}
		if !vecClose(vec, got, 1e-6) {
			t.Error("Get after reload returned different vector")
		}
	}
}

func TestCacheClear(t *testing.T) {
	path := t.TempDir() + "/clear.pcc"
	c, err := pistadb.OpenCache(path, testDim, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	for i := 0; i < 10; i++ {
		_ = c.Put(string(rune('a'+i)), randVec(testDim))
	}
	c.Clear()
	if c.Count() != 0 {
		t.Errorf("Count after Clear: want 0, got %d", c.Count())
	}
}
