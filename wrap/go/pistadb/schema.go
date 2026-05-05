// Milvus-style schema layer for PistaDB.
//
// PistaDB's C core stores `(uint64 id, char[256] label, float[dim] vec)` per
// row.  This file adds a Go-side wrapper that maps a multi-field schema onto
// that storage:
//
//   - the primary INT64 field becomes the row id,
//   - the FLOAT_VECTOR field is stored in the .pst file,
//   - all other scalar fields are kept in a JSON sidecar (<path>.meta.json).
//
// Quick start:
//
//	fields := []pistadb.FieldSchema{
//	    {Name: "lc_id", DType: pistadb.DTypeInt64, IsPrimary: true, AutoID: true},
//	    {Name: "lc_section", DType: pistadb.DTypeVarChar, MaxLength: 100},
//	    {Name: "lc_vector", DType: pistadb.DTypeFloatVector, Dim: 1536},
//	}
//	coll, err := pistadb.CreateCollection(
//	    "common_text", fields, "Common text search",
//	    pistadb.CollectionOptions{
//	        Metric: pistadb.MetricCosine,
//	        Index:  pistadb.IndexHNSW,
//	        BaseDir: "./db",
//	    })
//	if err != nil { log.Fatal(err) }
//	defer coll.Close()
//
//	ids, err := coll.Insert([]map[string]any{
//	    {"lc_section": "common", "lc_vector": vec1},
//	    {"lc_section": "common", "lc_vector": vec2},
//	})
//	hits, err := coll.Search(query, 10, nil)
package pistadb

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
)

// ── DataType ──────────────────────────────────────────────────────────────────

// DataType is the per-field data type (mirrors pymilvus.DataType).
type DataType int

const (
	DTypeBool        DataType = 1
	DTypeInt8        DataType = 2
	DTypeInt16       DataType = 3
	DTypeInt32       DataType = 4
	DTypeInt64       DataType = 5
	DTypeFloat       DataType = 10
	DTypeDouble      DataType = 11
	DTypeVarChar     DataType = 21
	DTypeJSON        DataType = 23
	DTypeFloatVector DataType = 101
)

func (d DataType) String() string {
	switch d {
	case DTypeBool:
		return "BOOL"
	case DTypeInt8:
		return "INT8"
	case DTypeInt16:
		return "INT16"
	case DTypeInt32:
		return "INT32"
	case DTypeInt64:
		return "INT64"
	case DTypeFloat:
		return "FLOAT"
	case DTypeDouble:
		return "DOUBLE"
	case DTypeVarChar:
		return "VARCHAR"
	case DTypeJSON:
		return "JSON"
	case DTypeFloatVector:
		return "FLOAT_VECTOR"
	default:
		return fmt.Sprintf("DataType(%d)", int(d))
	}
}

func parseDataType(s string) (DataType, error) {
	switch s {
	case "BOOL":
		return DTypeBool, nil
	case "INT8":
		return DTypeInt8, nil
	case "INT16":
		return DTypeInt16, nil
	case "INT32":
		return DTypeInt32, nil
	case "INT64":
		return DTypeInt64, nil
	case "FLOAT":
		return DTypeFloat, nil
	case "DOUBLE":
		return DTypeDouble, nil
	case "VARCHAR":
		return DTypeVarChar, nil
	case "JSON":
		return DTypeJSON, nil
	case "FLOAT_VECTOR":
		return DTypeFloatVector, nil
	}
	return 0, fmt.Errorf("pistadb: unknown DataType %q", s)
}

func (d DataType) isInt() bool {
	return d == DTypeInt8 || d == DTypeInt16 || d == DTypeInt32 || d == DTypeInt64
}

func (d DataType) isFloat() bool {
	return d == DTypeFloat || d == DTypeDouble
}

// ── FieldSchema ──────────────────────────────────────────────────────────────

// FieldSchema describes one field in a collection.  Mirrors
// pymilvus.FieldSchema.
type FieldSchema struct {
	Name        string
	DType       DataType
	IsPrimary   bool
	AutoID      bool
	MaxLength   int    // VARCHAR only; 0 = no limit
	Dim         int    // FLOAT_VECTOR only; required
	Description string
}

func (f *FieldSchema) validate() error {
	if f.Name == "" {
		return errors.New("pistadb: field name must not be empty")
	}
	if f.DType == DTypeFloatVector && f.Dim <= 0 {
		return fmt.Errorf("pistadb: FLOAT_VECTOR field %q requires positive Dim", f.Name)
	}
	if f.DType == DTypeVarChar && f.MaxLength < 0 {
		return fmt.Errorf("pistadb: VARCHAR field %q: MaxLength must be >= 0", f.Name)
	}
	if f.IsPrimary && f.DType != DTypeInt64 {
		return fmt.Errorf("pistadb: primary key field %q must be INT64", f.Name)
	}
	if f.AutoID && !f.IsPrimary {
		return fmt.Errorf("pistadb: AutoID is only valid on the primary field (got it on %q)", f.Name)
	}
	return nil
}

type fieldSchemaJSON struct {
	Name        string `json:"name"`
	DType       string `json:"dtype"`
	IsPrimary   bool   `json:"is_primary"`
	AutoID      bool   `json:"auto_id"`
	MaxLength   int    `json:"max_length,omitempty"`
	Dim         int    `json:"dim,omitempty"`
	Description string `json:"description,omitempty"`
}

func (f FieldSchema) toJSON() fieldSchemaJSON {
	return fieldSchemaJSON{
		Name:        f.Name,
		DType:       f.DType.String(),
		IsPrimary:   f.IsPrimary,
		AutoID:      f.AutoID,
		MaxLength:   f.MaxLength,
		Dim:         f.Dim,
		Description: f.Description,
	}
}

func (j fieldSchemaJSON) toField() (FieldSchema, error) {
	dt, err := parseDataType(j.DType)
	if err != nil {
		return FieldSchema{}, err
	}
	return FieldSchema{
		Name:        j.Name,
		DType:       dt,
		IsPrimary:   j.IsPrimary,
		AutoID:      j.AutoID,
		MaxLength:   j.MaxLength,
		Dim:         j.Dim,
		Description: j.Description,
	}, nil
}

// ── CollectionSchema ──────────────────────────────────────────────────────────

// CollectionSchema is an ordered list of FieldSchema with a description.
//
// A valid schema must contain exactly one primary key (INT64) field, exactly
// one FLOAT_VECTOR field, and unique field names.
type CollectionSchema struct {
	Fields      []FieldSchema
	Description string
}

// NewCollectionSchema constructs and validates a schema.
func NewCollectionSchema(fields []FieldSchema, description string) (*CollectionSchema, error) {
	s := &CollectionSchema{Fields: append([]FieldSchema(nil), fields...), Description: description}
	if err := s.validate(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *CollectionSchema) validate() error {
	seen := make(map[string]bool)
	primaryCount := 0
	vectorCount := 0
	for i := range s.Fields {
		f := &s.Fields[i]
		if err := f.validate(); err != nil {
			return err
		}
		if seen[f.Name] {
			return fmt.Errorf("pistadb: duplicate field name %q", f.Name)
		}
		seen[f.Name] = true
		if f.IsPrimary {
			primaryCount++
		}
		if f.DType == DTypeFloatVector {
			vectorCount++
		}
	}
	if primaryCount != 1 {
		return fmt.Errorf("pistadb: schema must have exactly one primary key (found %d)", primaryCount)
	}
	if vectorCount != 1 {
		return fmt.Errorf("pistadb: schema must have exactly one FLOAT_VECTOR field (found %d)", vectorCount)
	}
	return nil
}

// PrimaryField returns the primary key FieldSchema.
func (s *CollectionSchema) PrimaryField() FieldSchema {
	for _, f := range s.Fields {
		if f.IsPrimary {
			return f
		}
	}
	return FieldSchema{} // unreachable after validate()
}

// VectorField returns the FLOAT_VECTOR FieldSchema.
func (s *CollectionSchema) VectorField() FieldSchema {
	for _, f := range s.Fields {
		if f.DType == DTypeFloatVector {
			return f
		}
	}
	return FieldSchema{} // unreachable after validate()
}

// ScalarFields returns non-primary, non-vector fields in declaration order.
func (s *CollectionSchema) ScalarFields() []FieldSchema {
	out := make([]FieldSchema, 0, len(s.Fields))
	for _, f := range s.Fields {
		if !f.IsPrimary && f.DType != DTypeFloatVector {
			out = append(out, f)
		}
	}
	return out
}

// Field returns the FieldSchema for the given name, or an error.
func (s *CollectionSchema) Field(name string) (FieldSchema, error) {
	for _, f := range s.Fields {
		if f.Name == name {
			return f, nil
		}
	}
	return FieldSchema{}, fmt.Errorf("pistadb: no field named %q", name)
}

// ── Hit ──────────────────────────────────────────────────────────────────────

// Hit is one row returned by Collection.Search.  Fields holds the projected
// scalar columns (and optionally the vector / primary id when requested).
type Hit struct {
	ID       uint64
	Distance float32
	Fields   map[string]any
}

// Get retrieves a field value, or the zero value if absent.
func (h *Hit) Get(name string) any { return h.Fields[name] }

// ── Sidecar payload ──────────────────────────────────────────────────────────

const sidecarVersion = 1

type sidecar struct {
	Version     int                       `json:"version"`
	Name        string                    `json:"name"`
	Description string                    `json:"description"`
	Metric      string                    `json:"metric"`
	Index       string                    `json:"index"`
	NextID      uint64                    `json:"next_id"`
	Fields      []fieldSchemaJSON         `json:"fields"`
	Rows        map[string]map[string]any `json:"rows"`
}

// ── CollectionOptions ────────────────────────────────────────────────────────

// CollectionOptions configures CreateCollection / LoadCollection.
type CollectionOptions struct {
	Metric  Metric
	Index   IndexType
	Params  *Params
	BaseDir string // optional; if empty, files go in the current working dir
	Path    string // optional; explicit .pst path overrides BaseDir+Name
}

func (o CollectionOptions) resolvePath(name string) string {
	if o.Path != "" {
		return o.Path
	}
	if o.BaseDir == "" {
		return name + ".pst"
	}
	return filepath.Join(o.BaseDir, name+".pst")
}

func metricFromString(s string) (Metric, error) {
	switch s {
	case "L2":
		return MetricL2, nil
	case "Cosine":
		return MetricCosine, nil
	case "InnerProduct", "IP":
		return MetricIP, nil
	case "L1":
		return MetricL1, nil
	case "Hamming":
		return MetricHamming, nil
	}
	return MetricL2, fmt.Errorf("pistadb: unknown metric %q", s)
}

func indexFromString(s string) (IndexType, error) {
	switch s {
	case "Linear":
		return IndexLinear, nil
	case "HNSW":
		return IndexHNSW, nil
	case "IVF":
		return IndexIVF, nil
	case "IVF_PQ":
		return IndexIVFPQ, nil
	case "DiskANN":
		return IndexDiskANN, nil
	case "LSH":
		return IndexLSH, nil
	case "ScaNN":
		return IndexSCANN, nil
	case "SQ":
		return IndexSQ, nil
	}
	return IndexHNSW, fmt.Errorf("pistadb: unknown index type %q", s)
}

// ── Collection ────────────────────────────────────────────────────────────────

// Collection is a schema-backed wrapper over a Database, plus a JSON sidecar
// holding scalar fields for each row.
type Collection struct {
	Name   string
	Schema *CollectionSchema

	db      *Database
	path    string
	meta    string
	metric  Metric
	index   IndexType
	rows    map[uint64]map[string]any
	nextID  uint64
}

// CreateCollection creates a new collection and the underlying .pst /
// sidecar files.  Fails if either file already exists (no overwrite).
func CreateCollection(name string, fields []FieldSchema, description string, opt CollectionOptions) (*Collection, error) {
	schema, err := NewCollectionSchema(fields, description)
	if err != nil {
		return nil, err
	}
	pstPath := opt.resolvePath(name)
	metaPath := pstPath + ".meta.json"

	for _, p := range []string{pstPath, metaPath} {
		if _, err := os.Stat(p); err == nil {
			return nil, fmt.Errorf("pistadb: %s already exists", p)
		}
	}
	if dir := filepath.Dir(pstPath); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("pistadb: mkdir %s: %w", dir, err)
		}
	}

	vec := schema.VectorField()
	metric := opt.Metric
	idx := opt.Index
	if idx == 0 && metric == 0 {
		// zero values are fine: MetricL2 / IndexLinear; nothing to fix.
	}

	db, err := Open(pstPath, vec.Dim, metric, idx, opt.Params)
	if err != nil {
		return nil, err
	}
	c := &Collection{
		Name:   name,
		Schema: schema,
		db:     db,
		path:   pstPath,
		meta:   metaPath,
		metric: metric,
		index:  idx,
		rows:   make(map[uint64]map[string]any),
		nextID: 1,
	}
	if err := c.saveSidecar(); err != nil {
		_ = db.Save()
		db.Close()
		return nil, err
	}
	return c, nil
}

// LoadCollection re-opens an existing collection.  Either name (resolved
// against opt.BaseDir) or opt.Path must be provided.
func LoadCollection(name string, opt CollectionOptions) (*Collection, error) {
	pstPath := opt.resolvePath(name)
	metaPath := pstPath + ".meta.json"

	data, err := os.ReadFile(metaPath)
	if err != nil {
		return nil, fmt.Errorf("pistadb: read sidecar %s: %w", metaPath, err)
	}
	var sc sidecar
	if err := json.Unmarshal(data, &sc); err != nil {
		return nil, fmt.Errorf("pistadb: parse sidecar: %w", err)
	}

	fields := make([]FieldSchema, 0, len(sc.Fields))
	for _, fj := range sc.Fields {
		f, err := fj.toField()
		if err != nil {
			return nil, err
		}
		fields = append(fields, f)
	}
	schema, err := NewCollectionSchema(fields, sc.Description)
	if err != nil {
		return nil, err
	}
	metric, err := metricFromString(sc.Metric)
	if err != nil {
		return nil, err
	}
	idx, err := indexFromString(sc.Index)
	if err != nil {
		return nil, err
	}

	db, err := Open(pstPath, schema.VectorField().Dim, metric, idx, opt.Params)
	if err != nil {
		return nil, err
	}

	rows := make(map[uint64]map[string]any, len(sc.Rows))
	for k, v := range sc.Rows {
		id, err := strconv.ParseUint(k, 10, 64)
		if err != nil {
			db.Close()
			return nil, fmt.Errorf("pistadb: bad row key %q: %w", k, err)
		}
		rows[id] = v
	}

	collName := name
	if collName == "" {
		collName = sc.Name
	}
	return &Collection{
		Name:   collName,
		Schema: schema,
		db:     db,
		path:   pstPath,
		meta:   metaPath,
		metric: metric,
		index:  idx,
		rows:   rows,
		nextID: maxUint64(sc.NextID, 1),
	}, nil
}

// Path returns the .pst file path.
func (c *Collection) Path() string { return c.path }

// DB returns the underlying *Database (for advanced usage).
func (c *Collection) DB() *Database { return c.db }

// NumEntities returns the active row count (alias for Database.Count).
func (c *Collection) NumEntities() int { return c.db.Count() }

// Close flushes nothing — just releases native resources.  Call Flush first
// to persist.
func (c *Collection) Close() {
	if c.db != nil {
		c.db.Close()
		c.db = nil
	}
}

// Flush persists both the .pst and the JSON sidecar.
func (c *Collection) Flush() error {
	if err := c.db.Save(); err != nil {
		return err
	}
	return c.saveSidecar()
}

// Save is an alias for Flush.
func (c *Collection) Save() error { return c.Flush() }

// ── Insert ───────────────────────────────────────────────────────────────────

// Insert adds rows to the collection.  Each row is a map[string]any keyed by
// field name; the vector field accepts []float32, []float64, or []any of
// numbers.  Returns the assigned primary ids.
func (c *Collection) Insert(rows []map[string]any) ([]uint64, error) {
	pk := c.Schema.PrimaryField()
	vf := c.Schema.VectorField()
	scalars := c.Schema.ScalarFields()
	known := make(map[string]bool, len(c.Schema.Fields))
	for _, f := range c.Schema.Fields {
		known[f.Name] = true
	}

	out := make([]uint64, 0, len(rows))
	for _, row := range rows {
		// Reject unknown columns up-front.
		for k := range row {
			if !known[k] {
				return nil, fmt.Errorf("pistadb: unknown field %q in row", k)
			}
		}

		// Primary id.
		var id uint64
		if pk.AutoID {
			if v, ok := row[pk.Name]; ok && v != nil {
				return nil, fmt.Errorf("pistadb: AutoID enabled on %q — do not supply it", pk.Name)
			}
			id = c.nextID
			c.nextID++
		} else {
			v, ok := row[pk.Name]
			if !ok || v == nil {
				return nil, fmt.Errorf("pistadb: missing primary key %q", pk.Name)
			}
			x, err := coerceUint64(v)
			if err != nil {
				return nil, fmt.Errorf("pistadb: primary key %q: %w", pk.Name, err)
			}
			id = x
			if _, dup := c.rows[id]; dup {
				return nil, fmt.Errorf("pistadb: duplicate primary id=%d", id)
			}
			if id >= c.nextID {
				c.nextID = id + 1
			}
		}
		if id == 0 {
			return nil, errors.New("pistadb: primary key must be > 0")
		}

		// Vector.
		vraw, ok := row[vf.Name]
		if !ok {
			return nil, fmt.Errorf("pistadb: missing vector field %q", vf.Name)
		}
		vec, err := coerceFloat32Slice(vraw, vf.Dim)
		if err != nil {
			return nil, fmt.Errorf("pistadb: vector field %q: %w", vf.Name, err)
		}

		// Scalars.
		scalarVals := make(map[string]any, len(scalars))
		for _, f := range scalars {
			v, present := row[f.Name]
			if !present || v == nil {
				scalarVals[f.Name] = nil
				continue
			}
			cv, err := coerceScalar(v, f)
			if err != nil {
				return nil, fmt.Errorf("pistadb: field %q: %w", f.Name, err)
			}
			scalarVals[f.Name] = cv
		}

		if err := c.db.Insert(id, "", vec); err != nil {
			return nil, err
		}
		c.rows[id] = scalarVals
		out = append(out, id)
	}
	return out, nil
}

// ── Delete / Get ─────────────────────────────────────────────────────────────

// Delete removes one or more rows by primary id.  Missing ids are skipped
// silently; returns the number actually removed.
func (c *Collection) Delete(ids ...uint64) int {
	removed := 0
	for _, id := range ids {
		if err := c.db.Delete(id); err == nil {
			delete(c.rows, id)
			removed++
		}
	}
	return removed
}

// GetByID returns the full row (all fields including the vector) for the
// given primary id.
func (c *Collection) GetByID(id uint64) (map[string]any, error) {
	scalars, ok := c.rows[id]
	if !ok {
		return nil, fmt.Errorf("pistadb: id=%d not found", id)
	}
	vec, _, err := c.db.Get(id)
	if err != nil {
		return nil, err
	}
	out := make(map[string]any, len(c.Schema.Fields))
	out[c.Schema.PrimaryField().Name] = id
	for k, v := range scalars {
		out[k] = v
	}
	out[c.Schema.VectorField().Name] = vec
	return out, nil
}

// ── Search ───────────────────────────────────────────────────────────────────

// Search runs k-NN against the vector field and projects the requested
// outputFields back into each hit.  Pass outputFields=nil for all scalar
// fields, or a list of field names to include only those.
func (c *Collection) Search(query []float32, k int, outputFields []string) ([]Hit, error) {
	vf := c.Schema.VectorField()
	if len(query) != vf.Dim {
		return nil, fmt.Errorf("pistadb: query length %d != dim %d", len(query), vf.Dim)
	}
	want := outputFields
	if want == nil {
		for _, f := range c.Schema.ScalarFields() {
			want = append(want, f.Name)
		}
	} else {
		for _, n := range want {
			if n == c.Schema.PrimaryField().Name || n == vf.Name {
				continue
			}
			if _, err := c.Schema.Field(n); err != nil {
				return nil, err
			}
		}
	}

	raw, err := c.db.Search(query, k)
	if err != nil {
		return nil, err
	}
	pkName := c.Schema.PrimaryField().Name

	hits := make([]Hit, 0, len(raw))
	for _, r := range raw {
		fields := make(map[string]any, len(want))
		meta := c.rows[r.ID]
		for _, n := range want {
			switch n {
			case pkName:
				fields[n] = r.ID
			case vf.Name:
				v, _, err := c.db.Get(r.ID)
				if err == nil {
					fields[n] = v
				}
			default:
				fields[n] = meta[n]
			}
		}
		hits = append(hits, Hit{ID: r.ID, Distance: r.Distance, Fields: fields})
	}
	return hits, nil
}

// ── Sidecar I/O ──────────────────────────────────────────────────────────────

func (c *Collection) saveSidecar() error {
	fields := make([]fieldSchemaJSON, 0, len(c.Schema.Fields))
	for _, f := range c.Schema.Fields {
		fields = append(fields, f.toJSON())
	}
	rows := make(map[string]map[string]any, len(c.rows))
	// Sort to keep diffs stable across runs.
	keys := make([]uint64, 0, len(c.rows))
	for k := range c.rows {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	for _, k := range keys {
		rows[strconv.FormatUint(k, 10)] = c.rows[k]
	}
	sc := sidecar{
		Version:     sidecarVersion,
		Name:        c.Name,
		Description: c.Schema.Description,
		Metric:      c.metric.String(),
		Index:       c.index.String(),
		NextID:      c.nextID,
		Fields:      fields,
		Rows:        rows,
	}
	data, err := json.MarshalIndent(sc, "", "  ")
	if err != nil {
		return err
	}
	tmp := c.meta + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, c.meta)
}

// ── Coercion helpers ─────────────────────────────────────────────────────────

func coerceUint64(v any) (uint64, error) {
	switch x := v.(type) {
	case uint64:
		return x, nil
	case int64:
		if x < 0 {
			return 0, errors.New("negative value")
		}
		return uint64(x), nil
	case int:
		if x < 0 {
			return 0, errors.New("negative value")
		}
		return uint64(x), nil
	case int32:
		if x < 0 {
			return 0, errors.New("negative value")
		}
		return uint64(x), nil
	case float64:
		return uint64(x), nil
	case float32:
		return uint64(x), nil
	case string:
		return strconv.ParseUint(x, 10, 64)
	}
	return 0, fmt.Errorf("cannot coerce %T to uint64", v)
}

func coerceScalar(v any, f FieldSchema) (any, error) {
	switch f.DType {
	case DTypeBool:
		if b, ok := v.(bool); ok {
			return b, nil
		}
		return nil, fmt.Errorf("expected bool, got %T", v)
	case DTypeVarChar:
		s, ok := v.(string)
		if !ok {
			s = fmt.Sprint(v)
		}
		if f.MaxLength > 0 && len(s) > f.MaxLength {
			return nil, fmt.Errorf("value exceeds MaxLength=%d", f.MaxLength)
		}
		return s, nil
	case DTypeJSON:
		// Round-trip to verify JSON-serialisable.
		if _, err := json.Marshal(v); err != nil {
			return nil, fmt.Errorf("not JSON-serialisable: %w", err)
		}
		return v, nil
	default:
		if f.DType.isInt() {
			return coerceInt64(v)
		}
		if f.DType.isFloat() {
			return coerceFloat64(v)
		}
		return nil, fmt.Errorf("unsupported scalar dtype %s", f.DType)
	}
}

func coerceInt64(v any) (int64, error) {
	switch x := v.(type) {
	case int:
		return int64(x), nil
	case int8:
		return int64(x), nil
	case int16:
		return int64(x), nil
	case int32:
		return int64(x), nil
	case int64:
		return x, nil
	case uint:
		return int64(x), nil
	case uint8:
		return int64(x), nil
	case uint16:
		return int64(x), nil
	case uint32:
		return int64(x), nil
	case uint64:
		return int64(x), nil
	case float32:
		return int64(x), nil
	case float64:
		return int64(x), nil
	case string:
		return strconv.ParseInt(x, 10, 64)
	}
	return 0, fmt.Errorf("cannot coerce %T to int", v)
}

func coerceFloat64(v any) (float64, error) {
	switch x := v.(type) {
	case float32:
		return float64(x), nil
	case float64:
		return x, nil
	case int:
		return float64(x), nil
	case int32:
		return float64(x), nil
	case int64:
		return float64(x), nil
	case string:
		return strconv.ParseFloat(x, 64)
	}
	return 0, fmt.Errorf("cannot coerce %T to float", v)
}

func coerceFloat32Slice(v any, dim int) ([]float32, error) {
	switch x := v.(type) {
	case []float32:
		if len(x) != dim {
			return nil, fmt.Errorf("length %d != dim %d", len(x), dim)
		}
		return x, nil
	case []float64:
		if len(x) != dim {
			return nil, fmt.Errorf("length %d != dim %d", len(x), dim)
		}
		out := make([]float32, dim)
		for i, e := range x {
			out[i] = float32(e)
		}
		return out, nil
	case []any:
		if len(x) != dim {
			return nil, fmt.Errorf("length %d != dim %d", len(x), dim)
		}
		out := make([]float32, dim)
		for i, e := range x {
			f, err := coerceFloat64(e)
			if err != nil {
				return nil, fmt.Errorf("element %d: %w", i, err)
			}
			out[i] = float32(f)
		}
		return out, nil
	}
	// Fallback for any slice-of-numeric via reflection.
	rv := reflect.ValueOf(v)
	if rv.Kind() == reflect.Slice {
		n := rv.Len()
		if n != dim {
			return nil, fmt.Errorf("length %d != dim %d", n, dim)
		}
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			f, err := coerceFloat64(rv.Index(i).Interface())
			if err != nil {
				return nil, fmt.Errorf("element %d: %w", i, err)
			}
			out[i] = float32(f)
		}
		return out, nil
	}
	return nil, fmt.Errorf("cannot coerce %T to []float32", v)
}

func maxUint64(a, b uint64) uint64 {
	if a > b {
		return a
	}
	return b
}
