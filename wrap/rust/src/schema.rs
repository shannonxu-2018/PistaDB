//! Milvus-style schema layer for PistaDB.
//!
//! Enabled with `--features schema`.  Adds [`Collection`], [`FieldSchema`],
//! [`CollectionSchema`], [`DataType`], and the [`create_collection`] /
//! [`load_collection`] factories.  The vector + primary id are stored in the
//! underlying `.pst` file; remaining scalar fields are persisted in a JSON
//! sidecar (`<path>.meta.json`) via `serde_json`.
//!
//! ```no_run
//! # #[cfg(feature = "schema")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use pistadb::schema::{
//!     create_collection, CollectionOptions, DataType, FieldSchema,
//! };
//! use pistadb::{IndexType, Metric};
//! use serde_json::json;
//!
//! let fields = vec![
//!     FieldSchema { name: "lc_id".into(),     dtype: DataType::Int64,
//!                   is_primary: true, auto_id: true,  ..Default::default() },
//!     FieldSchema { name: "lc_section".into(),dtype: DataType::VarChar,
//!                   max_length: Some(100),    ..Default::default() },
//!     FieldSchema { name: "lc_vector".into(), dtype: DataType::FloatVector,
//!                   dim: Some(1536),          ..Default::default() },
//! ];
//! let coll = create_collection(
//!     "common_text", fields, "Common text search",
//!     CollectionOptions { metric: Metric::Cosine, index: IndexType::HNSW,
//!                         base_dir: Some("./db".into()), ..Default::default() },
//! )?;
//! coll.insert(vec![
//!     [("lc_section", json!("common")), ("lc_vector", json!(vec![0.0_f32; 1536]))]
//!         .into_iter().map(|(k,v)| (k.to_string(), v)).collect(),
//! ])?;
//! # Ok(()) }
//! # #[cfg(not(feature = "schema"))]
//! # fn main() {}
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Database, Error, IndexType, Metric, Params, Result};

// ── DataType ─────────────────────────────────────────────────────────────────

/// Per-field data type.  Mirrors `pymilvus.DataType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DataType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    Float,
    Double,
    VarChar,
    Json,
    FloatVector,
}

impl DataType {
    fn is_int(self) -> bool {
        matches!(self, Self::Int8 | Self::Int16 | Self::Int32 | Self::Int64)
    }

    fn is_float(self) -> bool {
        matches!(self, Self::Float | Self::Double)
    }
}

// ── FieldSchema ──────────────────────────────────────────────────────────────

/// A single field's description.  Only the relevant typing fields are used
/// for each [`DataType`]: `dim` for `FloatVector`, `max_length` for `VarChar`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    pub name: String,
    pub dtype: DataType,
    #[serde(default)]
    pub is_primary: bool,
    #[serde(default)]
    pub auto_id: bool,
    #[serde(default)]
    pub max_length: Option<usize>,
    #[serde(default)]
    pub dim: Option<usize>,
    #[serde(default)]
    pub description: String,
}

impl Default for FieldSchema {
    fn default() -> Self {
        Self {
            name: String::new(),
            dtype: DataType::Int64,
            is_primary: false,
            auto_id: false,
            max_length: None,
            dim: None,
            description: String::new(),
        }
    }
}

impl FieldSchema {
    fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(Error::Native("field name must not be empty".into()));
        }
        if self.dtype == DataType::FloatVector && self.dim.unwrap_or(0) == 0 {
            return Err(Error::Native(format!(
                "FLOAT_VECTOR field {:?} requires positive dim",
                self.name
            )));
        }
        if self.is_primary && self.dtype != DataType::Int64 {
            return Err(Error::Native(format!(
                "primary key {:?} must be INT64",
                self.name
            )));
        }
        if self.auto_id && !self.is_primary {
            return Err(Error::Native(format!(
                "auto_id is only valid on the primary field (got it on {:?})",
                self.name
            )));
        }
        Ok(())
    }
}

// ── CollectionSchema ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSchema {
    pub fields: Vec<FieldSchema>,
    #[serde(default)]
    pub description: String,
}

impl CollectionSchema {
    pub fn new(fields: Vec<FieldSchema>, description: impl Into<String>) -> Result<Self> {
        let s = Self {
            fields,
            description: description.into(),
        };
        s.validate()?;
        Ok(s)
    }

    fn validate(&self) -> Result<()> {
        let mut names = std::collections::HashSet::new();
        let mut primary = 0;
        let mut vector = 0;
        for f in &self.fields {
            f.validate()?;
            if !names.insert(&f.name) {
                return Err(Error::Native(format!("duplicate field name {:?}", f.name)));
            }
            if f.is_primary {
                primary += 1;
            }
            if f.dtype == DataType::FloatVector {
                vector += 1;
            }
        }
        if primary != 1 {
            return Err(Error::Native(format!(
                "schema must have exactly one primary key (found {primary})"
            )));
        }
        if vector != 1 {
            return Err(Error::Native(format!(
                "schema must have exactly one FLOAT_VECTOR field (found {vector})"
            )));
        }
        Ok(())
    }

    pub fn primary_field(&self) -> &FieldSchema {
        self.fields.iter().find(|f| f.is_primary).expect("validated")
    }

    pub fn vector_field(&self) -> &FieldSchema {
        self.fields
            .iter()
            .find(|f| f.dtype == DataType::FloatVector)
            .expect("validated")
    }

    pub fn scalar_fields(&self) -> impl Iterator<Item = &FieldSchema> {
        self.fields
            .iter()
            .filter(|f| !f.is_primary && f.dtype != DataType::FloatVector)
    }

    pub fn field(&self, name: &str) -> Result<&FieldSchema> {
        self.fields
            .iter()
            .find(|f| f.name == name)
            .ok_or_else(|| Error::Native(format!("no field named {name:?}")))
    }
}

// ── Hit ──────────────────────────────────────────────────────────────────────

/// A single search result enriched with the projected scalar fields.
#[derive(Debug, Clone)]
pub struct Hit {
    pub id: u64,
    pub distance: f32,
    pub fields: BTreeMap<String, Value>,
}

impl Hit {
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.fields.get(name)
    }
}

// ── Sidecar payload ──────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct Sidecar {
    version: u32,
    name: String,
    description: String,
    metric: String,
    index: String,
    next_id: u64,
    fields: Vec<FieldSchema>,
    rows: BTreeMap<String, BTreeMap<String, Value>>,
}

const SIDECAR_VERSION: u32 = 1;

fn metric_name(m: Metric) -> &'static str {
    match m {
        Metric::L2 => "L2",
        Metric::Cosine => "Cosine",
        Metric::IP => "IP",
        Metric::L1 => "L1",
        Metric::Hamming => "Hamming",
    }
}

fn metric_from_name(s: &str) -> Result<Metric> {
    match s {
        "L2" => Ok(Metric::L2),
        "Cosine" => Ok(Metric::Cosine),
        "IP" | "InnerProduct" => Ok(Metric::IP),
        "L1" => Ok(Metric::L1),
        "Hamming" => Ok(Metric::Hamming),
        _ => Err(Error::Native(format!("unknown metric {s:?}"))),
    }
}

fn index_name(i: IndexType) -> &'static str {
    match i {
        IndexType::Linear => "Linear",
        IndexType::HNSW => "HNSW",
        IndexType::IVF => "IVF",
        IndexType::IVF_PQ => "IVF_PQ",
        IndexType::DiskANN => "DiskANN",
        IndexType::LSH => "LSH",
        IndexType::ScaNN => "ScaNN",
    }
}

fn index_from_name(s: &str) -> Result<IndexType> {
    match s {
        "Linear" => Ok(IndexType::Linear),
        "HNSW" => Ok(IndexType::HNSW),
        "IVF" => Ok(IndexType::IVF),
        "IVF_PQ" => Ok(IndexType::IVF_PQ),
        "DiskANN" => Ok(IndexType::DiskANN),
        "LSH" => Ok(IndexType::LSH),
        "ScaNN" => Ok(IndexType::ScaNN),
        _ => Err(Error::Native(format!("unknown index type {s:?}"))),
    }
}

// ── CollectionOptions ────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct CollectionOptions {
    pub metric: Metric,
    pub index: IndexType,
    pub params: Option<Params>,
    /// Directory in which `<name>.pst` will be created.  Ignored if `path` is set.
    pub base_dir: Option<PathBuf>,
    /// Explicit path to the `.pst` file (overrides base_dir).
    pub path: Option<PathBuf>,
    /// If true, remove any existing `.pst` / `.meta.json` before creating.
    pub overwrite: bool,
}

impl CollectionOptions {
    fn resolve(&self, name: &str) -> PathBuf {
        if let Some(p) = &self.path {
            return p.clone();
        }
        match &self.base_dir {
            Some(d) => d.join(format!("{name}.pst")),
            None => PathBuf::from(format!("{name}.pst")),
        }
    }
}

impl Default for Metric {
    fn default() -> Self {
        Metric::L2
    }
}

impl Default for IndexType {
    fn default() -> Self {
        IndexType::HNSW
    }
}

// ── Collection ───────────────────────────────────────────────────────────────

/// A schema-backed wrapper over [`Database`] with a JSON sidecar.
pub struct Collection {
    pub name: String,
    pub schema: CollectionSchema,
    db: Option<Database>,
    path: PathBuf,
    meta_path: PathBuf,
    metric: Metric,
    index: IndexType,
    rows: BTreeMap<u64, BTreeMap<String, Value>>,
    next_id: u64,
}

impl Collection {
    /// Path to the `.pst` file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of active (non-deleted) rows.
    pub fn num_entities(&self) -> i32 {
        self.db_ref().count()
    }

    fn db_ref(&self) -> &Database {
        self.db.as_ref().expect("database closed")
    }

    /// Persist both the `.pst` and the JSON sidecar.
    pub fn flush(&self) -> Result<()> {
        self.db_ref().save()?;
        self.save_sidecar()
    }

    /// Alias for [`Self::flush`].
    pub fn save(&self) -> Result<()> {
        self.flush()
    }

    /// Free native resources.  Does **not** save.
    pub fn close(&mut self) {
        self.db = None;
    }

    fn save_sidecar(&self) -> Result<()> {
        let rows = self
            .rows
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        let sc = Sidecar {
            version: SIDECAR_VERSION,
            name: self.name.clone(),
            description: self.schema.description.clone(),
            metric: metric_name(self.metric).into(),
            index: index_name(self.index).into(),
            next_id: self.next_id,
            fields: self.schema.fields.clone(),
            rows,
        };
        let json = serde_json::to_string_pretty(&sc)
            .map_err(|e| Error::Native(format!("serialize sidecar: {e}")))?;
        let mut tmp_os = self.meta_path.clone().into_os_string();
        tmp_os.push(".tmp");
        let tmp = PathBuf::from(tmp_os);
        fs::write(&tmp, json).map_err(|e| Error::Native(format!("write {tmp:?}: {e}")))?;
        fs::rename(&tmp, &self.meta_path)
            .map_err(|e| Error::Native(format!("rename {tmp:?}: {e}")))?;
        Ok(())
    }

    // ── Insert ────────────────────────────────────────────────────────────

    /// Insert one or more rows.  Each row is a map keyed by field name.  The
    /// vector field accepts a JSON array of numbers or `Value::Array`.
    /// Returns the assigned primary ids.
    pub fn insert(&mut self, rows: Vec<BTreeMap<String, Value>>) -> Result<Vec<u64>> {
        let pk_name = self.schema.primary_field().name.clone();
        let pk_auto = self.schema.primary_field().auto_id;
        let vec_name = self.schema.vector_field().name.clone();
        let vec_dim = self.schema.vector_field().dim.unwrap_or(0);
        let known: std::collections::HashSet<String> =
            self.schema.fields.iter().map(|f| f.name.clone()).collect();
        let scalars: Vec<FieldSchema> = self.schema.scalar_fields().cloned().collect();

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            // Reject unknown columns.
            for k in row.keys() {
                if !known.contains(k) {
                    return Err(Error::Native(format!("unknown field {k:?}")));
                }
            }

            // Primary id.
            let id = if pk_auto {
                if matches!(row.get(&pk_name), Some(v) if !v.is_null()) {
                    return Err(Error::Native(format!(
                        "auto_id enabled on {pk_name:?} — do not supply it"
                    )));
                }
                let i = self.next_id;
                self.next_id += 1;
                i
            } else {
                let v = row.get(&pk_name).ok_or_else(|| {
                    Error::Native(format!("missing primary key {pk_name:?}"))
                })?;
                let i = coerce_u64(v).map_err(|e| {
                    Error::Native(format!("primary key {pk_name:?}: {e}"))
                })?;
                if self.rows.contains_key(&i) {
                    return Err(Error::Native(format!("duplicate primary id={i}")));
                }
                if i >= self.next_id {
                    self.next_id = i + 1;
                }
                i
            };
            if id == 0 {
                return Err(Error::Native("primary key must be > 0".into()));
            }

            // Vector.
            let vraw = row
                .get(&vec_name)
                .ok_or_else(|| Error::Native(format!("missing vector field {vec_name:?}")))?;
            let vec = coerce_float_vec(vraw, vec_dim).map_err(|e| {
                Error::Native(format!("vector field {vec_name:?}: {e}"))
            })?;

            // Scalars.
            let mut scalar_vals: BTreeMap<String, Value> = BTreeMap::new();
            for f in &scalars {
                match row.get(&f.name) {
                    Some(v) if !v.is_null() => {
                        let cv = coerce_scalar(v, f).map_err(|e| {
                            Error::Native(format!("field {:?}: {}", f.name, e))
                        })?;
                        scalar_vals.insert(f.name.clone(), cv);
                    }
                    _ => {
                        scalar_vals.insert(f.name.clone(), Value::Null);
                    }
                }
            }

            self.db_ref().insert(id, &vec, None)?;
            self.rows.insert(id, scalar_vals);
            out.push(id);
        }
        Ok(out)
    }

    // ── Delete / Get ──────────────────────────────────────────────────────

    /// Delete rows by primary id.  Missing ids are skipped silently.  Returns
    /// the number of rows actually removed.
    pub fn delete(&mut self, ids: &[u64]) -> usize {
        let mut removed = 0;
        for &id in ids {
            if self.db_ref().delete(id).is_ok() {
                self.rows.remove(&id);
                removed += 1;
            }
        }
        removed
    }

    /// Get the full row (all fields, including the vector) by primary id.
    pub fn get(&self, id: u64) -> Result<BTreeMap<String, Value>> {
        let scalars = self
            .rows
            .get(&id)
            .ok_or_else(|| Error::Native(format!("id={id} not found")))?;
        let entry = self.db_ref().get(id)?;
        let mut out = scalars.clone();
        out.insert(self.schema.primary_field().name.clone(), Value::from(id));
        out.insert(
            self.schema.vector_field().name.clone(),
            Value::Array(entry.vector.into_iter().map(|f| {
                serde_json::Number::from_f64(f as f64)
                    .map(Value::Number)
                    .unwrap_or(Value::Null)
            }).collect()),
        );
        Ok(out)
    }

    // ── Search ────────────────────────────────────────────────────────────

    /// k-NN search.  `output_fields=None` projects all scalar fields; pass a
    /// list of names to project only those.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        output_fields: Option<&[&str]>,
    ) -> Result<Vec<Hit>> {
        let vec_field = self.schema.vector_field();
        if query.len() != vec_field.dim.unwrap_or(0) {
            return Err(Error::Native(format!(
                "query length {} != dim {}",
                query.len(),
                vec_field.dim.unwrap_or(0)
            )));
        }
        let want: Vec<String> = match output_fields {
            None => self.schema.scalar_fields().map(|f| f.name.clone()).collect(),
            Some(list) => {
                for n in list {
                    if *n != self.schema.primary_field().name && *n != vec_field.name {
                        self.schema.field(n)?;
                    }
                }
                list.iter().map(|s| (*s).to_string()).collect()
            }
        };

        let raw = self.db_ref().search(query, k)?;
        let pk_name = &self.schema.primary_field().name;
        let vec_name = &vec_field.name;

        let mut hits = Vec::with_capacity(raw.len());
        for r in raw {
            let mut fields: BTreeMap<String, Value> = BTreeMap::new();
            let meta = self.rows.get(&r.id);
            for n in &want {
                if n == pk_name {
                    fields.insert(n.clone(), Value::from(r.id));
                } else if n == vec_name {
                    if let Ok(entry) = self.db_ref().get(r.id) {
                        fields.insert(
                            n.clone(),
                            Value::Array(
                                entry
                                    .vector
                                    .into_iter()
                                    .map(|f| {
                                        serde_json::Number::from_f64(f as f64)
                                            .map(Value::Number)
                                            .unwrap_or(Value::Null)
                                    })
                                    .collect(),
                            ),
                        );
                    }
                } else {
                    let v = meta.and_then(|m| m.get(n)).cloned().unwrap_or(Value::Null);
                    fields.insert(n.clone(), v);
                }
            }
            hits.push(Hit {
                id: r.id,
                distance: r.distance,
                fields,
            });
        }
        Ok(hits)
    }
}

impl Drop for Collection {
    fn drop(&mut self) {
        // Database itself drops here if still owned — does not auto-save (matches
        // the bare Database semantics).
    }
}

// ── Factories ────────────────────────────────────────────────────────────────

/// Create a new collection.  Errors if files already exist (unless
/// `opt.overwrite == true`).
pub fn create_collection(
    name: impl Into<String>,
    fields: Vec<FieldSchema>,
    description: impl Into<String>,
    opt: CollectionOptions,
) -> Result<Collection> {
    let name = name.into();
    let schema = CollectionSchema::new(fields, description)?;
    let path = opt.resolve(&name);
    let meta_path = sidecar_path(&path);

    if opt.overwrite {
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&meta_path);
    } else {
        for p in [&path, &meta_path] {
            if p.exists() {
                return Err(Error::Native(format!("{p:?} already exists")));
            }
        }
    }
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            fs::create_dir_all(dir)
                .map_err(|e| Error::Native(format!("mkdir {dir:?}: {e}")))?;
        }
    }

    let dim = schema.vector_field().dim.unwrap_or(0) as i32;
    let db = Database::open(&path, dim, opt.metric, opt.index, opt.params.as_ref())?;

    let coll = Collection {
        name,
        schema,
        db: Some(db),
        path,
        meta_path,
        metric: opt.metric,
        index: opt.index,
        rows: BTreeMap::new(),
        next_id: 1,
    };
    coll.save_sidecar()?;
    Ok(coll)
}

/// Re-open an existing collection from disk.
pub fn load_collection(name: &str, opt: CollectionOptions) -> Result<Collection> {
    let path = opt.resolve(name);
    let meta_path = sidecar_path(&path);

    let raw = fs::read_to_string(&meta_path)
        .map_err(|e| Error::Native(format!("read {meta_path:?}: {e}")))?;
    let sc: Sidecar = serde_json::from_str(&raw)
        .map_err(|e| Error::Native(format!("parse sidecar: {e}")))?;
    let schema = CollectionSchema::new(sc.fields, sc.description)?;
    let metric = metric_from_name(&sc.metric)?;
    let index = index_from_name(&sc.index)?;

    let db = Database::open(
        &path,
        schema.vector_field().dim.unwrap_or(0) as i32,
        metric,
        index,
        opt.params.as_ref(),
    )?;

    let mut rows: BTreeMap<u64, BTreeMap<String, Value>> = BTreeMap::new();
    for (k, v) in sc.rows {
        let id: u64 = k
            .parse()
            .map_err(|e| Error::Native(format!("bad row key {k:?}: {e}")))?;
        rows.insert(id, v);
    }

    let resolved_name = if name.is_empty() { sc.name } else { name.to_string() };

    Ok(Collection {
        name: resolved_name,
        schema,
        db: Some(db),
        path,
        meta_path,
        metric,
        index,
        rows,
        next_id: sc.next_id.max(1),
    })
}

fn sidecar_path(pst: &Path) -> PathBuf {
    let mut s = pst.as_os_str().to_owned();
    s.push(".meta.json");
    PathBuf::from(s)
}

// ── Coercion helpers ─────────────────────────────────────────────────────────

fn coerce_u64(v: &Value) -> std::result::Result<u64, String> {
    match v {
        Value::Number(n) => n
            .as_u64()
            .or_else(|| n.as_i64().and_then(|i| if i >= 0 { Some(i as u64) } else { None }))
            .or_else(|| n.as_f64().map(|f| f as u64))
            .ok_or_else(|| "not a non-negative integer".into()),
        Value::String(s) => s.parse::<u64>().map_err(|e| e.to_string()),
        _ => Err(format!("cannot coerce {v:?} to u64")),
    }
}

fn coerce_i64(v: &Value) -> std::result::Result<i64, String> {
    match v {
        Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .ok_or_else(|| "not an integer".into()),
        Value::String(s) => s.parse::<i64>().map_err(|e| e.to_string()),
        _ => Err(format!("cannot coerce {v:?} to i64")),
    }
}

fn coerce_f64(v: &Value) -> std::result::Result<f64, String> {
    match v {
        Value::Number(n) => n.as_f64().ok_or_else(|| "not a number".into()),
        Value::String(s) => s.parse::<f64>().map_err(|e| e.to_string()),
        _ => Err(format!("cannot coerce {v:?} to f64")),
    }
}

fn coerce_scalar(v: &Value, f: &FieldSchema) -> std::result::Result<Value, String> {
    match f.dtype {
        DataType::Bool => v
            .as_bool()
            .map(Value::Bool)
            .ok_or_else(|| format!("expected bool, got {v:?}")),
        DataType::VarChar => {
            let s = match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            if let Some(max) = f.max_length {
                if s.as_bytes().len() > max {
                    return Err(format!("value exceeds max_length={max}"));
                }
            }
            Ok(Value::String(s))
        }
        DataType::Json => Ok(v.clone()),
        DataType::FloatVector => Err("vector field cannot be coerced as scalar".into()),
        d if d.is_int() => coerce_i64(v).map(Value::from),
        d if d.is_float() => {
            let x = coerce_f64(v)?;
            serde_json::Number::from_f64(x)
                .map(Value::Number)
                .ok_or_else(|| "non-finite float".into())
        }
        _ => Err(format!("unsupported dtype {:?}", f.dtype)),
    }
}

fn coerce_float_vec(v: &Value, dim: usize) -> std::result::Result<Vec<f32>, String> {
    let arr = v.as_array().ok_or_else(|| "expected array".to_string())?;
    if arr.len() != dim {
        return Err(format!("length {} != dim {}", arr.len(), dim));
    }
    let mut out = Vec::with_capacity(dim);
    for (i, e) in arr.iter().enumerate() {
        let f = coerce_f64(e).map_err(|e| format!("element {i}: {e}"))?;
        out.push(f as f32);
    }
    Ok(out)
}
