"""
PistaDB - Milvus-style schema layer.

Provides ``FieldSchema`` / ``CollectionSchema`` / ``Collection`` and a
``create_collection`` helper modelled on the pymilvus API, e.g.::

    from pistadb import (
        FieldSchema, DataType, create_collection, Metric, Index,
    )

    fields = [
        FieldSchema("lc_id",     DataType.INT64,        is_primary=True, auto_id=True),
        FieldSchema("lc_section",DataType.VARCHAR,      max_length=100),
        FieldSchema("lc_key",    DataType.VARCHAR,      max_length=200),
        FieldSchema("lc_lang",   DataType.VARCHAR,      max_length=10),
        FieldSchema("lc_lineno", DataType.INT64),
        FieldSchema("lc_tokens", DataType.INT64),
        FieldSchema("lc_vector", DataType.FLOAT_VECTOR, dim=1536),
    ]
    coll = create_collection("common_text", fields, "Common text search",
                             metric=Metric.COSINE, index=Index.HNSW)

PistaDB's C core only stores ``(uint64 id, char[256] label, float[dim] vec)``
per row, so this module persists the vector + primary id in the underlying
``.pst`` file and stores the remaining scalar fields in a JSON sidecar
``<path>.meta.json`` written next to it.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field as dc_field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from . import (
    PistaDB,
    Metric,
    Index,
    Params,
    SearchResult,
)


# ── Data types (mirror pymilvus.DataType) ─────────────────────────────────────

class DataType(IntEnum):
    BOOL         = 1
    INT8         = 2
    INT16        = 3
    INT32        = 4
    INT64        = 5
    FLOAT        = 10
    DOUBLE       = 11
    VARCHAR      = 21
    JSON         = 23
    FLOAT_VECTOR = 101


_INT_TYPES   = {DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64}
_FLOAT_TYPES = {DataType.FLOAT, DataType.DOUBLE}
_SCALAR_TYPES = (
    {DataType.BOOL, DataType.VARCHAR, DataType.JSON} | _INT_TYPES | _FLOAT_TYPES
)


# ── Field / Collection schema ────────────────────────────────────────────────

@dataclass
class FieldSchema:
    """Description of a single field in a collection.

    Mirrors :class:`pymilvus.FieldSchema`.  The constructor accepts both
    positional (``name, dtype``) and keyword forms used by pymilvus code so
    Milvus snippets port over with minimal edits.
    """
    name:        str
    dtype:       DataType
    is_primary:  bool          = False
    auto_id:     bool          = False
    max_length:  Optional[int] = None   # VARCHAR only
    dim:         Optional[int] = None   # FLOAT_VECTOR only
    description: str           = ""

    def __post_init__(self) -> None:
        # Allow callers to pass DataType ints directly.
        self.dtype = DataType(int(self.dtype))

        if self.dtype == DataType.FLOAT_VECTOR:
            if not self.dim or self.dim <= 0:
                raise ValueError(
                    f"FLOAT_VECTOR field {self.name!r} requires a positive dim"
                )
        elif self.dtype == DataType.VARCHAR:
            if self.max_length is not None and self.max_length <= 0:
                raise ValueError(
                    f"VARCHAR field {self.name!r}: max_length must be positive"
                )

        if self.is_primary and self.dtype != DataType.INT64:
            raise ValueError(
                f"Primary key field {self.name!r} must be DataType.INT64"
            )

        if self.auto_id and not self.is_primary:
            raise ValueError(
                f"auto_id is only valid on the primary key field "
                f"(got it on {self.name!r})"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":        self.name,
            "dtype":       self.dtype.name,
            "is_primary":  self.is_primary,
            "auto_id":     self.auto_id,
            "max_length":  self.max_length,
            "dim":         self.dim,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FieldSchema":
        return cls(
            name        = d["name"],
            dtype       = DataType[d["dtype"]] if isinstance(d["dtype"], str)
                          else DataType(int(d["dtype"])),
            is_primary  = bool(d.get("is_primary", False)),
            auto_id     = bool(d.get("auto_id", False)),
            max_length  = d.get("max_length"),
            dim         = d.get("dim"),
            description = d.get("description", ""),
        )


@dataclass
class CollectionSchema:
    """A list of :class:`FieldSchema` plus a description.

    Validates that exactly one primary key and exactly one FLOAT_VECTOR field
    are present, and that all field names are unique.
    """
    fields:      List[FieldSchema]
    description: str = ""

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            dups = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate field name(s): {dups}")

        primaries = [f for f in self.fields if f.is_primary]
        if len(primaries) != 1:
            raise ValueError(
                f"Schema must have exactly one primary key field "
                f"(found {len(primaries)})"
            )

        vectors = [f for f in self.fields if f.dtype == DataType.FLOAT_VECTOR]
        if len(vectors) != 1:
            raise ValueError(
                f"Schema must have exactly one FLOAT_VECTOR field "
                f"(found {len(vectors)})"
            )

    @property
    def primary_field(self) -> FieldSchema:
        return next(f for f in self.fields if f.is_primary)

    @property
    def vector_field(self) -> FieldSchema:
        return next(f for f in self.fields if f.dtype == DataType.FLOAT_VECTOR)

    @property
    def scalar_fields(self) -> List[FieldSchema]:
        """Non-primary, non-vector fields, in declaration order."""
        return [
            f for f in self.fields
            if not f.is_primary and f.dtype != DataType.FLOAT_VECTOR
        ]

    def field(self, name: str) -> FieldSchema:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"No field named {name!r} in schema")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields":      [f.to_dict() for f in self.fields],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CollectionSchema":
        return cls(
            fields      = [FieldSchema.from_dict(f) for f in d["fields"]],
            description = d.get("description", ""),
        )


# ── Type coercion helpers ────────────────────────────────────────────────────

def _coerce_scalar(value: Any, field: FieldSchema) -> Any:
    """Validate / coerce a Python value to the field's declared type."""
    if value is None:
        return None

    dt = field.dtype
    if dt == DataType.BOOL:
        return bool(value)
    if dt in _INT_TYPES:
        return int(value)
    if dt in _FLOAT_TYPES:
        return float(value)
    if dt == DataType.VARCHAR:
        s = str(value)
        if field.max_length is not None and len(s.encode("utf-8")) > field.max_length:
            raise ValueError(
                f"VARCHAR field {field.name!r}: value exceeds "
                f"max_length={field.max_length}"
            )
        return s
    if dt == DataType.JSON:
        # Round-trip to ensure it is JSON-serialisable.
        json.dumps(value)
        return value
    raise ValueError(f"Unsupported scalar dtype: {dt}")


def _coerce_vector(value: Any, dim: int) -> np.ndarray:
    arr = np.ascontiguousarray(value, dtype=np.float32).ravel()
    if arr.shape[0] != dim:
        raise ValueError(
            f"vector field expects dim={dim}, got length {arr.shape[0]}"
        )
    return arr


# ── Hit (search result row) ──────────────────────────────────────────────────

class Hit:
    """A single search hit exposing all fields declared in the schema.

    Iterable like a :class:`SearchResult`, but also supports
    ``hit['field_name']`` and ``hit.entity.get('field_name')`` for parity with
    pymilvus call-sites.
    """

    __slots__ = ("id", "distance", "_fields")

    def __init__(self, id: int, distance: float, fields: Dict[str, Any]):
        self.id       = id
        self.distance = distance
        self._fields  = fields

    def __getitem__(self, name: str) -> Any:
        return self._fields[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self._fields.get(name, default)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "distance": self.distance, **self._fields}

    @property
    def entity(self) -> "Hit":
        # pymilvus parity: ``hit.entity.get('foo')``.
        return self

    def __repr__(self) -> str:
        preview = ", ".join(f"{k}={v!r}" for k, v in list(self._fields.items())[:3])
        return f"Hit(id={self.id}, distance={self.distance:.4f}, {preview})"


# ── Collection ───────────────────────────────────────────────────────────────

class Collection:
    """A named collection backed by a :class:`PistaDB` and a JSON sidecar.

    The vector field is stored in the underlying ``.pst`` file; all other
    scalar fields are stored in ``<path>.meta.json`` next to it, keyed by the
    primary id.

    Do not construct directly — use :func:`create_collection` (to make a new
    collection) or :func:`load_collection` (to reopen an existing one).
    """

    SIDECAR_VERSION = 1

    def __init__(
        self,
        name:    str,
        path:    Union[str, Path],
        schema:  CollectionSchema,
        db:      PistaDB,
        rows:    Optional[Dict[int, Dict[str, Any]]] = None,
        next_id: int = 1,
        metric:  Metric = Metric.L2,
        index:   Index  = Index.HNSW,
    ):
        self.name     = name
        self._path    = str(path)
        self._meta    = self._path + ".meta.json"
        self.schema   = schema
        self._db      = db
        self._rows    = rows if rows is not None else {}
        self._next_id = next_id
        self._metric  = metric
        self._index   = index

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "Collection":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def description(self) -> str:
        return self.schema.description

    @property
    def num_entities(self) -> int:
        return self._db.count

    @property
    def db(self) -> PistaDB:
        """Underlying PistaDB handle (for advanced use)."""
        return self._db

    @property
    def path(self) -> str:
        return self._path

    # ── Insert ─────────────────────────────────────────────────────────────

    def insert(
        self,
        data: Union[
            Mapping[str, Sequence[Any]],
            Sequence[Mapping[str, Any]],
        ],
    ) -> List[int]:
        """Insert one or more rows.

        ``data`` is either:

        * a list of dicts, one per row::

              coll.insert([
                  {"lc_section": "common", "lc_key": "ok",     "lc_vector": v1, ...},
                  {"lc_section": "common", "lc_key": "cancel", "lc_vector": v2, ...},
              ])

        * or a dict of column lists (pymilvus-style)::

              coll.insert({
                  "lc_section": ["common", "common"],
                  "lc_key":     ["ok", "cancel"],
                  "lc_vector":  [v1, v2],
                  ...
              })

        Returns the list of primary-key ids that were assigned (useful when
        ``auto_id=True``).
        """
        rows = self._normalise_rows(data)

        pk        = self.schema.primary_field
        vec_field = self.schema.vector_field
        scalars   = self.schema.scalar_fields

        ids_out: List[int] = []
        for row in rows:
            # ── Primary id ────────────────────────────────────────────────
            if pk.auto_id:
                if pk.name in row and row[pk.name] is not None:
                    raise ValueError(
                        f"auto_id is enabled on {pk.name!r} — do not supply it"
                    )
                pk_id = self._next_id
                self._next_id += 1
            else:
                if pk.name not in row or row[pk.name] is None:
                    raise ValueError(f"missing primary key field {pk.name!r}")
                pk_id = int(row[pk.name])
                if pk_id in self._rows:
                    raise ValueError(f"duplicate primary key id={pk_id}")
                if pk_id >= self._next_id:
                    self._next_id = pk_id + 1

            # ── Vector ────────────────────────────────────────────────────
            if vec_field.name not in row:
                raise ValueError(f"missing vector field {vec_field.name!r}")
            vec = _coerce_vector(row[vec_field.name], vec_field.dim or 0)

            # ── Scalars ───────────────────────────────────────────────────
            scalar_values: Dict[str, Any] = {}
            for f in scalars:
                if f.name not in row:
                    # Allow null/missing scalar fields — they read back as None.
                    scalar_values[f.name] = None
                else:
                    scalar_values[f.name] = _coerce_scalar(row[f.name], f)

            # Reject unknown columns to catch typos early.
            unknown = (
                set(row.keys())
                - {f.name for f in self.schema.fields}
            )
            if unknown:
                raise ValueError(f"unknown field(s): {sorted(unknown)}")

            # ── Persist ──────────────────────────────────────────────────
            # PistaDB requires id > 0; auto_id starts at 1 and user ids are
            # validated below.
            if pk_id <= 0:
                raise ValueError(
                    f"primary key must be > 0 (PistaDB constraint); got {pk_id}"
                )
            self._db.insert(pk_id, vec, label="")
            self._rows[pk_id] = scalar_values
            ids_out.append(pk_id)

        return ids_out

    # ── Delete / get ───────────────────────────────────────────────────────

    def delete(self, ids: Union[int, Iterable[int]]) -> int:
        """Delete one or more rows by primary id.  Returns the count removed."""
        if isinstance(ids, (int, np.integer)):
            ids = [int(ids)]
        removed = 0
        for i in ids:
            i = int(i)
            try:
                self._db.delete(i)
            except RuntimeError:
                continue
            self._rows.pop(i, None)
            removed += 1
        return removed

    def get(self, id: int) -> Dict[str, Any]:
        """Return the full row (all fields) for the given primary id."""
        i = int(id)
        if i not in self._rows:
            raise KeyError(f"id={i} not in collection")
        vec, _label = self._db.get(i)
        out: Dict[str, Any] = {self.schema.primary_field.name: i}
        out.update(self._rows[i])
        out[self.schema.vector_field.name] = vec
        return out

    # ── Search ─────────────────────────────────────────────────────────────

    def search(
        self,
        data:           Union[np.ndarray, Sequence[Any]],
        anns_field:     Optional[str] = None,
        limit:          int = 10,
        output_fields:  Optional[Sequence[str]] = None,
    ) -> List[List[Hit]]:
        """K-NN search, mirroring :meth:`pymilvus.Collection.search`.

        Parameters
        ----------
        data
            Either a single query vector or a batch (list / 2-D array).
        anns_field
            Name of the vector field to search on.  Optional — defaults to the
            sole FLOAT_VECTOR field declared in the schema.
        limit
            Top-K per query.
        output_fields
            Names of scalar fields to attach to each hit.  ``None`` means all
            scalar fields.

        Returns
        -------
        list of lists of :class:`Hit`, one inner list per query.
        """
        vec_field = self.schema.vector_field
        if anns_field is not None and anns_field != vec_field.name:
            raise ValueError(
                f"unknown vector field {anns_field!r} "
                f"(only {vec_field.name!r} is indexed)"
            )

        queries = self._normalise_queries(data, vec_field.dim or 0)

        if output_fields is None:
            wanted = [f.name for f in self.schema.scalar_fields]
        else:
            wanted = []
            for n in output_fields:
                # Validate but allow primary / vector by name too.
                if n != self.schema.primary_field.name \
                        and n != vec_field.name:
                    self.schema.field(n)
                wanted.append(n)

        pk_name  = self.schema.primary_field.name
        vec_name = vec_field.name

        results: List[List[Hit]] = []
        for q in queries:
            raw = self._db.search(q, k=limit)
            hits: List[Hit] = []
            for r in raw:
                row_meta = self._rows.get(r.id, {})
                fields: Dict[str, Any] = {}
                for n in wanted:
                    if n == pk_name:
                        fields[n] = r.id
                    elif n == vec_name:
                        # Re-read the vector lazily — most callers do not ask
                        # for it back.
                        v, _ = self._db.get(r.id)
                        fields[n] = v
                    else:
                        fields[n] = row_meta.get(n)
                hits.append(Hit(r.id, r.distance, fields))
            results.append(hits)
        return results

    # ── Persistence ────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Persist both the vector store and the metadata sidecar."""
        self._db.save()
        self._save_sidecar()

    save = flush  # pymilvus uses flush(); keep save() as an alias.

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None  # type: ignore[assignment]

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ── Internal: sidecar I/O ──────────────────────────────────────────────

    def _save_sidecar(self) -> None:
        payload = {
            "version":     self.SIDECAR_VERSION,
            "name":        self.name,
            "schema":      self.schema.to_dict(),
            "metric":      self._metric.name,
            "index":       self._index.name,
            "next_id":     self._next_id,
            # JSON keys must be strings.
            "rows":        {str(k): v for k, v in self._rows.items()},
        }
        tmp = self._meta + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self._meta)

    @classmethod
    def _load_sidecar(cls, meta_path: str) -> Tuple[
        CollectionSchema, Dict[int, Dict[str, Any]], int, str, Metric, Index,
    ]:
        with open(meta_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        schema  = CollectionSchema.from_dict(payload["schema"])
        rows    = {int(k): v for k, v in payload.get("rows", {}).items()}
        next_id = int(payload.get("next_id", 1))
        name    = payload.get("name", "")
        metric  = Metric[payload.get("metric", "L2")]
        index   = Index[payload.get("index", "HNSW")]
        return schema, rows, next_id, name, metric, index

    # ── Internal: input normalisation ──────────────────────────────────────

    def _normalise_rows(
        self,
        data: Union[Mapping[str, Sequence[Any]], Sequence[Mapping[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if isinstance(data, Mapping):
            # Column-oriented: dict of equal-length lists.
            lengths = {len(v) for v in data.values()}
            if len(lengths) != 1:
                raise ValueError(
                    "column-oriented insert: all columns must have the same length"
                )
            n = lengths.pop()
            return [{k: data[k][i] for k in data} for i in range(n)]

        rows: List[Dict[str, Any]] = []
        for r in data:
            if not isinstance(r, Mapping):
                raise TypeError(
                    f"insert(): each row must be a dict, got {type(r).__name__}"
                )
            rows.append(dict(r))
        return rows

    @staticmethod
    def _normalise_queries(data: Any, dim: int) -> List[np.ndarray]:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1:
            return [_coerce_vector(arr, dim)]
        if arr.ndim == 2:
            return [_coerce_vector(row, dim) for row in arr]
        raise ValueError(
            f"search(): query must be 1-D or 2-D, got ndim={arr.ndim}"
        )

    # ── Repr ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Collection(name={self.name!r}, path={self._path!r}, "
            f"num_entities={self.num_entities}, "
            f"fields={[f.name for f in self.schema.fields]})"
        )


# ── Factory functions ────────────────────────────────────────────────────────

def _resolve_path(name: str, base_dir: Optional[Union[str, Path]]) -> str:
    if base_dir is None:
        return f"{name}.pst"
    return os.path.join(str(base_dir), f"{name}.pst")


def create_collection(
    name:        str,
    fields:      Union[Sequence[FieldSchema], CollectionSchema],
    description: str = "",
    *,
    metric:      Metric = Metric.L2,
    index:       Index  = Index.HNSW,
    params:      Optional[Params] = None,
    base_dir:    Optional[Union[str, Path]] = None,
    path:        Optional[Union[str, Path]] = None,
    overwrite:   bool   = False,
) -> Collection:
    """Create a new collection.

    Parameters mirror the pymilvus pattern from the user's snippet::

        new_collection = create_collection(name, field_schema_list, "description")

    Parameters
    ----------
    name
        Collection name.  Used as the file stem for the underlying ``.pst``.
    fields
        Either a list of :class:`FieldSchema` or a built :class:`CollectionSchema`.
    description
        Free-text description (ignored if ``fields`` is already a CollectionSchema
        with its own description).
    metric, index, params
        Forwarded to :class:`PistaDB`.  Defaults: L2 + HNSW.
    base_dir
        Directory in which to create ``<name>.pst``.  Defaults to CWD.
    path
        Explicit file path; overrides ``base_dir`` / ``name`` when given.
    overwrite
        If ``True``, remove any existing ``.pst`` / ``.meta.json`` before
        creating the collection.  Default ``False`` raises if files exist.
    """
    if isinstance(fields, CollectionSchema):
        schema = fields
    else:
        schema = CollectionSchema(list(fields), description=description)

    pst_path = str(path) if path is not None else _resolve_path(name, base_dir)
    meta     = pst_path + ".meta.json"

    if overwrite:
        for p in (pst_path, meta):
            if os.path.exists(p):
                os.remove(p)
    else:
        for p in (pst_path, meta):
            if os.path.exists(p):
                raise FileExistsError(
                    f"{p} already exists — pass overwrite=True or use load_collection()"
                )

    parent = os.path.dirname(pst_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    vec_field = schema.vector_field
    db = PistaDB(
        pst_path,
        dim    = vec_field.dim or 0,
        metric = metric,
        index  = index,
        params = params,
    )

    coll = Collection(
        name    = name,
        path    = pst_path,
        schema  = schema,
        db      = db,
        rows    = {},
        next_id = 1,
        metric  = metric,
        index   = index,
    )
    coll._save_sidecar()
    return coll


def load_collection(
    name:     Optional[str] = None,
    *,
    path:     Optional[Union[str, Path]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    params:   Optional[Params] = None,
) -> Collection:
    """Re-open a previously created collection.

    Provide either ``name`` (resolved against ``base_dir``) or an explicit
    ``path`` to the ``.pst`` file.
    """
    if path is not None:
        pst_path = str(path)
    elif name is not None:
        pst_path = _resolve_path(name, base_dir)
    else:
        raise ValueError("load_collection: pass either 'name' or 'path'")

    meta = pst_path + ".meta.json"
    if not os.path.exists(meta):
        raise FileNotFoundError(
            f"sidecar not found: {meta} — collection was not created via "
            f"create_collection()"
        )

    schema, rows, next_id, saved_name, metric, index = Collection._load_sidecar(meta)
    vec_field = schema.vector_field

    db = PistaDB(
        pst_path,
        dim    = vec_field.dim or 0,
        metric = metric,
        index  = index,
        params = params,
    )

    return Collection(
        name    = name or saved_name,
        path    = pst_path,
        schema  = schema,
        db      = db,
        rows    = rows,
        next_id = next_id,
        metric  = metric,
        index   = index,
    )


__all__ = [
    "DataType",
    "FieldSchema",
    "CollectionSchema",
    "Collection",
    "Hit",
    "create_collection",
    "load_collection",
]
