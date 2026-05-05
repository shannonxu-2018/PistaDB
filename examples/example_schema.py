#!/usr/bin/env python3
"""
PistaDB - Milvus-style schema / collection example.
====================================================

Demonstrates the ``FieldSchema`` / ``CollectionSchema`` / ``create_collection``
API, modelled on pymilvus.  The structure of ``create_database()`` below is a
direct port of a typical Milvus snippet:

    lc_id     = FieldSchema(name="lc_id",     dtype=DataType.INT64,
                            is_primary=True,  auto_id=True)
    lc_section= FieldSchema(name="lc_section",dtype=DataType.VARCHAR,
                            max_length=100,  description="...")
    ...
    new_collection = create_collection(
        "common_text", [lc_id, lc_section, ..., lc_vector],
        "Common text search",
    )

Run:

    python examples/example_schema.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "wrap", "python"))

from pistadb import (
    FieldSchema, CollectionSchema, DataType,
    create_collection, load_collection,
    Metric, Index, Params,
)


# ── Configuration mirrors the Milvus example's app_const / const modules ─────

class const:
    EMBEDDING_DIM_OPENAI = 1536


class app_const:
    DB_COLLECTION_NAME_G001 = "common_text_g001"
    COLLECTION_FIELD_COUNT  = 7

    class _F:
        def __init__(self, name: str, desc: str = ""):
            self.Name = name
            self.Desc = desc

    DB_FIELD_ID      = _F("lc_id",      "Auto primary key.")
    DB_FIELD_SECTION = _F("lc_section", "Localization section name.")
    DB_FIELD_KEY     = _F("lc_key",     "Localization key.")
    DB_FIELD_LANG    = _F("lc_lang",    "Two-letter language code.")
    DB_FIELD_LINENO  = _F("lc_lineno",  "Line number in source file.")
    DB_FIELD_TOKENS  = _F("lc_tokens",  "Token count of the raw string.")
    DB_FIELD_VECTOR  = _F("lc_vector",  "Embedding vector of the raw string.")


DB_DIR = os.path.join(os.path.dirname(__file__), "..", "example_dbs")
os.makedirs(DB_DIR, exist_ok=True)


# ── Direct port of the user's create_database() function ─────────────────────

def create_database():
    lc_id = FieldSchema(
        name        = app_const.DB_FIELD_ID.Name,
        dtype       = DataType.INT64,
        is_primary  = True,
        auto_id     = True,
    )

    lc_section = FieldSchema(
        name        = app_const.DB_FIELD_SECTION.Name,
        dtype       = DataType.VARCHAR,
        max_length  = 100,
        description = app_const.DB_FIELD_SECTION.Desc,
    )

    lc_key = FieldSchema(
        name        = app_const.DB_FIELD_KEY.Name,
        dtype       = DataType.VARCHAR,
        max_length  = 200,
        description = app_const.DB_FIELD_KEY.Desc,
    )

    lc_lang = FieldSchema(
        name        = app_const.DB_FIELD_LANG.Name,
        dtype       = DataType.VARCHAR,
        max_length  = 10,
        description = app_const.DB_FIELD_LANG.Desc,
    )

    lc_lineno = FieldSchema(
        name        = app_const.DB_FIELD_LINENO.Name,
        dtype       = DataType.INT64,
        description = app_const.DB_FIELD_LINENO.Desc,
    )

    lc_tokens = FieldSchema(
        name        = app_const.DB_FIELD_TOKENS.Name,
        dtype       = DataType.INT64,
        description = app_const.DB_FIELD_TOKENS.Desc,
    )

    lc_vector = FieldSchema(
        name        = app_const.DB_FIELD_VECTOR.Name,
        dtype       = DataType.FLOAT_VECTOR,
        dim         = const.EMBEDDING_DIM_OPENAI,
        description = app_const.DB_FIELD_VECTOR.Desc,
    )

    field_schema_list = [
        lc_id, lc_section, lc_key, lc_lang, lc_lineno, lc_tokens, lc_vector,
    ]

    if len(field_schema_list) == app_const.COLLECTION_FIELD_COUNT:
        new_collection = create_collection(
            app_const.DB_COLLECTION_NAME_G001,
            field_schema_list,
            "Common text search",
            metric    = Metric.COSINE,
            index     = Index.HNSW,
            base_dir  = DB_DIR,
            overwrite = True,           # demo idempotency
        )
        if new_collection is None:
            print("Create new collection failed.")
        else:
            print(f"Create new collection success: {new_collection}")
        return new_collection
    else:
        print(
            f"create database collection failed: field num"
            f"({len(field_schema_list)}) not match COLLECTION_FIELD_COUNT"
            f"({app_const.COLLECTION_FIELD_COUNT})."
        )
        return None


# ── End-to-end demo: create → insert → search → reload ───────────────────────

def main() -> None:
    print("=" * 60)
    print("Milvus-style create_collection() demo")
    print("=" * 60)

    coll = create_database()
    if coll is None:
        return

    rng = np.random.default_rng(42)
    dim = const.EMBEDDING_DIM_OPENAI

    rows = [
        {
            "lc_section": "common",
            "lc_key":     "btn_ok",
            "lc_lang":    "en",
            "lc_lineno":  12,
            "lc_tokens":  3,
            "lc_vector":  rng.random(dim, dtype=np.float32),
        },
        {
            "lc_section": "common",
            "lc_key":     "btn_cancel",
            "lc_lang":    "en",
            "lc_lineno":  13,
            "lc_tokens":  4,
            "lc_vector":  rng.random(dim, dtype=np.float32),
        },
        {
            "lc_section": "settings",
            "lc_key":     "title",
            "lc_lang":    "en",
            "lc_lineno":  42,
            "lc_tokens":  2,
            "lc_vector":  rng.random(dim, dtype=np.float32),
        },
    ]

    ids = coll.insert(rows)
    print(f"\nInserted {len(ids)} rows; auto-assigned ids = {ids}")
    print(f"num_entities = {coll.num_entities}")

    # Search the first row's vector — it should rank itself first.
    q = rows[0]["lc_vector"]
    hits = coll.search(q, limit=3)[0]
    print("\nTop-3 hits for row 0:")
    for h in hits:
        print(
            f"  id={h.id}  dist={h.distance:.5f}  "
            f"section={h['lc_section']!r}  key={h['lc_key']!r}  "
            f"lang={h['lc_lang']!r}"
        )

    # Selective output_fields, pymilvus-style.
    hits = coll.search(q, limit=2, output_fields=["lc_key", "lc_lineno"])[0]
    print("\nWith output_fields=['lc_key','lc_lineno']:")
    for h in hits:
        print(f"  id={h.id}  {h.entity.get('lc_key')!r}  line={h.entity.get('lc_lineno')}")

    # Persist and reload.
    coll.flush()
    coll.close()
    print("\nFlushed to disk; reopening...")

    reopened = load_collection(
        app_const.DB_COLLECTION_NAME_G001, base_dir=DB_DIR,
    )
    print(f"Reopened: {reopened}")
    row = reopened.get(ids[0])
    print(
        f"  get(id={ids[0]}) → section={row['lc_section']!r}  "
        f"key={row['lc_key']!r}  vector_shape={row['lc_vector'].shape}"
    )
    reopened.close()


if __name__ == "__main__":
    main()
