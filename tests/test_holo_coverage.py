"""Read-only embedding coverage (observability #7).

Offline: a provider-shaped ``tmp_path`` database with canned vectors, and
factkit's vectors DB redirected into ``tmp_path`` as well. No host paths, no
network, no live store.
"""

from __future__ import annotations

import sqlite3

import pytest

from factkit import holo, vectors

MODEL = "text-embedding-nomic-embed-text-v1.5"


def _provider_db(path, contents):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE facts(fact_id INTEGER PRIMARY KEY, content TEXT, category TEXT,
                           tags TEXT, trust_score REAL, created_at TEXT, updated_at TEXT);
        """
    )
    for fact_id, content in enumerate(contents, 1):
        conn.execute(
            "INSERT INTO facts VALUES (?,?,?,?,?,?,?)",
            (
                fact_id,
                content,
                "general",
                "",
                0.9,
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
            ),
        )
    conn.commit()
    conn.close()


def _vectors_db(path, rows):
    conn = sqlite3.connect(path)
    conn.executescript(holo._FACTKIT_VECTORS_SCHEMA)
    for fact_id, model in rows:
        conn.execute(
            "INSERT INTO fact_embeddings(fact_id, model, dims, vec) VALUES (?,?,?,?)",
            (fact_id, model, 2, vectors.pack([1.0, 0.0])),
        )
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
def _no_plugin_health(monkeypatch):
    """Never import the host's installed plugin from a unit test."""
    monkeypatch.setattr(holo, "plugin_health", lambda: {})


@pytest.fixture()
def provider_store(tmp_path, monkeypatch):
    provider = tmp_path / "provider.db"
    _provider_db(
        provider,
        [
            "Atlas Server has 16 GB RAM",
            "Atlas Server listens on port 8010",
            "Beacon Mini has 1 GB RAM",
        ],
    )
    factkit_db = tmp_path / "factkit-vectors.db"
    monkeypatch.setattr(holo, "_factkit_vectors_db_path", lambda **kw: factkit_db)
    store = holo.ReadOnlyStore(provider)
    try:
        yield store, factkit_db
    finally:
        store.close()


def test_empty_store_reports_full_coverage(tmp_path, monkeypatch):
    provider = tmp_path / "empty.db"
    _provider_db(provider, [])
    factkit_db = tmp_path / "factkit-vectors.db"
    monkeypatch.setattr(holo, "_factkit_vectors_db_path", lambda **kw: factkit_db)
    store = holo.ReadOnlyStore(provider)

    try:
        out = holo.embedding_coverage(store, model=MODEL)
    finally:
        store.close()

    assert out["facts"] == 0
    assert out["embedded_for_model"] == 0
    assert out["missing"] == 0
    assert out["coverage"] == 1.0
    assert out["last_error"] == ""
    assert not factkit_db.exists()


def test_missing_and_coverage_math(provider_store):
    store, factkit_db = provider_store
    _vectors_db(factkit_db, [(1, MODEL), (2, MODEL)])

    out = holo.embedding_coverage(store, model=MODEL)

    assert out["facts"] == 3
    assert out["embedded_for_model"] == 2
    assert out["missing"] == 1
    assert out["coverage"] == 0.667


def test_absent_vectors_db_is_not_created(provider_store):
    store, factkit_db = provider_store

    out = holo.embedding_coverage(store, model=MODEL)

    assert out["embedded_for_model"] == 0
    assert out["coverage"] == 0.0
    assert not factkit_db.exists()


def test_vectors_for_another_model_do_not_count(provider_store):
    store, factkit_db = provider_store
    _vectors_db(factkit_db, [(1, "text-embedding-3-small")])

    out = holo.embedding_coverage(store, model=MODEL)

    assert out["embedded_for_model"] == 0
    assert out["missing"] == 3


def test_last_error_comes_from_plugin_health(provider_store, monkeypatch):
    store, _ = provider_store
    monkeypatch.setattr(holo, "plugin_health", lambda: {"last_error": "endpoint down"})

    assert holo.embedding_coverage(store, model=MODEL)["last_error"] == "endpoint down"


def test_plugin_health_degrades_without_the_plugin(monkeypatch):
    import importlib

    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", missing)

    assert holo.plugin_health() == {}
