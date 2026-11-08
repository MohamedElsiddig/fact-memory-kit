"""Vector storage routing: provider handles keep vectors in factkit's own DB.

No network and no host paths: the provider DB and factkit's vectors DB both
live under ``tmp_path``, and every vector is canned.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from factkit import holo, vectors

MODEL = "text-embedding-nomic-embed-text-v1.5"
QUERY = "atlas memory specs"
QUERY_VEC = [1.0, 0.0, 0.0, 0.0]


def _at(cosine: float) -> list[float]:
    """A 4-dim vector whose cosine against QUERY_VEC is exactly ``cosine``."""
    return [cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine)), 0.0, 0.0]


class _MapClient:
    """Canned text -> vector mapping; deterministic and offline."""

    def __init__(self, table):
        self.table = {" ".join(k.lower().split()): list(v) for k, v in table.items()}
        self.timeout, self.retries = 1.0, 0

    def embed_one(self, text):
        return list(self.table[" ".join((text or "").lower().split())])

    def embed(self, texts):
        return [self.embed_one(t) for t in texts]


def _make_provider(tmp_path, facts):
    """Provider-shaped DB. ``fact_embeddings`` deliberately has no ``created_at``."""
    db = tmp_path / "provider.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE facts(fact_id INTEGER PRIMARY KEY, content TEXT, category TEXT,
                           tags TEXT, trust_score REAL, created_at TEXT, updated_at TEXT);
        CREATE TABLE fact_embeddings(fact_id INTEGER, model TEXT, dims INTEGER, vec BLOB);
        """
    )
    table = {}
    for fact_id, (content, cosine) in enumerate(facts, 1):
        conn.execute(
            "INSERT INTO facts(fact_id,content,category,tags,trust_score,created_at,updated_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                fact_id,
                content,
                "entity",
                "",
                0.9,
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
            ),
        )
        vec = _at(cosine)
        conn.execute(
            "INSERT INTO fact_embeddings VALUES (?,?,?,?)",
            (fact_id, MODEL, len(vec), vectors.pack(vec)),
        )
        table[content] = vec
    conn.commit()
    conn.close()
    return db, table


def _install(monkeypatch, table):
    import factkit.config as config_mod
    import factkit.embed as embed_mod

    table.setdefault(QUERY, QUERY_VEC)

    class _Cfg:
        embed_model = MODEL

    monkeypatch.setattr(config_mod.Config, "load", lambda db_path=None: _Cfg())
    monkeypatch.setattr(embed_mod, "from_config", lambda cfg: _MapClient(table))
    # never touch the host embedding cache
    monkeypatch.setattr(holo, "_query_cache", lambda: None)


def _scalar(path, sql):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql).fetchone()[0]
    finally:
        conn.close()


def test_prefetch_migrates_provider_vectors_into_factkit_db(
    monkeypatch, tmp_path, caplog
):
    provider, table = _make_provider(
        tmp_path,
        [
            ("Atlas Server has 16 GB RAM", 0.90),
            ("Backup job runs at 02:00 daily", 0.10),
        ],
    )
    factkit_db = tmp_path / "factkit-vectors.db"
    monkeypatch.setattr(holo, "_factkit_vectors_db_path", lambda: factkit_db)
    monkeypatch.setattr(holo, "_VECTOR_MIGRATION_ATTEMPTED", set())
    _install(monkeypatch, table)

    calls = {"count": 0}
    real_migrate = holo.migrate_vectors_from_provider

    def counting_migrate(path=None):
        calls["count"] += 1
        return real_migrate(path)

    monkeypatch.setattr(holo, "migrate_vectors_from_provider", counting_migrate)

    store = holo.ReadOnlyStore(provider)
    try:
        assert (
            not factkit_db.exists()
            or _scalar(factkit_db, "SELECT COUNT(*) FROM fact_embeddings") == 0
        )
        with caplog.at_level("INFO", logger="factkit.holo"):
            hits = holo.prefetch(store, QUERY, min_score=0.0, min_trust=0.0, limit=5)
    finally:
        store.close()

    assert calls["count"] == 1  # attempted exactly once
    assert _scalar(factkit_db, "SELECT COUNT(*) FROM fact_embeddings") == 2
    assert _scalar(provider, "SELECT COUNT(*) FROM fact_embeddings") == 2
    assert [h["fact_id"] for h in hits][0] == 1  # cosine 0.90 wins
    assert "migrated 2 vectors from provider DB" in caplog.text


def test_prefetch_does_not_remigrate_when_factkit_db_has_rows(monkeypatch, tmp_path):
    provider, table = _make_provider(
        tmp_path,
        [
            ("Atlas Server has 16 GB RAM", 0.90),
        ],
    )
    factkit_db = tmp_path / "factkit-vectors.db"
    conn = sqlite3.connect(factkit_db)
    conn.executescript(holo._FACTKIT_VECTORS_SCHEMA)
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, model, dims, vec) VALUES (?,?,?,?)",
        (1, MODEL, 4, vectors.pack(_at(0.90))),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(holo, "_factkit_vectors_db_path", lambda: factkit_db)
    monkeypatch.setattr(holo, "_VECTOR_MIGRATION_ATTEMPTED", set())
    _install(monkeypatch, table)

    called = {"count": 0}

    def counting_migrate(path=None):
        called["count"] += 1
        return {"ok": True, "migrated": 0}

    monkeypatch.setattr(holo, "migrate_vectors_from_provider", counting_migrate)

    store = holo.ReadOnlyStore(provider)
    try:
        hits = holo.prefetch(store, QUERY, min_score=0.0, min_trust=0.0, limit=5)
    finally:
        store.close()

    assert called["count"] == 0
    assert [h["fact_id"] for h in hits] == [1]


def test_vectors_conn_routes_provider_handle_to_factkit_db(monkeypatch, tmp_path):
    provider, _ = _make_provider(tmp_path, [("Atlas Server has 16 GB RAM", 0.9)])
    factkit_db = tmp_path / "factkit-vectors.db"
    monkeypatch.setattr(holo, "_factkit_vectors_db_path", lambda: factkit_db)

    store = holo.ReadOnlyStore(provider)
    try:
        conn, close = holo._vectors_conn(store, readonly=True)
        try:
            assert close is True
            path = conn.execute("PRAGMA database_list").fetchone()[2]
            assert Path(path).resolve() == factkit_db.resolve()
            assert (
                conn.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()[0] == 0
            )
        finally:
            conn.close()
    finally:
        store.close()


def test_vectors_conn_keeps_factkit_store_on_own_db(tmp_path):
    from factkit.store import Store

    path = tmp_path / "facts.db"
    store = Store(path)
    try:
        conn, close = holo._vectors_conn(store)
        assert close is False
        assert conn is store.db
        conn.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()
    finally:
        store.close()

    readonly = Store(path, readonly=True)
    try:
        conn, close = holo._vectors_conn(readonly, readonly=True)
        assert close is False
        assert conn is readonly.db
        conn.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()
    finally:
        readonly.close()


def test_duck_adapter_keeps_legacy_read_through(monkeypatch, tmp_path):
    """A non-provider adapter that owns its vectors still works (test_scoring shape)."""
    path = tmp_path / "duck.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE facts(fact_id INTEGER PRIMARY KEY, content TEXT, category TEXT,
                           tags TEXT, trust_score REAL, created_at TEXT, updated_at TEXT);
        CREATE TABLE fact_embeddings(fact_id INTEGER, model TEXT, dims INTEGER, vec BLOB);
        """
    )
    conn.execute(
        "INSERT INTO facts VALUES (1,'Atlas Server has 16 GB RAM','entity','',0.9,"
        "'2026-01-01 00:00:00','2026-01-01 00:00:00')"
    )
    conn.execute(
        "INSERT INTO fact_embeddings VALUES (1,?,4,?)", (MODEL, vectors.pack(_at(0.90)))
    )
    conn.commit()
    conn.close()

    class _Duck:
        db_path = str(path)

        def read(self, sql, params=()):
            c = sqlite3.connect(self.db_path)
            c.row_factory = sqlite3.Row
            try:
                return list(c.execute(sql, params))
            finally:
                c.close()

    monkeypatch.setattr(
        holo, "_factkit_vectors_db_path", lambda: tmp_path / "factkit-vectors.db"
    )
    monkeypatch.setattr(holo, "_VECTOR_MIGRATION_ATTEMPTED", set())
    _install(monkeypatch, {"Atlas Server has 16 GB RAM": _at(0.90)})

    called = {"count": 0}

    def counting_migrate(path=None):
        called["count"] += 1
        return {"ok": True, "migrated": 0}

    monkeypatch.setattr(holo, "migrate_vectors_from_provider", counting_migrate)

    hits = holo.prefetch(_Duck(), QUERY, min_score=0.0, min_trust=0.0, limit=5)
    assert [h["fact_id"] for h in hits] == [1]
    assert called["count"] == 0  # duck adapters are not migrated
