"""Tombstone/restore tests.

The fixture uses the holographic plugin's real ``MemoryStore`` when it is
importable (the production shape: FTS triggers, declared entities, HRR vector).
Where the plugin is absent, a minimal store with the same write contract and an
FTS5 table is used, so the suite still runs against the documented API.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3

import pytest

from factkit import maintenance

_MINI_SCHEMA = """
CREATE TABLE facts(
    fact_id INTEGER PRIMARY KEY,
    content TEXT NOT NULL UNIQUE,
    category TEXT,
    tags TEXT,
    trust_score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    hrr_vector BLOB
);
CREATE TABLE entities(entity_id INTEGER PRIMARY KEY, name TEXT,
                      entity_type TEXT DEFAULT 'unknown');
CREATE TABLE fact_entities(fact_id INTEGER, entity_id INTEGER);
CREATE VIRTUAL TABLE facts_fts USING fts5(content, tags);
CREATE TRIGGER facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags) VALUES (new.fact_id, new.content, new.tags);
END;
CREATE TRIGGER facts_ad AFTER DELETE ON facts BEGIN
    DELETE FROM facts_fts WHERE rowid = old.fact_id;
END;
"""


class _MiniStore:
    """Minimal stand-in for holographic.MemoryStore (same private write contract)."""

    def __init__(self, path):
        self.db_path = str(path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False,
                                    isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_MINI_SCHEMA)

    def read(self, sql, params=()):
        return self.conn.execute(sql, params).fetchall()

    def _write(self, sql, params=()):
        return self.conn.execute(sql, params)

    @contextlib.contextmanager
    def _tx(self):
        if self.conn.in_transaction:
            yield
            return
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    def count_facts(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])

    def _entity(self, name: str) -> int:
        row = self.conn.execute("SELECT entity_id FROM entities WHERE name = ?",
                                (name,)).fetchone()
        if row is not None:
            self.conn.execute("UPDATE entities SET entity_type = 'declared' "
                              "WHERE entity_id = ?", (row[0],))
            return int(row[0])
        cur = self.conn.execute("INSERT INTO entities(name, entity_type) VALUES (?, 'declared')",
                                (name,))
        return int(cur.lastrowid)

    def add_fact_status(self, content, category="general", tags="", *, trust=None,
                        entities=()):
        content = (content or "").strip()
        if not content:
            raise ValueError("content must not be empty")
        with self._tx():
            try:
                cur = self._write(
                    "INSERT INTO facts(content, category, tags, trust_score) VALUES (?,?,?,?)",
                    (content, category, tags, 0.5 if trust is None else trust))
                fact_id = int(cur.lastrowid)
            except sqlite3.IntegrityError:
                return int(self.conn.execute("SELECT fact_id FROM facts WHERE content = ?",
                                             (content,)).fetchone()[0]), False
            for name in entities or ():
                if isinstance(name, str) and name.strip():
                    entity_id = self._entity(name.strip())
                    self._write("INSERT OR IGNORE INTO fact_entities VALUES (?,?)",
                                (fact_id, entity_id))
            return fact_id, True

    def close(self):
        self.conn.close()


def _make_store(path):
    try:
        from factkit import holo
        return holo.load_plugin_store(path)
    except Exception:  # noqa: BLE001 - plugin not installed in this environment
        return _MiniStore(path)


@pytest.fixture()
def store(tmp_path):
    s = _make_store(tmp_path / "memory_store.db")
    try:
        yield s
    finally:
        s.close()


def seed(store, content, *, category="entity", tags="hw", trust=0.8, entities=()):
    fact_id, _created = store.add_fact_status(content, category=category, tags=tags,
                                              trust=trust, entities=list(entities))
    return int(fact_id)


def fts_ids(store, term):
    rows = store.read("SELECT rowid FROM facts_fts WHERE facts_fts MATCH ?", (term,))
    return [int(row[0]) for row in rows]


def tombstone_row(store, fact_id):
    rows = store.read("SELECT * FROM fact_tombstones WHERE fact_id = ?", (fact_id,))
    return dict(rows[0]) if rows else None


# ------------------------------------------------------------- table creation
def test_ensure_tombstones_is_idempotent(store):
    maintenance.ensure_tombstones(store)
    maintenance.ensure_tombstones(store)
    tables = store.read("SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name='fact_tombstones'")
    indexes = store.read("SELECT name FROM sqlite_master WHERE type='index' "
                         "AND name='idx_fact_tombstones_restored'")
    assert len(tables) == 1
    assert len(indexes) == 1
    assert maintenance.list_tombstones(store) == []


# ----------------------------------------------------------------- tombstone
def test_tombstone_captures_content_and_metadata(store):
    fid = seed(store, "Atlas Server has 16 GB RAM", category="entity", tags="pi,hw",
               trust=0.83)
    row = maintenance.tombstone_fact(store, fid, reason="stale ram", batch_id="b-1",
                                     by="tester")
    assert row["fact_id"] == fid
    assert row["content"] == "Atlas Server has 16 GB RAM"
    assert row["category"] == "entity"
    assert row["tags"] == "pi,hw"
    assert row["trust_score"] == pytest.approx(0.83)
    assert row["reason"] == "stale ram"
    assert row["batch_id"] == "b-1"
    assert row["retired_by"] == "tester"
    assert row["retired_at"] and row["restored_at"] is None
    # tombstoning snapshots; it does not retire on its own
    assert store.count_facts() == 1


def test_tombstone_captures_declared_entities(store):
    fid = seed(store, "Atlas Server has 16 GB RAM", entities=["Atlas Server"])
    row = maintenance.tombstone_fact(store, fid)
    names = json.loads(row["entities"])
    assert "Atlas Server" in names
    assert all(isinstance(name, str) for name in names)


def test_tombstone_stores_hrr_vector_when_present(store):
    fid = seed(store, "Atlas Server has 16 GB RAM")
    row = maintenance.tombstone_fact(store, fid)
    # the plugin computes a vector on insert; the mini store leaves it NULL
    columns = {str(r[0]) for r in store.read("PRAGMA table_info(facts)")}
    if "hrr_vector" in columns:
        live = store.read("SELECT hrr_vector FROM facts WHERE fact_id = ?", (fid,))[0][0]
        if live is not None:
            assert bytes(row["hrr_vector"]) == bytes(live)


def test_tombstone_replaces_previous_snapshot(store):
    fid = seed(store, "Atlas Server has 16 GB RAM")
    maintenance.tombstone_fact(store, fid, reason="first")
    maintenance.tombstone_fact(store, fid, reason="second")
    rows = maintenance.list_tombstones(store)
    assert len(rows) == 1 and rows[0]["reason"] == "second"


def test_tombstone_missing_fact_raises(store):
    with pytest.raises(ValueError, match="not found"):
        maintenance.tombstone_fact(store, 4242)


def test_tombstone_joins_an_open_transaction(store):
    fid = seed(store, "Atlas Server has 16 GB RAM")
    with pytest.raises(RuntimeError):
        with store._tx():
            maintenance.tombstone_fact(store, fid, reason="rolled back")
            raise RuntimeError("abort")
    # written on the same connection, so the rollback took it with it
    assert maintenance.list_tombstones(store) == []


# ------------------------------------------------------------------ restore
def test_restore_readds_a_searchable_fact(store):
    fid = seed(store, "Transcriber listens on port 8010", entities=["Transcriber"])
    maintenance.tombstone_fact(store, fid)
    assert store.count_facts() == 1
    assert _remove(store, fid) is True
    assert store.count_facts() == 0
    assert fts_ids(store, "Transcriber") == []

    result = maintenance.restore_fact(store, fid)
    assert result["restored_from"] == fid
    assert result["content"] == "Transcriber listens on port 8010"
    assert result["fact_id"] != fid
    assert store.count_facts() == 1
    assert fts_ids(store, "Transcriber") == [result["fact_id"]]


def test_restore_marks_restored_at_and_hides_from_default_listing(store):
    fid = seed(store, "Backup runs at 02:00 daily")
    maintenance.tombstone_fact(store, fid)
    assert len(maintenance.list_tombstones(store)) == 1
    maintenance.restore_fact(store, fid)

    assert maintenance.list_tombstones(store) == []
    rows = maintenance.list_tombstones(store, include_restored=True)
    assert len(rows) == 1 and rows[0]["restored_at"] is not None


def test_restore_relinks_declared_entities(store):
    fid = seed(store, "Atlas Server has 16 GB RAM", entities=["Atlas Server"])
    maintenance.tombstone_fact(store, fid)
    _remove(store, fid)
    new_id = maintenance.restore_fact(store, fid)["fact_id"]
    linked = {row["name"] for row in store.read(
        "SELECT e.name FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id "
        "WHERE fe.fact_id = ? AND e.entity_type = 'declared'", (new_id,))}
    assert "Atlas Server" in linked


def test_restore_twice_raises(store):
    fid = seed(store, "Proxy listens on port 443")
    maintenance.tombstone_fact(store, fid)
    _remove(store, fid)
    maintenance.restore_fact(store, fid)
    with pytest.raises(ValueError, match="already restored"):
        maintenance.restore_fact(store, fid)


def test_restore_without_tombstone_raises(store):
    with pytest.raises(ValueError, match="no tombstone"):
        maintenance.restore_fact(store, 999)


def test_restore_after_content_reappears_reuses_the_row(store):
    fid = seed(store, "Backup runs at 02:00 daily")
    maintenance.tombstone_fact(store, fid)
    # the fact was never deleted: add_fact_status returns the existing id
    result = maintenance.restore_fact(store, fid)
    assert result["fact_id"] == fid
    assert store.count_facts() == 1
    assert maintenance.list_tombstones(store) == []


# -------------------------------------------------------------------- purge
def test_purge_removes_restored_and_old_rows_only(store):
    restored = seed(store, "Atlas Server has 16 GB RAM")
    old = seed(store, "Beacon Mini has 1 GB RAM")
    fresh = seed(store, "Proxy listens on port 443")
    for fid in (restored, old, fresh):
        maintenance.tombstone_fact(store, fid)

    maintenance.restore_fact(store, restored)
    store._write("UPDATE fact_tombstones SET retired_at = datetime('now', '-120 days') "
                 "WHERE fact_id = ?", (old,))

    assert maintenance.purge_tombstones(store, days=90) == 2
    remaining = maintenance.list_tombstones(store, include_restored=True)
    assert [row["fact_id"] for row in remaining] == [fresh]


def test_purge_keeps_recent_unrestored_rows(store):
    fid = seed(store, "Backup runs at 02:00 daily")
    maintenance.tombstone_fact(store, fid)
    assert maintenance.purge_tombstones(store, days=90) == 0
    assert len(maintenance.list_tombstones(store)) == 1


def test_purge_zero_days_uses_now_as_cutoff(store):
    fid = seed(store, "Backup runs at 02:00 daily")
    maintenance.tombstone_fact(store, fid)
    # a fresh row is not *older* than now; a minute-old row is
    assert maintenance.purge_tombstones(store, days=0) == 0
    store._write("UPDATE fact_tombstones SET retired_at = datetime('now', '-1 minute') "
                 "WHERE fact_id = ?", (fid,))
    assert maintenance.purge_tombstones(store, days=0) == 1
    assert maintenance.list_tombstones(store, include_restored=True) == []


def test_tombstone_restore_without_plugin(tmp_path):
    """The module works with any store exposing the documented contract."""
    store = _MiniStore(tmp_path / "mini.db")
    try:
        fid = seed(store, "Atlas Server has 16 GB RAM", entities=["Atlas Server"])
        maintenance.tombstone_fact(store, fid, reason="mini")
        assert _remove(store, fid) is True
        out = maintenance.restore_fact(store, fid)
        assert fts_ids(store, "Atlas") == [out["fact_id"]]
        assert maintenance.list_tombstones(store) == []
    finally:
        store.close()


# ------------------------------------------------------------------- helper
def _remove(store, fact_id):
    """Retire a fact through the store API (plugin or mini)."""
    remover = getattr(store, "remove_fact", None)
    if callable(remover):
        return bool(remover(fact_id))
    with store._tx():
        store._write("DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,))
        cur = store._write("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
    return bool(cur.rowcount)
