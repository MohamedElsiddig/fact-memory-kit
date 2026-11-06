"""Temporal-validity tests: additive schema, stamping, as-of reads, backfill.

No network and no live store: every case runs against a ``tmp_path`` database,
either the factkit ``Store`` or a minimal stand-in with the holographic
provider's private write contract (``_write``/``_tx``/``read``).
"""
from __future__ import annotations

import contextlib
import sqlite3

import pytest

from factkit import holo, temporal
from factkit.store import Store

AT = "2026-03-10T12:00:00Z"
BEFORE = "2026-03-01T00:00:00Z"
AFTER = "2026-04-01T00:00:00Z"


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "facts.db")
    yield s
    s.close()


def seed(store, *contents: str) -> list[int]:
    return [store.add_fact(text, category="entity", tags="test")["fact_id"]
            for text in contents]


def columns(db_path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(facts)")}
    finally:
        conn.close()


# ------------------------------------------------------------------- schema
def test_ensure_schema_creates_columns_and_index_on_bare_store(tmp_path):
    bare = Store(tmp_path / "bare.db", init=False)  # connection, no schema at all
    try:
        assert temporal.ensure_schema(bare)["added"] == ["valid_from", "valid_to"]
        assert {"valid_from", "valid_to"} <= columns(tmp_path / "bare.db")
        names = {row[0] for row in bare.db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'")}
        assert temporal.INDEX_NAME in names
        # the minimal table is usable, not just present
        assert bare.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
    finally:
        bare.close()


def test_ensure_schema_is_idempotent(store):
    seed(store, "Atlas Server has 16 GB RAM")
    first = temporal.ensure_schema(store)
    second = temporal.ensure_schema(store)
    assert first["added"] == ["valid_from", "valid_to"]
    assert second["added"] == []
    assert {"valid_from", "valid_to"} <= set(second["columns"])
    count = store.db.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?",
        (temporal.INDEX_NAME,)).fetchone()[0]
    assert count == 1


def test_existing_rows_get_null_bounds(store):
    ids = seed(store, "Atlas Server has 16 GB RAM", "Beacon Mini runs Caddy 2.7.6")
    temporal.ensure_schema(store)
    for fid in ids:
        row = store.get(fid)
        assert row["valid_from"] is None
        assert row["valid_to"] is None
    assert temporal.is_valid(store.get(ids[0]), as_of="1999-01-01T00:00:00Z") is True


# -------------------------------------------------------------------- stamp
def test_stamp_sets_and_clears_each_bound(store):
    fid = seed(store, "Atlas Server has 16 GB RAM")[0]
    temporal.ensure_schema(store)

    row = temporal.stamp(store, fid, valid_from=BEFORE, valid_to=AFTER)
    assert row["ok"] is True
    assert row["valid_from"] == BEFORE
    assert row["valid_to"] == AFTER

    # clear only valid_to; valid_from must survive
    row = temporal.stamp(store, fid, valid_to=None)
    assert row["valid_to"] is None
    assert row["valid_from"] == BEFORE

    # clear only valid_from
    row = temporal.stamp(store, fid, valid_from=None)
    assert row["valid_from"] is None
    assert store.get(fid)["valid_to"] is None


def test_stamp_refuses_the_no_op(store):
    fid = seed(store, "Atlas Server has 16 GB RAM")[0]
    temporal.ensure_schema(store)
    before = store.get(fid)

    refused = temporal.stamp(store, fid)
    assert refused["ok"] is False
    assert refused["refused"] is True
    assert "nothing to stamp" in refused["reason"]
    assert store.get(fid) == before

    # clearing two already-NULL bounds is also a no-op, not a silent write
    refused = temporal.stamp(store, fid, valid_from=None, valid_to=None)
    assert refused["refused"] is True
    assert store.get(fid) == before


def test_stamp_rejects_unknown_fact(store):
    temporal.ensure_schema(store)
    with pytest.raises(ValueError):
        temporal.stamp(store, 999, valid_from=BEFORE)


def test_stamp_uses_the_store_write_contract(tmp_path):
    """A provider-shaped store writes through _write/_tx, not a second handle."""
    class PluginStore:
        def __init__(self, path):
            self.db_path = str(path)
            self.conn = sqlite3.connect(self.db_path, isolation_level=None)
            self.conn.row_factory = sqlite3.Row
            self.conn.executescript(
                "CREATE TABLE facts(fact_id INTEGER PRIMARY KEY, content TEXT NOT NULL,"
                " created_at TEXT, updated_at TEXT);"
                "INSERT INTO facts(content) VALUES ('Atlas Server has 16 GB RAM');")

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

    s = PluginStore(tmp_path / "plugin.db")
    temporal.ensure_schema(s)
    row = temporal.stamp(s, 1, valid_from=BEFORE)
    assert row["valid_from"] == BEFORE
    assert s.conn.execute("SELECT valid_from FROM facts WHERE fact_id=1").fetchone()[0] == BEFORE


# ----------------------------------------------------------------- validity
def test_is_valid_open_bounds():
    assert temporal.is_valid({"valid_from": None, "valid_to": None}, as_of=AT) is True
    assert temporal.is_valid({}, as_of=AT) is True
    assert temporal.is_valid({"valid_from": BEFORE}, as_of=AT) is True
    assert temporal.is_valid({"valid_to": AFTER}, as_of=AT) is True


def test_is_valid_past_valid_to_and_exact_boundary():
    fact = {"valid_from": BEFORE, "valid_to": AFTER}
    assert temporal.is_valid(fact, as_of="2026-03-15T00:00:00Z") is True
    assert temporal.is_valid(fact, as_of=AFTER) is True          # inclusive
    assert temporal.is_valid(fact, as_of="2026-04-01T00:00:01Z") is False


def test_is_valid_future_valid_from_and_exact_boundary():
    fact = {"valid_from": AT, "valid_to": None}
    assert temporal.is_valid(fact, as_of="2026-03-10T11:59:59Z") is False
    assert temporal.is_valid(fact, as_of=AT) is True             # inclusive
    assert temporal.is_valid(fact, as_of=AFTER) is True


def test_is_valid_accepts_sqlite_timestamps_and_objects():
    fact = {"valid_from": "2026-03-01 00:00:00", "valid_to": None}
    assert temporal.is_valid(fact, as_of="2026-03-02T00:00:00Z") is True
    assert temporal.is_valid(fact, as_of="2026-02-01T00:00:00Z") is False


def test_filter_preserves_order_and_does_not_mutate():
    rows = [
        {"fact_id": 1, "valid_from": None, "valid_to": "2026-01-01T00:00:00Z"},
        {"fact_id": 2, "valid_from": BEFORE, "valid_to": None},
        {"fact_id": 3, "valid_from": "2026-05-01T00:00:00Z", "valid_to": None},
        {"fact_id": 4, "valid_from": None, "valid_to": None},
    ]
    snapshot = [dict(r) for r in rows]
    kept = temporal.filter_valid(rows, as_of=AT)
    assert [r["fact_id"] for r in kept] == [2, 4]
    assert rows == snapshot
    assert temporal.filter_valid([], as_of=AT) == []


# ---------------------------------------------------------------- supersede
def test_supersede_sets_both_rows_and_deletes_nothing(store):
    old_id, new_id = seed(store, "Atlas Server has 16 GB RAM",
                          "Atlas Server has 32 GB RAM")
    temporal.ensure_schema(store)
    before_count = store.count()

    result = temporal.supersede(store, old_id, new_id, at=AT)
    assert result["ok"] is True
    assert result["at"] == AT
    assert result["old"]["valid_to"] == AT
    assert result["new"]["valid_from"] == AT
    assert result["old"]["valid_from"] is None
    assert result["new"]["valid_to"] is None

    assert store.count() == before_count == 2
    assert store.get(old_id)["content"] == "Atlas Server has 16 GB RAM"
    assert store.get(new_id)["content"] == "Atlas Server has 32 GB RAM"

    # the old claim is still answerable for a date before the switch
    assert temporal.is_valid(store.get(old_id), as_of=BEFORE) is True
    assert temporal.is_valid(store.get(old_id), as_of=AT) is True
    assert temporal.is_valid(store.get(old_id), as_of=AFTER) is False
    assert temporal.is_valid(store.get(new_id), as_of=BEFORE) is False


def test_supersede_same_id_refused_and_rows_untouched(store):
    fid = seed(store, "Atlas Server has 16 GB RAM")[0]
    temporal.ensure_schema(store)
    before = store.get(fid)
    with pytest.raises(ValueError):
        temporal.supersede(store, fid, fid, at=AT)
    assert store.get(fid) == before


def test_supersede_defaults_to_now(store):
    old_id, new_id = seed(store, "port 8080", "port 9090")
    temporal.ensure_schema(store)
    result = temporal.supersede(store, old_id, new_id)
    assert result["old"]["valid_to"] == result["new"]["valid_from"]
    assert temporal.is_valid(store.get(old_id)) is True   # inclusive "now"
    assert temporal.is_valid(store.get(new_id)) is True


# ------------------------------------------------------------------ backfill
def test_backfill_fills_valid_from_and_is_idempotent(store):
    ids = seed(store, "Atlas Server has 16 GB RAM", "Beacon Mini runs Caddy 2.7.6")
    temporal.ensure_schema(store)
    created = [store.get(fid)["created_at"] for fid in ids]
    contents = [store.get(fid)["content"] for fid in ids]

    first = temporal.backfill(store)
    assert first["total"] == 2
    assert first["backfilled"] == 2
    assert first["already"] == 0
    for fid, stamp in zip(ids, created):
        row = store.get(fid)
        assert row["valid_to"] is None
        assert temporal.is_valid(row, as_of=stamp) is True
        assert temporal.is_valid(row, as_of="2000-01-01T00:00:00Z") is False
    assert [store.get(fid)["content"] for fid in ids] == contents

    second = temporal.backfill(store)
    assert second["backfilled"] == 0
    assert second["already"] == 2
    assert [store.get(fid)["valid_from"] for fid in ids] == \
           [store.get(fid)["valid_from"] for fid in ids]


# -------------------------------------------------------------- as_of_query
def test_as_of_query_filters_hits(store, monkeypatch):
    expired, current, future = seed(
        store, "Atlas Server has 16 GB RAM",
        "Atlas Server has 32 GB RAM",
        "Atlas Server will have 64 GB RAM")
    temporal.ensure_schema(store)
    temporal.stamp(store, expired, valid_to="2026-02-01T00:00:00Z")
    temporal.stamp(store, future, valid_from="2026-06-01T00:00:00Z")
    temporal.stamp(store, current, valid_from=BEFORE)

    canned = [{"fact_id": expired, "content": "old", "score": 0.91},
              {"fact_id": current, "content": "now", "score": 0.88},
              {"fact_id": future, "content": "later", "score": 0.80}]
    seen = {}

    def fake_prefetch(store_, message, *, limit=5, **kwargs):
        seen.update({"message": message, "limit": limit, "kwargs": kwargs})
        return [dict(hit) for hit in canned[:limit]]

    monkeypatch.setattr(holo, "prefetch", fake_prefetch)

    result = temporal.as_of_query(store, "how much RAM does Atlas have?",
                                  as_of=AT, limit=5, min_score=0.2)
    assert result["as_of"] == AT
    assert result["for_query"] == "how much RAM does Atlas have?"
    assert result["available"] is True
    assert [h["fact_id"] for h in result["hits"]] == [current]
    assert result["hidden"] == 2
    hit = result["hits"][0]
    assert set(hit) == {"fact_id", "content", "score", "valid_from", "valid_to"}
    assert hit["valid_from"] == BEFORE and hit["valid_to"] is None
    assert seen["kwargs"] == {"min_score": 0.2}
    assert seen["limit"] >= 5  # bounded over-fetch fills the answer


def test_as_of_query_respects_limit_after_filtering(store, monkeypatch):
    ids = seed(store, *[f"Atlas slot {i} holds value {i}" for i in range(6)])
    temporal.ensure_schema(store)
    canned = [{"fact_id": fid, "content": "x", "score": 0.9 - i / 100}
              for i, fid in enumerate(ids)]
    monkeypatch.setattr(holo, "prefetch", lambda *a, **k: list(canned))
    result = temporal.as_of_query(store, "atlas", as_of=AT, limit=2)
    assert [h["fact_id"] for h in result["hits"]] == ids[:2]
    assert result["hidden"] == 0


def test_as_of_query_reports_unavailable_embeddings(store, monkeypatch):
    temporal.ensure_schema(store)
    monkeypatch.setattr(holo, "prefetch", lambda *a, **k: None)
    result = temporal.as_of_query(store, "atlas", as_of=AT)
    assert result["available"] is False
    assert result["hits"] == []
    assert result["hidden"] == 0
    assert result["as_of"] == AT


# ------------------------------------------------------------------- safety
def test_read_only_helpers_do_not_touch_the_store(store):
    fid = seed(store, "Atlas Server has 16 GB RAM")[0]
    temporal.ensure_schema(store)
    temporal.stamp(store, fid, valid_from=BEFORE, valid_to=AFTER)
    before = [store.get(fid), store.count(),
              store.db.execute("SELECT COUNT(*) FROM audit").fetchone()[0]]
    rows = [store.get(fid)]
    assert temporal.filter_valid(rows, as_of=AT) == rows
    assert temporal.filter_valid(rows, as_of=AFTER)
    assert temporal.is_valid(rows[0], as_of=AFTER) is True
    after = [store.get(fid), store.count(),
             store.db.execute("SELECT COUNT(*) FROM audit").fetchone()[0]]
    assert before == after


def test_empty_store_is_tolerated(store, monkeypatch):
    assert temporal.ensure_schema(store)["added"] == ["valid_from", "valid_to"]
    assert temporal.backfill(store) == {"ok": True, "total": 0, "backfilled": 0,
                                        "already": 0, "missing_created_at": 0}
    monkeypatch.setattr(holo, "prefetch", lambda *a, **k: [])
    result = temporal.as_of_query(store, "anything", as_of=AT)
    assert result == {"as_of": AT, "for_query": "anything", "available": True,
                      "hits": [], "hidden": 0}
    assert temporal.filter_valid([], as_of=AT) == []
