"""Provenance tests: schema, backfill, record, snapshot diff and report.

Everything runs offline against a throwaway SQLite store built from the same
shape as the provider's ``facts`` table, so these tests do not need the
holographic plugin or a live memory database.
"""

from __future__ import annotations

import contextlib
import sqlite3

import pytest

from factkit import provenance

_FACTS_SCHEMA = """
CREATE TABLE facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_facts_trust ON facts(trust_score DESC);
"""

_BARE_SCHEMA = "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"


class _Store:
    """Duck-typed stand-in for holographic.MemoryStore (same write contract)."""

    def __init__(self, path, schema=_FACTS_SCHEMA):
        self.db_path = str(path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False,
                                    isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(schema)

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

    def close(self):
        self.conn.close()


class _ReadOnlyStore(_Store):
    """A store whose database refuses writes (permission failure)."""

    def _write(self, sql, params=()):
        raise sqlite3.OperationalError("attempt to write a readonly database")


@pytest.fixture()
def store(tmp_path):
    s = _Store(tmp_path / "facts.db")
    try:
        yield s
    finally:
        s.close()


def seed(store, content, *, trust=0.5):
    cur = store.conn.execute(
        "INSERT INTO facts(content, trust_score) VALUES (?, ?)", (content, trust))
    return int(cur.lastrowid)


def edit(store, fact_id, content):
    store.conn.execute("UPDATE facts SET content = ? WHERE fact_id = ?",
                       (content, fact_id))


def facts_rows(store):
    return [tuple(row) for row in store.conn.execute(
        "SELECT * FROM facts ORDER BY fact_id")]


def prov_row(store, fact_id):
    rows = store.read("SELECT * FROM fact_provenance WHERE fact_id = ?", (fact_id,))
    return dict(rows[0]) if rows else None


# --------------------------------------------------------------------- schema
def test_ensure_schema_creates_table_and_index_on_bare_store(tmp_path):
    store = _Store(tmp_path / "bare.db", schema=_BARE_SCHEMA)
    try:
        provenance.ensure_schema(store)

        columns = {row["name"]: row["type"] for row in
                   store.read("PRAGMA table_info(fact_provenance)")}
        assert list(columns) == ["fact_id", "origin", "actor", "first_seen",
                                 "last_seen", "content_hash", "revision", "note"]
        assert columns["fact_id"] == "INTEGER"
        indexes = {row["name"] for row in store.read(
            "SELECT name FROM sqlite_master WHERE type = 'index'")}
        assert provenance.INDEX in indexes
    finally:
        store.close()


def test_ensure_schema_is_idempotent_and_keeps_existing_rows(store):
    provenance.ensure_schema(store)
    store.conn.execute(
        "INSERT INTO fact_provenance(fact_id, origin, actor, first_seen, last_seen,"
        " content_hash, revision, note) VALUES (1, 'legacy', 'tester', 't0', 't0',"
        " 'abc', 7, 'keep me')")
    provenance.ensure_schema(store)

    assert len(store.read("SELECT name FROM sqlite_master WHERE type = 'table' "
                          "AND name = 'fact_provenance'")) == 1
    row = prov_row(store, 1)
    assert row["origin"] == "legacy" and row["revision"] == 7 and row["note"] == "keep me"


# ------------------------------------------------------------------- backfill
def test_backfill_inserts_one_row_per_fact(store):
    ids = [seed(store, "Atlas Server has 16 GB RAM"),
           seed(store, "Backup runs at 02:00 daily"),
           seed(store, "Proxy listens on port 443")]

    summary = provenance.backfill(store, origin="provider")

    assert summary["inserted"] == 3
    assert summary["scanned"] == 3
    assert summary["existing"] == 0
    rows = store.read("SELECT * FROM fact_provenance ORDER BY fact_id")
    assert [row["fact_id"] for row in rows] == ids
    for row in rows:
        content = store.read("SELECT content FROM facts WHERE fact_id = ?",
                             (row["fact_id"],))[0]["content"]
        assert row["origin"] == "provider"
        assert row["actor"] == "factkit"
        assert row["content_hash"] == provenance.hash_content(content)
        assert row["revision"] == 1
        assert row["first_seen"] == row["last_seen"] and row["first_seen"]


def test_backfill_is_idempotent_and_honours_batch(store):
    ids = [seed(store, "Atlas Server has 16 GB RAM"),
           seed(store, "Backup runs at 02:00 daily")]

    first = provenance.backfill(store, origin="provider", batch=1)
    second = provenance.backfill(store, origin="other", batch=1)

    assert first["inserted"] == 2
    assert second["inserted"] == 0
    assert second["existing"] == 2
    rows = store.read("SELECT fact_id, origin FROM fact_provenance ORDER BY fact_id")
    assert [row["fact_id"] for row in rows] == ids
    assert {row["origin"] for row in rows} == {"provider"}  # no re-stamping


def test_backfill_does_not_write_to_facts(store):
    ids = [seed(store, "Atlas Server has 16 GB RAM"),
           seed(store, "Backup runs at 02:00 daily")]
    before = facts_rows(store)

    provenance.backfill(store, origin="provider")

    assert facts_rows(store) == before
    assert store.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == len(ids)


# --------------------------------------------------------------------- record
def test_record_increments_revision_only_on_hash_change(store):
    fid = seed(store, "Atlas Server has 16 GB RAM")

    first = provenance.record(store, fid, origin="provider")
    assert first["revision"] == 1
    assert first["content_hash"] == provenance.hash_content("Atlas Server has 16 GB RAM")

    same = provenance.record(store, fid, origin="provider", note="seen again")
    assert same["revision"] == 1
    assert same["note"] == "seen again"

    edit(store, fid, "Atlas Server has 32 GB RAM")
    changed = provenance.record(store, fid, origin="provider")
    assert changed["revision"] == 2
    assert changed["content_hash"] == provenance.hash_content("Atlas Server has 32 GB RAM")

    again = provenance.record(store, fid, origin="provider")
    assert again["revision"] == 2


def test_record_preserves_first_seen_and_refreshes_origin(store):
    fid = seed(store, "Proxy listens on port 443")
    first = provenance.record(store, fid, origin="unknown", actor="factkit")

    second = provenance.record(store, fid, origin="hermes", actor="holographic",
                               note="provider touch")

    assert second["first_seen"] == first["first_seen"]
    assert second["origin"] == "hermes"
    assert second["actor"] == "holographic"
    assert second["note"] == "provider touch"
    assert second["revision"] == 1
    assert second["last_seen"] >= second["first_seen"]


def test_record_ignores_whitespace_only_changes(store):
    fid = seed(store, "Atlas Server has 16 GB RAM")
    provenance.record(store, fid, origin="provider")

    edit(store, fid, "  Atlas   Server has\n16 GB RAM  ")
    row = provenance.record(store, fid, origin="provider")

    assert row["revision"] == 1
    assert row["content_hash"] == provenance.hash_content("Atlas Server has 16 GB RAM")


def test_record_missing_fact_raises(store):
    with pytest.raises(ValueError, match="not found"):
        provenance.record(store, 4242, origin="provider")
    with pytest.raises(ValueError, match="invalid fact_id"):
        provenance.record(store, "not-an-id", origin="provider")


# -------------------------------------------------------------- snapshot_diff
def test_snapshot_diff_detects_changed_content(store):
    fid = seed(store, "Atlas Server has 16 GB RAM")
    other = seed(store, "Backup runs at 02:00 daily")
    provenance.backfill(store, origin="provider")

    edit(store, fid, "Atlas Server has 32 GB RAM")
    diff = provenance.snapshot_diff(store)

    assert [item["fact_id"] for item in diff["changed"]] == [fid]
    assert [item["fact_id"] for item in diff["missing"]] == []
    assert [item["fact_id"] for item in diff["new"]] == []
    changed = diff["changed"][0]
    assert changed["stored_hash"] == provenance.hash_content("Atlas Server has 16 GB RAM")
    assert changed["current_hash"] == provenance.hash_content("Atlas Server has 32 GB RAM")
    assert changed["origin"] == "provider"
    assert other not in [item["fact_id"] for item in diff["changed"]]


def test_snapshot_diff_reports_new_fact(store):
    first = seed(store, "Atlas Server has 16 GB RAM")
    provenance.backfill(store, origin="provider")
    fresh = seed(store, "Proxy listens on port 443")

    diff = provenance.snapshot_diff(store)

    assert [item["fact_id"] for item in diff["new"]] == [fresh]
    assert diff["changed"] == [] and diff["missing"] == []
    assert diff["new"][0]["content_hash"] == provenance.hash_content(
        "Proxy listens on port 443")
    assert first not in [item["fact_id"] for item in diff["new"]]


def test_snapshot_diff_reports_missing_fact(store):
    keep = seed(store, "Atlas Server has 16 GB RAM")
    gone = seed(store, "Backup runs at 02:00 daily")
    provenance.backfill(store, origin="provider")

    store.conn.execute("DELETE FROM facts WHERE fact_id = ?", (gone,))
    diff = provenance.snapshot_diff(store)

    assert [item["fact_id"] for item in diff["missing"]] == [gone]
    assert diff["missing"][0]["origin"] == "provider"
    assert diff["changed"] == [] and diff["new"] == []
    assert keep not in [item["fact_id"] for item in diff["missing"]]


def test_snapshot_diff_without_provenance_table_marks_every_fact_new(tmp_path):
    store = _Store(tmp_path / "untracked.db")
    try:
        ids = [seed(store, "Atlas Server has 16 GB RAM"),
               seed(store, "Proxy listens on port 443")]
        diff = provenance.snapshot_diff(store)
        assert [item["fact_id"] for item in diff["new"]] == ids
        assert diff["changed"] == [] and diff["missing"] == []
    finally:
        store.close()


# --------------------------------------------------------------------- report
def test_report_counts_origins_newest_and_untracked(store):
    first = seed(store, "Atlas Server has 16 GB RAM")
    second = seed(store, "Backup runs at 02:00 daily")
    third = seed(store, "Proxy listens on port 443")
    provenance.backfill(store, origin="legacy")
    provenance.record(store, third, origin="provider", actor="holographic",
                      note="provider write")
    untracked = seed(store, "Transcriber listens on port 8010")

    result = provenance.report(store)

    assert result["facts"] == 4
    assert result["tracked"] == 3
    assert result["untracked"] == 1
    assert result["by_origin"] == {"legacy": 2, "provider": 1}
    assert result["newest"][0]["fact_id"] == third
    assert result["newest"][0]["origin"] == "provider"
    assert [item["fact_id"] for item in result["no_provenance"]] == [untracked]
    assert {item["fact_id"] for item in result["newest"]} == {first, second, third}


def test_report_limit_caps_lists_but_not_counts(store):
    ids = [seed(store, f"Fact number {n} is stable") for n in range(1, 6)]
    provenance.backfill(store, origin="provider")
    seed(store, "Transcriber listens on port 8010")

    result = provenance.report(store, limit=1)

    assert result["facts"] == 6 and result["tracked"] == len(ids)
    assert len(result["newest"]) == 1
    assert len(result["no_provenance"]) == 1


# ----------------------------------------------------------- degraded stores
def test_missing_facts_table_is_tolerated(tmp_path):
    store = _Store(tmp_path / "bare.db", schema=_BARE_SCHEMA)
    try:
        provenance.ensure_schema(store)          # still creates provenance
        summary = provenance.backfill(store)
        assert summary["inserted"] == 0 and summary["scanned"] == 0
        assert "facts" in summary["error"]

        assert provenance.snapshot_diff(store) == {"changed": [], "missing": [],
                                                   "new": []}
        result = provenance.report(store)
        assert result["facts"] == 0 and result["untracked"] == 0
        assert result["newest"] == [] and result["no_provenance"] == []
    finally:
        store.close()


def test_read_only_store_is_reported_not_raised(tmp_path):
    store = _ReadOnlyStore(tmp_path / "locked.db")
    try:
        seed(store, "Atlas Server has 16 GB RAM")
        summary = provenance.backfill(store)
        assert summary["inserted"] == 0
        assert "readonly" in summary["error"]

        # read paths still work and see the fact as untracked
        diff = provenance.snapshot_diff(store)
        assert [item["fact_id"] for item in diff["new"]] == [1]
        result = provenance.report(store)
        assert result["facts"] == 1 and result["untracked"] == 1
    finally:
        store.close()


def test_full_cycle_leaves_the_facts_table_byte_identical(store):
    seed(store, "Atlas Server has 16 GB RAM")
    guard = seed(store, "Proxy listens on port 443")
    before = facts_rows(store)

    provenance.ensure_schema(store)
    provenance.backfill(store, origin="provider")
    provenance.record(store, guard, origin="hermes", note="touch")
    provenance.snapshot_diff(store)
    provenance.report(store)

    assert facts_rows(store) == before
    assert store.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == len(before)
