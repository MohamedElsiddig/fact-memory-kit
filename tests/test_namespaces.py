"""Namespaced recall: detection, assignment, filtering, scoping and stats.

The stores here are throwaway SQLite files with the provider's shape (a
``facts`` table with ``tags`` / ``category``) plus, where a test needs it, a
bank registry and a membership link table.  No holographic plugin, no live
database, no network: ``holo.prefetch`` is monkeypatched.
"""

from __future__ import annotations

import contextlib
import sqlite3

import pytest

from factkit import holo, namespaces

_FACTS = """
CREATE TABLE facts (
    fact_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    content    TEXT NOT NULL UNIQUE,
    category   TEXT DEFAULT 'general',
    tags       TEXT DEFAULT '',
    trust_score REAL DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# A bank registry with no fact->bank membership: this is exactly the live
# provider's shape (HRR category cache, NOT NULL vector, keyed cat:<category>).
_BANK_CACHE_ONLY = """
CREATE TABLE memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# A bank store that can answer "which facts belong to this bank".
_BANK_MEMBERSHIP = """
CREATE TABLE memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE fact_banks (
    fact_id INTEGER NOT NULL,
    bank_id INTEGER NOT NULL,
    PRIMARY KEY (fact_id, bank_id)
);
"""

_CATEGORIES_ONLY = """
CREATE TABLE facts (
    fact_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    content  TEXT NOT NULL UNIQUE,
    category TEXT DEFAULT 'general'
);
"""


class _Store:
    """Duck-typed stand-in for holographic.MemoryStore (same write contract)."""

    def __init__(self, path, schema=_FACTS):
        self.db_path = str(path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False,
                                    isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        if schema:
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

    def add(self, content, *, tags="", category="general", stamp=None):
        cur = self.conn.execute(
            "INSERT INTO facts (content, category, tags) VALUES (?,?,?)",
            (content, category, tags))
        if stamp:
            self.conn.execute("UPDATE facts SET updated_at = ? WHERE fact_id = ?",
                              (stamp, cur.lastrowid))
        return int(cur.lastrowid)

    def tags(self, fact_id):
        row = self.conn.execute("SELECT tags FROM facts WHERE fact_id = ?", (fact_id,)).fetchone()
        return row["tags"]

    def close(self):
        self.conn.close()


@pytest.fixture()
def store(tmp_path):
    s = _Store(tmp_path / "facts.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def bank_store(tmp_path):
    s = _Store(tmp_path / "banks.db", _FACTS + _BANK_MEMBERSHIP)
    try:
        yield s
    finally:
        s.close()


def _add_bank(store, name, *, count=0):
    cur = store.conn.execute("INSERT INTO memory_banks (bank_name, fact_count) VALUES (?,?)",
                             (name, count))
    return int(cur.lastrowid)


def _link(store, fact_id, bank_id):
    store.conn.execute("INSERT INTO fact_banks (fact_id, bank_id) VALUES (?,?)",
                       (fact_id, bank_id))


# ----------------------------------------------------------------- detection
def test_detect_bare_store_reports_tags_not_banks(store):
    info = namespaces.detect(store)
    assert info["banks"] is False
    assert info["tags"] is True
    assert info["categories"] is True
    assert "tags" in info["columns"] and "category" in info["columns"]
    assert set(info) == {"banks", "tags", "categories", "columns"}


def test_detect_requires_membership_not_just_the_bank_registry(tmp_path):
    """The live provider's memory_banks is a vector cache, not a membership map."""
    s = _Store(tmp_path / "cache.db", _FACTS + _BANK_CACHE_ONLY)
    try:
        s.conn.execute("INSERT INTO memory_banks (bank_name, vector, dim, fact_count) "
                       "VALUES ('cat:memory', x'00', 1024, 1)")
        s.add("Atlas has 16 GB RAM", tags="hardware")
        info = namespaces.detect(s)
        assert info["banks"] is False          # no fact_banks / bank_id column
        assert namespaces.list_namespaces(s) == []   # cat:memory is not a namespace
    finally:
        s.close()


def test_detect_fake_bank_table_with_membership(bank_store):
    info = namespaces.detect(bank_store)
    assert info["banks"] is True
    assert info["tags"] is True
    assert info["categories"] is True


# ------------------------------------------------------------------- listing
def test_list_counts_and_ordering_tags(store):
    a = store.add("a", tags="ns:work, project", stamp="2026-01-01 10:00:00")
    store.add("b", tags="ns:work", stamp="2026-02-01 10:00:00")
    store.add("c", tags="ns:home", stamp="2026-03-01 10:00:00")
    store.add("d", tags="plain, ns:")
    rows = namespaces.list_namespaces(store)
    assert [(r["namespace"], r["facts"]) for r in rows] == [("work", 2), ("home", 1)]
    assert rows[0]["last_seen"] == "2026-02-01 10:00:00"
    assert rows[0]["mechanism"] == "tags"
    assert store.tags(a).startswith("ns:work")   # listing is read-only


def test_list_bank_namespaces_excludes_category_banks(bank_store):
    f1 = bank_store.add("one")
    f2 = bank_store.add("two")
    f3 = bank_store.add("three")
    alpha = _add_bank(bank_store, "ns:alpha")
    beta = _add_bank(bank_store, "ns:beta")
    _add_bank(bank_store, "cat:memory", count=99)
    bank_store.conn.execute("UPDATE facts SET updated_at = '2026-05-01 00:00:00' WHERE fact_id IN (?,?)",
                            (f1, f2))
    _link(bank_store, f1, alpha)
    _link(bank_store, f2, alpha)
    _link(bank_store, f3, beta)

    rows = namespaces.list_namespaces(bank_store)
    assert [(r["namespace"], r["facts"]) for r in rows] == [("alpha", 2), ("beta", 1)]
    assert rows[0]["last_seen"] == "2026-05-01 00:00:00"
    assert all(r["mechanism"] == "banks" for r in rows)


def test_list_categories_when_no_tags_column(tmp_path):
    s = _Store(tmp_path / "cat.db", _CATEGORIES_ONLY)
    try:
        s.conn.execute("INSERT INTO facts (content, category) VALUES ('a','work')")
        s.conn.execute("INSERT INTO facts (content, category) VALUES ('b','work')")
        s.conn.execute("INSERT INTO facts (content, category) VALUES ('c','home')")
        rows = namespaces.list_namespaces(s)
        assert [(r["namespace"], r["facts"]) for r in rows] == [("work", 2), ("home", 1)]
        assert rows[0]["mechanism"] == "categories"
        # single-valued: assignment is refused rather than rewriting the category
        refusal = namespaces.assign(s, 1, "work", create=True)
        assert refusal["ok"] is False and "category" in refusal["reason"]
    finally:
        s.close()


# -------------------------------------------------------------------- assign
def test_assign_tag_form_appends_namespace_tag(store):
    store.add("seed", tags="ns:work")
    target = store.add("laptop specs", tags="hardware, shadow-keep")
    result = namespaces.assign(store, target, "work")
    assert result["ok"] is True
    assert result["mechanism"] == "tags" and result["changed"] is True
    assert store.tags(target) == "hardware, shadow-keep, ns:work"


def test_assign_tag_form_accepts_prefixed_namespace(store):
    store.add("seed", tags="ns:work")
    target = store.add("note")
    assert namespaces.assign(store, target, "ns:work")["ok"] is True
    assert store.tags(target) == "ns:work"


def test_assign_bank_form_links_membership(bank_store):
    fact = bank_store.add("one")
    bank = _add_bank(bank_store, "ns:alpha")
    result = namespaces.assign(bank_store, fact, "alpha")
    assert result["ok"] is True
    assert result["mechanism"] == "banks" and result["changed"] is True
    rows = bank_store.read("SELECT bank_id FROM fact_banks WHERE fact_id = ?", (fact,))
    assert [int(r["bank_id"]) for r in rows] == [bank]
    assert namespaces.stats(bank_store, "alpha")["facts"] == 1


def test_unknown_namespace_refused_when_create_is_false(store):
    store.add("seed", tags="ns:work")
    target = store.add("note")
    result = namespaces.assign(store, target, "typo")
    assert result["ok"] is False
    assert "unknown namespace" in result["reason"]
    assert store.tags(target) == ""


def test_unknown_bank_namespace_refused_when_create_is_false(bank_store):
    fact = bank_store.add("one")
    result = namespaces.assign(bank_store, fact, "ghost")
    assert result["ok"] is False and "unknown namespace" in result["reason"]
    assert bank_store.read("SELECT * FROM memory_banks") == []


def test_create_true_creates_namespace(store):
    target = store.add("note")
    result = namespaces.assign(store, target, "atlas", create=True)
    assert result["ok"] is True
    assert [r["namespace"] for r in namespaces.list_namespaces(store)] == ["atlas"]
    assert namespaces.stats(store)["namespaces"]["atlas"]["facts"] == 1


def test_create_true_creates_bank_row_and_membership(bank_store):
    fact = bank_store.add("one")
    result = namespaces.assign(bank_store, fact, "atlas", create=True)
    assert result["ok"] is True and result["created"] is True
    banks = [r["bank_name"] for r in bank_store.read("SELECT bank_name FROM memory_banks")]
    assert banks == ["ns:atlas"]
    assert namespaces.list_namespaces(bank_store)[0]["namespace"] == "atlas"


def test_bank_create_refuses_when_provider_needs_a_vector(tmp_path):
    """A NOT NULL HRR vector cannot be invented; report instead of fabricating."""
    s = _Store(tmp_path / "vec.db", _FACTS + _BANK_CACHE_ONLY
               + "CREATE TABLE fact_banks (fact_id INTEGER, bank_id INTEGER);")
    try:
        fact = s.add("one")
        result = namespaces.assign(s, fact, "atlas", create=True)
        assert result["ok"] is False
        assert "vector" in result["reason"]
        assert s.read("SELECT * FROM memory_banks") == []
    finally:
        s.close()


def test_assign_is_idempotent_tag(store):
    store.add("seed", tags="ns:work")
    target = store.add("note")
    first = namespaces.assign(store, target, "work")
    second = namespaces.assign(store, target, "work")
    assert first["changed"] is True and second["changed"] is False
    assert second["ok"] is True
    assert store.tags(target) == "ns:work"
    assert namespaces.stats(store, "work")["facts"] == 2


def test_assign_is_idempotent_bank(bank_store):
    fact = bank_store.add("one")
    _add_bank(bank_store, "ns:alpha")
    first = namespaces.assign(bank_store, fact, "alpha")
    second = namespaces.assign(bank_store, fact, "alpha")
    assert first["changed"] is True and second["changed"] is False
    assert len(bank_store.read("SELECT * FROM fact_banks")) == 1


@pytest.mark.parametrize("bad", ["", "   ", "has space", "comma,name", "a" * 65, "ns:"])
def test_assign_rejects_invalid_names(store, bad):
    target = store.add("note")
    result = namespaces.assign(store, target, bad, create=True)
    assert result["ok"] is False
    assert "invalid namespace" in result["reason"]
    assert store.tags(target) == ""


def test_assign_unknown_fact_is_refused(store):
    store.add("seed", tags="ns:work")
    result = namespaces.assign(store, 999, "work")
    assert result["ok"] is False and "not found" in result["reason"]


# ------------------------------------------------------------------ filtering
def test_filter_rows_handles_list_string_and_missing_tags():
    rows = [
        {"fact_id": 1, "tags": ["ns:work", "x"]},
        {"fact_id": 2, "tags": "x, ns:work"},
        {"fact_id": 3, "tags": "ns:home"},
        {"fact_id": 4},
        {"fact_id": 5, "tags": []},
        {"fact_id": 6, "tags": "xns:work, ns:workshop"},
    ]
    kept = namespaces.filter_rows(rows, "work")
    assert [r["fact_id"] for r in kept] == [1, 2]
    assert [r["fact_id"] for r in namespaces.filter_rows(rows, "ns:home")] == [3]
    assert namespaces.filter_rows(rows, "") == []
    assert namespaces.filter_rows([], "work") == []


# ------------------------------------------------------------------ scoping
def _seed_workspace(store):
    a = store.add("atlas backup runs at 02:00", tags="ns:atlas")
    b = store.add("atlas ssh port is 2222", tags="ns:atlas")
    c = store.add("unrelated homelab note", tags="ns:homelab")
    d = store.add("plain note")
    return a, b, c, d


def test_scope_query_postfilters_and_reports_dropped(store, monkeypatch):
    a, b, c, d = _seed_workspace(store)
    seen = {}

    def fake_prefetch(store_, message, *, limit=5, **kwargs):
        seen["limit"] = limit
        seen["message"] = message
        return [{"fact_id": a, "content": "a", "score": 0.9},
                {"fact_id": c, "content": "c", "score": 0.8},
                {"fact_id": b, "content": "b", "score": 0.7},
                {"fact_id": d, "content": "d", "score": 0.6}]

    monkeypatch.setattr(holo, "prefetch", fake_prefetch)
    out = namespaces.scope_query(store, "atlas backup", "atlas", limit=5)
    assert seen["limit"] == 24                     # 5 * 4 + 4
    assert seen["message"] == "atlas backup"
    assert [h["fact_id"] for h in out["hits"]] == [a, b]   # order preserved
    assert out["dropped"] == 2 and out["truncated"] == 0
    assert out["candidates"] == 4
    assert out["namespace"] == "atlas" and out["for_query"] == "atlas backup"
    assert out["mechanism"] == "tags" and out["available"] is True


def test_scope_query_truncates_to_limit_and_reports_it(store, monkeypatch):
    a, b, c, d = _seed_workspace(store)
    monkeypatch.setattr(holo, "prefetch", lambda *args, **kwargs: [
        {"fact_id": a}, {"fact_id": b}, {"fact_id": c}, {"fact_id": d}])
    out = namespaces.scope_query(store, "atlas", "atlas", limit=1)
    assert [h["fact_id"] for h in out["hits"]] == [a]
    assert out["dropped"] == 2 and out["truncated"] == 1


def test_scope_query_small_namespace_can_return_fewer_than_limit(store, monkeypatch):
    a, b, c, d = _seed_workspace(store)
    monkeypatch.setattr(holo, "prefetch", lambda *args, **kwargs: [
        {"fact_id": d}, {"fact_id": c}])
    out = namespaces.scope_query(store, "atlas", "atlas", limit=5)
    assert out["hits"] == [] and out["dropped"] == 2


def test_scope_query_bank_mechanism_uses_membership(bank_store, monkeypatch):
    inside = bank_store.add("inside")
    outside = bank_store.add("outside")
    bank = _add_bank(bank_store, "ns:alpha")
    _link(bank_store, inside, bank)
    monkeypatch.setattr(holo, "prefetch", lambda *args, **kwargs: [
        {"fact_id": inside}, {"fact_id": outside}])
    out = namespaces.scope_query(bank_store, "q", "alpha")
    assert out["mechanism"] == "banks"
    assert [h["fact_id"] for h in out["hits"]] == [inside]
    assert out["dropped"] == 1


def test_scope_query_reports_unavailable_channel(store, monkeypatch):
    monkeypatch.setattr(holo, "prefetch", lambda *args, **kwargs: None)
    out = namespaces.scope_query(store, "q", "atlas")
    assert out["available"] is False
    assert out["hits"] == [] and out["dropped"] == 0


# -------------------------------------------------------------------- stats
def test_stats_share_math(store):
    a, b, c, d = _seed_workspace(store)
    out = namespaces.stats(store)
    assert out["total_facts"] == 4
    assert out["count"] == 2
    assert out["namespaces"]["atlas"] == {
        "facts": 2, "share": 0.5, "last_seen": out["namespaces"]["atlas"]["last_seen"]}
    assert out["namespaces"]["homelab"]["share"] == 0.25
    assert sum(v["share"] for v in out["namespaces"].values()) == 0.75  # untagged rest


def test_stats_single_namespace(store):
    _seed_workspace(store)
    out = namespaces.stats(store, "atlas")
    assert (out["facts"], out["total_facts"], out["share"]) == (2, 4, 0.5)
    assert out["mechanism"] == "tags"
    missing = namespaces.stats(store, "nope")
    assert (missing["facts"], missing["share"]) == (0, 0.0)


def test_empty_store_is_tolerated(tmp_path, monkeypatch):
    monkeypatch.setattr(holo, "prefetch", lambda *args, **kwargs: [])
    s = _Store(tmp_path / "empty.db")
    try:
        info = namespaces.detect(s)
        assert info["tags"] is True and info["banks"] is False
        assert namespaces.list_namespaces(s) == []
        out = namespaces.stats(s)
        assert out == {"mechanism": "tags", "total_facts": 0, "count": 0, "namespaces": {}}
        assert namespaces.stats(s, "atlas")["share"] == 0.0
        assert namespaces.assign(s, 1, "atlas", create=True)["ok"] is False
        assert namespaces.scope_query(s, "q", "atlas")["hits"] == []
    finally:
        s.close()


def test_detect_on_path_and_missing_file(tmp_path, store):
    store.add("x", tags="ns:work")
    by_path = namespaces.detect(store.db_path)
    assert by_path["tags"] is True and by_path["columns"] == namespaces.detect(store)["columns"]
    missing = namespaces.detect(tmp_path / "absent.db")
    assert missing == {"banks": False, "tags": False, "categories": False, "columns": []}
    assert not (tmp_path / "absent.db").exists()   # detection never creates the file
