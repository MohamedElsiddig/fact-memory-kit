"""Offline tests for entity-aware ranking.

No network and no real model: every vector is canned, so a cosine is whatever
the fixture says it is. All names and addresses are synthetic (Atlas Server,
Beacon Mini, 192.0.2.x).
"""
from __future__ import annotations

import math
import sqlite3

import pytest

from factkit import holo, vectors

MODEL = "text-embedding-nomic-embed-text-v1.5"
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
        key = " ".join((text or "").lower().split())
        if not key:
            return [0.0, 0.0, 0.0, 0.0]
        return list(self.table[key])

    def embed(self, texts):
        return [self.embed_one(t) for t in texts]


class _Store:
    db_path = ""

    def __init__(self, path):
        self.db_path = str(path)

    def read(self, sql, params=()):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            # Like the plugin's MemoryStore: sqlite3.Row, not dicts.
            return list(conn.execute(sql, params))
        finally:
            conn.close()


def _make_store(tmp_path, facts, *, entities_column=False, entity_rows=()):
    """facts: (content, cosine[, entities_cell]) tuples."""
    db = tmp_path / "scoring.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    extra = ", entities TEXT" if entities_column else ""
    conn = sqlite3.connect(db)
    conn.executescript(f"""
    CREATE TABLE facts(fact_id INTEGER PRIMARY KEY, content TEXT UNIQUE, category TEXT,
                       tags TEXT, trust_score REAL, created_at TEXT, updated_at TEXT{extra});
    CREATE TABLE entities(entity_id INTEGER PRIMARY KEY, name TEXT,
                          entity_type TEXT DEFAULT 'unknown');
    CREATE TABLE fact_entities(fact_id INTEGER, entity_id INTEGER);
    CREATE TABLE fact_embeddings(fact_id INTEGER, model TEXT, dims INTEGER, vec BLOB);
    """)
    table = {}
    for fact_id, row in enumerate(facts, 1):
        content, cosine = row[0], row[1]
        cell = row[2] if len(row) > 2 else None
        if entities_column:
            conn.execute(
                "INSERT INTO facts(fact_id,content,category,tags,trust_score,"
                "created_at,updated_at,entities) VALUES (?,?,?,?,?,?,?,?)",
                (fact_id, content, "entity", "", 0.9,
                 "2026-01-01 00:00:00", "2026-01-01 00:00:00", cell))
        else:
            conn.execute(
                "INSERT INTO facts(fact_id,content,category,tags,trust_score,"
                "created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                (fact_id, content, "entity", "", 0.9,
                 "2026-01-01 00:00:00", "2026-01-01 00:00:00"))
        vec = _at(cosine)
        conn.execute("INSERT INTO fact_embeddings VALUES (?,?,?,?)",
                     (fact_id, MODEL, len(vec), vectors.pack(vec)))
        table[content] = vec
    for name in entity_rows:
        conn.execute("INSERT INTO entities(name) VALUES (?)", (name,))
    conn.commit()
    conn.close()
    return _Store(db), table


def _install(monkeypatch, table, query):
    import factkit.config as config_mod
    import factkit.embed as embed_mod

    table.setdefault(query, QUERY_VEC)

    class _Cfg:
        embed_model = MODEL

    monkeypatch.setattr(config_mod.Config, "load", lambda db_path=None: _Cfg())
    monkeypatch.setattr(embed_mod, "from_config", lambda cfg: _MapClient(table))


def _prefetch(monkeypatch, tmp_path, facts, query, *, entities_column=False,
              entity_rows=(), **kwargs):
    store, table = _make_store(tmp_path, facts, entities_column=entities_column,
                               entity_rows=entity_rows)
    _install(monkeypatch, table, query)
    return holo.prefetch(store, query, **kwargs)


def _ids(hits):
    return [hit["fact_id"] for hit in hits]


# ------------------------------------------------------------- paraphrase
def test_memory_query_picks_ram_fact_over_higher_cosine(monkeypatch, tmp_path):
    hits = _prefetch(monkeypatch, tmp_path, [
        ("Atlas Server backup job runs at 02:00 daily", 0.532),
        ("Beacon Mini has 1 GB RAM", 0.515),
    ], "how much memory does the small computer have")
    # Raw cosine favours the unrelated fact; the memory->RAM attribute bonus flips it.
    assert _ids(hits)[0] == 2
    assert hits[0]["boost"] == holo.ATTRIBUTE_BOOST


def test_paraphrase_puts_small_machine_in_top_three(monkeypatch, tmp_path):
    hits = _prefetch(monkeypatch, tmp_path, [
        ("Atlas Server runs Debian bookworm", 0.568),
        ("Backup job runs at 02:00 daily", 0.552),
        ("Dashboard listens behind the reverse proxy", 0.537),
        ("Wi-Fi network is called homelab", 0.529),
        ("Beacon Mini has 1 GB RAM", 0.521),   # 5th on raw cosine
        ("Printer queue is shared", 0.505),
    ], "how much memory does the small computer have", limit=6)
    ranked = _ids(hits)
    assert len(ranked) == 6
    assert ranked.index(5) <= 2, ranked


# -------------------------------------------------------------- entities
def test_entity_token_boosts_named_entity_fact(monkeypatch, tmp_path):
    hits = _prefetch(monkeypatch, tmp_path, [
        ("Beacon Mini has 1 GB RAM", 0.50),
        ("Atlas Server has 16 GB RAM", 0.50),
    ], "Beacon Mini specs")
    assert _ids(hits)[0] == 1
    assert hits[0]["boost"] == holo.ENTITY_BOOST
    assert hits[1].get("boost", 0.0) == 0.0


def test_entities_column_names_match_lowercase_query(monkeypatch, tmp_path):
    hits = _prefetch(monkeypatch, tmp_path, [
        ("Node Alpha has 1 GB RAM", 0.50, "Beacon Mini"),
        ("Node Beta has 16 GB RAM", 0.50, "Atlas Server"),
    ], "beacon mini ram", entities_column=True)
    assert _ids(hits)[0] == 1
    # entity (0.04) + ram attribute (0.02)
    assert hits[0]["boost"] == pytest.approx(
        holo.ENTITY_BOOST + holo.ATTRIBUTE_BOOST)


# ------------------------------------------------------------ attributes
def test_attribute_word_boosts_without_any_entity(monkeypatch, tmp_path):
    hits = _prefetch(monkeypatch, tmp_path, [
        ("Node Alpha has a 512 GB SSD", 0.52),
        ("Backup job runs at 02:00 daily", 0.53),
    ], "which machine has more storage")
    assert _ids(hits)[0] == 1
    assert hits[0]["boost"] == holo.ATTRIBUTE_BOOST
    assert hits[1].get("boost", 0.0) == 0.0


def test_port_query_picks_port_fact(monkeypatch, tmp_path):
    hits = _prefetch(monkeypatch, tmp_path, [
        ("Dashboard listens on port 9119", 0.52),
        ("Backup job runs at 02:00 daily", 0.53),
    ], "which port does the service listen on")
    assert _ids(hits)[0] == 1
    assert hits[0]["boost"] == holo.ATTRIBUTE_BOOST


def test_boost_is_capped(monkeypatch, tmp_path):
    boost = holo.entity_boost(
        "Beacon Mini and Atlas Server memory ram storage cpu port ip version",
        "Beacon Mini has 16 GB RAM, 512 GB storage, 8 cores on port 8080",
        "Beacon Mini, Atlas Server")
    assert holo.ENTITY_BOOST == 0.04
    assert holo.ATTRIBUTE_BOOST == 0.02
    assert holo.MAX_ENTITY_BOOST == 0.08
    assert boost == holo.MAX_ENTITY_BOOST


def test_no_entity_tokens_leaves_score_unchanged(monkeypatch, tmp_path):
    assert holo.entity_boost("what is the schedule", "Beacon Mini has 1 GB RAM") == 0.0
    hits = _prefetch(monkeypatch, tmp_path, [
        ("Atlas Server backup job runs at 02:00 daily", 0.42),
    ], "what is the schedule", min_score=0.0)
    assert hits[0]["score"] == 0.42
    assert "boost" not in hits[0]


def test_empty_query_does_not_crash(monkeypatch, tmp_path):
    assert holo.entity_boost("", "Beacon Mini has 1 GB RAM") == 0.0
    hits = _prefetch(monkeypatch, tmp_path, [
        ("Beacon Mini has 1 GB RAM", 0.50),
    ], "", min_score=0.0)
    assert hits == []


# ----------------------------------------------------------- determinism
def test_scores_are_deterministic_across_runs(monkeypatch, tmp_path):
    facts = [
        ("Atlas Server runs Debian bookworm", 0.532),
        ("Beacon Mini has 1 GB RAM", 0.515),
    ]
    first = _prefetch(monkeypatch, tmp_path, facts,
                      "how much memory does the Beacon Mini have")
    second = _prefetch(monkeypatch, tmp_path / "again", facts,
                       "how much memory does the Beacon Mini have")
    assert [(h["fact_id"], h["score"]) for h in first] == \
           [(h["fact_id"], h["score"]) for h in second]


# ------------------------------------------------------- fuse contract
def test_provider_hits_still_outrank_semantic_reserve(monkeypatch, tmp_path):
    store, table = _make_store(tmp_path, [
        ("Beacon Mini has 1 GB RAM", 0.60),
        ("Atlas Server has 16 GB RAM", 0.55),
        ("Printer queue is shared", 0.40),
    ])
    _install(monkeypatch, table, "Beacon Mini memory")
    provider = [{"fact_id": 3, "content": "Printer queue is shared",
                 "category": "tool", "tags": "", "trust_score": 0.9}]
    out = holo.fuse_results(store, "Beacon Mini memory", provider, limit=3)
    assert out and out[0]["fact_id"] == 3           # provider hit always first
    assert out[0].get("source") != "semantic"
