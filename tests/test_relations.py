"""Relation-graph tests: additive schema, link/unlink, traversal, inference.

Every case runs against a ``tmp_path`` database -- the factkit ``Store`` where
the real schema matters, and a provider-shaped stand-in for the private write
contract (``_write``/``_tx``/``read``).  No network, no live store.
"""
from __future__ import annotations

import contextlib
import sqlite3

import pytest

from factkit import holo, relations
from factkit.store import Store


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "facts.db")
    yield s
    s.close()


def seed(store, *contents: str) -> list[int]:
    return [store.add_fact(text, category="entity", tags="test")["fact_id"]
            for text in contents]


def table_names(store) -> set[str]:
    return {row[0] for row in store.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def index_names(store) -> set[str]:
    return {row[0] for row in store.db.execute(
        "SELECT name FROM sqlite_master WHERE type='index'")}


# ------------------------------------------------------------------- schema
def test_ensure_schema_is_idempotent(store):
    relations.ensure_schema(store)
    relations.ensure_schema(store)
    created = store.db.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='fact_relations'"
    ).fetchone()[0]
    assert created == 1
    assert {"idx_fact_relations_src", "idx_fact_relations_dst"} <= index_names(store)
    columns = [row[1] for row in store.db.execute("PRAGMA table_info(fact_relations)")]
    assert columns == ["id", "src_id", "dst_id", "rel", "weight", "created_at"]


def test_ensure_schema_leaves_facts_untouched(store):
    first, second = seed(store, "Atlas Server has 16 GB RAM",
                        "Atlas Server runs on port 8080")
    before = [tuple(row) for row in store.db.execute(
        "SELECT * FROM facts ORDER BY fact_id")]
    relations.ensure_schema(store)
    relations.link(store, first, second, rel="runs_on")
    after = [tuple(row) for row in store.db.execute(
        "SELECT * FROM facts ORDER BY fact_id")]
    assert after == before


# --------------------------------------------------------------- link/unlink
def test_link_and_unlink_roundtrip(store):
    a, b = seed(store, "Atlas Server has 16 GB RAM", "Atlas Server runs Caddy")
    edge = relations.link(store, a, b, rel="runs")
    assert edge["src_id"] == a and edge["dst_id"] == b
    assert edge["created"] is True and edge["weight"] == 1.0
    assert relations.unlink(store, a, b) == 1
    assert relations.neighbors(store, a, direction="both") == []


def test_unlink_can_target_one_label(store):
    a, b = seed(store, "Atlas Server has 16 GB RAM", "Atlas Server runs Caddy")
    relations.link(store, a, b, rel="runs")
    relations.link(store, a, b, rel="documents")
    assert relations.unlink(store, a, b, rel="runs") == 1
    left = relations.neighbors(store, a)
    assert [row["rel"] for row in left] == ["documents"]
    assert relations.unlink(store, a, b) == 1
    assert relations.unlink(store, a, b) == 0


def test_link_refuses_self_link(store):
    (a,) = seed(store, "Atlas Server has 16 GB RAM")
    with pytest.raises(ValueError, match="self-link"):
        relations.link(store, a, a, rel="same")
    assert relations.stats(store)["edges"] == 0


def test_link_refuses_unknown_fact(store):
    (a,) = seed(store, "Atlas Server has 16 GB RAM")
    with pytest.raises(ValueError, match="999 not found"):
        relations.link(store, a, 999, rel="runs")
    with pytest.raises(ValueError, match="999 not found"):
        relations.link(store, 999, a, rel="runs")
    assert relations.stats(store)["edges"] == 0


def test_link_refuses_empty_rel(store):
    a, b = seed(store, "Atlas Server has 16 GB RAM", "Atlas Server runs Caddy")
    for bad in ("", "   ", "---"):
        with pytest.raises(ValueError, match="rel must not be empty"):
            relations.link(store, a, b, rel=bad)


def test_rel_is_normalised_to_lowercase_snake_case(store):
    a, b = seed(store, "Atlas Server has 16 GB RAM", "Atlas Server runs Caddy")
    edge = relations.link(store, a, b, rel="  Part Of ")
    assert edge["rel"] == "part_of"
    assert store.db.execute(
        "SELECT rel FROM fact_relations WHERE id = ?", (edge["id"],)).fetchone()[0] == "part_of"
    relations.link(store, a, b, rel="PART-OF")   # same normalised label: update
    assert relations.stats(store)["edges"] == 1
    assert relations.unlink(store, a, b, rel="Part_of") == 1


def test_duplicate_link_updates_weight_and_keeps_created_at(store):
    a, b = seed(store, "Atlas Server has 16 GB RAM", "Atlas Server runs Caddy")
    first = relations.link(store, a, b, rel="runs", weight=1.0)
    second = relations.link(store, a, b, rel="runs", weight=2.5)
    assert second["created"] is False
    assert second["weight"] == 2.5
    assert second["created_at"] == first["created_at"]
    assert second["id"] == first["id"]
    assert store.db.execute("SELECT COUNT(*) FROM fact_relations").fetchone()[0] == 1


def test_link_persists_across_reopen(tmp_path):
    """The edge must be committed, not left in the writer's open transaction."""
    path = tmp_path / "persist.db"
    s = Store(path)
    a, b = seed(s, "Atlas Server has 16 GB RAM", "Atlas Server runs Caddy")
    relations.link(s, a, b, rel="runs")
    s.close()

    reopened = Store(path)
    try:
        assert [row["fact_id"] for row in relations.neighbors(reopened, a)] == [b]
    finally:
        reopened.close()
    assert relations.stats(holo.ReadOnlyStore(path))["edges"] == 1


def test_link_uses_the_provider_write_contract(tmp_path):
    class PluginStore:
        def __init__(self, path):
            self.db_path = str(path)
            self.conn = sqlite3.connect(self.db_path, isolation_level=None,
                                        check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.executescript(
                "CREATE TABLE facts(fact_id INTEGER PRIMARY KEY, content TEXT NOT NULL);"
                "INSERT INTO facts(content) VALUES ('Atlas Server has 16 GB RAM');"
                "INSERT INTO facts(content) VALUES ('Atlas Server runs Caddy');")

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
    edge = relations.link(s, 1, 2, rel="Runs On", weight=0.5)
    assert edge["rel"] == "runs_on" and edge["weight"] == 0.5
    row = s.conn.execute("SELECT src_id, dst_id, rel FROM fact_relations").fetchone()
    assert tuple(row) == (1, 2, "runs_on")


# ---------------------------------------------------------------- neighbours
def test_neighbors_out_and_in(store):
    a, b, c = seed(store, "A fact", "B fact", "C fact")
    relations.link(store, a, b, rel="next")
    relations.link(store, a, c, rel="other")
    out = relations.neighbors(store, a)
    assert {row["fact_id"] for row in out} == {b, c}
    assert {row["direction"] for row in out} == {"out"}
    incoming = relations.neighbors(store, b, direction="in")
    assert [(row["fact_id"], row["rel"], row["direction"]) for row in incoming] == \
        [(a, "next", "in")]


def test_neighbors_both_tags_each_side(store):
    a, b, c = seed(store, "A fact", "B fact", "C fact")
    relations.link(store, b, a, rel="parent")
    relations.link(store, a, c, rel="child")
    both = relations.neighbors(store, a, direction="both")
    assert {(row["fact_id"], row["direction"]) for row in both} == \
        {(b, "in"), (c, "out")}


def test_neighbors_rel_filter(store):
    a, b, c = seed(store, "A fact", "B fact", "C fact")
    relations.link(store, a, b, rel="next")
    relations.link(store, a, c, rel="other")
    rows = relations.neighbors(store, a, rel="NEXT")
    assert [(row["fact_id"], row["rel"]) for row in rows] == [(b, "next")]


def test_neighbors_order_by_weight_then_fact_id(store):
    a, b, c, d = seed(store, "A fact", "B fact", "C fact", "D fact")
    relations.link(store, a, b, rel="next", weight=1.0)
    relations.link(store, a, c, rel="next", weight=5.0)
    relations.link(store, a, d, rel="next", weight=1.0)
    rows = relations.neighbors(store, a)
    assert [row["fact_id"] for row in rows] == [c, b, d]
    assert [row["weight"] for row in rows] == [5.0, 1.0, 1.0]


def test_neighbors_limit_and_truncated_content(store):
    long_text = "Atlas " + ("detail " * 60)          # ~400 chars
    a, b = seed(store, "A fact", long_text)
    c = seed(store, "C fact")[0]
    relations.link(store, a, b, rel="next", weight=2.0)
    relations.link(store, a, c, rel="next", weight=1.0)
    rows = relations.neighbors(store, a, limit=1)
    assert len(rows) == 1 and rows[0]["fact_id"] == b
    rows = relations.neighbors(store, a)
    assert rows[0]["fact_id"] == b
    assert len(rows[0]["content"]) == relations.CONTENT_CHARS
    assert rows[0]["content"] == long_text[:200]


def test_neighbors_rejects_unknown_direction(store):
    (a,) = seed(store, "A fact")
    with pytest.raises(ValueError, match="direction"):
        relations.neighbors(store, a, direction="sideways")


def test_neighbors_empty_store(store):
    (a,) = seed(store, "A fact")
    assert relations.neighbors(store, a) == []
    assert relations.neighbors(store, a, direction="both") == []


# ---------------------------------------------------------------------- path
def test_path_direct_link(store):
    a, b = seed(store, "A fact", "B fact")
    relations.link(store, a, b, rel="next")
    assert relations.path(store, a, b) == [
        {"fact_id": a, "via_rel": None},
        {"fact_id": b, "via_rel": "next"},
    ]


def test_path_two_hops(store):
    a, b, c = seed(store, "A fact", "B fact", "C fact")
    relations.link(store, a, b, rel="next")
    relations.link(store, b, c, rel="then")
    hops = relations.path(store, a, c)
    assert [step["fact_id"] for step in hops] == [a, b, c]
    assert [step["via_rel"] for step in hops] == [None, "next", "then"]


def test_path_none_when_unreachable(store):
    a, b, c = seed(store, "A fact", "B fact", "C fact")
    relations.link(store, a, b, rel="next")
    assert relations.path(store, b, a) is None      # directed, not symmetric
    assert relations.path(store, a, c) is None
    assert relations.path(store, a, 999) is None


def test_path_respects_max_depth(store):
    a, b, c, d = seed(store, "A fact", "B fact", "C fact", "D fact")
    relations.link(store, a, b, rel="next")
    relations.link(store, b, c, rel="next")
    relations.link(store, c, d, rel="next")
    assert relations.path(store, a, c, max_depth=1) is None
    assert [s["fact_id"] for s in relations.path(store, a, c, max_depth=2)] == [a, b, c]
    assert relations.path(store, a, d, max_depth=3) is not None
    assert relations.path(store, a, d, max_depth=2) is None


def test_path_rel_filter(store):
    a, b, c = seed(store, "A fact", "B fact", "C fact")
    relations.link(store, a, b, rel="next")
    relations.link(store, b, c, rel="next")
    relations.link(store, a, c, rel="shortcut")
    assert [s["fact_id"] for s in relations.path(store, a, c, rel="shortcut")] == [a, c]
    assert [s["fact_id"] for s in relations.path(store, a, c, rel="next")] == [a, b, c]


def test_path_is_cycle_safe(store):
    a, b, c = seed(store, "A fact", "B fact", "C fact")
    relations.link(store, a, b, rel="next")
    relations.link(store, b, c, rel="next")
    relations.link(store, c, a, rel="back")
    hops = relations.path(store, a, c)
    assert [s["fact_id"] for s in hops] == [a, b, c]
    assert relations.path(store, a, a) == [{"fact_id": a, "via_rel": None}]


# --------------------------------------------------------------------- infer
def test_infer_proposes_attribute_edges_and_writes_nothing(store):
    a, b, unrelated = seed(
        store,
        "Atlas Server has 16 GB RAM",
        "Atlas Server has 16 GB RAM and 8 cores",
        "Beacon Mini runs Caddy 2.7.6",
    )
    before = [tuple(row) for row in store.db.execute("SELECT * FROM facts ORDER BY fact_id")]
    proposals = relations.infer(store)
    after = [tuple(row) for row in store.db.execute("SELECT * FROM facts ORDER BY fact_id")]

    assert proposals == [p for p in proposals if {p["src_id"], p["dst_id"]} == {a, b}]
    assert len(proposals) == 1
    p = proposals[0]
    assert (p["src_id"], p["dst_id"]) == (a, b)
    assert p["rel"] == "shares_ram"
    assert "ram=16gb" in p["reason"]
    assert 0.0 < p["confidence"] <= 1.0
    assert after == before
    assert "fact_relations" not in table_names(store)     # no implicit write


def test_infer_skips_existing_edges(store):
    a, b = seed(store, "Atlas Server has 16 GB RAM",
                "Atlas Server has 16 GB RAM and 8 cores")
    relations.link(store, a, b, rel="shares_ram")
    assert relations.infer(store) == []


def test_infer_is_empty_without_attribute_pairs(store):
    seed(store, "Atlas Server has 16 GB RAM", "Beacon Mini runs Caddy 2.7.6")
    assert relations.infer(store) == []


# --------------------------------------------------------------------- stats
def test_stats_counts_edges_rels_orphans_and_degree(store):
    a, b, c, orphan = seed(store, "A fact", "B fact", "C fact", "D fact")
    relations.link(store, a, b, rel="part_of")
    relations.link(store, b, c, rel="runs_on")
    relations.link(store, a, c, rel="part_of")
    s = relations.stats(store)
    assert s["edges"] == 3
    assert s["distinct_rels"] == s["rels"] == 2
    assert s["orphan_facts"] == s["orphans"] == 1
    assert s["max_degree"] == 2


def test_stats_empty_store(store):
    s = relations.stats(store)
    assert s == {"edges": 0, "rels": 0, "distinct_rels": 0,
                 "orphan_facts": 0, "orphans": 0, "max_degree": 0}


def test_stats_tolerates_a_missing_table(store, tmp_path):
    seed(store, "A fact", "B fact")
    assert relations.stats(store)["edges"] == 0
    assert relations.stats(store)["orphan_facts"] == 2   # facts exist, graph does not

    bare = Store(tmp_path / "bare.db", init=False)
    try:
        relations.ensure_schema(bare)
        bare.db.execute(
            "INSERT INTO fact_relations(src_id, dst_id, rel, weight) VALUES (1, 2, 'next', 1.0)")
        bare.db.commit()
        s = relations.stats(bare)
        assert s["edges"] == 1 and s["distinct_rels"] == 1
        assert s["orphan_facts"] == 0 and s["max_degree"] == 1
        assert "facts" not in table_names(bare)
    finally:
        bare.close()
