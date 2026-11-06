"""Multi-hop retrieval tests: BFS shape, decay, cycles, and the query merge.

No network, no live store and no dependency on the real relations module: the
graph is a canned ``factkit.relations`` stand-in installed in ``sys.modules``
and ``holo.prefetch`` is monkeypatched.  All ids and texts are synthetic.
"""
from __future__ import annotations

import sys
import types

import pytest

from factkit import holo, multihop


class _Store:
    """Opaque handle: the fake relations module never reads it."""

    db_path = ""

    def read(self, sql, params=()):  # pragma: no cover - must not be called
        raise AssertionError(f"unexpected store read: {sql}")


STORE = _Store()


def fake_relations(edges, contents=None):
    """A stand-in for factkit.relations; ``edges`` are (src, dst, rel, weight)."""
    mod = types.ModuleType("factkit.relations")
    mod.calls = []

    def neighbors(store, fact_id, rel=None, direction="out", limit=50):
        mod.calls.append((fact_id, rel, direction, limit))
        rows = []
        for src, dst, name, weight in edges:
            if direction == "out" and src != fact_id:
                continue
            if direction == "in" and dst != fact_id:
                continue
            if rel is not None and name != rel:
                continue
            other = dst if src == fact_id else src
            rows.append({"fact_id": other, "rel": name, "weight": weight,
                         "direction": direction,
                         "content": (contents or {}).get(other, f"fact {other}")})
        rows.sort(key=lambda row: (-row["weight"], row["fact_id"]))
        return rows[:limit] if limit is not None and limit >= 0 else rows

    mod.neighbors = neighbors
    mod.ensure_schema = lambda store: None
    return mod


@pytest.fixture()
def graph(monkeypatch):
    """Install a fake graph: ``graph(edges, contents=None) -> module``."""
    def install(edges, contents=None):
        mod = fake_relations(edges, contents)
        monkeypatch.setitem(sys.modules, "factkit.relations", mod)
        return mod
    return install


def _prefetch(monkeypatch, hits):
    """Patch holo.prefetch with canned hits; a list is returned as-is."""
    seen: dict = {}

    def fake(store, message, *, limit=5, **kwargs):
        seen.update({"message": message, "limit": limit, "kwargs": kwargs})
        if hits is None:
            return None
        return [dict(hit) for hit in hits][:limit]

    monkeypatch.setattr(holo, "prefetch", fake)
    return seen


def _ids(rows):
    return [row["fact_id"] for row in rows]


# ------------------------------------------------------------------ BFS shape
def test_depth_one_returns_only_direct_neighbours(graph):
    graph([(1, 2, "uses", 2.0), (1, 3, "uses", 1.0), (2, 4, "uses", 1.0)])
    out = multihop.walk(STORE, [1], depth=1)
    assert out["seeds"] == [1]
    assert _ids(out["hops"]) == [2, 3]
    assert {hop["depth"] for hop in out["hops"]} == {1}
    assert out["visited"] == [1, 2, 3]
    assert out["degraded"] is False


def test_depth_two_reaches_the_second_hop(graph):
    graph([(1, 2, "uses", 1.0), (2, 3, "depends_on", 1.0), (2, 4, "uses", 1.0)])
    out = multihop.walk(STORE, [1], depth=2)
    assert _ids(out["hops"]) == [2, 3, 4]
    assert [hop["depth"] for hop in out["hops"]] == [1, 2, 2]
    assert out["visited"] == [1, 2, 3, 4]


def test_hop_carries_relation_parent_and_content(graph):
    graph([(1, 2, "depends_on", 1.0)],
          contents={2: "Proxy sits in front of the dashboard"})
    hop = multihop.walk(STORE, [1], depth=1)["hops"][0]
    assert hop == {"fact_id": 2, "via": "depends_on", "from_id": 1, "depth": 1,
                   "content": "Proxy sits in front of the dashboard", "score": 0.6}


def test_walk_accepts_prefetch_hits_as_seeds(graph):
    graph([(1, 2, "uses", 1.0)])
    out = multihop.walk(STORE, [{"fact_id": 1, "score": 0.9}], depth=1)
    assert out["seeds"] == [1]
    assert _ids(out["hops"]) == [2]


# ------------------------------------------------------------- caps & budget
def test_fanout_caps_children_per_node(graph):
    edges = [(1, node, "uses", float(10 - node)) for node in (2, 3, 4, 5)]
    edges += [(2, 6, "uses", 3.0), (2, 7, "uses", 2.0), (2, 8, "uses", 1.0)]
    graph(edges)
    out = multihop.walk(STORE, [1], depth=2, fanout=2)
    depth1 = [hop["fact_id"] for hop in out["hops"] if hop["depth"] == 1]
    depth2 = [hop["fact_id"] for hop in out["hops"] if hop["depth"] == 2]
    assert depth1 == [2, 3]      # heaviest two children of node 1
    assert depth2 == [6, 7]      # fanout is per node, not global


def test_budget_caps_total_expansion(graph):
    graph([(seed, 10 + seed, "uses", 1.0) for seed in range(1, 6)])
    out = multihop.walk(STORE, [1, 2, 3, 4, 5], depth=2, fanout=5, budget=4)
    assert len(out["hops"]) == 4
    assert len(out["visited"]) == 9          # 5 seeds + 4 expanded nodes
    assert _ids(out["hops"]) == [11, 12, 13, 14]


def test_budget_zero_returns_seeds_only(graph):
    graph([(1, 2, "uses", 1.0)])
    out = multihop.walk(STORE, [1], depth=2, budget=0)
    assert out["hops"] == []
    assert out["visited"] == [1]
    assert out["degraded"] is False


# --------------------------------------------------------- cycles & filters
def test_cycle_is_visited_once(graph):
    graph([(1, 2, "uses", 1.0), (2, 1, "uses", 1.0), (2, 3, "uses", 1.0)])
    out = multihop.walk(STORE, [1], depth=3)
    assert _ids(out["hops"]) == [2, 3]
    assert out["visited"].count(1) == 1
    assert out["visited"].count(2) == 1


def test_self_loop_does_not_duplicate_a_seed(graph):
    graph([(1, 1, "same_as", 1.0), (1, 2, "uses", 1.0)])
    out = multihop.walk(STORE, [1], depth=2)
    assert _ids(out["hops"]) == [2]


def test_rel_filter_limits_the_walk(graph):
    graph([(1, 2, "uses", 2.0), (1, 3, "owns", 3.0), (2, 4, "uses", 1.0)])
    out = multihop.walk(STORE, [1], depth=2, rel="uses")
    assert _ids(out["hops"]) == [2, 4]      # the heavier "owns" edge is filtered out
    assert 3 not in _ids(out["hops"])


def test_incoming_edges_answer_what_depends_on_x(graph):
    # edges point service -> proxy; "what uses the proxy" is the inbound side
    graph([(7, 1, "uses", 2.0), (8, 1, "uses", 1.0)])
    out = multihop.walk(STORE, [1], depth=1)
    assert _ids(out["hops"]) == [7, 8]
    assert multihop.walk(STORE, [1], depth=1, direction="out")["hops"] == []


def test_unknown_direction_is_rejected(graph):
    graph([])
    with pytest.raises(ValueError):
        multihop.walk(STORE, [1], direction="sideways")


# -------------------------------------------------------------------- decay
def test_scores_decay_by_sixty_percent_per_hop(graph):
    graph([(1, 2, "uses", 1.0), (2, 3, "uses", 1.0)])
    out = multihop.walk(STORE, [1], depth=2)
    assert multihop.DECAY == 0.6
    assert [hop["score"] for hop in out["hops"]] == [0.6, 0.36]


# ------------------------------------------------------------- degradation
def test_missing_relations_module_degrades_without_raising(monkeypatch):
    monkeypatch.setitem(sys.modules, "factkit.relations", None)
    out = multihop.walk(STORE, [1, 2], depth=2)
    assert out == {"seeds": [1, 2], "hops": [], "visited": [1, 2], "degraded": True}


def test_walk_with_no_seeds_reads_nothing(graph):
    mod = graph([(1, 2, "uses", 1.0)])
    out = multihop.walk(STORE, [], depth=2)
    assert out == {"seeds": [], "hops": [], "visited": [], "degraded": False}
    assert mod.calls == []


# ---------------------------------------------------------------- merge
def test_expand_query_merges_hops_below_direct_hits(graph, monkeypatch):
    graph([(1, 3, "uses", 1.0), (3, 4, "depends_on", 1.0)],
          contents={4: "Caddy fronts the dashboard"})
    _prefetch(monkeypatch, [
        {"fact_id": 1, "content": "Proxy config", "score": 0.8, "trust": 0.9},
        {"fact_id": 2, "content": "Unrelated", "score": 0.5, "trust": 0.9},
    ])
    out = multihop.expand_query(STORE, "what uses the proxy", depth=2, limit=5)
    assert out["for_query"] == "what uses the proxy"
    assert _ids(out["hits"]) == [1, 2, 3, 4]
    assert [hit["score"] for hit in out["hits"]] == [0.8, 0.5, 0.48, 0.288]
    assert out["hits"][0]["source"] == "direct" and out["hits"][0]["depth"] == 0
    assert out["hits"][3]["source"] == "hop" and out["hits"][3]["depth"] == 2
    assert out["hits"][3]["content"] == "Caddy fronts the dashboard"
    assert len(out["direct"]) == 2 and _ids(out["hops"]) == [3, 4]
    assert out["degraded"] is False


def test_expand_query_hop_score_uses_its_own_seed(graph, monkeypatch):
    graph([(1, 2, "uses", 1.0), (3, 4, "uses", 1.0)])
    _prefetch(monkeypatch, [
        {"fact_id": 1, "content": "Strong seed", "score": 0.9},
        {"fact_id": 3, "content": "Weak seed", "score": 0.5},
    ])
    out = multihop.expand_query(STORE, "q", depth=1, limit=5)
    assert {hit["fact_id"]: hit["score"] for hit in out["hops"]} == {
        2: 0.54, 4: 0.3}   # 0.9 * 0.6, 0.5 * 0.6


def test_expand_query_dedupe_prefers_the_direct_hit(graph, monkeypatch):
    graph([])
    _prefetch(monkeypatch, [{"fact_id": 1, "content": "Proxy config", "score": 0.9}])
    monkeypatch.setattr(multihop, "walk", lambda *args, **kwargs: {
        "seeds": [1], "degraded": False, "visited": [1, 9],
        "hops": [
            {"fact_id": 9, "via": "uses", "from_id": 1, "depth": 1,
             "content": "Neighbour", "score": 0.6},
            {"fact_id": 1, "via": "uses", "from_id": 9, "depth": 1,
             "content": "Proxy config", "score": 0.6},
        ]})
    out = multihop.expand_query(STORE, "proxy", limit=5)
    assert _ids(out["hits"]) == [1, 9]
    assert out["hits"][0]["score"] == 0.9
    assert out["hits"][0]["source"] == "direct"
    assert _ids(out["hops"]) == [9]


def test_direct_hits_are_not_duplicated_when_reachable_by_an_edge(graph, monkeypatch):
    graph([(1, 2, "mentions", 1.0), (2, 3, "uses", 1.0)])
    _prefetch(monkeypatch, [
        {"fact_id": 1, "content": "A", "score": 0.7},
        {"fact_id": 2, "content": "B", "score": 0.6},
    ])
    out = multihop.expand_query(STORE, "q", depth=2)
    assert _ids(out["hits"]) == [1, 2, 3]
    assert out["hits"][1]["source"] == "direct"


def test_expand_query_respects_limit(graph, monkeypatch):
    graph([(1, 2, "uses", 1.0), (2, 3, "uses", 1.0)])
    _prefetch(monkeypatch, [{"fact_id": 1, "content": "A", "score": 0.9}])
    out = multihop.expand_query(STORE, "q", limit=2, depth=2)
    assert _ids(out["hits"]) == [1, 2]
    # the uncapped candidate lists stay available for inspection
    assert len(out["direct"]) == 1 and len(out["hops"]) == 2


def test_expand_query_breaks_ties_by_fact_id(graph, monkeypatch):
    graph([(1, 4, "uses", 1.0), (1, 3, "uses", 1.0), (1, 2, "uses", 1.0)])
    _prefetch(monkeypatch, [{"fact_id": 1, "content": "Seed", "score": 0.5}])
    out = multihop.expand_query(STORE, "q", limit=5, depth=1)
    assert _ids(out["hits"]) == [1, 2, 3, 4]
    assert {hop["score"] for hop in out["hops"]} == {0.3}   # equal: id decides


def test_expand_query_without_seeds_returns_empty(graph, monkeypatch):
    graph([(1, 2, "uses", 1.0)])
    _prefetch(monkeypatch, [])
    out = multihop.expand_query(STORE, "q")
    assert out["direct"] == [] and out["hops"] == [] and out["hits"] == []
    assert out["degraded"] is False


def test_expand_query_degrades_when_embeddings_are_unavailable(graph, monkeypatch):
    graph([(1, 2, "uses", 1.0)])
    _prefetch(monkeypatch, None)
    out = multihop.expand_query(STORE, "q")
    assert out["degraded"] is True
    assert out["hits"] == []


def test_expand_query_keeps_direct_hits_without_relations(monkeypatch):
    monkeypatch.setitem(sys.modules, "factkit.relations", None)
    _prefetch(monkeypatch, [{"fact_id": 5, "content": "Dashboard on port 9119",
                             "score": 0.8}])
    out = multihop.expand_query(STORE, "which port")
    assert out["degraded"] is True
    assert _ids(out["hits"]) == [5]
    assert out["hits"][0]["score"] == 0.8


def test_prefetch_knobs_go_to_prefetch_and_rel_to_the_walk(graph, monkeypatch):
    graph([(1, 2, "uses", 1.0), (1, 3, "owns", 3.0)])
    seen = _prefetch(monkeypatch, [{"fact_id": 1, "content": "A", "score": 0.9}])
    out = multihop.expand_query(STORE, "q", limit=4, rel="uses", min_score=0.2)
    assert seen["kwargs"] == {"min_score": 0.2}
    assert seen["limit"] == 4
    assert _ids(out["hits"]) == [1, 2]
