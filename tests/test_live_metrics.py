"""Offline tests for live-store retrieval metrics.

No network and no live Hermes database: a deterministic hash embedder and a
``tmp_path`` store stand in. The store is seeded with six synthetic facts, each
with an embedding, so every test exercises the same retrieval path the live
memory store uses.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3

import pytest

from factkit import live_metrics
from factkit.store import Store

FACTS = [
    "Atlas Server has 16 GB RAM",
    "Atlas Server listens on port 8010",
    "Beacon Mini has 1 GB RAM",
    "Backup runs at 02:00 daily",
    "Mazda 3 fuel tank is 50 litres",
    "Transcriber uses the whisper model",
]

CASES = [
    {"query": "atlas server ram", "expect": [1]},
    {"query": "atlas server port 8010", "expect": [2]},
    {"query": "beacon mini ram", "expect": [3]},
    {"query": "backup daily runs", "expect": [4]},
    {"query": "mazda fuel tank", "expect": [5]},
    {"query": "transcriber whisper", "expect": [6]},
]


class FakeEmbedder:
    """Hash-based bag of words -> fixed dims, deterministic and offline."""

    model = "fake-embed"
    dims = 64

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for token in (text or "").lower().split():
            h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            vec[h % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_one(self, text: str) -> list[float]:
        return self._vec(text)

    def embed(self, texts):
        return [self._vec(t) for t in texts]


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "facts.db")
    yield s
    s.close()


@pytest.fixture()
def seeded(store):
    """The store plus a FakeEmbedder, with all six facts indexed."""
    embedder = FakeEmbedder()
    ids = []
    for content in FACTS:
        fact = store.add_fact(content, trust_score=0.9)
        store.put_embedding(fact["fact_id"], embedder.model,
                            embedder.embed_one(content))
        ids.append(fact["fact_id"])
    store.db.commit()
    assert ids == [1, 2, 3, 4, 5, 6]
    return store, embedder


def report(store, embedder, cases, k=5):
    return live_metrics.evaluate(store, cases, k=k, embedder=embedder)


# ------------------------------------------------------------------ evaluate
def test_perfect_run_has_full_hit_rate_and_mrr(seeded):
    store, embedder = seeded
    result = report(store, embedder, CASES)

    assert result["cases"] == 6
    assert result["hit_rate"] == 1.0
    assert result["mrr"] == 1.0
    assert result["misses"] == []
    assert [row["rank"] for row in result["results"]] == [1] * 6
    assert all(row["rr"] == 1.0 for row in result["results"])


def test_miss_lowers_mrr_and_is_reported(seeded):
    store, embedder = seeded
    # k=1 and an expected fact that is not the top hit -> guaranteed miss.
    result = report(store, embedder, [{"query": "atlas server ram", "expect": [5]}], k=1)

    assert result["hit_rate"] == 0.0
    assert result["mrr"] == 0.0
    assert result["results"][0]["rank"] is None
    assert result["results"][0]["hit"] is False
    assert result["misses"] == [
        {"query": "atlas server ram", "expected": [5], "got": [1]}]

    # A hit mixed with a miss averages the reciprocal ranks.
    cases = [{"query": "atlas server ram", "expect": [1]},
             {"query": "atlas server ram", "expect": [5]}]
    mixed = report(store, embedder, cases, k=1)
    assert mixed["hit_rate"] == 0.5
    assert mixed["mrr"] == 0.5
    assert len(mixed["misses"]) == 1


def test_unknown_expected_id_is_reported_not_scored(seeded):
    store, embedder = seeded
    result = report(store, embedder, [{"query": "atlas server ram", "expect": [999]}])

    assert result["unknown"] == [{"query": "atlas server ram", "missing": [999]}]
    assert result["cases"] == 0            # nothing scorable
    assert result["total"] == 1
    assert result["misses"] == []          # a stale id is not a retrieval miss
    assert result["results"][0]["scored"] is False
    assert result["results"][0]["rank"] is None


def test_partially_unknown_case_scores_the_known_id(seeded):
    store, embedder = seeded
    result = report(store, embedder, [{"query": "atlas server ram", "expect": [999, 1]}])

    assert result["cases"] == 1
    assert result["hit_rate"] == 1.0
    assert result["unknown"] == [{"query": "atlas server ram", "missing": [999]}]
    assert result["results"][0]["rank"] == 1


def test_k_is_respected(seeded):
    store, embedder = seeded
    one = report(store, embedder, CASES, k=1)
    assert all(len(row["got"]) == 1 for row in one["results"])
    # Fact 1 wins "atlas server ram"; asking for fact 2 with k=1 must miss.
    probe = report(store, embedder, [{"query": "atlas server ram", "expect": [2]}], k=1)
    assert probe["results"][0]["rank"] is None

    three = report(store, embedder, CASES, k=3)
    assert all(len(row["got"]) == 3 for row in three["results"])


def test_empty_cases_and_empty_store(tmp_path):
    empty_store = Store(tmp_path / "empty.db")
    try:
        result = live_metrics.evaluate(empty_store, [], embedder=FakeEmbedder())
        assert result["cases"] == 0
        assert result["hit_rate"] == 0.0
        assert result["mrr"] == 0.0
        assert result["misses"] == []
        assert result["results"] == []
        assert "store has no facts" in result["warnings"]
        assert live_metrics.gate(result) == {
            "ok": False, "reasons": ["no scored cases"], "warnings": []}
    finally:
        empty_store.close()


def test_results_are_deterministic_and_ordered(seeded):
    store, embedder = seeded
    first = report(store, embedder, CASES)
    second = report(store, embedder, CASES)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert [row["query"] for row in first["results"]] == [c["query"] for c in CASES]


def test_evaluate_is_read_only(seeded):
    store, embedder = seeded

    def snapshot():
        facts = [(row["fact_id"], row["updated_at"])
                 for row in store.all_facts()]
        audit = store.db.execute("SELECT COUNT(*) FROM audit").fetchone()[0]
        embeddings = store.db.execute(
            "SELECT COUNT(*) FROM fact_embeddings").fetchone()[0]
        return {"facts": facts, "count": store.count(),
                "audit": audit, "embeddings": embeddings}

    before = snapshot()
    report(store, embedder, CASES)
    report(store, embedder, CASES)
    assert snapshot() == before


def test_no_embedder_degrades_to_lexical(seeded):
    store, _ = seeded
    result = live_metrics.evaluate(store, CASES, k=5)

    assert result["retrieval"] == "lexical"
    assert result["model"] == "fake-embed"      # most common stored model
    assert result["hit_rate"] == 1.0
    assert result["warnings"] == []


def test_report_is_json_serialisable(seeded):
    store, embedder = seeded
    result = report(store, embedder, CASES)
    json.dumps(result)                          # no dataclasses, arrays or Rows

    broken = live_metrics.evaluate(store, [{"query": "", "expect": [1]},
                                           {"query": "x", "expect": ["1"]},
                                           "junk"],
                                   embedder=embedder)
    json.dumps(broken)
    assert broken["invalid"] == 3
    assert broken["cases"] == 0


# ------------------------------------------------------------------- compare
def test_compare_detects_regression(seeded):
    store, embedder = seeded
    case = [{"query": "atlas server ram", "expect": [1]}]
    before = report(store, embedder, case, k=1)

    decoy = store.add_fact("atlas server ram", trust_score=0.9)
    store.put_embedding(decoy["fact_id"], embedder.model,
                        embedder.embed_one("atlas server ram"))
    store.db.commit()
    after = report(store, embedder, case, k=1)

    diff = live_metrics.compare(before, after)
    assert before["hit_rate"] == 1.0 and after["hit_rate"] == 0.0
    assert diff["degraded"] is True
    assert diff["regressed"] == ["atlas server ram"]
    assert diff["hit_lost"] == ["atlas server ram"]
    assert diff["deltas"][0]["before_rank"] == 1
    assert diff["deltas"][0]["after_rank"] is None


def test_compare_detects_improvement(seeded):
    store, embedder = seeded
    decoy = store.add_fact("atlas server ram", trust_score=0.9)
    store.put_embedding(decoy["fact_id"], embedder.model,
                        embedder.embed_one("atlas server ram"))
    store.db.commit()
    case = [{"query": "atlas server ram", "expect": [1]}]
    before = report(store, embedder, case, k=1)

    store.remove_fact(decoy["fact_id"])
    after = report(store, embedder, case, k=1)

    diff = live_metrics.compare(before, after)
    assert after["hit_rate"] == 1.0
    assert diff["degraded"] is False
    assert diff["improved"] == ["atlas server ram"]
    assert diff["regressed"] == []
    assert diff["deltas"][0]["delta_rr"] == 1.0


def test_compare_flags_mrr_drop_even_without_a_lost_hit():
    before = {"mrr": 1.0, "hit_rate": 1.0, "cases": 2, "results": [
        {"query": "a", "rank": 1, "hit": True, "got": [1]},
        {"query": "b", "rank": 1, "hit": True, "got": [2]}]}
    after = {"mrr": 0.5, "hit_rate": 1.0, "cases": 2, "results": [
        {"query": "a", "rank": 2, "hit": True, "got": [9, 1]},
        {"query": "b", "rank": 2, "hit": True, "got": [9, 2]}]}

    diff = live_metrics.compare(before, after)
    assert diff["hit_lost"] == []
    assert diff["degraded"] is True          # 1.0 - 0.5 > 0.05
    assert diff["regressed"] == ["a", "b"]
    assert diff["mrr_delta"] == -0.5


def test_compare_unchanged_and_added_cases_are_not_regressions():
    before = {"mrr": 1.0, "hit_rate": 1.0, "cases": 1, "results": [
        {"query": "a", "rank": 1, "hit": True, "got": [1]}]}
    after = {"mrr": 1.0, "hit_rate": 1.0, "cases": 2, "results": [
        {"query": "a", "rank": 1, "hit": True, "got": [1]},
        {"query": "new", "rank": 1, "hit": True, "got": [7]}]}

    diff = live_metrics.compare(before, after)
    assert diff["unchanged"] == ["a"]
    assert diff["improved"] == [] and diff["regressed"] == []
    assert diff["added"] == ["new"]
    assert diff["degraded"] is False


def test_compare_tolerates_malformed_reports():
    diff = live_metrics.compare(None, {"results": "nope"})
    assert diff["degraded"] is False
    assert diff["deltas"] == []


# ---------------------------------------------------------------------- gate
def test_gate_passes_on_a_good_run(seeded):
    store, embedder = seeded
    result = report(store, embedder, CASES)
    assert live_metrics.gate(result) == {"ok": True, "reasons": [], "warnings": []}


def test_gate_fails_with_threshold_reasons():
    result = {"cases": 4, "hit_rate": 0.5, "mrr": 0.4, "unknown": []}
    gate = live_metrics.gate(result)
    assert gate["ok"] is False
    assert gate["reasons"] == ["hit_rate 0.500 < 0.850", "mrr 0.400 < 0.800"]

    only_mrr = live_metrics.gate({"cases": 4, "hit_rate": 1.0, "mrr": 0.4})
    assert only_mrr["reasons"] == ["mrr 0.400 < 0.800"]

    custom = live_metrics.gate({"cases": 4, "hit_rate": 0.9, "mrr": 0.85},
                               min_hit=0.95, min_mrr=0.90)
    assert custom["ok"] is False and len(custom["reasons"]) == 2


def test_gate_surfaces_unknown_ids_as_warnings(seeded):
    store, embedder = seeded
    result = report(store, embedder, [{"query": "atlas server ram", "expect": [999, 1]}])
    gate = live_metrics.gate(result)
    assert gate["ok"] is True
    assert gate["warnings"] == ["unknown expected ids: 999"]


# ----------------------------------------------------------------- load_cases
def test_load_cases_accepts_list_and_wrapped_formats(tmp_path, seeded):
    store, _ = seeded
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([
        {"query": "atlas server ram", "expect": [1]},
        {"q": "beacon mini ram", "facts": [3]},          # golden-style aliases
        {"query": "no expect"},                          # dropped
        "junk",                                          # dropped
    ]))
    cases = live_metrics.load_cases(path, store)
    assert [c["query"] for c in cases] == ["atlas server ram", "beacon mini ram"]
    assert cases[0] == {"query": "atlas server ram", "expect": [1]}

    wrapped = tmp_path / "golden.json"
    wrapped.write_text(json.dumps({"thresholds": {"mrr": 0.8}, "cases": [
        {"query": "mazda fuel tank", "expect": [5, 999]}]}))
    loaded = live_metrics.load_cases(wrapped, store)
    assert loaded == [{"query": "mazda fuel tank", "expect": [5, 999],
                       "unknown": [999]}]


def test_load_cases_missing_file_is_empty_and_bad_json_raises(tmp_path):
    assert live_metrics.load_cases(tmp_path / "absent.json") == []

    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(ValueError, match="invalid JSON"):
        live_metrics.load_cases(broken)


def test_validate_cases_separates_unknown_ids(seeded):
    store, _ = seeded
    report_ = live_metrics.validate_cases(store, [
        {"query": "a", "expect": [1, 2]},
        {"query": "b", "expect": [42, 43]},
        {"query": "", "expect": [1]},
    ])
    assert report_["cases"] == [
        {"query": "a", "expect": [1, 2]}, {"query": "b", "expect": [42, 43]}]
    assert report_["unknown"] == [{"query": "b", "missing": [42, 43]}]
    assert report_["invalid"] == 1


# ------------------------------------------------------- live-store interface
class ReadOnlyView:
    """Stand-in for the plugin store: only ``read`` + ``db_path``."""

    def __init__(self, path):
        self.db_path = str(path)

    def read(self, sql, params=()):
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute(sql, params)]
        finally:
            conn.close()


def test_evaluate_works_through_plugin_style_store(seeded):
    store, embedder = seeded
    view = ReadOnlyView(store.path)

    result = live_metrics.evaluate(view, live_metrics.load_cases(
        "/nonexistent", view) or CASES, k=5, embedder=embedder)

    assert result["retrieval"] == "hybrid"
    assert result["hit_rate"] == 1.0
    assert result["mrr"] == 1.0
    # read-only view is still readable afterwards, and the file was not touched
    assert view.read("SELECT COUNT(*) AS n FROM facts")[0]["n"] == 6
