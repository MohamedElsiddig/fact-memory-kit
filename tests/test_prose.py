"""Prose contradiction detection: candidate pairs, verdicts and proposals."""
import json
from types import SimpleNamespace

import pytest

from factkit import holo, prose
from factkit.store import Store


# ------------------------------------------------------------------ helpers
def make_store(tmp_path, contents, *, name="m.db", entities=()):
    store = Store(tmp_path / name)
    ids = []
    for content in contents:
        fid, _ = store.add_fact_status(content, category="tool", tags="", trust=0.7,
                                       entities=list(entities))
        ids.append(fid)
    return store, ids


def pair(a_id=1, b_id=2, a="The proxy runs on node alpha",
         b="The proxy runs on node beta"):
    return {"a": {"fact_id": a_id, "content": a},
            "b": {"fact_id": b_id, "content": b},
            "shared": ["node", "proxy", "runs"]}


def answer(contradiction=True, confidence=0.9, attribute="host",
           evidence=("node alpha", "node beta")):
    return json.dumps({"contradiction": contradiction, "confidence": confidence,
                       "attribute": attribute, "evidence": list(evidence)})


class FakeClient:
    """OpenAI-compatible client stand-in; never touches the network."""

    def __init__(self, answers=(), error_after=None):
        self.answers = list(answers)
        self.error_after = error_after
        self.calls = []
        self.model = "fake-model"
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error_after is not None and len(self.calls) > self.error_after:
            raise OSError("connection refused")
        if not self.answers:
            raise AssertionError("fake client has no canned answer left")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.answers.pop(0)))])


# ------------------------------------------------------------------- pairs
def test_candidate_pairs_finds_shared_topic(tmp_path):
    store, ids = make_store(tmp_path, [
        "The proxy runs on node alpha",
        "The proxy runs on node beta",
        "Backups land on the vault disk",
    ])

    pairs = prose.candidate_pairs(store)

    assert len(pairs) == 1
    found = pairs[0]
    assert {found["a"]["fact_id"], found["b"]["fact_id"]} == set(ids[:2])
    assert found["shared"] == ["node", "proxy", "runs"]
    assert set(found) == {"a", "b", "shared"}
    assert set(found["a"]) == {"fact_id", "content"}
    assert found["a"]["content"] == "The proxy runs on node alpha"


def test_candidate_pairs_ignores_stopwords_and_numbers(tmp_path):
    store, ids = make_store(tmp_path, [
        "The value of it is 12345",
        "That value of it is 67890",
        "The proxy runs on node alpha",
        "The proxy runs on node beta",
    ])

    pairs = prose.candidate_pairs(store)

    # only "value" is shared by the first two: below min_overlap and no payload
    assert [(p["a"]["fact_id"], p["b"]["fact_id"]) for p in pairs] == [(ids[2], ids[3])]
    for found in pairs:
        assert not set(found["shared"]) & holo.STOP
        assert not any(token.isdigit() for token in found["shared"])


def test_candidate_pairs_excludes_numeric_conflicts(tmp_path):
    store, _ = make_store(tmp_path, [
        "Test Node alpha runs on port 7000",
        "Test Node alpha runs on port 7001",
    ], entities=["Test Node alpha"])

    assert holo.find_conflicts(store.db_path)  # the numeric path owns this pair
    assert prose.candidate_pairs(store) == []


def test_candidate_pairs_respects_min_overlap(tmp_path):
    store, _ = make_store(tmp_path, [
        "Gateway listens on rack seven",
        "Gateway listens on rack nine",
    ])

    assert len(prose.candidate_pairs(store, min_overlap=3)) == 1
    assert prose.candidate_pairs(store, min_overlap=4) == []


def test_candidate_pairs_is_deterministic_and_capped(tmp_path):
    store, _ = make_store(tmp_path, [
        "The proxy runs on node alpha",
        "The proxy runs on node beta",
        "Nightly backup lands on vault disk",
        "Nightly backup lands on mirror disk",
        "Resolver answers queries from rack seven",
        "Resolver answers queries from rack nine",
    ])

    first = prose.candidate_pairs(store)
    second = prose.candidate_pairs(store)
    assert first == second
    assert len(first) == 3

    capped = prose.candidate_pairs(store, max_pairs=2)
    assert capped == first[:2]
    assert len(capped) == 2


# -------------------------------------------------------------- adjudicate
def test_adjudicate_pairs_parses_contradiction():
    client = FakeClient([answer(True, 0.9)])

    results = prose.adjudicate_pairs([pair()], client)

    assert len(results) == 1
    assert results[0]["verdict"] == "contradiction"
    assert results[0]["confidence"] == pytest.approx(0.9)
    assert results[0]["attribute"] == "host"
    assert results[0]["evidence"] == ["node alpha", "node beta"]
    assert results.error is None


def test_adjudicate_pairs_parses_compatible():
    results = prose.adjudicate_pairs([pair()], FakeClient([answer(False, 0.95)]))

    assert len(results) == 1
    assert results[0]["verdict"] == "compatible"
    assert results[0]["contradiction"] is False


def test_low_confidence_is_downgraded_to_unknown():
    results = prose.adjudicate_pairs([pair()], FakeClient([answer(True, 0.2)]))

    assert results[0]["verdict"] == "unknown"
    assert results[0]["confidence"] == pytest.approx(0.2)


def test_missing_confidence_is_unknown():
    results = prose.adjudicate_pairs(
        [pair()], FakeClient(['{"contradiction": true, "attribute": "host"}']))

    assert results[0]["verdict"] == "unknown"
    assert results[0]["confidence"] == 0.0


def test_unparseable_answer_is_skipped():
    results = prose.adjudicate_pairs([pair()], FakeClient(["I cannot help with that"]))

    assert results == []
    assert results.error is None


def test_endpoint_failure_keeps_partial_results():
    client = FakeClient([answer(True, 0.9), answer(False, 0.9)], error_after=1)

    results = prose.adjudicate_pairs([pair(1, 2), pair(3, 4)], client)

    assert len(results) == 1
    assert results[0]["verdict"] == "contradiction"
    assert "connection refused" in results.error
    assert results["error"] == results.error
    assert "error" in results
    assert results.get("error") == results.error


def test_model_is_passed_to_client():
    client = FakeClient([answer()])
    prose.adjudicate_pairs([pair()], client, model="tiny")
    assert client.calls[0]["model"] == "tiny"


def test_callable_client_is_supported():
    results = prose.adjudicate_pairs([pair()], lambda prompt: answer(True))
    assert results[0]["verdict"] == "contradiction"


# -------------------------------------------------------------- proposals
def test_prose_scan_records_pending_proposal(tmp_path):
    store, ids = make_store(tmp_path, [
        "The proxy runs on node alpha",
        "The proxy runs on node beta",
    ])

    summary = prose.prose_scan(store, FakeClient([answer(True, 0.9)]))

    assert summary == {"pairs": 1, "judged": 1, "proposed": 1, "skipped": 0}
    row = store._one("SELECT * FROM conflict_proposals")
    assert row["status"] == "pending"
    assert (row["keep_id"], row["retire_id"]) == (ids[1], ids[0])
    assert row["slots"] == "prose"
    assert row["batch_id"].startswith("prose-")
    assert row["note"] == f"batch={row['batch_id']}"
    assert row["jaccard"] > 0


def test_record_is_idempotent(tmp_path):
    store, _ = make_store(tmp_path, [
        "The proxy runs on node alpha",
        "The proxy runs on node beta",
    ])

    first = prose.prose_scan(store, FakeClient([answer(True)]))
    second = prose.prose_scan(store, FakeClient([answer(True)]))

    rows = store.read("SELECT * FROM conflict_proposals")
    assert len(rows) == 1
    assert first["proposed"] == 1
    assert second["proposed"] == 0
    assert second["judged"] == 1          # still judged, just already proposed
    assert rows[0]["batch_id"].startswith("prose-")


def test_proposal_evidence_is_json(tmp_path):
    store, ids = make_store(tmp_path, [
        "The proxy runs on node alpha",
        "The proxy runs on node beta",
    ])

    prose.prose_scan(store, FakeClient([answer(True, 0.8)]))

    evidence = json.loads(store._one("SELECT evidence FROM conflict_proposals")["evidence"])
    assert evidence["kind"] == "prose"
    assert evidence["verdict"] == "contradiction"
    assert evidence["confidence"] == pytest.approx(0.8)
    assert evidence["shared"] == ["node", "proxy", "runs"]
    assert (evidence["keep_id"], evidence["retire_id"]) == (ids[1], ids[0])
    assert evidence["keep_text"] == "The proxy runs on node beta"
    assert evidence["retire_text"] == "The proxy runs on node alpha"


def test_prose_scan_leaves_facts_untouched(tmp_path):
    store, _ = make_store(tmp_path, [
        "The proxy runs on node alpha",
        "The proxy runs on node beta",
    ])
    before = [tuple(r) for r in store.read(
        "SELECT fact_id, content, updated_at FROM facts ORDER BY fact_id")]

    prose.prose_scan(store, FakeClient([answer(True)]))

    after = [tuple(r) for r in store.read(
        "SELECT fact_id, content, updated_at FROM facts ORDER BY fact_id")]
    assert after == before
    assert store.count_facts() == 2


def test_compatible_pairs_are_not_proposed(tmp_path):
    store, _ = make_store(tmp_path, [
        "The proxy runs on node alpha",
        "The proxy runs on node beta",
    ])

    summary = prose.prose_scan(store, FakeClient([answer(False, 0.9)]))

    assert summary["judged"] == 1
    assert summary["proposed"] == 0
    tables = {r[0] for r in store.read(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "conflict_proposals" not in tables or store.read(
        "SELECT * FROM conflict_proposals") == []


def test_empty_store(tmp_path):
    store = Store(tmp_path / "empty.db")

    assert prose.candidate_pairs(store) == []
    assert prose.prose_scan(store, FakeClient([])) == {
        "pairs": 0, "judged": 0, "proposed": 0, "skipped": 0}
    assert prose.record(store, [], batch_id="b-empty") == 0


def test_prose_scan_honours_max_pairs(tmp_path):
    store, _ = make_store(tmp_path, [
        "The proxy runs on node alpha",
        "The proxy runs on node beta",
        "Nightly backup lands on vault disk",
        "Nightly backup lands on mirror disk",
        "Resolver answers queries from rack seven",
        "Resolver answers queries from rack nine",
    ])
    client = FakeClient([answer(True), answer(True)])

    summary = prose.prose_scan(store, client, max_pairs=2)

    assert summary["pairs"] == 2
    assert summary["judged"] == 2
    assert len(client.calls) == 2
    assert len(store.read("SELECT * FROM conflict_proposals")) == 2


def test_scan_without_client_only_lists_pairs(tmp_path):
    store, _ = make_store(tmp_path, [
        "The proxy runs on node alpha",
        "The proxy runs on node beta",
    ])

    summary = prose.prose_scan(store)

    assert summary == {"pairs": 1, "judged": 0, "proposed": 0, "skipped": 1}
    assert "error" not in summary


def test_endpoint_failure_reports_partial_scan(tmp_path):
    store, _ = make_store(tmp_path, [
        "The proxy runs on node alpha",
        "The proxy runs on node beta",
        "Nightly backup lands on vault disk",
        "Nightly backup lands on mirror disk",
    ])
    client = FakeClient([answer(True)], error_after=1)

    summary = prose.prose_scan(store, client)

    assert summary["pairs"] == 2
    assert summary["judged"] == 1
    assert summary["proposed"] == 1
    assert summary["skipped"] == 1
    assert "connection refused" in summary["error"]
