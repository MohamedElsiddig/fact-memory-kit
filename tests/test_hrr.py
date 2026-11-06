"""Tests for the HRR candidate re-ranker (no network, no live hook)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from factkit import hrr
from factkit.store import Store

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hrr_bench.py"


FACTS = [
    {"fact_id": 1, "content": "Atlas Server has 16 GB RAM"},
    {"fact_id": 2, "content": "Beacon Mini has 1 GB RAM"},
    {"fact_id": 3, "content": "Backup runs at 02:00 daily"},
]
BY_ID = {f["fact_id"]: f for f in FACTS}


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "facts.db")
    for content in ("Atlas Server has 16 GB RAM",
                    "Beacon Mini has 1 GB RAM",
                    "Backup runs at 02:00 daily"):
        s.add_fact(content, category="entity", trust_score=0.9)
    yield s
    s.close()


# ------------------------------------------------------------------ encode
def test_encode_is_deterministic_for_the_same_seed():
    first = hrr.encode(FACTS, seed=3)
    second = hrr.encode(FACTS, seed=3)
    assert first == second
    assert first[1] == hrr.encode(FACTS, seed=3)[1]


def test_different_seeds_produce_different_vectors():
    zero = hrr.encode(FACTS, seed=0)
    other = hrr.encode(FACTS, seed=1)
    assert zero[1] != other[1]
    assert hrr.similarity(zero[1], other[1]) < 0.99


def test_encoded_vectors_are_unit_length():
    encoded = hrr.encode(FACTS)
    assert set(encoded) == {1, 2, 3}
    for vec in encoded.values():
        assert len(vec) == hrr.DIM
        norm = sum(x * x for x in vec) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-9)


def test_empty_content_encodes_to_zero_vector():
    encoded = hrr.encode([{"fact_id": 9, "content": ""}])
    assert encoded[9] == [0.0] * hrr.DIM
    assert hrr.similarity(encoded[9], encoded[9]) == 0.0


def test_position_binding_differentiates_word_order():
    a = hrr.encode([{"fact_id": 1, "content": "alpha beta gamma"}], seed=5)[1]
    b = hrr.encode([{"fact_id": 1, "content": "gamma beta alpha"}], seed=5)[1]
    assert a != b
    # same bag, different order: similar but not identical
    assert hrr.similarity(a, b) < 0.99


def test_large_unknown_vocabulary_stays_bounded():
    # hashing caps the index space even well beyond VOCAB_CAP distinct tokens
    indices = {hrr._token_index(f"token{i}", 0) for i in range(5_000)}
    assert all(0 <= i < hrr.VOCAB_CAP for i in indices)

    facts = [{"fact_id": i, "content": f"unique{i} word{i}"} for i in range(200)]
    encoded = hrr.encode(facts)
    assert len(encoded) == 200
    assert all(len(v) == hrr.DIM for v in encoded.values())
    assert len(hrr._VECTOR_CACHE) <= hrr.CACHE_MAX


# -------------------------------------------------------------- similarity
def test_similarity_is_symmetric_and_self_is_one():
    encoded = hrr.encode(FACTS)
    a, b = encoded[1], encoded[2]
    assert hrr.similarity(a, a) == pytest.approx(1.0, abs=1e-12)
    assert hrr.similarity(a, b) == pytest.approx(hrr.similarity(b, a), abs=1e-12)


def test_similarity_zero_for_zero_vectors_and_length_mismatch():
    zero = [0.0] * 8
    assert hrr.similarity(zero, [1.0] * 8) == 0.0
    assert hrr.similarity([1.0] * 8, zero) == 0.0
    assert hrr.similarity([1.0] * 8, [1.0] * 4) == 0.0
    assert hrr.similarity([], []) == 0.0


# -------------------------------------------------------------------- rank
def test_rank_orders_by_descending_score():
    encoded = hrr.encode(FACTS)
    rows = hrr.rank("atlas server 16 gb ram", encoded, BY_ID, k=3)
    assert rows[0]["fact_id"] == 1
    assert [r["score"] for r in rows] == sorted((r["score"] for r in rows), reverse=True)
    assert rows[0]["score"] > rows[-1]["score"]


def test_rank_ties_break_on_ascending_fact_id():
    same = [{"fact_id": 7, "content": "same words here"},
            {"fact_id": 3, "content": "same words here"}]
    encoded = hrr.encode(same)
    rows = hrr.rank("same words here", encoded, {f["fact_id"]: f for f in same}, k=2)
    assert [r["fact_id"] for r in rows] == [3, 7]
    assert rows[0]["score"] == rows[1]["score"]


def test_rank_caps_at_k_and_handles_non_positive_k():
    encoded = hrr.encode(FACTS)
    assert len(hrr.rank("ram", encoded, BY_ID, k=1)) == 1
    assert len(hrr.rank("ram", encoded, BY_ID, k=99)) == 3
    assert hrr.rank("ram", encoded, BY_ID, k=0) == []
    assert hrr.rank("ram", encoded, BY_ID, k=-3) == []


def test_rank_empty_facts_is_empty():
    assert hrr.encode([]) == {}
    assert hrr.rank("anything", {}, {}, k=5) == []


def test_rank_no_token_overlap_scores_exactly_zero():
    encoded = hrr.encode(FACTS)
    rows = hrr.rank("kubernetes helm chart", encoded, BY_ID, k=5)
    assert [r["fact_id"] for r in rows] == [1, 2, 3]     # tie broken by id
    assert all(r["score"] == 0.0 for r in rows)


def test_rank_stopword_only_query_scores_zero():
    encoded = hrr.encode(FACTS)
    rows = hrr.rank("what is the of", encoded, BY_ID, k=3)
    assert all(r["score"] == 0.0 for r in rows)


# ------------------------------------------------------- compare + baseline
def test_compare_with_provider_known_ranks_and_math(store):
    cases = [
        {"query": "atlas server ram", "expect": [1]},
        {"query": "atlas 16 gb ram", "expect": [2]},      # lexical prefers 1
        {"query": "backup daily", "expect": [3]},
        {"query": "no such words", "expect": [999]},      # both miss
        {"query": "backup daily", "expect": []},          # excluded from aggregates
    ]
    result = hrr.compare_with_provider(store, cases, k=2)

    ranks = [(c["hrr_rank"], c["lexical_rank"]) for c in result["cases"]]
    assert ranks == [(1, 1), (1, 2), (1, 1), (None, None), (None, None)]

    # 3 hits in top-2 of 4 counted cases, reciprocal ranks 1,1,1,0
    assert result["hrr"] == {"hit_at_k": 0.75, "mrr": 0.75, "n": 4}
    # lexical: 1, 1/2, 1, 0 -> MRR 0.625
    assert result["lexical"] == {"hit_at_k": 0.75, "mrr": 0.625, "n": 4}
    assert (result["wins"], result["losses"], result["ties"]) == (1, 0, 3)
    assert result["k"] == 2
    assert result["cases"][1]["hrr_top"] == [2, 1]
    assert result["cases"][1]["lexical_top"] == [1, 2]


def test_compare_counts_losses(store):
    result = hrr.compare_with_provider(
        store, [{"query": "atlas 16 gb ram", "expect": [1]}], k=2)
    assert result["cases"][0]["hrr_rank"] == 2
    assert result["cases"][0]["lexical_rank"] == 1
    assert (result["wins"], result["losses"], result["ties"]) == (0, 1, 0)
    assert result["wins"] + result["losses"] + result["ties"] == 1


def test_compare_is_read_only(store):
    before = (store.path.stat().st_mtime_ns, store.path.stat().st_size)
    before_rows = [dict(r) for r in store.db.execute(
        "SELECT fact_id, content, updated_at FROM facts ORDER BY fact_id")]
    hrr.compare_with_provider(
        store, [{"query": "atlas ram", "expect": [1]}], k=2)
    after = (store.path.stat().st_mtime_ns, store.path.stat().st_size)
    after_rows = [dict(r) for r in store.db.execute(
        "SELECT fact_id, content, updated_at FROM facts ORDER BY fact_id")]
    assert after == before
    assert after_rows == before_rows


# ------------------------------------------------------------------- bench
def test_bench_script_help_works_without_network():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                          cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "--k" in proc.stdout and "cases" in proc.stdout


def test_bench_script_reports_and_gates_on_verdict(tmp_path):
    db = tmp_path / "bench.db"
    s = Store(db)
    # A long distractor wins the lexical tie by id; HRR prefers the focused fact.
    s.add_fact("Atlas ram notes " + " ".join(f"filler{i}" for i in range(30)))
    s.add_fact("Atlas ram")
    s.close()

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"query": "atlas ram", "expect": [2]},
    ]))
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(cases), "--db", str(db), "--k", "1"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert "HRR" in proc.stdout and "W/L/T" in proc.stdout
    assert "VERDICT: HRR beats lexical" in proc.stdout
    assert proc.returncode == 0

    # a losing fixture must trip the gate
    cases.write_text(json.dumps([{"query": "atlas ram", "expect": [1]}]))
    losing = subprocess.run(
        [sys.executable, str(SCRIPT), str(cases), "--db", str(db), "--k", "1"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert "does NOT beat lexical" in losing.stdout
    assert losing.returncode == 1

    # missing cases file fails clearly, not with a traceback
    bad = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "absent.json")],
                         cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert bad.returncode == 2
    assert "cannot read cases file" in bad.stderr
    assert "Traceback" not in bad.stderr
