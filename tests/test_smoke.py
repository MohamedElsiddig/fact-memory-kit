"""Tests for the retrieval smoke gate. Every prefetch call is stubbed offline."""

from __future__ import annotations

import json

from factkit import smoke


# ------------------------------------------------------------------ helpers
def hits(*fact_ids):
    """Prefetch-shaped hits, best first."""
    return [{"fact_id": fid, "content": f"fact {fid}", "score": round(1.0 - i * 0.01, 4)}
            for i, fid in enumerate(fact_ids)]


def case(query, *expect):
    return {"query": query, "expect": list(expect)}


def stub(table, calls=None):
    """Fake prefetch: query -> hits, or None to simulate a dead endpoint."""
    def fake(store, query, *, limit=5, timeout=1.5):
        if calls is not None:
            calls.append({"query": query, "limit": limit, "timeout": timeout})
        value = table.get(query, [])
        if isinstance(value, Exception):
            raise value
        return value
    return fake


# -------------------------------------------------------------------- pass
def test_all_cases_pass():
    cases = [case("atlas port", 1, 2), case("laptop ram", 3)]
    result = smoke.run_smoke(
        object(), cases,
        prefetch=stub({"atlas port": hits(9, 1), "laptop ram": hits(3, 4)}))

    assert result["ok"] is True
    assert result["checked"] == 2
    assert result["skipped"] == 0
    assert result["failures"] == []
    assert result["degraded"] is False


def test_one_failure_is_reported_with_got_ids():
    cases = [case("atlas port", 1), case("laptop ram", 3)]
    result = smoke.run_smoke(
        object(), cases,
        prefetch=stub({"atlas port": hits(1), "laptop ram": hits(7, 8)}))

    assert result["ok"] is False
    assert result["checked"] == 2
    assert len(result["failures"]) == 1
    failure = result["failures"][0]
    assert failure == {"query": "laptop ram", "expect": [3], "got": [7, 8]}


def test_any_expected_id_counts_as_a_pass():
    cases = [case("atlas specs", 99, 2)]
    result = smoke.run_smoke(object(), cases, prefetch=stub({"atlas specs": hits(2)}))

    assert result["ok"] is True and result["checked"] == 1


# ------------------------------------------------------------------ top_k
def test_expected_id_outside_top_k_fails():
    # fact 1 sits at rank 6; the default top_k=5 cannot see it.
    table = {"atlas specs": hits(7, 8, 9, 10, 11, 1)}
    result = smoke.run_smoke(object(), [case("atlas specs", 1)], prefetch=stub(table))

    assert result["ok"] is False
    assert len(result["failures"][0]["got"]) == 5
    assert 1 not in result["failures"][0]["got"]

    wider = smoke.run_smoke(object(), [case("atlas specs", 1)],
                            prefetch=stub(table), top_k=6)
    assert wider["ok"] is True
    assert wider["cases"][0]["got"] == [7, 8, 9, 10, 11, 1]


def test_top_k_and_timeout_are_forwarded():
    calls = []
    smoke.run_smoke(object(), [case("q", 1)], prefetch=stub({"q": hits(1)}, calls),
                    top_k=2, timeout=0.25)

    assert calls == [{"query": "q", "limit": 2, "timeout": 0.25}]


# --------------------------------------------------------------- degraded
def test_endpoint_down_degrades_instead_of_failing():
    cases = [case("a", 1), case("b", 2), case("c", 3)]
    result = smoke.run_smoke(object(), cases,
                             prefetch=stub({"a": None, "b": None, "c": None}))

    assert result["ok"] is True
    assert result["checked"] == 0
    assert result["skipped"] == 3
    assert result["degraded"] is True
    assert result["failures"] == []


def test_partial_endpoint_failure_keeps_checked_cases():
    cases = [case("a", 1), case("b", 2)]
    result = smoke.run_smoke(object(), cases,
                             prefetch=stub({"a": hits(1), "b": None}))

    assert result["ok"] is True
    assert result["checked"] == 1
    assert result["skipped"] == 1
    assert result["degraded"] is False


def test_prefetch_exception_is_a_skip_not_a_crash():
    cases = [case("a", 1)]
    result = smoke.run_smoke(object(), cases,
                             prefetch=stub({"a": OSError("connection refused")}))

    assert result["ok"] is True
    assert result["skipped"] == 1 and result["checked"] == 0
    assert result["degraded"] is True


def test_default_prefetch_resolves_to_holo(monkeypatch):
    """No endpoint is touched: the holo prefetch hook itself is replaced."""
    from factkit import holo

    calls = []

    def fake(store, query, *, limit=5, timeout=1.5):
        calls.append(query)
        return hits(1)

    monkeypatch.setattr(holo, "prefetch", fake)
    result = smoke.run_smoke(object(), [case("atlas", 1)])

    assert calls == ["atlas"]
    assert result["ok"] is True and result["checked"] == 1


# ---------------------------------------------------------------- compare
def test_compare_detects_regression():
    cases = [case("atlas port", 1), case("laptop ram", 3)]
    before = smoke.run_smoke(
        object(), cases,
        prefetch=stub({"atlas port": hits(1), "laptop ram": hits(3)}))
    after = smoke.run_smoke(
        object(), cases,
        prefetch=stub({"atlas port": hits(1), "laptop ram": hits(7)}))

    verdict = smoke.compare(before, after)
    assert verdict == {"regressed": ["laptop ram"], "ok": False}


def test_compare_ignores_newly_added_cases():
    before_cases = [case("atlas port", 1)]
    after_cases = before_cases + [case("brand new", 42)]
    before = smoke.run_smoke(object(), before_cases, prefetch=stub({"atlas port": hits(1)}))
    after = smoke.run_smoke(object(), after_cases,
                            prefetch=stub({"atlas port": hits(1), "brand new": hits(7)}))

    assert smoke.compare(before, after) == {"regressed": [], "ok": True}


def test_compare_ignores_cases_skipped_after():
    cases = [case("atlas port", 1), case("laptop ram", 3)]
    before = smoke.run_smoke(
        object(), cases, prefetch=stub({"atlas port": hits(1), "laptop ram": hits(3)}))
    after = smoke.run_smoke(
        object(), cases, prefetch=stub({"atlas port": hits(1), "laptop ram": None}))

    assert smoke.compare(before, after) == {"regressed": [], "ok": True}


def test_compare_ignores_case_skipped_before():
    cases = [case("laptop ram", 3)]
    before = smoke.run_smoke(object(), cases, prefetch=stub({"laptop ram": None}))
    after = smoke.run_smoke(object(), cases, prefetch=stub({"laptop ram": hits(7)}))

    assert smoke.compare(before, after) == {"regressed": [], "ok": True}


def test_compare_does_not_flag_still_failing_case():
    cases = [case("atlas port", 1)]
    before = smoke.run_smoke(object(), cases, prefetch=stub({"atlas port": hits(7)}))
    after = smoke.run_smoke(object(), cases, prefetch=stub({"atlas port": hits(8)}))

    assert smoke.compare(before, after) == {"regressed": [], "ok": True}


# ------------------------------------------------------------ edges / junk
def test_empty_case_list_is_ok_but_degraded():
    result = smoke.run_smoke(object(), [], prefetch=stub({}))

    assert result["ok"] is True
    assert result["checked"] == 0
    assert result["skipped"] == 0
    assert result["failures"] == []
    assert result["degraded"] is True
    assert smoke.compare(result, result) == {"regressed": [], "ok": True}


def test_junk_cases_are_skipped_without_raising():
    junk = [
        None,
        "not a case",
        {},
        {"query": "no expect"},
        {"query": "", "expect": [1]},
        {"query": "string id", "expect": ["1"]},
        {"query": "bool id", "expect": [True]},
        {"query": "empty expect", "expect": []},
        {"query": "expect not a list", "expect": 7},
        {"query": "mixed ids", "expect": [1, "2"]},
        {"query": "  ", "expect": [1]},
    ]
    calls = []
    result = smoke.run_smoke(object(), junk + [case("real", 5)],
                             prefetch=stub({"real": hits(5)}, calls))

    assert result["ok"] is True
    assert result["checked"] == 1
    assert result["invalid"] == len(junk)
    assert calls == [{"query": "real", "limit": 5, "timeout": 1.5}]


def test_cases_that_are_not_a_list_do_not_raise():
    for bad in (None, 7, "nope"):
        result = smoke.run_smoke(object(), bad, prefetch=stub({}))
        assert result["ok"] is True and result["checked"] == 0


def test_load_cases_reads_json_list(tmp_path):
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps([
        {"query": "atlas port", "expect": [1, 2]},
        {"query": "laptop ram", "expect": [3]},
        "junk entry",
    ]))

    cases = smoke.load_cases(path)
    assert cases == [{"query": "atlas port", "expect": [1, 2]},
                     {"query": "laptop ram", "expect": [3]}]


def test_load_cases_missing_file_is_empty(tmp_path):
    assert smoke.load_cases(tmp_path / "absent.json") == []
