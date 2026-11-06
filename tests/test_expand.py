"""Tests for the curated query expansion.

Everything here is pure: no store, no embedding endpoint, no tmp database.
Fixture text is generic and synthetic.
"""
from __future__ import annotations

import pytest

from factkit import expand as qe


def terms(query: str, **kwargs) -> list[str]:
    return qe.expand(query, **kwargs)["terms"]


def added(query: str, **kwargs) -> list[str]:
    return qe.expand(query, **kwargs)["added"]


# ------------------------------------------------------------------ aliases
def test_laptop_group_expands_both_directions():
    assert "notebook" in terms("laptop specifications")
    assert "laptop" in terms("notebook specifications")


def test_ram_memory_group_expands_both_directions():
    assert "memory" in terms("how much ram")
    assert "ram" in terms("how much memory")


def test_storage_group_expands_to_disk_and_ssd():
    assert terms("storage") == ["storage", "disk", "ssd"]
    assert "storage" in terms("ssd")
    assert "disk" in terms("ssd")


def test_arabic_transliteration_group():
    # سيارة -> "sayara" is an obvious transliteration, kept in the car group.
    assert {"vehicle", "sayara"} <= set(terms("car"))
    assert {"car", "vehicle"} <= set(terms("sayara"))


def test_alias_table_stays_small_and_consistent():
    assert 0 < len(qe.ALIASES) < 40
    seen: set[str] = set()
    for canonical, members in qe.ALIASES.items():
        assert canonical in members
        assert len(set(members)) == len(members), members
        assert not (set(members) & seen), members          # one group per word
        seen.update(members)
        for member in members:
            if len(member) < qe.MIN_EXPAND_LEN:
                continue
            for other in members:
                if other != member:
                    assert other in terms(member), (member, other)


# --------------------------------------------------------------- guards
def test_short_tokens_never_expand():
    assert qe.expand("ip")["terms"] == ["ip"]
    assert added("ip") == []
    assert terms("tts") == ["tts", "voice"]      # 3 chars, specific enough
    # the longer member still reaches the short one
    assert added("address") == ["ip"]


def test_numbers_and_units_never_expand():
    assert added("16 32 512 16gb 512mb gb tb ms") == []
    assert added("min") == []                               # unit word


def test_unknown_words_are_untouched():
    result = qe.expand("flux capacitor")
    assert result["terms"] == ["flux", "capacitor"]
    assert result["added"] == []
    assert result["dropped"] == []


def test_quoted_phrase_is_one_token_and_not_expanded():
    result = qe.expand('find "listen port" now')
    assert result["terms"] == ["find", '"listen port"', "now"]
    assert result["added"] == []
    assert qe.expand('"laptop"')["added"] == []


def test_empty_query():
    assert qe.expand("") == {"original": "", "terms": [], "added": [], "dropped": []}
    assert qe.for_prefetch("") == ""
    assert qe.apply("", []) == []


# ------------------------------------------------------- dedupe and cap
def test_dedupe_is_case_insensitive_and_order_is_stable():
    result = qe.expand("Laptop LAPTOP specs")
    assert result["terms"] == ["laptop", "specs", "notebook"]
    assert result["added"] == ["notebook"]
    assert result["original"] == "Laptop LAPTOP specs"
    assert result["terms"][:2] == ["laptop", "specs"]       # originals first


def test_originals_do_not_reappear_as_additions():
    assert added("laptop notebook") == []
    assert terms("memory ram") == ["memory", "ram"]


def test_max_terms_cap_and_dropped_report():
    result = qe.expand("laptop ram storage", max_terms=4)
    assert result["terms"] == ["laptop", "ram", "storage", "notebook"]
    assert result["added"] == ["notebook"]
    assert result["dropped"] == ["memory", "disk", "ssd"]
    assert len(result["terms"]) == 4


def test_max_terms_zero_drops_everything():
    result = qe.expand("laptop ram", max_terms=0)
    assert result["terms"] == [] and result["added"] == []
    assert result["dropped"] == ["laptop", "ram", "notebook", "memory"]


# --------------------------------------------------------------- apply
def test_apply_never_drops_and_does_not_mutate_input():
    hits = [
        {"fact_id": 1, "content": "printer queue is shared", "score": 0.62},
        {"fact_id": 2, "content": "notebook has 16 GB memory", "score": 0.40},
        {"fact_id": 3, "content": "backup job runs at 02:00", "score": 0.31},
    ]
    out = qe.apply("laptop ram", hits)
    assert sorted(h["fact_id"] for h in out) == [1, 2, 3]
    assert [h["score"] for h in hits] == [0.62, 0.40, 0.31]
    assert "expand_bonus" not in hits[1]


def test_apply_scores_stay_monotonic():
    hits = [
        {"content": "printer queue is shared", "score": 0.55},
        {"content": "notebook and memory", "score": 0.20},
        {"content": "unrelated note", "score": 0.44},
        {"content": "printer queue is shared", "score": 0.20},
    ]
    out = qe.apply("laptop ram", hits)
    assert [h["score"] for h in out] == sorted((h["score"] for h in out), reverse=True)
    assert out[-1]["score"] == 0.20                         # unmatched, still present


def test_apply_bonus_is_capped_below_the_gap():
    hits = [
        {"content": "notebook memory disk ssd", "score": 0.20},
        {"content": "printer queue is shared", "score": 0.60},
    ]
    out = qe.apply("laptop ram storage", hits)
    assert out[0]["content"] == "printer queue is shared"
    assert out[1]["score"] == pytest.approx(0.20 + qe.MAX_BONUS)
    assert out[1]["expand_bonus"] == pytest.approx(qe.MAX_BONUS)
    assert qe.MAX_BONUS <= 0.15


def test_apply_ties_keep_input_order():
    hits = [
        {"id": "a", "content": "notebook one", "score": 0.50},
        {"id": "b", "content": "notebook two", "score": 0.50},
    ]
    out = qe.apply("laptop", hits)
    assert [h["id"] for h in out] == ["a", "b"]


def test_apply_equal_content_keeps_higher_score_first():
    hits = [
        {"id": "low", "content": "notebook", "score": 0.30},
        {"id": "high", "content": "notebook", "score": 0.60},
    ]
    out = qe.apply("laptop", hits)
    assert [h["id"] for h in out] == ["high", "low"]
    assert [h["score"] for h in out] == [0.75, 0.45]       # same bonus, order kept


def test_apply_matches_whole_words_only():
    hits = [
        {"id": "inside", "content": "support ticket", "score": 0.50},
        {"id": "real", "content": "service listens on port 8080", "score": 0.40},
    ]
    out = qe.apply("listen", hits)
    assert [h["id"] for h in out] == ["real", "inside"]
    assert "expand_bonus" not in next(h for h in out if h["id"] == "inside")


def test_apply_ignores_original_tokens():
    hits = [{"content": "laptop only", "score": 0.50}]
    out = qe.apply("laptop", hits)
    assert out[0]["score"] == 0.50
    assert "expand_bonus" not in out[0]


def test_apply_weight_zero_is_identity():
    hits = [{"content": "notebook memory", "score": 0.2}]
    out = qe.apply("laptop ram", hits, weight=0.0)
    assert out == hits


# --------------------------------------------------------- for_prefetch
def test_for_prefetch_appends_expansions():
    text = qe.for_prefetch("laptop specifications")
    assert text == "laptop specifications notebook"
    assert text.startswith("laptop specifications")


def test_for_prefetch_length_guard_keeps_short_queries_alone():
    # "laptop notebook" is 15 chars against a 12-char budget.
    assert qe.for_prefetch("laptop") == "laptop"
    assert qe.for_prefetch("car") == "car"


def test_for_prefetch_never_exceeds_double_length():
    for query in ["laptop", "car", "ram", "laptop specifications",
                  "which notebook has more memory"]:
        out = qe.for_prefetch(query)
        assert out.startswith(query)
        assert len(out) <= 2 * len(query)


def test_for_prefetch_empty_query():
    assert qe.for_prefetch("") == ""
    assert qe.for_prefetch("   ") == ""


# ---------------------------------------------------------- determinism
def test_output_is_deterministic():
    query = "which laptop has more memory and storage"
    assert qe.expand(query) == qe.expand(query)
    hits = [{"content": "notebook memory", "score": 0.4},
            {"content": "printer", "score": 0.5}]
    assert qe.apply(query, hits) == qe.apply(query, hits)
    assert qe.for_prefetch(query) == qe.for_prefetch(query)
