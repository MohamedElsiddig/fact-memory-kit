"""Policy tests: every dict mirrors ``holo``'s real detector output.

``conflict()`` starts from exactly the keys ``holo.find_conflicts`` emits and
adds only the optional category/trust enrichment the caller can supply.
``dupe()`` starts from ``holo.merge_duplicates`` (which uses ``ratio``).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from factkit.policy import Policy, classify, plan

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
AUTO = Policy(auto=True)


def conflict(**overrides):
    pair = {
        "keep": 2,
        "retire": 1,
        "keep_text": "Atlas Server listens on port 8010",
        "retire_text": "Atlas Server listens on port 8000-8100",
        "keep_updated": "2026-09-01 00:00:00",
        "retire_updated": "2026-01-01 00:00:00",
        "shared_entities": ["Atlas Server"],
        "jaccard": 0.62,
        "diffs": [{"slot": "port", "keep": ["8010"], "retire": ["8000-8100"]}],
        "keep_category": "entity",
        "retire_category": "entity",
        "keep_trust": 0.9,
        "retire_trust": 0.7,
    }
    pair.update(overrides)
    return pair


def dupe(**overrides):
    pair = {
        "keep": 3,
        "retire": 4,
        "ratio": 0.995,
        "keep_text": "Proxy listens on port 8080",
        "retire_text": "Proxy listens on port 8080.",
    }
    pair.update(overrides)
    return pair


# ------------------------------------------------------------------ near dupes
def test_near_duplicate_is_auto():
    assert classify(dupe(), now=NOW, policy=AUTO) == "auto"


def test_auto_disabled_turns_auto_into_propose():
    assert classify(dupe(), now=NOW) == "propose"
    assert classify(conflict(), now=NOW) == "propose"


def test_similarity_aliases_and_threshold():
    assert classify(dupe(ratio=None, score=0.99), now=NOW, policy=AUTO) == "auto"
    assert classify(dupe(ratio=0.98), now=NOW, policy=AUTO) == "auto"
    assert classify(dupe(ratio=0.979), now=NOW, policy=AUTO) == "propose"


def test_jaccard_is_not_a_duplicate_ratio():
    # Jaccard is the conflict detector's overlap; 1.0 with no shared entity and
    # no trust must still need review rather than auto-merge.
    pair = conflict(jaccard=1.0, shared_entities=[])
    pair.pop("keep_trust")
    pair.pop("retire_trust")
    assert classify(pair, now=NOW, policy=AUTO) == "propose"


# ------------------------------------------------------------------ conflicts
def test_port_conflict_with_entity_age_trust_is_auto():
    assert classify(conflict(), now=NOW, policy=AUTO) == "auto"


def test_no_shared_entity_proposes():
    assert classify(conflict(shared_entities=[]), now=NOW, policy=AUTO) == "propose"
    assert classify(conflict(shared_entities=None), now=NOW, policy=AUTO) == "propose"


def test_attribute_not_allowed_proposes():
    pair = conflict(diffs=[{"slot": "ram", "keep": ["16gb"], "retire": ["8gb"]}])
    assert classify(pair, now=NOW, policy=AUTO) == "propose"


def test_age_gap_too_small_proposes():
    assert classify(conflict(retire_updated="2026-08-29 00:00:00"),
                    now=NOW, policy=AUTO) == "propose"


def test_older_higher_trust_proposes():
    assert classify(conflict(retire_trust=0.95, keep_trust=0.8),
                    now=NOW, policy=AUTO) == "propose"


def test_cool_off_window_proposes():
    recent = (NOW - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    assert classify(conflict(keep_updated=recent), now=NOW, policy=AUTO) == "propose"


def test_reversed_pair_proposes():
    # 'retire' newer than 'keep' is inconsistent input, never automatic.
    assert classify(conflict(keep_updated="2026-01-01 00:00:00",
                             retire_updated="2026-09-01 00:00:00"),
                    now=NOW, policy=AUTO) == "propose"


def test_nested_fact_rows_supply_metadata():
    pair = {
        "keep": 2,
        "retire": 1,
        "keep_updated": "2026-09-01 00:00:00",
        "retire_updated": "2026-01-01 00:00:00",
        "shared_entities": ["Atlas Server"],
        "diffs": [{"slot": "port"}],
        "keep_fact": {"fact_id": 2, "category": "entity", "trust_score": 0.9},
        "retire_fact": {"fact_id": 1, "category": "entity", "trust_score": 0.7},
    }
    assert classify(pair, now=NOW, policy=AUTO) == "auto"


# --------------------------------------------------------------------- never
def test_denied_category_never():
    policy = Policy(auto=True, deny_categories=frozenset({"user_pref"}))
    pair = conflict(keep_category="user_pref", retire_category="user_pref")
    assert classify(pair, now=NOW, policy=policy) == "never"


def test_cross_category_never():
    assert classify(conflict(keep_category="entity", retire_category="tool"),
                    now=NOW, policy=AUTO) == "never"


def test_user_origin_marker_never():
    assert classify(conflict(source="user"), now=NOW, policy=AUTO) == "never"
    assert classify(conflict(retire_source="user"), now=NOW, policy=AUTO) == "never"
    assert classify(conflict(retire_verbatim=True), now=NOW, policy=AUTO) == "never"


def test_near_duplicate_does_not_bypass_denied_category():
    policy = Policy(auto=True, deny_categories=frozenset({"secret"}))
    assert classify(dupe(keep_category="secret", retire_category="secret"),
                    now=NOW, policy=policy) == "never"


# ---------------------------------------------------------------- bad shapes
def test_missing_keys_propose():
    # Literal find_conflicts shape without category/trust enrichment.
    pair = {
        "keep": 2,
        "retire": 1,
        "keep_text": "Atlas Server listens on port 8010",
        "retire_text": "Atlas Server listens on port 8000-8100",
        "keep_updated": "2026-09-01 00:00:00",
        "retire_updated": "2026-01-01 00:00:00",
        "shared_entities": ["Atlas Server"],
        "jaccard": 0.62,
        "diffs": [{"slot": "port", "keep": ["8010"], "retire": ["8000-8100"]}],
    }
    assert classify(pair, now=NOW, policy=AUTO) == "propose"
    assert classify({"keep": 1}, now=NOW, policy=AUTO) == "propose"


def test_unknown_pair_shape_proposes():
    assert classify({}, now=NOW, policy=AUTO) == "propose"
    assert classify(None, now=NOW, policy=AUTO) == "propose"  # type: ignore[arg-type]
    assert classify({"ratio": 0.5, "note": "odd"}, now=NOW, policy=AUTO) == "propose"
    assert classify({"ratio": 0.99, "keep_text": "x"}, now=NOW, policy=AUTO) == "propose"


def test_timestamps_accept_z_and_naive_now():
    assert classify(conflict(keep_updated="2026-09-01T00:00:00Z"),
                    now=NOW, policy=AUTO) == "auto"
    naive = datetime(2026, 9, 21, 12, 0)
    assert classify(conflict(), now=naive, policy=AUTO) == "auto"


# ---------------------------------------------------------------------- plan
def test_plan_caps_apply_count():
    pairs = [conflict(keep=i + 2, retire=i + 1) for i in range(3)]
    decisions = plan(pairs, now=NOW, policy=Policy(auto=True, max_apply=2, max_ratio=1.0))
    assert decisions == ["auto", "auto", "propose"]
    assert sum(d == "auto" for d in decisions) == 2


def test_plan_ratio_budget_and_master_switch():
    pairs = [conflict(keep=i + 2, retire=i + 1) for i in range(10)]
    assert sum(d == "auto" for d in plan(pairs, now=NOW, policy=Policy(auto=True))) == 2
    assert plan(pairs, now=NOW) == ["propose"] * 10
    assert plan([], now=NOW, policy=AUTO) == []


def test_plan_keeps_never_and_prioritises_dupes():
    pairs = [conflict(keep=2, retire=1), dupe(), conflict(keep=6, retire=5)]
    decisions = plan(pairs, now=NOW, policy=Policy(auto=True, max_apply=1, max_ratio=0.2))
    assert decisions == ["propose", "auto", "propose"]
