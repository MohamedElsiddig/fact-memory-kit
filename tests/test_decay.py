"""Tests for age-based confidence decay. Synthetic, offline, read-only-store safe."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from factkit import decay, holo
from factkit.store import Store

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


def _ago(days: float) -> str:
    return (NOW - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def _fact(**over) -> dict:
    fact = {
        "fact_id": 1,
        "content": "Synthetic fact alpha",
        "category": "entity",
        "trust_score": 0.5,
        "created_at": _ago(0),
        "updated_at": _ago(0),
    }
    fact.update(over)
    return fact


def _store(tmp_path, specs):
    """A throwaway plugin store whose rows carry explicit ages."""
    store = Store(tmp_path / "decay.db")
    ids = []
    for spec in specs:
        fact_id, _ = store.add_fact_status(
            spec["content"], category=spec.get("category", "entity"),
            tags="", trust=spec.get("trust", 0.5))
        ids.append(int(fact_id))
    con = sqlite3.connect(store.db_path)
    try:
        for fact_id, spec in zip(ids, specs):
            con.execute(
                "UPDATE facts SET created_at = ?, updated_at = ? WHERE fact_id = ?",
                (_ago(spec["age"]), _ago(spec["age"]), fact_id))
        con.commit()
    finally:
        con.close()
    try:
        store.db.execute("DELETE FROM audit")
        store.db.commit()
    except Exception:
        pass
    return store, ids


class _ListStore:
    """Standalone-shaped store: only all_facts(), no read(), no write path."""

    def __init__(self, rows):
        self._rows = rows

    def all_facts(self):
        return [dict(row) for row in self._rows]


# ------------------------------------------------------------------- curve
def test_factor_zero_one_and_two_half_lives():
    assert decay.decay_factor(0) == 1.0
    assert decay.decay_factor(180) == pytest.approx(0.5)
    assert decay.decay_factor(90) == pytest.approx(0.5 ** 0.5)
    assert decay.decay_factor(90, half_life_days=90) == pytest.approx(0.5)
    # two half-lives only reaches 0.25 when the floor is out of the way
    assert decay.decay_factor(360, floor=0.0) == pytest.approx(0.25)


def test_factor_clamps_at_floor():
    assert decay.decay_factor(360) == pytest.approx(0.5)
    assert decay.decay_factor(10_000) == pytest.approx(0.5)
    assert decay.decay_factor(360, floor=0.3) == pytest.approx(0.3)
    assert decay.decay_factor(180, floor=0.3) == pytest.approx(0.5)
    assert decay.decay_factor(180, floor=2.0) == 1.0


def test_factor_is_total_for_odd_inputs():
    assert decay.decay_factor(-5) == 1.0                     # future timestamp
    assert decay.decay_factor("not-a-number") == 1.0        # unknown age: no decay
    assert decay.decay_factor(float("nan")) == 1.0
    assert decay.decay_factor(float("inf")) == pytest.approx(0.5)
    assert decay.decay_factor(180, half_life_days=0) == pytest.approx(0.5)


# --------------------------------------------------------------- effective
def test_effective_trust_uses_updated_at_then_created_at():
    old_updated = _fact(trust_score=0.8, updated_at=_ago(180), created_at=_ago(1))
    assert decay.effective_trust(old_updated, now=NOW) == pytest.approx(0.4)

    created_only = _fact(trust_score=0.8, created_at=_ago(180))
    created_only.pop("updated_at")
    assert decay.effective_trust(created_only, now=NOW) == pytest.approx(0.4)


def test_effective_trust_without_timestamps_does_not_decay():
    bare = {"fact_id": 7, "trust_score": 0.8}
    assert decay.effective_trust(bare, now=NOW) == pytest.approx(0.8)
    assert decay.effective_trust({"fact_id": 8}, now=NOW) == 0.0


def test_effective_trust_tolerates_malformed_timestamps():
    garbage = _fact(trust_score=0.8, updated_at="yesterday-ish", created_at=None)
    assert decay.effective_trust(garbage, now=NOW) == pytest.approx(0.8)

    # a broken updated_at falls through to a usable created_at
    fallback = _fact(trust_score=0.8, updated_at="???", created_at=_ago(180))
    assert decay.effective_trust(fallback, now=NOW) == pytest.approx(0.4)

    # epoch seconds and datetime objects are timestamps too
    epoch = _fact(trust_score=0.8, updated_at=NOW.timestamp() - 180 * 86400)
    assert decay.effective_trust(epoch, now=NOW) == pytest.approx(0.4)
    dt = _fact(trust_score=0.8, updated_at=NOW - timedelta(days=180))
    assert decay.effective_trust(dt, now=NOW) == pytest.approx(0.4)


def test_now_injection_changes_only_the_reference_time():
    fact = _fact(trust_score=0.5, updated_at=_ago(0))
    first = decay.effective_trust(fact, now=NOW)
    again = decay.effective_trust(fact, now=NOW.isoformat())
    assert first == pytest.approx(0.5) and again == pytest.approx(first)
    later = decay.effective_trust(fact, now=NOW + timedelta(days=360))
    assert later == pytest.approx(0.25)


# -------------------------------------------------------------------- bands
@pytest.mark.parametrize("trust,age,min_trust,band", [
    (0.50, 0, 0.25, "fresh"),      # 0.500 == 2.0 * min_trust
    (0.375, 0, 0.25, "aging"),     # exact 1.5 * min_trust edge
    (0.40, 0, 0.25, "aging"),
    (0.50, 180, 0.25, "stale"),    # one half-life lands exactly on min_trust
    (0.40, 180, 0.25, "expired"),  # 0.200 < min_trust
    (0.50, 180, 0.30, "expired"),  # same score against a stricter caller
    (0.50, 180, 0.20, "stale"),
])
def test_band_boundaries(trust, age, min_trust, band):
    result = decay.classify_fact(
        _fact(trust_score=trust, updated_at=_ago(age)), now=NOW, min_trust=min_trust)
    assert result["band"] == band


def test_classify_fact_reports_the_documented_fields():
    result = decay.classify_fact(_fact(trust_score=0.6, updated_at=_ago(180)), now=NOW)
    assert result["fact_id"] == 1
    assert result["trust_score"] == pytest.approx(0.6)
    assert result["age_days"] == pytest.approx(180.0)
    assert result["factor"] == pytest.approx(0.5)
    assert result["effective"] == pytest.approx(0.3)
    assert result["band"] == "stale"


# --------------------------------------------------------------- categories
def test_slow_categories_get_double_the_half_life():
    at_180 = {"updated_at": _ago(180), "trust_score": 0.6}
    entity = decay.classify_fact(_fact(category="entity", **at_180), now=NOW)
    system = decay.classify_fact(_fact(category="system", **at_180), now=NOW)
    pref = decay.classify_fact(_fact(category="user_pref", **at_180), now=NOW)
    project = decay.classify_fact(_fact(category="project", **at_180), now=NOW)

    assert entity["factor"] == pytest.approx(0.5)
    assert system["factor"] == pytest.approx(0.5)
    # half rate => 0.5 ** (180 / 360)
    assert pref["factor"] == pytest.approx(0.5 ** 0.5)
    assert project["factor"] == pytest.approx(0.5 ** 0.5)
    assert pref["effective"] > entity["effective"]
    assert project["effective"] > entity["effective"]


def test_category_name_is_normalised():
    spaced = _fact(category="User-Pref", trust_score=0.6, updated_at=_ago(180))
    assert decay.classify_fact(spaced, now=NOW)["factor"] == pytest.approx(0.5 ** 0.5)


def test_unknown_category_uses_the_normal_rate():
    other = _fact(category="tool", trust_score=0.6, updated_at=_ago(180))
    assert decay.classify_fact(other, now=NOW)["factor"] == pytest.approx(0.5)


# ------------------------------------------------------------------ scans
_SCAN_SPECS = [
    {"content": "Synthetic entity alpha runs on port 7000", "category": "entity",
     "trust": 0.6, "age": 180},                                      # 0.300 stale
    {"content": "Synthetic entity beta runs on port 7001", "category": "entity",
     "trust": 0.2, "age": 0},                                        # 0.200 expired
    {"content": "Synthetic preference: dark mode", "category": "user_pref",
     "trust": 0.6, "age": 180},                                      # 0.424 aging
    {"content": "Synthetic project uses pytest", "category": "project",
     "trust": 0.5, "age": 0},                                        # 0.500 fresh
]


def test_scan_counts_and_orders_candidates(tmp_path):
    store, _ = _store(tmp_path, _SCAN_SPECS)
    result = decay.scan(store, now=NOW)

    assert result["scanned"] == 4
    assert result["counts"] == {"fresh": 1, "aging": 1, "stale": 1, "expired": 1}
    contents = [c["content"] for c in result["candidates"]]
    assert contents == [
        "Synthetic entity beta runs on port 7001",
        "Synthetic entity alpha runs on port 7000",
        "Synthetic preference: dark mode",
    ]
    assert [c["effective"] for c in result["candidates"]] == sorted(
        c["effective"] for c in result["candidates"])
    assert set(result["candidates"][0]) == {"fact_id", "content", "effective", "age_days"}


def test_scan_is_read_only(tmp_path):
    store, _ = _store(tmp_path, _SCAN_SPECS)
    before = [dict(r) for r in store.read("SELECT * FROM facts ORDER BY fact_id")]

    decay.scan(store, now=NOW)

    after = [dict(r) for r in store.read("SELECT * FROM facts ORDER BY fact_id")]
    assert after == before
    assert store.count_facts() == len(before) == 4


def test_scan_empty_store(tmp_path):
    store = Store(tmp_path / "empty.db")
    assert decay.scan(store, now=NOW) == {
        "scanned": 0,
        "counts": {"fresh": 0, "aging": 0, "stale": 0, "expired": 0},
        "candidates": [],
    }


def test_scan_respects_limit_without_hiding_the_total(tmp_path):
    specs = [{"content": f"Synthetic stale fact {i}", "category": "entity",
              "trust": 0.2, "age": 200 + i} for i in range(5)]
    store, _ = _store(tmp_path, specs)

    limited = decay.scan(store, now=NOW, limit=2)
    assert limited["scanned"] == 5
    assert sum(limited["counts"].values()) == 5
    assert len(limited["candidates"]) == 2
    assert limited["counts"]["expired"] == 5

    full = decay.scan(store, now=NOW)
    assert len(full["candidates"]) == 5
    assert [c["fact_id"] for c in limited["candidates"]] == [
        c["fact_id"] for c in full["candidates"][:2]]


def test_scan_accepts_a_store_with_only_all_facts():
    rows = [_fact(fact_id=1, trust_score=0.5, updated_at=_ago(180)),
            _fact(fact_id=2, trust_score=0.9, updated_at=_ago(0))]
    result = decay.scan(_ListStore(rows), now=NOW)
    assert result["scanned"] == 2
    assert result["counts"]["stale"] == 1 and result["counts"]["fresh"] == 1


# --------------------------------------------------------------- apply_touch
def test_apply_touch_changes_updated_at_only(tmp_path):
    store, ids = _store(tmp_path, [
        {"content": "Synthetic fact one", "trust": 0.4, "age": 400},
        {"content": "Synthetic fact two", "trust": 0.7, "age": 400},
    ])
    before = {row["fact_id"]: dict(row) for row in
              store.read("SELECT * FROM facts ORDER BY fact_id")}

    result = decay.apply_touch(store, [ids[0], ids[0]], reason="decay-review")

    assert result["ok"] is True
    assert result["touched"] == [ids[0]]          # duplicates collapse
    after = {row["fact_id"]: dict(row) for row in
             store.read("SELECT * FROM facts ORDER BY fact_id")}
    assert set(after) == set(before)
    for fact_id, old in before.items():
        new = after[fact_id]
        if fact_id == ids[0]:
            assert new["updated_at"] != old["updated_at"]
            old.pop("updated_at")
            new.pop("updated_at")
        assert new == old
    audit = store.read("SELECT action, fact_id, reason FROM audit")
    assert [(r["action"], r["fact_id"], r["reason"]) for r in audit] == [
        ("review-touch", ids[0], "decay-review")]


def test_apply_touch_reports_unknown_ids(tmp_path):
    store, ids = _store(tmp_path, [{"content": "Synthetic fact", "age": 10}])
    result = decay.apply_touch(store, [ids[0], 999_999])
    assert result["ok"] is False
    assert result["touched"] == ids
    assert result["failed"] == [{"fact_id": 999_999, "reason": "fact not found"}]
