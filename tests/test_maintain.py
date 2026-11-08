"""End-to-end tests for the maintenance batch (apply, undo, circuit breakers)."""
import sqlite3

import pytest

from factkit import holo, maintenance, policy, smoke
from factkit.store import Store


def _store(tmp_path, *, older="2026-10-01 00:00:00", newer="2026-11-04 00:00:00",
           old_port="7000", new_port="7001", n_extra=0):
    store = Store(tmp_path / "m.db")
    store.add_fact_status(f"Test Node alpha runs on port {old_port}", category="tool",
                          tags="", trust=0.7, entities=["Test Node alpha"])
    store.add_fact_status(f"Test Node alpha runs on port {new_port}", category="tool",
                          tags="", trust=0.8, entities=["Test Node alpha"])
    for i in range(n_extra):
        store.add_fact_status(f"Filler fact number {i}", category="tool", tags="",
                              trust=0.5, entities=[])
    con = sqlite3.connect(store.db_path)
    con.execute("UPDATE facts SET updated_at=? WHERE fact_id=1", (older,))
    con.execute("UPDATE facts SET updated_at=? WHERE fact_id=2", (newer,))
    con.commit(); con.close()
    return store


def _policy(**kw):
    return policy.Policy(auto=True, max_ratio=1.5, **kw)


def test_dry_run_reports_but_writes_nothing(tmp_path):
    store = _store(tmp_path)
    result = holo.maintain(store, policy=_policy(), dry_run=True)
    assert result["auto"] == 1 and result["applied"] == 0
    assert store.count_facts() == 2
    assert maintenance.list_tombstones(store) == []


def test_apply_retires_and_tombstones(tmp_path):
    store = _store(tmp_path)
    result = holo.maintain(store, policy=_policy(), dry_run=False)
    assert result["applied"] == 1
    assert store.count_facts() == 1
    rows = maintenance.list_tombstones(store)
    assert len(rows) == 1 and rows[0]["reason"] == "conflict"
    assert rows[0]["batch_id"] == result["batch"]


def test_undo_restores_the_retired_fact(tmp_path):
    store = _store(tmp_path)
    holo.maintain(store, policy=_policy(), dry_run=False)
    restored = maintenance.restore_fact(store, 1)
    assert restored["content"] == "Test Node alpha runs on port 7000"
    assert store.count_facts() == 2


def test_ratio_abort_leaves_store_untouched(tmp_path):
    store = _store(tmp_path)          # 1 candidate in a 2-fact store
    result = holo.maintain(store, policy=policy.Policy(auto=True, max_ratio=0.2), dry_run=False)
    assert result["aborted"] and result["applied"] == 0
    assert store.count_facts() == 2


def test_max_apply_caps_the_batch(tmp_path):
    store = _store(tmp_path, n_extra=8)
    result = holo.maintain(store, policy=_policy(max_apply=1), dry_run=False)
    assert result["applied"] == 1
    assert store.count_facts() == 9          # exactly one retirement


def test_smoke_regression_reverts_the_batch(tmp_path, monkeypatch):
    store = _store(tmp_path)
    calls = {"n": 0}

    def fake_smoke(store, cases, **kw):
        calls["n"] += 1
        # first run (before) passes, second run (after) regresses
        return {"ok": calls["n"] == 1, "checked": 1, "failures": [] if calls["n"] == 1 else [{"query": "q"}]}

    monkeypatch.setattr(smoke, "run_smoke", fake_smoke)
    monkeypatch.setattr(smoke, "compare", lambda before, after: {"ok": after["ok"], "regressed": ["q"]})
    result = holo.maintain(store, policy=_policy(), dry_run=False,
                           smoke_cases=[{"query": "q", "expect": [1]}])
    assert result["reverted"] is True
    assert store.count_facts() == 2          # restored
    assert result["applied"] == 1


def test_auto_off_only_proposes(tmp_path):
    store = _store(tmp_path)
    result = holo.maintain(store, policy=policy.Policy(auto=False, max_ratio=1.5), dry_run=False)
    assert result["applied"] == 0
    assert result["proposed"] >= 1
    assert store.count_facts() == 2
    pending = store.read("SELECT status FROM conflict_proposals")
    assert any(r["status"] == "pending" for r in pending)
