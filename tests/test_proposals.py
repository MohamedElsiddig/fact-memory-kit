"""Proposals carry their batch id and evidence; old databases migrate additively."""
import json
import sqlite3

from factkit import holo, maintenance, policy
from factkit.store import Store


def _store(tmp_path, *, older="2026-10-01 00:00:00", newer="2026-11-04 00:00:00",
           old_port="7000", new_port="7001"):
    store = Store(tmp_path / "m.db")
    store.add_fact_status(f"Test Node alpha runs on port {old_port}", category="tool",
                          tags="", trust=0.7, entities=["Test Node alpha"])
    store.add_fact_status(f"Test Node alpha runs on port {new_port}", category="tool",
                          tags="", trust=0.8, entities=["Test Node alpha"])
    con = sqlite3.connect(store.db_path)
    con.execute("UPDATE facts SET updated_at=? WHERE fact_id=1", (older,))
    con.execute("UPDATE facts SET updated_at=? WHERE fact_id=2", (newer,))
    con.commit(); con.close()
    return store


def _columns(store):
    return {r[1] for r in store._conn.execute("PRAGMA table_info(conflict_proposals)")}


def _pair(**kw):
    pair = {"keep": 2, "retire": 1, "jaccard": 0.5,
            "keep_text": "Test Node alpha runs on port 7001",
            "retire_text": "Test Node alpha runs on port 7000",
            "diffs": [{"slot": "port", "keep": ["7001"], "retire": ["7000"]}]}
    pair.update(kw)
    return pair


def _policy(**kw):
    return policy.Policy(auto=True, max_ratio=1.5, **kw)


def test_columns_created_on_a_bare_store(tmp_path):
    store = Store(tmp_path / "bare.db")
    store.db.execute("DROP TABLE IF EXISTS conflict_proposals")
    store.db.commit()
    assert "conflict_proposals" not in {r[0] for r in store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}

    holo._record_proposal(store, _pair(), "pending", "b-bare")

    assert {"batch_id", "evidence"} <= _columns(store)
    row = store._one("SELECT * FROM conflict_proposals")
    assert row["batch_id"] == "b-bare"


def test_old_table_is_migrated_without_losing_rows(tmp_path):
    store = Store(tmp_path / "old.db")
    maintenance.ensure_tombstones(store)          # legacy shape, no new columns
    with store._tx():
        store._write(
            "INSERT INTO conflict_proposals"
            "(keep_id, retire_id, jaccard, slots, delta, status, note) VALUES (?,?,?,?,?,?,?)",
            (10, 11, 0.25, "port", "port: a vs b", "pending", "legacy"))

    holo._record_proposal(store, _pair(), "pending", "b-new")

    rows = {r["keep_id"]: r for r in store.read("SELECT * FROM conflict_proposals")}
    assert set(rows) == {2, 10}
    assert rows[10]["note"] == "legacy" and rows[10]["batch_id"] is None
    assert rows[2]["batch_id"] == "b-new"


def test_pending_proposal_carries_batch_and_evidence(tmp_path):
    store = _store(tmp_path)
    result = holo.maintain(store, policy=policy.Policy(auto=False, max_ratio=1.5),
                           dry_run=False)

    row = store._one("SELECT * FROM conflict_proposals")
    assert row["status"] == "pending"
    assert row["batch_id"] == result["batch"]
    assert row["note"] == f"batch={result['batch']}"
    evidence = json.loads(row["evidence"])
    assert evidence["kind"] == "conflict"
    assert evidence["slot"] == "port"
    assert evidence["keep_text"] == "Test Node alpha runs on port 7001"
    assert evidence["retire_text"] == "Test Node alpha runs on port 7000"
    assert evidence["ratio"] == row["jaccard"] > 0


def test_applied_proposal_is_recorded_as_applied(tmp_path):
    store = _store(tmp_path)
    result = holo.maintain(store, policy=_policy(), dry_run=False)

    assert result["applied"] == 1
    row = store._one("SELECT * FROM conflict_proposals")
    assert row["status"] == "applied"
    assert row["batch_id"] == result["batch"]
    assert json.loads(row["evidence"])["kind"] == "conflict"


def test_repeating_the_same_pair_does_not_duplicate(tmp_path):
    store = _store(tmp_path)
    first = holo.maintain(store, policy=policy.Policy(auto=False, max_ratio=1.5), dry_run=False)
    second = holo.maintain(store, policy=policy.Policy(auto=False, max_ratio=1.5), dry_run=False)

    rows = store.read("SELECT keep_id, retire_id, status, batch_id FROM conflict_proposals")
    assert len(rows) == 1
    assert (rows[0]["keep_id"], rows[0]["retire_id"]) == (2, 1)
    assert rows[0]["status"] == "pending"
    assert rows[0]["batch_id"] == first["batch"] != second["batch"]


def test_duplicate_evidence_is_valid_json(tmp_path):
    store = Store(tmp_path / "dupe.db")
    store.add_fact_status("Backup drive alpha holds nightly snapshots", category="tool",
                          tags="", trust=0.6, entities=[])
    store.add_fact_status("Backup drive alpha holds nightly snapshot", category="tool",
                          tags="", trust=0.9, entities=[])

    result = holo.maintain(store, policy=policy.Policy(auto=False, max_ratio=1.5),
                           dry_run=False)

    assert result["proposed"] == 1
    row = store._one("SELECT * FROM conflict_proposals")
    evidence = json.loads(row["evidence"])
    assert evidence["kind"] == "duplicate"
    assert evidence["slot"] is None
    assert evidence["ratio"] > 0.9
    assert evidence["keep_text"] and evidence["retire_text"]
    assert row["batch_id"] == result["batch"]
