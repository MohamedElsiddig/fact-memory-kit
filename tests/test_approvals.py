"""Chat approval flow: codes, dry runs, tombstones, audit and idempotency."""
import hashlib
import json

from factkit import approvals, holo, maintenance, policy
from factkit.store import Store

# ------------------------------------------------------------------ fixtures
def _conflict_store(tmp_path, name="conflict.db"):
    """Two facts on one declared subject that disagree on one port.

    The newer row (higher id, newer port) is the keep side, matching how the
    conflict scan nominates a pair.
    """
    store = Store(tmp_path / name)
    store.add_fact_status("Test Node alpha listens on port 7000", category="tool",
                          tags="", trust=0.7, entities=["Test Node alpha"])
    store.add_fact_status("Test Node alpha listens on port 7001", category="tool",
                          tags="", trust=0.8, entities=["Test Node alpha"])
    return store


def _two_conflict_store(tmp_path):
    store = Store(tmp_path / "two.db")
    for text, trust in (
        ("Test Node alpha listens on port 7001", 0.8),
        ("Test Node alpha listens on port 7000", 0.7),
        ("Test Node beta listens on port 8101", 0.8),
        ("Test Node beta listens on port 8100", 0.7),
    ):
        store.add_fact_status(text, category="tool", tags="", trust=trust,
                              entities=[text.split(" listens")[0]])
    return store


def _duplicate_store(tmp_path):
    store = Store(tmp_path / "dupe.db")
    store.add_fact_status("Backup drive alpha holds nightly snapshots", category="tool",
                          tags="", trust=0.6)
    store.add_fact_status("Backup drive alpha holds nightly snapshot", category="tool",
                          tags="", trust=0.9)
    return store


def _propose(store):
    return holo.maintain(store, policy=policy.Policy(auto=False, max_ratio=1.5),
                         dry_run=False)


def _insert(store, proposal_id, *, keep, retire, slots="port", ratio=0.5,
            status="pending", keep_text="keep text", retire_text="retire text",
            batch="b-test"):
    maintenance.ensure_tombstones(store)
    holo._ensure_proposal_columns(store)
    evidence = {"kind": "duplicate" if slots == "duplicate" else "conflict",
                "slot": None if slots == "duplicate" else slots,
                "keep_text": keep_text, "retire_text": retire_text, "ratio": ratio}
    with store._tx():
        store._write(
            "INSERT INTO conflict_proposals"
            "(id, keep_id, retire_id, jaccard, slots, delta, status, note, batch_id, evidence) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (proposal_id, keep, retire, ratio, slots, "port: a vs b", status, "",
             batch, json.dumps(evidence)))


def _audit(store, action):
    try:
        rows = store.read(
            "SELECT action, fact_id, actor, reason FROM audit WHERE action = ? ORDER BY audit_id", (action,))
    except Exception:
        rows = store.read(
            "SELECT action, fact_id, actor, reason FROM audit WHERE action = ? ORDER BY rowid", (action,))
    return [dict(row) for row in rows]


# --------------------------------------------------------------------- reads
def test_pending_lists_codes_and_fields(tmp_path):
    store = _conflict_store(tmp_path)
    _propose(store)

    rows = approvals.pending(store)

    assert len(rows) == 1
    row = rows[0]
    assert {"code", "proposal_id", "kind", "keep_id", "retire_id", "keep_text",
            "retire_text", "ratio", "created_at", "batch_id"} <= set(row)
    assert len(row["code"]) == 4 and set(row["code"]) <= set("0123456789abcdef")
    assert row["code"] == hashlib.sha1(str(row["proposal_id"]).encode()).hexdigest()[:4]
    assert row["kind"] == "conflict"
    assert (row["keep_id"], row["retire_id"]) == (2, 1)
    assert row["keep_text"] == "Test Node alpha listens on port 7001"
    assert row["retire_text"] == "Test Node alpha listens on port 7000"
    assert row["ratio"] > 0
    assert row["created_at"] and row["batch_id"]


def test_pending_truncates_text_to_100(tmp_path):
    store = Store(tmp_path / "long.db")
    _insert(store, 1, keep=10, retire=11, keep_text="k" * 180, retire_text="r" * 180)

    row = approvals.pending(store)[0]

    assert len(row["keep_text"]) == 100
    assert len(row["retire_text"]) == 100


def test_codes_are_stable_across_calls(tmp_path):
    store = _conflict_store(tmp_path)
    _propose(store)

    first = [(row["proposal_id"], row["code"]) for row in approvals.pending(store)]
    second = [(row["proposal_id"], row["code"]) for row in approvals.pending(store)]

    assert first == second


def test_resolve_code_is_case_insensitive(tmp_path):
    store = _conflict_store(tmp_path)
    _propose(store)
    row = approvals.pending(store)[0]

    for typed in (row["code"].upper(), f" #{row['code'].upper()} "):
        resolved = approvals.resolve_code(store, typed)
        assert resolved is not None
        assert resolved["proposal_id"] == row["proposal_id"]
        assert resolved["code"] == row["code"]


def test_unknown_code_resolves_to_none(tmp_path):
    store = _conflict_store(tmp_path)
    _propose(store)

    assert approvals.resolve_code(store, "0000") is None
    assert approvals.resolve_code(store, "not-a-code") is None
    assert approvals.resolve_code(store, "") is None


def test_empty_store_is_total(tmp_path):
    store = Store(tmp_path / "empty.db")

    assert approvals.pending(store) == []
    assert approvals.history(store) == []
    assert "No pending proposals" in approvals.render([])
    result = approvals.decide(store, "abcd", action="accept")
    assert result == {"ok": False, "reason": "unknown code", "code": "abcd", "action": "accept"}


def test_ambiguous_duplicate_code_is_handled(tmp_path):
    store = Store(tmp_path / "amb.db")
    seen, pair = {}, []
    for candidate in range(1, 20000):
        prefix = hashlib.sha1(str(candidate).encode()).hexdigest()[:4]
        if prefix in seen:
            pair = [seen[prefix], candidate]
            break
        seen[prefix] = candidate
    assert pair, "expected a four-hex sha1 collision in the search range"

    for proposal_id in pair:
        _insert(store, proposal_id, keep=100000 + proposal_id, retire=200000 + proposal_id)

    rows = approvals.pending(store)
    assert len({row["code"] for row in rows}) == 2          # listing is unambiguous
    base = hashlib.sha1(str(pair[0]).encode()).hexdigest()[:4]
    assert all(row["code"].startswith(base) for row in rows)
    assert all(len(row["code"]) > 4 for row in rows)
    assert approvals.resolve_code(store, base) is None      # bare prefix is ambiguous
    for row in rows:
        assert approvals.resolve_code(store, row["code"])["proposal_id"] == row["proposal_id"]
    assert approvals.decide(store, base, action="accept")["reason"] == "ambiguous code"


# ----------------------------------------------------------------- decisions
def test_accept_applies_and_tombstones(tmp_path):
    store = _conflict_store(tmp_path)
    _propose(store)
    before = store.count_facts()
    row = approvals.pending(store)[0]

    result = approvals.decide(store, row["code"], action="accept", actor="telegram:test")

    assert result["ok"] is True
    assert result["batch_id"].startswith("chat-")
    assert store.count_facts() == before - 1
    assert store._one("SELECT content FROM facts WHERE fact_id = ?", (row["keep_id"],)) is not None
    assert store._one("SELECT content FROM facts WHERE fact_id = ?", (row["retire_id"],)) is None
    tombstones = maintenance.list_tombstones(store)
    assert [item["fact_id"] for item in tombstones] == [row["retire_id"]]
    assert tombstones[0]["batch_id"] == result["batch_id"]
    proposal = store._one("SELECT status, decided_by, resolved_at, note FROM conflict_proposals")
    assert proposal["status"] == "accepted"
    assert proposal["decided_by"] == "telegram:test"
    assert proposal["resolved_at"]
    assert row["code"] in proposal["note"]


def test_accept_duplicate_uses_merge_path(tmp_path):
    store = _duplicate_store(tmp_path)
    _propose(store)
    before = store.count_facts()
    row = approvals.pending(store)[0]
    assert row["kind"] == "duplicate"

    result = approvals.decide(store, row["code"], action="accept")

    assert result["ok"] is True
    assert store.count_facts() == before - 1
    tombstones = maintenance.list_tombstones(store)
    assert len(tombstones) == 1 and tombstones[0]["reason"] == "duplicate"


def test_reject_leaves_facts_untouched(tmp_path):
    store = _conflict_store(tmp_path)
    _propose(store)
    before = store.count_facts()
    row = approvals.pending(store)[0]

    result = approvals.decide(store, row["code"], action="reject", actor="telegram:test")

    assert result["ok"] is True and result["status"] == "rejected"
    assert store.count_facts() == before
    assert maintenance.list_tombstones(store) == []
    proposal = store._one("SELECT status, decided_by, resolved_at FROM conflict_proposals")
    assert proposal["status"] == "rejected"
    assert proposal["decided_by"] == "telegram:test"
    assert proposal["resolved_at"]
    assert approvals.pending(store) == []


def test_double_accept_is_refused(tmp_path):
    store = _conflict_store(tmp_path)
    _propose(store)
    row = approvals.pending(store)[0]
    assert approvals.decide(store, row["code"], action="accept")["ok"] is True
    after_first = store.count_facts()

    second = approvals.decide(store, row["code"], action="accept")

    assert second["ok"] is False
    assert second["reason"] == "already decided"
    assert store.count_facts() == after_first
    assert len(maintenance.list_tombstones(store)) == 1
    assert len(_audit(store, "proposal-accept")) == 1


def test_dry_run_changes_nothing(tmp_path):
    store = _conflict_store(tmp_path)
    _propose(store)
    before = store.count_facts()
    row = approvals.pending(store)[0]

    result = approvals.decide(store, row["code"], action="accept", dry_run=True)

    assert result["ok"] is True and result["dry_run"] is True
    assert result["result"]["dry_run"] is True
    assert store.count_facts() == before
    assert maintenance.list_tombstones(store) == []
    assert store._one("SELECT status FROM conflict_proposals")["status"] == "pending"
    assert len(approvals.pending(store)) == 1
    assert _audit(store, "proposal-accept") == []
    columns = {row[1] for row in store.read("PRAGMA table_info(conflict_proposals)")}
    assert "decided_by" not in columns       # a dry run does not even migrate


def test_reject_dry_run_changes_nothing(tmp_path):
    store = _conflict_store(tmp_path)
    _propose(store)

    result = approvals.decide(store, approvals.pending(store)[0]["code"],
                              action="reject", dry_run=True)

    assert result["ok"] is True and result["dry_run"] is True
    assert store._one("SELECT status FROM conflict_proposals")["status"] == "pending"
    assert _audit(store, "proposal-reject") == []


def test_audit_rows_written_for_accept_and_reject(tmp_path):
    store = _two_conflict_store(tmp_path)
    _propose(store)
    assert len(approvals.pending(store)) == 2

    accepted = approvals.pending(store)[0]
    approvals.decide(store, accepted["code"], action="accept", actor="telegram")
    rejected = approvals.pending(store)[0]
    approvals.decide(store, rejected["code"], action="reject", actor="telegram")

    for action, decided in (("proposal-accept", accepted), ("proposal-reject", rejected)):
        rows = _audit(store, action)
        assert len(rows) == 1
        assert rows[0]["actor"] == "telegram"
        assert f"code={decided['code']}" in rows[0]["reason"]
        assert f"proposal={decided['proposal_id']}" in rows[0]["reason"]
        assert action.split("-")[1] in rows[0]["reason"]


def test_history_lists_decisions_with_actor_and_time(tmp_path):
    store = _two_conflict_store(tmp_path)
    _propose(store)
    with store._tx():                      # make a decision time distinguishable
        store._write("UPDATE conflict_proposals SET created_at = '2020-01-01 00:00:00'")
    first = approvals.pending(store)[0]
    approvals.decide(store, first["code"], action="accept", actor="telegram")
    second = approvals.pending(store)[0]
    approvals.decide(store, second["code"], action="reject", actor="telegram")

    rows = approvals.history(store)

    assert len(rows) == 2
    by_id = {row["proposal_id"]: row for row in rows}
    assert by_id[first["proposal_id"]]["status"] == "accepted"
    assert by_id[second["proposal_id"]]["status"] == "rejected"
    assert {row["actor"] for row in rows} == {"telegram"}
    assert all(row["decided_at"] for row in rows)
    assert all(row["decided_at"] != "2020-01-01 00:00:00" for row in rows)


def test_history_keeps_legacy_applied_proposals(tmp_path):
    store = _duplicate_store(tmp_path)
    _propose(store)
    with store._tx():
        store._write("UPDATE conflict_proposals SET status = 'applied' WHERE status = 'pending'")

    rows = approvals.history(store)

    assert len(rows) == 1
    assert rows[0]["status"] == "applied"
    assert rows[0]["actor"] == "factkit"
    assert rows[0]["decided_at"]


# -------------------------------------------------------------------- render
def test_render_ten_proposals_stays_under_15_lines(tmp_path):
    store = Store(tmp_path / "render.db")
    for proposal_id in range(1, 11):
        _insert(store, proposal_id, keep=100 + proposal_id, retire=200 + proposal_id,
                keep_text=f"keep text {proposal_id}", retire_text=f"retire text {proposal_id}")

    rows = approvals.pending(store)
    text = approvals.render(rows)
    lines = text.splitlines()

    assert len(rows) == 10
    assert len(lines) <= 15
    assert lines[0].startswith("Pending proposals (10)")
    for row in rows:
        assert any(line.startswith(row["code"]) for line in lines[1:-2])
    assert "approve" in lines[-2] and "reject" in lines[-2]


def test_render_limits_a_long_listing(tmp_path):
    store = Store(tmp_path / "many.db")
    for proposal_id in range(1, 21):
        _insert(store, proposal_id, keep=100 + proposal_id, retire=200 + proposal_id)

    text = approvals.render(approvals.pending(store, limit=20))
    lines = text.splitlines()

    assert len(lines) <= 15
    assert lines[0] == "Pending proposals (12 of 20)"
    assert "first 12" in lines[-1]


def test_render_shows_ids_and_reason():
    text = approvals.render([{
        "code": "ab12", "proposal_id": 7, "kind": "duplicate", "keep_id": 5,
        "retire_id": 3, "ratio": 0.97,
    }])

    assert "ab12" in text
    assert "keep #5" in text and "retire #3" in text
    assert "duplicate 97% similar" in text
