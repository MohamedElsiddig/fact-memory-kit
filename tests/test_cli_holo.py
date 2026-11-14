"""CLI wiring for ``factkit holo``: write routing, guard output, approvals JSON."""
import json
import sqlite3

import pytest

from factkit import approvals, cli, holo, maintenance, policy
from factkit.store import Store


def _conflict_store(tmp_path, name="cli-conflict.db"):
    """Two port facts on one declared entity, timestamped so a policy may apply."""
    store = Store(tmp_path / name)
    store.add_fact_status("Test Node alpha runs on port 7000", category="tool",
                          tags="", trust=0.7, entities=["Test Node alpha"])
    store.add_fact_status("Test Node alpha runs on port 7001", category="tool",
                          tags="", trust=0.8, entities=["Test Node alpha"])
    con = sqlite3.connect(store.db_path)
    con.execute("UPDATE facts SET updated_at=? WHERE fact_id=1", ("2026-10-01 00:00:00",))
    con.execute("UPDATE facts SET updated_at=? WHERE fact_id=2", ("2026-11-04 00:00:00",))
    con.commit()
    con.close()
    return store


def _propose(store):
    return holo.maintain(store, policy=policy.Policy(auto=False, max_ratio=1.5),
                         dry_run=False)


# ------------------------------------------------------------ store routing
@pytest.mark.parametrize("argv, expected", [
    (["holo", "guard", "--text", "x"], False),
    (["holo", "decay"], False),
    (["holo", "metrics"], False),
    (["holo", "asof", "--text", "x", "--as-of", "2026-01-01"], False),
    (["holo", "hrr"], False),
    (["holo", "rerank", "--text", "x"], False),
    (["holo", "expand", "--text", "x"], False),
    (["holo", "hops", "--text", "x"], False),
    (["holo", "ns"], False),
    (["holo", "ns", "--namespace", "work", "--assign-id", "1"], False),
    (["holo", "approve"], False),
    (["holo", "approve", "--code", "1a2b"], True),
    (["holo", "conflicts"], False),
    (["holo", "conflicts", "--apply"], True),
    (["holo", "stale", "--touch", "1"], False),
    (["holo", "stale", "--touch", "1", "--apply"], True),
    (["holo", "dupes", "--apply"], True),
    (["holo", "maintain", "--apply"], True),
    (["holo", "undo", "--fact-id", "1", "--apply"], True),
    (["holo", "purge"], True),
    (["holo", "provenance"], False),
    (["holo", "provenance", "--apply"], True),
    (["holo", "prose"], False),
    (["holo", "prose", "--apply"], True),
    (["holo", "limits"], False),
    (["holo", "limits", "--apply"], True),
    (["holo", "relations"], False),
    (["holo", "relations", "--src", "1", "--dst", "2", "--rel", "uses"], False),
    (["holo", "relations", "--src", "1", "--dst", "2", "--rel", "uses", "--apply"], True),
    (["holo", "ns", "--namespace", "work", "--assign-id", "1", "--apply"], True),
    (["holo", "migrate"], False),
    (["holo", "migrate", "--apply"], True),
    (["holo", "migrate", "--rollback"], True),
])
def test_holo_needs_write(argv, expected):
    args = cli.build_parser().parse_args(argv)

    assert cli._holo_needs_write(args) is expected


def test_maintain_apply_uses_writable_store(tmp_path, monkeypatch):
    store = _conflict_store(tmp_path)
    seen = {}

    def writable(path):
        seen["path"] = path
        return Store(path)

    monkeypatch.setattr(holo, "get_memory_store", writable)
    monkeypatch.setattr(
        holo, "load_plugin_store",
        lambda *a, **k: pytest.fail("mutating action used the read-only store"))

    rc = cli.main(["holo", "maintain", "--db", store.db_path, "--auto", "--apply",
                   "--max-ratio", "1.5", "--limit", "5"])

    assert rc == 0
    assert seen["path"] == store.db_path
    reopened = Store(store.db_path)
    assert reopened.count_facts() == 1
    tombstones = maintenance.list_tombstones(reopened)
    assert len(tombstones) == 1 and tombstones[0]["reason"] == "conflict"


def test_guard_is_read_only_and_never_opens_writable_store(tmp_path, monkeypatch, capsys):
    store = _conflict_store(tmp_path)
    monkeypatch.setattr(
        holo, "get_memory_store",
        lambda *a, **k: pytest.fail("read-only action opened the writable store"))

    rc = cli.main(["holo", "guard", "--db", store.db_path, "--json",
                   "--text", "Test Node alpha runs on port 9999"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["verdict"] == "conflict"
    assert {"verdict", "matches", "suggestion", "human"} <= set(payload)
    assert payload["human"].startswith("CONFLICT")


def test_guard_default_prints_human_report(tmp_path, capsys):
    store = _conflict_store(tmp_path)

    rc = cli.main(["holo", "guard", "--db", store.db_path,
                   "--text", "Test Node alpha runs on port 9999"])

    out = capsys.readouterr().out
    assert rc == 1
    assert out.startswith("CONFLICT")


def test_guard_ok_exits_zero(tmp_path, capsys):
    store = _conflict_store(tmp_path)

    rc = cli.main(["holo", "guard", "--db", store.db_path,
                   "--text", "Unrelated service listens on port 4321"])

    assert rc == 0
    assert capsys.readouterr().out.startswith("OK")


def test_approve_listing_is_read_only_and_json_matches_pending(tmp_path, monkeypatch, capsys):
    store = _conflict_store(tmp_path)
    _propose(store)
    expected = approvals.pending(store)
    monkeypatch.setattr(
        holo, "get_memory_store",
        lambda *a, **k: pytest.fail("listing opened the writable store"))

    rc = cli.main(["holo", "approve", "--db", store.db_path, "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["kind"] is None
    assert payload["pending"] == len(expected) == 1
    assert payload["rows"] == expected


def test_approve_code_decision_uses_writable_store(tmp_path, monkeypatch):
    store = _conflict_store(tmp_path)
    _propose(store)
    code = approvals.pending(store)[0]["code"]
    seen = {}

    def writable(path):
        seen["path"] = path
        return Store(path)

    monkeypatch.setattr(holo, "get_memory_store", writable)
    monkeypatch.setattr(
        holo, "load_plugin_store",
        lambda *a, **k: pytest.fail("a decision used the read-only store"))

    rc = cli.main(["holo", "approve", "--db", store.db_path, "--code", code, "--apply"])

    assert rc == 0
    assert seen["path"] == store.db_path
    reopened = Store(store.db_path)
    assert reopened.count_facts() == 1
    assert len(maintenance.list_tombstones(reopened)) == 1
    assert approvals.pending(reopened) == []


def test_maintain_apply_end_to_end(tmp_path):
    """The real CLI path: writable store, tombstone, retirement, no monkeypatch."""
    path = tmp_path / "provider-shaped.db"
    store = holo.StandaloneStore(path)
    store.add_fact_status("Test Node alpha runs on port 7000", category="tool",
                          tags="", trust=0.7, entities=["Test Node alpha"])
    store.add_fact_status("Test Node alpha runs on port 7001", category="tool",
                          tags="", trust=0.8, entities=["Test Node alpha"])
    con = sqlite3.connect(path)
    con.execute("UPDATE facts SET updated_at=? WHERE fact_id=1", ("2026-10-01 00:00:00",))
    con.execute("UPDATE facts SET updated_at=? WHERE fact_id=2", ("2026-11-04 00:00:00",))
    con.commit()
    con.close()
    store.close()

    rc = cli.main(["holo", "maintain", "--db", str(path), "--auto", "--apply",
                   "--max-ratio", "1.5", "--limit", "5"])

    assert rc == 0
    reopened = holo.StandaloneStore(path)
    assert reopened.count_facts() == 1
    tombstones = maintenance.list_tombstones(reopened)
    assert len(tombstones) == 1 and tombstones[0]["reason"] == "conflict"
    reopened.close()
