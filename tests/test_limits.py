"""Store size limits, archival and vacuum tests.

Every fixture is a private SQLite file under ``tmp_path``; the live
``~/.hermes/memory_store.db`` is never opened or written.  Content is generic
and synthetic.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from factkit import limits

SCHEMA = """
CREATE TABLE facts(
    fact_id INTEGER PRIMARY KEY,
    content TEXT NOT NULL UNIQUE,
    category TEXT DEFAULT 'general',
    tags TEXT DEFAULT '',
    trust_score REAL DEFAULT 0.5,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    hrr_vector BLOB
);
CREATE TABLE entities(entity_id INTEGER PRIMARY KEY, name TEXT,
                      entity_type TEXT DEFAULT 'declared');
CREATE TABLE fact_entities(fact_id INTEGER, entity_id INTEGER);
CREATE VIRTUAL TABLE facts_fts USING fts5(content, tags);
CREATE TRIGGER facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags) VALUES (new.fact_id, new.content, new.tags);
END;
CREATE TRIGGER facts_ad AFTER DELETE ON facts BEGIN
    DELETE FROM facts_fts WHERE rowid = old.fact_id;
END;
CREATE TABLE fact_embeddings(
    fact_id INTEGER, model TEXT, dims INTEGER, vec BLOB,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(fact_id, model)
);
CREATE TABLE audit(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    action TEXT, fact_id INTEGER, actor TEXT, reason TEXT, before TEXT, after TEXT
);
CREATE TABLE fact_tombstones(
    fact_id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT,
    tags TEXT,
    trust_score REAL,
    entities TEXT,
    hrr_vector BLOB,
    retired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    retired_by TEXT,
    reason TEXT,
    batch_id TEXT,
    restored_at TIMESTAMP
);
CREATE TABLE conflict_proposals(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    keep_id INTEGER NOT NULL,
    retire_id INTEGER NOT NULL,
    jaccard REAL NOT NULL,
    slots TEXT NOT NULL,
    delta TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    resolved_at TIMESTAMP,
    note TEXT DEFAULT ''
);
CREATE TABLE fact_provenance(fact_id INTEGER, origin TEXT);
CREATE TABLE ttl_flags(fact_id INTEGER, reason TEXT);
"""


# ------------------------------------------------------------------ fixtures
def make_db(tmp_path, name="store.db"):
    """Create a full fixture store and return ``(path, connection)``."""
    path = tmp_path / name
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return path, conn


def seed(conn, count, *, category="user_pref", age_days=400, prefix="fixture"):
    """Insert *count* distinguishable facts and return their ids."""
    ids = []
    for i in range(count):
        cur = conn.execute(
            "INSERT INTO facts(content, category, tags, trust_score, created_at, updated_at) "
            "VALUES (?,?,?,?,datetime('now',?),datetime('now',?))",
            (f"{prefix} {category} note {i}", category, "fixture", 0.7,
             f"-{age_days} days", f"-{age_days} days"))
        ids.append(int(cur.lastrowid))
    return ids


def add_tombstone(conn, fact_id, *, age_days=400, restored=False):
    conn.execute(
        "INSERT INTO fact_tombstones"
        "(fact_id, content, category, retired_at, restored_at) "
        "VALUES (?,?,?,datetime('now',?),?)",
        (fact_id, f"retired fixture {fact_id}", "user_pref",
         f"-{age_days} days", None))
    if restored:
        conn.execute("UPDATE fact_tombstones SET restored_at = datetime('now', '-1 day') "
                     "WHERE fact_id = ?", (fact_id,))


def add_audit(conn, age_days, *, reason="fixture"):
    conn.execute(
        "INSERT INTO audit(at, action, fact_id, actor, reason) "
        "VALUES (datetime('now', ?), 'test', 1, 'tester', ?)",
        (f"-{age_days} days", reason))


def kinds(plan_result):
    return [action["kind"] for action in plan_result["actions"]]


def fact_count(conn):
    return int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])


# ------------------------------------------------------------------- inspect
def test_inspect_bare_store(tmp_path):
    path = tmp_path / "bare.db"
    sqlite3.connect(str(path)).close()  # zero-byte but valid empty database

    snap = limits.inspect(path)

    assert snap["db_bytes"] == 0
    assert snap["wal_bytes"] == 0
    assert snap["facts"] == snap["tombstones"] == 0
    assert snap["audit_rows"] == snap["proposals"] == snap["embeddings"] == 0
    assert snap["freelist_bytes"] == 0
    assert snap["page_count"] == 0
    assert snap["page_size"] > 0


def test_inspect_populated_store(tmp_path):
    path, conn = make_db(tmp_path)
    seed(conn, 3)
    add_tombstone(conn, 900, age_days=400)
    add_tombstone(conn, 901, age_days=10)
    add_audit(conn, 400)
    add_audit(conn, 30)
    add_audit(conn, 1)
    conn.execute("INSERT INTO conflict_proposals(keep_id, retire_id, jaccard, slots, delta) "
                 "VALUES (1, 2, 0.5, 'port', 'x')")
    conn.execute("INSERT INTO fact_embeddings(fact_id, model, dims, vec) VALUES (1,'m',2,x'0000')")
    conn.execute("INSERT INTO fact_embeddings(fact_id, model, dims, vec) VALUES (2,'m',2,x'0000')")
    try:
        snap = limits.inspect(path)
        freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    finally:
        conn.close()

    assert snap["facts"] == 3
    assert snap["tombstones"] == 2
    assert snap["audit_rows"] == 3
    assert snap["proposals"] == 1
    assert snap["embeddings"] == 2
    assert snap["db_bytes"] == path.stat().st_size
    assert snap["page_size"] == page_size
    assert snap["page_count"] == page_count
    assert snap["freelist_bytes"] == freelist * page_size


def test_inspect_tolerates_missing_tables(tmp_path):
    path = tmp_path / "partial.db"
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("CREATE TABLE facts(fact_id INTEGER PRIMARY KEY, content TEXT, "
                 "category TEXT, updated_at TEXT)")
    conn.execute("INSERT INTO facts(content, category, updated_at) "
                 "VALUES ('only a facts table', 'general', datetime('now'))")
    conn.close()

    snap = limits.inspect(path)

    assert snap["facts"] == 1
    assert snap["tombstones"] == 0
    assert snap["audit_rows"] == 0
    assert snap["proposals"] == 0
    assert snap["embeddings"] == 0
    assert snap["freelist_bytes"] == 0
    assert isinstance(snap["db_bytes"], int)


# ---------------------------------------------------------------------- plan
def test_plan_on_empty_store(tmp_path):
    path = tmp_path / "empty.db"
    sqlite3.connect(str(path)).close()

    result = limits.plan(path)

    assert result["actions"] == []
    assert result["blocked"] == []
    assert result["estimated_bytes_reclaimed"] == 0


def test_plan_proposes_purge_only_above_limit(tmp_path):
    path, conn = make_db(tmp_path)
    seed(conn, 1)
    add_tombstone(conn, 900, age_days=400)
    add_tombstone(conn, 901, age_days=10)
    conn.close()

    over = limits.plan(path, max_tombstones_days=180)
    purge = [a for a in over["actions"] if a["kind"] == "purge_tombstones"]
    assert len(purge) == 1
    assert purge[0]["count"] == 1

    under = limits.plan(path, max_tombstones_days=9999)
    assert "purge_tombstones" not in kinds(under)


def test_plan_proposes_archive_only_above_max_facts(tmp_path):
    path, conn = make_db(tmp_path)
    seed(conn, 12, category="user_pref", age_days=400)
    conn.close()

    over = limits.plan(path, max_facts=10)
    archive = [a for a in over["actions"] if a["kind"] == "archive_facts"]
    assert len(archive) == 1
    assert archive[0]["count"] == 2  # 12 facts, limit 10

    under = limits.plan(path, max_facts=5000)
    assert "archive_facts" not in kinds(under)


def test_plan_never_writes(tmp_path):
    path, conn = make_db(tmp_path)
    seed(conn, 5)
    add_tombstone(conn, 900, age_days=400)
    add_audit(conn, 400)
    before_mtime = path.stat().st_mtime_ns
    before = (fact_count(conn),
              conn.execute("SELECT COUNT(*) FROM fact_tombstones").fetchone()[0],
              conn.execute("SELECT COUNT(*) FROM audit").fetchone()[0])

    result = limits.plan(path, max_tombstones_days=180, max_audit_days=365)

    after = (fact_count(conn),
             conn.execute("SELECT COUNT(*) FROM fact_tombstones").fetchone()[0],
             conn.execute("SELECT COUNT(*) FROM audit").fetchone()[0])
    conn.close()
    assert result["actions"]
    assert after == before
    assert path.stat().st_mtime_ns == before_mtime


def test_estimated_bytes_reclaimed_is_non_negative_int(tmp_path):
    path, conn = make_db(tmp_path)
    seed(conn, 4)
    add_tombstone(conn, 900, age_days=400)
    conn.close()

    for result in (limits.plan(path), limits.plan(path, max_facts=1)):
        value = result["estimated_bytes_reclaimed"]
        assert isinstance(value, int) and not isinstance(value, bool)
        assert value >= 0


def test_plan_reports_vacuum_blocked_on_low_disk(tmp_path, monkeypatch):
    path, conn = make_db(tmp_path)
    seed(conn, 200)
    conn.execute("DELETE FROM facts WHERE fact_id <= 150")
    snap = limits.inspect(path)
    conn.close()
    assert snap["freelist_bytes"] > 0

    monkeypatch.setattr(limits.shutil, "disk_usage",
                        lambda _path: SimpleNamespace(free=1))
    result = limits.plan(path)

    assert "vacuum" not in kinds(result)
    assert any(entry["kind"] == "vacuum" for entry in result["blocked"])


# ------------------------------------------------------------------- archive
def test_archive_dry_run_deletes_nothing(tmp_path):
    path, conn = make_db(tmp_path)
    ids = seed(conn, 8, category="user_pref", age_days=400)
    seed(conn, 2, category="project", age_days=400)
    conn.close()

    result = limits.archive_facts(path, backup_dir=tmp_path / "archive")

    assert result["ok"] is True and result["dry_run"] is True
    assert result["deleted"] == 0
    assert result["path"] is None
    # default floor of 5 protects the five newest; three are candidates
    assert sorted(result["candidates"]) == sorted(ids[:3])
    assert not (tmp_path / "archive").exists()

    conn = sqlite3.connect(str(path))
    try:
        assert fact_count(conn) == 10
    finally:
        conn.close()


def test_archive_writes_jsonl_before_delete_and_verifies(tmp_path):
    path, conn = make_db(tmp_path)
    old_ids = seed(conn, 2, category="user_pref", age_days=400, prefix="old")
    fresh_id = seed(conn, 1, category="user_pref", age_days=5, prefix="fresh")[0]
    seed(conn, 2, category="project", age_days=400, prefix="project")
    contents = {int(row["fact_id"]): row["content"] for row in
                conn.execute("SELECT fact_id, content FROM facts")}
    conn.close()

    result = limits.archive_facts(path, backup_dir=tmp_path / "archive", dry_run=False,
                                  keep_min_per_category=0)

    assert result["ok"] is True
    assert result["deleted"] == 2
    assert result["exported"] == 2
    archive = Path(result["path"])
    assert archive.exists()
    lines = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert {row["fact_id"] for row in lines} == set(old_ids)
    for row in lines:
        assert row["sha256"] == hashlib.sha256(contents[row["fact_id"]].encode()).hexdigest()

    conn = sqlite3.connect(str(path))
    try:
        remaining = {int(row[0]) for row in conn.execute("SELECT fact_id FROM facts")}
    finally:
        conn.close()
    assert old_ids[0] not in remaining and old_ids[1] not in remaining
    assert fresh_id in remaining
    assert len(remaining) == 3


def test_archive_refuses_unwritable_target(tmp_path):
    path, conn = make_db(tmp_path)
    seed(conn, 2, category="user_pref", age_days=400)
    conn.close()
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("occupied", encoding="utf-8")

    result = limits.archive_facts(path, backup_dir=blocker, dry_run=False,
                                  keep_min_per_category=0)

    assert result["ok"] is False
    assert "refused" in result
    assert result["deleted"] == 0
    conn = sqlite3.connect(str(path))
    try:
        assert fact_count(conn) == 2
    finally:
        conn.close()
    assert list(tmp_path.rglob("*.jsonl")) == []


def test_archive_refuses_when_verification_fails(tmp_path, monkeypatch):
    path, conn = make_db(tmp_path)
    ids = seed(conn, 2, category="user_pref", age_days=400)
    conn.close()
    monkeypatch.setattr(limits, "_read_archive",
                        lambda _path: {fact_id: "0" * 64 for fact_id in ids})

    result = limits.archive_facts(path, backup_dir=tmp_path / "archive", dry_run=False,
                                  keep_min_per_category=0)

    assert result["ok"] is False
    assert "verification" in result["refused"]
    conn = sqlite3.connect(str(path))
    try:
        assert fact_count(conn) == 2
    finally:
        conn.close()


def test_archive_respects_keep_min_per_category(tmp_path):
    path, conn = make_db(tmp_path)
    ids = seed(conn, 7, category="user_pref", age_days=400)
    seed(conn, 3, category="project", age_days=400)
    conn.close()

    result = limits.archive_facts(path, backup_dir=tmp_path / "archive", dry_run=False,
                                  keep_min_per_category=5)

    assert result["ok"] is True
    assert result["deleted"] == 2
    assert sorted(result["candidates"]) == [ids[0], ids[1]]
    assert result["kept"]["user_pref"] == 5
    assert "project" not in result["kept"]  # only selected categories are reported

    conn = sqlite3.connect(str(path))
    try:
        remaining = sorted(int(row[0]) for row in conn.execute(
            "SELECT fact_id FROM facts WHERE category = 'user_pref'"))
        project_count = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE category = 'project'").fetchone()[0]
    finally:
        conn.close()
    assert remaining == ids[2:]
    assert project_count == 3


# --------------------------------------------------------------------- apply
def test_apply_requires_confirm_token(tmp_path):
    path, conn = make_db(tmp_path)
    seed(conn, 1)
    add_tombstone(conn, 900, age_days=400)
    conn.close()
    result = limits.plan(path, max_tombstones_days=180)

    for token in (None, "", "APPLY-LIMITS", "apply"):
        with pytest.raises(PermissionError):
            limits.apply(path, result, confirm_token=token)

    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("SELECT COUNT(*) FROM fact_tombstones").fetchone()[0] == 1
    finally:
        conn.close()


def test_apply_refuses_archive_actions(tmp_path):
    path, conn = make_db(tmp_path)
    seed(conn, 2, category="user_pref", age_days=400)
    add_tombstone(conn, 900, age_days=400)
    conn.close()
    plan_result = {
        "actions": [
            {"kind": "purge_tombstones", "count": 1, "detail": ""},
            {"kind": "archive_facts", "count": 2, "detail": ""},
        ],
        "params": {"max_tombstones_days": 180, "max_audit_days": 365},
    }

    result = limits.apply(path, plan_result, confirm_token="apply-limits")

    assert result["ok"] is False
    assert result["refused"] == "archive_facts"
    assert result["executed"] == []
    conn = sqlite3.connect(str(path))
    try:
        assert fact_count(conn) == 2
        # the purge in the same plan must not run either: refusal is atomic
        assert conn.execute("SELECT COUNT(*) FROM fact_tombstones").fetchone()[0] == 1
    finally:
        conn.close()


def test_apply_prunes_audit_window_and_keeps_newer_rows(tmp_path):
    path, conn = make_db(tmp_path)
    seed(conn, 1)
    add_audit(conn, 60, reason="old-a")
    add_audit(conn, 90, reason="old-b")
    add_audit(conn, 2, reason="new-a")
    add_audit(conn, 1, reason="new-b")
    conn.close()

    planned = limits.plan(path, max_audit_days=30)
    prune = [a for a in planned["actions"] if a["kind"] == "prune_audit"]
    assert prune and prune[0]["count"] == 2

    result = limits.apply(path, planned, confirm_token="apply-limits")

    assert result["ok"] is True
    assert result["counts"]["prune_audit"] == 2
    conn = sqlite3.connect(str(path))
    try:
        reasons = {row[0] for row in conn.execute("SELECT reason FROM audit")}
    finally:
        conn.close()
    assert reasons == {"new-a", "new-b"}


def test_apply_purges_only_old_tombstones(tmp_path):
    path, conn = make_db(tmp_path)
    add_tombstone(conn, 900, age_days=400)
    add_tombstone(conn, 901, age_days=5)
    conn.close()

    planned = limits.plan(path, max_tombstones_days=180)
    result = limits.apply(path, planned, confirm_token="apply-limits")

    assert result["ok"] is True
    assert result["counts"]["purge_tombstones"] == 1
    conn = sqlite3.connect(str(path))
    try:
        remaining = [int(row[0]) for row in conn.execute(
            "SELECT fact_id FROM fact_tombstones")]
    finally:
        conn.close()
    assert remaining == [901]


def test_apply_vacuum_blocked_on_low_disk(tmp_path, monkeypatch):
    path, conn = make_db(tmp_path)
    seed(conn, 200)
    conn.execute("DELETE FROM facts WHERE fact_id <= 150")
    conn.close()
    assert limits.inspect(path)["freelist_bytes"] > 0

    monkeypatch.setattr(limits.shutil, "disk_usage",
                        lambda _path: SimpleNamespace(free=1))
    planned = limits.plan(path)
    assert "vacuum" not in kinds(planned)
    assert any(entry["kind"] == "vacuum" for entry in planned["blocked"])

    forced = {"actions": [{"kind": "vacuum", "count": 1, "detail": ""}], "params": {}}
    result = limits.apply(path, forced, confirm_token="apply-limits")

    assert result["ok"] is False
    assert result["refused"] == "insufficient_disk_for_vacuum"
    assert limits.inspect(path)["freelist_bytes"] > 0


def test_apply_vacuum_reclaims_freelist(tmp_path):
    path, conn = make_db(tmp_path)
    seed(conn, 300)
    conn.execute("DELETE FROM facts WHERE fact_id <= 250")
    conn.close()
    before = limits.inspect(path)
    assert before["freelist_bytes"] > 0

    planned = limits.plan(path)
    assert "vacuum" in kinds(planned)
    result = limits.apply(path, planned, confirm_token="apply-limits")

    after = limits.inspect(path)
    assert result["ok"] is True
    assert result["vacuum"] in ("full", "incremental")
    assert after["freelist_bytes"] == 0
    assert after["facts"] == 50
    assert after["db_bytes"] <= before["db_bytes"]


def test_apply_vacuum_checkpoints_wal(tmp_path):
    path, conn = make_db(tmp_path)
    conn.execute("PRAGMA journal_mode=WAL")
    seed(conn, 200)
    conn.execute("DELETE FROM facts WHERE fact_id <= 180")
    conn.close()

    planned = limits.plan(path)
    result = limits.apply(path, planned, confirm_token="apply-limits")

    conn = sqlite3.connect(str(path))
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
    finally:
        conn.close()
    assert result["vacuum"] in ("full", "incremental")
    assert mode == "wal"
    assert freelist == 0


# ------------------------------------------------- live-shaped schema variants
def test_live_shaped_tables_use_ts_and_tombstoned_facts(tmp_path):
    path = tmp_path / "live_shape.db"
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.executescript("""
        CREATE TABLE facts(
            fact_id INTEGER PRIMARY KEY, content TEXT, category TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE audit(
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT, ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            action TEXT, fact_id INTEGER, actor TEXT, reason TEXT
        );
        CREATE TABLE tombstoned_facts(
            fact_id INTEGER PRIMARY KEY, content TEXT,
            retired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute("INSERT INTO facts(content, category, updated_at) "
                 "VALUES ('live shape fact', 'general', datetime('now'))")
    conn.execute("INSERT INTO tombstoned_facts(fact_id, content, retired_at) "
                 "VALUES (1, 'old tombstone', datetime('now','-400 days'))")
    conn.execute("INSERT INTO tombstoned_facts(fact_id, content, retired_at) "
                 "VALUES (2, 'fresh tombstone', datetime('now','-1 day'))")
    conn.execute("INSERT INTO audit(ts, action, reason) "
                 "VALUES (datetime('now','-90 days'), 'test', 'old')")
    conn.execute("INSERT INTO audit(ts, action, reason) "
                 "VALUES (datetime('now','-1 day'), 'test', 'new')")
    conn.close()

    snap = limits.inspect(path)
    assert snap["tombstones"] == 2
    assert snap["audit_rows"] == 2

    planned = limits.plan(path, max_tombstones_days=180, max_audit_days=30)
    assert any(a["kind"] == "purge_tombstones" and a["count"] == 1
               for a in planned["actions"])
    assert any(a["kind"] == "prune_audit" and a["count"] == 1
               for a in planned["actions"])

    result = limits.apply(path, planned, confirm_token="apply-limits")
    assert result["ok"] is True
    conn = sqlite3.connect(str(path))
    try:
        assert [int(r[0]) for r in conn.execute(
            "SELECT fact_id FROM tombstoned_facts")] == [2]
        assert [r[0] for r in conn.execute("SELECT reason FROM audit")] == ["new"]
    finally:
        conn.close()
