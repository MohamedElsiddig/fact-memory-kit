"""Store size limits: inspection, planning, archival and vacuum.

The live holographic store grows in three places that never shrink on their
own: retired facts, tombstones and the audit table.  This module answers three
questions without guessing:

* :func:`inspect` -- how big is the store, and where did the rows go?
* :func:`plan` -- what would a cleanup do, and how much would it reclaim?
  Planning is strictly read-only.
* :func:`archive_facts` / :func:`apply` -- carry the plan out, behind an
  explicit confirmation token.

Two safety rules shape the design:

* Facts are only ever deleted by :func:`archive_facts`, and only after the
  matching rows have been written to a JSONL file and read back to verify the
  ids and content hashes.  :func:`apply` refuses a plan containing an archive
  action, so an unattended run cannot drop facts.
* ``VACUUM`` rewrites the whole database file, so it needs free disk of at
  least **twice** the current database size (the old copy plus the new one).
  That is checked with :func:`shutil.disk_usage` before the operation and the
  operation is blocked when the space is not there.  On a WAL database the WAL
  is checkpointed first so the vacuum sees every committed page; when
  ``auto_vacuum=INCREMENTAL`` is set, ``PRAGMA incremental_vacuum`` is used
  instead of a full rewrite.

The module is stdlib-only.  ``store`` may be a path or any object exposing a
``db_path`` attribute, matching the rest of factkit.  All SQL is defensive:
older stores may lack ``fact_tombstones`` (some spell it ``tombstoned_facts``)
or the ``audit`` table, and the live ``audit`` table uses ``ts`` where
factkit-created stores use ``at``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "APPLY_TOKEN",
    "inspect",
    "plan",
    "archive_facts",
    "apply",
]

#: The only value of ``confirm_token`` that lets :func:`apply` write anything.
APPLY_TOKEN = "apply-limits"

#: Default archive location; ``FACTKIT_ARCHIVE_DIR`` overrides it.
_ARCHIVE_DIR = Path.home() / ".hermes" / "backups" / "archive"

#: Tombstone tables seen in the wild.  factkit's maintenance module creates the
#: first; some older stores use the second.
_TOMBSTONE_TABLES = ("fact_tombstones", "tombstoned_facts")

#: Candidate column names for "when did this row happen", newest schema last.
_TIME_COLUMNS = ("at", "ts", "created_at", "timestamp", "retired_at")

#: Related tables that carry a ``fact_id`` and would be orphaned by a delete.
_FACT_CHILD_TABLES = ("fact_entities", "fact_embeddings", "fact_provenance",
                      "ttl_flags", "fact_proposals")

#: Rough per-row overhead (headers, indexes, page slack) for estimates.
_ROW_OVERHEAD = 64


# ------------------------------------------------------------------ plumbing
def _db_path(store) -> Path:
    """Resolve a path-or-store argument to the SQLite file it refers to."""
    if isinstance(store, (str, os.PathLike)):
        return Path(store).expanduser()
    path = getattr(store, "db_path", None)
    if not path:
        raise TypeError("store must be a path or expose a db_path attribute")
    return Path(str(path)).expanduser()


def _connect(path: Path, *, read_only: bool = False,
             timeout: float = 10.0) -> sqlite3.Connection:
    """Open *path*; read-only unless a write is intended.

    A read-only open can fail for a WAL database whose ``-shm`` file the caller
    may not create.  The fallback opens read/write but sets ``query_only`` so
    the connection still cannot change the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"store does not exist: {path}")
    if read_only:
        uri = "file:" + urllib.parse.quote(path.resolve().as_posix(), safe="/") + "?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=timeout)
        except sqlite3.OperationalError:
            conn = sqlite3.connect(str(path), timeout=timeout)
        conn.execute("PRAGMA query_only=ON")
    else:
        conn = sqlite3.connect(str(path), timeout=timeout, isolation_level=None)
        conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row
    return conn


def _tables(conn) -> set[str]:
    try:
        return {str(row[0]) for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    except sqlite3.OperationalError:
        return set()


def _columns(conn, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.OperationalError:
        return set()


def _first_table(conn, names) -> str | None:
    have = _tables(conn)
    for name in names:
        if name in have:
            return name
    return None


def _pick(columns, names) -> str | None:
    for name in names:
        if name in columns:
            return name
    return None


def _pragma_int(conn, name: str) -> int:
    try:
        value = conn.execute(f"PRAGMA {name}").fetchone()
        return int(value[0] or 0) if value else 0
    except (sqlite3.OperationalError, TypeError, IndexError):
        return 0


def _count(conn, table: str | None) -> int:
    if not table:
        return 0
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except sqlite3.OperationalError:
        return 0


def _avg_row_bytes(conn, table: str | None, columns) -> int:
    """Average stored bytes per row for *table*, plus a small fixed overhead."""
    if not table:
        return _ROW_OVERHEAD
    cols = [name for name in columns if name in _columns(conn, table)]
    if not cols:
        return _ROW_OVERHEAD
    expr = " + ".join(f"COALESCE(LENGTH({name}), 0)" for name in cols)
    try:
        rows, total = conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM({expr}), 0) FROM {table}").fetchone()
        if rows:
            return max(1, int(total) // int(rows)) + _ROW_OVERHEAD
    except sqlite3.OperationalError:
        pass
    return _ROW_OVERHEAD


def _age_column(conn, table: str) -> str | None:
    return _pick(_columns(conn, table), _TIME_COLUMNS)


# -------------------------------------------------------------------- inspect
def inspect(store) -> dict:
    """Report the store's size and row counts.

    All values are derived from PRAGMAs, ``os.stat`` and ``COUNT(*)``; nothing
    is written and missing tables read as zero.
    """
    path = _db_path(store)
    conn = _connect(path, read_only=True)
    try:
        page_size = _pragma_int(conn, "page_size")
        page_count = _pragma_int(conn, "page_count")
        freelist_count = _pragma_int(conn, "freelist_count")
        tables = _tables(conn)
        tombstones = _first_table(conn, _TOMBSTONE_TABLES)
        wal = path.with_name(path.name + "-wal")
        return {
            "db_bytes": int(path.stat().st_size),
            "wal_bytes": int(wal.stat().st_size) if wal.exists() else 0,
            "facts": _count(conn, "facts" if "facts" in tables else None),
            "tombstones": _count(conn, tombstones),
            "audit_rows": _count(conn, "audit" if "audit" in tables else None),
            "proposals": _count(conn, "conflict_proposals"
                                if "conflict_proposals" in tables else None),
            "embeddings": _count(conn, "fact_embeddings"
                                 if "fact_embeddings" in tables else None),
            "freelist_bytes": int(freelist_count * page_size),
            "page_count": page_count,
            "page_size": page_size,
        }
    finally:
        conn.close()


# ---------------------------------------------------------- eligibility
def _tombstone_eligible(conn, days: int) -> int:
    """Rows a purge would delete: already restored, or retired beyond *days*."""
    table = _first_table(conn, _TOMBSTONE_TABLES)
    if not table:
        return 0
    cols = _columns(conn, table)
    clauses = []
    if "restored_at" in cols:
        clauses.append("restored_at IS NOT NULL")
    age = _pick(cols, ("retired_at", "created_at", "at", "ts"))
    if age:
        clauses.append(f"datetime({age}) < datetime('now', ?)")
    if not clauses:
        return 0
    try:
        row = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {' OR '.join(clauses)}",
            (f"-{max(0, int(days))} days",)).fetchone()
        return int(row[0] or 0) if row else 0
    except sqlite3.OperationalError:
        return 0


def _audit_eligible(conn, days: int) -> int:
    if "audit" not in _tables(conn):
        return 0
    age = _age_column(conn, "audit")
    if not age:
        return 0
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM audit WHERE datetime(" + age + ") < datetime('now', ?)",
            (f"-{max(0, int(days))} days",)).fetchone()
        return int(row[0] or 0) if row else 0
    except sqlite3.OperationalError:
        return 0


def _list_categories(categories) -> tuple[str, ...]:
    if isinstance(categories, str):
        categories = (categories,)
    return tuple(str(c) for c in categories or () if str(c))


def _archive_candidates(conn, *, older_than_days: int,
                        categories, keep_min_per_category: int = 5) -> list[dict]:
    """Facts eligible for archival, oldest first, floor respected per category.

    The newest ``keep_min_per_category`` facts of each category are never
    candidates, even when they are older than the cutoff: the floor exists so a
    category cannot be archived down to nothing.
    """
    cats = _list_categories(categories)
    if not cats or "facts" not in _tables(conn):
        return []
    cols = _columns(conn, "facts")
    age = _pick(cols, ("updated_at", "created_at"))
    if not age:
        return []
    wanted = [name for name in ("fact_id", "content", "category", "tags",
                                "trust_score", "created_at", "updated_at")
              if name in cols]
    placeholders = ",".join("?" for _ in cats)
    cutoff = f"-{max(0, int(older_than_days))} days"
    sql = (f"SELECT {', '.join(wanted)}, "
           f"CASE WHEN datetime({age}) < datetime('now', ?) THEN 1 ELSE 0 END AS is_old "
           f"FROM facts WHERE category IN ({placeholders}) "
           f"ORDER BY category, {age} DESC, fact_id DESC")
    try:
        rows = [dict(row) for row in conn.execute(sql, (cutoff, *cats))]
    except sqlite3.OperationalError:
        return []

    out: list[dict] = []
    floor = max(0, int(keep_min_per_category))
    index = 0
    while index < len(rows):
        category = rows[index].get("category")
        group = []
        while index < len(rows) and rows[index].get("category") == category:
            group.append(rows[index])
            index += 1
        # group is newest-first: the first `floor` rows are the protected ones.
        for row in group[floor:]:
            if not row.get("is_old"):
                continue
            row.pop("is_old", None)
            out.append(row)
    out.sort(key=lambda row: (str(row.get(age) or ""), int(row["fact_id"])))
    return out


# ----------------------------------------------------------------------- plan
def plan(store, *, max_facts: int = 5000, max_tombstones_days: int = 180,
         max_audit_days: int = 365, max_bytes: int | None = None) -> dict:
    """Describe a cleanup without touching the store.

    Returns ``actions`` (each ``{"kind", "count", "detail"}``),
    ``estimated_bytes_reclaimed`` and ``blocked`` (a list of
    ``{"kind", "reason"}`` for work that cannot run, currently low disk for
    vacuum or an over-limit store with nothing old enough to archive).

    Actions are limited to ``purge_tombstones``, ``prune_audit``,
    ``archive_facts`` and ``vacuum``.  ``archive_facts`` is only proposed, never
    carried out by :func:`apply`.
    """
    path = _db_path(store)
    snapshot = inspect(path)
    conn = _connect(path, read_only=True)
    try:
        tomb_table = _first_table(conn, _TOMBSTONE_TABLES)
        tombstones = _tombstone_eligible(conn, max_tombstones_days) if tomb_table else 0
        audit_rows = _audit_eligible(conn, max_audit_days)
        candidates = _archive_candidates(conn, older_than_days=365,
                                         categories=("user_pref",),
                                         keep_min_per_category=5)
        tomb_bytes = _avg_row_bytes(conn, tomb_table, ("content", "entities", "reason"))
        audit_bytes = _avg_row_bytes(conn, "audit", ("action", "reason", "before", "after"))
    finally:
        conn.close()

    actions: list[dict] = []
    blocked: list[dict] = []
    estimated = 0

    if tombstones:
        actions.append({
            "kind": "purge_tombstones",
            "count": int(tombstones),
            "detail": (f"delete {tombstones} tombstone(s) already restored or "
                       f"older than {int(max_tombstones_days)} days"),
        })
        estimated += tombstones * tomb_bytes

    if audit_rows:
        actions.append({
            "kind": "prune_audit",
            "count": int(audit_rows),
            "detail": f"delete {audit_rows} audit row(s) older than {int(max_audit_days)} days",
        })
        estimated += audit_rows * audit_bytes

    excess = max(0, int(snapshot["facts"]) - max(0, int(max_facts)))
    archive_count = min(excess, len(candidates))
    over_bytes = max_bytes is not None and snapshot["db_bytes"] > int(max_bytes)
    if over_bytes and not archive_count:
        # A byte limit can be hit long before a row limit; archive what the
        # policy allows rather than waiting for max_facts.
        archive_count = len(candidates)
    if archive_count:
        chosen = candidates[:archive_count]
        actions.append({
            "kind": "archive_facts",
            "count": int(archive_count),
            "detail": (f"archive {archive_count} fact(s) older than 365 days in "
                       f"categories user_pref, keeping at least 5 per category"),
        })
        estimated += sum(len(str(row.get("content") or "").encode("utf-8")) + 96
                         for row in chosen)
    elif excess:
        blocked.append({
            "kind": "archive_facts",
            "reason": (f"{excess} fact(s) over max_facts={int(max_facts)} but no "
                       f"user_pref fact older than 365 days is above the 5-per-category floor"),
        })

    if max_bytes is not None and over_bytes:
        blocked.append({
            "kind": "max_bytes",
            "reason": (f"database is {snapshot['db_bytes']} bytes, over the "
                       f"{int(max_bytes)} byte limit"),
        })

    if snapshot["freelist_bytes"] or tombstones or audit_rows or archive_count:
        reason = _disk_block(path, snapshot["db_bytes"])
        if reason:
            blocked.append({"kind": "vacuum", "reason": reason})
        else:
            actions.append({
                "kind": "vacuum",
                "count": int(snapshot["freelist_bytes"]),
                "detail": (f"checkpoint WAL, then reclaim ~{snapshot['freelist_bytes']} "
                           f"freelist byte(s); needs {2 * snapshot['db_bytes']} free bytes"),
            })
            estimated += snapshot["freelist_bytes"]

    return {
        "actions": actions,
        "estimated_bytes_reclaimed": int(max(0, estimated)),
        "blocked": blocked,
        # apply() needs the windows; keeping them here makes a plan
        # self-describing without widening the action tuples.
        "params": {
            "max_tombstones_days": int(max_tombstones_days),
            "max_audit_days": int(max_audit_days),
        },
    }


# -------------------------------------------------------------------- archive
def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _archive_path(backup_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return backup_dir / f"archive-{stamp}-{uuid.uuid4().hex[:8]}.jsonl"


def _read_archive(path: Path) -> dict[int, str]:
    """Read a JSONL archive back as ``{fact_id: sha256}``."""
    out: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            out[int(record["fact_id"])] = str(record["sha256"])
    return out


def archive_facts(store, *, older_than_days: int = 365, categories=("user_pref",),
                  dry_run: bool = True, keep_min_per_category: int = 5,
                  backup_dir=None) -> dict:
    """Export old facts to JSONL, verify the export, then delete those rows.

    The export is written to a temporary file and atomically renamed into
    ``~/.hermes/backups/archive/`` (or ``FACTKIT_ARCHIVE_DIR`` / *backup_dir*)
    before any row is removed.  The file is then read back and each id and
    content hash is compared with the live row; only rows that verify are
    deleted.  An unwritable target is a refusal, not an exception.

    ``dry_run=True`` (the default) reports candidates and writes nothing.
    The newest *keep_min_per_category* facts of each category are never
    archived.
    """
    path = _db_path(store)
    conn = _connect(path, read_only=bool(dry_run))
    try:
        candidates = _archive_candidates(
            conn, older_than_days=older_than_days, categories=categories,
            keep_min_per_category=keep_min_per_category)
        if dry_run:
            return {
                "ok": True, "dry_run": True, "path": None,
                "exported": 0, "deleted": 0,
                "candidates": [int(row["fact_id"]) for row in candidates],
                "unverified": [], "kept": _category_counts(conn, categories),
            }
        if not candidates:
            return {"ok": True, "dry_run": False, "path": None,
                    "exported": 0, "deleted": 0, "candidates": [],
                    "unverified": [], "kept": _category_counts(conn, categories)}

        target_dir = Path(backup_dir).expanduser() if backup_dir else Path(
            os.environ.get("FACTKIT_ARCHIVE_DIR") or _ARCHIVE_DIR).expanduser()
        target = _archive_path(target_dir)
        temp = target.with_name(target.name + ".tmp")
        records = []
        archived_at = datetime.now(timezone.utc).isoformat()
        for row in candidates:
            content = str(row.get("content") or "")
            records.append({
                "fact_id": int(row["fact_id"]),
                "content": content,
                "category": row.get("category"),
                "tags": row.get("tags"),
                "trust_score": row.get("trust_score"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "sha256": _sha256(content),
                "archived_at": archived_at,
            })

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            with temp.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True,
                                            ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, target)
        except OSError as exc:
            try:
                temp.unlink()
            except OSError:
                pass
            return {"ok": False, "dry_run": False, "refused": "archive file is not writable",
                    "error": str(exc), "path": None, "exported": 0, "deleted": 0,
                    "candidates": [int(row["fact_id"]) for row in candidates],
                    "unverified": [], "kept": _category_counts(conn, categories)}

        # ---- verify: re-read the file, then compare with the rows as they are
        # now (not as they were selected), so a concurrent edit cannot be
        # deleted under an archive that holds the older text.
        try:
            written = _read_archive(target)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return {"ok": False, "dry_run": False, "refused": "archive verification failed",
                    "error": str(exc), "path": str(target), "exported": len(records),
                    "deleted": 0, "candidates": [int(row["fact_id"]) for row in candidates],
                    "unverified": [int(row["fact_id"]) for row in candidates],
                    "kept": _category_counts(conn, categories)}

        live = _live_hashes(conn, list(written))
        verified = [fact_id for fact_id, digest in written.items()
                    if live.get(fact_id) == digest]
        unverified = [int(row["fact_id"]) for row in candidates
                      if int(row["fact_id"]) not in verified]
        if not verified:
            return {"ok": False, "dry_run": False, "refused": "archive verification failed",
                    "error": "no exported row matched its live content hash",
                    "path": str(target), "exported": len(records), "deleted": 0,
                    "candidates": [int(row["fact_id"]) for row in candidates],
                    "unverified": unverified, "kept": _category_counts(conn, categories)}

        try:
            deleted = _delete_facts(conn, verified)
        except sqlite3.Error as exc:
            return {"ok": False, "dry_run": False, "refused": "delete failed",
                    "error": str(exc), "path": str(target), "exported": len(records),
                    "deleted": 0, "candidates": [int(row["fact_id"]) for row in candidates],
                    "unverified": unverified, "kept": _category_counts(conn, categories)}

        return {
            "ok": deleted == len(verified),
            "dry_run": False,
            "path": str(target),
            "exported": len(records),
            "deleted": int(deleted),
            "candidates": [int(row["fact_id"]) for row in candidates],
            "unverified": unverified,
            "kept": _category_counts(conn, categories),
        }
    finally:
        conn.close()


def _category_counts(conn, categories) -> dict[str, int]:
    cats = _list_categories(categories)
    if not cats or "facts" not in _tables(conn):
        return {}
    placeholders = ",".join("?" for _ in cats)
    try:
        rows = conn.execute(
            f"SELECT category, COUNT(*) FROM facts WHERE category IN ({placeholders}) "
            f"GROUP BY category", cats)
        return {str(category): int(count) for category, count in rows}
    except sqlite3.OperationalError:
        return {}


def _live_hashes(conn, fact_ids) -> dict[int, str]:
    """Current content hashes for *fact_ids*, read in one query per 500 ids."""
    out: dict[int, str] = {}
    ids = [int(fact_id) for fact_id in fact_ids]
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        marks = ",".join("?" for _ in chunk)
        for row in conn.execute(
                f"SELECT fact_id, content FROM facts WHERE fact_id IN ({marks})", chunk):
            out[int(row[0])] = _sha256(str(row[1] or ""))
    return out


def _delete_facts(conn, fact_ids) -> int:
    """Delete facts and their child rows; return the number of facts removed.

    Everything happens in one immediate transaction, and the fact-row count is
    checked before commit so a partial delete is rolled back rather than
    reported as success.
    """
    ids = sorted({int(fact_id) for fact_id in fact_ids})
    if not ids:
        return 0
    conn.execute("BEGIN IMMEDIATE")
    try:
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ",".join("?" for _ in chunk)
            if "facts_fts" in _tables(conn):
                try:
                    conn.execute(f"DELETE FROM facts_fts WHERE rowid IN ({marks})", chunk)
                except sqlite3.OperationalError:
                    pass
            for table in _FACT_CHILD_TABLES:
                if table in _tables(conn) and "fact_id" in _columns(conn, table):
                    conn.execute(f"DELETE FROM {table} WHERE fact_id IN ({marks})", chunk)
            cur = conn.execute(f"DELETE FROM facts WHERE fact_id IN ({marks})", chunk)
            if int(cur.rowcount or 0) != len(chunk):
                raise sqlite3.IntegrityError(
                    f"expected to delete {len(chunk)} facts, deleted {cur.rowcount}")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return len(ids)


# ---------------------------------------------------------------------- apply
def _disk_block(path: Path, db_bytes: int) -> str | None:
    """Reason vacuum cannot run, or ``None`` when there is room.

    A full ``VACUUM`` writes a new copy before replacing the old one, so free
    space must be at least twice the current file size.
    """
    needed = 2 * int(db_bytes)
    try:
        free = int(shutil.disk_usage(str(path.parent)).free)
    except OSError as exc:
        return f"cannot determine free disk for {path.parent}: {exc}"
    if free < needed:
        return (f"VACUUM needs {needed} free bytes (twice the {db_bytes}-byte "
                f"database), only {free} available")
    return None


def _purge_tombstones(conn, days: int) -> int:
    table = _first_table(conn, _TOMBSTONE_TABLES)
    if not table:
        return 0
    cols = _columns(conn, table)
    clauses = []
    if "restored_at" in cols:
        clauses.append("restored_at IS NOT NULL")
    age = _pick(cols, ("retired_at", "created_at", "at", "ts"))
    if age:
        clauses.append(f"datetime({age}) < datetime('now', ?)")
    if not clauses:
        return 0
    cur = conn.execute(f"DELETE FROM {table} WHERE {' OR '.join(clauses)}",
                       (f"-{max(0, int(days))} days",))
    return int(cur.rowcount or 0)


def _prune_audit(conn, days: int) -> int:
    if "audit" not in _tables(conn):
        return 0
    age = _age_column(conn, "audit")
    if not age:
        return 0
    cur = conn.execute(
        "DELETE FROM audit WHERE datetime(" + age + ") < datetime('now', ?)",
        (f"-{max(0, int(days))} days",))
    return int(cur.rowcount or 0)


def _vacuum(conn) -> str:
    """Checkpoint WAL, then incremental or full vacuum. Returns the mode used."""
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    except sqlite3.OperationalError:
        pass  # not a WAL database (or another connection is mid-checkpoint)
    auto_vacuum = _pragma_int(conn, "auto_vacuum")
    if auto_vacuum == 2:
        try:
            conn.execute("PRAGMA incremental_vacuum").fetchall()
            return "incremental"
        except sqlite3.OperationalError:
            pass  # fall through to a full rewrite
    conn.execute("VACUUM")
    return "full"


def apply(store, plan_result, *, confirm_token) -> dict:
    """Execute the non-destructive part of *plan_result*.

    Only ``purge_tombstones``, ``prune_audit`` and ``vacuum`` run.  A plan that
    contains ``archive_facts`` is refused outright -- use
    :func:`archive_facts` with its export-and-verify path for that -- and no
    other action is executed in that case.

    ``confirm_token`` must be exactly ``"apply-limits"``; anything else raises
    :class:`PermissionError` before the store is opened.  Vacuum is preceded by
    a WAL checkpoint and is blocked when free disk is below twice the file size.
    """
    if confirm_token != APPLY_TOKEN:
        raise PermissionError("apply requires confirm_token='apply-limits'")

    actions = list(plan_result.get("actions") or [])
    kinds = {str(action.get("kind")) for action in actions}
    if "archive_facts" in kinds:
        return {
            "ok": False,
            "refused": "archive_facts",
            "executed": [],
            "counts": {},
            "vacuum": None,
            "blocked": [{"kind": "archive_facts",
                         "reason": "apply never deletes facts; run archive_facts explicitly"}],
        }

    path = _db_path(store)
    if "vacuum" in kinds:
        reason = _disk_block(path, path.stat().st_size)
        if reason:
            return {"ok": False, "refused": "insufficient_disk_for_vacuum",
                    "executed": [], "counts": {}, "vacuum": None,
                    "blocked": [{"kind": "vacuum", "reason": reason}]}

    params = plan_result.get("params") or {}
    tomb_days = int(params.get("max_tombstones_days", 180))
    audit_days = int(params.get("max_audit_days", 365))
    ignored = sorted(kind for kind in kinds
                     if kind not in {"purge_tombstones", "prune_audit", "vacuum"})

    executed: list[dict] = []
    counts: dict[str, int] = {}
    vacuum_mode = None
    conn = _connect(path)
    try:
        if "purge_tombstones" in kinds:
            count = _purge_tombstones(conn, tomb_days)
            counts["purge_tombstones"] = count
            executed.append({"kind": "purge_tombstones", "count": count})
        if "prune_audit" in kinds:
            count = _prune_audit(conn, audit_days)
            counts["prune_audit"] = count
            executed.append({"kind": "prune_audit", "count": count})
        if "vacuum" in kinds:
            vacuum_mode = _vacuum(conn)
            executed.append({"kind": "vacuum", "count": 0})
    finally:
        conn.close()

    return {"ok": True, "executed": executed, "counts": counts,
            "vacuum": vacuum_mode, "ignored": ignored, "blocked": []}
