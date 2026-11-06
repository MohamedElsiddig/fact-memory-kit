"""Provenance tracking for facts in a holographic store.

The provider (the holographic plugin) writes facts through its own API, so
those writes never appear in factkit's ``audit`` table. Provenance closes that
gap without intercepting the provider: factkit records a content hash for each
fact it knows about, and :func:`snapshot_diff` re-reads ``facts`` to show which
hashes moved. A provider write therefore becomes visible on the next snapshot.

The module is stdlib-only and never writes to ``facts``. It uses the store
duck-typed, like :mod:`factkit.maintenance`:

* ``store.db_path`` -- database path (fallback when no connection is exposed)
* ``store.read(sql, params)`` -- rows in a mapping shape
* ``store._write`` / ``store.write`` -- single-statement writer, joined to an
  open ``store._tx()`` when the store provides one
* ``store._conn`` / ``store.conn`` -- raw SQLite connection fallbacks

Schema creation is additive: ``CREATE TABLE IF NOT EXISTS`` plus an index. An
existing ``fact_provenance`` table and its rows are left untouched.
"""

from __future__ import annotations

import contextlib
import hashlib
import sqlite3
import unicodedata
from datetime import datetime, timezone
from typing import Any, Iterator

__all__ = [
    "ensure_schema",
    "backfill",
    "record",
    "snapshot_diff",
    "report",
    "normalize_content",
    "hash_content",
]

TABLE = "fact_provenance"
INDEX = "idx_fact_provenance_origin"

_COLUMNS = ("fact_id", "origin", "actor", "first_seen", "last_seen",
            "content_hash", "revision", "note")

_CREATE = (
    "CREATE TABLE IF NOT EXISTS fact_provenance ("
    "fact_id INTEGER PRIMARY KEY, "
    "origin TEXT, "
    "actor TEXT, "
    "first_seen TEXT, "
    "last_seen TEXT, "
    "content_hash TEXT, "
    "revision INTEGER, "
    "note TEXT)"
)

_CREATE_INDEX = (
    f"CREATE INDEX IF NOT EXISTS {INDEX} ON fact_provenance(origin)"
)

_UPSERT = (
    "INSERT INTO fact_provenance "
    "(fact_id, origin, actor, first_seen, last_seen, content_hash, revision, note) "
    "VALUES (?, ?, ?, ?, ?, ?, 1, ?) "
    "ON CONFLICT(fact_id) DO UPDATE SET "
    "origin = excluded.origin, "
    "actor = excluded.actor, "
    "last_seen = excluded.last_seen, "
    "note = excluded.note, "
    # SQLite evaluates the right-hand side against the pre-update row, so the
    # revision moves only when the stored hash and the fresh hash differ.
    "revision = fact_provenance.revision + "
    "(fact_provenance.content_hash IS NOT excluded.content_hash), "
    "content_hash = excluded.content_hash"
)


# ------------------------------------------------------------------- helpers
def _now() -> str:
    """UTC timestamp with fixed width, so text ordering is chronological."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def normalize_content(content: Any) -> str:
    """Canonical text for hashing: NFC, trimmed, whitespace collapsed.

    Case and punctuation are part of the claim and are *not* normalised; only
    layout differences (newlines, repeated spaces, padding) hash the same.
    """
    text = unicodedata.normalize("NFC", str(content or ""))
    return " ".join(text.split())


def hash_content(content: Any) -> str:
    """SHA-256 hex digest of :func:`normalize_content` output."""
    return hashlib.sha256(normalize_content(content).encode("utf-8")).hexdigest()


def _origin(value: Any) -> str:
    text = str(value or "").strip()
    return text or "unknown"


def _fact_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid fact_id: {value!r}") from None


@contextlib.contextmanager
def _raw_conn(store) -> Iterator[sqlite3.Connection]:
    """Yield the store's connection, opening (and closing) one by path if needed."""
    if isinstance(store, sqlite3.Connection):
        yield store
        return
    conn = getattr(store, "_conn", None)
    if not isinstance(conn, sqlite3.Connection):
        conn = getattr(store, "conn", None)
    if isinstance(conn, sqlite3.Connection):
        yield conn
        return
    path = getattr(store, "db_path", None)
    if not path:
        raise AttributeError(
            "store exposes no SQLite connection (need _conn, conn or db_path)")
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _rows(store, sql: str, params=()) -> list[dict]:
    """Read rows as plain dicts, through ``store.read`` when available."""
    reader = getattr(store, "read", None)
    if callable(reader):
        raw = reader(sql, params)
        out: list[dict] = []
        for row in raw:
            if isinstance(row, dict):
                out.append(dict(row))
                continue
            try:
                out.append(dict(row))
            except (TypeError, ValueError):  # tuple rows from a plain connection
                out.append({str(i): value for i, value in enumerate(row)})
        return out
    with _raw_conn(store) as conn:
        cur = conn.execute(sql, params)
        names = [str(d[0]) for d in (cur.description or ())]
        return [dict(zip(names, row)) for row in cur.fetchall()]


def _write(store, sql: str, params=()):
    """Run one write statement through the store's own write path when it has one."""
    fn = getattr(store, "_write", None)
    if callable(fn):
        return fn(sql, params)
    fn = getattr(store, "write", None)
    if callable(fn):
        return fn(sql, params)
    with _raw_conn(store) as conn:
        started = bool(getattr(conn, "in_transaction", False))
        cur = conn.execute(sql, params)
        # Commit only a transaction this call opened; never commit a caller's.
        if not started and bool(getattr(conn, "in_transaction", False)):
            conn.commit()
        return cur


@contextlib.contextmanager
def _tx(store) -> Iterator[None]:
    """Join the store's transaction when it has one, else start one locally."""
    fn = getattr(store, "_tx", None)
    if callable(fn):
        with fn():
            yield
        return

    conn = store if isinstance(store, sqlite3.Connection) else None
    if conn is None:
        candidate = getattr(store, "_conn", None)
        if not isinstance(candidate, sqlite3.Connection):
            candidate = getattr(store, "conn", None)
        conn = candidate if isinstance(candidate, sqlite3.Connection) else None
    if conn is None:
        yield
        return

    if getattr(conn, "in_transaction", False):
        yield
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _columns(store, table: str) -> set[str]:
    try:
        return {str(row.get("name")) for row in _rows(store, f"PRAGMA table_info({table})")}
    except sqlite3.Error:  # absent or unreadable schema must not be fatal
        return set()


def _facts(store) -> dict[int, str] | None:
    """Current ``{fact_id: content}``, or ``None`` when ``facts`` is unavailable."""
    columns = _columns(store, "facts")
    if not {"fact_id", "content"} <= columns:
        return None
    try:
        rows = _rows(store, "SELECT fact_id, content FROM facts ORDER BY fact_id")
    except sqlite3.Error:
        return None
    return {int(row["fact_id"]): row["content"] or "" for row in rows}


def _fact_content(store, fact_id: int) -> str | None:
    """Current content of one fact, or ``None`` when it (or ``facts``) is absent."""
    if not {"fact_id", "content"} <= _columns(store, "facts"):
        return None
    try:
        rows = _rows(store, "SELECT content FROM facts WHERE fact_id = ?", (fact_id,))
    except sqlite3.Error:
        return None
    return (rows[0].get("content") or "") if rows else None


def _provenance_rows(store) -> list[dict]:
    return _rows(store, f"SELECT {', '.join(_COLUMNS)} FROM fact_provenance")


def _provenance_row(store, fact_id: int) -> dict | None:
    rows = _rows(store, f"SELECT {', '.join(_COLUMNS)} FROM fact_provenance "
                        "WHERE fact_id = ?", (fact_id,))
    return rows[0] if rows else None


# ---------------------------------------------------------------------- API
def ensure_schema(store) -> None:
    """Create ``fact_provenance`` and its origin index if missing. Idempotent.

    Additive only: an existing table (and every row in it) is untouched. Safe
    on a bare store that has no ``facts`` table yet.
    """
    with _tx(store):
        _write(store, _CREATE)
        _write(store, _CREATE_INDEX)


def backfill(store, *, origin: str = "unknown", batch: int = 500) -> dict:
    """Give every untracked fact a provenance row. Never modifies ``facts``.

    ``origin`` is stamped on the new rows (typically the provider or importer
    that owned the fact before tracking started). ``batch`` bounds how many
    rows are written per transaction. Returns a summary; a missing ``facts``
    table or a read-only database is reported, not raised.
    """
    origin = _origin(origin)
    summary = {"origin": origin, "scanned": 0, "inserted": 0, "existing": 0}
    try:
        ensure_schema(store)
    except sqlite3.OperationalError as exc:  # read-only database
        return {**summary, "error": str(exc)}

    facts = _facts(store)
    if facts is None:
        return {**summary, "error": "facts table not found"}
    summary["scanned"] = len(facts)

    try:
        tracked = {int(row["fact_id"]) for row in _provenance_rows(store)}
    except sqlite3.Error as exc:
        return {**summary, "error": str(exc)}
    summary["existing"] = len(set(facts) & tracked)

    pending = [(fid, facts[fid]) for fid in sorted(facts) if fid not in tracked]
    step = max(1, int(batch or 500))
    stamp = _now()

    for start in range(0, len(pending), step):
        chunk = pending[start:start + step]
        with _tx(store):
            for fid, content in chunk:
                cur = _write(
                    store,
                    "INSERT OR IGNORE INTO fact_provenance "
                    "(fact_id, origin, actor, first_seen, last_seen, content_hash, "
                    " revision, note) VALUES (?, ?, ?, ?, ?, ?, 1, NULL)",
                    (fid, origin, "factkit", stamp, stamp, hash_content(content)),
                )
                # rowcount is 1 for a fresh row, 0 when a concurrent writer won.
                if cur is None:
                    summary["inserted"] += 1
                elif int(getattr(cur, "rowcount", 1) or 0) != 0:
                    summary["inserted"] += 1
    return summary


def record(store, fact_id, *, origin: str, actor: str = "factkit",
           note: str | None = None) -> dict:
    """Upsert the provenance row for one fact and return it.

    ``first_seen`` is preserved. ``revision`` starts at 1 and increments only
    when the fact's normalized content hash differs from the stored one, so a
    re-record of unchanged text is not a revision. ``origin``, ``actor`` and
    ``note`` always reflect the latest call (``note=None`` clears the note).

    Raises ``ValueError`` when the fact does not exist or ``fact_id`` is invalid.
    """
    fid = _fact_id(fact_id)
    ensure_schema(store)
    content = _fact_content(store, fid)
    if content is None:
        raise ValueError(f"fact {fid} not found")

    stamp = _now()
    with _tx(store):
        _write(store, _UPSERT, (fid, _origin(origin), str(actor or "factkit"),
                                stamp, stamp, hash_content(content), note))
    row = _provenance_row(store, fid)
    if row is None:  # pragma: no cover - the upsert either lands or raises
        raise RuntimeError(f"provenance for fact {fid} was not stored")
    return row


def snapshot_diff(store) -> dict:
    """Compare stored provenance hashes with the live ``facts`` table.

    Returns ``{"changed": [...], "missing": [...], "new": [...]}``, each list
    sorted by ``fact_id``:

    * ``changed`` -- a provenance row whose hash no longer matches the fact
    * ``missing`` -- a provenance row with no live fact (retired outside factkit)
    * ``new`` -- a live fact with no provenance row

    Read-only; a store without ``facts`` yields three empty lists.
    """
    facts = _facts(store)
    if facts is None:
        return {"changed": [], "missing": [], "new": []}

    try:
        rows = _provenance_rows(store)
    except sqlite3.Error:  # no provenance table yet: every fact is new
        rows = []
    stored = {int(row["fact_id"]): row for row in rows}

    changed, missing, new = [], [], []
    for fid in sorted(facts):
        content = facts[fid]
        digest = hash_content(content)
        row = stored.get(fid)
        if row is None:
            new.append({"fact_id": fid, "content_hash": digest, "content": content})
        elif (row.get("content_hash") or "") != digest:
            changed.append({
                "fact_id": fid,
                "stored_hash": row.get("content_hash"),
                "current_hash": digest,
                "origin": row.get("origin"),
                "revision": row.get("revision"),
                "last_seen": row.get("last_seen"),
                "content": content,
            })
    for fid in sorted(stored):
        if fid not in facts:
            row = stored[fid]
            missing.append({
                "fact_id": fid,
                "stored_hash": row.get("content_hash"),
                "origin": row.get("origin"),
                "last_seen": row.get("last_seen"),
            })
    return {"changed": changed, "missing": missing, "new": new}


def report(store, *, limit: int = 20) -> dict:
    """Coverage and recent activity for the provenance table.

    Returns counts by origin, the newest provenance rows, and the facts that
    still have no provenance row. ``limit`` caps the two lists (not the counts).
    """
    facts = _facts(store) or {}
    try:
        rows = _provenance_rows(store)
    except sqlite3.Error:
        rows = []

    by_origin: dict[str, int] = {}
    for row in rows:
        key = _origin(row.get("origin"))
        by_origin[key] = by_origin.get(key, 0) + 1

    tracked_ids = {int(row["fact_id"]) for row in rows}
    tracked = len(set(facts) & tracked_ids)
    cap = max(0, int(limit if limit is not None else 20))

    newest = sorted(rows, key=lambda row: (str(row.get("last_seen") or ""),
                                          int(row["fact_id"])), reverse=True)[:cap]
    newest = [{**row, "fact_id": int(row["fact_id"]),
               "content": facts.get(int(row["fact_id"]))} for row in newest]

    untracked = [{"fact_id": fid, "content": facts[fid]}
                 for fid in sorted(facts) if fid not in tracked_ids][:cap]

    return {
        "facts": len(facts),
        "tracked": tracked,
        "untracked": len(facts) - tracked,
        "by_origin": dict(sorted(by_origin.items(), key=lambda item: (-item[1], item[0]))),
        "newest": newest,
        "no_provenance": untracked,
    }
