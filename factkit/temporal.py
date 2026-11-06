"""Temporal validity for facts: *when* a fact was true, not just whether it is.

The live store's conflict rule is "newer wins": the losing row is retired, so
the answer to "what did we believe in March?" is gone.  This module adds two
nullable bounds to ``facts`` and leaves every existing operation alone:

``valid_from``
    the instant the claim started being true (NULL = since forever)
``valid_to``
    the instant it stopped being true (NULL = still true)

Both are ISO-8601 UTC, second precision (``2026-03-01T00:00:00Z``).  The
columns are additive: a provider that does not know about them keeps working,
and dropping the two columns (or just ignoring them) reverts the feature.  No
row is ever deleted here, and no fact's ``content`` is touched.

Store shapes
------------
The store is duck-typed, exactly like :mod:`factkit.maintenance`:

* ``store._write(sql, params)`` / ``store.write`` / ``store.db`` /
  ``store.conn`` / ``store._conn`` -- the write path, in that order
* ``store._tx()`` -- an optional transaction context manager
* ``store.read(sql, params)`` -- rows in a mapping shape, when available

``UPDATE`` statements name only the two temporal columns, so FTS triggers,
``updated_at`` recency and trust scores are untouched: recording *when* a fact
held must not make it look freshly confirmed.

Timestamps are parsed, never compared as strings, because a provider's
``created_at`` is ``"YYYY-MM-DD HH:MM:SS"`` while the bounds are written with a
``T`` and a ``Z``.  Unparseable bounds are treated as open-ended: a corrupt
timestamp must not silently hide a fact from recall.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import date, datetime, timezone
from typing import Any, Iterable, Iterator, Mapping

__all__ = [
    "ensure_schema",
    "stamp",
    "is_valid",
    "filter_valid",
    "supersede",
    "as_of_query",
    "backfill",
    "normalize_ts",
]

INDEX_NAME = "idx_facts_validity"
BOUNDS = ("valid_from", "valid_to")

# Minimal table, used only when ``facts`` itself is absent (a bare database).
# An existing provider schema is never redefined -- only two columns and one
# index are added to it.
_BARE_FACTS = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    content    TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


# --------------------------------------------------------------------- time
def _parse(value: Any) -> datetime | None:
    """Parse a stored bound into an aware UTC datetime; None when unknown."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            dt = datetime.fromtimestamp(float(value), timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:  # SQLite CURRENT_TIMESTAMP is UTC, unmarked
            dt = dt.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0)


def _fmt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_ts(value: Any) -> str | None:
    """Canonical ``YYYY-MM-DDTHH:MM:SSZ`` text for any accepted timestamp.

    Accepts ``datetime``/``date`` objects, epoch seconds, and ISO-8601 strings
    with or without a timezone (naive values are read as UTC).  Raises
    ``ValueError`` for anything else, so a caller never stores a bound that can
    never be compared.
    """
    if value is None:
        return None
    dt = _parse(value)
    if dt is None:
        raise ValueError(f"not an ISO-8601 timestamp: {value!r}")
    return _fmt(dt)


def _now() -> str:
    return _fmt(datetime.now(timezone.utc))


# -------------------------------------------------------------------- store
def _conn(store) -> sqlite3.Connection | None:
    for attr in ("db", "conn", "_conn"):
        candidate = getattr(store, attr, None)
        if isinstance(candidate, sqlite3.Connection):
            return candidate
    return None


def _write(store, sql: str, params=()):
    """Run one write on the store's own connection (never a second one)."""
    for attr in ("_write", "write"):
        fn = getattr(store, attr, None)
        if callable(fn):
            return fn(sql, params)
    conn = _conn(store)
    if conn is None:
        raise AttributeError("store exposes no write path (_write, write, db, conn)")
    cur = conn.execute(sql, params)
    if not getattr(conn, "in_transaction", False):
        conn.commit()
    return cur


@contextlib.contextmanager
def _tx(store) -> Iterator[None]:
    """Join the store's transaction when it has one, else commit ourselves."""
    fn = getattr(store, "_tx", None)
    if callable(fn):
        tx: Any = fn()
        with tx:
            yield
        return
    conn = _conn(store)
    try:
        yield
    except BaseException:
        if conn is not None:
            conn.rollback()
        raise
    if conn is not None and not getattr(conn, "in_transaction", False):
        conn.commit()


def _rows(store, sql: str, params=()) -> list[dict]:
    fn = getattr(store, "read", None)
    if callable(fn):
        rows: Any = fn(sql, params)
        if not isinstance(rows, (list, tuple)):
            rows = list(rows)
    else:
        conn = _conn(store)
        if conn is None:
            raise AttributeError("store exposes no read path (read, db, conn)")
        rows = conn.execute(sql, params).fetchall()
    out = []
    for row in rows:
        try:
            out.append(dict(row))
        except (TypeError, ValueError):  # tuple rows from a plain connection
            out.append({str(i): value for i, value in enumerate(row)})
    return out


def _one(store, sql: str, params=()) -> dict | None:
    rows = _rows(store, sql, params)
    return rows[0] if rows else None


def _columns(store) -> set[str]:
    try:
        return {str(row.get("name")) for row in _rows(store, "PRAGMA table_info(facts)")}
    except sqlite3.Error:
        return set()


def _has_table(store, name: str) -> bool:
    try:
        row = _one(store, "SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (name,))
    except sqlite3.Error:
        return False
    return row is not None


# --------------------------------------------------------------------- API
def ensure_schema(store) -> dict:
    """Add the two validity columns and their index. Idempotent.

    Safe on a bare database (``facts`` is created minimally when absent) and on
    a store that already has the columns: the presence check is
    ``PRAGMA table_info``, and the index uses ``IF NOT EXISTS``.  Nothing else
    in the schema is read or written.
    """
    if not _has_table(store, "facts"):
        _write(store, _BARE_FACTS)
    columns = _columns(store)
    added: list[str] = []
    for name in BOUNDS:
        if name not in columns:
            _write(store, f"ALTER TABLE facts ADD COLUMN {name} TEXT")
            added.append(name)
    _write(
        store,
        f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON facts(valid_from, valid_to)",
    )
    conn = _conn(store)
    if conn is not None and not getattr(conn, "in_transaction", False):
        try:
            conn.commit()
        except sqlite3.Error:
            pass
    return {"ok": True, "added": added, "index": INDEX_NAME,
            "columns": sorted(_columns(store))}


def _row(store, fact_id: int) -> dict | None:
    return _one(store, "SELECT * FROM facts WHERE fact_id = ?", (int(fact_id),))


_UNSET = object()


def stamp(store, fact_id: int, *, valid_from=_UNSET, valid_to=_UNSET) -> dict:
    """Set or clear either bound, writing through the store's write path.

    Omit a keyword to leave that bound alone; pass ``None`` to clear it.  A call
    that asks for nothing (both omitted), or that would leave an already
    unbounded fact unbounded, is refused rather than quietly bumping the row.
    The success value is the refreshed fact row with ``ok=True``; a refusal is
    ``{"ok": False, "refused": True, ...}`` with the row untouched.
    """
    fid = int(fact_id)
    if valid_from is _UNSET and valid_to is _UNSET:
        return {"ok": False, "refused": True, "fact_id": fid,
                "reason": "both valid_from and valid_to omitted: nothing to stamp"}
    current = _row(store, fid)
    if current is None:
        raise ValueError(f"fact {fid} not found")
    start = current.get("valid_from") if valid_from is _UNSET else normalize_ts(valid_from)
    end = current.get("valid_to") if valid_to is _UNSET else normalize_ts(valid_to)
    if start is None and end is None and valid_from is None and valid_to is None:
        return {"ok": False, "refused": True, "fact_id": fid,
                "reason": "both bounds are already NULL: nothing to clear"}

    fields, values = [], []
    if valid_from is not _UNSET:
        fields.append("valid_from = ?")
        values.append(start)
    if valid_to is not _UNSET:
        fields.append("valid_to = ?")
        values.append(end)
    values.append(fid)
    with _tx(store):
        _write(store, f"UPDATE facts SET {', '.join(fields)} WHERE fact_id = ?", values)
    row = _row(store, fid)
    if row is None:  # a concurrent delete must not be reported as success
        raise ValueError(f"fact {fid} disappeared")
    row["ok"] = True
    return row


def _bound(fact: Any, name: str):
    if fact is None:
        return None
    if isinstance(fact, Mapping):
        return fact.get(name)
    if isinstance(fact, sqlite3.Row):  # rows straight from a provider read()
        try:
            return fact[name]
        except (IndexError, KeyError):
            return None
    getter = getattr(fact, "get", None)
    if callable(getter):
        try:
            return getter(name)
        except Exception:  # noqa: BLE001 - sqlite3.Row raises on a missing column
            return None
    return getattr(fact, name, None)


def is_valid(fact: Any, as_of=None) -> bool:
    """Is ``fact`` true at ``as_of`` (default now)?

    NULL bounds are open-ended; both comparisons are inclusive, so a fact whose
    ``valid_to`` is exactly ``as_of`` is still valid at that instant (it stops
    being valid one moment later).  Unparseable bounds are treated as open.
    """
    at = _parse(as_of) if as_of is not None else _parse(_now())
    if at is None:
        raise ValueError(f"not an ISO-8601 timestamp: {as_of!r}")
    start = _parse(_bound(fact, "valid_from"))
    end = _parse(_bound(fact, "valid_to"))
    if start is not None and at < start:
        return False
    if end is not None and at > end:
        return False
    return True


def filter_valid(rows: Iterable[Any], as_of=None) -> list:
    """Drop rows that are not valid at ``as_of``, preserving input order.

    Rows are returned as the same objects; the input sequence is not mutated and
    no store is touched, so this is safe on cached hits.
    """
    at = _parse(as_of) if as_of is not None else _parse(_now())
    if at is None:
        raise ValueError(f"not an ISO-8601 timestamp: {as_of!r}")
    return [row for row in rows if is_valid(row, at)]


def supersede(store, old_id: int, new_id: int, *, at=None) -> dict:
    """Close ``old`` and open ``new`` at the same instant. Never deletes.

    Both updates run in one transaction: ``old.valid_to = at`` and
    ``new.valid_from = at``.  Returns ``{"ok", "at", "old", "new"}`` with the
    refreshed rows.  The old fact stays in the store and remains recallable for
    an ``as_of`` before ``at``; only the current view hides it.
    """
    old, new = int(old_id), int(new_id)
    if old == new:
        raise ValueError("cannot supersede a fact with itself")
    when = normalize_ts(at) if at is not None else _now()
    if _row(store, old) is None:
        raise ValueError(f"fact {old} not found")
    if _row(store, new) is None:
        raise ValueError(f"fact {new} not found")

    with _tx(store):
        _write(store, "UPDATE facts SET valid_to = ? WHERE fact_id = ?", (when, old))
        _write(store, "UPDATE facts SET valid_from = ? WHERE fact_id = ?", (when, new))
    old_row, new_row = _row(store, old), _row(store, new)
    if old_row is None or new_row is None:
        raise ValueError("fact disappeared during supersede")
    return {"ok": True, "at": when, "old": old_row, "new": new_row}


def _bounds(store, ids: list[int]) -> dict[int, dict]:
    """Read validity bounds for ``ids``; empty when the columns are absent."""
    if not ids:
        return {}
    marks = ",".join("?" for _ in ids)
    try:
        rows = _rows(
            store,
            f"SELECT fact_id, valid_from, valid_to FROM facts WHERE fact_id IN ({marks})",
            ids,
        )
    except sqlite3.Error:  # provider schema without the additive columns
        return {}
    return {int(r["fact_id"]): r for r in rows if r.get("fact_id") is not None}


def as_of_query(store, message: str, *, as_of, limit: int = 5, **kwargs) -> dict:
    """Provider recall, then post-filtered to what was true at ``as_of``.

    This is deliberately *not* a retrieval engine.  ``factkit.holo.prefetch``
    is called unchanged (its ``limit``, ``min_trust``, ``min_score`` and
    ``timeout`` keep their meaning and can be overridden through ``kwargs``),
    and its hits are then filtered by :func:`filter_valid`.  The provider's own
    search is not time-aware: it ranks by cosine similarity over the current
    corpus, so an expired fact can occupy a top slot and the result can hold
    fewer than ``limit`` rows.  ``hidden`` reports how many hits were dropped;
    ``available`` is False when embeddings were unavailable (``prefetch``
    returns ``None``) rather than pretending the answer is "nothing".
    """
    from . import holo

    when = normalize_ts(as_of) if as_of is not None else _now()
    at = _parse(when)
    wanted = int(limit)
    # Bounded over-fetch: expired neighbours would otherwise starve the answer
    # when they crowd the top of the provider's ranking.
    fetch = max(wanted, wanted * 4)
    hits = holo.prefetch(store, message, limit=fetch, **kwargs)
    base = {"as_of": when, "for_query": message}
    if hits is None:
        return {**base, "available": False, "hits": [], "hidden": 0,
                "reason": "embeddings unavailable"}

    bounds = _bounds(store, [int(h["fact_id"]) for h in hits if h.get("fact_id") is not None])
    kept: list[dict] = []
    hidden = 0
    for hit in hits:
        fid = hit.get("fact_id")
        if fid is None:
            continue
        meta = bounds.get(int(fid), {})
        row = {"fact_id": int(fid), "content": hit.get("content"),
               "score": hit.get("score"),
               "valid_from": meta.get("valid_from"),
               "valid_to": meta.get("valid_to")}
        if is_valid(row, at):
            if len(kept) < wanted:
                kept.append(row)
        else:
            hidden += 1
    return {**base, "available": True, "hits": kept, "hidden": hidden}


def backfill(store) -> dict:
    """Set ``valid_from`` from ``created_at`` where NULL; leave ``valid_to`` NULL.

    Every fact is assumed to have been true since it was written -- the only
    claim the store can support without inventing history.  Content, tags,
    trust and timestamps are not touched, so a second run reports
    ``backfilled: 0`` and changes nothing.
    """
    rows = _rows(store, "SELECT fact_id, content, created_at, valid_from, valid_to FROM facts")
    todo: list[tuple[str, int]] = []
    already = 0
    missing = 0
    for row in rows:
        if row.get("valid_from") is not None:
            already += 1
            continue
        created = _parse(row.get("created_at"))
        if created is None:
            missing += 1
            continue
        todo.append((_fmt(created), int(row["fact_id"])))

    if todo:
        with _tx(store):
            for start, fid in todo:
                _write(store, "UPDATE facts SET valid_from = ? WHERE fact_id = ?", (start, fid))
    return {"ok": True, "total": len(rows), "backfilled": len(todo),
            "already": already, "missing_created_at": missing}
