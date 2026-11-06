"""Typed relations between facts: the edge layer over ``factkit``'s fact table.

Facts are provider-owned: this module adds exactly one table, ``fact_relations``,
and never alters ``facts`` or any other existing table.  Every edge is a
directed, weighted, labelled link ``src_id --rel--> dst_id`` whose endpoints are
fact ids, so the graph follows the store's lifecycle (a retired fact simply
leaves a dangling edge that :func:`stats` reports through its degree/orphan
counts rather than silently rewriting history).

Store shapes
------------
The store is duck-typed, like :mod:`factkit.maintenance` and
:mod:`factkit.temporal`:

* ``store.read(sql, params)`` -- rows in a mapping shape (preferred)
* ``store._write(sql, params)`` / ``store.write`` / ``store.db`` /
  ``store.conn`` / ``store._conn`` -- the write path, in that order
* ``store._tx()`` -- an optional transaction context manager
* ``store.db_path`` / ``store.path`` -- informational only

Writes go through the provider's own connection.  Reads never call
``ensure_schema``: :func:`infer` and :func:`stats` are pure reads, so a caller
cannot create a table by asking a question.

Path convention
---------------
:func:`path` returns the node chain **including the source**, each entry
``{"fact_id": int, "via_rel": str | None}``: the first entry is the source with
``via_rel=None``, every later entry names the edge used to reach it.  A direct
link therefore returns two entries, and ``len(result) - 1`` is the hop count.
"""

from __future__ import annotations

import contextlib
import math
import re
import sqlite3
from collections import deque
from datetime import datetime, timezone
from typing import Any, Iterator

from . import holo

__all__ = [
    "ensure_schema",
    "link",
    "unlink",
    "neighbors",
    "path",
    "infer",
    "stats",
]

TABLE = "fact_relations"
INDEXES = (
    "idx_fact_relations_src",
    "idx_fact_relations_dst",
)
MAX_VISITED = 2000          # hard ceiling for one BFS
CONTENT_CHARS = 200         # neighbours carry a preview, not the whole fact
DIRECTIONS = ("out", "in", "both")

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS fact_relations ("
    " id INTEGER PRIMARY KEY,"
    " src_id INTEGER NOT NULL,"
    " dst_id INTEGER NOT NULL,"
    " rel TEXT NOT NULL,"
    " weight REAL DEFAULT 1.0,"
    " created_at TEXT,"
    " UNIQUE(src_id, dst_id, rel))"
)

_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_fact_relations_src ON fact_relations(src_id, rel)",
    "CREATE INDEX IF NOT EXISTS idx_fact_relations_dst ON fact_relations(dst_id, rel)",
)


# --------------------------------------------------------------------- store
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
    # Without a provider ``_tx`` we own the transaction: commit unconditionally.
    # A deferred connection is "in transaction" after the write, so testing
    # ``in_transaction`` here would silently discard everything on close.
    if conn is not None:
        try:
            conn.commit()
        except sqlite3.Error:
            pass


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


def _has_table(store, name: str) -> bool:
    try:
        row = _one(store, "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
                   (name,))
    except sqlite3.Error:
        return False
    return row is not None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _as_id(value, label: str) -> int:
    try:
        fid = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid {label}: {value!r}") from None
    if fid < 0:
        raise ValueError(f"invalid {label}: {value!r}")
    return fid


def _as_limit(value, default: int = 50) -> int:
    if value is None:
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        raise ValueError(f"invalid limit: {value!r}") from None


# -------------------------------------------------------------------- labels
_REL_JUNK = re.compile(r"[^a-z0-9]+")


def normalize_rel(rel) -> str:
    """Canonical label: lower-case ``snake_case``, empty labels refused.

    ``"Part Of"``, ``"part-of"`` and ``"PART_OF"`` all become ``"part_of"``.
    """
    text = _REL_JUNK.sub("_", str(rel or "").strip().lower()).strip("_")
    if not text:
        raise ValueError("rel must not be empty")
    return text


def _as_weight(weight) -> float:
    try:
        value = float(weight)
    except (TypeError, ValueError):
        raise ValueError(f"invalid weight: {weight!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"invalid weight: {weight!r}")
    return value


# -------------------------------------------------------------------- schema
def ensure_schema(store) -> dict:
    """Create ``fact_relations`` and both indexes. Idempotent.

    ``CREATE ... IF NOT EXISTS`` only: an existing table (with its rows and its
    UNIQUE constraint) is never rewritten, and no other table is touched.
    """
    _write(store, _SCHEMA)
    for sql in _INDEX_SQL:
        _write(store, sql)
    # A provider with ``_tx`` keeps schema creation in the caller's transaction;
    # a plain connection must be committed here or the DDL is lost on close.
    if not callable(getattr(store, "_tx", None)):
        conn = _conn(store)
        if conn is not None:
            try:
                conn.commit()
            except sqlite3.Error:
                pass
    return {"ok": True, "table": TABLE, "indexes": list(INDEXES)}


# ---------------------------------------------------------------------- CRUD
def _fact_exists(store, fact_id: int) -> bool:
    if not _has_table(store, "facts"):
        return False
    try:
        return _one(store, "SELECT fact_id FROM facts WHERE fact_id = ?", (fact_id,)) is not None
    except sqlite3.Error:
        return False


def link(store, src_id, dst_id, *, rel, weight: float = 1.0) -> dict:
    """Create an edge, or update the weight of the existing one.

    Refused with ``ValueError``: a self-link, an endpoint that is not a live
    fact, an empty label, an unusable weight.  ``created`` in the result says
    whether the row is new; ``created_at`` survives a weight update.
    """
    src, dst = _as_id(src_id, "src_id"), _as_id(dst_id, "dst_id")
    label = normalize_rel(rel)
    value = _as_weight(weight)
    if src == dst:
        raise ValueError(f"refusing self-link on fact {src}")

    ensure_schema(store)
    for fid in (src, dst):
        if not _fact_exists(store, fid):
            raise ValueError(f"fact {fid} not found")

    before = _one(store,
                  "SELECT id, created_at FROM fact_relations "
                  "WHERE src_id = ? AND dst_id = ? AND rel = ?", (src, dst, label))
    with _tx(store):
        _write(
            store,
            "INSERT INTO fact_relations(src_id, dst_id, rel, weight, created_at) "
            "VALUES (?,?,?,?,?) "
            "ON CONFLICT(src_id, dst_id, rel) DO UPDATE SET weight = excluded.weight",
            (src, dst, label, value, _now()),
        )
    row = _one(store,
               "SELECT id, src_id, dst_id, rel, weight, created_at FROM fact_relations "
               "WHERE src_id = ? AND dst_id = ? AND rel = ?", (src, dst, label))
    if row is None:  # pragma: no cover - the upsert either lands or raises
        raise RuntimeError(f"relation {src} -{label}-> {dst} was not stored")
    return {
        "id": int(row["id"]),
        "src_id": int(row["src_id"]),
        "dst_id": int(row["dst_id"]),
        "rel": str(row["rel"]),
        "weight": float(row["weight"]),
        "created_at": row["created_at"],
        "created": before is None,
    }


def unlink(store, src_id, dst_id, *, rel=None) -> int:
    """Delete matching edges and return the number removed.

    ``rel=None`` removes every label between the pair.  A fresh database simply
    has nothing to remove (0); the label is normalised like in :func:`link`.
    """
    src, dst = _as_id(src_id, "src_id"), _as_id(dst_id, "dst_id")
    label = normalize_rel(rel) if rel is not None else None
    ensure_schema(store)
    if label is None:
        sql, params = ("DELETE FROM fact_relations WHERE src_id = ? AND dst_id = ?",
                       (src, dst))
    else:
        sql, params = ("DELETE FROM fact_relations "
                       "WHERE src_id = ? AND dst_id = ? AND rel = ?", (src, dst, label))
    with _tx(store):
        cur = _write(store, sql, params)
    count = getattr(cur, "rowcount", 0)
    return int(count) if count and count > 0 else 0


# ----------------------------------------------------------------- neighbours
def _neighbor_rows(store, fact_id: int, rel: str | None, direction: str,
                   limit: int) -> list[dict]:
    """One direction's rows, already tagged and truncated."""
    has_facts = _has_table(store, "facts")
    endpoint = "dst_id" if direction == "out" else "src_id"
    join = f"LEFT JOIN facts f ON f.fact_id = r.{endpoint}" if has_facts else ""
    content = "f.content" if has_facts else "NULL"
    where = ["r.src_id = ?" if direction == "out" else "r.dst_id = ?"]
    params: list[Any] = [fact_id]
    if rel is not None:
        where.append("r.rel = ?")
        params.append(rel)
    params.append(limit)
    sql = (
        f"SELECT r.{endpoint} AS fact_id, r.rel AS rel, r.weight AS weight, "
        f"{content} AS content FROM fact_relations r {join} "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY r.weight DESC, fact_id ASC, r.rel ASC LIMIT ?"
    )
    out = []
    for row in _rows(store, sql, tuple(params)):
        out.append({
            "fact_id": int(row["fact_id"]),
            "rel": str(row["rel"]),
            "weight": float(row["weight"]),
            "direction": direction,
            "content": str(row.get("content") or "")[:CONTENT_CHARS],
        })
    return out


def neighbors(store, fact_id, *, rel=None, direction: str = "out",
              limit: int = 50) -> list[dict]:
    """Edges incident to ``fact_id``, as ``{fact_id, rel, weight, direction, content}``.

    ``fact_id`` in each row is the *other* endpoint.  Rows are ordered by
    ``weight`` descending then endpoint id ascending; ``content`` is the joined
    fact text truncated to 200 characters.
    """
    fid = _as_id(fact_id, "fact_id")
    label = normalize_rel(rel) if rel is not None else None
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    lim = _as_limit(limit)
    if lim == 0 or not _has_table(store, TABLE):
        return []

    rows: list[dict] = []
    if direction in ("out", "both"):
        rows.extend(_neighbor_rows(store, fid, label, "out", lim))
    if direction in ("in", "both"):
        rows.extend(_neighbor_rows(store, fid, label, "in", lim))

    seen: set[tuple[int, str, str]] = set()
    unique = []
    for row in rows:
        key = (row["fact_id"], row["rel"], row["direction"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    unique.sort(key=lambda r: (-r["weight"], r["fact_id"], r["rel"]))
    return unique[:lim]


# ----------------------------------------------------------------------- path
def _out_edges(store, node: int, rel: str | None) -> list[dict]:
    where = "src_id = ?"
    params: list[Any] = [node]
    if rel is not None:
        where += " AND rel = ?"
        params.append(rel)
    return _rows(store, "SELECT dst_id, rel FROM fact_relations "
                        f"WHERE {where} ORDER BY weight DESC, dst_id ASC", tuple(params))


def path(store, src_id, dst_id, *, max_depth: int = 3,
         rel=None) -> list[dict] | None:
    """Shortest directed path from ``src_id`` to ``dst_id``, or ``None``.

    Breadth-first over outbound edges, so the result has the fewest hops (ties
    broken by weight then id).  Returns the node chain including the source:
    ``[{"fact_id": src, "via_rel": None}, ..., {"fact_id": dst, "via_rel": r}]``.
    The visited set is capped at :data:`MAX_VISITED` nodes, which also makes
    cycles harmless for a graph that is larger than the cap.
    """
    src, dst = _as_id(src_id, "src_id"), _as_id(dst_id, "dst_id")
    label = normalize_rel(rel) if rel is not None else None
    try:
        depth_cap = max(0, int(max_depth))
    except (TypeError, ValueError):
        raise ValueError(f"invalid max_depth: {max_depth!r}") from None

    # A read-only lookup: no ensure_schema, no implicit table creation.
    if not _has_table(store, TABLE):
        return None
    if not _has_table(store, "facts"):
        return None
    if not _fact_exists(store, src) or not _fact_exists(store, dst):
        return None
    if src == dst:
        return [{"fact_id": src, "via_rel": None}]

    parents: dict[int, tuple[int, str]] = {}
    depths = {src: 0}
    visited = {src}
    queue: deque[int] = deque([src])

    while queue:
        node = queue.popleft()
        depth = depths[node]
        if depth >= depth_cap:
            continue
        for edge in _out_edges(store, node, label):
            nxt = int(edge["dst_id"])
            if nxt in visited:
                continue
            visited.add(nxt)
            parents[nxt] = (node, str(edge["rel"]))
            depths[nxt] = depth + 1
            if nxt == dst:
                chain = [{"fact_id": dst, "via_rel": str(edge["rel"])}]
                cur = node
                while cur != src:
                    parent, via = parents[cur]
                    chain.append({"fact_id": cur, "via_rel": via})
                    cur = parent
                chain.append({"fact_id": src, "via_rel": None})
                chain.reverse()
                return chain
            if len(visited) > MAX_VISITED:
                return None
            queue.append(nxt)
    return None


# ---------------------------------------------------------------------- infer
# Attribute words are not subject words: "both facts say RAM" is not evidence
# that they are about the same thing.
_ATTR_WORDS = {
    "memory", "ram", "storage", "disk", "ssd", "nvme", "drive", "cpu",
    "processor", "core", "cores", "port", "ip", "address", "version",
    "release", "clock", "ghz", "mhz", "gb", "mb", "tb", "gib",
}
_INFER_BUCKET_CAP = 1000     # an attribute on more facts than this is not evidence


def _subject_tokens(text: str) -> set[str]:
    return {t for t in holo._tokens(text) if t not in _ATTR_WORDS}


def _attr_vals(attrs: dict[str, set[str]], key: str) -> str:
    return "/".join(sorted(str(v) for v in attrs.get(key, ())))


def infer(store, *, limit: int = 20) -> list[dict]:
    """Propose same-subject attribute edges.  Read-only: nothing is written.

    The attribute keys come from :func:`factkit.holo._attributes` -- the same
    parser the conflict scan uses -- so "16 GB RAM" and "512 GB SSD" are the
    ``ram`` and ``ssd`` slots.  Two facts become a proposal when they share at
    least one subject token *and* at least one attribute key:

    * agreeing values -> ``shares_<attrs>`` (higher confidence)
    * contradictory values -> ``contradicts_<attrs>`` (lower confidence)

    Proposals are ordered by confidence (desc), then ids, capped by ``limit``.
    Edges that already exist in either direction are skipped.
    """
    lim = _as_limit(limit, default=20)
    if lim == 0 or not _has_table(store, "facts"):
        return []

    facts = _rows(store, "SELECT fact_id, content FROM facts ORDER BY fact_id")
    attrs = {int(f["fact_id"]): holo._attributes(str(f.get("content") or ""))
             for f in facts}
    subjects = {int(f["fact_id"]): _subject_tokens(str(f.get("content") or ""))
                for f in facts}

    existing: set[tuple[int, int]] = set()
    if _has_table(store, TABLE):
        for row in _rows(store, "SELECT src_id, dst_id FROM fact_relations"):
            a, b = int(row["src_id"]), int(row["dst_id"])
            existing.add((a, b))
            existing.add((b, a))

    by_attr: dict[str, list[int]] = {}
    for fid, slots in attrs.items():
        for key in slots:
            by_attr.setdefault(key, []).append(fid)

    candidates: set[tuple[int, int]] = set()
    for ids in by_attr.values():
        if len(ids) > _INFER_BUCKET_CAP:
            continue
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                candidates.add((a, b) if a < b else (b, a))

    out: list[dict] = []
    for a, b in candidates:
        if (a, b) in existing:
            continue
        shared_subject = subjects[a] & subjects[b]
        if not shared_subject:
            continue
        shared = sorted(set(attrs[a]) & set(attrs[b]))
        if not shared:
            continue

        agreeing, conflicting = [], []
        for key in shared:
            av, bv = attrs[a][key], attrs[b][key]
            same = all(any(holo._same(key, x, y) for y in bv) for x in av) and \
                all(any(holo._same(key, y, x) for x in av) for y in bv)
            (agreeing if same else conflicting).append(key)
        if not agreeing and not conflicting:
            continue

        keys = agreeing or conflicting
        rel = f"{'shares' if agreeing else 'contradicts'}_{'_'.join(keys)}"
        if agreeing:
            detail = ", ".join(f"{k}={_attr_vals(attrs[a], k)}" for k in agreeing)
            reason = f"same subject ({'/'.join(sorted(shared_subject))}) and attribute: {detail}"
            base = 0.62
        else:
            detail = ", ".join(
                f"{k}: {_attr_vals(attrs[a], k)} vs {_attr_vals(attrs[b], k)}"
                for k in conflicting)
            reason = f"same subject ({'/'.join(sorted(shared_subject))}), conflicting {detail}"
            base = 0.45

        union = subjects[a] | subjects[b]
        overlap = len(shared_subject) / len(union) if union else 0.0
        confidence = round(min(0.95, base + min(0.2, 0.05 * len(keys)) + 0.2 * overlap), 3)
        out.append({"src_id": a, "dst_id": b, "rel": rel,
                    "reason": reason, "confidence": confidence})

    out.sort(key=lambda p: (-p["confidence"], p["src_id"], p["dst_id"], p["rel"]))
    return out[:lim]


# ---------------------------------------------------------------------- stats
def stats(store) -> dict:
    """Graph counters.  Pure read; a store without either table yields zeros.

    ``orphan_facts`` counts live facts with no edge at all, ``max_degree`` the
    highest number of incident edges on one fact.  ``rels`` and ``orphans`` are
    short aliases of ``distinct_rels`` and ``orphan_facts``.
    """
    edges = rels = max_degree = 0
    if _has_table(store, TABLE):
        row = _one(store, "SELECT COUNT(*) AS n, COUNT(DISTINCT rel) AS r FROM fact_relations")
        if row is not None:
            edges, rels = int(row["n"] or 0), int(row["r"] or 0)
        if edges:
            degree = _one(
                store,
                "SELECT MAX(n) AS m FROM ("
                " SELECT fid, COUNT(*) AS n FROM ("
                "  SELECT src_id AS fid FROM fact_relations"
                "  UNION ALL SELECT dst_id FROM fact_relations"
                " ) GROUP BY fid)")
            if degree is not None:
                max_degree = int(degree["m"] or 0)

    orphans = 0
    if _has_table(store, "facts"):
        row = _one(
            store,
            "SELECT COUNT(*) AS n FROM facts f WHERE NOT EXISTS ("
            " SELECT 1 FROM fact_relations r"
            " WHERE r.src_id = f.fact_id OR r.dst_id = f.fact_id)") \
            if _has_table(store, TABLE) else _one(store, "SELECT COUNT(*) AS n FROM facts")
        if row is not None:
            orphans = int(row["n"] or 0)

    return {
        "edges": edges,
        "rels": rels,
        "distinct_rels": rels,
        "orphan_facts": orphans,
        "orphans": orphans,
        "max_degree": max_degree,
    }
