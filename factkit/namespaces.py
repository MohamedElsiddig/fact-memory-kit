"""Namespaced recall: partition a fact store into named scopes.

Why this module exists
----------------------
A single flat fact store mixes unrelated lives: work notes, homelab specs,
personal preferences.  Recall then has to *post-filter* whatever the embedding
channel returned, which wastes the candidate budget and can starve a small
namespace of hits.  Namespaces make the partition explicit so a caller can ask
"what do I know in *project-atlas* about backups?" instead of "what do I know
about backups?".

How a namespace is stored
-------------------------
The provider exposes two candidate mechanisms and this module detects which one
a store *actually* supports before using it; nothing is assumed from a version
string or from the mere presence of a table.

1. **Banks** (``memory_banks`` + a membership channel).  A bank is usable as a
   namespace only when the schema can answer *which facts belong to it*: a link
   table (``fact_banks`` / ``bank_facts`` / ``memory_bank_facts``) or a
   ``facts.bank_id`` column.  A registry alone is **not** enough.

   This matters on the live holographic store: ``memory_banks`` exists, but it
   is an HRR *category vector cache* keyed ``cat:<category>`` with a NOT NULL
   ``vector``.  There is no fact→bank membership column or join table, and the
   current provider no longer even reads banks (``_invalidate_bank`` drops them
   and rebuilds lazily).  Treating that table as membership would invent data
   and corrupt the cache, so :func:`detect` reports ``banks: False`` there and
   namespaces fall back to tags.

2. **Tags** (default).  A namespace is the tag ``ns:<name>`` on a fact.  Tags
   are already a comma-separated multi-value column, survive category changes,
   and are indexed by the provider's FTS triggers, so assignment is a plain
   fact update.

3. **Categories** are reported for information only.  ``facts.category`` is
   single-valued: using it as a namespace would overwrite the fact's
   classification, so :func:`assign` refuses on a category-only store and
   :func:`list_namespaces` exposes categories read-only.

Names are case-sensitive, at most 64 characters, and restricted to letters,
digits and ``_ . / -``.  No spaces or commas: the tag column is comma-separated
and a name that needs quoting is a name that will be mis-parsed later.

Filtering is always post-hoc
----------------------------
:func:`scope_query` asks the semantic channel for ``limit * 4 + 4`` candidates
and then drops the ones outside the namespace.  It is *not* an index-level
filter: a namespace that holds few facts can return fewer than ``limit`` hits
even when the store is large.  ``dropped`` reports how many candidates the
namespace filter removed; ``truncated`` reports how many in-namespace hits the
``limit`` cut.  Callers that need a guaranteed fill must raise ``limit``.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from . import holo

TAG_PREFIX = "ns:"
MAX_NAME = 64
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._/-]*$")

_BANK_LINK_TABLES = ("fact_banks", "bank_facts", "member_banks", "memory_bank_facts")
_BANK_REGISTRY_TABLES = ("memory_banks", "banks")
_COUNT_COLUMNS = ("facts", "fact_count", "count")


# ------------------------------------------------------------------ plumbing
def _path(store) -> str:
    if isinstance(store, (str, Path)):
        return str(Path(store).expanduser())
    return str(getattr(store, "db_path", "") or "")


def _read(store, sql: str, params=()) -> list[dict]:
    """One read; uses the store's own lock when it exposes ``read``."""
    if hasattr(store, "read"):
        return [dict(row) for row in store.read(sql, params)]
    path = _path(store)
    if not path:
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql, params)]
    finally:
        conn.close()


def _execute(store, sql: str, params=()) -> int:
    """One write statement; rowcount.  Prefers the provider's transaction path."""
    if hasattr(store, "_write") and hasattr(store, "_tx"):
        with store._tx():
            return int(store._write(sql, params).rowcount)
    conn = sqlite3.connect(_path(store), timeout=10.0)
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        return int(cur.rowcount)
    finally:
        conn.close()


def _tables(store) -> set[str]:
    try:
        return {str(r["name"]) for r in _read(
            store, "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    except sqlite3.Error:
        return set()


def _columns(store, table: str) -> list[dict]:
    """PRAGMA columns of a known table; the name is checked against the schema."""
    if table not in _tables(store):
        return []
    try:
        return _read(store, f"PRAGMA table_info({table})")
    except sqlite3.Error:
        return []


def _column_names(store, table: str) -> set[str]:
    return {str(c.get("name")) for c in _columns(store, table)}


# -------------------------------------------------------------------- names
def _clean(namespace) -> str:
    text = "" if namespace is None else str(namespace).strip()
    if text.lower().startswith(TAG_PREFIX):
        text = text[len(TAG_PREFIX):].strip()
    return text


def _valid(name: str) -> bool:
    return bool(name) and len(name) <= MAX_NAME and bool(_NAME_RE.match(name))


def _tag_name(namespace: str) -> str:
    return TAG_PREFIX + namespace


def _tag_list(value) -> list[str]:
    """Tags from a list, a JSON-ish list string, or a comma/semicolon string."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        parts = [str(item) for item in value]
    else:
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                import json
                data = json.loads(text)
                parts = [str(item) for item in data] if isinstance(data, list) else [text]
            except (ValueError, TypeError):
                parts = re.split(r"[,;]", text)
        else:
            parts = re.split(r"[,;]", text)
    out: list[str] = []
    for part in parts:
        tag = part.strip()
        if tag and tag not in out:
            out.append(tag)
    return out


def _tag_namespaces(value) -> list[str]:
    out: list[str] = []
    for tag in _tag_list(value):
        if tag.lower().startswith(TAG_PREFIX):
            name = tag[len(TAG_PREFIX):].strip()
            if name and name not in out:
                out.append(name)
    return out


# ---------------------------------------------------------------- detection
def _bank_channel(store, tables: set[str]) -> dict | None:
    """Describe the membership channel, or None when banks cannot hold namespaces."""
    columns = _column_names(store, "facts")
    for table in _BANK_LINK_TABLES:
        if table not in tables:
            continue
        cols = _column_names(store, table)
        if "fact_id" not in cols:
            continue
        if "bank_name" in cols or "namespace" in cols:
            name_col = "bank_name" if "bank_name" in cols else "namespace"
            return {"table": table, "id_col": None, "name_col": name_col,
                    "registry": None}
        if "bank_id" in cols:
            registry = next((t for t in _BANK_REGISTRY_TABLES if t in tables), None)
            if registry and "bank_name" in _column_names(store, registry):
                return {"table": table, "id_col": "bank_id", "name_col": None,
                        "registry": registry}
    # A fact column works only when a registry exists to hold the namespace row.
    for col in ("bank_id", "namespace"):
        if col not in columns:
            continue
        registry = next((t for t in _BANK_REGISTRY_TABLES if t in tables), None)
        if registry and "bank_name" in _column_names(store, registry):
            return {"table": "facts", "id_col": None, "name_col": None,
                    "registry": registry, "fact_column": col}
    return None


def detect(store) -> dict:
    """Report the namespace mechanisms the store's schema actually supports.

    Read-only: two sqlite_master / PRAGMA reads and nothing else.  ``banks``
    requires a membership channel, not just the ``memory_banks`` registry — see
    the module docstring for why the live provider's bank table does not count.
    """
    tables = _tables(store)
    columns = []
    if "facts" in tables:
        columns = [str(c.get("name")) for c in _columns(store, "facts")]
    return {
        "banks": _bank_channel(store, tables) is not None,
        "tags": "tags" in columns,
        "categories": "category" in columns,
        "columns": columns,
    }


def _mechanism(store, info: dict | None = None) -> str:
    info = info or detect(store)
    if info["banks"]:
        return "banks"
    if info["tags"]:
        return "tags"
    return "categories" if info["categories"] else "none"


# ------------------------------------------------------------------ listing
def _tag_counts(store, has_updated_at: bool) -> list[dict]:
    select = "SELECT tags" + (", updated_at" if has_updated_at else "") + " FROM facts"
    counts: dict[str, int] = {}
    seen: dict[str, str] = {}
    for row in _read(store, select):
        stamp = str(row.get("updated_at") or "") if has_updated_at else ""
        for name in _tag_namespaces(row.get("tags")):
            counts[name] = counts.get(name, 0) + 1
            if stamp > seen.get(name, ""):
                seen[name] = stamp
    return [{"namespace": name, "facts": count,
             "last_seen": seen.get(name) or None, "mechanism": "tags"}
            for name, count in counts.items()]


def _bank_counts(store, channel: dict | None) -> list[dict]:
    if channel is None:
        return []
    table = channel["table"]
    where = f" WHERE l.{channel['name_col']} LIKE '{TAG_PREFIX}%'" if channel["name_col"] else ""
    if channel["name_col"]:
        sql = (f"SELECT l.{channel['name_col']} AS ns, COUNT(*) AS n "
               f"FROM {table} l{where} GROUP BY 1")
    else:
        sql = (f"SELECT b.bank_name AS ns, COUNT(*) AS n FROM {table} l "
               f"JOIN {channel['registry']} b ON b.bank_id = l.bank_id "
               f"WHERE b.bank_name LIKE '{TAG_PREFIX}%' GROUP BY 1")
    try:
        rows = _read(store, sql)
    except sqlite3.Error:
        return []

    seen: dict[str, str] = {}
    if "updated_at" in _column_names(store, "facts"):
        try:
            if channel["name_col"]:
                last_sql = (f"SELECT l.{channel['name_col']} AS ns, MAX(f.updated_at) AS last_seen "
                            f"FROM {table} l JOIN facts f ON f.fact_id = l.fact_id "
                            f"GROUP BY 1")
            else:
                last_sql = (f"SELECT b.bank_name AS ns, MAX(f.updated_at) AS last_seen "
                            f"FROM {table} l JOIN facts f ON f.fact_id = l.fact_id "
                            f"JOIN {channel['registry']} b ON b.bank_id = l.bank_id "
                            f"GROUP BY 1")
            seen = {str(r["ns"]): str(r["last_seen"] or "") for r in _read(store, last_sql)}
        except sqlite3.Error:
            seen = {}

    out = []
    for row in rows:
        bank = str(row.get("ns") or "")
        name = _clean(bank)
        if not name:
            continue
        out.append({"namespace": name, "facts": int(row.get("n") or 0),
                    "last_seen": seen.get(bank) or None, "mechanism": "banks"})
    return out


def _category_counts(store) -> list[dict]:
    has_stamp = "updated_at" in _column_names(store, "facts")
    select = "SELECT category, COUNT(*) AS n" + (", MAX(updated_at) AS last_seen" if has_stamp else "")
    out = []
    for row in _read(store, select + " FROM facts GROUP BY category"):
        name = str(row.get("category") or "").strip()
        if not name:
            continue
        out.append({"namespace": name, "facts": int(row.get("n") or 0),
                    "last_seen": row.get("last_seen") or None, "mechanism": "categories"})
    return out


def list_namespaces(store) -> list[dict]:
    """Namespaces with fact counts and ``last_seen``; most populous first.

    Banks when the schema supports membership, otherwise tags, otherwise the
    read-only category view.  Ties break on the name, so ordering is stable.
    """
    info = detect(store)
    if info["banks"]:
        rows = _bank_counts(store, _bank_channel(store, _tables(store)))
    elif info["tags"]:
        rows = _tag_counts(store, "updated_at" in info["columns"])
    elif info["categories"]:
        rows = _category_counts(store)
    else:
        rows = []
    return sorted(rows, key=lambda r: (-r["facts"], r["namespace"]))


def _known(store, namespace: str, mechanism: str) -> bool:
    if mechanism == "banks":
        channel = _bank_channel(store, _tables(store))
        if channel is None:
            return False
        bank = _tag_name(namespace)
        if channel["name_col"]:
            rows = _read(store, f"SELECT 1 FROM {channel['table']} "
                                f"WHERE {channel['name_col']} = ? LIMIT 1", (bank,))
        else:
            rows = _read(store, f"SELECT 1 FROM {channel['registry']} "
                                f"WHERE bank_name = ? LIMIT 1", (bank,))
        return bool(rows)
    if mechanism == "tags":
        for row in _read(store, "SELECT tags FROM facts"):
            if namespace in _tag_namespaces(row.get("tags")):
                return True
        return False
    return False


# ------------------------------------------------------------------- assign
def assign(store, fact_id, namespace, *, create: bool = False) -> dict:
    """Put one fact in a namespace.  Idempotent; ``create=False`` is the default.

    Banks when the store can hold membership, otherwise the ``ns:<name>`` tag.
    Unknown namespace + ``create=False`` is refused rather than silently
    invented, so a typo cannot scatter facts into a new scope.
    """
    name = _clean(namespace)
    if not _valid(name):
        return {"ok": False, "reason": f"invalid namespace {namespace!r}: "
                                       f"1-{MAX_NAME} chars of letters, digits, _ . / -"}

    info = detect(store)
    mechanism = _mechanism(store, info)
    if mechanism == "categories":
        return {"ok": False, "namespace": name, "mechanism": mechanism,
                "reason": "store has no tag or bank channel; a category is single-valued"}
    if mechanism == "none":
        return {"ok": False, "namespace": name, "mechanism": mechanism,
                "reason": "store has neither a tags column nor a bank membership channel"}

    try:
        fact = _read(store, "SELECT fact_id FROM facts WHERE fact_id = ?", (int(fact_id),))
    except (sqlite3.Error, TypeError, ValueError) as exc:
        return {"ok": False, "namespace": name, "mechanism": mechanism,
                "reason": f"fact lookup failed: {exc}"}
    if not fact:
        return {"ok": False, "namespace": name, "mechanism": mechanism,
                "reason": f"fact {fact_id} not found"}

    if mechanism == "tags":
        return _assign_tag(store, int(fact_id), name, create=create)
    return _assign_bank(store, int(fact_id), name, create=create)


def _assign_tag(store, fact_id: int, name: str, *, create: bool) -> dict:
    row = _read(store, "SELECT tags FROM facts WHERE fact_id = ?", (fact_id,))[0]
    tags = _tag_list(row.get("tags"))
    tag = _tag_name(name)
    if tag in tags:
        return {"ok": True, "fact_id": fact_id, "namespace": name,
                "mechanism": "tags", "created": False, "changed": False}
    if not create and not _known(store, name, "tags"):
        return {"ok": False, "fact_id": fact_id, "namespace": name, "mechanism": "tags",
                "reason": f"unknown namespace {name!r}; pass create=True"}
    tags.append(tag)
    try:
        _execute(store, "UPDATE facts SET tags = ? WHERE fact_id = ?",
                 (", ".join(tags), fact_id))
    except sqlite3.Error as exc:
        return {"ok": False, "fact_id": fact_id, "namespace": name,
                "mechanism": "tags", "reason": f"write failed: {exc}"}
    return {"ok": True, "fact_id": fact_id, "namespace": name,
            "mechanism": "tags", "created": False, "changed": True}


def _assign_bank(store, fact_id: int, name: str, *, create: bool) -> dict:
    channel = _bank_channel(store, _tables(store))
    if channel is None:  # detection said banks, schema changed underneath us
        return {"ok": False, "fact_id": fact_id, "namespace": name,
                "mechanism": "banks", "reason": "bank membership channel disappeared"}
    bank = _tag_name(name)
    created = False

    if channel["name_col"]:
        exists = _read(store, f"SELECT 1 FROM {channel['table']} "
                              f"WHERE {channel['name_col']} = ? LIMIT 1", (bank,))
    else:
        exists = _read(store, f"SELECT 1 FROM {channel['registry']} "
                              f"WHERE bank_name = ? LIMIT 1", (bank,))

    if not exists:
        if not create:
            return {"ok": False, "fact_id": fact_id, "namespace": name,
                    "mechanism": "banks",
                    "reason": f"unknown namespace {name!r}; pass create=True"}
        try:
            _create_bank(store, channel, bank)
        except (sqlite3.Error, ValueError) as exc:
            return {"ok": False, "fact_id": fact_id, "namespace": name,
                    "mechanism": "banks", "reason": f"bank create failed: {exc}"}
        created = True

    try:
        if channel["name_col"]:
            changed = _execute(
                store,
                f"INSERT OR IGNORE INTO {channel['table']} "
                f"(fact_id, {channel['name_col']}) VALUES (?, ?)", (fact_id, bank))
        elif channel.get("fact_column"):
            changed = _execute(
                store,
                f"UPDATE facts SET {channel['fact_column']} = "
                f"(SELECT bank_id FROM {channel['registry']} WHERE bank_name = ?) "
                f"WHERE fact_id = ?", (bank, fact_id))
        else:
            row = _read(store, f"SELECT bank_id FROM {channel['registry']} "
                               f"WHERE bank_name = ?", (bank,))[0]
            changed = _execute(
                store,
                f"INSERT OR IGNORE INTO {channel['table']} (fact_id, bank_id) "
                f"VALUES (?, ?)", (fact_id, int(row["bank_id"])))
    except (sqlite3.Error, IndexError) as exc:
        return {"ok": False, "fact_id": fact_id, "namespace": name,
                "mechanism": "banks", "reason": f"assign failed: {exc}"}

    _refresh_bank_count(store, channel, bank)
    return {"ok": True, "fact_id": fact_id, "namespace": name, "mechanism": "banks",
            "created": created, "changed": bool(changed)}


def _create_bank(store, channel: dict, bank: str) -> None:
    """Insert the registry row, or refuse if the provider demands a vector we cannot invent."""
    registry = channel["registry"]
    if registry is None:
        raise ValueError("link table has no bank registry to create rows in")
    cols = _columns(store, registry)
    names = {str(c.get("name")) for c in cols}
    if "bank_name" not in names:
        raise ValueError(f"{registry} has no bank_name column")
    values = {"bank_name": bank}
    # A NOT NULL column with no default would have to be fabricated (a HRR
    # vector, a dimension).  Never invent one: report the schema requirement.
    missing = [str(c["name"]) for c in cols
               if c.get("notnull") and c.get("dflt_value") is None
               and not c.get("pk") and str(c["name"]) not in values]
    if missing:
        raise ValueError(f"{registry} requires {', '.join(missing)}; "
                         "cannot create a namespace bank without a real provider vector")
    _execute(store, f"INSERT INTO {registry} ({', '.join(values)}) "
                    f"VALUES ({', '.join('?' for _ in values)})", tuple(values.values()))


def _refresh_bank_count(store, channel: dict, bank: str) -> None:
    """Best-effort fact_count for the registry; membership is the link table's truth."""
    if not channel["registry"] or channel["name_col"]:
        return
    try:
        _execute(store,
                 f"UPDATE {channel['registry']} SET fact_count = "
                 f"(SELECT COUNT(*) FROM {channel['table']} l WHERE l.bank_id = "
                 f"{channel['registry']}.bank_id) WHERE bank_name = ?", (bank,))
    except sqlite3.Error:
        pass


# ------------------------------------------------------------------ filtering
def filter_rows(rows, namespace) -> list:
    """Keep rows whose tags include ``ns:<namespace>``; order is preserved.

    Accepts dicts or sqlite3.Row, with ``tags`` as a list or as the provider's
    comma-separated string.  A row with no tags is simply not a member.
    """
    name = _clean(namespace)
    if not name:
        return []
    tag = _tag_name(name)
    out = []
    for row in rows:
        data = dict(row) if not isinstance(row, dict) else row
        if tag in _tag_list(data.get("tags")):
            out.append(row)
    return out


def _members(store, name: str) -> set[int]:
    """fact ids in a bank namespace (used when tags are not the mechanism)."""
    channel = _bank_channel(store, _tables(store))
    if channel is None:
        return set()
    bank = _tag_name(name)
    try:
        if channel["name_col"]:
            rows = _read(store, f"SELECT fact_id FROM {channel['table']} "
                                f"WHERE {channel['name_col']} = ?", (bank,))
        elif channel.get("fact_column"):
            rows = _read(store, f"SELECT f.fact_id FROM facts f JOIN {channel['registry']} b "
                                f"ON b.bank_id = f.{channel['fact_column']} "
                                f"WHERE b.bank_name = ?", (bank,))
        else:
            rows = _read(store, f"SELECT l.fact_id FROM {channel['table']} l "
                                f"JOIN {channel['registry']} b ON b.bank_id = l.bank_id "
                                f"WHERE b.bank_name = ?", (bank,))
    except sqlite3.Error:
        return set()
    return {int(r["fact_id"]) for r in rows}


def _tags_for(store, fact_ids) -> dict[int, str | None]:
    ids = sorted({int(f) for f in fact_ids})
    if not ids or "tags" not in _column_names(store, "facts"):
        return {}
    marks = ", ".join("?" for _ in ids)
    try:
        rows = _read(store, f"SELECT fact_id, tags FROM facts WHERE fact_id IN ({marks})", ids)
    except sqlite3.Error:
        return {}
    return {int(r["fact_id"]): r.get("tags") for r in rows}


# -------------------------------------------------------------------- recall
def scope_query(store, message, namespace, *, limit: int = 5, **kwargs) -> dict:
    """Semantic recall restricted to one namespace, by post-filtering.

    ``holo.prefetch`` is asked for ``limit * 4 + 4`` candidates, hits outside
    the namespace are dropped, and the rest are cut to ``limit``.  This is not
    an index-level filter: a namespace holding few facts returns fewer than
    ``limit`` even in a large store, and ``dropped`` / ``truncated`` make that
    visible to the caller.  ``available`` is False when the embedding endpoint
    is down (``prefetch`` returns ``None``) — the caller can trip its breaker.
    """
    name = _clean(namespace)
    mechanism = _mechanism(store)
    window = max(int(limit), 0) * 4 + 4
    kwargs.pop("limit", None)  # the window is this module's decision, not the caller's
    raw = holo.prefetch(store, message, limit=window, **kwargs)
    out = {"namespace": name, "for_query": message, "hits": [],
           "dropped": 0, "truncated": 0, "candidates": 0,
           "mechanism": mechanism, "available": raw is not None}
    if not name or raw is None:
        return out

    rows = [dict(hit) for hit in raw]
    if mechanism == "banks":
        members = _members(store, name)
        inside = [row for row in rows if int(row.get("fact_id", -1)) in members]
    else:
        tags = _tags_for(store, [row.get("fact_id") for row in rows if "fact_id" in row])
        for row in rows:
            row["tags"] = tags.get(int(row["fact_id"])) if "fact_id" in row else None
        inside = filter_rows(rows, name)

    out["candidates"] = len(rows)
    out["dropped"] = len(rows) - len(inside)
    out["truncated"] = max(0, len(inside) - max(int(limit), 0))
    out["hits"] = inside[:max(int(limit), 0)]
    return out


# --------------------------------------------------------------------- stats
def stats(store, namespace=None) -> dict:
    """Fact counts per namespace, or one namespace's count and share of the store."""
    total = 0
    if "facts" in _tables(store):
        try:
            rows = _read(store, "SELECT COUNT(*) AS n FROM facts")
            total = int(rows[0]["n"]) if rows else 0
        except sqlite3.Error:
            total = 0
    rows = list_namespaces(store)
    mechanism = _mechanism(store)
    if namespace is not None:
        name = _clean(namespace)
        row = next((r for r in rows if r["namespace"] == name), None)
        facts = int(row["facts"]) if row else 0
        return {"namespace": name, "mechanism": mechanism, "total_facts": total,
                "facts": facts, "share": round(facts / total, 4) if total else 0.0,
                "last_seen": row["last_seen"] if row else None}
    return {
        "mechanism": mechanism,
        "total_facts": total,
        "count": len(rows),
        "namespaces": {r["namespace"]: {
            "facts": int(r["facts"]),
            "share": round(int(r["facts"]) / total, 4) if total else 0.0,
            "last_seen": r["last_seen"],
        } for r in rows},
    }
