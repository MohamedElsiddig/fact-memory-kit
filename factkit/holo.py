"""Maintenance operations over a live holographic store.

All writes go through the holographic plugin's own ``MemoryStore`` API
(``add_fact_status`` / ``update_fact`` / ``replace_fact`` / ``remove_fact``), so
FTS, entity links, HRR vectors, trust scores and bank invalidation stay
consistent — the same path the agent's ``fact_store`` tool uses.

Detection is read-only. Every operation supports ``dry_run`` and records what it
did in the ``audit`` table, written by factkit with actor ``factkit``.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import sqlite3
import sys
import types
from pathlib import Path

logger = logging.getLogger(__name__)

PLUGIN_DIR = Path.home() / ".hermes" / "plugins" / "holographic"


# --------------------------------------------------------------------- store
def _install_shims() -> None:
    """Stand-ins for Hermes runtime helpers when factkit runs outside the gateway."""
    try:
        import hermes_state_wal  # noqa: F401
    except ImportError:
        mod = types.ModuleType("hermes_state_wal")

        def apply_wal_with_fallback(conn, **_kw):
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                pass

        mod.apply_wal_with_fallback = apply_wal_with_fallback
        sys.modules["hermes_state_wal"] = mod
    try:
        from hermes_cli.sqlite_util import add_column_if_missing  # noqa: F401
    except ImportError:
        pkg = types.ModuleType("hermes_cli")
        pkg.__path__ = []  # namespace only, no discovery
        util = types.ModuleType("hermes_cli.sqlite_util")

        def add_column_if_missing(conn, table, column, decl):
            cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

        util.add_column_if_missing = add_column_if_missing
        sys.modules.setdefault("hermes_cli", pkg)
        sys.modules["hermes_cli.sqlite_util"] = util


def load_plugin_store(db_path: str | Path | None = None):
    """Import the holographic plugin's MemoryStore without running its __init__."""
    _install_shims()
    if "holographic" not in sys.modules:
        pkg = types.ModuleType("holographic")
        pkg.__path__ = [str(PLUGIN_DIR)]
        sys.modules["holographic"] = pkg
    from holographic.store import MemoryStore  # noqa: WPS433 - deliberate late import

    path = str(Path(db_path).expanduser()) if db_path else str(Path.home() / ".hermes" / "memory_store.db")
    return MemoryStore(path)


def _ro_conn(path: str | Path):
    conn = sqlite3.connect(f"file:{Path(path).expanduser()}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


# ------------------------------------------------------------ conflict scan
STOP = {
    "the", "and", "for", "with", "that", "this", "from", "has", "have", "was",
    "are", "its", "his", "her", "their", "not", "but", "you", "what", "when",
    "which", "how", "there", "these", "those", "into", "about", "will", "can",
    "full", "specs", "more", "info", "details", "note", "notes", "file",
}

# Attribute = (name, value). Only equal attribute names can contradict: two
# different paths are not a disagreement, two different RAM sizes are.
ATTR_PATTERNS = [
    # 16 GB RAM / 512 gb ssd / 6 cores
    ("unit", re.compile(r"\b(\d+(?:\.\d+)?)\s*(gb|tb|mb)\s+(ram|memory|ssd|storage|disk|swap)\b")),
    ("cores", re.compile(r"\b(\d+)\s*(?:cores?|threads?)\b")),
    # 2.4 GHz
    ("clock", re.compile(r"\b(\d+(?:\.\d+)?)\s*ghz\b")),
    # port 8080 / port=8080 / :8080
    ("port", re.compile(r"\bport\s*[=:]?\s*(\d{2,5})\b")),
    ("port", re.compile(r"(?<![\d.]):(\d{2,5})\b")),
    # 10.0.0.5
    ("ip", re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")),
    # caddy 2.7.6 / python3.12 / v1.2.3
    ("version", re.compile(r"\b([a-z][a-z0-9_-]{2,})\s+v?(\d+\.\d+(?:\.\d+)?)\b")),
]

# Paths, dates and generic locations are context, not attributes.
VOLATILE = ("port", "version", "ip", "clock", "unit", "cores")


def _tokens(text: str) -> set[str]:
    t = set(re.findall(r"[a-z0-9_-]+", (text or "").lower()))
    return {x for x in t if len(x) > 2 and x not in STOP and not x.isdigit()}


def _attributes(text: str) -> dict[str, set[str]]:
    low = (text or "").lower()
    out: dict[str, set[str]] = {}

    def put(name: str, value: str) -> None:
        out.setdefault(name, set()).add(value)

    for name, rx in ATTR_PATTERNS:
        for m in rx.finditer(low):
            if name == "unit":
                put(m.group(3), f"{m.group(1)}{m.group(2)}")
            elif name == "version":
                put(m.group(1), m.group(2))
            else:
                put(name, m.group(1))
    # a bare "16 GB" with no noun is storage/memory only when another fact
    # names the same attribute; do not guess here.
    return out


GENERIC_ATTRS = {"port", "ip", "clock", "cores"}


def _value_conflict(a: dict, b: dict, shared=()) -> list[tuple[str, set, set]]:
    aa, ab = _attributes(a["content"]), _attributes(b["content"])
    out = []
    for k in set(aa) & set(ab):
        if aa[k] == ab[k]:
            continue
        # A bare port or IP is only a conflict when both facts are about the
        # same declared subject; otherwise two services may legitimately differ.
        if k in GENERIC_ATTRS and not shared:
            continue
        out.append((k, aa[k], ab[k]))
    return out


def find_conflicts(db_path: str | Path, overlap_min: float = 0.35,
                   entity_min: float = 0.15) -> list[dict]:
    """Same declared subject + same attribute + different value = conflict.

    Two gates, both required: the facts must share a *declared* entity (or be
    near-duplicates by content), and they must disagree on the same named
    attribute. Similarity alone never creates a conflict.
    """
    conn = _ro_conn(db_path)
    try:
        facts = [dict(r) for r in conn.execute(
            "SELECT fact_id, content, category, trust_score, created_at, updated_at FROM facts")]
        links: dict[int, set[str]] = {}
        for r in conn.execute(
                "SELECT fe.fact_id, e.name FROM fact_entities fe "
                "JOIN entities e ON e.entity_id = fe.entity_id "
                "WHERE e.entity_type = 'declared'"):
            links.setdefault(int(r["fact_id"]), set()).add(r["name"].lower())
    finally:
        conn.close()

    toks = {f["fact_id"]: _tokens(f["content"]) for f in facts}
    out = []
    for i, a in enumerate(facts):
        for b in facts[i + 1:]:
            ta, tb = toks[a["fact_id"]], toks[b["fact_id"]]
            jac = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
            shared = set(links.get(a["fact_id"], ())) & set(links.get(b["fact_id"], ()))
            if not (jac >= overlap_min or (shared and jac >= entity_min)):
                continue
            diffs = _value_conflict(a, b, shared)
            if not diffs:
                continue
            newer, older = (a, b) if (str(a["updated_at"]), a["fact_id"]) >= (str(b["updated_at"]), b["fact_id"]) else (b, a)
            out.append({
                "keep": newer["fact_id"], "retire": older["fact_id"],
                "keep_text": newer["content"], "retire_text": older["content"],
                "keep_updated": newer["updated_at"], "retire_updated": older["updated_at"],
                "shared_entities": sorted(shared), "jaccard": round(jac, 3),
                "diffs": [{"slot": s_, "keep": sorted(k), "retire": sorted(r)} for s_, k, r in diffs],
            })
    return out


# ------------------------------------------------------------------ apply
def apply_conflict(store, pair: dict, *, dry_run: bool = True) -> dict:
    """Retire the older fact, keeping the newer text as the single truth."""
    keep_id, retire_id = int(pair["keep"]), int(pair["retire"])
    keep = store._one("SELECT fact_id, content, category, tags FROM facts WHERE fact_id = ?", (keep_id,))
    retire = store._one("SELECT fact_id, content FROM facts WHERE fact_id = ?", (retire_id,))
    if keep is None or retire is None:
        return {"ok": False, "reason": "fact disappeared"}
    if keep["content"] == retire["content"]:
        return {"ok": False, "reason": "identical content"}

    if dry_run:
        return {"ok": True, "dry_run": True, "keep": keep_id, "retire": retire_id,
                "action": "update keep + remove retire"}

    # 1. keep the authoritative text (identical => already correct, no touch)
    store.update_fact(keep_id, content=keep["content"], category=keep["category"],
                      tags=keep["tags"], touch=False)
    # 2. retire the stale one through the plugin's own delete path
    store.remove_fact(retire_id)
    # 3. provenance
    with store._tx():
        store._write(
            "INSERT INTO audit(action, fact_id, actor, reason, before, after) VALUES (?,?,?,?,?,?)",
            ("conflict-apply", keep_id, "factkit",
             f"retired #{retire_id} on slot {','.join(d['slot'] for d in pair['diffs'])}",
             retire["content"], keep["content"]))
    return {"ok": True, "keep": keep_id, "retire": retire_id, "removed": True}


# --------------------------------------------------------------- TTL flags
def find_stale(db_path: str | Path, days: int = 90, slot_days: int = 180) -> list[dict]:
    """Flag old facts, and older facts that hold volatile (rot-prone) slots."""
    conn = _ro_conn(db_path)
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT fact_id, content, category, trust_score, created_at, updated_at FROM facts")]
    finally:
        conn.close()
    out = []
    for r in rows:
        try:
            from datetime import datetime as _dt
            age = (_dt.now() - _dt.fromisoformat(str(r["updated_at"]))).days
        except ValueError:
            continue
        slots = _attributes(r["content"])
        volatile = [name for name in VOLATILE if name in slots]
        if age >= days:
            out.append({"fact_id": r["fact_id"], "age_days": age, "reason": "age",
                        "slots": volatile, "content": r["content"]})
        elif volatile and age >= slot_days:
            out.append({"fact_id": r["fact_id"], "age_days": age, "reason": "stale-slot",
                        "slots": volatile, "content": r["content"]})
    return sorted(out, key=lambda x: -x["age_days"])


def touch_fact(store, fact_id: int, *, dry_run: bool = True, reason: str = "reviewed, still correct") -> dict:
    """Confirm a fact is still true: bump updated_at through the store API."""
    if dry_run:
        return {"ok": True, "dry_run": True, "fact_id": fact_id}
    ok = store.update_fact(int(fact_id), content=None, touch=True)
    with store._tx():
        store._write("INSERT INTO audit(action, fact_id, actor, reason) VALUES (?,?,?,?)",
                     ("review-touch", int(fact_id), "factkit", reason))
    return {"ok": bool(ok), "fact_id": int(fact_id)}


# ------------------------------------------------------------------ merge
def merge_duplicates(db_path: str | Path, threshold: float = 0.92) -> list[dict]:
    """Near-identical facts: keep the highest trust (then newest), retire the rest."""
    conn = _ro_conn(db_path)
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT fact_id, content, trust_score, updated_at FROM facts")]
    finally:
        conn.close()
    from difflib import SequenceMatcher
    out = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            ratio = SequenceMatcher(None, a["content"].lower(), b["content"].lower()).ratio()
            if ratio < threshold:
                continue
            keep, drop = sorted([a, b], key=lambda r: (r["trust_score"], r["updated_at"]), reverse=True)
            out.append({"keep": keep["fact_id"], "retire": drop["fact_id"],
                        "ratio": round(ratio, 3), "keep_text": keep["content"],
                        "retire_text": drop["content"]})
    return out


def apply_merge(store, pair: dict, *, dry_run: bool = True) -> dict:
    keep_id, retire_id = int(pair["keep"]), int(pair["retire"])
    if dry_run:
        return {"ok": True, "dry_run": True, "keep": keep_id, "retire": retire_id}
    keep = store._one("SELECT content FROM facts WHERE fact_id = ?", (keep_id,))
    retire = store._one("SELECT content FROM facts WHERE fact_id = ?", (retire_id,))
    if keep is None or retire is None:
        return {"ok": False, "reason": "fact disappeared"}
    store.remove_fact(retire_id)
    with store._tx():
        store._write("INSERT INTO audit(action, fact_id, actor, reason, before, after) VALUES (?,?,?,?,?,?)",
                     ("merge", keep_id, "factkit", f"merged duplicate #{retire_id}",
                      retire["content"], keep["content"]))
    return {"ok": True, "keep": keep_id, "retire": retire_id}


# --------------------------------------------------------------- prefetch
def prefetch(store, message: str, *, limit: int = 5, min_trust: float = 0.3,
             min_score: float = 0.35) -> list[dict]:
    """Semantic recall for a user message, read from the live store.

    Uses the embeddings in ``fact_embeddings`` and the configured embedding
    endpoint. Returns [] when embeddings are unavailable, so the caller can
    fall back silently.
    """
    from .config import Config
    from .embed import from_config

    cfg = Config.load(str(getattr(store, "db_path", "") or "") or None)
    try:
        qvec = from_config(cfg).embed_one(message)
    except Exception:  # noqa: BLE001 - endpoint down, degrade to no recall
        return []
    if not qvec:
        return []

    import math
    import struct

    rows = store.read(
        "SELECT f.fact_id, f.content, f.trust_score, e.vec FROM fact_embeddings e "
        "JOIN facts f ON f.fact_id = e.fact_id WHERE e.model = ?", (cfg.embed_model,))
    out = []
    for row in rows:
        blob = row["vec"]
        n = len(blob) // 4
        vec = struct.unpack(f"<{n}f", blob)
        if len(vec) != len(qvec):
            continue
        dot = sum(a * b for a, b in zip(qvec, vec))
        na = math.sqrt(sum(a * a for a in qvec))
        nb = math.sqrt(sum(b * b for b in vec))
        if not na or not nb:
            continue
        score = dot / (na * nb)
        if score < min_score or float(row["trust_score"] or 0) < min_trust:
            continue
        out.append({"fact_id": int(row["fact_id"]), "content": row["content"],
                    "score": round(score, 4), "trust": round(float(row["trust_score"] or 0), 2)})
    out.sort(key=lambda r: -r["score"])
    return out[:limit]


def index_pending(store, *, batch: int = 32, force: bool = False) -> dict:
    """Embed facts that have no vector for the configured model."""
    from .config import Config
    from .embed import from_config
    from .vectors import pack

    cfg = Config.load(str(getattr(store, "db_path", "") or "") or None)
    client = from_config(cfg)
    model = cfg.embed_model
    have = {int(r["fact_id"]) for r in store.read(
        "SELECT fact_id FROM fact_embeddings WHERE model = ?", (model,))}
    rows = [dict(r) for r in store.read("SELECT fact_id, content FROM facts ORDER BY fact_id")]
    todo = [r for r in rows if force or int(r["fact_id"]) not in have]
    if not todo:
        return {"model": model, "indexed": 0, "pending": 0}

    done = 0
    try:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            vecs = client.embed([r["content"] for r in chunk])
            if not vecs:
                break
            conn = sqlite3.connect(store.db_path, timeout=10.0)
            try:
                for row, vec in zip(chunk, vecs):
                    conn.execute(
                        "INSERT OR REPLACE INTO fact_embeddings(fact_id, model, dims, vec) "
                        "VALUES (?,?,?,?)", (row["fact_id"], model, len(vec), pack(vec)))
                conn.commit()
            finally:
                conn.close()
            done += len(chunk)
    except Exception as exc:  # noqa: BLE001 - endpoint down: report, do not raise
        return {"model": model, "indexed": done, "pending": len(todo) - done, "error": str(exc)}
    return {"model": model, "indexed": done, "pending": len(todo) - done}
