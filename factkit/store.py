"""SQLite store: facts, per-model embeddings, entities, audit, proposals, metrics.

Design notes
------------
* Facts are the only source of truth; FTS5 (external content) gives BM25.
* Embeddings live in their own table keyed by ``(fact_id, model)`` so several
  models coexist and a swap never corrupts the corpus.
* Every mutation writes an append-only audit row (before/after JSON).
* Nothing is deleted by automation: TTL and conflicts only *propose*.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import vectors

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    source          TEXT DEFAULT 'manual',
    confidence      REAL DEFAULT 1.0,
    review_after    TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_embeddings (
    fact_id    INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    model      TEXT NOT NULL,
    dims       INTEGER NOT NULL,
    vec        BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (fact_id, model)
);
CREATE INDEX IF NOT EXISTS idx_emb_model ON fact_embeddings(model);

CREATE TABLE IF NOT EXISTS entities (
    entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL UNIQUE,
    kind      TEXT DEFAULT 'thing'
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias     TEXT PRIMARY KEY,
    entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, entity_id)
);

CREATE TABLE IF NOT EXISTS audit (
    audit_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    action     TEXT NOT NULL,
    fact_id    INTEGER,
    actor      TEXT DEFAULT 'cli',
    reason     TEXT DEFAULT '',
    before     TEXT,
    after      TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_fact ON audit(fact_id);

CREATE TABLE IF NOT EXISTS hermes_map (
    hermes_id  INTEGER PRIMARY KEY,
    fact_id    INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    synced_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conflict_proposals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    keep_id     INTEGER NOT NULL,
    retire_id   INTEGER NOT NULL,
    jaccard     REAL NOT NULL,
    slots       TEXT NOT NULL,
    delta       TEXT NOT NULL,
    status      TEXT DEFAULT 'pending',
    resolved_at TIMESTAMP,
    note        TEXT DEFAULT '',
    UNIQUE(keep_id, retire_id, status)
);

CREATE TABLE IF NOT EXISTS fact_proposals (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    content    TEXT NOT NULL,
    category   TEXT DEFAULT 'general',
    tags       TEXT DEFAULT '',
    source     TEXT DEFAULT 'bridge',
    score      REAL DEFAULT 0.0,
    reasons    TEXT DEFAULT '',
    status     TEXT DEFAULT 'pending',
    note       TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS ttl_flags (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id     INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    detail      TEXT NOT NULL,
    age_days    REAL NOT NULL,
    status      TEXT DEFAULT 'pending',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(fact_id, kind, status)
);

CREATE TABLE IF NOT EXISTS metrics (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    kind    TEXT NOT NULL,
    name    TEXT NOT NULL,
    value   REAL DEFAULT 0,
    meta    TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_metrics_kind ON metrics(kind, ts);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    content, tags,
    content='facts', content_rowid='fact_id',
    tokenize="unicode61 remove_diacritics 2"
);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts(rowid, content, tags) VALUES (new.fact_id, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, content, tags)
    VALUES('delete', old.fact_id, old.content, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF content, tags ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, content, tags)
    VALUES('delete', old.fact_id, old.content, old.tags);
  INSERT INTO facts_fts(rowid, content, tags) VALUES (new.fact_id, new.content, new.tags);
END;
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class Store:
    def __init__(self, path: str | Path, *, init: bool = True, readonly: bool = False):
        self.path = Path(path).expanduser()
        self.readonly = bool(readonly)
        if self.readonly:
            # Never create, never migrate, never write. SQLite enforces this at
            # the file level; PRAGMA query_only is a second belt for the same
            # connection.
            if not self.path.exists():
                raise FileNotFoundError(f"database not found: {self.path}")
            uri = f"file:{self.path}?mode=ro"
            self.db = sqlite3.connect(uri, uri=True, timeout=10.0)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA query_only=ON")
            self.db.execute("SELECT COUNT(*) FROM facts").fetchone()  # prove it reads
            return
        if init:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=10.0)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        if init:
            self.db.executescript(SCHEMA)
            self.db.commit()

    # ------------------------------------------------------------------ utils
    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _audit(self, action: str, fact_id: int | None, before, after,
               actor: str = "cli", reason: str = "") -> None:
        self.db.execute(
            "INSERT INTO audit(action, fact_id, actor, reason, before, after) "
            "VALUES (?,?,?,?,?,?)",
            (action, fact_id, actor, reason,
             _json(before) if before is not None else None,
             _json(after) if after is not None else None),
        )

    def metric(self, kind: str, name: str, value: float = 0.0, **meta) -> None:
        self.db.execute(
            "INSERT INTO metrics(kind, name, value, meta) VALUES (?,?,?,?)",
            (kind, name, float(value), _json(meta)),
        )

    # ------------------------------------------------------------------ facts
    def add_fact(self, content: str, category: str = "general", tags: str = "",
                 trust_score: float = 0.5, source: str = "manual",
                 confidence: float = 1.0, review_after: str | None = None,
                 actor: str = "cli", reason: str = "") -> dict:
        content = " ".join(content.split())
        if not content:
            raise ValueError("content is empty")
        cur = self.db.execute(
            "INSERT INTO facts(content, category, tags, trust_score, source, "
            "confidence, review_after) VALUES (?,?,?,?,?,?,?)",
            (content, category, tags, trust_score, source, confidence, review_after),
        )
        fact_id = int(cur.lastrowid or 0)
        self._audit("add", fact_id, None, {"content": content,
                                           "category": category, "tags": tags},
                    actor, reason)
        self.db.commit()
        return self.get(fact_id)  # type: ignore[return-value]

    def update_fact(self, fact_id: int, *, content: str | None = None,
                    category: str | None = None, tags: str | None = None,
                    trust_score: float | None = None,
                    review_after: str | None = None,
                    actor: str = "cli", reason: str = "") -> dict:
        before = self.get(fact_id)
        if before is None:
            raise KeyError(f"no fact #{fact_id}")
        fields, values = [], []
        for name, value in (("content", content), ("category", category),
                            ("tags", tags), ("trust_score", trust_score),
                            ("review_after", review_after)):
            if value is not None:
                fields.append(f"{name}=?")
                values.append(value)
        if not fields:
            return before
        fields.append("updated_at=?")
        values.append(utcnow())
        values.append(fact_id)
        self.db.execute(f"UPDATE facts SET {', '.join(fields)} WHERE fact_id=?", values)
        after = self.get(fact_id)
        self._audit("update", fact_id, before, after, actor, reason)
        self.db.commit()
        return after  # type: ignore[return-value]

    def remove_fact(self, fact_id: int, *, actor: str = "cli",
                    reason: str = "") -> bool:
        before = self.get(fact_id)
        if before is None:
            return False
        self.db.execute("DELETE FROM facts WHERE fact_id=?", (fact_id,))
        self._audit("remove", fact_id, before, None, actor, reason)
        self.db.commit()
        return True

    def get(self, fact_id: int) -> dict | None:
        row = self.db.execute("SELECT * FROM facts WHERE fact_id=?", (fact_id,)).fetchone()
        return dict(row) if row else None

    def list_facts(self, limit: int = 200, offset: int = 0,
                   category: str | None = None) -> list[dict]:
        sql = "SELECT * FROM facts"
        args: list[Any] = []
        if category:
            sql += " WHERE category=?"
            args.append(category)
        sql += " ORDER BY fact_id LIMIT ? OFFSET ?"
        args += [limit, offset]
        return [dict(r) for r in self.db.execute(sql, args)]

    def all_facts(self) -> list[dict]:
        rows = [dict(r) for r in self.db.execute("SELECT * FROM facts ORDER BY fact_id")]
        links: dict[int, set[int]] = {}
        for r in self.db.execute("SELECT fact_id, entity_id FROM fact_entities"):
            links.setdefault(r["fact_id"], set()).add(r["entity_id"])
        for r in rows:
            r["entities"] = sorted(links.get(r["fact_id"], set()))
            r["updated_ts"] = self._ts(r["updated_at"])
        return rows

    @staticmethod
    def _ts(value) -> float:
        if value is None:
            return 0.0
        text = str(value).replace("Z", "")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(text[:26], fmt).replace(
                    tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return 0.0

    def count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0])

    # ------------------------------------------------------------- embeddings
    def put_embedding(self, fact_id: int, model: str, vec: Sequence[float]) -> None:
        blob = vectors.pack(list(vec))
        self.db.execute(
            "INSERT INTO fact_embeddings(fact_id, model, dims, vec) VALUES (?,?,?,?) "
            "ON CONFLICT(fact_id, model) DO UPDATE SET "
            "dims=excluded.dims, vec=excluded.vec, created_at=CURRENT_TIMESTAMP",
            (fact_id, model, len(vec), blob),
        )

    def get_embedding(self, fact_id: int, model: str):
        row = self.db.execute(
            "SELECT vec FROM fact_embeddings WHERE fact_id=? AND model=?",
            (fact_id, model)).fetchone()
        return vectors.unpack(row["vec"]) if row else None

    def embeddings(self, model: str) -> list[tuple[int, Any]]:
        rows = self.db.execute(
            "SELECT fact_id, vec FROM fact_embeddings WHERE model=?", (model,))
        return [(r["fact_id"], vectors.unpack(r["vec"])) for r in rows]

    def facts_missing_embedding(self, model: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT f.* FROM facts f LEFT JOIN fact_embeddings e "
            "ON e.fact_id=f.fact_id AND e.model=? WHERE e.fact_id IS NULL "
            "ORDER BY f.fact_id", (model,))
        return [dict(r) for r in rows]

    def embedding_models(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT model, COUNT(*) n, MAX(dims) dims, MAX(created_at) last "
            "FROM fact_embeddings GROUP BY model ORDER BY model")
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------- FTS
    def fts_search(self, query: str, limit: int = 50) -> list[tuple[int, float]]:
        match = fts_query(query)
        if not match:
            return []
        try:
            rows = self.db.execute(
                "SELECT f.fact_id, bm25(facts_fts) AS rank FROM facts_fts "
                "JOIN facts f ON f.fact_id = facts_fts.rowid "
                "WHERE facts_fts MATCH ? ORDER BY rank LIMIT ?",
                (match, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(int(r["fact_id"]), float(r["rank"])) for r in rows]

    # -------------------------------------------------------------- entities
    def ensure_entity(self, name: str, kind: str = "thing") -> int:
        name = name.strip()
        row = self.db.execute(
            "SELECT entity_id FROM entities WHERE name=? COLLATE NOCASE", (name,)).fetchone()
        if row:
            return int(row["entity_id"])
        alias = self.db.execute(
            "SELECT entity_id FROM entity_aliases WHERE alias=? COLLATE NOCASE",
            (name,)).fetchone()
        if alias:
            return int(alias["entity_id"])
        cur = self.db.execute("INSERT INTO entities(name, kind) VALUES (?,?)", (name, kind))
        return int(cur.lastrowid or 0)

    def add_alias(self, alias: str, entity_id: int) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO entity_aliases(alias, entity_id) VALUES (?,?)",
            (alias.strip(), entity_id))

    def entities(self) -> list[dict]:
        # factkit's schema calls it ``kind``; the live Hermes store calls it
        # ``entity_type``. Read either, so the dashboard can serve both.
        columns = {row["name"] for row in self.db.execute("PRAGMA table_info(entities)")}
        kind = ("e.entity_type" if "entity_type" in columns
                else "e.kind" if "kind" in columns else "'thing'")
        rows = self.db.execute(
            f"SELECT e.entity_id, e.name, {kind} AS kind, COUNT(fe.fact_id) n "
            "FROM entities e LEFT JOIN fact_entities fe ON fe.entity_id=e.entity_id "
            "GROUP BY e.entity_id ORDER BY n DESC, e.name")
        return [dict(r) for r in rows]

    def link_entity(self, fact_id: int, entity_id: int) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO fact_entities(fact_id, entity_id) VALUES (?,?)",
            (fact_id, entity_id))

    def unlink_entity(self, fact_id: int, entity_id: int) -> None:
        self.db.execute(
            "DELETE FROM fact_entities WHERE fact_id=? AND entity_id=?",
            (fact_id, entity_id))

    def fact_entities(self, fact_id: int) -> list[dict]:
        rows = self.db.execute(
            "SELECT e.entity_id, e.name FROM fact_entities fe "
            "JOIN entities e ON e.entity_id=fe.entity_id WHERE fe.fact_id=?",
            (fact_id,))
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ audit
    def audit_log(self, limit: int = 100, fact_id: int | None = None) -> list[dict]:
        if fact_id is None:
            rows = self.db.execute(
                "SELECT * FROM audit ORDER BY audit_id DESC LIMIT ?", (limit,))
        else:
            rows = self.db.execute(
                "SELECT * FROM audit WHERE fact_id=? ORDER BY audit_id DESC LIMIT ?",
                (fact_id, limit))
        return [dict(r) for r in rows]

    # -------------------------------------------------------------- proposals
    def add_conflict(self, keep_id: int, retire_id: int, jaccard: float,
                     slots: Iterable[str], delta: Iterable[str]) -> int:
        cur = self.db.execute(
            "INSERT OR IGNORE INTO conflict_proposals"
            "(keep_id, retire_id, jaccard, slots, delta) VALUES (?,?,?,?,?)",
            (keep_id, retire_id, jaccard, ",".join(slots), " | ".join(delta)))
        return int(cur.lastrowid or 0)

    def conflicts(self, status: str = "pending") -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM conflict_proposals WHERE status=? ORDER BY jaccard DESC, id",
            (status,))
        return [dict(r) for r in rows]

    def resolve_conflict(self, proposal_id: int, status: str, note: str = "") -> None:
        self.db.execute(
            "UPDATE conflict_proposals SET status=?, resolved_at=?, note=? WHERE id=?",
            (status, utcnow(), note, proposal_id))

    def add_proposal(self, content: str, category: str = "general", tags: str = "",
                     source: str = "bridge", score: float = 0.0,
                     reasons: str = "") -> int:
        cur = self.db.execute(
            "INSERT INTO fact_proposals(content, category, tags, source, score, reasons) "
            "VALUES (?,?,?,?,?,?)",
            (content, category, tags, source, score, reasons))
        return int(cur.lastrowid or 0)

    def proposals(self, status: str = "pending") -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM fact_proposals WHERE status=? ORDER BY score DESC, id",
            (status,))
        return [dict(r) for r in rows]

    def resolve_proposal(self, proposal_id: int, status: str, note: str = "") -> None:
        self.db.execute(
            "UPDATE fact_proposals SET status=?, note=? WHERE id=?",
            (status, note, proposal_id))

    # -------------------------------------------------------------------- ttl
    def flag_ttl(self, fact_id: int, kind: str, detail: str, age_days: float) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO ttl_flags(fact_id, kind, detail, age_days) "
            "VALUES (?,?,?,?)", (fact_id, kind, detail, age_days))

    def ttl_flags(self, status: str = "pending") -> list[dict]:
        rows = self.db.execute(
            "SELECT t.*, f.content, f.category FROM ttl_flags t "
            "JOIN facts f ON f.fact_id=t.fact_id WHERE t.status=? "
            "ORDER BY t.age_days DESC", (status,))
        return [dict(r) for r in rows]

    def resolve_ttl(self, flag_id: int, status: str) -> None:
        self.db.execute("UPDATE ttl_flags SET status=? WHERE id=?", (status, flag_id))

    # ---------------------------------------------------------------- metrics
    def metrics_summary(self, days: int = 30) -> dict:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        rows = self.db.execute(
            "SELECT kind, name, COUNT(*) n, AVG(value) avg, MAX(value) max "
            "FROM metrics WHERE ts >= ? GROUP BY kind, name ORDER BY kind, name",
            (since,)).fetchall()
        out: dict[str, list] = {}
        for r in rows:
            out.setdefault(r["kind"], []).append({
                "name": r["name"], "count": r["n"],
                "avg": round(r["avg"] or 0, 3), "max": round(r["max"] or 0, 3)})
        return out

    def latency_percentiles(self, days: int = 30) -> dict:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        vals = [float(r[0]) for r in self.db.execute(
            "SELECT value FROM metrics WHERE kind='search' AND name='latency_ms' "
            "AND ts >= ? ORDER BY value", (since,))]
        if not vals:
            return {"n": 0}
        def pct(p: float) -> float:
            idx = min(len(vals) - 1, max(0, int(round(p / 100 * (len(vals) - 1)))))
            return round(vals[idx], 1)
        hits = self.db.execute(
            "SELECT SUM(value), COUNT(*) FROM metrics WHERE kind='search' "
            "AND name='hit' AND ts >= ?", (since,)).fetchone()
        return {"n": len(vals), "p50_ms": pct(50), "p95_ms": pct(95),
                "max_ms": round(vals[-1], 1),
                "hit_rate": round((hits[0] or 0) / hits[1], 3) if hits[1] else 0.0}


FTS_STOP = {
    "the", "a", "an", "of", "for", "to", "in", "on", "at", "is", "are", "was",
    "were", "be", "and", "or", "not", "with", "from", "by", "as", "that", "this",
    "it", "its", "my", "me", "i", "you", "your", "we", "our", "do", "does", "did",
    "how", "what", "when", "where", "which", "who", "why", "can", "could",
}


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 OR-query (ranking sorts the noise out)."""
    import re
    tokens = [t for t in re.findall(r"[A-Za-z0-9_][A-Za-z0-9_.\-/]*", text or "")]
    terms = []
    for tok in tokens:
        low = tok.lower()
        if len(low) < 2 or low in FTS_STOP:
            continue
        terms.append('"' + tok.replace('"', "") + '"')
    # de-duplicate, keep order
    seen, uniq = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return " OR ".join(uniq[:24])
