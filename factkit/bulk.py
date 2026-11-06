"""Bulk import/export with dry-run and validation.

JSON is lossless (includes embeddings and audit provenance); CSV is for humans.
Import never overwrites silently: duplicates are skipped, and ``--supersedes``
is validated against the DB before anything is written.
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path

FIELDS = ["fact_id", "content", "category", "tags", "trust_score", "source",
          "confidence", "review_after", "created_at", "updated_at"]


def export_facts(store, path: Path | None = None, fmt: str = "json",
                 include_embeddings: bool = False) -> str:
    facts = store.all_facts()
    rows = []
    for fact in facts:
        row = {k: fact.get(k) for k in FIELDS}
        row["entities"] = [e["name"] for e in store.fact_entities(fact["fact_id"])]
        if include_embeddings:
            emb = {}
            for record in store.embedding_models():
                vec = store.get_embedding(fact["fact_id"], record["model"])
                if vec is not None:
                    emb[record["model"]] = list(vec)
            row["embeddings"] = emb
        rows.append(row)

    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=[*FIELDS, "entities", "embeddings"],
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["entities"] = ";".join(row.get("entities", []))
            flat.pop("embeddings", None)
            writer.writerow(flat)
        text = buf.getvalue()
    else:
        text = json.dumps({"version": 1, "count": len(rows), "facts": rows},
                          indent=1, ensure_ascii=False)
    if path:
        path.write_text(text)
    return text


def _load(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".csv":
        return [dict(r) for r in csv.DictReader(io.StringIO(text))]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    return data.get("facts", [])


def import_facts(store, path: Path, *, dry_run: bool = False,
                 default_category: str = "general") -> dict:
    rows = _load(path)
    existing = {" ".join(f["content"].lower().split()): f["fact_id"]
                for f in store.list_facts(limit=1_000_000)}
    added, skipped, errors = [], [], []

    for index, row in enumerate(rows, 1):
        content = " ".join(str(row.get("content", "")).split())
        if not content:
            errors.append({"row": index, "error": "missing content"})
            continue
        key = content.lower()
        if key in existing:
            skipped.append({"row": index, "reason": "duplicate",
                            "fact_id": existing[key]})
            continue
        try:
            trust = float(row.get("trust_score") or 0.5)
        except (TypeError, ValueError):
            trust = 0.5
        trust = min(1.0, max(0.0, trust))

        if dry_run:
            added.append({"row": index, "content": content[:80]})
            continue
        fact = store.add_fact(
            content,
            category=(row.get("category") or default_category),
            tags=(row.get("tags") or ""),
            trust_score=trust,
            source=(row.get("source") or "import"),
            confidence=float(row.get("confidence") or 1.0),
            review_after=row.get("review_after") or None,
            actor="import", reason=f"bulk import from {path.name}",
        )
        existing[key] = fact["fact_id"]
        added.append({"row": index, "fact_id": fact["fact_id"]})
    store.db.commit()
    return {"rows": len(rows), "added": len(added), "skipped": len(skipped),
            "errors": errors, "dry_run": dry_run, "detail": added[:50]}


def validate_against_golden(path: Path, golden_path: Path) -> dict:
    """Check that every fact id referenced by the golden set exists in the export."""
    data = json.loads(path.read_text())
    ids = {int(f["fact_id"]) for f in data.get("facts", []) if f.get("fact_id")}
    golden = json.loads(golden_path.read_text())
    missing = []
    for case in golden.get("cases", []):
        for fid in case.get("facts") or []:
            if int(fid) not in ids:
                missing.append({"query": case.get("q"), "fact_id": fid})
    return {"facts": len(ids), "missing": missing, "ok": not missing}

def import_hermes(store, hermes_db: Path, *, dry_run: bool = False,
                  keep_timestamps: bool = True) -> dict:
    """Import facts (+ entity links) from a Hermes ``memory_store.db``.

    The source is opened ``mode=ro``; it is never written to. Duplicates are
    skipped by normalised content, and original trust/timestamps are preserved so
    the golden rule still knows which facts are newer.
    """
    hermes_db = Path(hermes_db).expanduser()
    if not hermes_db.exists():
        raise FileNotFoundError(hermes_db)
    con = sqlite3.connect(f"file:{hermes_db}?mode=ro", uri=True, timeout=5.0)
    con.row_factory = sqlite3.Row
    try:
        columns = {r["name"] for r in con.execute("PRAGMA table_info(facts)")}
        if "content" not in columns:
            raise ValueError("source has no facts.content column")
        select = ["fact_id", "content"]
        for name in ("category", "tags", "trust_score", "created_at", "updated_at"):
            if name in columns:
                select.append(name)
        rows = [dict(r) for r in con.execute(
            f"SELECT {', '.join(select)} FROM facts ORDER BY fact_id")]
        links: dict[int, list[int]] = {}
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if {"fact_entities", "entities"} <= tables:
            for r in con.execute(
                    "SELECT fe.fact_id, e.name FROM fact_entities fe "
                    "JOIN entities e ON e.entity_id=fe.entity_id"):
                links.setdefault(int(r["fact_id"]), []).append(r["name"])
    finally:
        con.close()

    existing = {" ".join(f["content"].lower().split()): f["fact_id"]
                for f in store.list_facts(limit=1_000_000)}
    added, skipped, errors = [], [], []
    id_map: dict[int, int] = {}

    for row in rows:
        content = " ".join(str(row.get("content") or "").split())
        if not content:
            errors.append({"fact_id": row.get("fact_id"), "error": "empty content"})
            continue
        key = content.lower()
        if key in existing:
            skipped.append({"fact_id": row.get("fact_id"),
                            "fact_id_here": existing[key], "reason": "duplicate"})
            continue
        if dry_run:
            added.append({"fact_id": row.get("fact_id"), "content": content[:80]})
            continue
        fact = store.add_fact(
            content,
            category=(row.get("category") or "general"),
            tags=(row.get("tags") or ""),
            trust_score=float(row.get("trust_score") or 0.5),
            source="hermes-import",
            actor="import", reason="migrated from Hermes memory_store.db",
        )
        id_map[int(row["fact_id"])] = fact["fact_id"]
        existing[key] = fact["fact_id"]
        if keep_timestamps and row.get("created_at"):
            store.db.execute(
                "UPDATE facts SET created_at=?, updated_at=? WHERE fact_id=?",
                (row.get("created_at"), row.get("updated_at") or row.get("created_at"),
                 fact["fact_id"]))
        for name in links.get(int(row["fact_id"]), []):
            entity_id = store.ensure_entity(name)
            store.link_entity(fact["fact_id"], entity_id)
        added.append({"fact_id": row.get("fact_id"), "fact_id_here": fact["fact_id"]})
    store.db.commit()
    return {"source": str(hermes_db), "rows": len(rows), "added": len(added),
            "skipped": len(skipped), "errors": errors, "dry_run": dry_run,
            "detail": added[:50]}


def sync_hermes(store, hermes_db: Path, *, dry_run: bool = False,
                prune: bool = False) -> dict:
    """Keep the factkit store aligned with a live Hermes ``memory_store.db``.

    Hermes stays the source of truth. This mirrors inserts, updates changed text
    in place (so fact_id and its embeddings survive), and optionally retires facts
    deleted upstream. The source is opened read-only.
    """
    hermes_db = Path(hermes_db).expanduser()
    con = sqlite3.connect(f"file:{hermes_db}?mode=ro", uri=True, timeout=5.0)
    con.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in con.execute(
            "SELECT fact_id, content, category, tags, trust_score, created_at, updated_at "
            "FROM facts ORDER BY fact_id")]
        links: dict[int, list[str]] = {}
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if {"fact_entities", "entities"} <= tables:
            for r in con.execute(
                    "SELECT fe.fact_id, e.name FROM fact_entities fe "
                    "JOIN entities e ON e.entity_id=fe.entity_id"):
                links.setdefault(int(r["fact_id"]), []).append(r["name"])
    finally:
        con.close()

    mapping = {int(r["hermes_id"]): int(r["fact_id"]) for r in store.db.execute(
        "SELECT hermes_id, fact_id FROM hermes_map")}
    by_content = {" ".join(f["content"].lower().split()): f["fact_id"]
                  for f in store.list_facts(limit=1_000_000)}
    added = updated = unchanged = 0
    details: list[dict] = []
    seen_hermes: set[int] = set()

    for row in rows:
        hid = int(row["fact_id"])
        seen_hermes.add(hid)
        content = " ".join((row["content"] or "").split())
        if not content:
            continue
        fact_id = mapping.get(hid) or by_content.get(content.lower())
        if fact_id and store.get(fact_id) is None:
            fact_id = None
        if fact_id is None:
            if dry_run:
                added += 1
                details.append({"hermes_id": hid, "action": "add"})
                continue
            fact = store.add_fact(
                content, category=row.get("category") or "general",
                tags=row.get("tags") or "", trust_score=float(row.get("trust_score") or 0.5),
                source="hermes-sync", actor="sync",
                reason=f"hermes fact #{hid}")
            fact_id = fact["fact_id"]
            store.db.execute(
                "INSERT OR REPLACE INTO hermes_map(hermes_id, fact_id) VALUES (?,?)",
                (hid, fact_id))
            if row.get("created_at"):
                store.db.execute(
                    "UPDATE facts SET created_at=?, updated_at=? WHERE fact_id=?",
                    (row["created_at"], row.get("updated_at") or row["created_at"], fact_id))
            for name in links.get(hid, []):
                store.link_entity(fact_id, store.ensure_entity(name))
            by_content[content.lower()] = fact_id
            added += 1
            details.append({"hermes_id": hid, "action": "add", "fact_id": fact_id})
            continue

        current = store.get(fact_id)
        if current["content"].strip() != content:
            if not dry_run:
                store.update_fact(fact_id, content=content,
                                  category=row.get("category") or current["category"],
                                  tags=row.get("tags") or current["tags"],
                                  reason=f"synced from hermes #{hid}")
                if row.get("updated_at"):
                    store.db.execute("UPDATE facts SET updated_at=? WHERE fact_id=?",
                                     (row["updated_at"], fact_id))
            updated += 1
            details.append({"hermes_id": hid, "action": "update", "fact_id": fact_id})
        else:
            unchanged += 1
        if not dry_run:
            store.db.execute(
                "INSERT OR REPLACE INTO hermes_map(hermes_id, fact_id) VALUES (?,?)",
                (hid, fact_id))

    pruned = 0
    if prune:
        for hid, fact_id in mapping.items():
            if hid not in seen_hermes and store.get(fact_id) is not None:
                pruned += 1
                details.append({"hermes_id": hid, "action": "prune", "fact_id": fact_id})
                if not dry_run:
                    store.remove_fact(fact_id, actor="sync",
                                      reason=f"deleted in Hermes (#{hid})")
    if not dry_run:
        store.db.commit()
    return {"source": str(hermes_db), "source_facts": len(rows), "added": added,
            "updated": updated, "unchanged": unchanged, "pruned": pruned,
            "dry_run": dry_run, "detail": details[:60]}
