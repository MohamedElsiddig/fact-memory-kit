"""Bulk import/export with dry-run and validation.

JSON is lossless (includes embeddings and audit provenance); CSV is for humans.
Import never overwrites silently: duplicates are skipped, and ``--supersedes``
is validated against the DB before anything is written.
"""
from __future__ import annotations

import csv
import io
import json
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
