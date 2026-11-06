"""Prose contradiction detection: candidate pairs, model verdicts, proposals.

Numeric attribute disagreement is handled by :func:`factkit.holo.find_conflicts`
(16 GB vs 32 GB, port 7000 vs 7001). This module covers the other half: two
facts about the same subject whose *wording* disagrees, such as "the proxy runs
on node A" versus "the proxy runs on node B". The two facts share the subject
tokens; only the model can tell that the remainder contradicts.

Pipeline::

    candidate_pairs(store)          # read-only, deterministic, capped
    adjudicate_pairs(pairs, client) # JSON verdict per pair, partial on failure
    record(store, results, batch_id=...)  # conflict_proposals, status pending

Rules:
  * proposals only - nothing here writes to ``facts``;
  * pairs that ``find_conflicts`` already flagged are left to that path;
  * an unparseable answer is skipped, never guessed at;
  * a pair already in ``conflict_proposals`` is not inserted twice.
"""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .adjudicate import parse_verdict
from .holo import _tokens, _value_conflict, find_conflicts

MAX_PAIRS = 50
MIN_OVERLAP = 2

# Below this the model is guessing; the pair is reported but not proposed.
MIN_CONFIDENCE = 0.6

VERDICTS = ("contradiction", "compatible", "unknown")

PROMPT = (
    "You compare two stored facts about the same subject. Decide whether they "
    "contradict each other: they cannot both be true at the same time.\n"
    "Ignore differences in wording, detail level or age. A newer value that "
    "replaces an older one IS a contradiction. If the facts describe different "
    "subjects, or one merely adds detail, they are compatible.\n"
    "Answer with JSON only: "
    '{"contradiction": true|false, "confidence": 0.0-1.0, '
    '"attribute": "<short name>", '
    '"evidence": ["<span from fact 1>", "<span from fact 2>"]}\n\n'
    "FACT 1: {a}\nFACT 2: {b}\n"
)


# ------------------------------------------------------------------- pairs
def _fact_rows(store) -> list[dict]:
    """Facts as plain dicts, from a plugin store or a database path."""
    if hasattr(store, "read"):
        rows = store.read("SELECT fact_id, content FROM facts ORDER BY fact_id")
        return [{"fact_id": int(r["fact_id"]), "content": str(r["content"] or "")}
                for r in rows]

    path = f"file:{Path(store).expanduser()}?mode=ro"
    conn = sqlite3.connect(path, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT fact_id, content FROM facts ORDER BY fact_id")
        return [{"fact_id": int(r["fact_id"]), "content": str(r["content"] or "")}
                for r in rows]
    finally:
        conn.close()


def _already_flagged(store) -> set[frozenset]:
    """fact-id pairs that numeric conflict detection already owns."""
    db_path = getattr(store, "db_path", None)
    if db_path is None and not hasattr(store, "read"):
        db_path = store
    if db_path is None:
        return set()
    try:
        found = find_conflicts(db_path)
    except Exception:  # noqa: BLE001 - older/foreign store: fall back below
        return set()
    return {frozenset((int(p["keep"]), int(p["retire"]))) for p in found}


def candidate_pairs(store, *, max_pairs: int = MAX_PAIRS,
                    min_overlap: int = MIN_OVERLAP) -> list[dict]:
    """Pairs sharing ``min_overlap`` content tokens, minus numeric conflicts.

    Tokens come from :func:`factkit.holo._tokens`, so stopwords, short words and
    pure numbers do not count. A pair is skipped when ``find_conflicts`` already
    flagged it or when it disagrees on a named numeric attribute (that is the
    numeric path's job). Order is deterministic: most shared tokens first, then
    fact ids. Returns at most ``max_pairs`` entries shaped::

        {"a": {"fact_id": int, "content": str},
         "b": {"fact_id": int, "content": str},
         "shared": ["proxy", "runs", ...]}
    """
    need = max(1, int(min_overlap))
    facts = _fact_rows(store)
    if len(facts) < 2 or max_pairs <= 0:
        return []

    flagged = _already_flagged(store)
    tokens = {f["fact_id"]: _tokens(f["content"]) for f in facts}
    out: list[dict] = []
    for i, a in enumerate(facts):
        for b in facts[i + 1:]:
            key = frozenset((a["fact_id"], b["fact_id"]))
            if key in flagged:
                continue
            if a["content"].strip().lower() == b["content"].strip().lower():
                continue
            shared = sorted(tokens[a["fact_id"]] & tokens[b["fact_id"]])
            if len(shared) < need:
                continue
            # Same attribute, different value: leave it to find_conflicts.
            if _value_conflict(a, b, shared):
                continue
            out.append({"a": {"fact_id": a["fact_id"], "content": a["content"]},
                        "b": {"fact_id": b["fact_id"], "content": b["content"]},
                        "shared": shared})

    out.sort(key=lambda p: (-len(p["shared"]), p["a"]["fact_id"], p["b"]["fact_id"]))
    return out[:int(max_pairs)]


# -------------------------------------------------------------- adjudicate
class Verdicts(list):
    """List of judged pairs that also carries the endpoint error, if any."""

    error: str | None = None

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key == "error":
                return self.error
            raise KeyError(key)
        return super().__getitem__(key)

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, str) and item == "error":
            return self.error is not None
        return super().__contains__(item)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style access for the one non-list key: ``error``."""
        if key == "error":
            return self.error if self.error is not None else default
        return default


def _ask(client, prompt: str, model: str | None) -> str:
    """One completion: OpenAI-compatible client, or a plain prompt callable."""
    if callable(client) and not hasattr(client, "chat"):
        return client(prompt)

    create = getattr(client, "chat").completions.create
    kwargs: dict = {"messages": [{"role": "user", "content": prompt}],
                    "temperature": 0}
    resolved = (model or getattr(client, "model", None)
                or getattr(client, "default_model", None))
    if resolved:
        kwargs["model"] = resolved
    response = create(**kwargs)
    if isinstance(response, str):
        return response
    return str(response.choices[0].message.content or "")


def _confidence(data: dict) -> float:
    """Clamp the model's confidence to 0..1; missing or junk means 0.0."""
    raw = data.get("confidence")
    if isinstance(raw, bool):
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, value))


def _texts(pair: dict) -> tuple[str, str]:
    def side(name: str, legacy: str) -> str:
        value = pair.get(name)
        if isinstance(value, dict):
            value = value.get("content")
        if not value:
            value = pair.get(legacy)
        return str(value or "")

    return side("a", "keep_content"), side("b", "retire_content")


def adjudicate_pairs(pairs: list[dict], client, *, model: str | None = None,
                     max_pairs: int = MAX_PAIRS) -> Verdicts:
    """Ask ``client`` to judge each pair; return parsed verdicts.

    Result entries keep the original pair and add ``verdict`` (one of
    ``contradiction`` / ``compatible`` / ``unknown``), ``confidence``,
    ``evidence`` and ``attribute``. Confidence below :data:`MIN_CONFIDENCE`
    downgrades any answer to ``unknown``. An unparseable answer is skipped.

    An endpoint failure ends the run, keeping what was judged so far; the
    returned list carries the message on ``results.error`` (and
    ``results["error"]``).
    """
    results = Verdicts()
    results.error = None
    if client is None:
        results.error = "no client"
        return results

    for pair in pairs[:max(0, int(max_pairs))]:
        a, b = _texts(pair)
        if not a or not b:
            continue
        prompt = PROMPT.replace("{a}", a).replace("{b}", b)
        try:
            answer = _ask(client, prompt, model)
        except Exception as exc:  # noqa: BLE001 - endpoint down: keep partial results
            results.error = f"{type(exc).__name__}: {exc}"
            break
        verdict = parse_verdict(answer)
        if verdict is None:
            continue
        confidence = _confidence(verdict)
        if confidence < MIN_CONFIDENCE:
            label = "unknown"
        else:
            label = "contradiction" if verdict["contradiction"] else "compatible"
        results.append({**pair, "verdict": label, "confidence": confidence,
                        "contradiction": bool(verdict["contradiction"]),
                        "evidence": list(verdict.get("evidence") or []),
                        "attribute": verdict.get("attribute") or ""})
    return results


# --------------------------------------------------------------- proposals
def _keep_retire(store, a_id: int, b_id: int) -> tuple[int, int] | None:
    """Newer fact is ``keep`` (the repo's convention); None if one vanished."""
    rows = store.read("SELECT fact_id, updated_at FROM facts WHERE fact_id IN (?, ?)",
                      (a_id, b_id))
    stamps = {int(r["fact_id"]): str(r["updated_at"] or "") for r in rows}
    if len(stamps) != 2:
        return None
    keep, retire = sorted((a_id, b_id), key=lambda fid: (stamps[fid], fid), reverse=True)
    return keep, retire


def _exists(store, keep_id: int, retire_id: int) -> bool:
    """A proposal for this pair in either orientation already exists."""
    row = store.read(
        "SELECT 1 FROM conflict_proposals WHERE "
        "(keep_id = ? AND retire_id = ?) OR (keep_id = ? AND retire_id = ?) LIMIT 1",
        (keep_id, retire_id, retire_id, keep_id))
    return bool(row)


def _jaccard(a_text: str, b_text: str) -> float:
    ta, tb = _tokens(a_text), _tokens(b_text)
    return round(len(ta & tb) / len(ta | tb), 4) if (ta | tb) else 0.0


def record(store, results, *, batch_id: str) -> int:
    """Store adjudicated pairs as pending proposals; return the inserted count.

    Uses the existing ``conflict_proposals`` table, adding its ``batch_id`` /
    ``evidence`` columns additively on older databases. Idempotent: a pair that
    is already proposed - in either keep/retire orientation - is skipped, and
    the UNIQUE constraint is the final guard. ``facts`` is never touched.
    """
    results = list(results or [])
    if not results:
        return 0

    from . import holo, maintenance

    maintenance.ensure_tombstones(store)   # owns the proposals DDL
    holo._ensure_proposal_columns(store)   # additive on older databases

    inserted = 0
    for result in results:
        a, b = result.get("a") or {}, result.get("b") or {}
        try:
            a_id, b_id = int(a["fact_id"]), int(b["fact_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if a_id == b_id:
            continue
        ordered = _keep_retire(store, a_id, b_id)
        if ordered is None or _exists(store, *ordered):
            continue
        keep_id, retire_id = ordered

        a_text, b_text = str(a.get("content") or ""), str(b.get("content") or "")
        keep_text, retire_text = ((a_text, b_text) if keep_id == a_id
                                  else (b_text, a_text))
        verdict = str(result.get("verdict") or "unknown")
        try:
            confidence = max(0.0, min(1.0, float(result.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        shared = sorted({str(t) for t in (result.get("shared") or [])})
        jaccard = _jaccard(a_text, b_text)
        evidence = json.dumps(
            {"kind": "prose", "shared": shared, "verdict": verdict,
             "confidence": confidence, "keep_id": keep_id, "retire_id": retire_id,
             "keep_text": keep_text, "retire_text": retire_text,
             "attribute": str(result.get("attribute") or ""),
             "spans": list(result.get("evidence") or [])},
            sort_keys=True, ensure_ascii=False)
        delta = f"prose {verdict} ({confidence:.2f})"
        if shared:
            delta += f": {' '.join(shared)}"

        with store._tx():
            cur = store._write(
                "INSERT OR IGNORE INTO conflict_proposals"
                "(keep_id, retire_id, jaccard, slots, delta, status, note, batch_id, evidence) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (keep_id, retire_id, jaccard, "prose", delta, "pending",
                 f"batch={batch_id}" if batch_id else "", batch_id, evidence))
        if getattr(cur, "rowcount", 1):
            inserted += 1
    return inserted


# -------------------------------------------------------------------- scan
def prose_scan(store, client=None, *, max_pairs: int = MAX_PAIRS,
               apply_proposals: bool = True) -> dict:
    """Candidate pairs -> model verdicts -> pending proposals.

    Returns ``{"pairs", "judged", "proposed", "skipped"}`` plus ``"error"`` when
    the endpoint failed part-way. ``pairs`` counts candidates considered,
    ``judged`` parsed verdicts, ``skipped`` candidates that never got a verdict
    and ``proposed`` rows actually inserted (contradictions only). Nothing under
    ``facts`` is modified.
    """
    pairs = candidate_pairs(store, max_pairs=max_pairs)
    summary: dict = {"pairs": len(pairs), "judged": 0, "proposed": 0,
                     "skipped": len(pairs)}
    if not pairs or client is None:
        return summary

    results = adjudicate_pairs(pairs, client, max_pairs=max_pairs)
    summary["judged"] = len(results)
    summary["skipped"] = len(pairs) - len(results)
    if results.error:
        summary["error"] = results.error

    hits = [r for r in results if r.get("verdict") == "contradiction"]
    if apply_proposals and hits:
        batch_id = ("prose-" + _dt.datetime.now().strftime("%Y%m%d%H%M%S")
                    + "-" + uuid.uuid4().hex[:6])
        summary["proposed"] = record(store, hits, batch_id=batch_id)
    return summary
