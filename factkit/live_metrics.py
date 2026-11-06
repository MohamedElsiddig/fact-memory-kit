"""Retrieval metrics over a *live* store (the Hermes holographic memory).

The golden gate in :mod:`factkit.golden` only scores the standalone corpus. This
module replays hand-labelled ``{"query": str, "expect": [fact_id, ...]}`` cases
against whatever store the caller passes -- the plugin's ``MemoryStore`` (via
``store.read``), a :class:`factkit.store.Store`, or any object with a read-only
SQLite connection -- and reports hit rate, MRR and per-case ranks.

Everything here is **read-only**: the only SQL issued is ``SELECT`` / ``PRAGMA``
and nothing is written to ``facts``, ``fact_embeddings``, ``metrics`` or
``audit``. Scoring never touches the network either: pass an ``embedder`` to add
the vector channel, otherwise retrieval is lexical (BM25) only.

Ranking mirrors :class:`factkit.hybrid.HybridSearch`: BM25 over the fact text and
cosine over the stored embeddings, fused with reciprocal-rank fusion and a
deterministic ``fact_id`` tie-break. No trust/score threshold is applied, so a
miss is a pure ranking miss, not a filtered-out fact.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from . import vectors
from .store import FTS_STOP

RRF_K = 60
DEFAULT_K = 5

_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-/]*")


# --------------------------------------------------------------------- cases
def _expect_ids(value: Any) -> tuple[int, ...] | None:
    """Validated expected ids, or None when the case is junk.

    Same rules as :func:`factkit.smoke._expect_ids`: a non-empty list/tuple/set
    of real ints (``bool`` is rejected even though it subclasses ``int``).
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set)):
        return None
    if isinstance(value, set):
        value = sorted(value)          # sets iterate in hash order, not input order
    ids: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        ids.append(int(item))
    return tuple(ids) or None


def _query_of(case: dict) -> str | None:
    query = case.get("query", case.get("q"))
    if not isinstance(query, str) or not query.strip():
        return None
    return query


def _expect_of(case: dict) -> tuple[int, ...] | None:
    return _expect_ids(case.get("expect", case.get("facts")))


def _items(cases: Any) -> list:
    """Normalise the many accepted containers into a list of case objects."""
    if cases is None:
        return []
    if isinstance(cases, dict):
        if "query" in cases or "q" in cases:      # a single case, not a report
            return [cases]
        cases = cases.get("cases") or []
    try:
        return list(cases)
    except TypeError:
        return []


def _case_dict(item) -> dict:
    """Map a dict or a golden-style ``Case``/row object onto query/expect keys."""
    if isinstance(item, dict):
        return item
    return {"query": getattr(item, "q", None),
            "expect": getattr(item, "facts", None)}


def load_cases(path, store=None) -> list[dict]:
    """Read ``{"query": str, "expect": [int, ...]}`` cases from ``path``.

    The input format is the one :func:`factkit.smoke.load_cases` accepts: either
    a bare JSON list or a dict with a ``"cases"`` list (so a ``golden.json``
    wrapper works too). ``q``/``facts`` are accepted as aliases for
    ``query``/``expect``. A missing file yields ``[]``; malformed JSON raises
    ``ValueError`` (a broken baseline should be loud, not silently empty).

    When ``store`` is given, every expected id is checked against the live
    store. Unknown ids are attached to their case as ``"unknown": [id, ...]``
    so a stale expectation is visible instead of being scored as a miss. Use
    :func:`validate_cases` if you want the report on its own.
    """
    p = Path(path).expanduser()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {p}: {exc}") from exc
    if isinstance(data, dict):
        data = data.get("cases") or []
    if not isinstance(data, list):
        return []

    out: list[dict] = []
    known = _known_ids(store) if store is not None else set()
    for item in data:
        if not isinstance(item, dict):
            continue
        query, expect = _query_of(item), _expect_of(item)
        if query is None or expect is None:
            continue
        case = {"query": query, "expect": list(expect)}
        if store is not None:
            missing = [fid for fid in expect if fid not in known]
            if missing:
                case["unknown"] = missing
        out.append(case)
    return out


def validate_cases(store, cases) -> dict:
    """Split cases into scorable ones and cases with unknown expected ids.

    Returns ``{"cases": [...], "unknown": [{"query", "missing"}], "invalid": n}``.
    A case is still scorable when *some* of its ids exist; only the unknown ids
    are reported, never scored as zero.
    """
    known = _known_ids(store)
    valid: list[dict] = []
    unknown: list[dict] = []
    invalid = 0
    for item in _items(cases):
        raw = _case_dict(item)
        query, expect = _query_of(raw), _expect_of(raw)
        if query is None or expect is None:
            invalid += 1
            continue
        missing = [fid for fid in expect if fid not in known]
        if missing:
            unknown.append({"query": query, "missing": missing})
        valid.append({"query": query, "expect": list(expect)})
    return {"cases": valid, "unknown": unknown, "invalid": invalid}


def _known_ids(store) -> set[int]:
    rows = _safe_rows(store, "SELECT fact_id FROM facts ORDER BY fact_id")
    return {int(r["fact_id"]) for r in rows}


# ------------------------------------------------------------------- reading
def _rows(store, sql: str, params=()) -> list[dict]:
    """Run one read against ``store``, whatever read surface it exposes."""
    read = getattr(store, "read", None)
    if callable(read):
        return [dict(r) for r in read(sql, params)]

    conn = getattr(store, "db", None)
    if conn is not None and hasattr(conn, "execute"):
        cur = conn.execute(sql, params)
        columns = [d[0] for d in (cur.description or [])]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    path = getattr(store, "db_path", None) or getattr(store, "path", None)
    if path:
        ro = sqlite3.connect(f"file:{Path(path).expanduser()}?mode=ro", uri=True,
                             timeout=5.0)
        try:
            ro.row_factory = sqlite3.Row
            return [dict(r) for r in ro.execute(sql, params)]
        finally:
            ro.close()

    raise TypeError("store must expose read(sql, params), db, or a db path")


def _safe_rows(store, sql: str, params=()) -> list[dict]:
    try:
        return _rows(store, sql, params)
    except (sqlite3.Error, TypeError, AttributeError):
        return []


class _Snapshot:
    """One read-only view of the facts and embeddings used for a run."""

    def __init__(self, facts: list[dict], model: str, embedded: dict[int, Any],
                 warnings: list[str]):
        self.facts = facts
        self.model = model
        self.embedded = embedded
        self.warnings = warnings


def _snapshot(store, embedder) -> _Snapshot:
    warnings: list[str] = []
    columns = {str(r.get("name") or "") for r in _safe_rows(store, "PRAGMA table_info(facts)")}
    if columns:
        entity_col = "entities" if "entities" in columns else "NULL AS entities"
        facts = _safe_rows(
            store,
            "SELECT fact_id, content, category, tags, trust_score, "
            f"{entity_col} FROM facts ORDER BY fact_id")
    else:  # unknown schema: the two columns scoring cannot work without
        facts = _safe_rows(store, "SELECT fact_id, content FROM facts ORDER BY fact_id")
    facts = [f for f in facts if f.get("fact_id") is not None]
    if not facts:
        warnings.append("store has no facts")

    model = ""
    if embedder is not None:
        model = str(getattr(embedder, "model", "") or "")
    if not model:
        models = _safe_rows(
            store,
            "SELECT model, COUNT(*) AS n FROM fact_embeddings "
            "GROUP BY model ORDER BY n DESC, model")
        model = str(models[0]["model"]) if models else ""

    embedded: dict[int, Any] = {}
    if model and embedder is not None:
        rows = _safe_rows(store, "SELECT fact_id, vec FROM fact_embeddings WHERE model = ?",
                          (model,))
        for row in rows:
            blob = row.get("vec")
            if blob is None:
                continue
            try:
                if isinstance(blob, (bytes, bytearray, memoryview)):
                    vec = vectors.unpack(bytes(blob))
                else:  # already-unpacked sequence from a custom store
                    vec = vectors.unpack(vectors.pack([float(x) for x in blob]))
                embedded[int(row["fact_id"])] = vec
            except (TypeError, ValueError):
                continue
        if not embedded:
            warnings.append(f"no embeddings for model {model!r}; scoring lexically")
    return _Snapshot(facts, model, embedded, warnings)


# ------------------------------------------------------------------- ranking
def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower())
            if len(t) >= 2 and t not in FTS_STOP]


def _bm25(query: str, facts: list[dict], k1: float = 1.2, b: float = 0.75):
    docs = [(int(f["fact_id"]), _tokens(str(f.get("content") or ""))) for f in facts]
    if not docs:
        return []
    terms = _tokens(query)
    if not terms:
        return []
    total = len(docs)
    avgdl = sum(len(d) for _, d in docs) / total or 1.0
    dfs = Counter()
    for _, tokens in docs:
        dfs.update(set(tokens))

    scored: list[tuple[int, float]] = []
    for fid, tokens in docs:
        tf = Counter(tokens)
        length = len(tokens) or 1
        score = 0.0
        for term in terms:
            if term not in tf:
                continue
            df = dfs[term] or 1
            idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            score += idf * (tf[term] * (k1 + 1.0)) / (
                tf[term] + k1 * (1.0 - b + b * length / avgdl))
        if score > 0:
            scored.append((fid, round(score, 9)))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored


def _embed_query(embedder, text: str):
    if embedder is None:
        return None
    try:
        if hasattr(embedder, "embed_one"):
            vec = embedder.embed_one(text)
        else:
            vecs = embedder.embed([text])
            vec = vecs[0] if vecs else None
    except Exception as exc:  # noqa: BLE001 - endpoint down means lexical scoring
        return {"error": f"{type(exc).__name__}: {exc}"}
    if not vec:
        return None
    try:
        return vectors.unpack(vectors.pack([float(x) for x in vec]))
    except (TypeError, ValueError):
        return None


def _vector_ranks(qvec, embedded: dict[int, Any]):
    if qvec is None or isinstance(qvec, dict) or not embedded:
        return []
    scored = [(fid, vectors.cosine(qvec, vec)) for fid, vec in embedded.items()]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored


def _search(query: str, snapshot: _Snapshot, k: int, embedder):
    """Top-``k`` fact ids for ``query``; RRF of BM25 and cosine ranks."""
    pool = max(k * 6, 40)
    lexical = _bm25(query, snapshot.facts)[:pool]
    qvec = _embed_query(embedder, query)
    warning = None
    if isinstance(qvec, dict):
        warning = f"embedding failed; scoring lexically: {qvec['error']}"
        qvec = None
    vector = _vector_ranks(qvec, snapshot.embedded)[:pool]

    scores: dict[int, float] = {}
    for rank, (fid, _) in enumerate(lexical, 1):
        scores[fid] = scores.get(fid, 0.0) + 1.0 / (RRF_K + rank)
    for rank, (fid, _) in enumerate(vector, 1):
        scores[fid] = scores.get(fid, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return [fid for fid, _ in ordered[:k]], warning, bool(vector)


# ------------------------------------------------------------------ evaluate
def evaluate(store, cases, *, k: int = DEFAULT_K, embedder=None) -> dict:
    """Score retrieval over the live store. Read-only and JSON-serialisable.

    ``cases`` is a list of ``{"query": str, "expect": [fact_id, ...]}`` dicts
    (the output of :func:`load_cases`, golden ``Case`` objects and a
    ``{"cases": [...]}`` wrapper are accepted too). A case hits when any of its
    expected ids appears in the top ``k``; its rank is the position of the first
    expected id and its reciprocal rank is ``1 / rank`` (0 for a miss).

    Cases whose expected ids are all unknown are *not* scored as misses: they
    are listed under ``unknown`` and excluded from the aggregates, so a stale
    golden file cannot silently depress the numbers.
    """
    try:
        k = max(1, int(k))
    except (TypeError, ValueError):
        k = DEFAULT_K

    snapshot = _snapshot(store, embedder)
    known = {int(f["fact_id"]) for f in snapshot.facts}
    items = _items(cases)

    results: list[dict] = []
    misses: list[dict] = []
    unknown: list[dict] = []
    invalid = 0
    scored = 0
    hits = 0
    rr_sum = 0.0
    hybrid_used = False
    warnings = list(snapshot.warnings)

    for item in items:
        raw = _case_dict(item)
        query, expect = _query_of(raw), _expect_of(raw)
        if query is None or expect is None:
            invalid += 1
            continue

        missing = [fid for fid in expect if fid not in known]
        usable = [fid for fid in expect if fid in known]
        if missing:
            unknown.append({"query": query, "missing": missing})
        if not usable:
            results.append({"query": query, "expect": list(expect), "got": [],
                            "rank": None, "hit": False, "hit_at_k": False,
                            "rr": 0.0, "reciprocal_rank": 0.0,
                            "scored": False, "unknown": missing})
            continue

        got, warning, used_vectors = _search(query, snapshot, k, embedder)
        hybrid_used = hybrid_used or used_vectors
        if warning and warning not in warnings:
            warnings.append(warning)
        rank = next((i for i, fid in enumerate(got, 1) if fid in usable), None)
        hit = rank is not None
        rr = 1.0 / rank if rank else 0.0

        scored += 1
        hits += 1 if hit else 0
        rr_sum += rr
        record = {"query": query, "expect": list(expect), "got": got,
                  "rank": rank, "hit": hit, "hit_at_k": hit, "rr": round(rr, 4),
                  "reciprocal_rank": round(rr, 4), "scored": True}
        if missing:
            record["unknown"] = missing
        results.append(record)
        if not hit:
            misses.append({"query": query, "expected": list(expect), "got": got})

    return {
        "k": k,
        "model": snapshot.model,
        "retrieval": "hybrid" if hybrid_used else "lexical",
        "cases": scored,
        "total": len(results),
        "hit_rate": round(hits / scored, 3) if scored else 0.0,
        "mrr": round(rr_sum / scored, 3) if scored else 0.0,
        "misses": misses,
        "unknown": unknown,
        "invalid": invalid,
        "warnings": warnings,
        "results": results,
    }


# ------------------------------------------------------------------- compare
def _report_rows(report) -> list[dict]:
    if not isinstance(report, dict):
        return []
    rows = report.get("results")
    if not isinstance(rows, list):
        cases = report.get("cases")
        rows = cases if isinstance(cases, list) else []
    return [dict(r) for r in rows if isinstance(r, dict)]


def _report_mrr(report) -> float:
    if isinstance(report, dict):
        try:
            return float(report.get("mrr") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _rank(row) -> int | None:
    rank = row.get("rank") if isinstance(row, dict) else None
    if isinstance(rank, bool) or not isinstance(rank, int):
        return None
    return rank if rank > 0 else None


def compare(before, after) -> dict:
    """Diff two :func:`evaluate` reports, case by case.

    Cases are matched by ``(query, occurrence)`` so duplicate queries do not
    collapse. Only cases present in both runs are judged; cases added or dropped
    between runs are reported separately and never look like a regression.

    ``degraded`` follows the operational definition: some case hit before and
    misses now, or overall MRR dropped by more than 0.05.
    """
    was_rows, now_rows = _report_rows(before), _report_rows(after)

    was: dict[tuple[str, int], dict] = {}
    seen: Counter = Counter()
    for row in was_rows:
        query = row.get("query")
        if not isinstance(query, str):
            continue
        was[(query, seen[query])] = row
        seen[query] += 1

    now: dict[tuple[str, int], dict] = {}
    seen = Counter()
    for row in now_rows:
        query = row.get("query")
        if not isinstance(query, str):
            continue
        now[(query, seen[query])] = row
        seen[query] += 1

    deltas: list[dict] = []
    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []
    hit_lost: list[str] = []

    def judge(key: tuple[str, int], old: dict, new: dict) -> None:
        query = key[0]
        old_rank, new_rank = _rank(old), _rank(new)
        old_hit = bool(old.get("hit")) or old_rank is not None
        new_hit = bool(new.get("hit")) or new_rank is not None
        old_rr = 1.0 / old_rank if old_rank else 0.0
        new_rr = 1.0 / new_rank if new_rank else 0.0

        if new_hit and not old_hit:
            verdict = "improved"
        elif old_hit and not new_hit:
            verdict = "regressed"
        elif old_rank and new_rank and new_rank < old_rank:
            verdict = "improved"
        elif old_rank and new_rank and new_rank > old_rank:
            verdict = "regressed"
        else:
            verdict = "unchanged"

        {"improved": improved, "regressed": regressed,
         "unchanged": unchanged}[verdict].append(query)
        if old_hit and not new_hit:
            hit_lost.append(query)
        deltas.append({"query": query, "before_rank": old_rank,
                       "after_rank": new_rank,
                       "before_rr": round(old_rr, 4), "after_rr": round(new_rr, 4),
                       "delta_rr": round(new_rr - old_rr, 4),
                       "change": verdict})

    for key, new in now.items():
        old = was.get(key)
        if old is not None:
            judge(key, old, new)
    for key, old in was.items():
        if key not in now:
            old_rank = _rank(old)
            deltas.append({"query": key[0], "before_rank": old_rank,
                           "after_rank": None,
                           "before_rr": round(1.0 / old_rank, 4) if old_rank else 0.0,
                           "after_rr": None, "delta_rr": None, "change": "removed"})

    before_mrr, after_mrr = _report_mrr(before), _report_mrr(after)
    mrr_drop = round(before_mrr - after_mrr, 6)
    return {
        "deltas": deltas,
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "hit_lost": hit_lost,
        "added": [key[0] for key in now if key not in was],
        "removed": [key[0] for key in was if key not in now],
        "before_mrr": before_mrr,
        "after_mrr": after_mrr,
        "mrr_delta": round(after_mrr - before_mrr, 6),
        "degraded": bool(hit_lost) or mrr_drop > 0.05,
    }


# ---------------------------------------------------------------------- gate
def gate(result, *, min_hit: float = 0.85, min_mrr: float = 0.80) -> dict:
    """Threshold gate over an :func:`evaluate` report: ``{"ok", "reasons"}``.

    An empty run fails loudly (a gate with nothing to score is not a pass).
    Unknown expected ids are surfaced under ``warnings``; they are excluded from
    scoring, so they cannot mask a threshold failure.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    try:
        cases = int((result or {}).get("cases") or 0)
    except (TypeError, ValueError, AttributeError):
        cases = 0
    try:
        hit_rate = float((result or {}).get("hit_rate") or 0.0)
        mrr = float((result or {}).get("mrr") or 0.0)
    except (TypeError, ValueError, AttributeError):
        hit_rate = mrr = 0.0

    if cases <= 0:
        reasons.append("no scored cases")
    else:
        if hit_rate < min_hit:
            reasons.append(f"hit_rate {hit_rate:.3f} < {min_hit:.3f}")
        if mrr < min_mrr:
            reasons.append(f"mrr {mrr:.3f} < {min_mrr:.3f}")

    unknown = (result or {}).get("unknown") if isinstance(result, dict) else None
    if isinstance(unknown, list) and unknown:
        ids = sorted({fid for item in unknown if isinstance(item, dict)
                      for fid in (item.get("missing") or [])})
        warnings.append("unknown expected ids: " + ", ".join(str(i) for i in ids))

    return {"ok": not reasons, "reasons": reasons, "warnings": warnings}
