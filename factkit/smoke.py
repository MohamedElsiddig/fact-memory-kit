"""Retrieval smoke gate for maintenance batches.

Hand-labelled ``{"query": str, "expect": [fact_id, ...]}`` cases are replayed
against semantic recall twice: before a maintenance batch and after it. A case
passes when *any* expected fact id shows up in the top-k hits.

The gate is deliberately forgiving about the embedding endpoint: ``prefetch``
returning ``None`` (endpoint down) marks the case skipped instead of failed, so a
dead embedder cannot block a maintenance run. A run where nothing could be
checked reports ``ok=True`` with ``degraded=True`` — callers should say so
loudly rather than treat it as a pass.

``compare`` only judges queries that were actually checked in *both* runs, so
newly added cases and endpoint hiccups never look like regressions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_TIMEOUT = 1.5


# ------------------------------------------------------------------- cases
def load_cases(path: str | Path) -> list[dict]:
    """Read a JSON list of ``{"query": str, "expect": [int, ...]}`` cases.

    A missing file yields ``[]`` (nothing to gate on). A dict payload with a
    ``"cases"`` list is accepted as a convenience for golden-style files.
    Non-dict entries are dropped; full validation happens in ``run_smoke``.
    """
    p = Path(path).expanduser()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    if isinstance(data, dict):
        data = data.get("cases") or []
    if not isinstance(data, list):
        return []
    return [c for c in data if isinstance(c, dict)]


def _expect_ids(value) -> tuple[int, ...] | None:
    """Validated expected ids, or None when the case is junk."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set)):
        return None
    ids: list[int] = []
    for item in value:
        # bool is an int subclass; True must not silently become fact 1.
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        ids.append(item)
    return tuple(ids) or None


def _hit_ids(hits, top_k: int) -> list[int]:
    """Fact ids from a prefetch result, tolerating dicts, rows and junk."""
    out: list[int] = []
    for hit in hits:
        if len(out) >= top_k:
            break
        fid = hit.get("fact_id") if isinstance(hit, dict) else getattr(hit, "fact_id", None)
        if isinstance(fid, bool) or not isinstance(fid, int):
            try:
                fid = int(fid)
            except (TypeError, ValueError):
                continue
        out.append(fid)
    return out


# -------------------------------------------------------------------- run
def run_smoke(store, cases, *, prefetch=None, top_k: int = DEFAULT_TOP_K,
              timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Replay ``cases`` through semantic recall. Never raises.

    ``prefetch`` is called as ``prefetch(store, query, limit=top_k,
    timeout=timeout)`` — the same shape as :func:`factkit.holo.prefetch`. It may
    return a list of hit dicts, or ``None`` to mean "endpoint unavailable".

    Returns ``ok``, ``checked`` (cases actually scored), ``failures``
    (``{"query", "expect", "got"}``), ``skipped`` (endpoint-down cases),
    ``degraded`` (nothing was checked) and per-case ``cases`` records, which
    ``compare`` needs.
    """
    try:
        return _run_smoke(store, cases, prefetch=prefetch, top_k=top_k, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - a smoke gate must never block a batch
        logger.warning("smoke gate crashed: %s", exc)
        return {"ok": False, "checked": 0, "failures": [], "skipped": 0,
                "degraded": True, "invalid": 0, "cases": [],
                "error": f"{type(exc).__name__}: {exc}"}


def _run_smoke(store, cases, *, prefetch=None, top_k=DEFAULT_TOP_K,
               timeout=DEFAULT_TIMEOUT) -> dict:
    try:
        k = max(1, int(top_k))
    except (TypeError, ValueError):
        k = DEFAULT_TOP_K
    try:
        seconds = float(timeout)
    except (TypeError, ValueError):
        seconds = DEFAULT_TIMEOUT
    if seconds <= 0:
        seconds = DEFAULT_TIMEOUT

    recall = prefetch
    if recall is None:
        try:
            from .holo import prefetch as recall  # late: keeps holo optional
        except Exception:  # noqa: BLE001 - no embedder module: everything skips
            recall = None

    try:
        items = list(cases or [])
    except TypeError:
        items = []

    records: list[dict] = []
    failures: list[dict] = []
    skipped = 0
    invalid = 0

    for case in items:
        query = case.get("query") if isinstance(case, dict) else None
        expect = _expect_ids(case.get("expect")) if isinstance(case, dict) else None
        if not isinstance(query, str) or not query.strip() or expect is None:
            invalid += 1
            continue

        try:
            hits = recall(store, query, limit=k, timeout=seconds) if recall else None
        except Exception:  # noqa: BLE001 - a flaky endpoint is a skip, not a fail
            hits = None

        if hits is None:
            skipped += 1
            records.append({"query": query, "expect": list(expect), "got": [],
                            "ok": False, "skipped": True})
            continue

        try:
            got = _hit_ids(hits, k)
        except Exception:  # noqa: BLE001 - malformed hits score as a miss
            got = []

        passed = any(fid in got for fid in expect)
        records.append({"query": query, "expect": list(expect), "got": got,
                        "ok": passed, "skipped": False})
        if not passed:
            failures.append({"query": query, "expect": list(expect), "got": got})

    checked = sum(1 for r in records if not r["skipped"])
    return {
        "ok": not failures,
        "checked": checked,
        "failures": failures,
        "skipped": skipped,
        "degraded": checked == 0,
        "invalid": invalid,
        "cases": records,
        "top_k": k,
    }


# ---------------------------------------------------------------- compare
def _checked(report) -> dict[str, bool]:
    """Query -> passed, for cases the report actually scored."""
    out: dict[str, bool] = {}
    if not isinstance(report, dict):
        return out
    records = report.get("cases")
    if records is None:
        records = report.get("results")
    if isinstance(records, list):
        for rec in records:
            if not isinstance(rec, dict) or rec.get("skipped"):
                continue
            query = rec.get("query")
            if not isinstance(query, str) or not query:
                continue
            passed = rec.get("ok")
            if not isinstance(passed, bool):
                passed = not rec.get("got")
            out[query] = passed
        return out
    # Minimal report without per-case rows: only failures are known.
    for fail in report.get("failures") or []:
        if isinstance(fail, dict) and isinstance(fail.get("query"), str):
            out[fail["query"]] = False
    return out


def compare(before: dict, after: dict) -> dict:
    """Queries that passed before and fail after.

    Only queries checked (not skipped) in both reports are judged; cases added
    after the baseline are ignored, and a case the endpoint skipped after the
    batch is inconclusive, not a regression.
    """
    try:
        was, now = _checked(before), _checked(after)
    except Exception:  # noqa: BLE001 - compare must not block a report either
        was, now = {}, {}
    regressed = [q for q, passed in was.items() if passed and q in now and not now[q]]
    return {"regressed": regressed, "ok": not regressed}
