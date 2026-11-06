"""Golden gate: fail-to-pass regression tests for retrieval.

Ship-with defaults, no LLM, no embeddings needed to *score*: the gate replays
hand-labelled queries against the live index and compares against thresholds.
If no golden file exists, one is seeded from the corpus so the gate is never
silently skipped.
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_THRESHOLDS = {
    "r_at_5": 0.85,     # first expected fact in the top 5
    "mrr": 0.80,        # mean reciprocal rank
    "fp": 0,            # negatives that returned anything
    "hit_rate": 0.90,   # fraction of positive queries with any hit
}

GOLDEN_NAME = "golden.json"


@dataclass
class Case:
    q: str
    facts: list[int] = field(default_factory=list)
    kind: str = "lexical"          # lexical | paraphrase | negative
    note: str = ""


@dataclass
class Result:
    metrics: dict
    rows: list[dict]
    failures: list[str]

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self, verbose: bool = False) -> str:
        lines = []
        if verbose:
            lines.append(f"{'query':<46} {'kind':<11} {'expect':<12} {'result':<14} got")
            for r in self.rows:
                lines.append(f"{r['q'][:45]:<46} {r['kind']:<11} "
                             f"{str(r['expect'])[:11]:<12} {r['result']:<14} {r['got']}")
            lines.append("")
        for key, value in self.metrics.items():
            lines.append(f"{key:<20} {value}")
        if self.failures:
            lines.append("")
            lines.append("GATE FAIL: " + "; ".join(self.failures))
        else:
            lines.append("")
            lines.append("GATE PASS")
        return "\n".join(lines)


def _default_path(root: Path | None = None) -> Path:
    base = root or Path.cwd()
    return base / GOLDEN_NAME


def seed_cases(store, limit: int = 24) -> list[dict]:
    """Derive lexical queries from the corpus: 4 significant words per fact."""
    cases: list[dict] = []
    stop = {"the", "and", "for", "with", "that", "this", "from", "are", "was",
            "has", "have", "its", "but", "you", "your", "our", "all", "any",
            "can", "when", "then", "than", "into", "out", "use", "used", "via",
            "only", "also", "must", "should", "always", "never", "user", "wants"}
    for fact in store.list_facts(limit=limit):
        words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.\-/]+",
                                       fact["content"].lower())
                 if len(w) > 3 and w not in stop]
        if len(words) < 3:
            continue
        q = " ".join(words[:4])
        cases.append({"q": q, "facts": [fact["fact_id"]], "kind": "lexical"})
    return cases


def load_cases(path: Path | None, store=None) -> tuple[list[Case], Path, dict]:
    path = path or _default_path()
    data = None
    if path.exists():
        data = json.loads(path.read_text())
    if not data:
        seeded = seed_cases(store) if store is not None else []
        data = {"thresholds": DEFAULT_THRESHOLDS, "cases": seeded}
        if seeded and path:
            path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    cases = [Case(q=c["q"], facts=list(c.get("facts") or []),
                  kind=c.get("kind", "lexical"), note=c.get("note", ""))
             for c in data.get("cases", [])]
    thresholds = {**DEFAULT_THRESHOLDS, **(data.get("thresholds") or {})}
    return cases, path, thresholds


def rank_of(expected: set[int], got: list[int]) -> int | None:
    for i, fid in enumerate(got, 1):
        if fid in expected:
            return i
    return None


def run(store, search_fn, cases: list[Case], k: int = 5) -> Result:
    positives = [c for c in cases if c.kind != "negative"]
    negatives = [c for c in cases if c.kind == "negative"]

    r_at_k = 0.0
    mrr = 0.0
    hits_any = 0
    rows: list[dict] = []

    for case in positives:
        got = [h.fact_id for h in search_fn(case.q, k=8)]
        rank = rank_of(set(case.facts), got)
        if rank and rank <= k:
            r_at_k += 1
        if rank:
            mrr += 1.0 / rank
            hits_any += 1
        rows.append({"q": case.q, "kind": case.kind,
                     "expect": case.facts[:2],
                     "result": f"rank {rank}" if rank else "MISS",
                     "got": got[:5]})

    fp = 0
    fp_rows: list[dict] = []
    for case in negatives:
        got = [h.fact_id for h in search_fn(case.q, k=5)]
        if got:
            fp += 1
        fp_rows.append({"q": case.q, "kind": "negative", "expect": [],
                        "result": "FALSE POSITIVE" if got else "clean",
                        "got": got[:5]})

    n = len(positives)
    metrics = {
        "queries_scored": n,
        "negatives": len(negatives),
        "r_at_5": round(r_at_k / n, 3) if n else 0.0,
        "mrr": round(mrr / n, 3) if n else 0.0,
        "hit_rate": round(hits_any / n, 3) if n else 0.0,
        "false_positives": fp,
    }
    return Result(metrics=metrics, rows=rows + fp_rows, failures=[])


def check(result: Result, thresholds: dict | None = None) -> Result:
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    failures: list[str] = []
    m = result.metrics
    if m["queries_scored"] == 0 and m.get("negatives", 0) == 0:
        failures.append("empty golden set: no cases to score")
    if m["queries_scored"]:
        if m["r_at_5"] < th["r_at_5"]:
            failures.append(f"r_at_5 {m['r_at_5']} < {th['r_at_5']}")
        if m["mrr"] < th["mrr"]:
            failures.append(f"mrr {m['mrr']} < {th['mrr']}")
        if m["hit_rate"] < th["hit_rate"]:
            failures.append(f"hit_rate {m['hit_rate']} < {th['hit_rate']}")
    if m["false_positives"] > th["fp"]:
        failures.append(f"false_positives {m['false_positives']} > {th['fp']}")
    result.failures = failures
    return result


def write_cases(cases: list[Case], path: Path | None = None,
                thresholds: dict | None = None) -> Path:
    path = path or _default_path()
    payload = {
        "thresholds": {**DEFAULT_THRESHOLDS, **(thresholds or {})},
        "cases": [{"q": c.q, "facts": c.facts, "kind": c.kind,
                   **({"note": c.note} if c.note else {})} for c in cases],
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    return path


def summary_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": round(statistics.fmean(values), 3),
            "p50": round(statistics.median(values), 3)}


def remap(path: Path | None, old_id: int, new_id: int) -> int:
    """Point golden cases at a survivor after a retirement (keeps the gate honest)."""
    path = path or _default_path()
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    changed = 0
    for case in data.get("cases", []):
        facts = case.get("facts") or []
        if old_id in facts:
            case["facts"] = [new_id if f == old_id else f for f in facts]
            changed += 1
    if changed:
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    return changed
