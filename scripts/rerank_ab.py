#!/usr/bin/env python3
"""A/B the embedding-mode re-ranker on prefetched candidates.

For every labelled case the script takes the live ``prefetch`` head (k=8 by
default), runs it through :func:`factkit.rerank.bench` in ``embedding`` mode and
reports hit@k, MRR, mean latency, accepted/timed-out/error counts before and
after. The run is read-only: the live store is opened through
``holo.ReadOnlyStore`` and the re-ranker never writes.

Cases are ``{"query": ..., "expect": [fact_id, ...]}``. A bare list, an object
with a ``"cases"`` list and the ``q``/``facts`` aliases used by
``examples/golden.example.json`` are all accepted.

Exit codes:

* ``0`` — hit@k or MRR improved (or, for ``--fake-embed``, the plumbing ran).
* ``1`` — no gain: leave the embedding re-ranker off.
* ``2`` — unusable run: no cases, no endpoint for every case, no candidates at
  all, or the re-ranker failed for every case.

Examples::

    python scripts/rerank_ab.py cases/live_cases.json
    python scripts/rerank_ab.py cases/live_cases.json --db ~/.hermes/memory_store.db --k 5
    python scripts/rerank_ab.py cases/live_cases.json --json
    python scripts/rerank_ab.py cases/live_cases.json --fake-embed   # plumbing only

``--fake-embed`` swaps in a deterministic hash embedder and bypasses
``prefetch``'s query cache. Its metrics are synthetic and are never a quality
signal; the quality gate is therefore suppressed for that mode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # running as a plain script, not an installed module
    sys.path.insert(0, str(ROOT))

DEFAULT_DB = Path(
    os.environ.get("FACTKIT_DB", "~/.hermes/memory_store.db")
).expanduser()
EXPECT_KEYS = (
    "expect",
    "expected",
    "expected_ids",
    "expected_facts",
    "facts",
    "fact_ids",
    "ids",
    "relevant",
)
QUERY_KEYS = ("query", "q", "question")
_DIGITS = re.compile(r"[+-]?\d+")


class UnusableRun(RuntimeError):
    """The A/B cannot produce a meaningful number."""


# ---------------------------------------------------------------- cases file
def _query_of(item: dict) -> str:
    for key in QUERY_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _as_id(value: object) -> object | None:
    """Normalise an expected id: numeric strings become ints, dicts unwrap."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return int(text) if _DIGITS.fullmatch(text) else text
    if isinstance(value, dict):
        for key in ("fact_id", "id"):
            if key in value:
                return _as_id(value[key])
        return None
    return None


def _expect_of(item: dict) -> tuple[bool, list]:
    """Return ``(found, ids)``; ``found`` distinguishes [] from a missing key."""
    for key in EXPECT_KEYS:
        if key not in item:
            continue
        raw = item[key]
        if raw is None:
            return True, []
        if isinstance(raw, (int, str, dict)):
            raw = [raw]
        if not isinstance(raw, (list, tuple, set)):
            raise ValueError(
                f"'expect' must be a list of fact ids, got {type(raw).__name__}"
            )
        ids = []
        for value in raw:
            normalised = _as_id(value)
            if normalised is not None:
                ids.append(normalised)
        return True, ids
    return False, []


def load_cases(path: str | Path) -> tuple[list[dict], int]:
    p = Path(path).expanduser()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"cases file not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {p}: {exc}") from exc
    if isinstance(data, dict):
        data = data.get("cases")
    if not isinstance(data, list):
        raise ValueError("cases file must be a list or an object with a 'cases' list")

    cases: list[dict] = []
    skipped = 0
    for item in data:
        if not isinstance(item, dict):
            skipped += 1
            continue
        query = _query_of(item)
        found, expect = _expect_of(item)
        if not query or not found:
            skipped += 1
            continue
        cases.append({"query": query, "expect": expect})
    return cases, skipped


# -------------------------------------------------------------- local store
def _open_store(path: str | Path):
    from factkit.holo import ReadOnlyStore

    try:
        return ReadOnlyStore(path)
    except Exception as exc:  # noqa: BLE001 - a bad path must be one clear error
        raise UnusableRun(
            f"cannot open store {Path(path).expanduser()}: {exc}"
        ) from exc


def _known_ids(store) -> set:
    try:
        return {int(row["fact_id"]) for row in store.read("SELECT fact_id FROM facts")}
    except Exception:  # noqa: BLE001 - diagnostics only
        return set()


def _all_facts(store) -> list[dict]:
    rows = [
        dict(row)
        for row in store.read("SELECT fact_id, content FROM facts ORDER BY fact_id")
    ]
    return [
        {"fact_id": int(row["fact_id"]), "content": str(row.get("content") or "")}
        for row in rows
        if row.get("fact_id") is not None
    ]


def _sorted_candidates(rows: list[dict], limit: int) -> list[dict]:
    def key(hit: dict) -> tuple[float, tuple[int, object]]:
        fact_id = hit.get("fact_id")
        if isinstance(fact_id, int) and not isinstance(fact_id, bool):
            id_key = (0, fact_id)
        else:
            try:
                id_key = (0, int(str(fact_id)))
            except (TypeError, ValueError):
                id_key = (1, str(fact_id))
        return (-float(hit.get("score") or 0.0), id_key)

    return sorted(rows, key=key)[:limit]


# ------------------------------------------------------------- fake embedder
class HashEmbedder:
    """Deterministic bag-of-(tokens, bigrams) embedder for offline plumbing."""

    def __init__(self, dim: int = 128):
        self.dim = max(8, int(dim))

    def embed(self, texts) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]

    def try_embed(self, texts) -> list[list[float]] | None:
        return self.embed(list(texts))

    def embed_one(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
        grams = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        if not grams:
            grams = [str(text or "")]
        vector = [0.0] * self.dim
        for gram in grams:
            hashed = zlib.crc32(gram.encode("utf-8"))
            vector[hashed % self.dim] += 1.0 if (hashed >> 16) & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def health(self) -> dict:
        return {"ok": True, "synthetic": True, "model": f"hash-{self.dim}"}


def _cosine(left, right) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = norm_a = norm_b = 0.0
    for a, b in zip(left, right):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / math.sqrt(norm_a * norm_b)))


def _fake_prefetch(
    facts: list[dict], query: str, embedder: HashEmbedder, limit: int
) -> list[dict]:
    """Hash-based stand-in for ``holo.prefetch``; same candidate shape."""
    query_vector = embedder.embed_one(query)
    rows = [
        {
            "fact_id": fact["fact_id"],
            "content": fact["content"],
            "score": round(max(0.0, _cosine(query_vector, fact["vector"])), 6),
        }
        for fact in facts
    ]
    return _sorted_candidates(rows, limit)


def _synthetic_candidates(case: dict, embedder: HashEmbedder, limit: int) -> list[dict]:
    """Candidates for ``--fake-embed`` when no store is usable."""
    query = case["query"]
    query_vector = embedder.embed_one(query)
    rows = []
    for index, fact_id in enumerate(case["expect"] or [-1]):
        content = f"synthetic answer {index} for {query}"
        rows.append({"fact_id": fact_id, "content": content})
    for index in range(max(0, limit - len(rows))):
        rows.append(
            {
                "fact_id": -(1000 + index),
                "content": f"unrelated synthetic distractor {index}",
            }
        )
    for row in rows:
        row["score"] = round(
            max(0.0, _cosine(query_vector, embedder.embed_one(row["content"]))), 6
        )
    return _sorted_candidates(rows, limit)


# ----------------------------------------------------------------- real path
def _embedding_client(db: str | Path, timeout_ms: float):
    from factkit.config import Config
    from factkit.embed import from_config

    client = from_config(Config.load(str(Path(db).expanduser())))
    # One request may not outlive the re-rank budget; the deadline is enforced
    # after each call, so a loose socket timeout would defeat it.
    client.timeout = max(0.02, min(float(timeout_ms) / 1000.0, 1.5))
    client.retries = 0
    return client


def _real_prefetch(store, query: str, limit: int) -> tuple[list[dict], bool]:
    from factkit import holo

    hits = holo.prefetch(store, query, limit=limit, min_score=0.0, min_trust=0.0)
    if hits is None:
        return [], True
    return [
        {
            "fact_id": int(hit["fact_id"]),
            "content": str(hit.get("content") or ""),
            "score": float(hit.get("score") or 0.0),
        }
        for hit in hits
    ], False


# --------------------------------------------------------------------- rerank
def _rerank_params(fn) -> set[str]:
    """Fail loudly when the checkout has no first-party embedding mode."""
    import inspect

    try:
        params = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError) as exc:
        raise UnusableRun(f"cannot inspect factkit.rerank.rerank: {exc}") from exc
    missing = {"mode", "embed_client"} - params
    if missing:
        raise UnusableRun(
            "factkit.rerank has no first-party embedding mode "
            f"(missing parameters: {', '.join(sorted(missing))}); "
            "update factkit/rerank.py before running this A/B"
        )
    return params


def _first_rank(order: list, relevant: set):
    for position, fact_id in enumerate(order, 1):
        if fact_id in relevant:
            return position
    return None


def _fmt_rank(rank) -> str:
    return "-" if rank is None else str(rank)


def _build_report(
    args,
    cases: list[dict],
    skipped: int,
    db: str,
    store_label: str,
    no_endpoint: int,
    empty: int,
    prefetch_ms: float,
    unknown: list[dict],
    bench: dict,
) -> dict:
    n = int(bench.get("n") or 0)
    hit_before = float(bench.get("hit_at_k_before") or 0.0)
    hit_after = float(bench.get("hit_at_k_after") or 0.0)
    mrr_before = float(bench.get("mrr_before") or 0.0)
    mrr_after = float(bench.get("mrr_after") or 0.0)
    gain = (hit_after > hit_before + 1e-12) or (mrr_after - mrr_before > args.min_gain)

    details = list(bench.get("details") or [])
    for detail, case in zip(details, cases):
        expect = set(case["expect"])
        detail["expect"] = list(case["expect"])
        detail["before_rank"] = _first_rank(detail.get("before_order") or [], expect)
        detail["after_rank"] = _first_rank(detail.get("after_order") or [], expect)

    if args.fake_embed:
        verdict = "SYNTHETIC run: plumbing only, metrics are NOT a quality signal"
    elif gain:
        verdict = (
            f"GAIN: hit@{args.k} {hit_before:.3f}->{hit_after:.3f}, "
            f"MRR {mrr_before:.3f}->{mrr_after:.3f}"
        )
    else:
        verdict = (
            f"NO GAIN: hit@{args.k} {hit_before:.3f}->{hit_after:.3f}, "
            f"MRR {mrr_before:.3f}->{mrr_after:.3f}; leave embedding re-rank off"
        )

    report = dict(bench)
    report.update(
        {
            "synthetic": bool(args.fake_embed),
            "gate": "suppressed (synthetic)" if args.fake_embed else "quality",
            "db": db,
            "store": store_label,
            "n_total": len(cases),
            "skipped_unlabelled": skipped,
            "prefetch_limit": args.limit,
            "timeout_ms": float(args.timeout_ms),
            "min_gain": float(args.min_gain),
            "mean_prefetch_latency_ms": round(prefetch_ms / (n or 1), 3),
            "no_endpoint": no_endpoint,
            "empty_candidates": empty,
            "unknown_expectations": unknown,
            "gain": bool(gain),
            "verdict": verdict,
            "details": details,
        }
    )
    return report


def _unusable_reason(report: dict) -> str | None:
    n = int(report.get("n") or 0)
    if n == 0:
        return "no labelled cases to score"
    if report["no_endpoint"] == n:
        return (
            "the embedding endpoint returned nothing for every case "
            "(start it or use --fake-embed for a plumbing check)"
        )
    if report["empty_candidates"] == n:
        return "prefetch returned no candidates for every case; check the store and its embeddings"
    if report["errors"] == n:
        return "the re-ranker failed for every case; see details"
    return None


def _print_human(report: dict) -> None:
    if report["synthetic"]:
        print(
            "SYNTHETIC — deterministic hash embedder; plumbing check only, "
            "these metrics are NOT a quality signal."
        )
    print(
        f"db={report['db']} ({report['store']})  cases={report['n_total']}  "
        f"k={report['k']}  prefetch={report['prefetch_limit']}  "
        f"timeout={report['timeout_ms']:g}ms  min_gain={report['min_gain']:g}"
    )
    if report["skipped_unlabelled"]:
        print(
            f"skipped {report['skipped_unlabelled']} case(s) without a query or label"
        )
    if report["no_endpoint"]:
        print(
            f"warning: {report['no_endpoint']} case(s) had no embedding endpoint; "
            "scored as empty"
        )
    if report["empty_candidates"]:
        print(f"warning: {report['empty_candidates']} case(s) had zero candidates")
    if report["unknown_expectations"]:
        print(
            f"warning: {len(report['unknown_expectations'])} case(s) expect ids "
            "that are not in the store"
        )

    print(f"{'query':<44} {'before':>6} {'after':>6} {'reason':>13} {'ms':>8}")
    print("-" * 81)
    for detail in report["details"]:
        query = str(detail.get("query") or "")
        if len(query) > 44:
            query = query[:41] + "..."
        print(
            f"{query:<44} {_fmt_rank(detail.get('before_rank')):>6} "
            f"{_fmt_rank(detail.get('after_rank')):>6} "
            f"{str(detail.get('reason') or ''):>13} "
            f"{float(detail.get('latency_ms') or 0.0):>8.1f}"
        )
    print("-" * 81)

    print(
        f"hit@{report['k']}  before {report['hit_at_k_before']:.3f}  "
        f"after {report['hit_at_k_after']:.3f}  "
        f"delta {report['hit_at_k_after'] - report['hit_at_k_before']:+.3f}"
    )
    print(
        f"MRR{'':<5} before {report['mrr_before']:.3f}  "
        f"after {report['mrr_after']:.3f}  "
        f"delta {report['mrr_after'] - report['mrr_before']:+.3f}"
    )
    print(
        f"mean latency {report['mean_latency_ms']:.1f} ms "
        f"(prefetch {report['mean_prefetch_latency_ms']:.1f} ms)  "
        f"accepted {report['accepted']}/{report['n']}  "
        f"timeouts {report['timeouts']}  errors {report['errors']}"
    )
    print(f"VERDICT: {report['verdict']}")


def _validate_args(parser: argparse.ArgumentParser, args) -> None:
    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.timeout_ms < 0:
        parser.error("--timeout-ms must be >= 0")
    if args.min_gain < 0:
        parser.error("--min-gain must be >= 0")
    if args.limit < args.k:
        print(
            f"warning: prefetch depth {args.limit} is smaller than k={args.k}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A/B the embedding-mode re-ranker on prefetched candidates."
    )
    parser.add_argument(
        "cases", help="JSON cases file with queries and expected fact ids"
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help="store to read (default: $FACTKIT_DB or ~/.hermes/memory_store.db)",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="cutoff for hit@k (default: 5)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="prefetch depth passed to holo.prefetch (default: 8)",
    )
    parser.add_argument(
        "--timeout-ms",
        type=float,
        default=250.0,
        help="re-rank latency budget in ms (default: 250)",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.01,
        help="MRR improvement the re-ranker must clear (default: 0.01)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the raw report as JSON"
    )
    parser.add_argument(
        "--fake-embed",
        action="store_true",
        help="offline hash embedder for plumbing only (not a quality signal)",
    )
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    try:
        cases, skipped = load_cases(args.cases)
    except ValueError as exc:
        parser.error(str(exc))
    if not cases:
        print(
            "error: no usable cases (each case needs a query and an expect list)",
            file=sys.stderr,
        )
        return 2

    try:
        from factkit import rerank as reranker
    except Exception as exc:  # noqa: BLE001 - missing/broken module is one clear error
        print(f"error: cannot import factkit.rerank: {exc}", file=sys.stderr)
        return 2

    try:
        _rerank_params(reranker.rerank)
        # bench forwards the same kwargs; reject a half-updated module up front
        # instead of raising a TypeError after the prefetch work is done.
        _rerank_params(reranker.bench)
    except UnusableRun as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    store = None
    store_facts: list[dict] = []
    known: set | None = None
    store_label = "synthetic-cases" if args.fake_embed else "read-only store"
    db_label = str(Path(args.db).expanduser())

    if args.fake_embed:
        embedder = HashEmbedder()
        try:
            store = _open_store(args.db)
            store_facts = _all_facts(store)
            if store_facts:
                for fact in store_facts:
                    fact["vector"] = embedder.embed_one(fact["content"])
                store_label = "store-facts (synthetic scorer)"
                known = {fact["fact_id"] for fact in store_facts}
        except Exception as exc:  # noqa: BLE001 - fake mode degrades to synthetic
            print(
                f"note: fake-mode store read failed ({exc}); "
                "using cases-only synthetic candidates",
                file=sys.stderr,
            )
            store, store_facts, known = None, [], None
        client = embedder
    else:
        try:
            store = _open_store(args.db)
        except UnusableRun as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        known = _known_ids(store)
        client = _embedding_client(args.db, args.timeout_ms)

    bench_cases: list[dict] = []
    no_endpoint = 0
    empty = 0
    prefetch_ms = 0.0
    unknown: list[dict] = []

    try:
        for case in cases:
            if known is not None:
                missing = [
                    fact_id for fact_id in case["expect"] if fact_id not in known
                ]
                if missing:
                    unknown.append({"query": case["query"], "missing": missing})
            if args.fake_embed:
                started = time.monotonic()
                if store_facts:
                    candidates = _fake_prefetch(
                        store_facts, case["query"], embedder, args.limit
                    )
                else:
                    candidates = _synthetic_candidates(case, embedder, args.limit)
                prefetch_ms += (time.monotonic() - started) * 1000.0
            else:
                started = time.monotonic()
                candidates, missing_endpoint = _real_prefetch(
                    store, case["query"], args.limit
                )
                prefetch_ms += (time.monotonic() - started) * 1000.0
                no_endpoint += int(missing_endpoint)
            empty += int(not candidates)
            bench_cases.append(
                {
                    "query": case["query"],
                    "expect": case["expect"],
                    "candidates": candidates,
                }
            )
    except Exception as exc:  # noqa: BLE001 - a store read failure is one clear error
        print(f"error: prefetch failed: {exc}", file=sys.stderr)
        return 2

    try:
        bench = reranker.bench(
            client,
            bench_cases,
            k=args.k,
            timeout_ms=args.timeout_ms,
            min_gain=args.min_gain,
            mode="embedding",
            embed_client=client,
        )
    except TypeError as exc:
        print(
            f"error: factkit.rerank.bench rejected the embedding-mode call: {exc}",
            file=sys.stderr,
        )
        return 2
    report = _build_report(
        args,
        cases,
        skipped,
        db_label,
        store_label,
        no_endpoint,
        empty,
        prefetch_ms,
        unknown,
        bench,
    )

    reason = _unusable_reason(report)
    if reason:
        report["gain"] = False
        report["verdict"] = f"UNUSABLE: {reason}"
    report["exit_code"] = (
        2 if reason else (0 if args.fake_embed or report["gain"] else 1)
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        _print_human(report)
    if reason:
        print(f"error: {reason}", file=sys.stderr)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
