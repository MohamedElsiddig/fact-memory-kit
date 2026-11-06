#!/usr/bin/env python3
"""Benchmark the HRR candidate re-ranker against a lexical baseline.

Reads a JSON cases file (a list of ``{"query": ..., "expect": [fact_id, ...]}``,
or an object with a ``"cases"`` list), ranks the live facts both ways and
prints a table plus a one-line verdict. Exit code is 1 unless HRR beats the
lexical baseline on *both* hit@k and MRR — a deliberate gate, because this
module is a candidate re-ranker and not yet wired into the live hook.

Examples::

    python scripts/hrr_bench.py cases.json
    python scripts/hrr_bench.py cases.json --db ~/.hermes/memory_store.db --k 5
    python scripts/hrr_bench.py cases.json --json

The run is read-only (``holo.ReadOnlyStore``); it never writes to the store.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # running as a plain script, not an installed module
    sys.path.insert(0, str(ROOT))

from factkit import hrr  # noqa: E402
from factkit.holo import ReadOnlyStore  # noqa: E402

DEFAULT_DB = Path(os.environ.get("FACTKIT_DB", "~/.hermes/memory_store.db")).expanduser()


def load_cases(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases", [])
    if not isinstance(data, list):
        raise SystemExit("cases file must be a list or an object with a 'cases' list")
    cases = []
    for item in data:
        if not isinstance(item, dict) or "query" not in item:
            raise SystemExit("each case needs at least a 'query'")
        expect = item.get("expect") or []
        if not isinstance(expect, list):
            raise SystemExit("'expect' must be a list of fact ids")
        cases.append({"query": str(item["query"]), "expect": [int(x) for x in expect]})
    return cases


def _fmt_rank(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def _fmt_case(case: dict) -> str:
    query = case["query"]
    if len(query) > 40:
        query = query[:37] + "..."
    if case["hrr_rank"] is None and case["lexical_rank"] is None:
        outcome = "miss"
    elif case["hrr_rank"] is None:
        outcome = "loss"
    elif case["lexical_rank"] is None:
        outcome = "win"
    elif case["hrr_rank"] < case["lexical_rank"]:
        outcome = "win"
    elif case["hrr_rank"] > case["lexical_rank"]:
        outcome = "loss"
    else:
        outcome = "tie"
    return f"{query:<40}  {_fmt_rank(case['hrr_rank']):>4}  {_fmt_rank(case['lexical_rank']):>4}  {outcome:>5}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the HRR candidate re-ranker with a lexical baseline.")
    parser.add_argument("cases", help="JSON cases file with queries and expected fact ids")
    parser.add_argument("--db", default=str(DEFAULT_DB),
                        help="store to read (default: $FACTKIT_DB or ~/.hermes/memory_store.db)")
    parser.add_argument("--k", type=int, default=5, help="cutoff for hit@k (default: 5)")
    parser.add_argument("--json", action="store_true", help="print the raw result as JSON")
    args = parser.parse_args(argv)

    try:
        cases = load_cases(args.cases)
    except (OSError, ValueError) as exc:
        parser.error(f"cannot read cases file {args.cases!r}: {exc}")
    store = ReadOnlyStore(args.db)
    result = hrr.compare_with_provider(store, cases, k=args.k)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"{'query':<40}  {'hrr':>4}  {'lex':>4}  {'':>5}")
        print("-" * 59)
        for case in result["cases"]:
            print(_fmt_case(case))
        print("-" * 59)
        h, l = result["hrr"], result["lexical"]
        print(f"HRR      hit@{result['k']}={h['hit_at_k']:.3f}  MRR={h['mrr']:.3f}  n={h['n']}")
        print(f"Lexical  hit@{result['k']}={l['hit_at_k']:.3f}  MRR={l['mrr']:.3f}  n={l['n']}")
        print(f"W/L/T    {result['wins']}/{result['losses']}/{result['ties']}")

    h, l = result["hrr"], result["lexical"]
    beat = h["hit_at_k"] > l["hit_at_k"] and h["mrr"] > l["mrr"]
    if beat:
        print("VERDICT: HRR beats lexical on both hit@k and MRR — candidate for A/B, still not wired in.")
    else:
        print(f"VERDICT: HRR does NOT beat lexical on both hit@k and MRR "
              f"(hit@k {h['hit_at_k']:.3f} vs {l['hit_at_k']:.3f}, "
              f"MRR {h['mrr']:.3f} vs {l['mrr']:.3f}) — do not wire it in.")
    return 0 if beat else 1


if __name__ == "__main__":
    raise SystemExit(main())
