"""Holographic reduced representation (HRR) scoring — a *candidate* re-ranker.

This module is deliberately additive. It does not replace the provider, the
embedding channel, :mod:`factkit.vectors` or :func:`factkit.holo.fuse_results`,
and nothing imports it from the live recall hook. It exists so the idea can be
measured against a lexical baseline before anyone wires it in.

How it works
------------
Each content token is hashed to one of ``VOCAB_CAP`` indices (so an unbounded
vocabulary never allocates an unbounded table), and that index seeds a
deterministic RNG that draws a unit vector of ``dim`` floats. A token at
position *p* is bound to its position by rolling that vector *p* slots. The
bound vectors are superposed (added) and the sum is L2-normalised, giving one
dense vector per fact. Queries are encoded by exactly the same function.

Because superposition is commutative, a bag-of-tokens similarity survives the
position binding, while the rolls make different orders produce different
vectors. A query that shares no indexed token with a fact scores exactly
``0.0``: random collisions would otherwise leak a small nonzero cosine, and a
token-composition model has no evidence for a fact it shares no token with.

Status: candidate, not replacement
----------------------------------
HRR is a cheap, local, deterministic signal. It is *not* a semantic embedder:
paraphrases with no shared tokens score zero, and it cannot replace the
provider's own ranking. Treat it as a re-ranking feature. Wiring it into the
live hook (``holo.fuse_results`` / ``holo.prefetch``) requires A/B numbers from
:func:`compare_with_provider` on a representative cases file first — in
particular, HRR must beat the lexical baseline on both hit@k and MRR before it
earns a place in the fused path. Until then this stays off the hook.

All functions here are read-only with respect to a store. ``compare_with_provider``
only issues ``SELECT`` queries.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "VOCAB_CAP",
    "CACHE_MAX",
    "tokens",
    "encode",
    "similarity",
    "rank",
    "compare_with_provider",
]

DIM = 1024
VOCAB_CAP = 4096
CACHE_MAX = 512

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words carry no retrieval signal and only add superposition noise.
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "his", "how", "i", "if", "in", "into", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "she", "should", "so", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "those",
    "to", "was", "we", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "would", "you", "your",
}


def tokens(text: Any) -> list[str]:
    """Lower-cased alphanumeric tokens, stop words and 1-char noise removed."""
    return [t for t in _TOKEN_RE.findall(str(text or "").lower())
            if len(t) > 1 and t not in STOP]


# --------------------------------------------------------------- token space
# Vectors are generated on demand and kept in a small LRU. Pre-building a
# VOCAB_CAP x dim table would cost hundreds of MB on a small machine, and most
# corpora only ever touch a fraction of the buckets.
_VECTOR_CACHE: "OrderedDict[tuple[int, int, int], tuple[float, ...]]" = OrderedDict()


def _token_index(token: str, seed: int) -> int:
    """Hash a token into the bounded vocabulary."""
    digest = hashlib.blake2b(f"{seed}:{token}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % VOCAB_CAP


def _unit(index: int, dim: int, seed: int) -> tuple[float, ...]:
    """Deterministic random unit vector for one vocabulary index."""
    key = (int(seed), int(dim), int(index))
    cached = _VECTOR_CACHE.get(key)
    if cached is not None:
        _VECTOR_CACHE.move_to_end(key)
        return cached

    rng = random.Random((int(seed) * 1_000_003) ^ (int(index) << 20) ^ int(dim))
    values = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:  # pragma: no cover - astronomically unlikely
        values[0] = 1.0
        norm = 1.0
    vec = tuple(v / norm for v in values)
    _VECTOR_CACHE[key] = vec
    while len(_VECTOR_CACHE) > CACHE_MAX:
        _VECTOR_CACHE.popitem(last=False)
    return vec


def _normalise(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in vec))
    if norm == 0.0:
        return [0.0] * len(vec)
    return [float(v) / norm for v in vec]


def _encode_text(text: Any, *, dim: int, seed: int) -> list[float]:
    """Superpose position-bound token vectors, then L2-normalise."""
    acc = [0.0] * dim
    for pos, token in enumerate(tokens(text)):
        vec = _unit(_token_index(token, seed), dim, seed)
        shift = pos % dim
        # roll the token vector by its position: binding to "where it occurred"
        for j in range(dim):
            acc[j] += vec[j - shift]
    return _normalise(acc)


def encode(facts: Iterable[Mapping[str, Any]], *, dim: int = DIM, seed: int = 0) -> dict[int, list[float]]:
    """Encode facts into deterministic dense HRR vectors.

    ``facts`` items must expose ``fact_id`` (or ``id``) and ``content``. The
    result maps fact id -> L2-normalised ``dim``-float vector. Same input and
    seed produce byte-identical output; a different seed produces different
    vectors. Tokens are hashed into ``VOCAB_CAP`` buckets and vectors are
    generated lazily behind a bounded LRU, so a large or unseen vocabulary does
    not grow memory without bound.
    """
    if int(dim) <= 0:
        raise ValueError("dim must be positive")
    out: dict[int, list[float]] = {}
    for fact in facts:
        fact_id = fact.get("fact_id", fact.get("id"))
        if fact_id is None:
            raise ValueError("fact is missing fact_id")
        out[int(fact_id)] = _encode_text(fact.get("content", ""), dim=int(dim), seed=int(seed))
    return out


# ------------------------------------------------------------------ scoring
def similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 when either vector has zero norm or lengths differ."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _content(fact: Any) -> str:
    if isinstance(fact, Mapping):
        return str(fact.get("content") or "")
    return str(fact or "")


def rank(query: str, encoded: Mapping[int, Sequence[float]], facts_by_id: Mapping[int, Any],
         *, k: int = 5) -> list[dict]:
    """Score every encoded fact against ``query`` and return the top ``k``.

    Scores are HRR cosine values in ``[-1, 1]``; a fact that shares no token
    with the query scores exactly ``0.0``. Ties break on ascending fact id so
    the result is deterministic. ``k`` is capped to the number of candidates;
    ``k <= 0`` returns an empty list.

    ``encoded`` is expected to come from :func:`encode` with the default seed
    (0); a query encoded under a different seed would not share token vectors
    with the facts.
    """
    limit = max(0, int(k))
    if not encoded or limit == 0:
        return []

    qvec = _encode_text(query, dim=_dim_of(encoded), seed=0)
    q_tokens = set(tokens(query))
    scored: list[dict] = []
    for raw_id, vec in encoded.items():
        fact_id = int(raw_id)
        fact = facts_by_id.get(fact_id)
        if fact is None and raw_id in facts_by_id:  # non-int-keyed mapping
            fact = facts_by_id[raw_id]  # type: ignore[index]
        score = 0.0
        if q_tokens and (set(tokens(_content(fact))) & q_tokens):
            score = similarity(qvec, vec)
        scored.append({"fact_id": fact_id, "score": round(score, 6)})

    scored.sort(key=lambda row: (-row["score"], row["fact_id"]))
    return scored[:limit]


def _dim_of(encoded: Mapping[int, Sequence[float]]) -> int:
    for vec in encoded.values():
        return len(vec)
    return DIM


# --------------------------------------------------------------- comparison
def _read_facts(store: Any) -> list[dict]:
    """Read ``(fact_id, content)`` from any supported store, without writing."""
    rows: list[dict] | None = None
    if hasattr(store, "read"):
        try:
            rows = [dict(r) for r in store.read(
                "SELECT fact_id, content FROM facts ORDER BY fact_id")]
        except Exception:  # noqa: BLE001 - fall through to the other shapes
            rows = None
    if rows is None and hasattr(store, "all_facts"):
        rows = [dict(r) for r in store.all_facts()]
    if rows is None and hasattr(store, "list_facts"):
        rows = [dict(r) for r in store.list_facts(limit=1_000_000)]
    if rows is None:
        path = str(Path(str(store)).expanduser())
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            rows = [dict(r) for r in conn.execute(
                "SELECT fact_id, content FROM facts ORDER BY fact_id")]
        finally:
            conn.close()
    return [{"fact_id": int(r["fact_id"]), "content": str(r.get("content") or "")}
            for r in rows]


def _lexical_scores(query_tokens: set[str], facts: Sequence[dict]) -> list[dict]:
    """Provider-style baseline: plain token overlap, no embeddings, no vectors."""
    if not query_tokens:
        return [{"fact_id": f["fact_id"], "score": 0.0} for f in facts]
    out = []
    for fact in facts:
        overlap = query_tokens & set(tokens(fact["content"]))
        out.append({"fact_id": fact["fact_id"],
                    "score": len(overlap) / len(query_tokens)})
    return out


def _order(scores: Sequence[dict]) -> list[int]:
    return [row["fact_id"] for row in
            sorted(scores, key=lambda row: (-row["score"], row["fact_id"]))]


def _rank_of(order: Sequence[int], target: int | None) -> int | None:
    if target is None:
        return None
    try:
        return list(order).index(target) + 1
    except ValueError:
        return None


def compare_with_provider(store: Any, cases: Iterable[Mapping[str, Any]], *, k: int = 5) -> dict:
    """Compare HRR ranking against a lexical baseline over a store's facts.

    ``cases`` is an iterable of ``{"query": str, "expect": [fact_id, ...]}``.
    For each case the *first* expected id is the target. Per-case output holds
    its rank (1-based, ``None`` when absent) under both rankers, plus the top
    ``k`` ids. Aggregates are::

        {"hrr": {"hit_at_k", "mrr", "n"},
         "lexical": {...}, "wins", "losses", "ties", "k", "cases"}

    ``hit_at_k`` is the fraction of cases whose target landed in the top ``k``;
    ``mrr`` is the mean reciprocal rank over the full ranking (0 for a miss).
    ``wins`` / ``losses`` / ``ties`` count pairwise rank improvements over the
    lexical baseline (a miss ranks below every hit). Cases without ``expect``
    are reported but excluded from aggregates.

    Read-only: only ``SELECT`` queries are issued, so a caller's store and its
    mtime are untouched.
    """
    limit = max(0, int(k))
    facts = _read_facts(store)
    by_id = {f["fact_id"]: f for f in facts}
    encoded = encode(facts) if facts else {}

    results: list[dict] = []
    hrr_ranks: list[int | None] = []
    lex_ranks: list[int | None] = []
    wins = losses = ties = 0

    for case in cases:
        query = str(case.get("query") or "")
        expect = [int(x) for x in (case.get("expect") or [])]
        target = expect[0] if expect else None

        hrr_order = _order(rank(query, encoded, by_id, k=len(facts))) if facts else []
        lex_order = _order(_lexical_scores(set(tokens(query)), facts)) if facts else []

        hrr_rank = _rank_of(hrr_order, target)
        lex_rank = _rank_of(lex_order, target)
        results.append({
            "query": query,
            "expect": expect,
            "target": target,
            "hrr_rank": hrr_rank,
            "lexical_rank": lex_rank,
            "hrr_top": hrr_order[:limit],
            "lexical_top": lex_order[:limit],
        })

        if target is None:
            continue
        hrr_ranks.append(hrr_rank)
        lex_ranks.append(lex_rank)

        # a miss ranks after every hit
        hrr_key = hrr_rank if hrr_rank is not None else math.inf
        lex_key = lex_rank if lex_rank is not None else math.inf
        if hrr_key < lex_key:
            wins += 1
        elif hrr_key > lex_key:
            losses += 1
        else:
            ties += 1

    def aggregate(ranks: Sequence[int | None]) -> dict:
        n = len(ranks)
        if not n:
            return {"hit_at_k": 0.0, "mrr": 0.0, "n": 0}
        hits = sum(1 for r in ranks if r is not None and r <= limit)
        reciprocal = sum(1.0 / r for r in ranks if r is not None)
        return {"hit_at_k": round(hits / n, 6),
                "mrr": round(reciprocal / n, 6),
                "n": n}

    return {
        "k": limit,
        "cases": results,
        "hrr": aggregate(hrr_ranks),
        "lexical": aggregate(lex_ranks),
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }
