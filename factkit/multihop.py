"""Multi-hop retrieval: follow ``factkit.relations`` edges out from search seeds.

Ordinary recall answers "what do I know about X" in one read.  Some questions are
about *connections* instead -- "what depends on X", "which service uses the
proxy", "what breaks if I move the proxy" -- and the answer sits one or two
edges away from anything the query text mentions.  :func:`expand_query` walks
those edges starting from the normal semantic seeds and folds the neighbours
back into the candidate set at a decayed score.

Cost
----
This is deliberately **not** part of every turn.  Each expanded node costs one
or two ``relations.neighbors`` reads (two when ``direction="both"``), so a
depth-2 walk is up to a few dozen extra queries against the store.  It is an
explicit path -- a CLI command or a tool call the agent chooses for a connection
question -- never the default recall path.  ``fanout`` caps children per node,
``budget`` caps the whole expansion, and ``depth`` stops the walk, so a dense
graph cannot turn one question into thousands of reads.

Scoring
-------
A hop scores ``0.6 ** depth`` relative to its seed; :func:`expand_query`
multiplies that by the seed's own prefetch score.  A direct hit therefore always
outranks its neighbour, and a two-hop fact has to be the only answer before it
displaces one-hop evidence.  Ordering is always ``score`` descending then
``fact_id`` ascending, so ties are stable across runs.

Degradation
-----------
``factkit.relations`` is imported inside the functions, never at module import
time.  When it is absent (older install, plugin not enabled) both entry points
return empty hops and ``degraded: True`` instead of raising: recall keeps
working with direct hits only.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping

logger = logging.getLogger(__name__)

__all__ = ["walk", "expand_query", "DECAY"]

#: Score multiplier per hop.  One hop keeps 60%, two hops 36%.
DECAY = 0.6

RELATIONS_MODULE = "factkit.relations"

# holo.prefetch tuning that a caller may forward through ``expand_query``; every
# other keyword belongs to ``walk``.
_PREFETCH_KWARGS = ("min_trust", "min_score", "timeout")


# --------------------------------------------------------------------- helpers
def _relations():
    """Import the relations module late, so its absence never breaks imports."""
    return importlib.import_module(f"{__package__}.relations")


def _seed_list(seed_ids) -> list[int]:
    """Normalise seeds to unique ints, preserving order.

    Accepts a bare id, or a sequence of ids, prefetch hits, or any mapping with
    a ``fact_id`` key -- callers usually already hold hits and should not have
    to project them first.
    """
    if seed_ids is None:
        return []
    if isinstance(seed_ids, Mapping) or isinstance(seed_ids, (int, str)):
        seed_ids = [seed_ids]
    out: list[int] = []
    for item in seed_ids:
        raw = item.get("fact_id") if isinstance(item, Mapping) else item
        if raw is None:
            continue
        try:
            fact_id = int(raw)
        except (TypeError, ValueError):
            continue
        if fact_id not in out:
            out.append(fact_id)
    return out


def _weight(row: Mapping) -> float:
    try:
        return float(row.get("weight") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _neighbours(mod, store, fact_id: int, *, rel, direction: str,
                fanout: int | None, skip=()) -> list[dict]:
    """One node's unvisited neighbours, merged across directions and ordered.

    A node reachable both ways is returned once, keeping the heavier edge.  A
    failure on one node is logged and skipped: a missing edge must not lose the
    rest of the answer.  ``skip`` (the visited set) is applied *before* the
    fanout cut, so edges back into the walked region never consume a slot that a
    new fact could use.
    """
    sides = ("out", "in") if direction == "both" else (direction,)
    limit = -1 if fanout is None else max(0, int(fanout))
    if limit == 0:
        return []

    best: dict[int, dict] = {}
    for side in sides:
        try:
            rows = mod.neighbors(store, fact_id, rel=rel, direction=side,
                                 limit=limit if limit > 0 else 50)
        except Exception as exc:  # noqa: BLE001 - one bad node must not abort the walk
            logger.warning("relations.neighbors failed for #%s: %s", fact_id, exc)
            continue
        for row in rows or ():
            row = dict(row)
            try:
                other = int(row["fact_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if other == fact_id or other in skip:
                continue
            current = best.get(other)
            if current is None or _weight(row) > _weight(current):
                best[other] = row

    ordered = sorted(best.values(), key=lambda r: (-_weight(r), int(r["fact_id"])))
    return ordered[:limit] if limit > 0 else ordered


# ------------------------------------------------------------------------ walk
def walk(store, seed_ids, *, depth: int = 2, rel: str | None = None,
         fanout: int = 5, budget: int | None = 40,
         direction: str = "both") -> dict:
    """Breadth-first expansion over the relation graph.

    ``seed_ids`` may be ids or prefetch hits.  ``depth`` is the number of edges
    followed (``depth=1`` returns direct neighbours only); ``fanout`` caps
    children per node; ``budget`` caps the total number of non-seed facts
    returned, so a dense graph degrades into "the strongest edges" rather than a
    stall.  ``direction="both"`` is the default because connection questions run
    in both directions: "what depends on X" reads the edges *into* X while
    "what does X use" reads the edges out of it.

    Returns ``{seeds, hops, visited, degraded}``.  Each hop is
    ``{fact_id, via, from_id, depth, content, score}`` where ``via`` is the
    relation label and ``score`` is ``0.6 ** depth`` relative to a seed score of
    1.0 (:func:`expand_query` re-bases it on the real seed score).  ``visited``
    is every id touched, in discovery order, seeds first; it is what makes the
    walk cycle-safe.
    """
    seeds = _seed_list(seed_ids)
    result = {"seeds": list(seeds), "hops": [], "visited": list(seeds),
              "degraded": False}

    levels = max(0, int(depth or 0))
    if not seeds or levels == 0 or budget == 0:
        return result

    if direction not in ("out", "in", "both"):
        raise ValueError(f"direction must be 'out', 'in' or 'both', got {direction!r}")

    try:
        mod = _relations()
        getattr(mod, "neighbors")
    except (ImportError, AttributeError):
        result["degraded"] = True
        return result

    cap = None if budget is None else max(0, int(budget))
    visited = set(seeds)
    frontier = list(seeds)

    for level in range(1, levels + 1):
        if not frontier or (cap is not None and len(result["hops"]) >= cap):
            break
        next_frontier: list[int] = []
        for from_id in frontier:
            for row in _neighbours(mod, store, from_id, rel=rel,
                                   direction=direction, fanout=fanout,
                                   skip=visited):
                if cap is not None and len(result["hops"]) >= cap:
                    break
                fact_id = int(row["fact_id"])
                if fact_id in visited:
                    continue
                visited.add(fact_id)
                result["hops"].append({
                    "fact_id": fact_id,
                    "via": row.get("rel"),
                    "from_id": from_id,
                    "depth": level,
                    "content": row.get("content") or "",
                    "score": round(DECAY ** level, 6),
                })
                next_frontier.append(fact_id)
            if cap is not None and len(result["hops"]) >= cap:
                break
        frontier = next_frontier

    result["visited"] = list(seeds) + [h["fact_id"] for h in result["hops"]]
    return result


# ---------------------------------------------------------------- query merge
def _root(fact_id: int, parents: Mapping[int, int]) -> int:
    """Follow ``from_id`` back to the seed a hop descends from."""
    seen: set[int] = set()
    current = fact_id
    while current in parents and current not in seen:
        seen.add(current)
        current = parents[current]
    return current


def expand_query(store, message: str, *, limit: int = 5, depth: int = 2,
                 seed_limit: int | None = None, **kwargs) -> dict:
    """Semantic seeds, one graph walk, one merged ranking.

    ``holo.prefetch`` supplies the seeds; :func:`walk` expands them; hop scores
    are re-based on the seed they came from (``0.6 ** depth * seed score``).
    Direct hits keep their prefetch score and win any duplicate, then everything
    is sorted by score descending and ``fact_id`` ascending and cut to ``limit``.

    ``kwargs`` go to :func:`walk` (``rel``, ``fanout``, ``budget``,
    ``direction``); the prefetch knobs ``min_trust``, ``min_score`` and
    ``timeout`` are routed to :func:`factkit.holo.prefetch` instead.

    Returns ``{for_query, direct, hops, hits, degraded}``.  ``direct`` is the
    prefetch list, ``hops`` the scored neighbours that were not already direct
    hits, and ``hits`` the merged, capped ranking.  ``degraded`` is true when the
    relations module is missing or the embedding endpoint is unavailable; in the
    latter case there are no seeds and everything is empty.
    """
    out = {"for_query": message, "direct": [], "hops": [], "hits": [],
           "degraded": False}
    if limit is not None and limit <= 0:
        return out

    from . import holo

    prefetch_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _PREFETCH_KWARGS}
    found = holo.prefetch(store, message,
                          limit=limit if seed_limit is None else seed_limit,
                          **prefetch_kwargs)
    if found is None:
        out["degraded"] = True
        return out

    direct: list[dict] = []
    seen: set[int] = set()
    for hit in found:
        row = dict(hit)
        fact_id = int(row["fact_id"])
        if fact_id in seen:
            continue
        seen.add(fact_id)
        row["source"] = "direct"
        row["depth"] = 0
        direct.append(row)

    walked = walk(store, [row["fact_id"] for row in direct], depth=depth, **kwargs)
    out["degraded"] = bool(walked.get("degraded"))

    seed_scores = {int(row["fact_id"]): float(row.get("score") or 0.0)
                   for row in direct}
    parents = {int(hop["fact_id"]): int(hop["from_id"])
               for hop in walked.get("hops", ())}

    hops: list[dict] = []
    for hop in walked.get("hops", ()):
        fact_id = int(hop["fact_id"])
        if fact_id in seen:            # direct evidence outranks the path to it
            continue
        root = _root(fact_id, parents)
        score = round(float(hop.get("score") or 0.0)
                      * seed_scores.get(root, 0.0), 6)
        hops.append({
            "fact_id": fact_id,
            "content": hop.get("content") or "",
            "score": score,
            "source": "hop",
            "depth": hop.get("depth"),
            "via": hop.get("via"),
            "from_id": hop.get("from_id"),
        })
    hops.sort(key=lambda row: (-row["score"], row["fact_id"]))

    merged = direct + hops
    merged.sort(key=lambda row: (-float(row.get("score") or 0.0), int(row["fact_id"])))

    out["direct"] = direct
    out["hops"] = hops
    out["hits"] = merged[:limit] if limit is not None else merged
    return out
