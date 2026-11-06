"""Pre-insert write-path guard: check a candidate fact against the live store.

The maintenance workflow finds duplicates and contradictions *after* they are
stored (``holo.find_conflicts`` / ``holo.merge_duplicates``).  This module is
the same inspection aimed one step earlier: given the text an agent is about to
store, decide whether it should be added at all.

The guard is read-only and pure with respect to the store.  It reads facts and
their entity links, never writes, and returns a plain dict::

    {"verdict": "ok" | "duplicate" | "conflict" | "near",
     "matches": [{"fact_id", "content", "reason", "similarity", "diffs"}],
     "suggestion": str | None}

Detection reuses the maintenance detectors rather than inventing new rules:

* duplicates use ``difflib.SequenceMatcher`` on case-folded, whitespace- and
  punctuation-normalised text -- the same measure ``holo.merge_duplicates``
  computes -- plus its numeric-identity rule (two facts whose digit sets differ
  are different claims, however similar the prose).  ``merge_duplicates`` is a
  store-wide O(n^2) scan with no candidate parameter, so the guard applies the
  same primitive to one candidate instead of calling it on every write.
* conflicts reuse ``holo._value_conflict`` / ``holo._same`` and the entity /
  token-overlap gate from ``holo.find_conflicts``, with values normalised by
  ``factkit.units``.  Ports compare by identity, except that a single port
  inside a declared range is agreement rather than a contradiction.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from . import holo, units

__all__ = ["check", "format_report", "normalize", "DUPLICATE_THRESHOLD", "NEAR_THRESHOLD"]

# 0.98 is the policy's auto-merge class and the bar for refusing a write as a
# duplicate; merge_duplicates' 0.92 default would call too much a duplicate
# here.  Anything in the 0.90-0.98 band is worth merging wording, not adding.
DUPLICATE_THRESHOLD = 0.98
NEAR_THRESHOLD = 0.90

# Same gates as holo.find_conflicts' defaults: a conflict needs textual overlap
# by itself, or a shared declared entity plus a little overlap.
OVERLAP_MIN = 0.35
ENTITY_MIN = 0.15

_TRAILING_PUNCT = re.compile(r"[\s.,;:!?]+$")
_SPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


# --------------------------------------------------------------------- text
def normalize(text: str) -> str:
    """Case-fold, collapse whitespace and strip trailing punctuation.

    This is the comparison form for duplicate detection.  It is deliberately
    conservative: it never rewrites words, digits or unit spellings, so a
    candidate that changed a number still has a different digit set.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    folded = _SPACE.sub(" ", text.casefold()).strip()
    return _TRAILING_PUNCT.sub("", folded)


def _numbers(text: str) -> set[str]:
    """The digit set of a text, as holo.merge_duplicates computes it."""
    return set(_NUMBER.findall(text or ""))


def _similarity(a: str, b: str) -> float:
    """Ratio of two *already normalised* strings (holo's measure)."""
    return SequenceMatcher(None, a, b).ratio()


# -------------------------------------------------------------------- store
def _rows(store, sql: str, params=()) -> list[dict]:
    """Rows as dicts, tolerating providers that return plain tuples."""
    out = []
    for row in store.read(sql, params):
        try:
            out.append(dict(row))
        except (TypeError, ValueError):
            out.append({"fact_id": row[0], "content": row[1]})
    return out


def _facts(store) -> list[dict]:
    """Live facts with the columns the guard needs.  Read-only."""
    try:
        rows = _rows(store, "SELECT fact_id, content, category, trust_score "
                            "FROM facts ORDER BY fact_id")
    except Exception:  # noqa: BLE001 - an empty/uninitialised store has no facts
        return []
    return [r for r in rows if r.get("content")]


def _entity_names(store) -> dict[int, set[str]]:
    """Entity names per fact for the "same declared subject" gate.

    Declared names are authoritative, as in ``holo.find_conflicts``.  When a
    fact carries no declaration (facts written without ``entities=`` still get
    auto-extracted links), fall back to the provider's *multi-word* names.  The
    provider itself treats those as safe to match in free text, and the
    candidate must still mention the name verbatim, so a port on a genuinely
    unrelated service is not dragged into a conflict.
    """
    queries = (
        ("SELECT fe.fact_id AS fact_id, e.name AS name, e.entity_type AS entity_type "
         "FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id"),
        ("SELECT fe.fact_id AS fact_id, e.name AS name, NULL AS entity_type "
         "FROM fact_entities fe JOIN entities e ON e.entity_id = fe.entity_id"),
    )
    for sql in queries:
        try:
            rows = _rows(store, sql)
        except Exception:  # noqa: BLE001 - missing table/column: try the next shape
            continue
        declared: dict[int, set[str]] = {}
        extracted: dict[int, set[str]] = {}
        for row in rows:
            try:
                fid = int(row["fact_id"])
            except (KeyError, TypeError, ValueError):
                continue
            name = str(row.get("name") or "").strip().casefold()
            if not name:
                continue
            if str(row.get("entity_type") or "") == "declared":
                declared.setdefault(fid, set()).add(name)
            elif " " in name:
                extracted.setdefault(fid, set()).add(name)
        return {fid: declared.get(fid) or names for fid, names in
                {**extracted, **declared}.items()}
    return {}


def _mentions(text: str, name: str) -> bool:
    """True when *name* occurs in *text* as a whole phrase (word boundaries)."""
    if not name:
        return False
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text) is not None


# ---------------------------------------------------------------- conflicts
def _port_bounds(value: tuple):
    if not value or value[0] not in ("single", "range"):
        return None
    if value[0] == "single":
        return value[1], value[1]
    return value[1], value[2]


def _port_compatible(a, b) -> bool:
    """Identity, plus the one benign case: a single port inside a range.

    ``units.conflicts`` treats overlapping-but-different port specs as a
    contradiction, which is right for two ranges but wrong for "8080" against
    a declared 8000-8100 range: the single port is already covered.
    """
    ca, cb = units.canonical("port", str(a)), units.canonical("port", str(b))
    if ca == cb:
        return True
    ba, bb = _port_bounds(ca), _port_bounds(cb)
    if ba is None or bb is None:
        return False
    if ca[0] == "single" and cb[0] == "range":
        return bb[0] <= ca[1] <= bb[1]
    if ca[0] == "range" and cb[0] == "single":
        return ba[0] <= cb[1] <= ba[1]
    return False


def _same(slot: str, a, b) -> bool:
    """Value equality for one slot; ``units`` normalises capacities/clocks."""
    if slot == "port":
        return _port_compatible(a, b)
    return holo._same(slot, a, b)


def _differs(slot: str, left: set, right: set) -> bool:
    """True when either side holds a value the other side cannot match."""
    left_ok = all(any(_same(slot, v, w) for w in right) for v in left)
    right_ok = all(any(_same(slot, w, v) for v in left) for w in right)
    return not (left_ok and right_ok)


def _conflicts(candidate: dict, existing: dict, shared: set[str]) -> list[dict]:
    """Attribute diffs for one pair, reusing holo's detector and gate."""
    left_tokens = holo._tokens(candidate["content"])
    right_tokens = holo._tokens(existing["content"])
    union = left_tokens | right_tokens
    overlap = len(left_tokens & right_tokens) / len(union) if union else 0.0
    if not (overlap >= OVERLAP_MIN or (shared and overlap >= ENTITY_MIN)):
        return []

    diffs = []
    for slot, cand_values, existing_values in holo._value_conflict(
            candidate, existing, shared):
        if not _differs(slot, cand_values, existing_values):
            continue
        diffs.append({
            "slot": slot,
            "candidate": sorted(str(v) for v in cand_values),
            "existing": sorted(str(v) for v in existing_values),
        })
    return diffs


# --------------------------------------------------------------------- API
def _trust_hint(candidate_trust, existing_trust) -> str:
    if candidate_trust is None or existing_trust is None:
        return ""
    try:
        if float(candidate_trust) > float(existing_trust) + 0.05:
            return (" The candidate carries the higher trust score; update or "
                    "supersede the existing fact after review instead of adding "
                    "a second one.")
    except (TypeError, ValueError):
        return ""
    return ""


def check(store, content: str, category: str | None = None,
          trust: float | None = None) -> dict:
    """Inspect *content* against the store before it is inserted.

    ``category`` and ``trust`` are advisory: they shape the suggestion text,
    never the verdict.  Matching is intentionally category-blind, because the
    same fact can be filed under a different category and still be a duplicate.

    The store is only read.  Every fact write remains the caller's job, through
    ``add_fact_status`` / ``update_fact`` after this returns.
    """
    normalized = normalize(content)
    result: dict = {"verdict": "ok", "matches": [], "suggestion": None}
    if not normalized:
        return result

    facts = _facts(store)
    if not facts:
        return result

    numbers = _numbers(normalized)
    entity_names = _entity_names(store)
    candidate = {"fact_id": None, "content": content}

    duplicates: list[dict] = []
    conflicts: list[dict] = []
    near: list[dict] = []

    for fact in facts:
        try:
            fact_id = int(fact["fact_id"])
        except (KeyError, TypeError, ValueError):
            continue
        existing_text = str(fact.get("content") or "")
        if not existing_text:
            continue
        existing_norm = normalize(existing_text)
        if not existing_norm:
            continue

        ratio = _similarity(normalized, existing_norm)
        same_numbers = numbers == _numbers(existing_norm)
        match = {
            "fact_id": fact_id,
            "content": existing_text,
            "reason": "",
            "similarity": round(ratio, 3),
            "diffs": [],
        }

        if same_numbers and ratio >= DUPLICATE_THRESHOLD:
            match["reason"] = "duplicate"
            duplicates.append(match)
            continue

        shared = {name for name in entity_names.get(fact_id, ()) if _mentions(normalized, name)}
        diffs = _conflicts(candidate, fact, shared)
        if diffs:
            match["reason"] = "conflict"
            match["diffs"] = diffs
            conflicts.append(match)
            continue

        # A different digit set is a different claim, so it is not a merge
        # candidate either: merging "filler fact 1" into "filler fact 2" would
        # destroy one of them.
        if same_numbers and ratio >= NEAR_THRESHOLD:
            match["reason"] = "near"
            near.append(match)

    matches = duplicates + conflicts + near
    matches.sort(key=lambda m: (-m["similarity"], m["fact_id"]))
    result["matches"] = matches

    if duplicates:
        best = max(duplicates, key=lambda m: m["similarity"])
        result["verdict"] = "duplicate"
        kind = f" {category}" if category else ""
        result["suggestion"] = (
            f"Fact #{best['fact_id']} already states this (similarity "
            f"{best['similarity']:.2f}); update that{kind} fact instead of "
            f"adding a duplicate." + _trust_hint(trust, _trust_of(facts, best["fact_id"]))
        )
    elif conflicts:
        best = max(conflicts, key=lambda m: m["similarity"])
        slots = ", ".join(sorted({d["slot"] for d in best["diffs"]}))
        result["verdict"] = "conflict"
        result["suggestion"] = (
            f"Fact #{best['fact_id']} disagrees on {slots}; update it or "
            f"supersede it after review instead of storing both values."
            + _trust_hint(trust, _trust_of(facts, best["fact_id"]))
        )
    elif near:
        best = max(near, key=lambda m: m["similarity"])
        result["verdict"] = "near"
        result["suggestion"] = (
            f"Fact #{best['fact_id']} is already {best['similarity']:.2f} similar; "
            f"merge the wording into it rather than adding a near-duplicate."
        )
    return result


def _trust_of(facts: list[dict], fact_id: int):
    for fact in facts:
        if fact.get("fact_id") == fact_id:
            return fact.get("trust_score")
    return None


def format_report(result: dict) -> str:
    """One line per match, ready for a CLI (``ok`` when nothing matched)."""
    matches = result.get("matches") or []
    if not matches:
        return "OK  no matching facts"
    lines = []
    for match in matches:
        reason = str(match.get("reason") or "match").upper()
        similarity = float(match.get("similarity") or 0.0)
        line = f"{reason:<9} #{match.get('fact_id')}  sim={similarity:.2f}"
        details = []
        for diff in match.get("diffs") or []:
            left = ", ".join(str(v) for v in diff.get("candidate") or ())
            right = ", ".join(str(v) for v in diff.get("existing") or ())
            details.append(f"{diff.get('slot')}: {left} vs {right}")
        if details:
            line += "  " + "; ".join(details)
        line += f"  | {match.get('content', '')}"
        lines.append(line)
    return "\n".join(lines)
