"""Age-based confidence decay — a suggestion layer, never a rewrite.

The provider ranks hits by ``facts.trust_score``. That number is earned (feedback,
corrections) and is provenance, so aging must not silently overwrite it: a
rewrite would be indistinguishable from a human downgrade and would erase the
history of what the store actually believed. This module therefore computes a
*separate* effective trust at scan time:

    effective = trust_score * decay_factor(age)

and exposes three read-only helpers (:func:`effective_trust`,
:func:`classify_fact`, :func:`scan`) plus one explicit write
(:func:`apply_touch`) that records "a human confirmed this is still true" by
bumping ``updated_at`` through :func:`factkit.holo.touch_fact`. Nothing here
edits ``trust_score``.

The curve
---------
Confidence halves every ``half_life_days`` (default 180) of age and stops at
``floor`` (default ``0.5``). The floor is deliberate: aging is weak evidence —
an old fact may still be exactly right — so decay may demote a fact below a
recall threshold but never erases it. Unknown or unparseable timestamps are
treated as age zero (factor 1.0): absence of evidence about age is not evidence
of staleness.

Categories
----------
``user_pref`` and ``project`` facts get *twice* the half-life (360 days).
A preference or a project decision describes the user's intent; it changes when
the user changes their mind, not when the environment drifts. Entity/system
facts (ports, versions, hosts, capacities) are claims about a world that moves
underneath them, so they keep the normal rate. The floor applies to both sets.

Bands
-----
Bands are relative to the caller's ``min_trust`` so the labels mean the same
thing at any recall threshold:

===================  ==========================================
``fresh``            ``effective >= 2.0 * min_trust``
``aging``            ``effective >= 1.5 * min_trust``
``stale``            ``effective >= min_trust``
``expired``          ``effective <  min_trust``
===================  ==========================================

At the default ``min_trust=0.25`` the edges are 0.50 / 0.375 / 0.25, so a fact
at default trust (0.5) is ``fresh`` on the day it is written and ``stale``
after exactly one half-life.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any

__all__ = [
    "BANDS",
    "DEFAULT_FLOOR",
    "DEFAULT_HALF_LIFE_DAYS",
    "DEFAULT_MIN_TRUST",
    "SLOW_CATEGORIES",
    "apply_touch",
    "classify_fact",
    "decay_factor",
    "effective_trust",
    "scan",
]

DEFAULT_HALF_LIFE_DAYS = 180.0
DEFAULT_FLOOR = 0.5
DEFAULT_MIN_TRUST = 0.25

# Categories whose truth is set by human intent rather than by the environment.
SLOW_CATEGORIES = frozenset({"user_pref", "project"})
SLOW_HALF_LIFE_MULTIPLIER = 2.0

BANDS = ("fresh", "aging", "stale", "expired")
FRESH_RATIO = 2.0
AGING_RATIO = 1.5

_MISSING = object()


# ------------------------------------------------------------------ coercion
def _finite(value):
    """A finite float from a real number (bool excluded), else None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value, low: float = 0.0, high: float = 1.0) -> float:
    number = _finite(value)
    if number is None:
        return low
    return min(high, max(low, number))


def _parse_dt(value):
    """UTC datetime from a timestamp-like value, else None. Never raises."""
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime(value.year, value.month, value.day)
    elif value is None or isinstance(value, bool):
        return None
    elif isinstance(value, (int, float)):
        stamp = _finite(value)
        if stamp is None:
            return None
        if abs(stamp) > 1e11:          # milliseconds, as some stores emit
            stamp /= 1000.0
        try:
            moment = datetime.fromtimestamp(stamp, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.endswith(("Z", "z")):
            raw = raw[:-1] + "+00:00"
        try:
            moment = datetime.fromisoformat(raw)
        except ValueError:
            moment = None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"):
                try:
                    moment = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
            if moment is None:
                return None
    else:
        return None

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    try:
        return moment.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def _now(now=None) -> datetime:
    """The reference instant; malformed input falls back to the clock."""
    return _parse_dt(now) or datetime.now(timezone.utc)


def _field(fact, name, default=None):
    if not isinstance(fact, dict):
        return default
    value = fact.get(name, _MISSING)
    return default if value is _MISSING else value


def _trust_value(fact) -> float:
    """``trust_score`` (then ``trust``), clamped to [0, 1]; missing means 0."""
    number = _finite(_field(fact, "trust_score"))
    if number is None:
        number = _finite(_field(fact, "trust", 0.0))
    return 0.0 if number is None else min(1.0, max(0.0, number))


def _category_name(fact) -> str:
    value = _field(fact, "category", "")
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _category_multiplier(fact) -> float:
    return SLOW_HALF_LIFE_MULTIPLIER if _category_name(fact) in SLOW_CATEGORIES else 1.0


def _age_days(fact, now=None) -> float:
    """Age from ``updated_at`` then ``created_at``; unknown/malformed is 0.0."""
    moment = _now(now)
    for key in ("updated_at", "created_at"):
        stamp = _parse_dt(_field(fact, key))
        if stamp is not None:
            return max(0.0, (moment - stamp).total_seconds() / 86400.0)
    return 0.0


# --------------------------------------------------------------------- curve
def decay_factor(age_days: float, *, half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
                 floor: float = DEFAULT_FLOOR) -> float:
    """``max(floor, 0.5 ** (age_days / half_life_days))``.

    Age zero (or negative, i.e. a future timestamp) is factor 1.0, as is an
    unparseable age: not knowing how old a fact is must not decay it. A
    missing, non-finite or non-positive half-life degrades to the floor rather
    than raising (an infinite age saturates there too). ``floor`` is clamped to
    [0, 1]; a floor above 1.0 yields 1.0.
    """
    floor = _clamp(floor)
    if age_days is None or isinstance(age_days, bool):
        return 1.0
    try:
        age = float(age_days)
    except (TypeError, ValueError, OverflowError):
        return 1.0
    if math.isnan(age) or age <= 0:
        return 1.0
    if math.isinf(age):
        return floor
    half_life = _finite(half_life_days)
    if half_life is None or half_life <= 0:
        return floor
    try:
        factor = 0.5 ** (age / half_life)
    except OverflowError:  # absurd age/half-life ratio
        return floor
    if not math.isfinite(factor):
        return floor
    return min(1.0, max(floor, factor))


def effective_trust(fact: dict, now=None, **kw) -> float:
    """``trust_score * decay_factor(age)``, category-aware.

    Accepts ``half_life_days`` and ``floor`` through ``kw``. Timestamps may be
    ISO-8601 text, datetimes, dates or epoch seconds; ``updated_at`` wins over
    ``created_at``, and missing or malformed values are treated as age zero.
    """
    age = _age_days(fact, now)
    half_life = _finite(kw.get("half_life_days", DEFAULT_HALF_LIFE_DAYS))
    if half_life is None:
        half_life = DEFAULT_HALF_LIFE_DAYS
    factor = decay_factor(age,
                          half_life_days=half_life * _category_multiplier(fact),
                          floor=kw.get("floor", DEFAULT_FLOOR))
    return _trust_value(fact) * factor


def _band(effective: float, min_trust: float) -> str:
    if effective >= FRESH_RATIO * min_trust:
        return "fresh"
    if effective >= AGING_RATIO * min_trust:
        return "aging"
    if effective >= min_trust:
        return "stale"
    return "expired"


def classify_fact(fact, now=None, *, min_trust: float = DEFAULT_MIN_TRUST, **kw) -> dict:
    """Score one fact and place it in a band. Read-only and total."""
    age = _age_days(fact, now)
    half_life = _finite(kw.get("half_life_days", DEFAULT_HALF_LIFE_DAYS))
    if half_life is None:
        half_life = DEFAULT_HALF_LIFE_DAYS
    factor = decay_factor(age,
                          half_life_days=half_life * _category_multiplier(fact),
                          floor=kw.get("floor", DEFAULT_FLOOR))
    trust = _trust_value(fact)
    effective = trust * factor
    threshold = _finite(min_trust)
    threshold = DEFAULT_MIN_TRUST if threshold is None else min(1.0, max(0.0, threshold))
    return {
        "fact_id": _field(fact, "fact_id"),
        "trust_score": trust,
        "age_days": age,
        "factor": factor,
        "effective": effective,
        "band": _band(effective, threshold),
    }


# --------------------------------------------------------------------- store
_FACT_SQL = ("SELECT fact_id, content, category, trust_score, created_at, updated_at "
             "FROM facts")


def _read_facts(store) -> list[dict]:
    """Every fact row, read-only. Works with the plugin and standalone stores."""
    reader: Any = getattr(store, "read", None)
    if callable(reader):
        return [dict(row) for row in reader(_FACT_SQL)]
    lister: Any = getattr(store, "all_facts", None)
    if callable(lister):
        return [dict(row) for row in lister()]
    raise TypeError("scan() needs a store exposing read(sql) or all_facts()")


def scan(store, *, now=None, limit: int = 0, **kw) -> dict:
    """Classify every fact without writing anything.

    Returns ``{"scanned", "counts", "candidates"}``. ``counts`` has one entry
    per band; ``candidates`` lists every non-``fresh`` fact as
    ``{fact_id, content, effective, age_days}``, worst (lowest effective) first.
    ``limit > 0`` truncates the candidate list only; ``scanned`` and the counts
    always describe the whole store.
    """
    moment = _now(now)
    rows = _read_facts(store)
    counts = {band: 0 for band in BANDS}
    candidates: list[dict] = []
    for row in rows:
        result = classify_fact(row, now=moment, **kw)
        counts[result["band"]] += 1
        if result["band"] != "fresh":
            candidates.append({
                "fact_id": result["fact_id"],
                "content": row.get("content", ""),
                "effective": result["effective"],
                "age_days": result["age_days"],
            })

    candidates.sort(key=lambda item: (item["effective"], str(item["fact_id"])))
    try:
        cap = int(limit)
    except (TypeError, ValueError, OverflowError):
        cap = 0
    if cap > 0:
        candidates = candidates[:cap]
    return {"scanned": len(rows), "counts": counts, "candidates": candidates}


def apply_touch(store, fact_ids, *, reason: str = "decay-review") -> dict:
    """Record a human confirmation by bumping ``updated_at`` only.

    Delegates to :func:`factkit.holo.touch_fact` (which uses the provider's own
    ``update_fact(..., touch=True)`` and writes an audit row); the content,
    category, tags and ``trust_score`` are left exactly as they were. This is
    the only writer in this module and it is never called by :func:`scan`.
    The auxiliary audit table is created first when missing, as holo's other
    writers do; no fact row gains or loses a column.
    """
    from . import holo
    from .maintenance import ensure_tombstones

    ensure_tombstones(store)

    touched: list[int] = []
    failed: list[dict] = []
    seen: set = set()
    ids: Any = [fact_ids] if isinstance(fact_ids, int) and not isinstance(fact_ids, bool) else fact_ids
    for raw in ids or ():
        try:
            fact_id = int(raw)
        except (TypeError, ValueError, OverflowError):
            failed.append({"fact_id": raw, "reason": "not an integer id"})
            continue
        if fact_id in seen:
            continue
        seen.add(fact_id)
        try:
            result = holo.touch_fact(store, fact_id, dry_run=False, reason=reason)
        except Exception as exc:  # noqa: BLE001 - one bad id must not kill the batch
            failed.append({"fact_id": fact_id, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if result.get("ok"):
            touched.append(fact_id)
        else:
            failed.append({"fact_id": fact_id, "reason": "fact not found"})
    return {"ok": not failed, "touched": touched, "failed": failed, "reason": reason}
