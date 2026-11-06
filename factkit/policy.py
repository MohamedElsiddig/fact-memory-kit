"""Which detected maintenance candidates may be applied without review?

Detection (:func:`factkit.holo.find_conflicts` and
:func:`factkit.holo.merge_duplicates`) is deliberately permissive: it reports
anything that *might* be a duplicate or a contradiction. This module is the
second gate. It answers one question per candidate:

``auto``
    Safe to apply unattended: either a near-identical duplicate or a conflict
    that satisfies every guard below.

``propose``
    A human must look at it. This is also the answer for anything the policy
    cannot fully verify, including malformed input.

``never``
    Must not be touched: a denied category, a cross-category pair, or a pair
    carrying a user-origin/verbatim marker. ``never`` is never relaxed by the
    master switch.

Pair shapes
-----------
:func:`~factkit.holo.find_conflicts` emits::

    {"keep", "retire", "keep_text", "retire_text", "keep_updated",
     "retire_updated", "shared_entities", "jaccard", "diffs": [{"slot", ...}]}

:func:`~factkit.holo.merge_duplicates` emits::

    {"keep", "retire", "ratio", "keep_text", "retire_text"}

Real conflict pairs carry no category or trust, so those guards cannot pass
on a raw pair; the caller may enrich a pair with the optional keys below
(flat, or inside a ``keep_fact`` / ``retire_fact`` mapping with the database
column names)::

    keep_category / retire_category
    keep_trust_score / retire_trust_score   (alias: keep_trust / retire_trust)
    keep_source / retire_source             ("user" marks user origin)
    keep_verbatim / retire_verbatim

``jaccard`` is intentionally *not* treated as a duplicate ratio: it is the
conflict detector's overlap score, and a genuine port contradiction can have
a Jaccard of 1.0 once digits are stripped from tokens.

Every function is total. Unknown or partial input returns ``"propose"`` rather
than raising. Pure stdlib.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal

__all__ = ["Policy", "classify", "plan"]

_MISSING = object()

# merge_duplicates emits 'ratio'; 'score' and 'similarity' cover other callers.
_SIMILARITY_KEYS = ("score", "similarity", "ratio")
_NEAR_DUPLICATE = 0.98

# Keys that may carry an operator identity inside a fact row.
_SOURCE_KEYS = ("source", "origin", "created_by", "author")
_USER_ORIGINS = frozenset({"user", "verbatim"})
_TRUTHY_TEXT = frozenset({"true", "yes", "1", "verbatim"})

_NESTED_HOLDERS = ("keep_fact", "retire_fact", "keep_meta", "retire_meta",
                   "keep_row", "retire_row")


@dataclass(frozen=True)
class Policy:
    """Limits for unattended maintenance.

    ``auto``
        Master switch. When false, every ``auto`` decision becomes ``propose``;
        ``never`` is unaffected.
    ``max_apply`` / ``max_ratio``
        Batch budget used by :func:`plan`: at most ``max_apply`` autos, and at
        most ``max_ratio`` of the candidate batch (minimum one when the batch
        is non-empty).
    ``min_age_gap_days``
        A conflict's retire side must be at least this much older than the
        keep side before it may be applied automatically.
    ``allow_attrs``
        Conflict slots whose values are cheap to verify and safe to overwrite.
    ``deny_categories``
        Categories that may never be touched, whatever the similarity.
    ``cool_off_hours``
        Neither fact may have been updated this recently. Zero or negative
        disables the window.
    """

    auto: bool = False
    max_apply: int = 5
    max_ratio: float = 0.20
    min_age_gap_days: int = 7
    allow_attrs: frozenset = frozenset({"port", "ip", "version"})
    deny_categories: frozenset = frozenset()
    cool_off_hours: int = 24


# --------------------------------------------------------------------- values
def _number(value):
    """Smallest finite float for a real number, else None (bools excluded)."""
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, Decimal)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _text(value):
    """Trimmed lower-case text for a scalar, else '' (never raises)."""
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - pathological input
            return ""
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:  # noqa: BLE001 - pathological __str__
            return ""
    return value.strip().lower()


def _names(values):
    """Normalise a category/attribute collection to a set of lower-case names."""
    out = set()
    if values is None or isinstance(values, (str, bytes)):
        values = (values,) if values else ()
    try:
        items = list(values)
    except TypeError:
        return out
    for item in items:
        name = _text(item)
        if name:
            out.add(name)
    return out


def _parse_dt(value):
    """UTC datetime from a timestamp-like value, else None."""
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime(value.year, value.month, value.day)
    elif isinstance(value, bool) or value is None:
        return None
    elif isinstance(value, (int, float, Decimal)):
        stamp = float(value)
        if not math.isfinite(stamp):
            return None
        if abs(stamp) > 1e11:      # milliseconds, as some stores emit
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


def _nested(pair):
    """The per-side fact mappings present in a pair."""
    for holder in _NESTED_HOLDERS:
        row = pair.get(holder)
        if isinstance(row, dict):
            yield row


def _side_value(pair, side, names):
    """Flat ``<side>_<name>`` first, then a nested fact row; MISSING if absent.

    A holder named for the other side is only trusted when its ``fact_id``
    proves it belongs to this side, so a mixed-up mapping cannot lend one
    fact's trust or timestamps to the other.
    """
    for name in names:
        key = f"{side}_{name}"
        if pair.get(key) is not None:
            return pair[key]
    preferred = (f"{side}_fact", f"{side}_meta", f"{side}_row")
    others = tuple(h for h in _NESTED_HOLDERS if h not in preferred)
    for holder in preferred + others:
        row = pair.get(holder)
        if not isinstance(row, dict):
            continue
        if holder not in preferred:
            holder_id = row.get("fact_id")
            if holder_id is None or pair.get(side) is None:
                continue
            if _text(holder_id) != _text(pair.get(side)):
                continue
        for name in names:
            if row.get(name) is not None:
                return row[name]
    return _MISSING


def _category(pair, side):
    value = _side_value(pair, side, ("category",))
    if value is _MISSING:
        return None
    return _text(value) or None


def _updated(pair, side):
    value = _side_value(pair, side, ("updated_at", "updated"))
    if value is _MISSING:
        return None
    return _parse_dt(value)


def _trust(pair, side):
    value = _side_value(pair, side, ("trust_score", "trust"))
    if value is _MISSING:
        return None
    return _number(value)


def _ids(pair):
    """(keep, retire) when both are present and distinct, else None."""
    keep, retire = pair.get("keep", _MISSING), pair.get("retire", _MISSING)
    if keep is _MISSING or retire is _MISSING:
        return None
    if keep is None or retire is None:
        return None
    try:
        if keep == retire:
            return None
    except Exception:  # noqa: BLE001 - exotic __eq__
        return None
    return keep, retire


# -------------------------------------------------------------------- markers
def _origin_marker(value):
    return _text(value) in _USER_ORIGINS


def _verbatim(value):
    if value is True:
        return True
    if isinstance(value, str):
        return _text(value) in _TRUTHY_TEXT
    return False


def _user_origin(pair):
    """True when any side carries a user/verbatim origin marker."""
    for key in _SOURCE_KEYS:
        if _origin_marker(pair.get(key)):
            return True
    for key in ("verbatim", "user_verbatim"):
        if _verbatim(pair.get(key)):
            return True
    for side in ("keep", "retire"):
        for key in _SOURCE_KEYS:
            if _origin_marker(pair.get(f"{side}_{key}")):
                return True
        if _verbatim(pair.get(f"{side}_verbatim")):
            return True
    for row in _nested(pair):
        for key in _SOURCE_KEYS:
            if _origin_marker(row.get(key)):
                return True
        if _verbatim(row.get("verbatim")):
            return True
    return False


# ------------------------------------------------------------------ similarity
def _similarity(pair):
    for key in _SIMILARITY_KEYS:
        if key in pair:
            score = _number(pair[key])
            if score is not None:
                return score
    return None


def _is_near_duplicate(pair):
    if not isinstance(pair, dict) or _ids(pair) is None:
        return False
    score = _similarity(pair)
    return score is not None and score >= _NEAR_DUPLICATE


# ------------------------------------------------------------------ conflict
def _shared_entity(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(isinstance(item, str) and item.strip() for item in value)
    return False


def _slot(pair):
    """The first differing attribute, from diffs[0]['slot'] or a flat key."""
    diffs = pair.get("diffs")
    first = None
    if isinstance(diffs, dict):
        first = diffs
    elif isinstance(diffs, (list, tuple)) and diffs:
        first = diffs[0]
    if isinstance(first, dict):
        for key in ("slot", "attribute", "attr", "name"):
            name = _text(first.get(key))
            if name:
                return name
    for key in ("slot", "attribute", "attr"):
        name = _text(pair.get(key))
        if name:
            return name
    return None


def _conflict_is_auto(pair, policy, now):
    """Every guard for an unattended conflict: entity, slot, age, trust, quiet."""
    if _ids(pair) is None:
        return False
    if not _shared_entity(pair.get("shared_entities")):
        return False

    slot = _slot(pair)
    if not slot or slot not in _names(policy.allow_attrs):
        return False

    keep_dt, retire_dt = _updated(pair, "keep"), _updated(pair, "retire")
    if keep_dt is None or retire_dt is None:
        return False
    # The retire side must actually be the older fact; a pair that would drop
    # newer knowledge is inconsistent and needs review.
    if retire_dt > keep_dt:
        return False

    try:
        min_gap = float(policy.min_age_gap_days)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(min_gap):
        return False
    gap_days = (keep_dt - retire_dt).total_seconds() / 86400.0
    if gap_days < min_gap:
        return False

    keep_trust, retire_trust = _trust(pair, "keep"), _trust(pair, "retire")
    if keep_trust is None or retire_trust is None:
        return False
    if retire_trust > keep_trust:
        return False

    try:
        cool_hours = float(policy.cool_off_hours)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(cool_hours):
        return False
    if cool_hours > 0:
        window = cool_hours * 3600.0
        if (now - keep_dt).total_seconds() < window:
            return False
        if (now - retire_dt).total_seconds() < window:
            return False
    return True


# ---------------------------------------------------------------------- API
def _classify(pair, now, policy):
    if not isinstance(pair, dict):
        return "propose"
    policy = policy if isinstance(policy, Policy) else Policy()

    deny = _names(policy.deny_categories)
    categories = [_category(pair, "keep"), _category(pair, "retire")]
    if any(cat in deny for cat in categories if cat):
        return "never"
    if categories[0] and categories[1] and categories[0] != categories[1]:
        return "never"
    if _user_origin(pair):
        return "never"

    decision = "propose"
    if _is_near_duplicate(pair):
        decision = "auto"
    elif _conflict_is_auto(pair, policy, now):
        decision = "auto"

    if decision == "auto" and not policy.auto:
        return "propose"
    return decision


def classify(pair: dict, *, now=None, policy: Policy = Policy()) -> str:
    """Return ``'auto'``, ``'propose'`` or ``'never'`` for one candidate.

    ``now`` defaults to the current UTC time and may be a datetime, date,
    ISO-8601 string or epoch number. Malformed pairs, missing ids, missing
    timestamps and missing trust all degrade to ``'propose'``.
    """
    try:
        if now is None:
            moment = datetime.now(timezone.utc)
        else:
            moment = _parse_dt(now)
            if moment is None:
                return "propose"
        return _classify(pair, moment, policy)
    except Exception:  # noqa: BLE001 - policy decisions must never crash a scan
        return "propose"


def plan(pairs, *, now=None, policy: Policy = Policy()) -> list[str]:
    """Classify a batch and enforce ``max_apply``/``max_ratio``.

    Near-duplicates consume the budget first because merging them is the
    least destructive action; remaining autos are demoted to ``propose`` in
    input order. The return value is one decision per input pair.
    """
    try:
        items = list(pairs)
    except TypeError:
        return []
    decisions = [classify(pair, now=now, policy=policy) for pair in items]
    if not isinstance(policy, Policy) or not policy.auto or not items:
        return decisions

    try:
        apply_cap = int(policy.max_apply)
    except (TypeError, ValueError, OverflowError):
        apply_cap = 0
    try:
        raw_ratio = float(policy.max_ratio)
        ratio_cap = math.floor(raw_ratio * len(items)) if math.isfinite(raw_ratio) else -1
    except (TypeError, ValueError, OverflowError):
        ratio_cap = -1
    if ratio_cap < 0:
        cap = 0                            # unusable ratio: apply nothing
    else:
        cap = max(0, min(apply_cap, max(1, ratio_cap)))

    order = sorted(
        range(len(items)),
        key=lambda i: (0 if decisions[i] == "auto" and _is_near_duplicate(items[i]) else 1, i),
    )
    allowed = set(order[:cap])
    return [
        "auto" if (decision == "auto" and i in allowed) else
        ("propose" if decision == "auto" else decision)
        for i, decision in enumerate(decisions)
    ]
