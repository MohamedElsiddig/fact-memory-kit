"""The golden rule: stale-fact detection, deterministic and explainable.

    same subject + same value slot + different value  ->  newer updated_at wins

Similarity alone cannot separate a contradiction from a duplicate (AUROC ~0.59
in published memory-conflict work), so this uses structure instead: entity set
equality (or a strict token-overlap bar when no entity is linked) plus slot
disagreement on IPs, ports, versions, dates, paths, model names and counts.

It only *proposes*. Applying is a reviewed step that writes an audit row.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

STOP = {
    "the", "and", "for", "with", "that", "this", "from", "not", "are", "was",
    "has", "have", "its", "but", "you", "your", "our", "all", "any", "can",
    "when", "then", "than", "into", "out", "use", "used", "using", "via",
    "only", "also", "must", "should", "always", "never", "user", "wants",
    "want", "fact", "there", "their", "which", "what", "how", "who", "will",
    "would", "may", "might", "one", "two", "new", "old", "same",
}

SLOTS = {
    "ip":      re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "port":    re.compile(r"(?:port\s*|:)(\d{2,5})\b"),
    "version": re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?)\b"),
    "date":    re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b"),
    "path":    re.compile(r"(~/[\w./-]+|/(?:opt|etc|var|srv|home|mnt|usr)/[\w./-]+)"),
    "model":   re.compile(r"\b([a-z0-9]+(?:-[a-z0-9]+){1,4})\b"),
    "count":   re.compile(r"\b(\d+)\s*(gb|tb|mb|ghz|mhz|core|cores|msg|messages|session|sessions)\b"),
    "host":    re.compile(r"\b([a-z0-9][a-z0-9-]*\.(?:local|lan|internal|home))\b", re.I),
}

OVERLAP_MIN = 0.25
NO_ENTITY_OVERLAP_MIN = 0.55


@dataclass
class Proposal:
    keep: int
    retire: int
    jaccard: float
    slots: list[str]
    delta: list[str]
    keep_text: str
    retire_text: str
    keep_age_d: float
    retire_age_d: float

    def as_dict(self) -> dict:
        return self.__dict__ | {"delta": list(self.delta), "slots": list(self.slots)}


def tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9_.\-/]+", (text or "").lower())
    return {t for t in raw if len(t) > 2 and t not in STOP}


def _value(name: str, match: re.Match) -> str:
    """Normalise one slot hit so equal values compare equal across phrasings."""
    if name == "count" and match.lastindex and match.lastindex >= 2:
        return f"{match.group(1)}{match.group(2).lower()}"
    if name == "port":
        digits = re.search(r"\d{2,5}", match.group(0))
        return digits.group(0) if digits else match.group(0)
    if name in ("version", "date", "model") and match.lastindex:
        return match.group(1).lower()
    return " ".join(match.group(0).lower().split())


def slots(text: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name, rx in SLOTS.items():
        values = {_value(name, m) for m in rx.finditer((text or "").lower())}
        if values:
            out[name] = values
    return out


def conflicting_slots(a: dict, b: dict) -> tuple[list[str], list[str]]:
    shared, delta = [], []
    for name in sorted(set(a["slots"]) & set(b["slots"])):
        va, vb = a["slots"][name], b["slots"][name]
        if va != vb:
            shared.append(name)
            delta.append(f"{name}: {sorted(va)} vs {sorted(vb)}")
    return shared, delta


def age_days(ts: float) -> float:
    import time
    if not ts:
        return 0.0
    return max(0.0, (time.time() - ts) / 86400.0)


def find(rows: list[dict], overlap_min: float = OVERLAP_MIN) -> list[Proposal]:
    """rows: dicts with fact_id, content, updated_ts, entities(set[int])."""
    for row in rows:
        row["tok"] = tokens(row["content"])
        row["slots"] = slots(row["content"])
    proposals: list[Proposal] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            ta, tb = a["tok"], b["tok"]
            if not ta or not tb:
                continue
            jac = len(ta & tb) / len(ta | tb)
            ea, eb = a.get("entities") or set(), b.get("entities") or set()
            if ea and eb:
                if ea != eb:
                    continue          # different subjects (Atlas Server vs Beacon Mini)
                if jac < overlap_min:
                    continue
            elif jac < max(NO_ENTITY_OVERLAP_MIN, overlap_min):
                continue
            shared, delta = conflicting_slots(a, b)
            if not shared:
                continue
            # newer updated_at wins; identical timestamps fall back to the
            # higher fact_id (the later insert), so ties are deterministic.
            key_a = (a["updated_ts"], a["fact_id"])
            key_b = (b["updated_ts"], b["fact_id"])
            new, old = (a, b) if key_a >= key_b else (b, a)
            proposals.append(Proposal(
                keep=new["fact_id"], retire=old["fact_id"],
                jaccard=round(jac, 3), slots=shared, delta=delta,
                keep_text=new["content"][:120], retire_text=old["content"][:120],
                keep_age_d=round(age_days(new["updated_ts"]), 1),
                retire_age_d=round(age_days(old["updated_ts"]), 1),
            ))
    proposals.sort(key=lambda p: (-p.jaccard, p.retire))
    return proposals


def scan(store, overlap_min: float = OVERLAP_MIN) -> list[Proposal]:
    """Find, persist as pending proposals, and return them."""
    rows = []
    for fact in store.all_facts():
        fact["entities"] = set(fact.get("entities") or [])
        rows.append(fact)
    proposals = find(rows, overlap_min)
    for p in proposals:
        store.add_conflict(p.keep, p.retire, p.jaccard, p.slots, p.delta)
    store.db.commit()
    return proposals


def apply(store, keep_id: int, retire_id: int, *, actor: str = "review",
          reason: str = "golden rule: newer supersedes older",
          golden_path=None) -> dict:
    """Retire one fact, recording provenance. Does not edit the keeper's text."""
    keeper = store.get(keep_id)
    retired = store.get(retire_id)
    if keeper is None or retired is None:
        raise KeyError("both facts must exist")
    store.remove_fact(retire_id, actor=actor,
                      reason=f"{reason} (kept #{keep_id})")
    store.update_fact(keep_id, trust_score=min(1.0, (keeper["trust_score"] or 0.5) + 0.05),
                      actor=actor, reason=f"superseded #{retire_id}")
    remapped = 0
    if golden_path is not None:
        from .golden import remap
        remapped = remap(golden_path, retire_id, keep_id)
    return {"kept": keep_id, "retired": retire_id, "golden_remapped": remapped}
