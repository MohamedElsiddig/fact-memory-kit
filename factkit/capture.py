"""Proactive capture: should this message become a fact?

The heuristic answers one question with a score and explicit reasons, so an
agent can ask "save this?" only when it is worth interrupting the user. It is
deliberately conservative: questions, greetings, one-off tasks and anything
time-stamped "today/now" score low.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

UNITS = r"(?:gb|tb|mb|kb|ghz|mhz|cores?|watts?|w|v|mah|fps|ms|mbps|gbps)"
SPEC_PATTERNS = [
    (r"\b\d+(?:\.\d+)?\s*" + UNITS + r"\b", 3.0),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", 3.0),
    (r"(?::|\bport\s+)\d{2,5}\b", 2.5),
    (r"\bv?\d+\.\d+(?:\.\d+)?\b", 1.5),
    (r"\b[A-Z0-9]{6,}\b", 1.0),          # VIN, model codes, serials
    (r"~/[\w./-]+|/(?:opt|etc|var|srv|home)/[\w./-]+", 1.5),
]
PREFERENCE_PATTERNS = [
    (r"\bi (?:prefer|like|want|hate|dislike|need)\b", 3.0),
    (r"\b(?:always|never)\b", 2.0),
    (r"\bfrom now on\b|\bgoing forward\b", 3.0),
    (r"\b(?:use|prefer)\b.{0,24}\b(?:instead of|rather than)\b", 2.5),
    (r"\bdon'?t (?:ever|suggest|ask|use)\b", 2.5),
]
IDENTITY_PATTERNS = [
    (r"\bmy (?:name|car|house|home|wife|son|daughter|brother|sister|laptop|phone|address|email|job|work)\b", 2.5),
    (r"\bi (?:live|work|study) (?:in|at|near)\b", 2.5),
    (r"\bi(?:'m| am) (?:from|based in)\b", 2.0),
]
DECISION_PATTERNS = [
    (r"\b(?:we|i) decided\b|\bdecided to\b", 2.5),
    (r"\blet'?s (?:use|go with|stick with)\b", 2.0),
    (r"\bthe (?:plan|rule|policy) is\b", 2.0),
]
CORRECTION_PATTERNS = [
    (r"\bactually\b|\bcorrection\b|\bi meant\b|\bnot .{0,20} but\b", 2.0),
    (r"\bno longer\b|\bdoesn'?t work anymore\b", 2.0),
]
NEGATIVE_PATTERNS = [
    (r"^\s*(?:hi|hello|hey|thanks|thank you|ok|okay|sure|yes|no)\b[\s!.?]*$", 4.0),
    (r"\?\s*$", 2.5),                     # a question is not a fact
    (r"^\s*(?:please|can you|could you|would you|run|check|show|open|find|search)\b", 2.0),
    (r"\b(?:right now|today only|for now|this once|temporarily)\b", 2.0),
    (r"^\s*(?:/|!)[a-z]", 3.0),          # slash/bang commands
]

CATEGORY_RULES = [
    ("user_pref", re.compile(r"\bi (?:prefer|like|want|hate|dislike)\b|\balways\b|\bnever\b", re.I)),
    ("project", re.compile(r"\b(?:project|repo|branch|deploy|migration|build)\b", re.I)),
    ("tool", re.compile(r"\b(?:port|container|config|service|install|version|api)\b", re.I)),
]


@dataclass
class Suggestion:
    score: float
    save: bool
    reasons: list[str] = field(default_factory=list)
    category: str = "general"
    tags: str = ""

    def as_dict(self) -> dict:
        return {"score": round(self.score, 2), "save": self.save,
                "reasons": self.reasons, "category": self.category,
                "tags": self.tags}


def _hits(text: str, patterns) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    for pattern, weight in patterns:
        match = re.search(pattern, text, re.I | re.M)
        if match:
            score += weight
            reasons.append(match.group(0).strip()[:60])
    return score, reasons


def suggest(text: str, threshold: float = 3.0) -> Suggestion:
    text = (text or "").strip()
    if not text:
        return Suggestion(0.0, False, ["empty"])

    score = 0.0
    reasons: list[str] = []
    for group in (SPEC_PATTERNS, PREFERENCE_PATTERNS, IDENTITY_PATTERNS,
                  DECISION_PATTERNS, CORRECTION_PATTERNS):
        gain, why = _hits(text, group)
        score += gain
        reasons += why

    penalty, why_not = _hits(text, NEGATIVE_PATTERNS)
    score -= penalty
    reasons += [f"not: {w}" for w in why_not]

    # Length sanity: a bare fragment or a wall of text is rarely a good fact.
    words = len(text.split())
    if words < 4:
        score -= 1.5
        reasons.append("too short")
    if words > 90:
        score -= 1.0
        reasons.append("too long / multi-topic")

    category = "general"
    for name, rx in CATEGORY_RULES:
        if rx.search(text):
            category = name
            break

    tags = ""
    for pattern, tag in ((r"\bport\b|:\d{2,5}\b", "config"),
                         (r"\bprefer|always|never\b", "preference"),
                         (r"\b(?:gb|tb|ghz|cores?)\b", "specs")):
        if re.search(pattern, text, re.I):
            tags = ",".join(filter(None, [tags, tag]))

    return Suggestion(score=round(score, 2), save=score >= threshold,
                      reasons=reasons[:6], category=category, tags=tags)


def dedupe(store, text: str, embedder=None, model: str = "",
           threshold: float = 0.92) -> int | None:
    """Return an existing fact id when this text is already stored."""
    norm = " ".join(text.lower().split())
    for fact in store.list_facts(limit=1_000_000):
        if " ".join(fact["content"].lower().split()) == norm:
            return fact["fact_id"]
    if embedder is None or not model:
        return None
    try:
        qvec = embedder.embed_one(text)
    except Exception:  # noqa: BLE001
        return None
    from .vectors import cosine, pack, unpack
    qa = unpack(pack(qvec))
    best, best_id = 0.0, None
    for fact_id, vec in store.embeddings(model):
        score = cosine(qa, vec)
        if score > best:
            best, best_id = score, fact_id
    return best_id if best >= threshold else None
