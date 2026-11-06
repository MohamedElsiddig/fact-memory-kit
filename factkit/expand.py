"""Query expansion from a small, curated alias table.

The goal is recall, not rewriting: a query keeps every one of its own tokens in
place and gains a few synonyms immediately after them. Nothing here calls a
model, touches the store or does stemming, so two calls with the same input
always return the same output.

    >>> expand("laptop memory")["terms"]
    ['laptop', 'memory', 'notebook', 'ram']
    >>> for_prefetch("laptop memory")
    'laptop memory notebook ram'

``apply`` is the optional second half: it reorders hits that are already
admitted by the normal gates, adding a small bonus when an *expansion* (never
an original token) appears in the content. See ``MAX_BONUS`` for the cap.
"""

from __future__ import annotations

import re

# Canonical word -> every word in its group, canonical first. Members expand to
# the *other* members of their group, in the order written here. Keep this
# table short and unambiguous: each extra synonym is a chance to retrieve an
# unrelated fact, and long ambiguous words ("drive", "board", "screen") are not
# worth that risk.
ALIASES: dict[str, tuple[str, ...]] = {
    "laptop": ("laptop", "notebook"),
    "ram": ("ram", "memory"),
    "storage": ("storage", "disk", "ssd"),
    "port": ("port", "listen"),
    "ip": ("ip", "address"),
    "router": ("router", "gateway"),
    "phone": ("phone", "mobile"),
    # "sayara" is the usual Latin transliteration of the Arabic سيارة (car).
    "car": ("car", "vehicle", "sayara"),
    "speaker": ("speaker", "assistant"),
    "stt": ("stt", "transcription"),
    "tts": ("tts", "voice"),
    "backup": ("backup", "snapshot"),
    "password": ("password", "credential"),
}

# word -> the other members of its group, in group order
_EXPANSIONS: dict[str, tuple[str, ...]] = {}
for _canonical, _members in ALIASES.items():
    for _member in _members:
        _EXPANSIONS.setdefault(_member, tuple(
            other for other in _members if other != _member))

# Tokens shorter than this never expand: "ip" is reachable from "address" but
# never injects a longer word on its own. Three-letter acronyms ("stt", "tts")
# are specific enough to expand.
MIN_EXPAND_LEN = 3

# Numbers with a unit are facts, not words to be rewritten ("16gb" must not
# become "16gb memory"). Short unit words are already excluded by length, but
# three-letter units are listed explicitly.
_UNITS = {
    "amp", "day", "days", "gib", "ghz", "hrs", "kwh", "mib", "min", "mins",
    "mhz", "sec", "secs", "tib", "volt", "watt", "week", "weeks",
}

# Quoted text is one token, including the quotes, and is never split or
# expanded; callers can hand it straight to a phrase search.
_TOKEN_RE = re.compile(r'"[^"]*"|\'[^\']*\'|[^\s"\']+')
_EDGE = ".,;:!?()[]{}<>"
_WORD_RE = re.compile(r"[a-z0-9]+")

# A hit can gain at most this much. It equals the default per-term weight, so
# extra matching terms never compound beyond one step: a hit 0.2 below another
# cannot overtake it no matter how many expansions it contains. Raise it only
# with a matching gap in mind, never to "make reranking stronger".
MAX_BONUS = 0.15


def _key(token: str) -> str:
    """Dedupe key: case-insensitive, quotes retained for phrases."""
    return token.lower()


def _clean(raw: str) -> str:
    """Lower-case a bare token and trim sentence punctuation from its edges."""
    return raw.strip(_EDGE).lower()


def _tokenize(query: str) -> list[str]:
    """Quote-aware tokens, lower-cased, in query order, duplicates removed."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(query or ""):
        quoted = len(raw) > 1 and raw[0] in "\"'" and raw[-1] == raw[0]
        token = raw if quoted else _clean(raw)
        if not token:
            continue
        key = _key(token)
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
    return out


def _expandable(token: str) -> bool:
    """True when a token is a plain word eligible for alias expansion."""
    if token[0] in "\"'":          # quoted phrase: preserve, never expand
        return False
    if len(token) < MIN_EXPAND_LEN:
        return False
    if not token.isalpha():        # digits, units, versions, snake_case
        return False
    return token not in _UNITS


def expand(query: str, *, max_terms: int = 8) -> dict:
    """Expand ``query`` with curated synonyms.

    Returns ``{"original": query, "terms": [...], "added": [...],
    "dropped": [...]}``. ``terms`` is the original tokens followed by their
    expansions, deduped case-insensitively and capped at ``max_terms``;
    ``added`` lists only the expansions that survived the cap and ``dropped``
    lists everything that did not (in candidate order), so a caller can see
    exactly what the cap cost.

    Original tokens come first, so a tight cap can only remove expansions.
    Tokens shorter than three characters, quoted phrases, numbers, unit words
    and words with punctuation in them are never expanded.
    """
    originals = _tokenize(query)
    candidates = list(originals)
    seen = {_key(token) for token in originals}

    for token in originals:
        if not _expandable(token):
            continue
        for other in _EXPANSIONS.get(token, ()):
            key = _key(other)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(other)

    cap = max(0, int(max_terms))
    terms = candidates[:cap]
    dropped = candidates[cap:]
    original_keys = {_key(token) for token in originals}
    added = [term for term in terms if _key(term) not in original_keys]
    return {"original": query, "terms": terms, "added": added, "dropped": dropped}


def _words(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _matches(term: str, content_words: str) -> bool:
    """Word-boundary match for a term (or phrase) in padded word text."""
    phrase = " ".join(_words(term.strip("\"'")))
    return bool(phrase) and f" {phrase} " in content_words


def apply(query: str, hits, *, weight: float = 0.15) -> list:
    """Rerank existing ``hits`` by alias overlap, without dropping any.

    ``hits`` is a list of dicts with ``content`` and ``score``. Each distinct
    expansion term found in a hit's content adds ``weight``, and the total
    bonus is capped at ``MAX_BONUS`` (0.15). A bonus is never negative and the
    input list is not modified; hits whose score is unchanged keep their input
    order (the sort is stable). An ``expand_bonus`` key is added to hits that
    earned one.

    The cap is the guarantee: no matter how many terms match, a hit can move
    up by at most 0.15, so a 0.6 hit cannot be passed by a 0.2 hit. Hits closer
    together than the cap may still swap — that is the intended reranking.
    """
    rows = [dict(hit) for hit in (hits or [])]
    added = expand(query)["added"]
    if not rows or not added:
        return rows

    try:
        step = float(weight)
    except (TypeError, ValueError):
        step = 0.0
    if step <= 0:
        return rows

    for row in rows:
        words = " " + " ".join(_words(row.get("content", ""))) + " "
        matches = sum(1 for term in added if _matches(term, words))
        if not matches:
            continue
        bonus = min(step * matches, MAX_BONUS)
        try:
            score = float(row.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        row["score"] = round(score + bonus, 6)
        row["expand_bonus"] = round(bonus, 6)

    # max_terms default keeps every alias; the sort is stable so equal scores
    # retain the order the caller supplied.
    rows.sort(key=lambda row: -float(row.get("score") or 0.0))
    return rows


def for_prefetch(query: str) -> str:
    """The text to embed for ``query``: itself plus at most three expansions.

    Expansions are appended only while the result stays within twice the
    original length, because a much longer string drifts away from what the
    user actually asked. Short queries therefore gain nothing ("laptop" is 6
    characters; " laptop notebook" would be 15), which is deliberate. Every
    term is tried in order and skipped if it does not fit, so one long synonym
    cannot hide a shorter one behind it.
    """
    text = query or ""
    if not text.strip():
        return ""
    added = expand(text)["added"]
    if not added:
        return text

    budget = 2 * len(text)
    chosen: list[str] = []
    for term in added[:3]:
        candidate = text + " " + " ".join(chosen + [term])
        if len(candidate) <= budget:
            chosen.append(term)
    if not chosen:
        return text
    return text + " " + " ".join(chosen)
