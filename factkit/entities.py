"""Entity extraction and auto-linking.

Deterministic on purpose: a regex-and-gazetteer pass beats a small local LLM for
this job (no hallucinated entities, no extra latency, testable). New proper-noun
phrases become entities, so subject matching in the golden-rule scan improves
over time instead of depending on whoever wrote the fact.
"""
from __future__ import annotations

import re

# Words that start sentences or label sections and must never become entities.
STOP_CAP = {
    "the", "a", "an", "this", "that", "these", "those", "it", "he", "she",
    "they", "we", "you", "i", "if", "when", "while", "after", "before", "for",
    "from", "with", "and", "but", "or", "not", "no", "yes", "use", "always",
    "never", "note", "note:", "warning", "summary", "full", "current", "new",
    "old", "user", "fact", "facts", "config", "service", "container", "run",
    "running", "installed", "set", "add", "remove", "update", "check", "see",
    "laptop", "desktop", "server", "phone", "container", "service", "device",
    "gb", "tb", "mb", "ram", "cpu", "gpu", "ssd", "hdd", "api", "url", "port",
}

# Multi-word product families worth treating as one entity.
FAMILIES = [
    "raspberry pi", "home assistant", "lm studio", "yandex station",
    "york station", "apple tv", "google home", "amazon echo", "mac mini",
    "thinkpad", "rog strix", "legion go", "steam deck", "nvidia shield",
]

CAP_SEQ = re.compile(r"\b(?:[A-Z][A-Za-z0-9]*(?:[ \-][A-Z0-9][A-Za-z0-9]*){0,3})\b")
HOST = re.compile(r"\b([a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.(?:local|lan|home|internal))\b", re.I)
IDENT = re.compile(r"\b([A-Za-z]+[0-9]+[A-Za-z0-9]*|[a-z]+-[a-z0-9-]{3,})\b")


def _clean(name: str) -> str:
    return " ".join(name.strip(" .,:;()[]\"'").split())


def candidates(text: str, limit: int = 8) -> list[str]:
    text = text or ""
    found: list[str] = []
    seen: set[str] = set()

    def push(name: str) -> None:
        name = _clean(name)
        if not name or len(name) < 3 or len(name) > 60:
            return
        low = name.lower()
        if low in STOP_CAP or low in seen:
            return
        if low.isdigit():
            return
        first = low.split()[0]
        if first in {"gb", "tb", "mb", "ram", "cpu", "gpu", "ssd", "hdd", "port"}:
            return
        seen.add(low)
        found.append(name)

    low = text.lower()
    for family in FAMILIES:
        for match in re.finditer(re.escape(family), low):
            # capture the original casing plus an optional trailing number
            tail = text[match.end(): match.end() + 4]
            num = re.match(r"\s+(\d{1,2})\b", tail)
            push(text[match.start(): match.end()] + (num.group(0) if num else ""))

    for match in HOST.finditer(text):
        push(match.group(1).lower())

    for match in CAP_SEQ.finditer(text):
        name = match.group(0)
        first = name.split()[0]
        if first.lower() in STOP_CAP:
            continue
        if len(name.split()) == 1 and len(name) < 4:
            continue
        push(name)

    for match in IDENT.finditer(text):
        token = match.group(1)
        if token.lower() in STOP_CAP:
            continue
        push(token)
    return found[:limit]


def link(store, fact_id: int, text: str | None = None,
         create: bool = True) -> list[str]:
    """Attach a fact to entities. Existing aliases win; new names are created."""
    fact = store.get(fact_id)
    if fact is None:
        return []
    text = text if text is not None else fact["content"]
    linked: list[str] = []
    for name in candidates(text):
        entity_id = store.ensure_entity(name) if create else None
        if entity_id is None:
            row = store.db.execute(
                "SELECT entity_id FROM entities WHERE name=? COLLATE NOCASE",
                (name,)).fetchone()
            if row is None:
                continue
            entity_id = int(row["entity_id"])
        store.link_entity(fact_id, entity_id)
        linked.append(name)
    store.db.commit()
    return linked


def relink_all(store, create: bool = True) -> dict:
    linked = 0
    for fact in store.list_facts(limit=1_000_000):
        names = link(store, fact["fact_id"], fact["content"], create=create)
        linked += len(names)
    return {"facts": store.count(), "links": linked,
            "entities": len(store.entities())}


def resolve(store, mention: str) -> int | None:
    """Resolve a mention to an entity id by exact name or alias."""
    row = store.db.execute(
        "SELECT entity_id FROM entities WHERE name=? COLLATE NOCASE",
        (mention,)).fetchone()
    if row:
        return int(row["entity_id"])
    row = store.db.execute(
        "SELECT entity_id FROM entity_aliases WHERE alias=? COLLATE NOCASE",
        (mention,)).fetchone()
    return int(row["entity_id"]) if row else None
