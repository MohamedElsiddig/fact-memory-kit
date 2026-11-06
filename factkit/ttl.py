"""Fact TTL: flag (never delete) facts that have aged out or whose config rotted.

Two signals:
* age — nothing has touched the fact in N days (default 90).
* slot rot — the fact carries a volatile slot (port, IP, version, path, host)
  and is older than the slot horizon (default 180). Those are the ones that go
  stale silently: a port moves, an IP is reassigned, a version is bumped.
"""
from __future__ import annotations

import time

from .conflicts import SLOTS, age_days

VOLATILE = ("port", "ip", "version", "path", "host")


def scan(store, ttl_days: int = 90, slot_ttl_days: int = 180) -> list[dict]:
    flags: list[dict] = []
    now = time.time()
    for fact in store.all_facts():
        age = age_days(fact["updated_ts"])
        found = SLOTS
        present = [name for name, rx in found.items() if rx.search(fact["content"] or "")]
        if age >= ttl_days:
            detail = f"untouched for {age:.0f} days (limit {ttl_days})"
            store.flag_ttl(fact["fact_id"], "age", detail, round(age, 1))
            flags.append({"fact_id": fact["fact_id"], "kind": "age",
                          "detail": detail, "age_days": round(age, 1),
                          "content": fact["content"]})
        volatile = [s for s in present if s in VOLATILE]
        if volatile and age >= slot_ttl_days:
            detail = (f"{'/'.join(volatile)} slot(s) older than {slot_ttl_days} days: "
                      f"{fact['content'][:90]}")
            store.flag_ttl(fact["fact_id"], "slot", detail, round(age, 1))
            flags.append({"fact_id": fact["fact_id"], "kind": "slot",
                          "detail": detail, "age_days": round(age, 1),
                          "content": fact["content"]})
    store.db.commit()
    return flags


def due(store) -> list[dict]:
    """Facts whose explicit review_after has passed."""
    out = []
    for fact in store.all_facts():
        review_after = fact.get("review_after")
        if not review_after:
            continue
        ts = store._ts(review_after)
        if ts and ts <= time.time():
            out.append({"fact_id": fact["fact_id"], "content": fact["content"],
                        "review_after": review_after,
                        "age_days": round(age_days(fact["updated_ts"]), 1)})
    return out


def acknowledge(store, flag_id: int, status: str = "ack") -> None:
    store.resolve_ttl(flag_id, status)
    store.db.commit()
