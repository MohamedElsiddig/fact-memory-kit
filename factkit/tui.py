"""Review TUI: one queue for conflicts, TTL flags and bridge proposals.

Interactive when a TTY is attached; every action is also a plain function so the
CLI and tests can drive it without a terminal.
"""
from __future__ import annotations

import sys

from pathlib import Path

from . import bridge, conflicts

CLEAR = "\033[2J\033[H"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def queue(store) -> list[dict]:
    items: list[dict] = []
    for row in store.conflicts("pending"):
        items.append({
            "kind": "conflict", "id": row["id"],
            "title": f"KEEP #{row['keep_id']}  <- retire #{row['retire_id']}",
            "detail": row["delta"], "jaccard": row["jaccard"],
            "keep_id": row["keep_id"], "retire_id": row["retire_id"],
        })
    for row in store.proposals("pending"):
        items.append({
            "kind": "proposal", "id": row["id"],
            "title": f"[{row['category']}] {row['content'][:70]}",
            "detail": row["reasons"], "score": row["score"],
        })
    for row in store.ttl_flags("pending"):
        items.append({
            "kind": "ttl", "id": row["id"],
            "title": f"#{row['fact_id']} {row['kind']} ({row['age_days']:.0f}d)",
            "detail": row["detail"], "score": row["age_days"],
        })
    return items


def decide(store, kind: str, item_id: int, action: str, note: str = "") -> dict:
    """action: accept | reject | skip."""
    if kind == "conflict":
        if action == "accept":
            proposal = store.db.execute(
                "SELECT * FROM conflict_proposals WHERE id=?", (item_id,)).fetchone()
            if proposal is None:
                return {"ok": False, "error": "proposal not found"}
            candidate = Path.cwd() / "golden.json"
            conflicts.apply(store, proposal["keep_id"], proposal["retire_id"],
                            reason=f"review accepted: {note or 'golden rule'}",
                            golden_path=candidate if candidate.exists() else None)
            store.resolve_conflict(item_id, "applied", note)
        else:
            store.resolve_conflict(item_id, "rejected" if action == "reject" else "skipped", note)
    elif kind == "proposal":
        if action == "accept":
            bridge.apply(store, item_id)
        else:
            store.resolve_proposal(item_id, "rejected" if action == "reject" else "skipped", note)
    elif kind == "ttl":
        if action == "accept":
            fact_id = store.db.execute(
                "SELECT fact_id FROM ttl_flags WHERE id=?", (item_id,)).fetchone()[0]
            store.update_fact(int(fact_id),
                              reason=f"TTL reviewed: {note or 'still valid'}",
                              actor="ttl-review")
            store.resolve_ttl(item_id, "ack")
        else:
            store.resolve_ttl(item_id, "rejected" if action == "reject" else "skipped")
    else:
        return {"ok": False, "error": f"unknown kind {kind}"}
    store.db.commit()
    return {"ok": True, "kind": kind, "id": item_id, "action": action}


def _render(items: list[dict], index: int) -> None:
    if sys.stdout.isatty():
        print(CLEAR, end="")
    print(f"{BOLD}factkit review{RESET}  {index + 1}/{len(items)}\n")
    item = items[index]
    print(f"  {BOLD}{item['kind'].upper()}{RESET}  {item['title']}\n")
    print(f"  {DIM}{item['detail']}{RESET}\n")
    print("  [a] accept   [r] reject   [s] skip   [q] quit")


def interactive(store) -> int:
    items = queue(store)
    if not items:
        print("nothing pending")
        return 0
    index, changed = 0, 0
    while index < len(items):
        _render(items, index)
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if choice in ("q", "quit"):
            break
        if choice in ("a", "r", "s"):
            action = {"a": "accept", "r": "reject", "s": "skip"}[choice]
            decide(store, items[index]["kind"], items[index]["id"], action)
            changed += 1
        elif choice.isdigit():
            index = max(0, min(len(items) - 1, int(choice) - 1))
            continue
        index += 1
    print(f"\n{changed} item(s) handled")
    return 0
