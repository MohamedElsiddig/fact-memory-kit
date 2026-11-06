"""Approve or reject recorded proposals from a chat reply.

``factkit holo maintain`` records ambiguous pairs in ``conflict_proposals``
with status ``pending``. The CLI can resolve them, but a chat user cannot open
a terminal, so each pending row gets a short code derived from its proposal id
and a reply such as ``approve 3f2a`` is translated back into the same apply
path the CLI uses.

Design rules:

* Codes are ``sha1(str(proposal_id))`` prefixes. Four hex characters are the
  normal length; when two pending proposals share that prefix the code is
  extended until it is unique in the listing, and a bare four-character
  prefix that stays ambiguous resolves to ``None`` instead of guessing.
* Acceptance goes through :func:`factkit.holo.apply_conflict` or
  :func:`factkit.holo.apply_merge`, never a direct delete, so the retire side
  is tombstoned first and stays restorable.
* A decision writes its own audit row (actor, code, proposal id, action).
  ``dry_run`` writes nothing at all.
* Every function is total on an empty or legacy store: a missing
  ``conflict_proposals`` table reads as "nothing pending".
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone

__all__ = ["pending", "resolve_code", "render", "decide", "history"]

CODE_LEN = 4
CODE_MAX = 40
TEXT_LIMIT = 100
MAX_RENDER_ROWS = 12

_HEX = frozenset("0123456789abcdef")
_ACTIONS = ("accept", "reject")


# ------------------------------------------------------------------- helpers
def _rows(store, sql: str, params=()) -> list[dict]:
    out = []
    for row in store.read(sql, params):
        try:
            out.append(dict(row))
        except (TypeError, ValueError):   # tuple rows from a plain connection
            out.append({str(i): value for i, value in enumerate(row)})
    return out


def _table_exists(store, name: str) -> bool:
    return bool(_rows(store, "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)))


def _columns(store, table: str) -> set[str]:
    """Column names of ``table``; empty when it (or PRAGMA support) is missing."""
    try:
        rows = store.read(f"PRAGMA table_info({table})")
    except Exception:  # noqa: BLE001 - schema probe must never be fatal
        return set()
    out = set()
    for row in rows:
        try:
            out.add(str(row["name"]))
            continue
        except (IndexError, KeyError, TypeError):
            pass
        try:                               # positional rows: (cid, name, type, ...)
            out.add(str(row[1]))
        except (IndexError, TypeError):
            continue
    return out


def _evidence(row: dict) -> dict:
    raw = row.get("evidence")
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _kind(row: dict) -> str:
    kind = str(_evidence(row).get("kind") or "").strip().lower()
    if kind in ("conflict", "duplicate"):
        return kind
    if str(row.get("slots") or "").strip().lower() == "duplicate":
        return "duplicate"
    return "conflict"


def _digest(proposal_id) -> str:
    return hashlib.sha1(str(proposal_id).encode("utf-8")).hexdigest()


def _codes(ids) -> list[str]:
    """Shortest stable sha1 prefix per id, unique within ``ids``.

    Four characters unless that prefix is shared; a shared prefix is extended
    until it names exactly one proposal in the set. The result depends only on
    the ids, so repeated calls over the same pending set return the same codes.
    """
    digests = [_digest(pid) for pid in ids]
    codes = [digest[:CODE_LEN] for digest in digests]
    counts = Counter(codes)
    for i, digest in enumerate(digests):
        if counts[codes[i]] == 1:
            continue
        for length in range(CODE_LEN + 1, CODE_MAX + 1):
            prefix = digest[:length]
            if sum(1 for other in digests if other.startswith(prefix)) == 1:
                codes[i] = prefix
                break
        else:                              # pragma: no cover - full sha1 collision
            codes[i] = digest
    return codes


def _clean_code(code) -> str:
    q = str(code or "").strip().lower().lstrip("#").strip()
    if not CODE_LEN <= len(q) <= CODE_MAX or not set(q) <= _HEX:
        return ""
    return q


def _select_columns(store) -> list[str]:
    """Column list shared by the read queries, tolerating an old schema."""
    cols = _columns(store, "conflict_proposals")
    wanted = ["id", "created_at", "resolved_at", "keep_id", "retire_id", "jaccard",
              "slots", "delta", "status", "note"]
    for name in ("batch_id", "evidence", "decided_by"):
        if name in cols:
            wanted.append(name)
    return wanted


def _fetch(store, where: str = "", params=()) -> list[dict]:
    if not _table_exists(store, "conflict_proposals"):
        return []
    cols = _select_columns(store)
    sql = f"SELECT {', '.join(cols)} FROM conflict_proposals"
    if where:
        sql += f" WHERE {where}"
    sql += " ORDER BY id"
    return _rows(store, sql, params)


def _fact_texts(store, rows) -> dict[int, str]:
    ids = sorted({int(row["keep_id"]) for row in rows} | {int(row["retire_id"]) for row in rows})
    if not ids:
        return {}
    marks = ",".join("?" for _ in ids)
    try:
        return {int(row["fact_id"]): str(row["content"] or "")
                for row in _rows(store, f"SELECT fact_id, content FROM facts WHERE fact_id IN ({marks})", ids)}
    except Exception:  # noqa: BLE001 - a missing facts table just means no text
        return {}


def _project(row: dict, texts: dict[int, str], code: str = "") -> dict:
    evidence = _evidence(row)
    keep_id, retire_id = int(row["keep_id"]), int(row["retire_id"])
    keep_text = str(evidence.get("keep_text") or texts.get(keep_id) or "")
    retire_text = str(evidence.get("retire_text") or texts.get(retire_id) or "")
    try:
        ratio = float(row.get("jaccard") or 0.0)
    except (TypeError, ValueError):
        ratio = 0.0
    return {
        "code": code,
        "proposal_id": int(row["id"]),
        "kind": _kind(row),
        "keep_id": keep_id,
        "retire_id": retire_id,
        "keep_text": keep_text[:TEXT_LIMIT],
        "retire_text": retire_text[:TEXT_LIMIT],
        "ratio": round(ratio, 3),
        "created_at": row.get("created_at"),
        "batch_id": row.get("batch_id"),
    }


def _match(rows: list[dict], code) -> dict | None:
    """The one pending row a code names, or None when missing or ambiguous."""
    q = _clean_code(code)
    if not q or not rows:
        return None
    codes = _codes([row["id"] for row in rows])
    exact = [row for row, assigned in zip(rows, codes) if assigned == q]
    if len(exact) == 1:
        return exact[0]
    prefixed = [row for row in rows if _digest(row["id"]).startswith(q)]
    return prefixed[0] if len(prefixed) == 1 else None


def _limit(value, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------- reads
def pending(store, *, limit=10) -> list[dict]:
    """Pending proposals, oldest first, each with the code the user types."""
    rows = _fetch(store, "status = 'pending'")
    if not rows:
        return []
    texts = _fact_texts(store, rows)
    codes = _codes([row["id"] for row in rows])
    return [_project(row, texts, code) for row, code in zip(rows, codes)][:_limit(limit, 10)]


def resolve_code(store, code) -> dict | None:
    """The pending proposal an exact code names; None when unknown or ambiguous."""
    rows = _fetch(store, "status = 'pending'")
    match = _match(rows, code)
    if match is None:
        return None
    codes = dict(zip([row["id"] for row in rows], _codes([row["id"] for row in rows])))
    return _project(match, _fact_texts(store, rows), codes[match["id"]])


def render(list_or_rows) -> str:
    """Telegram-friendly plain text: one line per proposal, at most ~15 lines."""
    items = []
    for raw in list_or_rows or []:
        item = dict(raw)
        pid = item.get("proposal_id", item.get("id"))
        item["proposal_id"] = pid
        if not item.get("code") and pid is not None:
            item["code"] = _codes([pid])[0]
        items.append(item)

    if not items:
        return "No pending proposals.\nNothing to approve or reject."

    total = len(items)
    shown = items[:MAX_RENDER_ROWS]
    if total <= MAX_RENDER_ROWS:
        lines = [f"Pending proposals ({total})"]
    else:
        lines = [f"Pending proposals ({len(shown)} of {total})"]

    for item in shown:
        code = str(item.get("code") or "")
        keep = item.get("keep_id", item.get("keep", "?"))
        retire = item.get("retire_id", item.get("retire", "?"))
        kind = str(item.get("kind") or "conflict")
        try:
            ratio = float(item.get("ratio") or 0.0)
        except (TypeError, ValueError):
            ratio = 0.0
        if kind == "duplicate":
            reason = f"duplicate {ratio:.0%} similar" if ratio else "duplicate"
        else:
            reason = f"conflict {ratio:.0%} overlap" if ratio else "conflict"
        lines.append(f"{code}  keep #{keep} / retire #{retire}  {reason}")

    lines.append("Reply: approve <code> or reject <code>")
    if total > len(shown):
        lines.append(f"Showing the first {len(shown)}; decide one to see the rest.")
    else:
        lines.append(f"e.g. approve {shown[0].get('code')}")
    return "\n".join(lines)


def history(store, *, limit=20) -> list[dict]:
    """Decided proposals, newest first, with status, decided_at and actor."""
    rows = _fetch(store, "status <> 'pending'")
    out = []
    for row in rows:
        try:
            ratio = float(row.get("jaccard") or 0.0)
        except (TypeError, ValueError):
            ratio = 0.0
        out.append({
            "proposal_id": int(row["id"]),
            "kind": _kind(row),
            "keep_id": int(row["keep_id"]),
            "retire_id": int(row["retire_id"]),
            "status": row.get("status"),
            "decided_at": row.get("resolved_at") or row.get("created_at"),
            "actor": row.get("decided_by") or "factkit",
            "ratio": round(ratio, 3),
            "batch_id": row.get("batch_id"),
        })
    out.sort(key=lambda item: str(item["decided_at"] or ""), reverse=True)
    return out[:_limit(limit, 20)]


# ---------------------------------------------------------------- decisions
def _ensure_decided_by(store) -> None:
    cols = _columns(store, "conflict_proposals")
    if cols and "decided_by" not in cols:
        with store._tx():
            store._write("ALTER TABLE conflict_proposals ADD COLUMN decided_by TEXT")


def _batch_id() -> str:
    return "chat-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _pair(row: dict) -> dict:
    """The holo pair shape rebuilt from the stored proposal row."""
    kind = _kind(row)
    slots = [slot.strip() for slot in str(row.get("slots") or "").split(",") if slot.strip()]
    if not slots or slots == ["duplicate"]:
        slots = ["duplicate" if kind == "duplicate" else "conflict"]
    try:
        ratio = float(row.get("jaccard") or 0.0)
    except (TypeError, ValueError):
        ratio = 0.0
    pair = {
        "keep": int(row["keep_id"]),
        "retire": int(row["retire_id"]),
        "kind": kind,
        "diffs": [{"slot": slot} for slot in slots],
    }
    if kind == "duplicate":
        pair["ratio"] = ratio
    else:
        pair["jaccard"] = ratio
    return pair


def _mark(store, proposal_id: int, status: str, actor: str, note: str) -> bool:
    with store._tx():
        cur = store._write(
            "UPDATE conflict_proposals SET status = ?, resolved_at = CURRENT_TIMESTAMP, "
            "decided_by = ?, note = CASE WHEN COALESCE(note, '') = '' THEN ? "
            "ELSE note || ' | ' || ? END WHERE id = ? AND status = 'pending'",
            (status, actor, note, note, proposal_id))
    return getattr(cur, "rowcount", 1) != 0


def _audit(store, actor: str, action: str, code: str, proposal_id: int, keep_id: int) -> None:
    with store._tx():
        store._write(
            "INSERT INTO audit(action, fact_id, actor, reason) VALUES (?,?,?,?)",
            (f"proposal-{action}", keep_id, actor,
             f"code={code} proposal={proposal_id} action={action}"))


def decide(store, code, *, action, actor="chat", dry_run=False) -> dict:
    """Accept or reject one pending proposal by code.

    ``accept`` calls holo's own apply path (tombstone first) and records the
    decision; ``reject`` only changes the proposal row. A dry run validates the
    decision and changes nothing. An already-decided proposal is refused with
    ``reason='already decided'`` so a repeated chat reply is harmless.
    """
    action = str(action or "").strip().lower()
    if action not in _ACTIONS:
        raise ValueError("action must be 'accept' or 'reject'")

    from . import holo, maintenance

    maintenance.ensure_tombstones(store)
    rows = _fetch(store)
    q = _clean_code(code)
    if not q:
        return {"ok": False, "reason": "unknown code", "code": str(code or ""), "action": action}

    row = _match([item for item in rows if item.get("status") == "pending"], q)
    if row is None:
        candidates = [item for item in rows if _digest(item["id"]).startswith(q)]
        if len(candidates) == 1:
            row = candidates[0]             # a stale code for a decided proposal
        elif len(candidates) > 1:
            return {"ok": False, "reason": "ambiguous code", "code": q, "action": action}
        else:
            return {"ok": False, "reason": "unknown code", "code": q, "action": action}

    proposal_id = int(row["id"])
    if row.get("status") != "pending":
        return {"ok": False, "reason": "already decided", "code": q,
                "proposal_id": proposal_id, "status": row.get("status"), "action": action}

    if action == "reject":
        if dry_run:
            return {"ok": True, "dry_run": True, "action": "reject", "code": q,
                    "proposal_id": proposal_id, "status": "rejected"}
        _ensure_decided_by(store)
        changed = _mark(store, proposal_id, "rejected", actor, f"rejected via chat code {q}")
        if not changed:
            return {"ok": False, "reason": "already decided", "code": q,
                    "proposal_id": proposal_id, "action": "reject"}
        _audit(store, actor, "reject", q, proposal_id, int(row["keep_id"]))
        return {"ok": True, "action": "reject", "code": q, "proposal_id": proposal_id,
                "status": "rejected"}

    pair = _pair(row)
    batch = _batch_id()
    fn = holo.apply_merge if pair["kind"] == "duplicate" else holo.apply_conflict
    result = fn(store, pair, dry_run=dry_run, batch_id=batch)
    if dry_run:
        return {"ok": bool(result.get("ok")), "dry_run": True, "action": "accept",
                "code": q, "proposal_id": proposal_id, "batch_id": batch, "result": result}

    if not result.get("ok"):
        return {"ok": False, "reason": result.get("reason", "apply failed"), "code": q,
                "proposal_id": proposal_id, "action": "accept", "result": result}

    _ensure_decided_by(store)
    changed = _mark(store, proposal_id, "accepted", actor,
                    f"accepted via chat code {q} batch={batch}")
    if not changed:                        # pragma: no cover - guarded by the status check
        return {"ok": False, "reason": "already decided", "code": q,
                "proposal_id": proposal_id, "action": "accept", "batch_id": batch}
    _audit(store, actor, "accept", q, proposal_id, int(row["keep_id"]))
    return {"ok": True, "action": "accept", "code": q, "proposal_id": proposal_id,
            "batch_id": batch, "keep": pair["keep"], "retire": pair["retire"],
            "status": "accepted", "result": result}
