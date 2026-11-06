"""Session -> fact bridge: propose durable facts from past conversations.

Opt-in by design. The LLM only *proposes*; every candidate is deduped and parked
in ``fact_proposals`` for review. Nothing enters the store without a human (or an
explicit ``--apply``) accepting it.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .capture import dedupe, suggest

EXTRACT_PROMPT = """You extract durable facts from a conversation transcript.

Rules:
- Only statements that stay true beyond this conversation: preferences, specs,
  decisions, configurations, identities, recurring constraints.
- Never extract questions, greetings, one-off tasks, or anything uncertain.
- One fact per line item, self-contained (resolve "it"/"that" to the real noun).
- Keep the user's meaning; do not invent numbers, ports, paths or names.
- Skip anything already obvious or generic.

Return JSON only:
{"facts": [{"content": "...", "category": "user_pref|project|tool|general",
            "tags": "comma,separated", "confidence": 0.0-1.0}]}

Transcript:
"""


@dataclass
class Candidate:
    content: str
    category: str = "general"
    tags: str = ""
    confidence: float = 0.7
    score: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__


def read_messages(path: Path, include_assistant: bool = False) -> list[tuple[str, str]]:
    """Return [(role, text)] from jsonl/json/markdown/plain transcripts."""
    raw = path.read_text(errors="replace")
    out: list[tuple[str, str]] = []
    if path.suffix in (".jsonl", ".ndjson"):
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.extend(_pair(obj, include_assistant))
    elif path.suffix == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        items = data if isinstance(data, list) else (data or {}).get("messages", [])
        for obj in items:
            out.extend(_pair(obj, include_assistant))
    else:
        role = "user"
        for block in re.split(r"\n{2,}", raw):
            match = re.match(r"^(user|assistant|human|ai)\s*:\s*(.*)$", block,
                             re.S | re.I)
            if match:
                role, block = match.group(1).lower(), match.group(2)
            if role.startswith(("user", "human")) or include_assistant:
                out.append((role, block.strip()))
    return out


def _pair(obj, include_assistant: bool) -> list[tuple[str, str]]:
    if not isinstance(obj, dict):
        return []
    role = str(obj.get("role") or obj.get("sender") or "").lower()
    content = obj.get("content") or obj.get("text") or ""
    if isinstance(content, list):
        content = " ".join(str(part.get("text", "")) if isinstance(part, dict) else str(part)
                           for part in content)
    if not content:
        msg = obj.get("message")
        if isinstance(msg, dict):
            role = str(msg.get("role", role)).lower()
            content = msg.get("content", "")
    if not isinstance(content, str) or not content.strip():
        return []
    if role.startswith(("user", "human")) or (include_assistant and role.startswith(("assistant", "ai"))):
        return [(role, content.strip())]
    return []


def chunks(messages: list[tuple[str, str]], size: int = 1400) -> list[str]:
    buf, out = [], []
    for _, text in messages:
        if len(text) > size:
            text = text[:size]
        if sum(len(t) for t in buf) + len(text) > size and buf:
            out.append("\n".join(buf))
            buf = []
        buf.append(text)
    if buf:
        out.append("\n".join(buf))
    return out


def _llm(base_url: str, model: str, api_key: str, prompt: str,
         timeout: float) -> dict | None:
    body = json.dumps({"model": model, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        text = data["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", text, re.S)
        payload = json.loads(match.group(0)) if match else {}
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001
        return None


def extract(messages: list[tuple[str, str]], cfg, *, max_chunks: int = 8) -> list[Candidate]:
    found: list[Candidate] = []
    for chunk in chunks(messages)[:max_chunks]:
        payload = _llm(cfg.llm_base_url, cfg.llm_model, cfg.llm_api_key,
                       EXTRACT_PROMPT + chunk, cfg.llm_timeout)
        if not payload:
            continue
        for item in payload.get("facts", []):
            if not isinstance(item, dict):
                continue
            content = " ".join(str(item.get("content", "")).split())
            if len(content) < 12:
                continue
            found.append(Candidate(
                content=content,
                category=str(item.get("category", "general")),
                tags=str(item.get("tags", "")),
                confidence=float(item.get("confidence", 0.7) or 0.7),
                score=suggest(content).score,
            ))
    return found


def run(store, cfg, path: Path, *, include_assistant: bool = False,
        embedder=None, model: str = "", max_chunks: int = 8,
        dry_run: bool = False) -> dict:
    messages = read_messages(path, include_assistant)
    candidates = extract(messages, cfg, max_chunks=max_chunks)
    kept, dupes = [], 0
    for cand in candidates:
        existing = dedupe(store, cand.content, embedder, model,
                          cfg.dedupe_threshold)
        if existing:
            dupes += 1
            continue
        kept.append(cand)
        if not dry_run:
            store.add_proposal(cand.content, cand.category, cand.tags,
                               source=f"bridge:{path.name}",
                               score=cand.score,
                               reasons=f"confidence={cand.confidence:.2f}")
    store.db.commit()
    return {"messages": len(messages), "chunks": len(chunks(messages)),
            "candidates": len(candidates), "duplicates": dupes,
            "proposed": len(kept), "dry_run": dry_run,
            "facts": [c.as_dict() for c in kept]}


def apply(store, proposal_id: int, *, actor: str = "bridge-review") -> dict:
    row = store.db.execute(
        "SELECT * FROM fact_proposals WHERE id=?", (proposal_id,)).fetchone()
    if row is None:
        raise KeyError(f"no proposal #{proposal_id}")
    fact = store.add_fact(row["content"], category=row["category"],
                          tags=row["tags"], source=row["source"],
                          actor=actor, reason="accepted proposal")
    store.resolve_proposal(proposal_id, "accepted")
    store.db.commit()
    return fact
