"""Optional LLM adjudication for prose contradictions.

Attribute matching (see :mod:`factkit.units`) only sees named values such as
RAM size or a port. It cannot tell that "the proxy runs on node A" and "the
proxy runs on node B" disagree. This module asks a local chat model to judge a
small number of candidate pairs and returns a strict verdict.

Rules:
  * proposals only - nothing here writes to the store;
  * capped per run, and skipped entirely when the endpoint is unavailable;
  * any malformed answer is discarded, never guessed at.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

MAX_PAIRS = 50

PROMPT = (
    "You compare two stored facts about the same subject. Decide whether they "
    "contradict each other: they cannot both be true at the same time.\n"
    "Ignore differences in wording, detail level or age. A newer value that "
    "replaces an older one IS a contradiction.\n"
    "Answer with JSON only: "
    '{"contradiction": true|false, "attribute": "<short name>", '
    '"evidence": ["<span from fact 1>", "<span from fact 2>"]}\n\n'
    "FACT 1: {a}\nFACT 2: {b}\n"
)


def _post(cfg, prompt: str, timeout: float = 60.0) -> str:
    body = json.dumps({
        "model": cfg.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 200,
    }).encode()
    req = urllib.request.Request(
        f"{cfg.llm_base_url.rstrip('/')}/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {cfg.llm_api_key}"} if cfg.llm_api_key else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        payload = json.loads(resp.read())
    return payload["choices"][0]["message"]["content"]


def parse_verdict(text: str) -> dict | None:
    """Extract the JSON object from a model answer; None when unusable."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict) or not isinstance(data.get("contradiction"), bool):
        return None
    evidence = data.get("evidence")
    if not isinstance(evidence, list) or not all(isinstance(x, str) for x in evidence):
        data["evidence"] = []
    data["attribute"] = str(data.get("attribute") or "")[:60]
    return data


def adjudicate(pairs: list[dict], cfg, *, limit: int = MAX_PAIRS,
               timeout: float = 60.0, post=None) -> list[dict]:
    """Annotate pairs with a model verdict; drop anything unparseable.

    Never raises: an endpoint failure ends adjudication and returns what was
    judged so far.
    """
    post = post or (lambda prompt: _post(cfg, prompt, timeout=timeout))
    out: list[dict] = []
    for pair in pairs[:limit]:
        a = pair.get("keep_content") or pair.get("a") or ""
        b = pair.get("retire_content") or pair.get("b") or ""
        if not a or not b:
            continue
        try:
            verdict = parse_verdict(post(PROMPT.replace("{a}", a).replace("{b}", b)))
        except Exception:  # noqa: BLE001 - endpoint down: stop, keep proposals
            break
        if verdict is None:
            continue
        if verdict["contradiction"]:
            out.append({**pair, "verdict": verdict, "source": "llm"})
    return out
