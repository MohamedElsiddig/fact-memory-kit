"""Query router: facts | wiki | live | external.

Cheap heuristics decide most queries; an optional LLM fallback only runs when
the heuristic is unsure. The router returns a *hint* (the command to run), so an
agent with no built-in policy still gets a deterministic next step.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field

ROUTES = ("facts", "wiki", "live", "external", "mixed")

RULES = {
    "live": [
        (r"\b(?:is|are)\b.{0,30}\b(?:up|running|down|healthy|listening|reachable)\b", 2.5),
        (r"\b(?:service|container|process|daemon)\b.{0,20}\b(?:status|state)\b", 2.5),
        (r"\b(?:uptime|load average|free (?:ram|memory)|disk (?:space|usage)|cpu usage)\b", 3),
        (r"\b(?:current|right now|at the moment|live) (?:status|state|usage|version)\b", 3),
        (r"\b(?:logs?|journalctl|dmesg|ps aux|ss -tlnp|lsof)\b", 2.5),
        (r"\bwhat(?:'s| is) (?:using|running|listening)\b", 2.5),
        (r"\bip address of (?:this|the) (?:machine|host|pi|server)\b", 2),
    ],
    "external": [
        (r"\b(?:latest|newest|current)\b.{0,30}\b(?:release|version|price|news|update)\b", 3),
        (r"\b(?:release|version)\b.{0,20}\b(?:of|for)\b", 1.5),
        (r"\b(?:news|headlines|weather|forecast|stock|price of|exchange rate)\b", 3),
        (r"\b(?:today|this week|yesterday)\b", 1.5),
        (r"\bupstream\b|\bchangelog\b|\brelease notes\b", 2),
        (r"\bwho (?:won|is winning)\b", 2),
    ],
    "wiki": [
        (r"\bhow (?:do|does|to|can) (?:i|we|you)\b", 2.5),
        (r"\b(?:documentation|docs|guide|runbook|notes|wiki|procedure|playbook)\b", 2.5),
        (r"\b(?:why|explain|walk me through)\b", 1.5),
        (r"\b(?:setup|configure|install) (?:steps|instructions)\b", 2),
    ],
    "facts": [
        (r"\bmy (?:[\w-]+ )?(?:specs?|ip|address|password|config|setup|car|laptop|phone|home)\b", 3),
        (r"\bremember (?:when|that|what)\b", 3),
        (r"\b(?:did|do) (?:i|we) (?:tell|say|store|save|prefer)\b", 2.5),
        (r"\bwhat (?:is|was) (?:my|our)\b", 2.5),
        (r"\b(?:preferred|favourite|favorite|default) (?:[\w-]+ )?(?:way|method|port|model)\b", 2),
        (r"\bspecs?\b", 1.5),
    ],
}

HINT = {
    "facts": "factkit search \"{q}\"",
    "wiki": "search the knowledge base (OpenKnowledge MCP or ~/wikis)",
    "live": "run a read-only system command (ss/lsof/systemctl/df)",
    "external": "web search, then cite the source",
    "mixed": "facts first, then wiki, then live",
}


@dataclass
class RouteResult:
    route: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    hint: str = ""
    method: str = "heuristic"

    def as_dict(self) -> dict:
        return self.__dict__


def _score(text: str, patterns) -> tuple[float, list[str]]:
    total, why = 0.0, []
    for pattern, weight in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            total += weight
            why.append(match.group(0)[:50])
    return total, why


def heuristic(query: str) -> RouteResult:
    query = (query or "").strip()
    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}
    for route, patterns in RULES.items():
        score, why = _score(query, patterns)
        scores[route] = score
        reasons[route] = why

    best = max(scores, key=lambda r: scores[r])
    total = sum(scores.values()) or 1.0
    confidence = scores[best] / total
    if scores[best] == 0:
        best, confidence = "mixed", 0.0
        reasons.setdefault("mixed", ["no source signal"])
    return RouteResult(route=best, confidence=round(confidence, 2),
                       reasons=reasons.get(best, [])[:4],
                       hint=HINT[best].format(q=query))


LLM_PROMPT = """Classify the user's query into exactly one route.
Routes:
- facts: personal or stored knowledge (their devices, preferences, past decisions)
- wiki: how-to, documentation, explanations, procedures
- live: current machine state (running services, ports, disk, logs, uptime)
- external: public/current information (news, prices, releases, weather)
- mixed: genuinely needs more than one source

Reply with JSON only: {"route": "...", "confidence": 0.0-1.0, "reason": "..."}
Query: """


def llm_classify(query: str, base_url: str, model: str, api_key: str = "",
                 timeout: float = 30.0) -> RouteResult | None:
    body = json.dumps({
        "model": model, "temperature": 0,
        "messages": [{"role": "user", "content": LLM_PROMPT + query}],
    }).encode()
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        text = data["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", text, re.S)
        payload = json.loads(match.group(0)) if match else {}
        route = str(payload.get("route", "")).lower()
        if route not in ROUTES:
            return None
        conf = float(payload.get("confidence", 0.5))
        return RouteResult(route=route, confidence=round(min(max(conf, 0.0), 1.0), 2),
                           reasons=[str(payload.get("reason", ""))[:80]],
                           hint=HINT[route].format(q=query), method="llm")
    except Exception:  # noqa: BLE001 - routing must never break a turn
        return None


def route(query: str, cfg=None, use_llm: bool = True) -> RouteResult:
    result = heuristic(query)
    if use_llm and result.confidence < 0.5 and cfg is not None:
        fallback = llm_classify(query, cfg.llm_base_url, cfg.llm_model,
                                cfg.llm_api_key, cfg.llm_timeout)
        if fallback is not None:
            return fallback
    return result
