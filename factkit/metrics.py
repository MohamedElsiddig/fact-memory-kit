"""Metrics: retrieval quality, latency, embedding coverage and cleanup backlog."""
from __future__ import annotations

import html
from datetime import datetime, timezone

from . import golden


def summary(store, days: int = 30) -> dict:
    models = store.embedding_models()
    total = store.count()
    embedded = max((m["n"] for m in models), default=0)
    latency = store.latency_percentiles(days)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "facts": total,
        "entities": len(store.entities()),
        "embeddings": {
            "models": models,
            "coverage": round(min(1.0, embedded / total), 3) if total else 0.0,
        },
        "retrieval": latency,
        "audit_entries": len(store.audit_log(limit=1_000_000)),
        "backlog": {
            "conflicts_pending": len(store.conflicts("pending")),
            "proposals_pending": len(store.proposals("pending")),
            "ttl_pending": len(store.ttl_flags("pending")),
        },
        "events": store.metrics_summary(days),
    }


def render_text(data: dict) -> str:
    lines = [
        f"factkit report  ({data['generated_at']})",
        "-" * 46,
        f"facts              {data['facts']}",
        f"entities           {data['entities']}",
        f"embedding coverage {data['embeddings']['coverage']:.0%}",
    ]
    for model in data["embeddings"]["models"]:
        lines.append(f"  {model['model']:<34} {model['n']:>4} x {model['dims']}")
    retrieval = data["retrieval"]
    if retrieval.get("n"):
        lines += [
            f"search p50 / p95   {retrieval['p50_ms']} ms / {retrieval['p95_ms']} ms",
            f"search hit rate    {retrieval['hit_rate']:.0%}  (n={retrieval['n']})",
        ]
    by = data["backlog"]
    lines += [
        f"pending            {by['conflicts_pending']} conflicts, "
        f"{by['proposals_pending']} proposals, {by['ttl_pending']} TTL flags",
        f"audit entries      {data['audit_entries']}",
    ]
    return "\n".join(lines)


def render_html(data: dict) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(m['model'])}</td><td>{m['n']}</td>"
        f"<td>{m['dims']}</td><td>{html.escape(str(m['last']))}</td></tr>"
        for m in data["embeddings"]["models"])
    retrieval = data["retrieval"]
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>factkit dashboard</title>
<style>
body{{font:15px/1.5 ui-sans-serif,system-ui,sans-serif;margin:2rem;color:#111}}
h1{{font-size:1.3rem}} table{{border-collapse:collapse;margin:.5rem 0 1.5rem}}
td,th{{padding:.35rem .8rem;border-bottom:1px solid #e5e5e5;text-align:left}}
.big{{font-size:2rem;font-weight:600}} .muted{{color:#666}}
.grid{{display:flex;gap:2.5rem;flex-wrap:wrap}}
</style></head><body>
<h1>factkit — {html.escape(data['generated_at'])}</h1>
<div class="grid">
  <div><div class="big">{data['facts']}</div><div class="muted">facts</div></div>
  <div><div class="big">{data['entities']}</div><div class="muted">entities</div></div>
  <div><div class="big">{data['embeddings']['coverage']:.0%}</div><div class="muted">embedded</div></div>
  <div><div class="big">{retrieval.get('p95_ms', 0)}</div><div class="muted">p95 ms</div></div>
  <div><div class="big">{data['backlog']['conflicts_pending']}</div><div class="muted">conflicts</div></div>
  <div><div class="big">{data['backlog']['ttl_pending']}</div><div class="muted">TTL flags</div></div>
</div>
<h2>Embedding models</h2>
<table><tr><th>model</th><th>facts</th><th>dims</th><th>last write</th></tr>{rows}</table>
<h2>Retrieval</h2>
<p>{retrieval.get('n', 0)} searches · hit rate {retrieval.get('hit_rate', 0):.0%}
 · p50 {retrieval.get('p50_ms', 0)} ms · p95 {retrieval.get('p95_ms', 0)} ms</p>
</body></html>"""
