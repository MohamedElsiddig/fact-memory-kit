---
name: fact-memory
description: Use when a task needs the user's stored facts, devices, preferences or past decisions — retrieve, store and maintain long-term memory with factkit.
tags: memory, facts, retrieval, rag, sqlite
---

# Fact Memory

Long-term memory lives in a local SQLite store managed by `factkit`
(this repo). Treat it as the first source for anything user-specific, and as a
system of record you must not corrupt.

Full command reference and failure modes: `AGENTS.md`. Human overview: `README.md`.

## Retrieval order

1. `factkit search "<query>" -k 5 --json` — stored facts.
2. Project wiki / knowledge base (OpenKnowledge MCP, then markdown files).
3. Live system (`ss`, `lsof`, `systemctl`, `df`) for current state.
4. Public web only when the above cannot answer.

Stored facts outrank live readings: the user's value is the intended one
(upgraded RAM, moved service, changed port).

Unsure which source applies? `factkit route "<query>"` returns the route and the
command to run next.

## Storing

Store only durable, self-contained statements: specs, preferences, decisions,
configurations, identities. Resolve pronouns to real nouns.

```bash
factkit add "Atlas Server runs Debian at 192.0.2.10 with 16 GB RAM" \
  -c entity -t pi5,specs --trust 0.9
```

Ask `factkit capture "<text>"` when unsure; it returns `save`, `score`, `reasons`.
Never store questions, greetings, task state or transient secrets.

## Changing

* `factkit update <id> --content "..." --reason "..."` — corrections.
* Contradiction: add the new fact, run `factkit conflicts scan`, then resolve in
  `factkit review`. The golden rule keeps the newer `updated_at` and proposes the
  older for retirement.
* Delete only on explicit user request, always with `--reason`.
* Every mutation is audited: `factkit audit --fact-id <id> --json`.

## Maintenance

```bash
factkit conflicts scan     # stale contradictions (proposals only)
factkit ttl scan           # age + rotted ports/IPs/versions (proposals only)
factkit bridge <file|state.db>  # opt-in: transcript or Hermes DB
factkit review             # accept / reject / skip
factkit gate               # retrieval regression gate
factkit report             # coverage, p50/p95 latency, hit rate, backlog
```

Automation only proposes. A human accepts. Refuse to auto-apply conflict or TTL
resolutions unless the user explicitly asks for that policy.

## Quality bar

* Run `factkit gate` after any change to retrieval, embeddings or the golden set.
* Quote `r_at_5`, `mrr`, `hit_rate` and `fp` — never say "search feels better".
* `factkit embed` backfills new facts; coverage below 100% means facts were added
  while the embedding endpoint was down.
* Embeddings are optional: with the endpoint down, reads degrade to BM25 and
  writes still work.

## Pitfalls

* **Do not edit the SQLite file directly.** FTS triggers, audit rows and
  embeddings all bypass you.
* **Different subjects are not conflicts.** Entity sets must match; Atlas Server vs Beacon Mini
  share tokens but are different machines.
* **Slots are case-normalised.** `16 GB` → `16gb`; a raw string comparison misses
  the conflict.
* **Retiring a fact invalidates golden cases** that reference it. The review path
  remaps them; manual deletions need `factkit golden-write`.
* **Per-model embeddings.** A new model indexes alongside the old one; switch with
  `FACTKIT_EMBED_MODEL`, do not delete the previous rows unless asked.
* **One writer at a time.** WAL allows concurrent readers, but keep batch writes
  short to avoid `database is locked`.
* **factkit and Hermes stores are separate.** Import with `factkit import --hermes`
  (read-only, preserves timestamps/entities); do not edit `memory_store.db` directly.
* **Dashboard auth is opt-in.** Set `FACTKIT_DASHBOARD_TOKEN` before exposing it;
  without a token every route is open to whoever can reach the port.
