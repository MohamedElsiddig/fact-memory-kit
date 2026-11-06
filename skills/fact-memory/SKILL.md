---
name: fact-memory
description: Use when retrieving or maintaining the user's stored facts.
tags: memory, facts, retrieval, rag, sqlite, holographic
---

# Fact Memory

Long-term memory is the Hermes holographic store at
`~/.hermes/memory_store.db`, managed by the `holographic` provider and enhanced
by the `factkit` companion plugin (this repo). Treat it as the first source for
anything user-specific, and as a system of record you must not corrupt.

Full command reference and failure modes: `AGENTS.md`. Human overview: `README.md`.

## Retrieval order

1. `fact_store` (holographic provider) — FTS + HRR + trust, the ranking of record.
2. Semantic block injected by the `factkit` `pre_llm_call` hook (up to 3 facts).
3. Project wiki / knowledge base (OpenKnowledge MCP, then markdown files).
4. Live system (`ss`, `lsof`, `systemctl`, `df`) for current state.
5. Public web only when the above cannot answer.

Stored facts outrank live readings: the user's value is the intended one
(upgraded RAM, moved service, changed port).

## Storing

Store only durable, self-contained statements: specs, preferences, decisions,
configurations, identities. Resolve pronouns to real nouns. Use the `fact_store`
tool, never raw SQL.

```bash
factkit capture "<text>"        # save / score / reasons before storing
```

Never store questions, greetings, task state or transient secrets.

## How the enhancement works

| Layer | Behaviour |
|---|---|
| `pre_llm_call` | embeds the user message, injects up to 3 facts; deduped against the session and the conversation history |
| `transform_tool_result` | re-ranks `fact_store` search results with RRF (provider order + embeddings); provider hits are never dropped |
| breaker | 3 failed embedding calls → 5 minutes off, turns pay 0ms |
| prewarm | 10-minute cron keeps the embedding model resident |

Embeddings live in `fact_embeddings` beside the facts; `hrr_vector` and the
provider's own ranking are untouched.

## Maintenance

Report-only by default; `--apply` is explicit and never implied.

```bash
DB=~/.hermes/memory_store.db

factkit holo conflicts --db $DB             # same subject, same attribute, different value
factkit holo stale     --db $DB             # age + volatile attributes
factkit holo dupes     --db $DB             # near-identical text (numbers must match)
factkit holo maintain  --db $DB             # classify + propose; --auto enables the auto class
factkit holo maintain  --db $DB --auto --apply --limit 5 --smoke cases.json
factkit holo tombstones --db $DB            # what a batch retired
factkit holo undo --db $DB --fact-id 17 --apply
```

`maintain` applies only the policy's `auto` class (near-duplicates ≥ 0.98, or a
`port`/`ip`/`version` conflict on a shared declared entity with a ≥ 7-day gap and
no trust downgrade), capped at 5 per run, aborted when candidates exceed 20% of
the store, reverted when the smoke gate regresses. Every retirement is
tombstoned before deletion.

Automation runs proposal-only by default (nightly cron). Do not enable the auto
class without the user's explicit instruction.

## Quality bar

* `factkit gate` guards the standalone corpus; the live store uses the smoke
  cases passed to `maintain`.
* Quote `r_at_5`, `mrr`, `hit_rate`, `fp` — never say "search feels better".
* `factkit embed` backfills new facts; coverage below 100% means facts were added
  while the endpoint was down.
* Values are compared after unit normalisation: `16GB` == `16384MB`,
  `3.6GHz` == `3600MHz`, `8080` conflicts with `8000-8100`.

## Pitfalls

* **Never edit `~/.hermes/plugins/holographic`.** Provider updates overwrite it.
  Writes go through its `MemoryStore` API (`add_fact_status`, `update_fact`,
  `replace_fact`, `remove_fact`).
* **Never edit `~/.hermes/plugins/factkit` by hand.** It is a copy: change the
  repo, re-run `install.sh`, verify with `install.sh --check` or
  `factkit plugin-status` (both must print MATCH).
* **Different subjects are not conflicts.** A shared city or vendor is not a
  subject; entity sets must match.
* **Same sentence, different number is not a duplicate.** `filler fact 1` and
  `filler fact 2` score 0.95 similarity and are different claims.
* **Ports and IPs are identity values.** On the same subject, `7000` vs `7001` is
  a conflict, not two acceptable truths.
* **Retiring a fact invalidates golden cases** that reference it.
* **One writer at a time.** WAL allows concurrent readers; keep batches short.
* **Dashboard auth is opt-in.** Set `FACTKIT_DASHBOARD_TOKEN` before exposing it.
