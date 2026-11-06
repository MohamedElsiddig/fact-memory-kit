# AGENTS.md — how another agent should use this repo

You are an agent with shell access. This file is your operating manual for
`fact-memory-kit`. Follow it instead of inventing a workflow.

## 1. Install (one time)

```bash
git clone https://github.com/MohamedElsiddig/fact-memory-kit.git
cd fact-memory-kit
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/factkit init
```

Add `.venv/bin` to `PATH`, or call `.venv/bin/factkit` explicitly. The only
runtime dependency is the Python standard library (3.10+).

Verify:

```bash
factkit doctor
```

`doctor` prints the database path, fact count, embedding endpoint/model, a health
probe and per-model embedding coverage. If the embedding probe fails, the kit
still works — every read path degrades to BM25.

## 2. Point it at an embedding server

Any OpenAI-compatible `/v1/embeddings` endpoint works. For a local server:

```bash
export FACTKIT_EMBED_BASE_URL=http://127.0.0.1:1234/v1
export FACTKIT_EMBED_MODEL=text-embedding-nomic-embed-text-v1.5
# only if your server requires a bearer token:
export FACTKIT_EMBED_API_KEY=...
```

Then index once and confirm the gate:

```bash
factkit embed           # backfill + golden gate
factkit report          # coverage, latency, backlog
```

## 3. Retrieval order — do not skip steps

For any question about the user's own machines, accounts, preferences or past
decisions:

1. `factkit search "<query>" -k 5 --json` — the fact store.
2. If empty or low confidence: the project wiki / knowledge base
   (OpenKnowledge MCP, or `~/wikis` markdown).
3. If still unanswered: live system commands (`ss`, `lsof`, `systemctl`, `df`).
4. Only then the public web.

Stored user facts outrank live readings: the user may have upgraded hardware or
moved a service, and their stored value is the intended one.

Use `factkit route "<query>"` when unsure which source applies. It returns
`route`, `confidence` and the next command to run.

```bash
factkit route "what are my laptop specs"
# {"route": "facts", "hint": "factkit search \"...\"", "confidence": 1.0}
```

## 4. Writing facts

Store a fact when *all* of these hold:

* it stays true beyond this conversation (spec, preference, decision, config);
* it is self-contained (resolve "it"/"that" to the real noun);
* it has a subject you can name and a value worth retrieving.

Never store: questions, greetings, one-off task state, secrets you were only
shown transiently, or anything the user asked you to forget.

```bash
factkit add "<one self-contained sentence>" -c user_pref -t style --trust 0.9
```

Prefer `factkit capture "<text>"` first when you are unsure. It returns
`{"save": true|false, "score": ..., "reasons": [...]}`.

`category` is one of `user_pref`, `project`, `tool`, `general`, or an entity kind
such as `entity`.

## 5. Changing facts — never overwrite silently

* Update: `factkit update <id> --content "..." --reason "why"`.
* Supersede: add the new fact, then check `factkit conflicts scan` — the golden
  rule nominates the pair and you accept it in `factkit review`.
* Delete only on an explicit user request, with `--reason`.

Every mutation writes an audit row; you can always answer "who changed this and
when" with `factkit audit --fact-id <id> --json`.

## 6. Maintenance loop (weekly or after a batch of changes)

```bash
factkit conflicts scan     # stale contradictions
factkit ttl scan           # aged facts and rotted ports/IPs/versions
factkit bridge <session>   # optional: propose facts from a transcript
factkit review             # human decides; never auto-apply
factkit gate               # retrieval is still healthy
factkit report             # numbers, not impressions
```

Rules:

* `conflicts` and `ttl` only *propose*. Do not auto-accept.
* `bridge` requires an LLM endpoint and is opt-in; proposals are deduped.
* Run `gate` after any change to retrieval or embeddings. A failed gate is a
  blocker: report the metric that dropped, do not call it "better".

## 7. Machine-readable use

Every read command accepts `--json` (search, list, audit, conflicts, ttl,
proposals, review queue, report). Example:

```bash
factkit search "transcriber port" -k 3 --json
```

```json
[{"fact_id": 94, "content": "Transcriber listens on 8010", "score": 0.0328,
  "bm25_rank": 1, "vec_rank": 2, "vec_score": 0.71, "entities": ["Whisper"]}]
```

Prefer these fields over parsing human output.

## 8. Bulk operations

```bash
factkit export --out facts.json --with-embeddings
factkit import facts.json --dry-run
factkit import facts.json
```

Import skips duplicates, validates trust scores, and never overwrites. Use
`--validate-golden golden.json` to confirm every referenced fact survived.

## 9. Failure modes

| Symptom | Meaning | Action |
|---|---|---|
| `embedding service unreachable` | endpoint down or wrong URL | facts unchanged; fix env, rerun `factkit embed` |
| `coverage` below 100% | new facts added while the service was down | rerun `factkit embed` |
| gate fails after adding facts | golden set is stale or retrieval regressed | `factkit gate -v`, then `golden-write` if the corpus changed |
| `database is locked` | two writers | retry; WAL is on, keep writes short |
| dashboard shows nothing | server started against another `FACTKIT_DB` | check `factkit doctor` |

## 10. What not to do

* Do not edit `memory_store.db` or the SQLite file directly; use the CLI/API so
  the audit trail and FTS triggers stay consistent.
* Do not delete facts to resolve a contradiction without review.
* Do not claim retrieval improved without quoting `r_at_5`, `mrr` and `fp`.
* Do not commit a `golden.json` generated from a demo database into a real
  deployment — generate it against the corpus you are shipping.
