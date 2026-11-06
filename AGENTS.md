# AGENTS.md — how another agent should use this repo

You are an agent with shell access. This file is your operating manual for
`fact-memory-kit`. Follow it instead of inventing a workflow.

## Rules for this repo

- Never edit the installed plugin by hand: `~/.hermes/plugins/factkit` is a copy.
  Change `integrations/hermes/factkit/` (or the module it vendors), then run
  `bash integrations/hermes/factkit/install.sh` and confirm with
  `install.sh --check` / `factkit plugin-status` (both must print MATCH).
- Never patch `~/.hermes/plugins/holographic`; provider updates overwrite it.
  All writes go through its `MemoryStore` API.
- Embedding calls are capped at 1.5s and guarded by a breaker. Do not raise the
  timeout to hide a slow endpoint; fix the endpoint or accept degraded recall.
- Conflict values are compared through `factkit.units`. Add a normalisation case
  there before widening any regex.
- LLM adjudication is proposals only. It never writes to the store.

## 0. Pick the mode

**Companion (default).** If the host already runs Hermes holographic memory,
`~/.hermes/memory_store.db` is the authoritative store. Do not create a second
database and do not edit the provider. Install the plugin and point factkit at
that file:

```bash
bash integrations/hermes/factkit/install.sh
hermes plugins enable factkit
export FACTKIT_DB=~/.hermes/memory_store.db
factkit embed
```

Maintenance then runs against the live store, and every write goes through the
provider's `MemoryStore` API (see section 8).

**Standalone.** Only when there is no existing store, or for a separate corpus:
use `factkit init` and the default `~/.factkit/facts.db`.

## 1. Install (one time)

```bash
git clone https://github.com/MohamedElsiddig/fact-memory-kit.git
cd fact-memory-kit
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/factkit init          # standalone mode only
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

Companion mode (the live Hermes store):

```bash
DB=~/.hermes/memory_store.db

factkit holo conflicts --db $DB   # same subject, same attribute, different value
factkit holo stale     --db $DB   # aged facts and rotted ports/IPs/versions
factkit holo dupes     --db $DB   # near-identical text; numbers must match

factkit holo maintain  --db $DB   # classify + record proposals (no writes)
factkit holo maintain  --db $DB --auto                     # auto class in the plan
factkit holo maintain  --db $DB --auto --apply --limit 5 \
        --smoke examples/smoke_cases.json                  # apply behind the gate

factkit holo tombstones --db $DB                       # what a batch retired
factkit holo undo       --db $DB --fact-id 17 --apply  # restore it under a new id
factkit holo purge      --db $DB --days 90             # no dry run
```

Standalone mode (the factkit corpus):

```bash
factkit conflicts scan     # stale contradictions
factkit ttl scan           # aged facts and rotted ports/IPs/versions
factkit bridge <file|state.db>  # optional: propose facts from a transcript or Hermes DB
factkit review             # human decides
factkit gate               # retrieval is still healthy
factkit report             # numbers, not impressions
```

Rules:

* Report-only by default. `--apply` acts on exactly the report it is attached to.
* `maintain` applies only the policy's auto class: near-duplicates ≥ 0.98, or a
  `port`/`ip`/`version` conflict on a shared declared entity with a ≥ 7-day gap
  and no trust downgrade. Hard limits: 5 per run, abort above 20% of the store,
  revert the batch when the smoke gate regresses. Every retirement is
  tombstoned first, so it is undoable.
* `--auto` is a policy switch, not consent. Do not enable it unless the user asks.
* Smoke-case ids in `examples/smoke_cases.json` are placeholders; replace them
  with ids from the store being guarded or the gate checks nothing.
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
factkit import --hermes ~/.hermes/memory_store.db   # from a Hermes agent
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
| dashboard returns 401 | token configured, request has none | send `Authorization: Bearer <token>` or open `/?token=...` |
| dashboard shows nothing | server started against another `FACTKIT_DB` | check `factkit doctor` |

## 10. What not to do

* Do not edit `memory_store.db` or the SQLite file directly; use the CLI/API so
  the audit trail and FTS triggers stay consistent.
* Do not delete facts to resolve a contradiction without review.
* Do not claim retrieval improved without quoting `r_at_5`, `mrr` and `fp`.
* Do not commit a `golden.json` generated from a demo database into a real
  deployment — generate it against the corpus you are shipping.
