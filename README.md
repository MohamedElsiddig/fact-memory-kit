# fact-memory-kit

Semantic recall and safe maintenance for an existing long-term fact store —
built for, and running against, Hermes' holographic memory.

It does not replace the memory provider and does not keep a second database.
It reads and writes `~/.hermes/memory_store.db`, the same file the `fact_store`
tool uses, and adds the parts that store lacks: embeddings, attribute-level
conflict detection, TTL review, duplicate merge, and an audit trail.

```
                    ┌─────────────── same SQLite file ───────────────┐
 fact_store tool ──►│ facts · FTS5 · HRR · entities · trust          │◄── provider prefetch
                    └───────▲───────────────────────────▲────────────┘
                            │                           │
          factkit holo ─────┘                           └───── factkit plugin
          (conflicts, TTL, merge, audit)                      (pre_llm_call semantic recall)
```

## What it adds

| Capability | Command / hook | Effect |
|---|---|---|
| Semantic recall | `pre_llm_call` hook | injects up to 3 embedding-matched facts per turn, beside the provider's keyword recall |
| Auto-indexing | `on_session_end` hook | embeds facts added since the last run |
| Conflict detection | `factkit holo conflicts` | same subject + same named attribute + different value → newer wins |
| Conflict apply | `factkit holo conflicts --apply` | retires the older fact through the provider's own API |
| TTL review | `factkit holo stale` | age and volatile-attribute flags; never deletes |
| Duplicate merge | `factkit holo dupes [--apply]` | near-identical facts, keep the higher-trust copy |
| Pre-insert guard | `factkit holo guard --text "..."` | `ok` / `duplicate` / `near` / `conflict` before a fact is written; exit 1 blocks |
| Chat approval | `factkit holo approve [--code X] [--apply]` | pending proposals with 4-char codes, accept or reject without the review TUI |
| Confidence decay | `factkit holo decay` | age bands (fresh/aging/stale/expired), read-only suggestions |
| Provenance | `factkit holo provenance [--apply]` | content hashes expose the provider's own writes |
| Prose conflicts | `factkit holo prose [--apply]` | LLM-judged contradictions between same-subject facts, proposals only |
| Live retrieval metrics | `factkit holo metrics` | hit@k, MRR, regression gate against the live store |
| Store limits | `factkit holo limits [--apply]` | archive, audit/tombstone pruning, WAL-safe vacuum |
| Time-aware query | `factkit holo asof --text "..." --as-of DATE` | what was true at a point in time |
| Backup + drill | `factkit_backup.py [--drill]` | sqlite backup API, integrity check, restore rehearsal |
| Query expansion | `factkit holo expand --text "..."` | alias-expanded query, then a capped re-rank of prefetch hits |
| Model migration | `factkit holo migrate [--apply] [--verify] [--rollback]` | resumable vectors for a second embedding model, old rows untouched |
| Relations | `factkit holo relations [--src N --dst M --rel R --apply]` | edge graph, inferred edges (proposals only), stats |
| Multi-hop | `factkit holo hops --text "..." --depth 2` | walks relation edges from prefetch seeds, score decays per hop |
| HRR benchmark | `factkit holo hrr` / `scripts/hrr_bench.py` | HRR vs lexical A/B; exits 1 unless it wins both metrics |
| Cross-encoder rerank | `factkit holo rerank --text "..." --llm-url ...` | opt-in, 250 ms budget, refuses the new order without a gain |
| Namespaces | `factkit holo ns [--namespace X --assign-id N]` | tag-based scoping (`ns:<name>`), post-filtered recall |
| Semantic query | `factkit prefetch "..."` | exactly what the hook would inject |
| Golden gate | `factkit gate` | retrieval regression suite with thresholds |
| Standalone mode | `factkit init` / `embed` / `search` | own store, for use without Hermes |

## Install — companion plugin (recommended)

```bash
git clone https://github.com/MohamedElsiddig/fact-memory-kit.git
cd fact-memory-kit
python3 -m venv .venv && . .venv/bin/activate && pip install -e .

bash integrations/hermes/factkit/install.sh
hermes plugins enable factkit

FACTKIT_DB=~/.hermes/memory_store.db factkit embed   # index existing facts
```

`install.sh` copies the plugin to `~/.hermes/plugins/factkit` and vendors the
modules it needs (`holo`, `config`, `embed`, `vectors`, `units`, `embedcache`,
`maintenance`). `install.sh --check` compares every vendored file by sha256 and
exits 1 on drift. The `holographic`
provider is not modified, so its updates cannot overwrite this.

Configure the endpoint in `~/.hermes/plugins/factkit/config.yaml` (mode `600`):

```yaml
embed_base_url: http://127.0.0.1:1234/v1
embed_api_key: ""
embed_model: text-embedding-nomic-embed-text-v1.5
prefetch: true
limit: 3
min_score: 0.55
```

Verify:

```bash
hermes plugins list | grep factkit        # enabled
factkit prefetch "what car do I drive"    # non-empty
```

## Install — standalone

```bash
factkit init --demo
factkit search "atlas server specs"
factkit add "The proxy listens on node-b at 192.0.2.20" -c tool -t proxy
```

Standalone uses `~/.factkit/facts.db`. Point `FACTKIT_DB` elsewhere to keep
several corpora apart. Python 3.10+, standard library only (`numpy` optional).

## Embeddings

Any OpenAI-compatible endpoint works; embeddings are optional and failures are
soft — reads degrade to BM25, the hook injects nothing, writes still work.

```bash
export FACTKIT_EMBED_BASE_URL=http://127.0.0.1:1234/v1
export FACTKIT_EMBED_MODEL=text-embedding-nomic-embed-text-v1.5
export FACTKIT_EMBED_API_KEY=sk-...        # only if the endpoint needs one
factkit embed --health
factkit embed
```

Vectors live per model in `fact_embeddings(fact_id, model, dims, vec)`, so a
second model can be indexed beside the first and switched without touching facts.

## Maintenance

Report-only by default. `--apply` is explicit and never implied.

### Report

```bash
DB=~/.hermes/memory_store.db

factkit holo conflicts --db $DB    # same subject + same attribute + different value
factkit holo stale     --db $DB    # age and volatile-attribute flags
factkit holo dupes     --db $DB    # near-identical text, digits included
```

Each prints JSON and writes nothing: `conflicts` nominates keep ← retire pairs with
the differing attributes, `stale` flags aged facts and rotted ports/IPs/versions,
`dupes` groups near-identical wording. `--apply` on any one of them acts on that
report alone (`stale --touch 17 --apply` marks a reviewed fact as still true).

### Classify and propose

```bash
factkit holo maintain --db $DB                                  # plan only
factkit holo maintain --db $DB --auto                           # auto class in the plan
factkit holo maintain --db $DB --auto --limit 3 --max-ratio 0.10
```

`maintain` runs both detectors, enriches each pair with category, trust and source,
and labels it `auto`, `propose` or `never` by the policy below. Nothing is written
without `--apply`; `--auto` only enables the `auto` class. `--limit` caps applied
changes (never above the policy's 5) and `--max-ratio` aborts the batch when
candidates exceed that fraction of the store.

### Apply behind a smoke gate

```bash
factkit holo maintain --db $DB --auto --apply --limit 5 \
  --smoke examples/smoke_cases.json
```

The cases file is a JSON list of `{"query": ..., "expect": [fact_id]}`; a case
passes when any expected id appears in the top 5 semantic hits. **Replace the
placeholder ids in `examples/smoke_cases.json` with ids from your own store** —
fact ids are local to a database. Recall runs before and after the batch; a query
that passed before and fails after reverts every retirement in the batch. An
unreachable embedder marks cases skipped (`degraded`), never passed.

### Tombstones and undo

```bash
factkit holo tombstones --db $DB                  # unrestored retirements
factkit holo undo --db $DB --fact-id 17 --apply   # re-add fact 17 from its tombstone
```

Every retirement is copied to `fact_tombstones` before the delete, so `undo`
restores content, category, tags, trust and declared entities through
`add_fact_status` — under a **new** fact id (`restored_from: 17`). `tombstones
--apply` also lists rows that were already restored. `purge --days 90` permanently
drops restored tombstones and unrestored ones older than the window; it has no dry
run.

### Policy

| Class | What falls in it | Behaviour |
|---|---|---|
| auto-applies | duplicate ratio ≥ 0.98; a `port`/`ip`/`version` conflict on a shared declared entity where the retire side is ≥ 7 days older, its trust is not higher and neither side changed in 24h | only with `--auto --apply` |
| proposed only | every other attribute slot, prose pairs, pairs missing category/trust/timestamps, `--adjudicate` output | queued in `conflict_proposals` on `--apply`, never retired |
| never touched | cross-category pairs, user-origin or verbatim facts, configured denied categories | excluded even with `--auto` |
| hard limits | 5 applies per run (`--limit` can only lower it); abort above 20% of the store (`--max-ratio` moves the line); revert the batch on a smoke regression; tombstone before delete | fixed in the policy, not bypassed by flags |

The nightly cron (`factkit_maintain.py`) is proposal-only outside the policy's auto
class, and silent unless it applied or proposed something.

Writes go through the provider's `MemoryStore` API (`update_fact`, `remove_fact`,
`replace_fact`, `add_fact_status`), so FTS triggers, HRR vectors, entity links,
trust scores and bank invalidation stay consistent. Every applied change writes
an `audit` row with actor `factkit`.

## Companion operations

### Pre-insert guard

Check a fact against the store before writing it. Read-only, and it exits 1
whenever the insert should be blocked, so a script can branch on it:

```bash
factkit holo guard --db $DB --text "Raspberry Pi 2 has 4 GB RAM"
# {"verdict": "conflict", "matches": [...], "suggestion": "...", "human": "..."}
```

`duplicate` (>= 0.98 with identical numeric tokens), `conflict` (same named
attribute, different value, unit-normalised), `near` (0.90-0.97, merge the
wording). The guard only sees attributes the text names: "has 16 GB" without
the word RAM is reported `ok`, which is the intended conservative failure.

### Chat approval

Proposals live in `conflict_proposals`; the chat path is a short code per row:

```bash
factkit holo approve --db $DB                          # list pending + rendered text
factkit holo approve --db $DB --code 1a2b --apply      # accept (tombstone path)
factkit holo approve --db $DB --code 1a2b --reject     # status only
```

Accept delegates to `apply_conflict`/`apply_merge` with a fresh `chat-<ts>`
batch, so the tombstone and undo path is unchanged. A second accept returns
`already decided` and writes nothing.

### Age, provenance, prose

```bash
factkit holo decay      --db $DB                  # bands by effective trust, read-only
factkit holo provenance --db $DB                  # hash diff: changed / new / missing
factkit holo provenance --db $DB --apply          # backfill rows for untracked facts
factkit holo prose      --db $DB --apply --limit 10 \
        --llm-url https://llm.local/v1 --llm-model <chat-model>
```

`decay` never rewrites `trust_score`: it reports what age does to it. Provenance
is how provider writes become visible without intercepting them - the hash of
each fact is compared on demand. `prose` queues only `contradiction` verdicts,
with the pair and confidence in the evidence JSON.

### Live retrieval metrics

```bash
factkit holo metrics --db $DB --smoke cases.json
```

Runs hybrid retrieval (BM25 + cosine) over the live store, reports hit@k, MRR
and per-case ranks, and exits 1 when the gate fails. Cases use
`{"query": ..., "expect": [fact_id, ...]}`; unknown ids are reported instead of
being scored as misses.

### Store limits, archive and backups

```bash
factkit holo limits --db $DB            # inspect + plan, writes nothing
factkit holo limits --db $DB --apply    # prune tombstones/audit, then vacuum
```

`--apply` refuses plans that contain archival actions; archiving facts is a
separate, explicit call (`factkit.limits.archive_facts`) that exports JSONL,
re-reads it, compares ids and hashes, and only then deletes inside the same
transaction. Vacuum is checkpointed first and blocked unless free disk is at
least twice the database size.

Backups use SQLite's backup API (not `cp`), force the copy out of WAL mode,
run `PRAGMA integrity_check`, and keep the newest 14:

```bash
~/.hermes/scripts/factkit_backup.py          # silent on success
~/.hermes/scripts/factkit_backup.py --drill  # restore + compare counts + FTS probe
```

Scheduled here: create daily at 02:40, restore drill Sundays at 05:00.

### Time-aware queries

`valid_from`/`valid_to` are additive columns on `facts` (NULL = open-ended):

```bash
factkit holo asof --db $DB --text "which proxy port" --as-of 2026-03-01
```

`as_of` is a **post-filter over `prefetch`** - the provider's own search is not
time-aware, so retired rows can still arrive and are dropped afterwards. The
live store was backfilled once (`valid_from` from `created_at`); `supersede`
closes the old row and opens the new one without deleting either.

### Embedding cache

Query embeddings are cached in `~/.hermes/embed_cache.db` (sqlite, float32,
768 dims ~3 KB/vector, 30-day TTL, 5000 entries). Warm queries drop from
~0.13s to ~0.02s; a cache fault falls back to the endpoint and never fails a
turn. Override the path with `FACTKIT_EMBED_CACHE`.

## Experiments and opt-in paths

None of these run inside the per-turn hook; each is an explicit command, and the
ones with a measured downside say so.

- **Query expansion** (`expand.py`) maps a curated alias set (laptop/notebook,
  memory/ram, stt/transcription...) onto the query, caps the bonus at 0.15 and
  leaves the original string alone when expansion would double its length.
- **Model migration** (`migrate.py`) adds vectors for a second model beside the
  existing rows (`INSERT OR IGNORE`, one transaction per batch, resumable).
  `verify` requires full coverage and no duplicate/dim-mismatched rows; `rollback`
  needs the literal token and refuses to remove the last model.
- **Relations and multi-hop** (`relations.py`, `multihop.py`) add an edge table and
  a BFS walk (decay 0.6 per hop, fanout 5, budget 40, cycle-safe). `infer` writes
  nothing. Multi-hop is for connection questions only - it costs extra reads.
- **HRR** (`hrr.py`) is a candidate re-ranker with a benchmark harness. On the
  current corpus it beats lexical on MRR but not hit@5, so the bench exits 1 and
  nothing is wired into the hook. Wire it only when the A/B wins both.
- **Rerank** (`rerank.py`) has two modes and stays off (`rerank: false`).
  `llm` mode measured ~2.3 s for five hits. `embedding` mode is wired into
  `transform_tool_result` with a 250 ms deadline, but on the live corpus it
  timed out on 25/25 cases (mean 330 ms, prefetch 163 ms), so hit@5 and MRR did
  not move (0.800/0.768). `scripts/rerank_ab.py` exits 1 on no gain: leave it off
  unless you raise the budget and can pay the latency, or precompute candidate
  vectors so the hook only embeds the query.
- **Namespaces** (`namespaces.py`) detects the store shape. The live provider has
  `memory_banks`, but only as a category-level vector cache with no fact->bank
  membership, so namespaces are the `ns:<name>` tag convention. Scoping is a
  post-filter: a small namespace returns fewer than `limit`.

## Operational hardening

- **Hooks never create the store.** The companion checks that the database
  exists before opening it; a missing store disables recall with one warning
  instead of creating an empty database beside the real one.
- **One cached handle per path**, closed on `on_session_end` (twice-safe).
- **Dedupe is per session**: a fact already injected earlier in the same
  conversation is skipped, and the next session starts clean.
- **`health()`** exposes `{recall_failures, embed_failures, breaker_open,
  last_error}` for the plugin; the first failure of each kind logs at WARNING.
- **One vendored-file manifest.** `integrations/hermes/factkit/manifest.txt` is
  read by both `install.sh` and `factkit plugin-status`, so they cannot disagree.
  Dist-owned extras (`update.yaml`, `README.md`, `LICENSE`, config examples) are
  ignored by the drift check; a missing or modified managed file is still DRIFT.
- **Tests do not depend on this host.** The plugin is loaded from
  `integrations/hermes/factkit` into a temp package, CI runs `unit` and `plugin`
  jobs separately, and `tests/test_no_host_paths.py` fails on any `/home/` path.

## Hybrid results

`fact_store` search results are re-ranked in a `transform_tool_result` hook: the
provider's ordering (FTS + HRR + trust) is fused with an embedding channel using
reciprocal rank fusion. Two invariants:

- a provider hit is never dropped or moved below a fact it already ranked over
  unless the embedding channel is decisively stronger;
- half the result slots stay open, so a fact the provider missed can enter.

`factkit plugin-status` proves the installed copy matches the repo; the hook
cannot reorder the provider's own prefetch block, only the tool result.

## Resilience

Every embedding call is capped at 1.5s. After three consecutive failures the
breaker opens for 5 minutes: turns then pay 0ms and lose semantic recall rather
than waiting on a dead endpoint. `~/.hermes/scripts/factkit_prewarm.py` runs on a
10-minute cron to keep the model resident, which removes the ~1.5s cold start.

## The golden rule

Cosine similarity cannot separate a contradiction from a duplicate. Structure can:

```
same subject  +  same attribute  +  different value  →  newer updated_at wins
```

Attributes are named, not just typed: `ram`, `storage`, `cores`, `clock`, `port`,
`ip`, and software versions (`caddy 2.7.6`). Two facts conflict only when the
attribute names match and the values differ — after unit normalisation, so
`16GB` and `16384MB` agree, `3.6GHz` and `3600MHz` agree, and `8080` conflicts
with the range `8000-8100` but not with `9000-9100`.

Subject matching requires a shared *declared* entity or a high content overlap.
A shared city or vendor alone is not a subject — that check produced 37 false
positives on an 82-fact store before it was tightened. A bare `port` or `ip`
disagreement also requires a shared declared subject, since two services may
legitimately use different ports.

## Golden gate

Retrieval quality is measured, not asserted.

```bash
factkit golden-write     # seed golden.json from the corpus, then edit
factkit gate             # PASS/FAIL
factkit gate -v          # per-query ranks and misses
```

| Metric | Default |
|---|---|
| `r_at_5` | 0.85 |
| `mrr` | 0.80 |
| `hit_rate` | 0.90 |
| `fp` | 0 |

A golden file describes one corpus. After an import or a database switch,
regenerate it before treating a failure as a retrieval regression. Retiring a
fact through the review path remaps its cases to the survivor.

## Config

| Variable | Default | Purpose |
|---|---|---|
| `FACTKIT_DB` | `~/.factkit/facts.db` | store path (set to `~/.hermes/memory_store.db` for companion mode) |
| `FACTKIT_EMBED_BASE_URL` | `http://127.0.0.1:1234/v1` | embeddings endpoint |
| `FACTKIT_EMBED_MODEL` | `text-embedding-nomic-embed-text-v1.5` | model id |
| `FACTKIT_EMBED_API_KEY` | empty | bearer token |
| `FACTKIT_EMBED_BATCH` | 32 | inputs per request |
| `FACTKIT_EMBED_TIMEOUT` | 20 | seconds |
| `FACTKIT_LLM_BASE_URL` / `_MODEL` / `_API_KEY` | local server / `local-model` | bridge and router fallback |
| `FACTKIT_TTL_DAYS` | 90 | age horizon |
| `FACTKIT_SLOT_TTL_DAYS` | 180 | volatile-attribute horizon |
| `FACTKIT_DEDUPE_THRESHOLD` | 0.92 | near-duplicate threshold |
| `FACTKIT_RRF_K` | 60 | RRF constant |

Plugin-only keys (`config.yaml`, env overrides in brackets): `prefetch`
[`FACTKIT_PREFETCH`], `limit` [`FACTKIT_PREFETCH_LIMIT`], `min_score`
[`FACTKIT_PREFETCH_MIN`], `db` [`FACTKIT_DB`].

## Standalone extras

These act on factkit's own store unless noted (the dashboard can also serve the
live Hermes store, read-only):

- `factkit dashboard --host 127.0.0.1 --port 8765 --token ...` — metrics plus
  accept/reject queues; bearer token and HttpOnly cookie when `--token` is set.
  Shows the live Hermes store (`~/.hermes/memory_store.db`) read-only by
  default; see [Dashboard database](#dashboard-database).
- `factkit review` — curses queue, `--line` for dumb terminals
- `factkit bridge <file|state.db>` — propose facts from a transcript or a Hermes
  session DB
- `factkit export` / `import` — JSON or CSV, dry-run, duplicate skip
- `factkit import --hermes ~/.hermes/memory_store.db` — read-only snapshot import
- `factkit sync --hermes ~/.hermes/memory_store.db` — mirror changes one way
- `factkit route`, `factkit capture`, `factkit report`, `factkit audit`

## Dashboard database

`factkit dashboard` picks its database explicitly and labels it in the page and
in `/api/summary` (`database.path`, `.mode`, `.readonly`, `.label`):

| setting | values | default |
|---|---|---|
| `--db` / `FACTKIT_DB` | any SQLite path | live default below |
| `FACTKIT_DB_MODE` / `--mode` | `live`, `corpus` | `live` |
| `FACTKIT_READONLY` / `--read-only` | truthy / flag | off (corpus only) |

Defaults: `live` is `~/.hermes/memory_store.db`; `corpus` is
`~/.factkit/facts.db`; an explicit path infers `live` only when it resolves to
the live file. Precedence is `--db` > `FACTKIT_DB` > the mode's default.

The live store is **always read-only**: it is opened with SQLite `mode=ro` and
`PRAGMA query_only=ON`, and the server answers review POSTs with `403`. There
is no flag that opens `~/.hermes/memory_store.db` read-write. Review actions
are available only on a corpus database, or on a custom path passed to `--db`.

```bash
factkit dashboard                                   # live store, read-only
factkit dashboard --mode corpus                     # factkit's own store
factkit dashboard --db ~/scratch/facts.db --read-only
```

## Safety guarantees

- No automatic writes. Detection is read-only; `--apply` is required.
- Deletes go through `remove_fact`, never raw `DELETE` on `facts`.
- Every applied change is audited with before/after content and a reason.
- The provider's retrieval, trust learning, entity index and multilingual
  handling are untouched and still own `fact_store`.

## Limits

- The hook cannot reorder `fact_store` results; it adds a separate recall block.
- Semantic recall needs the embedding endpoint to be up at query time.
- Conflict detection understands named numeric attributes, not free-form prose.
- Multi-hop reasoning, temporal validity and document-level memory are not
  implemented; see `ROADMAP.md`.

## License

MIT
