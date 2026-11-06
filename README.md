# fact-memory-kit

Local-first long-term memory for agents: one SQLite file, hybrid retrieval,
provenance for every change, and a deterministic rule for stale facts.

No vector database, no service to run. Embeddings are optional and come from any
OpenAI-compatible endpoint (Ollama, vLLM, llama.cpp, hosted APIs).

```
facts ──┬── FTS5 (BM25) ─────┐
        └── embeddings ──────┴── reciprocal rank fusion ── hits + metrics
                                  │
        audit ◄── every change ───┤
        conflicts / TTL ──────────┘  proposals only, never silent deletes
```

## What it does

| Feature | Command | Notes |
|---|---|---|
| Hybrid retrieval | `factkit search "..."` | BM25 + cosine, fused with RRF; degrades to lexical when the embedding service is down |
| Golden gate | `factkit gate` | 28-case style regression suite with thresholds; fails the build on a drop |
| Golden rule | `factkit conflicts scan` | same subject + same slot + different value → newer `updated_at` wins |
| TTL flags | `factkit ttl scan` | age and slot-rot flags, never deletions |
| Review queue | `factkit review` | full-screen curses UI (line mode with `--line`) for conflicts, proposals and TTL flags |
| Audit trail | `factkit audit` | append-only, before/after JSON on every mutation |
| Proactive capture | `factkit capture "..."` | scores whether a message is worth storing, with reasons |
| Session → facts | `factkit bridge session.jsonl` | opt-in LLM extraction, deduped, parked as proposals; reads Hermes `state.db` directly |
| Query router | `factkit route "..."` | facts / wiki / live / external / mixed + the next command to run |
| Bulk ops | `factkit export` / `import` | JSON or CSV, dry-run, duplicate skip, golden validation |
| Metrics | `factkit report` / `dashboard` | p50/p95 latency, hit rate, coverage, backlog, HTML dashboard |
| Entity linking | `factkit entity relink` | deterministic gazetteer + proper-noun extraction |

## Install

```bash
git clone https://github.com/MohamedElsiddig/fact-memory-kit.git
cd fact-memory-kit
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
```

Python 3.10+. The only dependency is the standard library (`numpy` optional and
detected automatically).

## Quickstart

```bash
factkit init --demo                 # creates ~/.factkit/facts.db + sample facts
factkit search "atlas server specs"
factkit add "The proxy listens on node-b at 192.0.2.20" -c tool -t proxy
factkit report
```

Point it at an embedding server and index:

```bash
export FACTKIT_EMBED_BASE_URL=http://127.0.0.1:1234/v1
export FACTKIT_EMBED_MODEL=text-embedding-nomic-embed-text-v1.5
factkit embed            # backfills, then runs the golden gate
```

With an API key (hosted or token-protected endpoint):

```bash
export FACTKIT_EMBED_API_KEY=sk-...
```

## Golden gate

Retrieval quality is not "it feels better" — it is measured.

```bash
factkit golden-write     # seed golden.json from the current corpus, then edit
factkit gate             # PASS/FAIL against thresholds
factkit gate -v          # per-query ranks and misses
```

Default thresholds (`golden.json` → `thresholds`):

| Metric | Default | Meaning |
|---|---|---|
| `r_at_5` | 0.85 | first expected fact lands in the top 5 |
| `mrr` | 0.80 | mean reciprocal rank |
| `hit_rate` | 0.90 | positive queries returning anything |
| `fp` | 0 | negatives that returned a hit |

`factkit embed` runs the gate after indexing, and `factkit gate` exits non-zero
on regression, so CI catches a bad retrieval change. When the golden rule retires
a fact, the gate remaps its cases to the survivor so the suite does not silently
rot (regenerate with `golden-write` if you retire facts by hand).

## The golden rule

Cosine similarity cannot tell a contradiction from a duplicate (published
memory-conflict work puts it near chance). Structure can:

```
same subject  +  same value slot  +  different value  →  newer updated_at wins
```

Slots are IPs, ports, versions, dates, paths, model names, hostnames and counts.
Subjects must match by entity set — so "node-a has 16 GB" and "node-b has 1 GB" are
not a conflict. Identical timestamps tie-break on the later insert.

```bash
factkit conflicts scan          # propose; --gate exits 1 when any exist
factkit conflicts list --json
factkit review                  # accept / reject / skip
```

Nothing is applied automatically. Accepting writes an audit row and bumps the
survivor's trust.

## Config

Every knob is an environment variable; all have offline-safe defaults.

| Variable | Default | Purpose |
|---|---|---|
| `FACTKIT_DB` | `~/.factkit/facts.db` | database file |
| `FACTKIT_EMBED_BASE_URL` | `http://127.0.0.1:1234/v1` | OpenAI-compatible embeddings |
| `FACTKIT_EMBED_MODEL` | `text-embedding-nomic-embed-text-v1.5` | model id |
| `FACTKIT_EMBED_API_KEY` | empty | bearer token |
| `FACTKIT_EMBED_BATCH` | 32 | inputs per request |
| `FACTKIT_EMBED_TIMEOUT` | 20 | seconds |
| `FACTKIT_LLM_BASE_URL` / `_MODEL` / `_API_KEY` | your server / `local-model` | bridge + router fallback |
| `FACTKIT_TTL_DAYS` | 90 | age flag horizon |
| `FACTKIT_SLOT_TTL_DAYS` | 180 | volatile-slot horizon |
| `FACTKIT_DEDUPE_THRESHOLD` | 0.92 | cosine dedupe for proposals |
| `FACTKIT_RRF_K` | 60 | RRF constant |

Embeddings are stored per model in `fact_embeddings(fact_id, model, dims, vec)`,
so you can index a second model beside the first and switch without re-writing
facts.

## Dashboard

```bash
factkit dashboard --host 127.0.0.1 --port 8765 --token "$(openssl rand -hex 16)"
```

Metrics plus one-click accept/reject for every queue. Loopback by default.
With `--token` (or `FACTKIT_DASHBOARD_TOKEN`) every route requires a bearer token;
open `http://host:8765/?token=...` once and the browser keeps an HttpOnly cookie.

## Use as a library

```python
from factkit import Config, Store, HybridSearch, EmbeddingClient

cfg = Config.load()
store = Store(cfg.db_path)
search = HybridSearch(store, EmbeddingClient(
    cfg.embed_base_url, cfg.embed_model, cfg.embed_api_key))

for hit in search.search("what port does transcriber use", k=5):
    print(hit.fact_id, round(hit.score, 4), hit.content)
```

## Migrating an existing store

If you already have a facts table (any schema with `fact_id`/`content`), export it
to JSON with `id`, `content`, `category`, `tags`, `trust_score`, `updated_at`,
then:

```bash
factkit import old_facts.json --dry-run
factkit import old_facts.json
factkit entity relink
factkit embed
factkit golden-write        # build a labelled set from the imported corpus
```

Coming from a Hermes agent? Import its store directly — the source is opened
read-only, timestamps and entity links are preserved:

```bash
factkit import --hermes ~/.hermes/memory_store.db --dry-run
factkit import --hermes ~/.hermes/memory_store.db
factkit embed
```

The importer keeps original timestamps out of the write path, so the golden rule
will prefer newly-confirmed facts; re-add anything you know is current.

## Design notes

* Facts are append-only in spirit: mutations are recorded in `audit`, and
  automation only ever writes *proposals*.
* FTS5 uses an external-content table with triggers, so BM25 never drifts from
  the base table.
* Vectors are float32 blobs; cosine is pure Python unless numpy is importable.
* The store is one file; copy it for a backup, WAL files included.

## License

MIT
