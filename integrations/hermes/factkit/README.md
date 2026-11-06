# Hermes companion plugin

Runs alongside the `holographic` memory provider — same database, no fork, no
changes to the provider. Adds:

- a semantic-recall block injected before each model call (`pre_llm_call`)
- embedding of facts written during a session (`on_session_end`)

Writes are never automatic. Conflict and TTL actions run through the provider's
own `MemoryStore` API via `factkit holo ...`.

## Install

```bash
bash integrations/hermes/factkit/install.sh
hermes plugins enable factkit
```

`install.sh` copies the plugin into `~/.hermes/plugins/factkit` and vendors the
five factkit modules it needs. Re-run it after updating the repo.

## Configure

```bash
cp ~/.hermes/plugins/factkit/config.example.yaml ~/.hermes/plugins/factkit/config.yaml
chmod 600 ~/.hermes/plugins/factkit/config.yaml
```

| key | default | meaning |
|---|---|---|
| `embed_base_url` | `http://127.0.0.1:1234/v1` | OpenAI-compatible embeddings endpoint |
| `embed_api_key` | `""` | bearer token, if required |
| `embed_model` | `text-embedding-nomic-embed-text-v1.5` | must match the model used by `factkit embed` |
| `prefetch` | `true` | inject the semantic block |
| `limit` | `3` | max facts injected per turn |
| `min_score` | `0.55` | minimum cosine similarity |
| `db` | `~/.hermes/memory_store.db` | store path |

Environment variables override the file (`FACTKIT_EMBED_BASE_URL`,
`FACTKIT_EMBED_API_KEY`, `FACTKIT_EMBED_MODEL`, `FACTKIT_DB`,
`FACTKIT_PREFETCH`, `FACTKIT_PREFETCH_LIMIT`, `FACTKIT_PREFETCH_MIN`).

## Index the existing store

```bash
factkit holo conflicts --db ~/.hermes/memory_store.db   # report only
factkit holo stale     --db ~/.hermes/memory_store.db
factkit holo dupes     --db ~/.hermes/memory_store.db
factkit prefetch "how much memory does the pi have"    # what the hook injects
```

Embeddings must exist before the hook returns anything:

```bash
FACTKIT_DB=~/.hermes/memory_store.db factkit embed
```

## Verify

```bash
hermes plugins list | grep factkit          # enabled
factkit prefetch "what car do I drive"      # non-empty
```

## Notes

- The hook fails soft: endpoint down, missing table or bad config returns no
  context rather than breaking a turn.
- Injected facts are labelled as data, never as instructions.
- `factkit holo ... --apply` retires facts through `remove_fact`, so FTS, HRR
  vectors, entity links and trust stay consistent.
