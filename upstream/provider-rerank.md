# Provider-native rerank: an optional `rerank` hook and a wider candidate window

**Status:** proposal + patch, not applied. Target file: `retrieval.py` in the
`holographic` provider (the installed copy is read-only for us; plugin updates
overwrite in-place edits, so this ships as a diff, not an edit).

**Summary:** `FactRetriever._rerank` scores every candidate the retrieval
channels return — each channel capped at `limit * 3` — and then throws away
everything past `limit`. No hook — ours or anyone else's — can see those
candidates again. This patch adds two optional constructor arguments:

- `rerank`: a callable invoked once per query, after scoring and **before** the
  final slice, over every scored candidate;
- `candidate_window`: how many candidates each retrieval channel may contribute
  (default: today's `limit * 3`).

Both default to today's behaviour exactly. With no hook the patch is inert.

---

## 1. Problem statement

Candidates are generated per channel at `limit * 3`, scored by `_rerank`, then
truncated by a single slice:

| Stage | Where | What it decides |
|---|---|---|
| generation ceiling | `search()`, the three `limit * 3` arguments | how many rows each of FTS, n-gram and word-scan may return |
| scoring | `_rerank()`, the loop over `candidates` | `score = relevance * trust_score`, then optional decay |
| **truncation** | `_rerank()`, `sorted(...)[:limit]` | **which `limit` rows ever leave the retriever** |

That last line is the whole problem. It is where *scored* candidates become
invisible, and it runs before any caller or hook is reached. (The generation
ceiling above it is where unscored candidates never exist at all; widening it
is the second half of this patch.) An external reranker attached to the tool
result therefore receives `limit` rows: it may reorder the provider's
shortlist, but it cannot promote a candidate the provider ranked `limit + 1`,
even when the provider already paid to score it. The provider computes two
thirds of a ranking and discards it.

The same ceiling applies to `probe` and `related` (they fall back to `search`)
and to `reason`'s unknown-entity path (it calls `_rerank` directly). The
entity-index paths do **not** pass through `_rerank` at all: `probe`/`reason`
with a known entity call `_linked_facts`, which orders by trust and never
scores. A hook placed in `_rerank` inherits exactly the coverage `_rerank`
already has — no more. Covering `_linked_facts` is a separate, larger change
(see Open questions).

### Why the current workaround is expensive

`fuse_results` compensates for the narrow page by running a *second* retrieval
over the whole store (`prefetch(limit=1_000_000)`) and fusing the two orderings
with reciprocal rank fusion. That works, but it pays for two retrievals per
query and can only fuse at page granularity. Reranking candidates the provider
has already scored and ranked is strictly cheaper than re-retrieving them.

---

## 2. Measured evidence

### 2.1 Live store (82 facts, 10 queries, limit 5 and 10)

Instrumented `_rerank` on a copy of the live database:

| metric | value |
|---|---|
| candidates entering `_rerank` | min 1, max 6, mean 2.7 |
| query/limit pairs where the window was saturated (`= limit * 3`) | 0 / 20 |
| query/limit pairs where candidates were cut | 1 / 20 |
| candidates cut overall | 1 / 54 (1.9%) |

Honest reading: **at 82 facts the truncation is not yet costing us results.**
The window is not saturated, so the ceiling is structural rather than active.
This is exactly why the change must default to today's behaviour: there is no
quality evidence to justify widening it for everyone.

### 2.2 Synthetic store (401 facts, one broad query matching all of them)

`limit` varied, `search()` timed on the same host:

| limit | candidates scored | returned | candidates never scored | hidden |
|---|---|---|---|---|
| 5 | 15 | 5 | 386 | 96.3% |
| 10 | 30 | 10 | 371 | 92.5% |
| 20 | 60 | 20 | 341 | 85.0% |

Full `search()` cost at these sizes: 6–10 ms, so widening the window to 60
candidates is cheap for lexical scoring (it is not free for a cross-encoder —
see §5).

Caveat, stated up front: in that corpus the planted answer still surfaced,
because its lexical signal was distinctive enough to rank inside the window.
Truncation hurts when the lexical rank is deep and the real signal is semantic
— which is the case the embedding channel exists for.

### 2.3 Fusion ordering and reserve slots (the workaround we measured)

`fuse_results` fuses provider order with an embedding channel:

- every provider row is kept (`by_id`), scored `1/(60 + i) + 0.5 * 1/(60 + semantic_rank)`;
- `limit // 2` slots are reserved for facts the provider missed entirely;
- those reserve slots are gated to the embedding channel's **top 5**
  (`cutoff = min(len(hits), 5)`), because a global cosine floor does not
  separate signal from noise for this embedding model.

Two consequences for this proposal:

1. The reserve path can only rescue a fact the *embedding* channel put in its
   top 5. A provider candidate ranked `limit + 1` by lexical score and 6th by
   embedding is reachable by neither path.
2. Provider hits keep priority, so fusion can reorder the page but cannot
   change *which* provider candidates were on it. The hook closes that gap
   without touching the invariants `fuse_results` already guarantees.

---

## 3. Two alternatives

### A. `rerank` hook + `candidate_window` (this patch)

The retriever calls an optional callable after scoring, before truncation, with
the full candidate list. The caller decides whether to spend latency; the
provider decides how wide the list is.

- **Pro:** uses candidates already scored (no second retrieval); one extension
  point covers `search`, `probe`, `related` and `reason`'s lexical path;
  opt-in per construction; a failing hook degrades to today's ordering.
- **Con:** a synchronous hook cannot be preempted (see §5); a caller that
  passes a slow hook owns the latency; `_linked_facts` still bypasses it.

### B. Raise the candidate window only

Add `candidate_window` and stop there. Cheaper to review, but on its own it is
**unobservable**: `_rerank` still returns `[:limit]`, so a wider window buys
extra scoring work and the same answer. It only becomes useful once something
consumes the wider list — i.e. it is a strict subset of A. Rejected as a
standalone change; kept inside A because the hook needs a window worth reading.

### Rejected: post-hoc re-sort in `transform_tool_result`

Already implemented in factkit and measured. It can only reorder `limit` rows;
promoting a dropped candidate requires re-retrieving it, which is what the
embedding channel does at `limit=1_000_000`. Correct, but it pays twice.

### Rejected: hard timeout via a worker thread

A watchdog thread that returns on timeout leaves the hook running, holding an
HTTP connection and possibly a lock, with no way to cancel it. One leaked
thread per slow call is worse than a logged overrun. See §5.

---

## 4. API sketch

```python
FactRetriever(
    store,
    rerank=None,                 # Callable[[str, list[dict], int], list[dict] | None]
    candidate_window=None,       # None => limit * 3, today's behaviour
    rerank_timeout_ms=250,       # soft budget: logged, never enforced
)
```

```python
def my_rerank(query: str, candidates: list[dict], limit: int) -> list[dict] | None:
    """candidates: copies of every scored row, best first, each with fact_id,
    content, score, trust_score. Return the rows you want, best first, or None
    to keep the provider order."""
    ranked = cross_encoder.score(query, [c["content"] for c in candidates])
    return [c for _, c in sorted(zip(ranked, candidates), reverse=True)][:limit]
```

Contract, as implemented in the patch:

- called once per `search()` (and per direct `_rerank` call), after `score` is
  set and after temporal decay, **before** `[:limit]`;
- receives **copies** of the rows, so mutating them cannot change the result;
- may return fewer rows than `limit` — the provider order backfills the rest;
- `None` or a non-list return keeps the provider order unchanged;
- ids not in the candidate set are dropped; duplicates and non-dicts ignored;
- any exception is caught and logged at WARNING; the search still returns;
- the rows returned are the provider's own, so `score` and `trust_score`
  cannot be forged by the hook;
- a call slower than `rerank_timeout_ms` logs a warning and is otherwise
  allowed to finish.

Wiring on the caller side (factkit, illustrative): construct the retriever with
`rerank=provider_rerank` and `candidate_window=max(60, limit * 3)`, and keep
`fuse_results` for the tool result until the golden gate shows the hook wins.
A remote reranker would be reached at something like
`http://192.0.2.10:8080/v1/rerank`; the example address is documentation-only
(RFC 5737 test space), not a deployed endpoint.

---

## 5. Compatibility

- **Defaults are inert.** `rerank=None` and `candidate_window=None` reproduce
  the old code path; verified by running the patched and unpatched modules over
  the same store: 21/21 query × limit pairs return identical ids and scores.
- **Signature is additive.** New parameters are appended after `hrr_dim`, so
  existing positional calls keep working. No new imports beyond `time` and
  `Callable`, no new dependency, no schema change, no migration.
- **Syntax** matches the module's existing `X | None` style, so the
  `requires-python >= 3.10` floor is unchanged.
- **`_rerank`'s signature is unchanged.** Subclasses that override it keep
  working but lose the hook (the call site is inside the base implementation).
  Documented rather than worked around.
- **Cost when unused:** one `is not None` check per query.
- **`candidate_window` is clamped** to `max(limit, window)`, so a window
  smaller than `limit` can never starve a result set.
- **Entity paths bypass the hook** (`_linked_facts`), so behaviour there is
  bit-for-bit unchanged.

---

## 6. Failure modes

| failure | behaviour | cost |
|---|---|---|
| hook raises | caught, logged, provider order returned | one WARNING line |
| hook returns `None` | provider order returned | none |
| hook returns non-list | logged, provider order returned | one WARNING |
| hook returns junk ids / duplicates | dropped/deduped, backfilled to `limit` | none |
| hook returns an empty list | provider order backfills | none |
| hook overruns the budget | completes, WARNING with elapsed ms | latency paid this turn |
| rerank endpoint down | hook's own error path; if it propagates, provider catches it | one failed request per turn until the caller's breaker opens (factkit: 3 failures → 5 min off) |
| hook returns NaN/inf scores | scores are never read back; provider rows win | none |
| hook mutates rows | no effect (it gets copies) | one dict copy per candidate |

### Timeout

`rerank_timeout_ms` is a **contract plus observability**, not a kill switch.
A synchronous call inside the retriever cannot be preempted safely: killing a
thread mid-HTTP leaves the connection and any store lock held. The provider
therefore logs the overrun and lets the call finish; the caller must enforce
its own deadline (`time.monotonic()` at entry, return `None` when exceeded).
A hard timeout is possible later with `concurrent.futures`, but only if the
maintainers accept thread leakage as the failure mode.

### Latency budget

The budget we can defend from measurement:

- 60 candidates, lexical scoring: 10 ms total search on a 401-fact store;
- cross-encoder on CPU: roughly 10–30 ms per candidate, i.e. 0.6–1.8 s for 60 —
  **over any interactive budget**. A cross-encoder hook must either cap the
  window at ~20 (0.2–0.6 s, still marginal) or run on a resident GPU service;
- the existing `transform_tool_result` hook budget is 250 ms and the embedding
  call is capped at 1.5 s, so a default `rerank_timeout_ms=250` matches the
  budget the caller already has.

Recommendation: default the hook off; let the caller choose 250 ms with a
remote reranker or 1 s with a warmed local one, and record the elapsed time in
the ranking metadata so the golden gate can attribute regressions.

---

## 7. Rollback plan

The patch is deliberately reversible at every layer:

1. **No state.** No schema change, no persisted config, no cache key. Reverting
   is code-only; no data to migrate back.
2. **Provider default is off.** With no `rerank` argument the behaviour is
   today's, so "rollback" for a caller is passing `rerank=None` (or setting the
   caller's env flag, `FACTKIT_PROVIDER_RERANK=0`, in our integration).
3. **Reverse the diff.** `patch -R -p1 < upstream/rerank.patch` restores the
   original file byte-for-byte; the patch touches one file and deletes nothing.
4. **Do not patch in place.** The installed provider is replaced on plugin
   update. Ship the change upstream and pin the provider version; if a release
   regresses, reinstall the previous version rather than editing the install.
5. **Quality gate.** Run the golden set (`factkit gate`) and MRR/hit@k
   (`live_metrics`) before and after. The existing batch-revert path restores
   the previous batch if retrieval regresses.
6. **Kill criteria.** If MRR does not improve on the golden set, or p95
   `search()` grows by more than the 250 ms budget, do not enable the hook.

---

## 8. Verification performed for this proposal

The patch was exercised against a scratch copy of the installed module (never
the installed file itself), on a temporary store:

| check | result |
|---|---|
| no hook: ids and scores vs unpatched module | identical on 21/21 query × limit pairs |
| `candidate_window` widening | 3 candidates (default) → 8 (window 9) |
| hook reorder | order changes, scores and rows preserved |
| hook raises | provider order kept, WARNING logged |
| hook returns a string | provider order kept, WARNING logged |
| hook returns duplicates, unknown ids, non-dicts | dropped, backfilled to `limit` |
| hook returns `None` / `[]` | provider order kept / backfilled |

`tests/test_upstream_patch.py` guards the artifact offline: the diff parses,
touches only `retrieval.py`, keeps every function it mentions, stays under 120
changed lines, and the hunks still match the installed module's line numbers.
The suite parses the patch; it never applies it.

## 9. Open questions

1. Should `_linked_facts` (entity-index `probe`/`reason`) also offer the hook?
   It never scores today, so a reranker there would need a score to reason
   about — a larger change than this patch.
2. Is an absolute `candidate_window` the right shape, or a multiplier of
   `limit`? Absolute is simpler to reason about for a caller that knows its
   reranker's cost; a multiplier tracks `limit` automatically. This patch
   chose absolute with a floor of `limit`.
3. Do the maintainers want a hard timeout (thread) or is contract +
   observability enough? Evidence so far says the latter.
