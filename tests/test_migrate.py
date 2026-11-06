"""Offline tests for per-model embedding migration.

No endpoint and no real data: ``FakeClient`` stands in for the model and every
fixture is synthetic (Atlas, Beacon, 192.0.2.x). Stores live under ``tmp_path``;
the Hermes memory store is never touched.
"""
from __future__ import annotations

import pytest

from factkit import vectors
from factkit.migrate import CONFIRM_TOKEN, plan, rollback, run, verify
from factkit.store import Store

MODEL = "text-embedding-3-small"
OLD_MODEL = "text-embedding-nomic-embed-text-v1.5"
DIMS = 16


# ------------------------------------------------------------------ helpers
def fake_vector(text: str, dims: int = DIMS) -> list[float]:
    """Deterministic per-text vector; values are exact in float32."""
    seed = sum(ord(ch) for ch in text) or 1
    return [((seed + i) % 97) / 128.0 for i in range(dims)]


class FakeClient:
    """Counts one ``embed`` call per batch; can fail chosen batches."""

    def __init__(self, dims: int = DIMS, fail_on: tuple[int, ...] = ()):
        self.dims = dims
        self.fail_on = set(fail_on)
        self.calls: list[list[str]] = []

    def embed(self, texts):
        texts = list(texts)
        index = len(self.calls)
        self.calls.append(texts)
        if index in self.fail_on:
            raise RuntimeError(f"endpoint blew up on batch {index}")
        return [fake_vector(t, self.dims) for t in texts]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "facts.db")
    try:
        yield s
    finally:
        s.close()


def add_facts(store, n: int, prefix: str = "Atlas") -> list[int]:
    return [store.add_fact(f"{prefix} node {i} sits at 192.0.2.{i + 10}",
                           category="entity")["fact_id"]
            for i in range(n)]


def seed_vectors(store, fact_ids, model: str = OLD_MODEL, dims: int = 8) -> None:
    for fact_id in fact_ids:
        fact = store.get(fact_id)
        store.put_embedding(fact_id, model, fake_vector(fact["content"], dims))
    store.db.commit()


def target_rows(store, model: str = MODEL):
    return list(store.db.execute(
        "SELECT fact_id, model, dims, vec FROM fact_embeddings "
        "WHERE model=? ORDER BY fact_id", (model,)))


def raw_rows(store, model: str = OLD_MODEL):
    """Byte-level snapshot, for proving other models are untouched."""
    return [(r[0], r[1], r[2], bytes(r[3])) for r in store.db.execute(
        "SELECT fact_id, model, dims, vec FROM fact_embeddings "
        "WHERE model=? ORDER BY fact_id", (model,))]


def make_pkless(store) -> None:
    """Rebuild fact_embeddings without its primary key, so duplicates can exist."""
    store.db.executescript(
        """
        DROP INDEX IF EXISTS idx_emb_model;
        ALTER TABLE fact_embeddings RENAME TO fe_pk;
        CREATE TABLE fact_embeddings (
            fact_id    INTEGER NOT NULL,
            model      TEXT NOT NULL,
            dims       INTEGER NOT NULL,
            vec        BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO fact_embeddings(fact_id, model, dims, vec, created_at)
            SELECT fact_id, model, dims, vec, created_at FROM fe_pk;
        DROP TABLE fe_pk;
        """)
    store.db.commit()


# --------------------------------------------------------------------- plan
def test_plan_empty_store(store):
    result = plan(store, MODEL)
    assert result["total_facts"] == 0
    assert result["missing"] == 0
    assert result["present"] == 0
    assert result["api_calls"] == 0
    assert result["estimated_s"] == 0.0
    assert result["blocked"] == []
    assert result["ready"] is True
    assert result["models"] == {}


def test_plan_counts_only_missing_facts(store):
    ids = add_facts(store, 5)
    seed_vectors(store, ids[1:3], model=OLD_MODEL)
    seed_vectors(store, ids[3:5], model=MODEL, dims=DIMS)

    result = plan(store, MODEL)

    assert result["total_facts"] == 5
    assert result["missing"] == 3
    assert result["present"] == 2
    assert result["counts_by_model"] == {OLD_MODEL: 2, MODEL: 2}
    assert result["models"][MODEL]["dims"] == [DIMS]
    assert result["blocked"] == []


def test_plan_estimates_calls_and_seconds(store):
    add_facts(store, 16)

    default = plan(store, MODEL)
    assert default["batch"] == 64
    assert default["api_calls"] == 1
    assert default["estimated_s"] == 2.0        # 16 / 8 per second

    small = plan(store, MODEL, batch=5)
    assert small["api_calls"] == 4              # ceil(16 / 5)

    slow = plan(store, MODEL, rate=4)
    assert slow["estimated_s"] == 4.0


def test_plan_is_read_only(store):
    add_facts(store, 4)
    before_changes = store.db.total_changes
    before = raw_rows(store)

    plan(store, MODEL)
    plan(store, MODEL)
    verify(store, MODEL)

    assert store.db.total_changes == before_changes
    assert target_rows(store) == []
    assert raw_rows(store) == before


def test_plan_reports_blank_content_as_blocked(store):
    add_facts(store, 2)
    store.db.execute("INSERT INTO facts(content) VALUES ('   ')")
    store.db.commit()

    result = plan(store, MODEL)

    assert result["missing"] == 3
    assert result["ready"] is False
    assert any("empty content" in reason for reason in result["blocked"])


def test_plan_blocks_on_unusable_existing_rows(store):
    ids = add_facts(store, 3)
    seed_vectors(store, ids, model=MODEL, dims=DIMS)
    store.db.execute(
        "UPDATE fact_embeddings SET dims = ? WHERE fact_id = ? AND model = ?",
        (DIMS - 1, ids[1], MODEL))
    store.db.commit()

    result = plan(store, MODEL)

    assert result["missing"] == 0
    assert result["models"][MODEL]["unusable"] == 1
    assert result["ready"] is False
    assert any("unusable" in reason for reason in result["blocked"])


def test_plan_rejects_bad_batch_and_rate(store):
    with pytest.raises(ValueError):
        plan(store, MODEL, batch=0)
    with pytest.raises(ValueError):
        plan(store, MODEL, rate=0)


# ---------------------------------------------------------------------- run
def test_run_inserts_exactly_the_missing_rows(store):
    ids = add_facts(store, 4)
    seed_vectors(store, ids[:2], model=OLD_MODEL, dims=8)   # old model only
    seed_vectors(store, ids[2:3], model=MODEL, dims=DIMS)   # already migrated
    already = target_rows(store)
    client = FakeClient()

    result = run(store, client, MODEL, batch=2, dry_run=False)

    assert result["planned"] == 3
    assert result["embedded"] == 3
    assert result["failed"] == 0
    assert result["skipped"] == 0
    assert {row[0] for row in target_rows(store)} == set(ids)
    assert tuple(target_rows(store)[2]) == tuple(already[0])  # untouched
    assert [len(call) for call in client.calls] == [2, 1]
    assert raw_rows(store) != []                # the old model survived


def test_run_is_idempotent(store):
    add_facts(store, 4)
    client = FakeClient()

    first = run(store, client, MODEL, batch=3, dry_run=False)
    snapshot = target_rows(store)
    second = run(store, client, MODEL, batch=3, dry_run=False)

    assert first["embedded"] == 4
    assert second["planned"] == 0
    assert second["embedded"] == 0
    assert second["skipped"] == 0
    assert len(client.calls) == 2               # second run called nobody
    assert target_rows(store) == snapshot


def test_dry_run_makes_no_client_calls_and_no_writes(store):
    add_facts(store, 5)
    client = FakeClient()

    result = run(store, client, MODEL, batch=2, dry_run=True)

    assert client.calls == []
    assert target_rows(store) == []
    assert result["planned"] == 5
    assert result["embedded"] == 0
    assert result["skipped"] == 0
    assert result["dims"] is None
    assert result["dry_run"] is True


def test_batching_respects_batch_size(store):
    add_facts(store, 7)
    client = FakeClient()

    result = run(store, client, MODEL, batch=3, dry_run=False)

    assert [len(call) for call in client.calls] == [3, 3, 1]
    assert result["embedded"] == 7


def test_limit_caps_the_work(store):
    add_facts(store, 5)
    client = FakeClient()

    result = run(store, client, MODEL, batch=4, limit=2, dry_run=False)

    assert result["planned"] == 2
    assert result["embedded"] == 2
    assert len(target_rows(store)) == 2


def test_failed_batch_is_reported_without_aborting_the_rest(store):
    ids = add_facts(store, 6)
    client = FakeClient(fail_on=(0,))

    result = run(store, client, MODEL, batch=2, dry_run=False)

    assert len(client.calls) == 3               # all three batches attempted
    assert result["failed"] == 2
    assert result["embedded"] == 4
    assert result["ok"] is False
    assert len(result["failures"]) == 1
    assert set(result["failures"][0]["fact_ids"]) == set(ids[:2])
    assert {row[0] for row in target_rows(store)} == set(ids[2:])


def test_blank_content_is_not_sent_to_the_endpoint(store):
    add_facts(store, 2)
    store.db.execute("INSERT INTO facts(content) VALUES ('   ')")
    store.db.commit()
    client = FakeClient()

    result = run(store, client, MODEL, batch=8, dry_run=False)

    assert result["planned"] == 3
    assert result["embedded"] == 2
    assert result["failed"] == 1
    assert result["failures"][0]["error"] == "empty content"
    assert all(text.strip() for call in client.calls for text in call)


def test_run_resumes_after_a_partial_run(store):
    add_facts(store, 5)
    client = FakeClient()

    first = run(store, client, MODEL, batch=2, limit=2, dry_run=False)
    second = run(store, client, MODEL, batch=2, dry_run=False)

    assert first["embedded"] == 2
    assert second["planned"] == 3
    assert second["embedded"] == 3
    assert len(target_rows(store)) == 5
    assert len({row[0] for row in target_rows(store)}) == 5


def test_dims_come_from_the_returned_vectors(store):
    add_facts(store, 3)
    client = FakeClient(dims=12)

    result = run(store, client, MODEL, batch=2, dry_run=False)

    assert result["dims"] == 12
    assert {row[2] for row in target_rows(store)} == {12}
    # 12 float32 values per row, and verify agrees
    assert {len(row[3]) for row in target_rows(store)} == {12 * 4}
    assert verify(store, MODEL)["ok"] is True


def test_progress_callback_runs_once_per_batch(store):
    add_facts(store, 5)
    client = FakeClient()
    events: list[dict] = []

    result = run(store, client, MODEL, batch=2, dry_run=False,
                 on_progress=events.append)

    assert len(events) == 3
    assert [e["batch"] for e in events] == [1, 2, 3]
    assert events[-1]["processed"] == result["planned"] == 5
    assert events[-1]["embedded"] == 5
    assert all(e["batches"] == 3 for e in events)


def test_progress_callback_error_does_not_break_the_run(store):
    add_facts(store, 2)

    def explode(_event):
        raise RuntimeError("UI died")

    result = run(store, FakeClient(), MODEL, batch=1, dry_run=False,
                 on_progress=explode)
    assert result["embedded"] == 2


def test_run_rejects_bad_arguments(store):
    add_facts(store, 1)
    with pytest.raises(ValueError):
        run(store, FakeClient(), MODEL, batch=0, dry_run=True)
    with pytest.raises(ValueError):
        run(store, FakeClient(), MODEL, limit=-1, dry_run=True)


# ------------------------------------------------------------------- verify
def test_verify_ok(store):
    ids = add_facts(store, 3)
    seed_vectors(store, ids, model=MODEL, dims=DIMS)

    result = verify(store, MODEL)

    assert result["total_facts"] == 3
    assert result["covered"] == 3
    assert result["coverage"] == 1.0
    assert result["dim_mismatches"] == 0
    assert result["duplicates"] == 0
    assert result["ok"] is True


def test_verify_reports_partial_coverage(store):
    ids = add_facts(store, 4)
    seed_vectors(store, ids[:3], model=MODEL, dims=DIMS)

    result = verify(store, MODEL)

    assert result["coverage"] == pytest.approx(0.75)
    assert result["missing"] == 1
    assert result["ok"] is False


def test_verify_counts_dim_mismatches(store):
    ids = add_facts(store, 2)
    seed_vectors(store, ids, model=MODEL, dims=DIMS)
    # the dims column lies about the BLOB length
    store.db.execute(
        "UPDATE fact_embeddings SET dims = ? WHERE fact_id = ? AND model = ?",
        (DIMS - 1, ids[0], MODEL))
    store.db.commit()

    result = verify(store, MODEL)

    assert result["dim_mismatches"] == 1
    assert result["ok"] is False


def test_verify_detects_mixed_dims(store):
    ids = add_facts(store, 3)
    seed_vectors(store, ids[:2], model=MODEL, dims=DIMS)
    seed_vectors(store, ids[2:], model=MODEL, dims=DIMS * 2)

    result = verify(store, MODEL)

    assert result["dim_mismatches"] == 1
    assert result["dims"] == [DIMS, DIMS * 2]
    assert result["ok"] is False


def test_verify_counts_duplicate_rows(store):
    ids = add_facts(store, 2)
    make_pkless(store)
    vec = vectors.pack(fake_vector("dup", DIMS))
    for fact_id in ids:
        store.db.execute(
            "INSERT INTO fact_embeddings(fact_id, model, dims, vec) "
            "VALUES (?,?,?,?)", (fact_id, MODEL, DIMS, vec))
    store.db.execute(
        "INSERT INTO fact_embeddings(fact_id, model, dims, vec) "
        "VALUES (?,?,?,?)", (ids[0], MODEL, DIMS, vec))
    store.db.commit()

    result = verify(store, MODEL)

    assert result["rows"] == 3
    assert result["duplicates"] == 1
    assert result["coverage"] == 1.0
    assert result["ok"] is False


# ----------------------------------------------------------------- rollback
def test_rollback_requires_the_literal_token(store):
    ids = add_facts(store, 2)
    seed_vectors(store, ids, model=MODEL, dims=DIMS)

    with pytest.raises(TypeError):
        rollback(store, MODEL)                       # keyword-only, no default
    with pytest.raises(ValueError):
        rollback(store, MODEL, confirm_token="please")
    assert len(target_rows(store)) == 2              # refused, nothing deleted


def test_rollback_refuses_to_remove_the_last_model(store):
    ids = add_facts(store, 2)
    seed_vectors(store, ids, model=MODEL, dims=DIMS)

    with pytest.raises(ValueError) as excinfo:
        rollback(store, MODEL, confirm_token=CONFIRM_TOKEN)
    assert "only model" in str(excinfo.value)
    assert len(target_rows(store)) == 2

    forced = rollback(store, MODEL, confirm_token=CONFIRM_TOKEN, force=True)
    assert forced["removed"] == 2
    assert forced["forced"] is True
    assert forced["remaining_models"] == []
    assert target_rows(store) == []


def test_rollback_removes_only_the_target_model(store):
    ids = add_facts(store, 3)
    seed_vectors(store, ids, model=OLD_MODEL, dims=8)
    seed_vectors(store, ids, model=MODEL, dims=DIMS)
    old_before = raw_rows(store, OLD_MODEL)

    result = rollback(store, MODEL, confirm_token=CONFIRM_TOKEN)

    assert result["removed"] == 3
    assert result["remaining_models"] == [OLD_MODEL]
    assert target_rows(store) == []
    assert raw_rows(store, OLD_MODEL) == old_before


def test_rollback_on_absent_model_is_a_noop(store):
    ids = add_facts(store, 1)
    seed_vectors(store, ids, model=OLD_MODEL, dims=8)

    result = rollback(store, "never-used", confirm_token=CONFIRM_TOKEN)

    assert result["removed"] == 0
    assert result["remaining_models"] == [OLD_MODEL]


# ------------------------------------------------------------------ models
@pytest.mark.parametrize("bad", ["", "   ", "m" * 129])
def test_invalid_model_names_are_rejected(store, bad):
    add_facts(store, 1)
    client = FakeClient()

    with pytest.raises(ValueError):
        plan(store, bad)
    with pytest.raises(ValueError):
        run(store, client, bad, dry_run=False)
    with pytest.raises(ValueError):
        verify(store, bad)
    with pytest.raises(ValueError):
        rollback(store, bad, confirm_token=CONFIRM_TOKEN)
    assert client.calls == []


def test_model_name_length_limit_is_inclusive(store):
    assert plan(store, "m" * 128)["target_model"] == "m" * 128


def test_model_name_is_stored_verbatim(store):
    add_facts(store, 2)
    name = "Org/Embed-Model.V2"

    result = run(store, FakeClient(), name, batch=2, dry_run=False)

    assert result["target_model"] == name
    assert {row[1] for row in target_rows(store, name)} == {name}


def test_migration_never_touches_other_models_bytes(store):
    ids = add_facts(store, 4)
    seed_vectors(store, ids[:3], model=OLD_MODEL, dims=8)
    old_before = raw_rows(store, OLD_MODEL)

    run(store, FakeClient(), MODEL, batch=3, dry_run=False)

    assert raw_rows(store, OLD_MODEL) == old_before
    assert len(old_before) == 3
    assert len(target_rows(store)) == 4
