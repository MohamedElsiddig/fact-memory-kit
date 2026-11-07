"""Embedding invalidation when a fact's content changes."""

from factkit.store import Store

MODEL = "test-model"


def test_invalidate_removes_only_that_fact(tmp_path):
    store = Store(tmp_path / "facts.db")
    try:
        first = store.add_fact("Atlas Server has 16 GB RAM")
        second = store.add_fact("Beacon Server has 32 GB RAM")
        store.put_embedding(first["fact_id"], MODEL, [1.0, 0.0])
        store.put_embedding(first["fact_id"], "other-model", [1.0, 0.0])
        store.put_embedding(second["fact_id"], MODEL, [0.0, 1.0])

        assert store.invalidate_embeddings(first["fact_id"]) == 2

        assert store.get_embedding(first["fact_id"], MODEL) is None
        assert store.get_embedding(first["fact_id"], "other-model") is None
        assert store.get_embedding(second["fact_id"], MODEL) is not None
    finally:
        store.close()


def test_update_identical_content_keeps_embeddings(tmp_path):
    store = Store(tmp_path / "facts.db")
    try:
        fact = store.add_fact("Atlas Server has 16 GB RAM")
        store.put_embedding(fact["fact_id"], MODEL, [1.0, 0.0])

        store.update_fact(
            fact["fact_id"], content="Atlas Server has 16 GB RAM", reason="noop"
        )

        assert store.get_embedding(fact["fact_id"], MODEL) is not None
    finally:
        store.close()


def test_update_changed_content_missing_embedding(tmp_path):
    store = Store(tmp_path / "facts.db")
    try:
        fact = store.add_fact("Atlas Server has 16 GB RAM")
        store.put_embedding(fact["fact_id"], MODEL, [1.0, 0.0])

        store.update_fact(
            fact["fact_id"], content="Atlas Server has 8 GB RAM", reason="upgrade"
        )

        missing = [f["fact_id"] for f in store.facts_missing_embedding(MODEL)]
        assert fact["fact_id"] in missing
        actions = [r["action"] for r in store.audit_log(fact_id=fact["fact_id"])]
        assert "embedding-invalidate" in actions
    finally:
        store.close()
