"""Retirement records validity: the tombstone keeps the retired fact's bounds.

``remove_fact`` deletes the facts row, so ``valid_to`` only survives if the
tombstone snapshot captured it first.  These cases run against the standalone
``Store`` (the same writable shape the CLI maintenance path uses): one store
with the additive temporal columns, one without, plus a dry run that must not
write anything.
"""

import sqlite3

import pytest

from factkit import holo, maintenance, temporal
from factkit.store import Store


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "facts.db")
    yield s
    s.close()


def _seed(store):
    keep = store.add_fact(
        "Atlas transcriber listens on 8010", category="entity", tags="infra"
    )["fact_id"]
    retire = store.add_fact(
        "Atlas transcriber listens on 8011", category="entity", tags="infra"
    )["fact_id"]
    return keep, retire


def _conflict(keep, retire):
    return {"keep": keep, "retire": retire, "diffs": [{"slot": "port"}]}


def test_conflict_records_validity_in_the_tombstone(store):
    keep, retire = _seed(store)
    temporal.ensure_schema(store)

    out = holo.apply_conflict(store, _conflict(keep, retire), dry_run=False)

    assert out["ok"] is True
    assert store.get(retire) is None  # the live row is gone
    tombstones = maintenance.list_tombstones(store)
    assert [t["fact_id"] for t in tombstones] == [retire]
    assert tombstones[0]["valid_to"] is not None
    kept = store.get(keep)
    assert kept["valid_from"] is not None  # the surviving claim is opened
    assert kept["valid_to"] is None


def test_merge_records_validity_in_the_tombstone(store):
    keep, retire = _seed(store)
    temporal.ensure_schema(store)

    out = holo.apply_merge(store, {"keep": keep, "retire": retire}, dry_run=False)

    assert out["ok"] is True
    assert store.get(retire) is None
    tombstone = maintenance.list_tombstones(store)[0]
    assert tombstone["fact_id"] == retire
    assert tombstone["valid_to"] is not None
    assert store.get(keep)["valid_from"] is not None


def test_dry_run_writes_nothing(store):
    keep, retire = _seed(store)
    temporal.ensure_schema(store)

    out = holo.apply_conflict(store, _conflict(keep, retire), dry_run=True)

    assert out["dry_run"] is True
    assert store.count_facts() == 2
    assert store.get(retire) is not None
    assert maintenance.list_tombstones(store) == []
    assert store.get(keep)["valid_from"] is None


def test_store_without_validity_columns_still_retires(store):
    keep, retire = _seed(store)
    columns = {row[1] for row in store.db.execute("PRAGMA table_info(facts)")}
    assert "valid_to" not in columns  # the additive schema was never applied

    out = holo.apply_conflict(store, _conflict(keep, retire), dry_run=False)

    assert out["ok"] is True
    assert store.get(retire) is None
    tombstone = maintenance.list_tombstones(store)[0]
    assert tombstone["valid_from"] is None
    assert tombstone["valid_to"] is None
    # no implicit migration: a missing temporal module/schema stays missing
    columns = {row[1] for row in store.db.execute("PRAGMA table_info(facts)")}
    assert "valid_to" not in columns


def test_tombstone_table_is_migrated_additively(tmp_path):
    """A tombstone table created before the validity columns gains them."""
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE fact_tombstones ("
            "fact_id INTEGER PRIMARY KEY, content TEXT NOT NULL, category TEXT, "
            "tags TEXT, trust_score REAL, entities TEXT, hrr_vector BLOB, "
            "retired_at TIMESTAMP, retired_by TEXT, reason TEXT, batch_id TEXT, "
            "restored_at TIMESTAMP)"
        )
        conn.commit()
    finally:
        conn.close()

    store = Store(path)
    try:
        maintenance.ensure_tombstones(store)
        cols = {
            row[1] for row in store.db.execute("PRAGMA table_info(fact_tombstones)")
        }
        assert {"valid_from", "valid_to"} <= cols
        maintenance.ensure_tombstones(store)  # idempotent
    finally:
        store.close()
