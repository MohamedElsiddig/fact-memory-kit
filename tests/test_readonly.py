"""Read-only store fallback used when the plugin host refuses a write-open."""
import sqlite3

import pytest

from factkit import holo


def _db(tmp_path):
    store = holo.load_plugin_store(str(tmp_path / "ro.db"))
    store.add_fact_status("Atlas Server has 16 GB RAM", category="entity", tags="", trust=0.9, entities=[])
    return store.db_path


def test_reads_facts(tmp_path):
    db = _db(tmp_path)
    ro = holo.ReadOnlyStore(db)
    rows = ro.read("SELECT fact_id, content FROM facts")
    assert rows and rows[0]["content"].startswith("Atlas Server")
    assert ro.count_facts() == 1


def test_reports_its_path(tmp_path):
    db = _db(tmp_path)
    assert holo.ReadOnlyStore(db).db_path == str(db)


def test_writes_are_refused(tmp_path):
    db = _db(tmp_path)
    ro = holo.ReadOnlyStore(db)
    with pytest.raises(sqlite3.OperationalError):
        ro._conn.execute("DELETE FROM facts")


def test_missing_file_is_not_created(tmp_path):
    target = tmp_path / "absent.db"
    with pytest.raises((sqlite3.OperationalError, FileNotFoundError)):
        holo.ReadOnlyStore(target)
    assert not target.exists()
