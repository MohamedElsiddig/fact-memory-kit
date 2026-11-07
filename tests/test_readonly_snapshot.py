"""ReadOnlyStore snapshot/WAL behavior (tmp_path only)."""
import sqlite3

from factkit import holo


def _make_wal_db(tmp_path):
    db = tmp_path / "wal.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO facts (fact_id, content) VALUES (1, 'seed')")
    conn.commit()
    # leave a WAL writer uncheckpointed
    conn.execute("INSERT INTO facts (fact_id, content) VALUES (2, 'wal-only')")
    conn.commit()
    return db, conn


def test_wal_only_write_visible(tmp_path):
    db, writer = _make_wal_db(tmp_path)
    try:
        ro = holo.ReadOnlyStore(db)
        try:
            rows = ro.read("SELECT fact_id, content FROM facts ORDER BY fact_id")
            assert [r["fact_id"] for r in rows] == [1, 2]
        finally:
            ro.close()
    finally:
        writer.close()


def test_snapshot_key_changes_when_wal_changes(tmp_path):
    db, writer = _make_wal_db(tmp_path)
    try:
        k1 = holo.ReadOnlyStore._snapshot_key(str(db))
        writer.execute("INSERT INTO facts (fact_id, content) VALUES (3, 'more')")
        writer.commit()
        k2 = holo.ReadOnlyStore._snapshot_key(str(db))
        assert k1 != k2
    finally:
        writer.close()


def test_repeated_reads_reuse_connection(tmp_path):
    db = tmp_path / "plain.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO facts (fact_id, content) VALUES (1, 'a')")
    conn.commit()
    conn.close()

    ro = holo.ReadOnlyStore(db)
    try:
        c1 = ro._connection()
        ro.read("SELECT 1")
        ro.read("SELECT 1")
        c2 = ro._connection()
        assert c1 is c2
    finally:
        ro.close()


def test_close_twice_safe(tmp_path):
    db = tmp_path / "plain2.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT)")
    conn.commit()
    conn.close()

    ro = holo.ReadOnlyStore(db)
    ro.close()
    ro.close()
