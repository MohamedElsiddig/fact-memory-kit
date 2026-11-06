"""Dashboard database selection and read-only guarantees. Offline, tmp_path only."""
from __future__ import annotations

import json
import sqlite3
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import pytest

from factkit import config
from factkit.config import DashboardDB, resolve_dashboard_db
from factkit.server import make_handler
from factkit.store import Store


@pytest.fixture()
def corpus(tmp_path):
    path = tmp_path / "facts.db"
    with Store(path) as store:
        store.add_fact("Atlas Server has 16 GB RAM")
        store.add_fact("Laptop has 32 GB RAM")
    return path


# ------------------------------------------------------------------ selection
def test_default_is_live_and_read_only():
    sel = resolve_dashboard_db(environ={})
    assert sel.mode == "live"
    assert sel.path == config.LIVE_DB
    assert sel.readonly is True
    assert "read-only" in sel.label
    assert str(config.LIVE_DB) in sel.label


def test_corpus_mode_default_path_and_writable(tmp_path):
    sel = resolve_dashboard_db(mode="corpus", environ={})
    assert sel.mode == "corpus"
    assert sel.path == config.CORPUS_DB
    assert sel.readonly is False
    assert "read-write" in sel.label


def test_factkit_db_env_selects_path(corpus):
    sel = resolve_dashboard_db(environ={"FACTKIT_DB": str(corpus)})
    assert sel.path == corpus
    assert sel.mode == "corpus"
    assert sel.readonly is False
    assert sel.source == "FACTKIT_DB"


def test_db_argument_beats_env(corpus, tmp_path):
    other = tmp_path / "other.db"
    sel = resolve_dashboard_db(str(corpus), environ={"FACTKIT_DB": str(other)})
    assert sel.path == corpus
    assert sel.source == "--db"


def test_live_path_is_read_only_even_in_corpus_mode():
    sel = resolve_dashboard_db(str(config.LIVE_DB), mode="corpus", environ={})
    assert sel.mode == "corpus"
    assert sel.readonly is True


def test_live_mode_forces_read_only_off_the_default_path(corpus):
    sel = resolve_dashboard_db(str(corpus), mode="live", environ={})
    assert sel.readonly is True


def test_read_only_env_for_corpus(corpus):
    sel = resolve_dashboard_db(environ={"FACTKIT_DB": str(corpus),
                                        "FACTKIT_READONLY": "1"})
    assert sel.readonly is True
    assert "read-only" in sel.label


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        resolve_dashboard_db(mode="staging", environ={})


def test_as_dict_is_json_serialisable(corpus):
    payload = resolve_dashboard_db(str(corpus), environ={}).as_dict()
    assert payload["writable"] is True
    assert payload["readonly"] is False
    assert json.loads(json.dumps(payload))["path"] == str(corpus)


# ------------------------------------------------------------------- store ro
def test_store_readonly_refuses_writes(corpus):
    with Store(corpus, readonly=True) as store:
        assert store.readonly is True
        assert store.count() == 2
        with pytest.raises(sqlite3.OperationalError):
            store.add_fact("should never land")
        with pytest.raises(sqlite3.OperationalError):
            store.db.execute("DELETE FROM facts")

    assert Store(corpus).count() == 2


def test_store_readonly_never_creates_a_file(tmp_path):
    target = tmp_path / "absent.db"
    with pytest.raises(FileNotFoundError):
        Store(target, readonly=True)
    assert not target.exists()


def test_readonly_reads_entity_type_column(tmp_path):
    """The live Hermes schema names the column ``entity_type``, not ``kind``."""
    path = tmp_path / "live.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT);
        CREATE TABLE entities (entity_id INTEGER PRIMARY KEY, name TEXT,
                               entity_type TEXT, aliases TEXT);
        CREATE TABLE fact_entities (fact_id INTEGER, entity_id INTEGER);
        INSERT INTO facts VALUES (1, 'Atlas Server has 16 GB RAM');
        INSERT INTO entities VALUES (1, 'Atlas Server', 'server', '');
        INSERT INTO fact_entities VALUES (1, 1);
    """)
    con.commit(); con.close()
    with Store(path, readonly=True) as store:
        rows = store.entities()
    assert rows[0]["name"] == "Atlas Server"
    assert rows[0]["kind"] == "server"
    assert rows[0]["n"] == 1


def test_readonly_search_does_not_write_metrics(corpus):
    """GET /api/search must work on a read-only store (no latency logging)."""
    from factkit.hybrid import HybridSearch

    with Store(corpus, readonly=True) as store:
        hits = HybridSearch(store).search("atlas server", k=2, use_vectors=False)
        assert hits
        assert store.db.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 0


# -------------------------------------------------------------------- http ui
@contextmanager
def _serve(selection):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(selection))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(url, path):
    with urllib.request.urlopen(url + path, timeout=5) as resp:
        return json.loads(resp.read().decode())


def test_api_summary_labels_database(corpus):
    selection = DashboardDB(path=corpus, mode="corpus", readonly=True, source="test")
    with _serve(selection) as url:
        data = _get(url, "/api/summary")
        assert data["database"]["path"] == str(corpus)
        assert data["database"]["mode"] == "corpus"
        assert data["database"]["readonly"] is True
        assert "read-only" in data["database"]["label"]
        assert _get(url, "/api/database")["readonly"] is True


def test_page_shows_database_label(corpus):
    selection = DashboardDB(path=corpus, mode="corpus", readonly=True, source="test")
    with _serve(selection) as url:
        with urllib.request.urlopen(url + "/", timeout=5) as resp:
            body = resp.read().decode()
    assert str(corpus) in body
    assert "read-only" in body


def test_read_only_dashboard_refuses_post(corpus):
    selection = DashboardDB(path=corpus, mode="corpus", readonly=True, source="test")
    with _serve(selection) as url:
        req = urllib.request.Request(url + "/api/proposal/1/apply", method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 403
        payload = json.loads(exc.value.read().decode())
        assert "read-only" in payload["error"]
        assert payload["database"]["readonly"] is True
