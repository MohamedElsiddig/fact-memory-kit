"""Recall path against the shipped (manifest-only) plugin subset.

``install.sh`` vendors only the files listed in ``manifest.txt``; the plugin
subset deliberately omits ``store``/``entities``/``standalone``/``policy``/
``smoke``. These tests build a temp copy of exactly that subset, import it under
a unique package name, and prove the ``pre_llm_call`` recall path works without
those modules, sends the secret-scope embedding key, and never trips the
breaker on a healthy turn.

No network and no host paths: the provider DB and factkit's vectors DB both live
under ``tmp_path`` and every vector is canned.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sqlite3
import sys
import types
from pathlib import Path

import pytest

from factkit import vectors

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "integrations/hermes/factkit/plugin_init.py"
MANIFEST = REPO / "integrations/hermes/factkit/manifest.txt"

MODEL = "text-embedding-nomic-embed-text-v1.5"
QUERY = "what are the specs of my atlas server"
QUERY_VEC = [1.0, 0.0, 0.0, 0.0]
NON_MANIFEST = ("store", "entities", "standalone", "policy", "smoke")

_counter = 0


def _at(cosine: float) -> list[float]:
    """A 4-dim vector whose cosine against ``QUERY_VEC`` is exactly ``cosine``."""
    return [cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine)), 0.0, 0.0]


class _MapClient:
    """Canned text -> vector mapping; deterministic and offline."""

    def __init__(self, table):
        self.table = {" ".join(k.lower().split()): list(v) for k, v in table.items()}
        self.timeout, self.retries = 1.0, 0

    def embed_one(self, text):
        key = " ".join((text or "").lower().split())
        return list(self.table.get(key, [0.0, 0.0, 0.0, 0.0]))

    def embed(self, texts):
        return [self.embed_one(t) for t in texts]


class _FakeStore:
    """Duck adapter: ``db_path`` + ``read()`` and no ``.db``/``.readonly``/``.path``.

    That shape is not a ``factkit.store.Store`` and not a provider handle, so
    ``prefetch`` keeps its legacy read-through fallback for the vectors.
    """

    def __init__(self, path):
        self.db_path = str(path)

    def read(self, sql, params=()):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            return list(conn.execute(sql, params))
        finally:
            conn.close()


def _make_store(tmp_path):
    """Provider-shaped DB plus canned vectors; returns ``(store, table)``."""
    db = tmp_path / "provider.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE facts(fact_id INTEGER PRIMARY KEY, content TEXT UNIQUE, category TEXT,
                           tags TEXT, trust_score REAL, created_at TEXT, updated_at TEXT,
                           entities TEXT);
        CREATE TABLE entities(entity_id INTEGER PRIMARY KEY, name TEXT,
                              entity_type TEXT DEFAULT 'unknown');
        CREATE TABLE fact_entities(fact_id INTEGER, entity_id INTEGER);
        CREATE TABLE fact_embeddings(fact_id INTEGER, model TEXT, dims INTEGER, vec BLOB);
        """
    )
    table = {}
    for fact_id, (content, cosine) in enumerate(
        [
            ("Atlas Server has 16 GB RAM", 0.95),
            ("Beacon Mini has 1 GB RAM", 0.40),
        ],
        1,
    ):
        conn.execute(
            "INSERT INTO facts(fact_id,content,category,tags,trust_score,"
            "created_at,updated_at,entities) VALUES (?,?,?,?,?,?,?,?)",
            (
                fact_id,
                content,
                "entity",
                "",
                0.9,
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
                None,
            ),
        )
        vec = _at(cosine)
        conn.execute(
            "INSERT INTO fact_embeddings VALUES (?,?,?,?)",
            (fact_id, MODEL, len(vec), vectors.pack(vec)),
        )
        table[content] = vec
    conn.execute("INSERT INTO entities(name) VALUES (?)", ("Atlas Server",))
    conn.commit()
    conn.close()
    return _FakeStore(db), table


def _load_dist_plugin(tmp_path):
    """Build the shipped subset in ``tmp_path`` and import it under a unique name.

    Returns ``(mod, holo, name)``. The five non-manifest modules must not exist
    in the copied tree; the test then imports and exercises the plugin exactly
    as it is installed on a Hermes host.
    """
    global _counter
    _counter += 1

    root = tmp_path / "plugin"
    pkg = root / "factkit"
    pkg.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text(
        SOURCE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (pkg / "__init__.py").write_text('__all__ = ["holo", "rerank"]\n', encoding="utf-8")
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        entry = raw.strip()
        if not entry or entry.startswith("#") or not entry.startswith("factkit/"):
            continue
        src = REPO / entry
        (pkg / Path(entry).name).write_text(
            src.read_text(encoding="utf-8"), encoding="utf-8"
        )
    for name in NON_MANIFEST:
        assert not (pkg / f"{name}.py").exists(), f"{name}.py must not be vendored"

    name = f"fkdist_{abs(hash(str(tmp_path)))}_{_counter}"
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            sys.modules.pop(key, None)
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    holo = importlib.import_module(f"{name}.factkit.holo")
    return mod, holo, name


@pytest.fixture
def dist_plugin(tmp_path):
    """Load the temp subset per test and drop its modules afterwards."""
    loaded = []
    try:
        result = _load_dist_plugin(tmp_path)
        loaded.append(result[2])
        yield result
    finally:
        for name in loaded:
            for key in list(sys.modules):
                if key == name or key.startswith(name + "."):
                    sys.modules.pop(key, None)


class _Cfg:
    embed_model = MODEL
    embed_base_url = "http://127.0.0.1:1234/v1"
    embed_api_key = ""
    embed_timeout = 20.0
    embed_batch = 32


def _patch_recall(monkeypatch, holo, name, tmp_path, table):
    """Pin the subset's config/embed/cache seams to deterministic fakes."""
    config_mod = importlib.import_module(f"{name}.factkit.config")
    embed_mod = importlib.import_module(f"{name}.factkit.embed")

    table = dict(table)
    table.setdefault(QUERY, list(QUERY_VEC))

    monkeypatch.setattr(config_mod.Config, "load", lambda db_path=None: _Cfg())
    client = _MapClient(table)
    monkeypatch.setattr(embed_mod, "from_config", lambda cfg: client)
    monkeypatch.setattr(holo, "_query_cache", lambda: None)
    monkeypatch.setattr(
        holo, "_factkit_vectors_db_path", lambda: tmp_path / "vectors.db"
    )
    monkeypatch.setattr(holo, "_VECTOR_MIGRATION_ATTEMPTED", set())
    return client


def _reset_plugin_state(mod, db_path):
    mod._CFG = {
        "prefetch": True,
        "limit": 3,
        "min_score": 0.0,
        "embed_timeout_ms": 1500,
        "breaker_seconds": 300,
        "db": str(db_path),
    }
    mod._STORES.clear()
    mod._SEEN.clear()
    if hasattr(mod, "_SEEN_GLOBAL"):
        mod._SEEN_GLOBAL.clear()
    mod._HEALTH.update(
        {
            "recall_failures": 0,
            "embed_failures": 0,
            "breaker_open": False,
            "last_error": "",
        }
    )
    if hasattr(mod, "_FIRST_FAILURE"):
        mod._FIRST_FAILURE.clear()
    mod._BREAKER.update({"fails": 0, "until": 0.0})


# ------------------------------------------------------ missing modules
def test_prefetch_returns_hits_without_non_manifest_modules(
    monkeypatch, tmp_path, dist_plugin
):
    mod, holo, name = dist_plugin
    store, table = _make_store(tmp_path)
    _patch_recall(monkeypatch, holo, name, tmp_path, table)

    hits = holo.prefetch(store, QUERY, min_score=0.0, min_trust=0.0, limit=5)

    assert hits, "prefetch returned no hits from the shipped subset"
    assert 1 in [hit["fact_id"] for hit in hits]


def test_pre_llm_call_survives_missing_modules_and_does_not_trip_breaker(
    monkeypatch, tmp_path, dist_plugin
):
    mod, holo, name = dist_plugin
    store, table = _make_store(tmp_path)
    _patch_recall(monkeypatch, holo, name, tmp_path, table)
    _reset_plugin_state(mod, store.db_path)
    monkeypatch.setattr(mod, "_store", lambda: store)

    out = mod.on_pre_llm_call(user_message=QUERY)

    text = "\n".join(block.get("context", "") for block in out)
    assert "<semantic-memory>" in text
    assert mod._BREAKER["fails"] == 0
    assert not mod._breaker_open()
    assert mod._HEALTH.get("embed_failures", 0) == 0


# ------------------------------------------------------ secret-scope key
def test_pre_llm_call_passes_secret_scope_key_to_client(
    monkeypatch, tmp_path, dist_plugin
):
    mod, holo, name = dist_plugin
    store, table = _make_store(tmp_path)

    for env in (
        "FACTKIT_EMBED_API_KEY",
        "FACTKIT_EMBED_BASE_URL",
        "FACTKIT_EMBED_MODEL",
        "FACTKIT_DB",
    ):
        monkeypatch.delenv(env, raising=False)
    # min_score comes from env when config.yaml does not set it; keep the hit.
    monkeypatch.setenv("FACTKIT_PREFETCH_MIN", "0.0")

    fake_agent = types.ModuleType("agent")
    fake_agent.secret_scope = types.SimpleNamespace(
        get_secret=lambda key: "sekrit" if key == "FACTKIT_EMBED_API_KEY" else None
    )
    monkeypatch.setitem(sys.modules, "agent", fake_agent)

    config_mod = importlib.import_module(f"{name}.factkit.config")
    embed_mod = importlib.import_module(f"{name}.factkit.embed")
    seen = {}

    class _Recorder:
        def __init__(
            self, base_url, model, api_key="", timeout=20.0, batch_size=32, **kwargs
        ):
            seen["api_key"] = api_key
            seen["base_url"] = base_url
            seen["model"] = model
            self.timeout, self.retries = timeout, 0

        def embed_one(self, text):
            return list(QUERY_VEC)

        def embed(self, texts):
            return [list(QUERY_VEC) for _ in texts]

    def _explode(cfg):  # prove the explicit-config path is used, not env defaults
        raise AssertionError("from_config should not be used when config is explicit")

    monkeypatch.setattr(embed_mod, "EmbeddingClient", _Recorder)
    monkeypatch.setattr(embed_mod, "from_config", _explode)
    monkeypatch.setattr(config_mod.Config, "load", lambda db_path=None: _Cfg())
    monkeypatch.setattr(holo, "_query_cache", lambda: None)
    monkeypatch.setattr(
        holo, "_factkit_vectors_db_path", lambda: tmp_path / "vectors.db"
    )
    monkeypatch.setattr(holo, "_VECTOR_MIGRATION_ATTEMPTED", set())

    mod._CFG = {}  # force a fresh _config() read of the secret scope
    mod._STORES.clear()
    mod._SEEN.clear()
    if hasattr(mod, "_SEEN_GLOBAL"):
        mod._SEEN_GLOBAL.clear()
    mod._HEALTH.update(
        {
            "recall_failures": 0,
            "embed_failures": 0,
            "breaker_open": False,
            "last_error": "",
        }
    )
    mod._BREAKER.update({"fails": 0, "until": 0.0})
    monkeypatch.setattr(mod, "_store", lambda: store)

    out = mod.on_pre_llm_call(user_message=QUERY)

    assert seen.get("api_key") == "sekrit"
    assert "FACTKIT_EMBED_API_KEY" not in os.environ
    text = "\n".join(block.get("context", "") for block in out)
    assert "<semantic-memory>" in text
