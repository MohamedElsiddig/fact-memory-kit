"""Lifecycle and isolation tests for the Hermes factkit plugin entry point.

These exercise the plugin module in a temp fake-plugin layout (no installed
plugin, no live DB). They target the contract in fix/lifecycle: missing store
is never created, handles are cached/closed, dedupe is session-keyed, health is
observable, and session buckets are dropped on session end.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "integrations/hermes/factkit/plugin_init.py"


def _load(tmp_path, db_path: Path | None = None):
    """Load plugin_init as an installed-style package with a stub holo module."""
    root = tmp_path / "plugin"
    pkg = root / "factkit"
    pkg.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text(SOURCE.read_text(), encoding="utf-8")
    # Minimal vendored package so ``from .factkit import holo`` resolves.
    (pkg / "__init__.py").write_text('__all__ = ["holo"]\n', encoding="utf-8")
    for name in ("holo", "config", "embed", "vectors", "units", "maintenance"):
        src = REPO / "factkit" / f"{name}.py"
        if src.exists():
            (pkg / f"{name}.py").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            (pkg / f"{name}.py").write_text("", encoding="utf-8")

    mod_name = f"fklife_{abs(hash((str(tmp_path), str(db_path))))}"
    for key in list(sys.modules):
        if key == mod_name or key.startswith(mod_name + "."):
            sys.modules.pop(key, None)
    spec = importlib.util.spec_from_file_location(
        mod_name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    importlib.import_module(f"{mod_name}.factkit.holo")

    # Isolate state between tests.
    mod._STORES.clear()
    mod._SEEN.clear()
    if hasattr(mod, "_SEEN_GLOBAL"):
        mod._SEEN_GLOBAL.clear()
    mod._HEALTH.update(
        {"recall_failures": 0, "embed_failures": 0, "breaker_open": False, "last_error": ""}
    )
    if hasattr(mod, "_FIRST_FAILURE"):
        mod._FIRST_FAILURE.clear()
    mod._BREAKER.update({"fails": 0, "until": 0.0})
    mod._CFG = {"db": str(db_path)} if db_path is not None else {}
    return mod


class _FakeStore:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def test_missing_db_returns_none_and_creates_no_file(tmp_path, caplog):
    db = tmp_path / "absent" / "memory_store.db"
    mod = _load(tmp_path, db_path=db)
    with caplog.at_level(logging.WARNING):
        assert mod._store() is None
    assert not db.exists()
    assert not db.parent.exists()
    assert any("factkit: store not found, recall disabled" in r.message for r in caplog.records)


def test_store_handle_cached_across_calls(tmp_path, monkeypatch):
    db = tmp_path / "mem.db"
    db.write_bytes(b"")
    mod = _load(tmp_path, db_path=db)
    sentinel = _FakeStore()
    calls = {"n": 0}

    def fake_open(holo, path):
        calls["n"] += 1
        return sentinel

    monkeypatch.setattr(mod, "_open_store", fake_open)
    assert mod._store() is sentinel
    assert mod._store() is sentinel
    assert calls["n"] == 1
    assert mod._STORES[str(db.resolve())] is sentinel


def test_session_end_closes_cached_handles(tmp_path, monkeypatch):
    db = tmp_path / "mem.db"
    db.write_bytes(b"")
    mod = _load(tmp_path, db_path=db)
    store = _FakeStore()
    monkeypatch.setattr(mod, "_open_store", lambda holo, path: store)
    monkeypatch.setattr(mod, "_index_enabled", lambda: False)
    assert mod._store() is store
    mod.on_session_end(session_id="s1")
    assert store.closed == 1
    assert mod._STORES == {}


def test_dedupe_is_per_session(tmp_path):
    mod = _load(tmp_path, db_path=tmp_path / "mem.db")
    fact = "Melsiddig uses a ThinkPad P14s"
    mod._mark_seen([fact], session="s1")
    seen_s1 = mod._seen_contents(session="s1")
    seen_s2 = mod._seen_contents(session="s2")
    prefix = mod._norm(fact)[:60]
    assert prefix in seen_s1
    assert prefix not in seen_s2
    # Same fact again in s1 is still represented (suppressed by caller logic);
    # once in s2 is allowed (not present yet).
    mod._mark_seen([fact], session="s2")
    assert prefix in mod._seen_contents(session="s2")


def test_health_counts_a_failure(tmp_path):
    db = tmp_path / "missing.db"
    mod = _load(tmp_path, db_path=db)
    before = mod.health()
    assert before["recall_failures"] == 0
    assert mod._store() is None
    after = mod.health()
    assert after["recall_failures"] >= 1
    assert isinstance(after["breaker_open"], bool)
    assert "last_error" in after
    # returns a copy
    after["recall_failures"] = 999
    assert mod.health()["recall_failures"] >= 1


def test_session_end_clears_session_bucket(tmp_path):
    mod = _load(tmp_path, db_path=tmp_path / "mem.db")
    mod._mark_seen(["alpha fact content"], session="s1")
    mod._mark_seen(["beta fact content"], session="s2")
    assert "s1" in mod._SEEN
    mod.on_session_end(session_id="s1")
    assert "s1" not in mod._SEEN
    # other sessions unaffected
    assert "s2" in mod._SEEN
