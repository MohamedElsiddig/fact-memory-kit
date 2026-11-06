"""The transform_tool_result hook must merge into the provider's real key."""
import importlib
import importlib.util
import pathlib
import json
import sys

REPO = pathlib.Path("/home/melsiddig/workspace/fact-memory-kit")
SOURCE = REPO / "integrations/hermes/factkit/plugin_init.py"


def _load(tmp_path):
    """Mirror an installed plugin: entry point plus the vendored factkit package."""
    root = tmp_path / "plugin"
    (root / "factkit").mkdir(parents=True)
    (root / "__init__.py").write_text(SOURCE.read_text())
    (root / "factkit/__init__.py").write_text('__all__ = ["holo"]\n')
    for name in ("config", "embed", "vectors", "units", "maintenance", "holo"):
        (root / f"factkit/{name}.py").write_text((REPO / f"factkit/{name}.py").read_text())
    name = f"fkmerge{abs(hash(str(tmp_path)))}"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    importlib.import_module(f"{name}.factkit.holo")
    return mod


def _stub(mod, monkeypatch, merged):
    monkeypatch.setattr(mod, "_merge_enabled", lambda: True)
    monkeypatch.setattr(mod, "_breaker_open", lambda: False)
    monkeypatch.setattr(mod, "_store", lambda: object())
    monkeypatch.setattr(mod, "_num", lambda key, default: default)
    monkeypatch.setattr(mod, "_mark_seen", lambda contents: None)
    monkeypatch.setattr(mod.factkit.holo, "fuse_results", lambda *a, **k: merged)


def _payload(key="results"):
    return json.dumps({key: [{"fact_id": 1, "content": "one"}], "count": 1})


def test_provider_results_key_is_merged(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    _stub(mod, monkeypatch, [{"fact_id": 1, "content": "one", "score": 0.5},
                              {"fact_id": 2, "content": "two", "score": 0.4}])
    out = mod.on_transform_tool_result(tool_name="fact_store",
                                       args={"action": "search", "query": "q"},
                                       result=_payload("results"))
    data = json.loads(out)
    assert data["results"] == [{"fact_id": 1, "content": "one", "score": 0.5},
                               {"fact_id": 2, "content": "two", "score": 0.4}]
    assert data["count"] == 2
    assert data["ranking"] == "fused"


def test_legacy_facts_key_still_works(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    _stub(mod, monkeypatch, [{"fact_id": 1, "content": "one", "score": 0.5}])
    out = mod.on_transform_tool_result(tool_name="fact_store",
                                       args={"action": "search", "query": "q"},
                                       result=_payload("facts"))
    assert json.loads(out)["facts"][0]["fact_id"] == 1


def test_non_search_action_untouched(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    _stub(mod, monkeypatch, [{"fact_id": 2, "content": "two", "score": 0.4}])
    assert mod.on_transform_tool_result(tool_name="fact_store", args={"action": "add"},
                                        result=_payload()) is None


def test_breaker_open_returns_original(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    _stub(mod, monkeypatch, [{"fact_id": 2, "content": "two", "score": 0.4}])
    monkeypatch.setattr(mod, "_breaker_open", lambda: True)
    assert mod.on_transform_tool_result(tool_name="fact_store",
                                        args={"action": "search", "query": "q"},
                                        result=_payload()) is None


def test_no_list_key_returns_original(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    _stub(mod, monkeypatch, [{"fact_id": 2}])
    assert mod.on_transform_tool_result(tool_name="fact_store",
                                        args={"action": "search", "query": "q"},
                                        result=json.dumps({"count": 0})) is None
