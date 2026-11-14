"""The plugin hook must pass the caller's limit through and keep provider rows.

Mirrors an installed plugin (entry point plus vendored factkit modules) from a
temporary directory, then spies on ``holo.fuse_results``. No host paths.

The provider page bounds the page the hook may return: ``want`` is clamped to
``len(facts)`` before fusion, so a caller limit larger than the provider page
cannot pull semantic rows past it.
"""

import importlib
import importlib.util
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SOURCE = REPO / "integrations/hermes/factkit/plugin_init.py"


def _load(tmp_path):
    root = tmp_path / "plugin"
    (root / "factkit").mkdir(parents=True)
    (root / "__init__.py").write_text(SOURCE.read_text())
    (root / "factkit/__init__.py").write_text('__all__ = ["holo"]\n')
    for name in ("config", "embed", "vectors", "units", "maintenance", "holo"):
        (root / f"factkit/{name}.py").write_text(
            (REPO / f"factkit/{name}.py").read_text()
        )
    name = f"fkpage{abs(hash(str(tmp_path)))}"
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    importlib.import_module(f"{name}.factkit.holo")
    return mod


def _facts(ids):
    return [{"fact_id": i, "content": f"fact {i}", "score": 1.0} for i in ids]


def _run(mod, monkeypatch, provider, fused, limit=None, namespace=None):
    monkeypatch.setattr(mod, "_merge_enabled", lambda: True)
    monkeypatch.setattr(mod, "_breaker_open", lambda: False)
    monkeypatch.setattr(mod, "_store", lambda: object())
    monkeypatch.setattr(mod, "_num", lambda key, default: default)
    monkeypatch.setattr(mod, "_mark_seen", lambda contents: None)
    monkeypatch.setattr(mod, "_rerank_enabled", lambda: False)
    monkeypatch.setattr(mod, "_namespace", lambda: namespace)
    calls = []

    def fake(*args, **kwargs):
        calls.append(kwargs)
        return [dict(fact) for fact in fused]

    monkeypatch.setattr(mod.factkit.holo, "fuse_results", fake)
    args = {"action": "search", "query": "q"}
    if limit is not None:
        args["limit"] = limit
    out = mod.on_transform_tool_result(
        tool_name="fact_store",
        args=args,
        result=json.dumps({"results": provider, "count": len(provider)}),
    )
    return json.loads(out), calls


def test_provider_10_limit_6_returns_first_six_provider_rows(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    provider = _facts(range(1, 11))
    data, calls = _run(mod, monkeypatch, provider, provider, limit=6)
    assert calls[-1]["limit"] == 6
    assert [f["fact_id"] for f in data["results"]] == list(range(1, 7))
    assert data["count"] == 6


def test_limit_3_with_3_provider_rows_stays_3(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    provider = _facts([1, 2, 3])
    semantic = [{"fact_id": 10, "content": "ten", "score": 0.2, "source": "semantic"}]
    data, calls = _run(mod, monkeypatch, provider, provider + semantic, limit=3)
    assert calls[-1]["limit"] == 3
    assert [f["fact_id"] for f in data["results"]] == [1, 2, 3]
    assert data["count"] == 3


def test_provider_3_limit_6_bounded_to_provider_page(tmp_path, monkeypatch):
    """A larger caller limit is clamped to the provider page before fusion."""
    mod = _load(tmp_path)
    provider = _facts([1, 2, 3])
    semantic = [
        {"fact_id": 10, "content": "ten", "score": 0.2, "source": "semantic"},
        {"fact_id": 11, "content": "eleven", "score": 0.1, "source": "semantic"},
    ]
    data, calls = _run(mod, monkeypatch, provider, provider + semantic, limit=6)
    assert calls[-1]["limit"] == 3
    assert [f["fact_id"] for f in data["results"]] == [1, 2, 3]
    assert data["count"] == 3


def test_namespace_is_forwarded_to_fusion(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    provider = _facts([1])
    _, calls = _run(mod, monkeypatch, provider, provider, limit=3, namespace="work")
    assert calls[-1]["namespace"] == "work"


def test_namespace_helper_reads_config(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    monkeypatch.setattr(mod, "_config", lambda: {"namespace": "  work  "})
    assert mod._namespace() == "work"
    monkeypatch.setattr(mod, "_config", lambda: {"namespace": ""})
    assert mod._namespace() is None
    monkeypatch.setattr(mod, "_config", lambda: {})
    assert mod._namespace() is None
