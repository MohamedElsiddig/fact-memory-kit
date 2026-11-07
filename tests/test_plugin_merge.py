"""The transform_tool_result hook must merge into the provider's real key."""
import importlib
import importlib.util
import pathlib
import json
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
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
    out = mod.on_transform_tool_result(
        tool_name="fact_store",
        args={"action": "search", "query": "q"},
        result=json.dumps({"results": [{"fact_id": 1, "content": "one"},
                                       {"fact_id": 2, "content": "two"}], "count": 2}),
    )
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


def _facts(ids):
    return [{"fact_id": i, "content": f"fact {i}", "score": round(1.0 - i / 100, 3)}
            for i in ids]


def _run(mod, monkeypatch, provider, fused, limit=None):
    """Run the hook with a spying fuse_results; return (payload, recorded kwargs)."""
    _stub(mod, monkeypatch, fused)
    monkeypatch.setattr(mod, "_rerank_enabled", lambda: False)
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


def test_provider_limit_3_returns_exactly_3(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    provider = _facts([1, 2, 3])
    data, calls = _run(mod, monkeypatch, provider, provider, limit=3)
    assert [f["fact_id"] for f in data["results"]] == [1, 2, 3]
    assert data["count"] == 3
    assert calls[-1]["limit"] == 3


def test_provider_3_limit_5_is_bounded_by_provider_page(tmp_path, monkeypatch):
    """The provider page bounds the response: no semantic rows beyond it."""
    mod = _load(tmp_path)
    provider = _facts([1, 2, 3])
    semantic = [{"fact_id": 10, "content": "semantic ten", "score": 0.2,
                 "source": "semantic"},
                {"fact_id": 11, "content": "semantic eleven", "score": 0.1,
                 "source": "semantic"}]
    data, calls = _run(mod, monkeypatch, provider, provider + semantic, limit=5)
    ids = [f["fact_id"] for f in data["results"]]
    assert ids == [1, 2, 3]
    assert data["count"] == 3
    assert calls[-1]["limit"] == 3
    assert 10 not in ids and 11 not in ids


def test_provider_6_limit_4_first_four_provider_order(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    provider = _facts([1, 2, 3, 4, 5, 6])
    data, calls = _run(mod, monkeypatch, provider, provider[:4], limit=4)
    assert [f["fact_id"] for f in data["results"]] == [1, 2, 3, 4]
    assert data["count"] == 4
    assert calls[-1]["limit"] == 4


def test_no_limit_falls_back_to_config_limit(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    provider = _facts([1, 2, 3])
    data, calls = _run(mod, monkeypatch, provider, provider)
    assert [f["fact_id"] for f in data["results"]] == [1, 2, 3]
    assert data["count"] == 3
    assert calls[-1]["limit"] == 3


def test_provider_3_limit_6_bounded_to_provider_page(tmp_path, monkeypatch):
    """A larger caller limit cannot pull semantic rows past the provider page."""
    mod = _load(tmp_path)
    provider = _facts([1, 2, 3])
    semantic = [{"fact_id": 10, "content": "semantic ten", "score": 0.2,
                 "source": "semantic"}]
    data, calls = _run(mod, monkeypatch, provider, provider + semantic, limit=6)
    ids = [f["fact_id"] for f in data["results"]]
    assert ids == [1, 2, 3]
    assert data["count"] == 3
    assert calls[-1]["limit"] == 3
    assert 10 not in ids


def test_provider_3_limit_3_returns_exactly_3(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    provider = _facts([1, 2, 3])
    semantic = [{"fact_id": 10, "content": "semantic ten", "score": 0.2,
                 "source": "semantic"}]
    data, calls = _run(mod, monkeypatch, provider, provider + semantic, limit=3)
    ids = [f["fact_id"] for f in data["results"]]
    assert ids == [1, 2, 3]
    assert data["count"] == 3
    assert calls[-1]["limit"] == 3


def test_provider_6_limit_6_unchanged(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    provider = _facts([1, 2, 3, 4, 5, 6])
    data, calls = _run(mod, monkeypatch, provider, provider, limit=6)
    ids = [f["fact_id"] for f in data["results"]]
    assert ids == [1, 2, 3, 4, 5, 6]
    assert data["count"] == 6
    assert calls[-1]["limit"] == 6


def test_provider_1_limit_6_bounded_to_one(tmp_path, monkeypatch):
    """A one-row provider page stays one row; no semantic expansion."""
    mod = _load(tmp_path)
    provider = _facts([1])
    semantic = [{"fact_id": 10, "content": "semantic ten", "score": 0.2,
                 "source": "semantic"},
                {"fact_id": 11, "content": "semantic eleven", "score": 0.1,
                 "source": "semantic"}]
    data, calls = _run(mod, monkeypatch, provider, provider + semantic, limit=6)
    ids = [f["fact_id"] for f in data["results"]]
    assert ids == [1]
    assert data["count"] == 1
    assert calls[-1]["limit"] == 1


def test_fused_page_longer_than_limit_is_trimmed_after_rerank(tmp_path, monkeypatch):
    """A fused/reranked page may be longer than the caller's page size."""
    mod = _load(tmp_path)
    provider = _facts([1, 2, 3, 4, 5, 6])
    data, calls = _run(mod, monkeypatch, provider, provider, limit=4)
    assert [f["fact_id"] for f in data["results"]] == [1, 2, 3, 4]
    assert data["count"] == 4
    assert calls[-1]["limit"] == 4


def test_trim_happens_after_rerank(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    provider = _facts([1, 2, 3, 4, 5, 6])
    _stub(mod, monkeypatch, provider)
    monkeypatch.setattr(mod, "_rerank", lambda q, fused: (list(reversed(fused)), True))
    monkeypatch.setattr(mod.factkit.holo, "fuse_results", lambda *a, **k: list(provider))
    out = mod.on_transform_tool_result(
        tool_name="fact_store",
        args={"action": "search", "query": "q", "limit": 4},
        result=json.dumps({"results": provider, "count": len(provider)}),
    )
    data = json.loads(out)
    assert [f["fact_id"] for f in data["results"]] == [6, 5, 4, 3]
    assert data["count"] == 4
    assert data["ranking"] == "reranked"


def test_limit_is_clamped_to_50(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    provider = _facts(range(1, 61))
    data, calls = _run(mod, monkeypatch, provider, provider, limit=500)
    assert calls[-1]["limit"] == 50
    assert data["count"] == 50


def test_limit_accepts_numeric_string(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    provider = _facts([1, 2, 3])
    data, calls = _run(mod, monkeypatch, provider, provider, limit="2")
    assert calls[-1]["limit"] == 2
    assert [f["fact_id"] for f in data["results"]] == [1, 2]


def test_invalid_limit_falls_back_to_default(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    provider = _facts([1, 2, 3])
    _, calls = _run(mod, monkeypatch, provider, provider, limit="abc")
    assert calls[-1]["limit"] == 3


def test_negative_limit_clamped_to_one(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    provider = _facts([1, 2, 3])
    data, calls = _run(mod, monkeypatch, provider, provider, limit=-5)
    assert calls[-1]["limit"] == 1
    assert len(data["results"]) == 1


def test_empty_results_list_returns_original(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    _stub(mod, monkeypatch, [{"fact_id": 1, "content": "one"}])
    assert mod.on_transform_tool_result(
        tool_name="fact_store",
        args={"action": "search", "query": "q"},
        result=json.dumps({"results": [], "count": 0}),
    ) is None


def test_missing_limit_uses_configured_value(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    provider = _facts(range(1, 11))
    calls = []
    _stub(mod, monkeypatch, provider)
    monkeypatch.setattr(mod, "_rerank_enabled", lambda: False)
    monkeypatch.setattr(mod, "_num", lambda key, default: 7 if key == "limit" else default)
    monkeypatch.setattr(mod.factkit.holo, "fuse_results",
                        lambda *a, **k: calls.append(k) or [dict(f) for f in provider])
    out = mod.on_transform_tool_result(
        tool_name="fact_store",
        args={"action": "search", "query": "q"},
        result=json.dumps({"results": provider, "count": len(provider)}),
    )
    assert calls[-1]["limit"] == 7
    assert json.loads(out)["count"] == 7


def test_remember_only_sees_emitted_rows(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    provider = _facts([1, 2, 3, 4, 5, 6])
    seen = []
    _stub(mod, monkeypatch, provider)
    monkeypatch.setattr(mod, "_rerank", lambda q, fused: (fused, False))
    monkeypatch.setattr(mod, "_mark_seen",
                        lambda contents, session=None: seen.extend(contents))
    monkeypatch.setattr(mod.factkit.holo, "fuse_results", lambda *a, **k: list(provider))
    out = mod.on_transform_tool_result(
        tool_name="fact_store",
        args={"action": "search", "query": "q", "limit": 3},
        result=json.dumps({"results": provider, "count": len(provider)}),
    )
    assert [f["fact_id"] for f in json.loads(out)["results"]] == [1, 2, 3]
    assert seen == ["fact 1", "fact 2", "fact 3"]
