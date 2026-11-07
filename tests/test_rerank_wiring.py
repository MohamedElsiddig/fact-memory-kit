"""The companion plugin re-ranks fused results only when explicitly enabled."""
import importlib
import importlib.util
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SOURCE = REPO / "integrations/hermes/factkit/plugin_init.py"
VENDORED = ("config", "embed", "vectors", "units", "embedcache", "maintenance",
            "rerank", "holo")


class VecClient:
    """Exact-text vectors; records every embedded text."""

    def __init__(self, vectors):
        self.vectors = dict(vectors)
        self.calls = []

    def embed_one(self, text):
        self.calls.append(text)
        return self.vectors[text]


class BoomEmbed:
    def embed_one(self, text):
        raise RuntimeError("embed endpoint down")


FUSED = [
    {"fact_id": 1, "content": "alpha doc", "score": 0.9},
    {"fact_id": 2, "content": "bravo doc", "score": 0.8},
]


def _load(tmp_path):
    """Mirror an installed plugin: entry point plus the vendored factkit package."""
    root = tmp_path / "plugin"
    (root / "factkit").mkdir(parents=True)
    (root / "__init__.py").write_text(SOURCE.read_text())
    (root / "factkit/__init__.py").write_text('__all__ = ["holo", "rerank"]\n')
    for name in VENDORED:
        (root / f"factkit/{name}.py").write_text(
            (REPO / f"factkit/{name}.py").read_text())
    name = f"fkrerank{id(tmp_path)}"
    # id() is reused across tests, so a stale package of the same name can
    # still be in sys.modules and shadow this one; drop it before loading.
    stale = [key for key in sys.modules
             if key == name or key.startswith(f"{name}.")]
    for key in stale:
        del sys.modules[key]
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    factkit_pkg = importlib.import_module(f"{name}.factkit")
    factkit_pkg.holo = importlib.import_module(f"{name}.factkit.holo")
    factkit_pkg.rerank = importlib.import_module(f"{name}.factkit.rerank")
    mod.factkit = factkit_pkg
    return mod


def _trip_should_not_happen():
    raise AssertionError("re-rank must not trip the breaker")


def _stub(mod, monkeypatch, fused=None, rerank_on=False):
    merged = [dict(hit) for hit in (fused or FUSED)]
    monkeypatch.setattr(mod, "_merge_enabled", lambda: True)
    monkeypatch.setattr(mod, "_breaker_open", lambda: False)
    monkeypatch.setattr(mod, "_breaker_reset", lambda: None)
    monkeypatch.setattr(mod, "_breaker_trip", _trip_should_not_happen)
    monkeypatch.setattr(mod, "_store", lambda: object())
    monkeypatch.setattr(mod, "_mark_seen", lambda contents: None)
    monkeypatch.setattr(mod, "_num", lambda key, default: default)
    monkeypatch.setattr(mod, "_rerank_enabled", lambda: rerank_on)
    monkeypatch.setattr(mod.factkit.holo, "fuse_results", lambda *a, **k: merged)
    return merged


def _payload():
    return json.dumps({"results": [dict(hit) for hit in FUSED], "count": 2})


def _run(mod):
    return mod.on_transform_tool_result(
        tool_name="fact_store",
        args={"action": "search", "query": "q"},
        result=_payload(),
    )


def test_rerank_off_leaves_fused_order_and_client_untouched(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    merged = _stub(mod, monkeypatch, rerank_on=False)

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("re-rank ran while disabled")

    monkeypatch.setattr(mod, "_rerank_client", _must_not_be_called)
    monkeypatch.setattr(mod.factkit.rerank, "rerank", _must_not_be_called)

    data = json.loads(_run(mod))
    assert data["ranking"] == "fused"
    assert data["results"] == merged
    assert data["count"] == 2


def test_rerank_on_reorders_by_embedding(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    _stub(mod, monkeypatch, rerank_on=True)
    client = VecClient({"q": [1.0, 0.0], "alpha doc": [0.0, 1.0],
                        "bravo doc": [1.0, 0.0]})
    monkeypatch.setattr(mod, "_rerank_client", lambda: client)

    data = json.loads(_run(mod))
    assert data["ranking"] == "reranked"
    assert [hit["fact_id"] for hit in data["results"]] == [2, 1]
    assert data["count"] == 2
    assert client.calls == ["q", "alpha doc", "bravo doc"]


def test_rerank_exception_keeps_fused_order(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    merged = _stub(mod, monkeypatch, rerank_on=True)
    monkeypatch.setattr(mod, "_rerank_client", BoomEmbed)

    data = json.loads(_run(mod))
    assert [hit["fact_id"] for hit in data["results"]] == [1, 2]
    assert data["results"] == merged
    assert data["ranking"] == "fused"


def test_rerank_timeout_result_keeps_fused_order(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    merged = _stub(mod, monkeypatch, rerank_on=True)
    monkeypatch.setattr(mod, "_rerank_client", lambda: object())
    monkeypatch.setattr(
        mod.factkit.rerank, "rerank",
        lambda *a, **k: {"hits": [dict(hit) for hit in merged],
                         "reranked": False, "reason": "timeout"},
    )

    data = json.loads(_run(mod))
    assert [hit["fact_id"] for hit in data["results"]] == [1, 2]
    assert data["results"] == merged
    assert data["ranking"] == "fused"


def test_rerank_enabled_without_client_keeps_fused_order(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    merged = _stub(mod, monkeypatch, rerank_on=True)
    monkeypatch.setattr(mod, "_rerank_client", lambda: None)

    data = json.loads(_run(mod))
    assert data["results"] == merged
    assert data["ranking"] == "fused"


def test_rerank_enabled_reads_env_override(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    monkeypatch.setattr(mod, "_config", lambda: {"rerank": "0"})
    monkeypatch.delenv("FACTKIT_RERANK", raising=False)
    assert mod._rerank_enabled() is False
    monkeypatch.setenv("FACTKIT_RERANK", "1")
    assert mod._rerank_enabled() is True
