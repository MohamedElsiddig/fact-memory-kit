"""Opt-in namespace scoping for prefetch and fusion.

Scoping is a post-filter over ``ns:<name>`` tags: no network, no host paths.
``prefetch`` is exercised end to end against a temporary ``Store`` with stubbed
embedding calls; ``fuse_results`` uses a dict store and a stubbed ``prefetch``.
"""

import sys

import pytest

from factkit import embedcache, holo, vectors
from factkit.store import Store

MODEL = "test-model"


class _Client:
    timeout = 1.5
    retries = 0

    def embed_one(self, text):  # pragma: no cover - cached_embed is stubbed
        return [1.0, 0.0]


@pytest.fixture()
def embedded_store(tmp_path, monkeypatch):
    store = Store(tmp_path / "facts.db")
    work = store.add_fact(
        "Atlas backups run at 02:40", category="project", tags="infra, ns:work"
    )["fact_id"]
    home = store.add_fact(
        "The kitchen lights are warm white", category="general", tags="ns:home"
    )["fact_id"]
    loose = store.add_fact("Beacon Mini has 32 GB RAM", category="general", tags="")[
        "fact_id"
    ]
    for fid, vec in ((work, [1.0, 0.0]), (home, [0.95, 0.31]), (loose, [0.9, 0.44])):
        store.db.execute(
            "INSERT INTO fact_embeddings(fact_id, model, dims, vec) VALUES (?,?,?,?)",
            (fid, MODEL, len(vec), vectors.pack(vec)),
        )
    store.db.commit()

    monkeypatch.setattr(holo, "_resolve_embed", lambda *a, **k: (_Client(), MODEL))
    monkeypatch.setattr(holo, "_query_cache", lambda: None)
    monkeypatch.setattr(
        embedcache, "cached_embed", lambda cache, fn, text, model: [1.0, 0.0]
    )
    yield store, {"work": work, "home": home, "loose": loose}
    store.close()


def _ids(hits):
    return [h["fact_id"] for h in hits]


def test_unscoped_prefetch_is_unchanged(embedded_store):
    store, ids = embedded_store
    hits = holo.prefetch(store, "atlas backups", limit=5, min_score=0.3)
    assert set(_ids(hits)) == {ids["work"], ids["home"], ids["loose"]}


def test_scoped_prefetch_only_returns_tagged_facts(embedded_store):
    store, ids = embedded_store
    work = holo.prefetch(
        store, "atlas backups", limit=5, min_score=0.3, namespace="work"
    )
    assert _ids(work) == [ids["work"]]
    home = holo.prefetch(
        store, "atlas backups", limit=5, min_score=0.3, namespace="home"
    )
    assert _ids(home) == [ids["home"]]


def test_scoped_prefetch_with_no_members_returns_empty(embedded_store):
    store, _ = embedded_store
    assert (
        holo.prefetch(
            store, "atlas backups", limit=5, min_score=0.3, namespace="missing"
        )
        == []
    )


class _DictStore:
    def __init__(self, rows):
        self._rows = {int(r["fact_id"]): dict(r) for r in rows}

    def read(self, sql, params=()):
        if params:
            row = self._rows.get(int(params[0]))
            return [row] if row else []
        return list(self._rows.values())


def _fuse(monkeypatch, namespace, rows):
    monkeypatch.setattr(
        holo,
        "prefetch",
        lambda *a, **k: [
            {"fact_id": 101, "score": 0.9},
            {"fact_id": 102, "score": 0.8},
        ],
    )
    provider = [{"fact_id": 1, "content": "one", "score": 1.0}]
    # limit 5 leaves room for the two-row semantic reserve under the invariant
    # that semantic rows fill at most half the page's spare slots.
    return holo.fuse_results(
        _DictStore(rows), "q", provider, limit=5, namespace=namespace
    )


def test_fusion_drops_out_of_scope_semantic_rows(monkeypatch):
    rows = [
        {
            "fact_id": 101,
            "content": "a",
            "category": "x",
            "tags": "ns:work",
            "trust_score": 0.8,
        },
        {
            "fact_id": 102,
            "content": "b",
            "category": "x",
            "tags": "ns:home",
            "trust_score": 0.8,
        },
    ]
    out = _fuse(monkeypatch, "work", rows)
    assert [r["fact_id"] for r in out] == [1, 101]


def test_fusion_unscoped_keeps_semantic_rows(monkeypatch):
    rows = [
        {
            "fact_id": 101,
            "content": "a",
            "category": "x",
            "tags": "ns:work",
            "trust_score": 0.8,
        },
        {
            "fact_id": 102,
            "content": "b",
            "category": "x",
            "tags": "ns:home",
            "trust_score": 0.8,
        },
    ]
    out = _fuse(monkeypatch, None, rows)
    assert [r["fact_id"] for r in out] == [1, 101, 102]


def test_missing_namespace_module_degrades_to_no_filter(monkeypatch):
    """A shipped subset without ``factkit.namespaces`` must not raise."""
    rows = [
        {
            "fact_id": 101,
            "content": "a",
            "category": "x",
            "tags": "ns:work",
            "trust_score": 0.8,
        }
    ]
    monkeypatch.setitem(sys.modules, "factkit.namespaces", None)
    out = _fuse(monkeypatch, "work", rows)
    assert [r["fact_id"] for r in out] == [1, 101]
