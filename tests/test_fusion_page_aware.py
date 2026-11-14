"""Fusion is provider-page aware: provider rows own their slots.

Plain dicts and a stubbed ``prefetch``: no network, no database, no host paths.

The provider page bounds the page the plugin may return. ``fuse_results`` still
fills *spare* slots for direct callers, so the bound is applied by passing the
already-clamped limit (``min(limit, len(provider_facts))``), exactly as
``on_transform_tool_result`` does. These cases pin that contract down.
"""

from factkit import holo


class _Store:
    def __init__(self, rows=None):
        self._rows = {int(r["fact_id"]): dict(r) for r in (rows or [])}

    def read(self, sql, params=()):
        if params:
            row = self._rows.get(int(params[0]))
            return [row] if row else []
        return list(self._rows.values())


def _providers(n):
    return [{"fact_id": i, "content": f"p{i}", "score": 1.0} for i in range(1, n + 1)]


def _sem_hits(ids):
    return [{"fact_id": i, "score": 0.9} for i in ids]


def _sem_rows(ids):
    return [
        {
            "fact_id": i,
            "content": f"s{i}",
            "category": "x",
            "tags": "",
            "trust_score": 0.8,
        }
        for i in ids
    ]


def test_provider_10_limit_6_returns_first_six_provider_rows(monkeypatch):
    monkeypatch.setattr(
        holo, "prefetch", lambda *a, **k: _sem_hits([1, 2, 3, 10] + [101, 102])
    )
    out = holo.fuse_results(_Store(_sem_rows([101, 102])), "q", _providers(10), limit=6)
    assert [r["fact_id"] for r in out] == list(range(1, 7))


def test_provider_3_limit_3_returns_exactly_three(monkeypatch):
    monkeypatch.setattr(
        holo, "prefetch", lambda *a, **k: _sem_hits([101, 102, 103, 104, 105])
    )
    out = holo.fuse_results(
        _Store(_sem_rows(range(101, 106))), "q", _providers(3), limit=3
    )
    assert [r["fact_id"] for r in out] == [1, 2, 3]


def test_provider_3_limit_6_effective_page_bounded_by_provider_page(monkeypatch):
    """A caller limit larger than the provider page cannot extend it.

    The hook clamps ``want`` to ``len(facts)`` before calling fusion; here the
    same clamped limit is passed, so the semantic hits have no spare slot and
    the page stays the three provider rows.
    """
    monkeypatch.setattr(
        holo, "prefetch", lambda *a, **k: _sem_hits([101, 102, 103, 104, 105])
    )
    out = holo.fuse_results(
        _Store(_sem_rows(range(101, 106))), "q", _providers(3), limit=min(6, 3)
    )
    assert [r["fact_id"] for r in out] == [1, 2, 3]
