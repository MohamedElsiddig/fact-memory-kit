"""Fusion invariants: provider rows own slots (plain dicts, tmp_path not required)."""
from factkit import holo


class _Store:
    def __init__(self, rows=None):
        self._rows = {int(r["fact_id"]): dict(r) for r in (rows or [])}

    def read(self, sql, params=()):
        if params:
            fid = int(params[0])
            row = self._rows.get(fid)
            return [row] if row else []
        return list(self._rows.values())


def _providers(n):
    return [{"fact_id": i, "content": f"p{i}", "score": 1.0} for i in range(1, n + 1)]


def _sem_hits(ids):
    return [{"fact_id": i, "score": 0.9} for i in ids]


def test_limit_10_keeps_all_10_provider_ids(monkeypatch):
    monkeypatch.setattr(holo, "prefetch", lambda *a, **k: _sem_hits([99, 98]))
    out = holo.fuse_results(_Store(), "q", _providers(10), limit=10)
    assert [r["fact_id"] for r in out] == list(range(1, 11))


def test_limit_6_returns_first_6_provider_ids(monkeypatch):
    monkeypatch.setattr(holo, "prefetch", lambda *a, **k: _sem_hits([99]))
    out = holo.fuse_results(_Store(), "q", _providers(10), limit=6)
    assert [r["fact_id"] for r in out] == list(range(1, 7))


def test_semantic_fills_only_spare_slots(monkeypatch):
    monkeypatch.setattr(holo, "prefetch", lambda *a, **k: _sem_hits([101, 102, 103, 104, 105]))
    rows = [{"fact_id": i, "content": f"s{i}", "category": "x", "tags": "", "trust_score": 0.8}
            for i in range(101, 106)]
    out = holo.fuse_results(_Store(rows), "q", _providers(10), limit=12)
    provider_ids = list(range(1, 11))
    assert [r["fact_id"] for r in out[:10]] == provider_ids
    added = [r["fact_id"] for r in out[10:]]
    assert len(added) <= 2
    assert set(added).issubset({101, 102, 103, 104, 105})


def test_limit_none_returns_every_provider_row(monkeypatch):
    monkeypatch.setattr(holo, "prefetch", lambda *a, **k: [])
    prov = _providers(10)
    out = holo.fuse_results(_Store(), "q", prov, limit=None)
    assert [r["fact_id"] for r in out] == [r["fact_id"] for r in prov]
