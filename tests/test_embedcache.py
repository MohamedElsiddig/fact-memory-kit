"""Offline tests for the persistent embedding cache.

No endpoint, no real data: a deterministic fake embedder stands in for the
model, and every fixture name is synthetic (Atlas Server, Beacon Mini,
192.0.2.x).
"""
from __future__ import annotations

import array
import math
import os
import sqlite3
import stat
from datetime import datetime, timedelta, timezone

import pytest

from factkit import embedcache
from factkit.embedcache import EmbedCache, cache_key, cached_embed

MODEL = "text-embedding-nomic-embed-text-v1.5"
OTHER_MODEL = "text-embedding-3-small"
DIMS = 768

ATLAS = "Atlas Server is at 192.0.2.10"
BEACON = "Beacon Mini reports 192.0.2.11 on port 8080"

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")


# ------------------------------------------------------------------ helpers
def fake_vector(text: str, dims: int = DIMS) -> list[float]:
    """Deterministic per-text vector; values land inside float32 exactly."""
    seed = sum(ord(ch) for ch in text) or 1
    return [((seed + i) % 97) / 128.0 for i in range(dims)]


class FakeEmbedder:
    """Records calls so a hit can be proven not to have reached the model."""

    def __init__(self, dims: int = DIMS):
        self.dims = dims
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        return fake_vector(text, self.dims)


def _backdate(path, text: str, model: str, days: float) -> None:
    """Rewrite created_at directly; the cache exposes no clock injection."""
    stamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(
        timespec="seconds")
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("UPDATE embedding_cache SET created_at = ? WHERE key = ?",
                     (stamp, cache_key(text, model)))
        conn.commit()
    finally:
        conn.close()


def _raw_row(path, text: str, model: str):
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute(
            "SELECT model, dim, vector, created_at, hits FROM embedding_cache "
            "WHERE key = ?", (cache_key(text, model),)).fetchone()
    finally:
        conn.close()


@pytest.fixture
def cache(tmp_path):
    c = EmbedCache(tmp_path / "embedding-cache.db")
    try:
        yield c
    finally:
        c.close()


# -------------------------------------------------------------------- tests
def test_miss_then_hit(cache):
    embed = FakeEmbedder()
    assert cache.get(ATLAS, MODEL) is None
    first = cached_embed(cache, embed, ATLAS, MODEL)
    second = cached_embed(cache, embed, ATLAS, MODEL)
    assert first == second
    assert embed.calls == [ATLAS]          # second call never reached the model
    assert cache.stats()["entries"] == 1


def test_key_stable_across_whitespace(cache):
    messy = "  Atlas \t Server\n is   at 192.0.2.10  "
    assert cache_key(messy, MODEL) == cache_key(ATLAS, MODEL)
    cache.put(ATLAS, MODEL, fake_vector(ATLAS))
    assert cache.get(messy, MODEL) == pytest.approx(fake_vector(ATLAS))
    assert cache.stats()["entries"] == 1   # one key, not two


def test_key_does_not_fold_case(cache):
    """embed.py sends text verbatim, so case is significant."""
    assert cache_key("Atlas Server", MODEL) != cache_key("atlas server", MODEL)
    cache.put("Atlas Server", MODEL, fake_vector(ATLAS))
    assert cache.get("atlas server", MODEL) is None


def test_key_is_sha256_hex(cache):
    key = cache_key(ATLAS, MODEL)
    assert len(key) == 64 and all(c in "0123456789abcdef" for c in key)


def test_model_isolation(cache):
    cache.put(ATLAS, MODEL, fake_vector(ATLAS))
    assert cache.get(ATLAS, OTHER_MODEL) is None
    cache.put(ATLAS, OTHER_MODEL, fake_vector(BEACON))
    assert cache.get(ATLAS, MODEL) == pytest.approx(fake_vector(ATLAS))
    assert cache.get(ATLAS, OTHER_MODEL) == pytest.approx(fake_vector(BEACON))
    assert cache.stats()["entries"] == 2


def test_dims_preserved_exactly(cache):
    vec = fake_vector(ATLAS)
    cache.put(ATLAS, MODEL, vec)
    got = cache.get(ATLAS, MODEL)
    assert len(got) == DIMS
    # float32 is lossy in general; this fixture's values are exact in binary
    assert got == list(array.array("f", vec))
    model, dim, blob, _, _ = _raw_row(cache.path, ATLAS, MODEL)
    assert (model, dim) == (MODEL, DIMS)
    assert len(blob) == DIMS * 4 == 3072
    assert cache.stats()["bytes"] == 3072


def test_put_rejects_empty_or_infinite_vectors(cache):
    with pytest.raises(ValueError):
        cache.put(ATLAS, MODEL, [])
    with pytest.raises(ValueError):
        cache.put(ATLAS, MODEL, [1.0, math.inf])
    assert cache.stats()["entries"] == 0


def test_corrupt_blob_is_a_miss(cache):
    cache.put(ATLAS, MODEL, fake_vector(ATLAS))
    conn = sqlite3.connect(str(cache.path))
    try:
        conn.execute("UPDATE embedding_cache SET vector = ? WHERE key = ?",
                     (b"\x01\x02\x03", cache_key(ATLAS, MODEL)))
        conn.commit()
    finally:
        conn.close()
    assert cache.get(ATLAS, MODEL) is None
    # the bad row is evicted, so the next write can heal it
    assert cache.stats()["entries"] == 0


def test_truncated_vector_is_a_miss(cache):
    cache.put(ATLAS, MODEL, fake_vector(ATLAS))
    conn = sqlite3.connect(str(cache.path))
    try:
        conn.execute("UPDATE embedding_cache SET dim = 767 WHERE key = ?",
                     (cache_key(ATLAS, MODEL),))
        conn.commit()
    finally:
        conn.close()
    assert cache.get(ATLAS, MODEL) is None


def test_unwritable_path_does_not_break_cached_embed(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("a regular file cannot hold a database")
    broken = EmbedCache(blocker / "cache.db")
    embed = FakeEmbedder()
    try:
        assert broken.get(ATLAS, "m") is None
        assert cached_embed(broken, embed, ATLAS, "m") == fake_vector(ATLAS)
        assert embed.calls == [ATLAS]
        assert broken.stats() == {"entries": 0, "hits": 0, "bytes": 0,
                                  "oldest": None}
        assert broken.prune()["removed"] == 0
    finally:
        broken.close()


def test_cached_embed_uses_cache_when_it_works(tmp_path):
    embed = FakeEmbedder()
    with EmbedCache(tmp_path / "ok.db") as c:
        for _ in range(3):
            assert cached_embed(c, embed, BEACON, MODEL) == fake_vector(BEACON)
        assert embed.calls == [BEACON]
        assert c.stats()["hits"] == 2


def test_hit_counter_increments(cache):
    cache.put(ATLAS, MODEL, fake_vector(ATLAS))
    assert cache.stats()["hits"] == 0
    cache.get(ATLAS, MODEL)
    cache.get(ATLAS, MODEL)
    cache.get(ATLAS, MODEL)
    assert cache.stats()["hits"] == 3
    assert cache.get(BEACON, MODEL) is None       # a miss is not a hit
    assert cache.stats()["hits"] == 3


def test_stats_numbers(cache):
    assert cache.stats() == {"entries": 0, "hits": 0, "bytes": 0, "oldest": None}
    cache.put(ATLAS, MODEL, fake_vector(ATLAS))
    _backdate(cache.path, ATLAS, MODEL, days=5)
    cache.put(BEACON, MODEL, fake_vector(BEACON))
    cache.get(ATLAS, MODEL)
    cache.get(BEACON, MODEL)
    cache.get(BEACON, MODEL)
    stats = cache.stats()
    assert stats["entries"] == 2
    assert stats["hits"] == 3
    assert stats["bytes"] == 2 * DIMS * 4
    assert stats["oldest"] == (datetime.now(timezone.utc) - timedelta(days=5)
                               ).isoformat(timespec="seconds")


def test_prune_by_ttl(tmp_path):
    c = EmbedCache(tmp_path / "ttl.db", ttl_days=30)
    try:
        c.put(ATLAS, MODEL, fake_vector(ATLAS))
        c.put(BEACON, MODEL, fake_vector(BEACON))
        c.put("192.0.2.12 is spare", MODEL, fake_vector("192.0.2.12 is spare"))
        _backdate(c.path, ATLAS, MODEL, days=60)
        _backdate(c.path, BEACON, MODEL, days=40)

        result = c.prune()
        assert result == {"expired": 2, "overflow": 0, "removed": 2, "entries": 1}
        assert c.get(ATLAS, MODEL) is None
        assert c.get("192.0.2.12 is spare", MODEL) is not None
        assert c.prune()["removed"] == 0          # idempotent
    finally:
        c.close()


def test_prune_by_max_entries(tmp_path):
    c = EmbedCache(tmp_path / "cap.db", max_entries=2, ttl_days=3650)
    texts = [f"Beacon Mini node {i} sits at 192.0.2.{i + 20}" for i in range(5)]
    try:
        for days, text in zip((10, 8, 6, 4, 2), texts):
            c.put(text, MODEL, fake_vector(text))
            _backdate(c.path, text, MODEL, days=days)

        result = c.prune()
        assert result == {"expired": 0, "overflow": 3, "removed": 3, "entries": 2}
        remaining = {c.get(t, MODEL) is not None for t in texts}
        assert remaining == {False, False, False, True, True}   # newest two
        assert c.stats()["entries"] == 2
    finally:
        c.close()


def test_file_permission_is_0600(tmp_path):
    path = tmp_path / "private" / "cache.db"
    with EmbedCache(path) as c:
        c.put(ATLAS, MODEL, fake_vector(ATLAS))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_creation_is_lazy(tmp_path):
    path = tmp_path / "lazy" / "cache.db"
    c = EmbedCache(path)
    try:
        assert not path.exists()          # constructing touches nothing
        assert c.get(ATLAS, MODEL) is None
        assert path.exists()
    finally:
        c.close()


def test_persists_across_instances(tmp_path):
    path = tmp_path / "shared.db"
    with EmbedCache(path) as first:
        first.put(ATLAS, MODEL, fake_vector(ATLAS))
    with EmbedCache(path) as second:
        assert second.get(ATLAS, MODEL) == pytest.approx(fake_vector(ATLAS))
        assert second.stats()["entries"] == 1


def test_rejects_negative_limits(tmp_path):
    with pytest.raises(ValueError):
        EmbedCache(tmp_path / "x.db", max_entries=-1)
    with pytest.raises(ValueError):
        EmbedCache(tmp_path / "x.db", ttl_days=-1)
