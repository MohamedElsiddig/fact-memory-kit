"""``factkit holo metrics`` CLI wiring (observability enhancement #5).

Only the metrics workflow lives here: ``--report``, ``--save`` and
``--min-score``. Cache and index CLI tests belong to the later observability
tasks. Everything is offline: a ``tmp_path`` store, a deterministic hash
embedder and a JSON cases file; no live database and no network.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from factkit import cli, embed, holo, vectors
from factkit.embedcache import EmbedCache
from factkit.store import Store

FACTS = [
    "Atlas Server has 16 GB RAM",
    "Atlas Server listens on port 8010",
    "Beacon Mini has 1 GB RAM",
    "Backup runs at 02:00 daily",
    "Mazda 3 fuel tank is 50 litres",
    "Transcriber uses the whisper model",
]

CASES = [
    {"query": "atlas server ram", "expect": [1]},
    {"query": "atlas server port 8010", "expect": [2]},
    {"query": "beacon mini ram", "expect": [3]},
    {"query": "backup daily runs", "expect": [4]},
    {"query": "mazda fuel tank", "expect": [5]},
    {"query": "transcriber whisper", "expect": [6]},
]


class FakeEmbedder:
    """Hash bag of words -> 64 dims, deterministic and offline."""

    model = "fake-embed"
    dims = 64

    def _vec(self, text):
        vec = [0.0] * self.dims
        for token in (text or "").lower().split():
            h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            vec[h % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_one(self, text):
        return self._vec(text)

    def embed(self, texts):
        return [self._vec(t) for t in texts]


@pytest.fixture()
def live_db(tmp_path):
    """Provider-shaped store with six embedded facts, closed for read-only use."""
    path = tmp_path / "live.db"
    store = Store(path)
    embedder = FakeEmbedder()
    for content in FACTS:
        fact = store.add_fact(content, trust_score=0.9)
        store.put_embedding(
            fact["fact_id"], embedder.model, embedder.embed_one(content)
        )
    store.db.commit()
    store.close()
    return path


@pytest.fixture()
def cases_file(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(CASES), encoding="utf-8")
    return path


@pytest.fixture()
def failing_cases(tmp_path):
    """One case that a ``k=1`` run must miss (fact 1 wins "atlas server ram")."""
    path = tmp_path / "failing.json"
    path.write_text(
        json.dumps([{"query": "atlas server ram", "expect": [2]}]), encoding="utf-8"
    )
    return path


@pytest.fixture(autouse=True)
def offline_embedder(monkeypatch):
    monkeypatch.setattr(embed, "from_config", lambda cfg: FakeEmbedder())


def run(argv, capsys):
    rc = cli.main(argv)
    out, err = capsys.readouterr()
    return rc, (json.loads(out) if out.strip() else None), err


def test_metrics_default_still_gates(live_db, cases_file, failing_cases, capsys):
    base = ["holo", "metrics", "--db", str(live_db), "--smoke"]

    rc, payload, _ = run([*base, str(cases_file)], capsys)
    assert rc == 0
    assert payload["gate"]["ok"] is True
    assert payload["min_score"] is None  # no floor unless asked

    rc, payload, _ = run([*base, str(failing_cases), "--limit", "1"], capsys)
    assert rc == 1
    assert payload["gate"]["ok"] is False
    assert payload["gate"]["reasons"]  # the failure is visible


def test_metrics_report_always_exits_zero(live_db, failing_cases, capsys):
    rc, payload, _ = run(
        [
            "holo",
            "metrics",
            "--db",
            str(live_db),
            "--smoke",
            str(failing_cases),
            "--limit",
            "1",
            "--report",
        ],
        capsys,
    )

    assert rc == 0
    assert payload["gate"]["ok"] is False


def test_metrics_save_appends_one_row_per_run(tmp_path, live_db, cases_file, capsys):
    out = tmp_path / "nested" / "history.jsonl"
    argv = [
        "holo",
        "metrics",
        "--db",
        str(live_db),
        "--smoke",
        str(cases_file),
        "--save",
        str(out),
    ]

    rc, payload, err = run(argv, capsys)
    assert rc == 0 and payload["gate"]["ok"] is True
    assert "saved metrics to" in err  # note goes to stderr, not stdout

    rc, _, _ = run(argv, capsys)
    assert rc == 0

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # appended, not overwritten
    row = json.loads(lines[0])
    assert set(row) == {"ts", "cases", "hit_rate", "mrr", "k", "gate_ok"}
    assert row["cases"] == 6 and row["k"] == 5
    assert row["hit_rate"] == 1.0 and row["mrr"] == 1.0
    assert row["gate_ok"] is True
    assert row["ts"].endswith("+00:00")  # ISO8601 UTC seconds
    datetime.fromisoformat(row["ts"])


def test_metrics_save_records_a_failed_gate_under_report(
    tmp_path, live_db, failing_cases, capsys
):
    out = tmp_path / "history.jsonl"
    rc, _, _ = run(
        [
            "holo",
            "metrics",
            "--db",
            str(live_db),
            "--smoke",
            str(failing_cases),
            "--limit",
            "1",
            "--report",
            "--save",
            str(out),
        ],
        capsys,
    )

    assert rc == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["cases"] == 1
    assert rows[0]["gate_ok"] is False


def test_metrics_save_default_path_comes_from_the_parser():
    args = cli.build_parser().parse_args(["holo", "metrics", "--save"])

    assert args.save == cli.METRICS_HISTORY_PATH
    assert args.save.endswith(".hermes/factkit-metrics.jsonl")


def test_metrics_min_score_reaches_evaluate(live_db, cases_file, capsys):
    rc, payload, _ = run(
        [
            "holo",
            "metrics",
            "--db",
            str(live_db),
            "--smoke",
            str(cases_file),
            "--min-score",
            "1.5",
        ],
        capsys,
    )

    assert rc == 0
    assert payload["min_score"] == 1.5
    # A floor above every cosine drops the vector channel; BM25 still carries
    # the cases, so the floor demonstrably reached evaluate().
    assert payload["retrieval"] == "lexical"
    assert payload["hit_rate"] == 1.0


def test_metrics_save_failure_exits_two(tmp_path, live_db, cases_file, capsys):
    target = tmp_path / "a-directory"
    target.mkdir()
    rc, payload, err = run(
        [
            "holo",
            "metrics",
            "--db",
            str(live_db),
            "--smoke",
            str(cases_file),
            "--save",
            str(target),
        ],
        capsys,
    )

    assert rc == 2
    assert payload["gate"]["ok"] is True  # the report was still printed
    assert "could not save metrics" in err


# ------------------------------------------------------------- holo cache (#6)
def _cache_path(tmp_path, monkeypatch):
    path = tmp_path / "embed_cache.db"
    monkeypatch.setenv("FACTKIT_EMBED_CACHE", str(path))
    return path


def test_cache_reports_zeros_without_creating_the_file(tmp_path, monkeypatch, capsys):
    path = _cache_path(tmp_path, monkeypatch)

    rc, payload, _ = run(["holo", "cache"], capsys)

    assert rc == 0
    assert payload == {"path": str(path), "entries": 0, "hits": 0,
                       "bytes": 0, "oldest": None}
    assert not path.exists()


def test_cache_reports_entries_bytes_and_oldest(tmp_path, monkeypatch, capsys):
    path = _cache_path(tmp_path, monkeypatch)
    cache = EmbedCache(path)
    cache.put("Atlas Server is at 192.0.2.10", "fake-embed", [0.5, 0.5])
    cache.close()

    rc, payload, _ = run(["holo", "cache"], capsys)

    assert rc == 0
    assert payload["path"] == str(path)
    assert payload["entries"] == 1
    assert payload["hits"] == 0
    assert payload["bytes"] == 8  # two float32 values
    assert isinstance(payload["oldest"], str)


def test_cache_prune_removes_expired_rows(tmp_path, monkeypatch, capsys):
    path = _cache_path(tmp_path, monkeypatch)
    cache = EmbedCache(path)
    cache.put("Atlas Server is at 192.0.2.10", "fake-embed", [0.5, 0.5])
    cache.close()
    stamp = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat(
        timespec="seconds")
    conn = sqlite3.connect(path)
    conn.execute("UPDATE embedding_cache SET created_at = ?", (stamp,))
    conn.commit()
    conn.close()

    rc, payload, _ = run(["holo", "cache", "--prune"], capsys)

    assert rc == 0
    assert payload["removed"] == 1
    assert payload["expired"] == 1
    assert payload["entries"] == 0
    assert payload["stats"]["entries"] == 0
    assert path.exists()  # pruning never deletes the cache file itself


def test_cache_prune_on_a_missing_file_is_a_noop(tmp_path, monkeypatch, capsys):
    path = _cache_path(tmp_path, monkeypatch)

    rc, payload, _ = run(["holo", "cache", "--prune"], capsys)

    assert rc == 0
    assert payload["removed"] == 0
    assert payload["entries"] == 0
    assert not path.exists()


# ------------------------------------------------------------- holo index (#7)
MODEL = "text-embedding-nomic-embed-text-v1.5"


def _provider_file(path, contents):
    store = holo.StandaloneStore(path)
    for content in contents:
        store.add_fact_status(content, category="general", tags="", trust=0.9)
    store.close()
    return path


@pytest.fixture()
def index_vectors(tmp_path, monkeypatch):
    """Redirect factkit's vectors DB into tmp_path and silence plugin health."""
    vectors_db = tmp_path / "factkit-vectors.db"
    monkeypatch.setattr(holo, "_factkit_vectors_db_path", lambda **kw: vectors_db)
    monkeypatch.setattr(holo, "plugin_health", lambda: {})
    monkeypatch.setenv("FACTKIT_EMBED_MODEL", MODEL)
    return vectors_db


def test_index_read_only_reports_coverage(tmp_path, index_vectors, capsys):
    db = _provider_file(tmp_path / "live.db", FACTS[:3])
    conn = sqlite3.connect(index_vectors)
    conn.executescript(holo._FACTKIT_VECTORS_SCHEMA)
    for fact_id in (1, 2):
        conn.execute(
            "INSERT INTO fact_embeddings(fact_id, model, dims, vec) VALUES (?,?,?,?)",
            (fact_id, MODEL, 2, vectors.pack([1.0, 0.0])),
        )
    conn.commit()
    conn.close()

    rc, payload, _ = run(["holo", "index", "--db", str(db)], capsys)

    assert rc == 0
    assert payload["facts"] == 3
    assert payload["embedded_for_model"] == 2
    assert payload["missing"] == 1
    assert payload["coverage"] == 0.667
    assert payload["last_error"] == ""


def test_index_read_only_creates_nothing(tmp_path, index_vectors, capsys):
    db = _provider_file(tmp_path / "live.db", FACTS[:2])

    rc, payload, _ = run(["holo", "index", "--db", str(db)], capsys)

    assert rc == 0
    assert payload["embedded_for_model"] == 0
    assert payload["coverage"] == 0.0
    assert not index_vectors.exists()


@pytest.mark.parametrize("extra", [[], ["--apply"]])
def test_index_refuses_a_missing_store(tmp_path, extra, capsys):
    db = tmp_path / "absent.db"

    rc, payload, err = run(["holo", "index", "--db", str(db), *extra], capsys)

    assert rc == 2
    assert payload is None
    assert "store not found" in err
    assert not db.exists()


def test_index_apply_calls_index_pending_with_a_budget(tmp_path, monkeypatch, capsys):
    db = _provider_file(tmp_path / "live.db", FACTS[:2])
    seen = {}

    def fake_pending(store, **kwargs):
        seen.update(kwargs)
        return {"model": "fake-embed", "indexed": 2, "pending": 1,
                "failed": 1, "skipped": 1, "timed_out": False}

    monkeypatch.setattr(holo, "load_plugin_store", lambda db_path=None: object())
    monkeypatch.setattr(holo, "index_pending", fake_pending)

    rc, payload, _ = run(["holo", "index", "--db", str(db), "--apply"], capsys)

    assert rc == 0
    assert seen == {"timeout": 30.0}
    assert payload["embedded"] == 2
    assert payload["failed"] == 1
    assert payload["skipped"] == 1
    assert payload["timed_out"] is False


def test_index_apply_honours_an_explicit_timeout(tmp_path, monkeypatch, capsys):
    db = _provider_file(tmp_path / "live.db", FACTS[:1])
    seen = {}

    def fake_pending(store, **kwargs):
        seen.update(kwargs)
        return {"model": "fake-embed", "indexed": 0, "pending": 1,
                "failed": 0, "skipped": 1, "timed_out": True}

    monkeypatch.setattr(holo, "load_plugin_store", lambda db_path=None: object())
    monkeypatch.setattr(holo, "index_pending", fake_pending)

    rc, payload, _ = run(
        ["holo", "index", "--db", str(db), "--apply", "--timeout", "5"], capsys)

    assert rc == 0
    assert seen == {"timeout": 5.0}
    assert payload["timed_out"] is True


def test_index_apply_reports_a_hard_error(tmp_path, monkeypatch, capsys):
    db = _provider_file(tmp_path / "live.db", FACTS[:1])
    monkeypatch.setattr(holo, "load_plugin_store", lambda db_path=None: object())
    monkeypatch.setattr(
        holo, "index_pending",
        lambda store, **kw: {"model": "fake-embed", "indexed": 0, "pending": 1,
                             "failed": 1, "skipped": 0, "timed_out": False,
                             "error": "endpoint down"})

    rc, payload, _ = run(["holo", "index", "--db", str(db), "--apply"], capsys)

    assert rc == 1
    assert payload["error"] == "endpoint down"


def test_index_pending_timeout_skips_remaining_rows(tmp_path, monkeypatch):
    db = _provider_file(tmp_path / "live.db", FACTS[:2])
    monkeypatch.setattr(holo, "_factkit_vectors_db_path",
                        lambda **kw: tmp_path / "vectors.db")

    class Client:
        timeout = 20.0

        def embed(self, texts):
            raise AssertionError("must not embed after the budget is spent")

    monkeypatch.setattr(holo, "_resolve_embed",
                        lambda cfg, **kw: (Client(), "fake-embed"))
    store = holo.StandaloneStore(db)
    try:
        out = holo.index_pending(store, timeout=1e-9)
    finally:
        store.close()

    assert out["timed_out"] is True
    assert out["indexed"] == 0
    assert out["failed"] == 0
    assert out["skipped"] == 2


def test_index_pending_untimed_clean_path_reports_zeroed_counters(tmp_path, monkeypatch):
    db = _provider_file(tmp_path / "live.db", FACTS[:2])
    monkeypatch.setattr(holo, "_factkit_vectors_db_path",
                        lambda **kw: tmp_path / "vectors.db")

    class Client:
        timeout = 20.0

        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    client = Client()
    monkeypatch.setattr(holo, "_resolve_embed",
                        lambda cfg, **kw: (client, "fake-embed"))
    store = holo.StandaloneStore(db)
    try:
        out = holo.index_pending(store)
    finally:
        store.close()

    assert out["indexed"] == 2
    assert out["failed"] == 0
    assert out["skipped"] == 0
    assert out["timed_out"] is False
    assert client.timeout == 20.0  # an untimed run leaves the client alone
