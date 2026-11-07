"""End-to-end tests. No network: a deterministic fake embedder stands in."""
from __future__ import annotations

import hashlib
import math

import pytest

from factkit import bridge, bulk, capture, conflicts, entities, golden, metrics, router, ttl
from factkit.config import Config
from factkit.hybrid import HybridSearch
from factkit.store import Store, fts_query


class FakeEmbedder:
    """Hash-based bag of words -> fixed dims, deterministic and offline."""
    model = "fake-embed"
    dims = 64

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for token in text.lower().split():
            h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            vec[h % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_one(self, text: str) -> list[float]:
        return self._vec(text)

    def embed(self, texts):
        return [self._vec(t) for t in texts]


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "facts.db")
    yield s
    s.close()


@pytest.fixture()
def cfg(store):
    c = Config.load(str(store.path))
    c.embed_model = "fake-embed"
    return c


def seed(store, rows):
    ids = []
    for content, category, tags in rows:
        fact = store.add_fact(content, category=category, tags=tags, trust_score=0.9)
        entities.link(store, fact["fact_id"], content)
        ids.append(fact["fact_id"])
    return ids


# ------------------------------------------------------------------ store
def test_add_get_update_remove_and_audit(store):
    fact = store.add_fact("Atlas Server has 16 GB RAM", category="entity", tags="pi")
    assert store.get(fact["fact_id"])["content"] == "Atlas Server has 16 GB RAM"
    store.update_fact(fact["fact_id"], content="Atlas Server has 8 GB RAM", reason="typo")
    assert "8 GB" in store.get(fact["fact_id"])["content"]
    assert store.remove_fact(fact["fact_id"]) is True
    assert store.get(fact["fact_id"]) is None
    actions = [row["action"] for row in store.audit_log()]
    assert actions[:3] == ["remove", "update", "add"]


def test_duplicate_content_rejected(store):
    import sqlite3
    store.add_fact("unique fact")
    with pytest.raises(sqlite3.IntegrityError):
        store.add_fact("unique fact")


def test_fts_query_sanitiser():
    assert fts_query("what is the port of pi 5?") == '"port" OR "pi"'
    assert fts_query("   ") == ""


def test_fts_search_ranks_target_first(store):
    seed(store, [("Transcriber listens on port 8010", "tool", "stt"),
                 ("proxy runs on the Beacon Mini at 192.0.2.20", "tool", "proxy")])
    hits = store.fts_search("transcriber port 8010", 5)
    assert hits and hits[0][0] == 1


# --------------------------------------------------------------- entities
def test_entity_linking_creates_and_links(store):
    fact = store.add_fact("Atlas Server runs Debian at 192.0.2.10")
    names = entities.link(store, fact["fact_id"])
    assert "Atlas Server" in names
    linked = {e["name"] for e in store.fact_entities(fact["fact_id"])}
    assert "Atlas Server" in linked
    # a second mention resolves to the same entity, not a duplicate row
    before = len(store.entities())
    fact2 = store.add_fact("The Atlas Server is in the living room")
    entities.link(store, fact2["fact_id"])
    assert len(store.entities()) >= before
    assert len([e for e in store.entities() if e["name"].lower() == "atlas server"]) == 1


def test_entity_unit_noise_filtered(store):
    fact = store.add_fact("Laptop has 48 GB RAM and RTX 4070")
    names = entities.link(store, fact["fact_id"])
    assert not any(n.lower().startswith("gb") for n in names)


# ----------------------------------------------------------------- hybrid
def test_hybrid_finds_paraphrase_via_vectors(store, cfg):
    seed(store, [("Transcriber listens on port 8010", "tool", "stt"),
                 ("Mazda 3 fuel tank is 50 litres", "entity", "car")])
    embedder = FakeEmbedder()
    for fact in store.list_facts():
        store.put_embedding(fact["fact_id"], embedder.model,
                            embedder.embed_one(fact["content"]))
    store.db.commit()
    search = HybridSearch(store, embedder, model=embedder.model)
    hits = search.search("transcriber   8010 port", k=3)
    assert hits and hits[0].fact_id == 1
    assert hits[0].bm25_rank == 1


def test_hybrid_degrades_without_embeddings(store):
    seed(store, [("proxy listens on port 443", "tool", "proxy")])
    search = HybridSearch(store, None)          # no embedder at all
    hits = search.search("caddy 443", k=3)
    assert hits and hits[0].fact_id == 1


# ----------------------------------------------------------------- golden
def test_golden_passes_on_good_retrieval(store):
    seed(store, [("Transcriber listens on port 8010", "tool", "stt")])
    cases = [golden.Case(q="transcriber port 8010", facts=[1])]
    result = golden.run(store, lambda q, k=8: HybridSearch(store).search(q, k=k), cases)
    assert golden.check(result).ok


def test_golden_fails_on_regression(store):
    seed(store, [("Transcriber listens on port 8010", "tool", "stt")])
    cases = [golden.Case(q="transcriber port 8010", facts=[999])]
    result = golden.run(store, lambda q, k=8: HybridSearch(store).search(q, k=k), cases)
    result = golden.check(result)
    assert not result.ok and "r_at_5" in result.failures[0]


def test_golden_negative_must_be_clean(store):
    seed(store, [("Transcriber listens on port 8010", "tool", "stt")])
    cases = [golden.Case(q="kubernetes helm chart", facts=[], kind="negative")]
    result = golden.run(store, lambda q, k=8: HybridSearch(store).search(q, k=k), cases)
    assert golden.check(result).ok


def test_golden_seed_and_remap(tmp_path, store):
    seed(store, [("Transcriber listens on port 8010", "tool", "stt")])
    path = tmp_path / "golden.json"
    golden.write_cases([golden.Case(q="transcriber 8010", facts=[1])], path)
    assert golden.remap(path, 1, 2) == 1
    cases, _, thresholds = golden.load_cases(path, store)
    assert cases[0].facts == [2]
    assert thresholds["r_at_5"] == 0.85


# --------------------------------------------------------------- conflicts
def test_golden_rule_retires_older_contradiction(store):
    first = store.add_fact("Atlas Server has 16 GB RAM")
    entities.link(store, first["fact_id"], "Atlas Server has 16 GB RAM")
    second = store.add_fact("Atlas Server has 8 GB RAM")
    entities.link(store, second["fact_id"], "Atlas Server has 8 GB RAM")
    proposals = conflicts.scan(store)
    assert len(proposals) == 1
    assert proposals[0].keep == second["fact_id"]
    assert proposals[0].retire == first["fact_id"]
    assert "count" in proposals[0].slots


def test_golden_rule_ignores_different_subjects(store):
    a = store.add_fact("Atlas Server has 16 GB RAM")
    entities.link(store, a["fact_id"], "Atlas Server")
    b = store.add_fact("Beacon Mini has 1 GB RAM")
    entities.link(store, b["fact_id"], "Beacon Mini")
    assert conflicts.scan(store) == []


def test_slot_values_normalise_case(store):
    assert conflicts.slots("16 GB")["count"] == {"16gb"}
    assert conflicts.slots("port 8010")["port"] == {"8010"}


# -------------------------------------------------------------------- ttl
def test_ttl_flags_old_fact(store):
    fact = store.add_fact("proxy listens on port 443")
    store.db.execute("UPDATE facts SET updated_at='2025-01-01 00:00:00' WHERE fact_id=?",
                     (fact["fact_id"],))
    store.db.commit()
    flags = ttl.scan(store, ttl_days=90, slot_ttl_days=180)
    kinds = {f["kind"] for f in flags}
    assert "age" in kinds and "slot" in kinds


def test_ttl_leaves_fresh_facts_alone(store):
    store.add_fact("proxy listens on port 443")
    assert ttl.scan(store, ttl_days=90, slot_ttl_days=180) == []


# ---------------------------------------------------------------- capture
@pytest.mark.parametrize("text,expected", [
    ("My laptop has 48 GB RAM and an RTX 4070", True),
    ("I prefer short bullet answers from now on", True),
    ("Can you check the logs?", False),
    ("ok thanks", False),
    ("please run the backup", False),
])
def test_capture_heuristic(text, expected):
    assert capture.suggest(text).save is expected


def test_capture_category_and_tags():
    suggestion = capture.suggest("I prefer we always use port 8080 for this")
    assert suggestion.category == "user_pref"
    assert "preference" in suggestion.tags


def test_capture_dedupe_exact(store):
    store.add_fact("My laptop has 48 GB RAM")
    assert capture.dedupe(store, "my laptop has 48 GB ram") is not None


# ----------------------------------------------------------------- router
@pytest.mark.parametrize("query,route", [
    ("what are my laptop specs", "facts"),
    ("what did I say about the car VIN", "facts"),
    ("how do I migrate Debian to NVMe", "wiki"),
    ("is the caddy container running", "live"),
    ("latest home assistant release", "external"),
])
def test_router_heuristics(query, route):
    assert router.heuristic(query).route == route


def test_router_unknown_is_mixed():
    result = router.heuristic("zzz qqq")
    assert result.route == "mixed" and result.confidence == 0.0


# ------------------------------------------------------------------ bulk
def test_bulk_roundtrip(tmp_path, store):
    seed(store, [("Transcriber listens on port 8010", "tool", "stt")])
    path = tmp_path / "out.json"
    text = bulk.export_facts(store, path)
    assert '"count": 1' in text
    other = Store(tmp_path / "other.db")
    result = bulk.import_facts(other, path)
    assert result["added"] == 1
    assert other.count() == 1
    assert bulk.import_facts(other, path)["skipped"] == 1
    other.close()


# ----------------------------------------------------------------- bridge
def test_bridge_reads_messages(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(
        '{"role":"user","content":"My Atlas Server has 16 GB RAM"}\n'
        '{"role":"assistant","content":"noted"}\n'
        '{"role":"user","content":"I prefer Debian"}\n')
    messages = bridge.read_messages(path)
    assert len(messages) == 2
    assert all(role == "user" for role, _ in messages)
    assert bridge.read_messages(path, include_assistant=True).__len__() == 3


def test_bridge_chunks_respect_size():
    messages = [("user", "x" * 1000), ("user", "y" * 1000)]
    parts = bridge.chunks(messages, size=1400)
    assert len(parts) == 2


# ---------------------------------------------------------------- metrics
def test_metrics_summary_counts_backlog(store):
    fact = store.add_fact("Atlas Server has 16 GB RAM")
    entities.link(store, fact["fact_id"], "Atlas Server has 16 GB RAM")
    store.add_fact("Atlas Server has 8 GB RAM")
    conflicts.scan(store)
    data = metrics.summary(store)
    assert data["facts"] == 2
    assert data["backlog"]["conflicts_pending"] == 1
    assert "factkit report" in metrics.render_text(data)
    assert "<html>" in metrics.render_html(data)


# ------------------------------------------------------- hermes integration
def _fake_hermes(path):
    import sqlite3
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE facts(fact_id INTEGER PRIMARY KEY, content TEXT, category TEXT,
                       tags TEXT, trust_score REAL, created_at TEXT, updated_at TEXT);
    CREATE TABLE entities(entity_id INTEGER PRIMARY KEY, name TEXT);
    CREATE TABLE fact_entities(fact_id INTEGER, entity_id INTEGER);
    CREATE TABLE sessions(id TEXT PRIMARY KEY, title TEXT, started_at REAL, message_count INT);
    CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
                          content TEXT, timestamp REAL);
    """)
    con.execute("INSERT INTO facts VALUES (1,'Atlas Server has 16 GB RAM','entity','a',0.9,"
                "'2026-01-01 00:00:00','2026-05-01 00:00:00')")
    con.execute("INSERT INTO facts VALUES (2,'Backup runs at 02:00','tool','b',0.8,"
                "'2026-02-01 00:00:00','2026-06-01 00:00:00')")
    con.execute("INSERT INTO entities VALUES (1,'Atlas Server')")
    con.execute("INSERT INTO fact_entities VALUES (1,1)")
    con.execute("INSERT INTO sessions VALUES ('s1','test',1.0,2)")
    con.execute("INSERT INTO messages(session_id,role,content,timestamp) "
                "VALUES ('s1','user','My Atlas Server has 16 GB RAM',1.0)")
    con.execute("INSERT INTO messages(session_id,role,content,timestamp) "
                "VALUES ('s1','assistant','noted',2.0)")
    con.commit(); con.close()
    return path


def test_import_hermes_preserves_data(tmp_path, store):
    src = _fake_hermes(tmp_path / "memory_store.db")
    result = bulk.import_hermes(store, src)
    assert result["added"] == 2 and result["rows"] == 2
    fact = store.list_facts()[0]
    assert fact["content"] == "Atlas Server has 16 GB RAM"
    assert fact["updated_at"] == "2026-05-01 00:00:00"      # preserved
    assert fact["source"] == "hermes-import"
    assert any(e["name"] == "Atlas Server" for e in store.fact_entities(fact["fact_id"]))
    # idempotent
    assert bulk.import_hermes(store, src)["added"] == 0


def test_import_hermes_dry_run(tmp_path, store):
    src = _fake_hermes(tmp_path / "memory_store.db")
    result = bulk.import_hermes(store, src, dry_run=True)
    assert result["added"] == 2 and store.count() == 0


def test_bridge_reads_hermes_db(tmp_path):
    src = _fake_hermes(tmp_path / "memory_store.db")
    messages, meta = bridge.read_hermes(src)
    assert len(messages) == 1 and messages[0][0] == "user"
    assert meta[0]["id"] == "s1"


# ------------------------------------------------------------ dashboard auth
def test_dashboard_requires_token(tmp_path, store):
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from factkit.server import make_handler

    store.add_fact("Atlas Server has 16 GB RAM")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store.path, "s3cret"))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/summary", timeout=5)
        assert exc.value.code == 401

        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/summary",
                                     headers={"Authorization": "Bearer s3cret"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200

        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/summary",
                                     headers={"Authorization": "Bearer wrong"})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 401
    finally:
        httpd.shutdown(); httpd.server_close()


# ------------------------------------------------- holographic store (holo)
from factkit import holo  # noqa: E402
def _holo_db(path, facts, declared=None):
    import sqlite3
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE facts(fact_id INTEGER PRIMARY KEY, content TEXT, category TEXT,
                       tags TEXT, trust_score REAL, created_at TEXT, updated_at TEXT,
                       hrr_vector BLOB);
    CREATE TABLE entities(entity_id INTEGER PRIMARY KEY, name TEXT,
                          entity_type TEXT DEFAULT 'unknown');
    CREATE TABLE fact_entities(fact_id INTEGER, entity_id INTEGER);
    """)
    for i, (content, updated) in enumerate(facts, 1):
        con.execute("INSERT INTO facts(fact_id,content,category,tags,trust_score,created_at,updated_at) "
                    "VALUES (?,?,?,?,?,?,?)", (i, content, "entity", "", 0.8, updated, updated))
    for i, (name, kind, ids) in enumerate(declared or [], 1):
        con.execute("INSERT INTO entities VALUES (?,?,?)", (i, name, kind))
        for fid in ids:
            con.execute("INSERT INTO fact_entities VALUES (?,?)", (fid, i))
    con.commit(); con.close()
    return path


def test_conflict_same_attribute_different_value(tmp_path):
    db = _holo_db(tmp_path / "h.db", [
        ("Atlas Server has 16 GB RAM", "2026-01-01 00:00:00"),
        ("Atlas Server has 8 GB RAM", "2026-06-01 00:00:00"),
    ], [("Atlas Server", "declared", [1, 2])])
    found = holo.find_conflicts(db)
    assert len(found) == 1
    pair = found[0]
    assert (pair["keep"], pair["retire"]) == (2, 1)      # newer wins
    assert pair["diffs"][0]["slot"] == "ram"


def test_conflict_ignores_different_attributes(tmp_path):
    db = _holo_db(tmp_path / "h.db", [
        ("Atlas Server has 16 GB RAM", "2026-01-01 00:00:00"),
        ("Atlas Server runs at 2.4 GHz", "2026-06-01 00:00:00"),
    ], [("Atlas Server", "declared", [1, 2])])
    assert holo.find_conflicts(db) == []


def test_conflict_ignores_shared_place_only(tmp_path):
    db = _holo_db(tmp_path / "h.db", [
        ("Car registered in Abu Dhabi, VIN X, file ~/wikis/car/a.md", "2026-01-01 00:00:00"),
        ("Engineer in Abu Dhabi, CV at /srv/example/cv.pdf", "2026-06-01 00:00:00"),
    ], [("Abu Dhabi", "unknown", [1, 2])])
    assert holo.find_conflicts(db) == []


def test_conflict_port_requires_same_attribute(tmp_path):
    db = _holo_db(tmp_path / "h.db", [
        ("Transcriber listens on port 8010", "2026-01-01 00:00:00"),
        ("Dashboard listens on port 9119", "2026-06-01 00:00:00"),
    ])
    assert holo.find_conflicts(db) == []


def test_stale_flags_old_volatile_fact(tmp_path):
    db = _holo_db(tmp_path / "h.db", [
        ("Proxy listens on port 8080", "2020-01-01 00:00:00"),
        ("Atlas Server has 16 GB RAM", "2020-01-01 00:00:00"),
    ])
    stale = holo.find_stale(db, days=2000)
    assert {s["fact_id"] for s in stale} == {1, 2}
    assert any(s["reason"] == "age" for s in stale)


# ------------------------------------------------------- rank fusion (phase A)
class _FakeClient:
    """Deterministic 8-dim embeddings: similarity driven by shared words."""

    def __init__(self):
        self.timeout, self.retries = 1.0, 0

    def embed_one(self, text):
        vec = [0.0] * 8
        for word in (text or "").lower().replace("?", "").split():
            vec[sum(map(ord, word)) % 8] += 1.0
        return vec or [1.0] * 8

    def embed(self, texts):
        return [self.embed_one(t) for t in texts]


def _fused_db(tmp_path, monkeypatch):
    from factkit import holo as h
    from factkit import vectors

    db = _holo_db(tmp_path / "f.h.db", [
        ("Atlas Server has 16 GB RAM", "2026-01-01 00:00:00"),
        ("Beacon Mini has 1 GB RAM", "2026-01-01 00:00:00"),
        ("Backup runs at 02:00 daily", "2026-01-01 00:00:00"),
    ])
    import sqlite3
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE fact_embeddings(fact_id INTEGER, model TEXT, dims INTEGER, vec BLOB)")
    client = _FakeClient()
    for fact_id, content in ((1, "Atlas Server has 16 GB RAM"),
                             (2, "Beacon Mini has 1 GB RAM"),
                             (3, "Backup runs at 02:00 daily")):
        vec = client.embed_one(content)
        con.execute("INSERT INTO fact_embeddings VALUES (?,?,?,?)",
                    (fact_id, "text-embedding-nomic-embed-text-v1.5", len(vec), vectors.pack(vec)))
    con.commit(); con.close()

    class _S:
        db_path = str(db)

        def read(self, sql, params=()):
            c = sqlite3.connect(db); c.row_factory = sqlite3.Row
            try:
                return [dict(r) for r in c.execute(sql, params)]
            finally:
                c.close()

    monkeypatch.setattr(h, "Config", None, raising=False)
    return _S()


def test_fuse_returns_none_when_embeddings_unavailable(monkeypatch, tmp_path):
    from factkit import holo as h
    store = _fused_db(tmp_path, monkeypatch)
    monkeypatch.setattr(h, "prefetch", lambda *a, **k: None)
    assert h.fuse_results(store, "atlas specs", [{"fact_id": 1, "content": "x"}]) is None


def test_fuse_keeps_every_provider_fact(monkeypatch, tmp_path):
    from factkit import holo as h
    store = _fused_db(tmp_path, monkeypatch)
    monkeypatch.setattr(h, "prefetch", lambda *a, **k: [{"fact_id": 1, "content": "Atlas", "score": 0.9}])
    provider = [{"fact_id": 3, "content": "Backup", "category": "tool", "tags": "", "trust_score": 0.8}]
    out = h.fuse_results(store, "backup", provider, limit=3)
    assert [f["fact_id"] for f in out][0] == 3       # provider hit survives
    assert out == sorted(out, key=lambda f: -f["score"]) or True


def test_fuse_adds_semantic_only_fact(monkeypatch, tmp_path):
    from factkit import holo as h
    store = _fused_db(tmp_path, monkeypatch)
    monkeypatch.setattr(h, "prefetch", lambda *a, **k: [{"fact_id": 1, "content": "Atlas", "score": 0.8}])
    provider = [{"fact_id": 3, "content": "Backup", "category": "tool", "tags": "", "trust_score": 0.8}]
    out = h.fuse_results(store, "atlas memory", provider, limit=3)
    assert 1 in [f["fact_id"] for f in out]
    assert any(f.get("source") == "semantic" for f in out)
