import json, tempfile, shutil, sqlite3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.make_live_cases import build_cases

def make_fixture():
    td = Path(tempfile.mkdtemp(prefix='test-live-'))
    db = td / 'facts.db'
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE facts (
        fact_id INTEGER PRIMARY KEY, content TEXT, category TEXT,
        tags TEXT, trust_score REAL, retrieval_count INTEGER,
        helpful_count INTEGER, created_at TEXT, updated_at TEXT,
        hrr_vector BLOB, valid_from TEXT, valid_to TEXT
    )""")
    data = [
        (1, "Car: Mazda 3 Sedan 2.5 S, 2021 (US spec), VIN JM1BPAAL5M1328615, Abu Dhabi-registered.", "entity"),
        (2, "Laptop ASUS ROG Strix G614JI (shadow-keep) — i7-13650HX, RTX 4070, 48 GB, 2×1 TB NVMe.", "entity"),
        (3, "When user asks for personal specifications, always check holographic memory first.", "user_pref"),
        (4, "Hermes backup script fix: Adding --ignore-failed-read flag to tar command.", "tool"),
        (5, "Nextcloud file ops: always use MCP nc_webdav_* tools for WebDAV operations.", "project"),
        (6, "Transcriber listens on 8010 and drives the CUDA worker at :9000.", "entity"),
        (7, "Short fact", "general"),
        (8, "Password is secret123 and api_key=abc", "user_pref"),
    ]
    con.executemany("INSERT INTO facts VALUES (?,?,?, '', 0.9, 0, 0, '2026-01-01', '2026-01-01', NULL, NULL, NULL)", data)
    con.commit()
    con.close()
    return td, db

def test_case_count_and_validity():
    td, db = make_fixture()
    try:
        cases_file = td / 'cases.json'
        cases = build_cases(db, cases_file)
        assert len(cases) >= 4
        for c in cases:
            assert 'query' in c and 'expect' in c and 'note' in c
            assert c['expect'][0] in {1,2,3,4,5,6}
    finally:
        shutil.rmtree(td)

def test_no_verbatim_queries():
    td, db = make_fixture()
    try:
        cases = build_cases(db, td / 'cases.json')
        for c in cases:
            assert len(set(c['query'].lower().split())) > 0
    finally:
        shutil.rmtree(td)

def test_no_duplicate_queries():
    td, db = make_fixture()
    try:
        cases = build_cases(db, td / 'cases.json')
        queries = [c['query'] for c in cases]
        assert len(queries) == len(set(queries))
    finally:
        shutil.rmtree(td)

def test_deterministic_output():
    td, db = make_fixture()
    try:
        c1 = build_cases(db, td / 'c1.json')
        c2 = build_cases(db, td / 'c2.json')
        assert c1 == c2
    finally:
        shutil.rmtree(td)

def test_refuses_missing_db():
    import pytest
    with pytest.raises((sqlite3.Error, FileNotFoundError)):
        build_cases(Path('/nonexistent/path.db'), Path('/tmp/x.json'))

def test_categories_represented():
    td, db = make_fixture()
    try:
        cases = build_cases(db, td / 'cases.json')
        cats = {c['note'] for c in cases}
        assert len(cats) >= 3
    finally:
        shutil.rmtree(td)
