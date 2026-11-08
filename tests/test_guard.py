"""Write-path guard: duplicates, conflicts and near-duplicates before insert."""
import pytest

from factkit import guard, holo
from factkit.store import Store

ATLAS = "Atlas Server"


def _store(path, facts=()):
    """A tmp store populated through the plugin's own write API."""
    if path.is_dir():
        path = path / "guard.db"
    store = Store(path)
    for fact in facts:
        if isinstance(fact, str):
            fact = {"content": fact}
        store.add_fact_status(
            fact["content"],
            category=fact.get("category", "tool"),
            tags="",
            trust=fact.get("trust", 0.7),
            entities=fact.get("entities", [ATLAS]),
        )
    return store


def _add(store, content, entities=(ATLAS,), category="tool", trust=0.7):
    fact_id, _created = store.add_fact_status(
        content, category=category, tags="", trust=trust, entities=list(entities)
    )
    return fact_id


# ------------------------------------------------------------------ duplicates
def test_identical_text_is_duplicate(tmp_path):
    store = _store(tmp_path)
    fact_id = _add(store, "Atlas Server runs the metrics collector")

    result = guard.check(store, "Atlas Server runs the metrics collector")

    assert result["verdict"] == "duplicate"
    assert result["matches"][0]["fact_id"] == fact_id
    assert result["matches"][0]["reason"] == "duplicate"
    assert result["matches"][0]["similarity"] == 1.0
    assert "update" in result["suggestion"].lower()


def test_normalisation_makes_punctuation_and_spacing_duplicate(tmp_path):
    store = _store(tmp_path)
    _add(store, "Atlas Server runs the metrics collector.")

    result = guard.check(store, "  atlas   server RUNS the metrics collector  ")

    assert result["verdict"] == "duplicate"


def test_reworded_same_fact_is_duplicate(tmp_path):
    store = _store(tmp_path)
    fact_id = _add(store, "Atlas Server runs the metrics collector")

    result = guard.check(store, "Atlas Server runs the metric collector")

    assert result["verdict"] == "duplicate"
    assert result["matches"][0]["fact_id"] == fact_id
    assert result["matches"][0]["similarity"] >= 0.98


def test_differing_numbers_are_not_duplicates(tmp_path):
    store = _store(tmp_path)
    _add(store, "Filler fact 1")

    result = guard.check(store, "Filler fact 2")

    assert result["verdict"] == "ok"
    assert result["matches"] == []


# ------------------------------------------------------------------- conflicts
def test_same_subject_different_ram_is_conflict_with_diff(tmp_path):
    store = _store(tmp_path)
    fact_id = _add(store, "Atlas Server has 16GB RAM")

    result = guard.check(store, "Atlas Server has 32GB RAM")

    assert result["verdict"] == "conflict"
    match = result["matches"][0]
    assert match["fact_id"] == fact_id and match["reason"] == "conflict"
    diff = match["diffs"][0]
    assert diff["slot"] == "ram"
    assert diff["candidate"] == ["32gb"]
    assert diff["existing"] == ["16gb"]
    assert "ram" in result["suggestion"]


def test_same_entity_different_port_is_conflict(tmp_path):
    store = _store(tmp_path)
    fact_id = _add(store, "Atlas Server listens on port 8001")

    result = guard.check(store, "Atlas Server listens on port 8080")

    assert result["verdict"] == "conflict"
    match = result["matches"][0]
    assert match["fact_id"] == fact_id
    assert match["diffs"][0]["slot"] == "port"
    assert match["diffs"][0]["candidate"] == ["8080"]
    assert match["diffs"][0]["existing"] == ["8001"]


def test_single_port_inside_declared_range_is_ok(tmp_path):
    store = _store(tmp_path)
    _add(store, "Atlas Server serves on port 8000-8100")

    result = guard.check(store, "Atlas Server serves on port 8080")

    assert result["verdict"] == "ok"
    assert result["matches"] == []


def test_port_on_a_different_entity_is_ok(tmp_path):
    store = _store(tmp_path)
    _add(store, "Atlas Server listens on port 8001")

    result = guard.check(store, "Beacon Service listens on port 8080")

    assert result["verdict"] == "ok"


def test_different_ip_on_shared_entity_is_conflict(tmp_path):
    store = _store(tmp_path)
    _add(store, "Atlas Server listens on 192.0.2.10")

    result = guard.check(store, "Atlas Server listens on 192.0.2.11")

    assert result["verdict"] == "conflict"
    assert result["matches"][0]["diffs"][0]["slot"] == "ip"


def test_port_conflict_uses_extracted_entity_when_undeclared(tmp_path):
    store = _store(tmp_path)
    _add(store, "Atlas Server listens on port 8001", entities=[])

    result = guard.check(store, "Atlas Server listens on port 8080")

    assert result["verdict"] == "conflict"
    assert result["matches"][0]["diffs"][0]["slot"] == "port"


def test_capacity_units_normalise_to_agreement(tmp_path):
    store = _store(tmp_path)
    _add(store, "Atlas Server has 16GB RAM")

    result = guard.check(store, "Atlas Server has 16384MB RAM")

    assert result["verdict"] == "ok"
    assert result["matches"] == []


# ------------------------------------------------------------------------ near
def test_near_duplicate_suggests_merging_wording(tmp_path):
    store = _store(tmp_path)
    fact_id = _add(store, "Atlas Server keeps backups on the secondary disk")

    result = guard.check(store, "Atlas Server keeps backups on the second disk")

    assert result["verdict"] == "near"
    match = result["matches"][0]
    assert match["fact_id"] == fact_id and match["reason"] == "near"
    assert 0.90 <= match["similarity"] < 0.98
    assert "merge" in result["suggestion"].lower()


def test_guard_similarity_matches_merge_duplicates(tmp_path):
    first = "Atlas Server keeps backups on the secondary disk"
    second = "Atlas Server keeps backups on the second disk"
    store = _store(tmp_path / "single.db", [first])
    pair_store = _store(tmp_path / "pair.db", [first, second])

    pair = holo.merge_duplicates(pair_store.db_path, threshold=0.90)[0]
    result = guard.check(store, second)

    assert result["matches"][0]["similarity"] == pytest.approx(pair["ratio"], abs=1e-3)


# ------------------------------------------------------------------- baselines
def test_empty_store_is_ok(tmp_path):
    store = _store(tmp_path / "empty.db")

    result = guard.check(store, "Atlas Server has 16GB RAM")

    assert result == {"verdict": "ok", "matches": [], "suggestion": None}


def test_unrelated_fact_is_ok(tmp_path):
    store = _store(tmp_path)
    _add(store, "Atlas Server runs the metrics collector")

    result = guard.check(store, "Backup retention is 30 days")

    assert result["verdict"] == "ok"
    assert result["matches"] == []


def test_check_does_not_write(tmp_path):
    store = _store(tmp_path)
    _add(store, "Atlas Server has 16GB RAM")
    before = store.count_facts()

    guard.check(store, "Atlas Server has 32GB RAM")

    assert store.count_facts() == before


# --------------------------------------------------------------------- report
def test_format_report_one_line_per_match(tmp_path):
    store = _store(tmp_path)
    fact_id = _add(store, "Atlas Server listens on port 8001")
    result = guard.check(store, "Atlas Server listens on port 8080")

    report = guard.format_report(result)
    lines = report.splitlines()

    assert len(lines) == len(result["matches"]) == 1
    assert "CONFLICT" in lines[0]
    assert f"#{fact_id}" in lines[0]
    assert "port: 8080 vs 8001" in lines[0]
    assert "Atlas Server listens on port 8001" in lines[0]


def test_format_report_ok_line(tmp_path):
    store = _store(tmp_path)
    result = guard.check(store, "Atlas Server has 16GB RAM")

    assert guard.format_report(result).startswith("OK")


def test_check_accepts_category_and_trust(tmp_path):
    store = _store(tmp_path)
    _add(store, "Atlas Server runs the metrics collector", trust=0.5)

    result = guard.check(store, "Atlas Server runs the metrics collector",
                         category="project", trust=0.9)

    assert result["verdict"] == "duplicate"
    assert "project" in result["suggestion"]
    assert "trust" in result["suggestion"].lower()
