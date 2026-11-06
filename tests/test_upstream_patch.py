"""Guards for the upstream provider-rerank proposal.

These tests read the artifacts only: they parse the unified diff, check that it
is minimal and still applies to the installed module, and audit the proposal
document. They never apply the patch and never import the provider.

Skipped checks (the hunk match) need the installed provider at its default
path; everywhere else the suite is offline and host-independent.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
PATCH = UPSTREAM / "rerank.patch"
DOC = UPSTREAM / "provider-rerank.md"
PROVIDER = pathlib.Path.home() / ".hermes" / "plugins" / "holographic" / "retrieval.py"

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)")


# ------------------------------------------------------------------ parsing
class Hunk:
    def __init__(self, match: re.Match):
        self.old_start = int(match.group(1))
        self.old_len = int(match.group(2) or 1)
        self.new_start = int(match.group(3))
        self.new_len = int(match.group(4) or 1)
        self.lines: list[tuple[str, str]] = []

    @property
    def old_text(self) -> list[str]:
        return [text for tag, text in self.lines if tag in (" ", "-")]

    @property
    def new_text(self) -> list[str]:
        return [text for tag, text in self.lines if tag in (" ", "+")]


class FileDiff:
    def __init__(self, old: str, new: str):
        self.old = old
        self.new = new
        self.hunks: list[Hunk] = []


def parse_patch(text: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    for raw in text.splitlines():
        if raw.startswith("--- "):
            files.append(FileDiff(raw[4:].split("\t")[0].strip(), ""))
        elif raw.startswith("+++ ") and files:
            files[-1].new = raw[4:].split("\t")[0].strip()
        elif raw.startswith("@@") and files:
            match = HUNK_RE.match(raw)
            assert match, f"malformed hunk header: {raw!r}"
            files[-1].hunks.append(Hunk(match))
        elif files and files[-1].hunks and raw[:1] in ("+", "-", " "):
            files[-1].hunks[-1].lines.append((raw[:1], raw[1:]))
        elif raw.startswith("\\"):
            continue  # "\ No newline at end of file"
    return files


@pytest.fixture(scope="module")
def patch_text() -> str:
    assert PATCH.is_file(), f"missing {PATCH}"
    return PATCH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def doc_text() -> str:
    assert DOC.is_file(), f"missing {DOC}"
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def diffs(patch_text: str) -> list[FileDiff]:
    parsed = parse_patch(patch_text)
    assert parsed, "the patch file contains no file diff"
    return parsed


def added_text(diffs: list[FileDiff]) -> str:
    return "\n".join(text for diff in diffs for hunk in diff.hunks
                     for tag, text in hunk.lines if tag == "+")


def removed_text(diffs: list[FileDiff]) -> str:
    return "\n".join(text for diff in diffs for hunk in diff.hunks
                     for tag, text in hunk.lines if tag == "-")


# ------------------------------------------------------------------- tests
def test_patch_is_a_well_formed_unified_diff(diffs):
    for diff in diffs:
        assert diff.old and diff.new, "each file diff needs --- and +++ headers"
        for hunk in diff.hunks:
            assert len(hunk.old_text) == hunk.old_len, (
                f"hunk at old line {hunk.old_start}: header says {hunk.old_len} "
                f"old lines, body has {len(hunk.old_text)}")
            assert len(hunk.new_text) == hunk.new_len, (
                f"hunk at new line {hunk.new_start}: header says {hunk.new_len} "
                f"new lines, body has {len(hunk.new_text)}")


def test_patch_touches_only_the_retrieval_module(diffs):
    assert len(diffs) == 1, f"patch touches {len(diffs)} files, expected 1"

    def strip_prefix(path: str) -> str:
        head, _, tail = path.partition("/")
        return tail if head in ("a", "b") else path

    paths = {strip_prefix(diffs[0].old), strip_prefix(diffs[0].new)}
    assert paths == {"retrieval.py"}, paths


def test_new_api_names_are_present(diffs):
    added = added_text(diffs)
    for name in ("rerank", "candidate_window", "rerank_timeout_ms", "rerank_hook", "_apply_rerank"):
        assert name in added, f"{name} does not appear in any added line"
    assert "Callable[[str, list[dict], int]" in added, "hook signature missing"


def test_no_existing_function_is_deleted(diffs):
    for diff in diffs:
        for hunk in diff.hunks:
            old_defs = {m.group(1) for text in hunk.old_text for m in [DEF_RE.match(text)] if m}
            new_defs = {m.group(1) for text in hunk.new_text for m in [DEF_RE.match(text)] if m}
            assert old_defs <= new_defs, f"function(s) removed: {sorted(old_defs - new_defs)}"
            for tag, text in hunk.lines:
                assert not (tag == "-" and DEF_RE.match(text)), f"deleted a def line: {text!r}"


def test_hook_and_window_default_to_todays_behaviour(diffs):
    added = added_text(diffs)
    assert "candidate_window: int | None = None" in added
    assert "rerank: Callable" in added and "= None" in added.split("rerank: Callable", 1)[1].split("\n")[0]
    assert "if self.rerank_hook is not None" in added, "hook is not guarded"
    assert "self.candidate_window or limit * 3" in added, "old window not preserved as the fallback"


def test_patch_is_minimal(diffs):
    added = sum(1 for diff in diffs for hunk in diff.hunks for tag, _ in hunk.lines if tag == "+")
    removed = sum(1 for diff in diffs for hunk in diff.hunks for tag, _ in hunk.lines if tag == "-")
    assert added + removed <= 120, f"{added + removed} changed lines, limit is 120"
    assert added >= 20, "patch looks truncated"


def test_hunks_still_match_the_installed_provider(diffs):
    if not PROVIDER.is_file():
        pytest.skip("installed provider not present on this host")
    lines = PROVIDER.read_text(encoding="utf-8").splitlines()
    for diff in diffs:
        for hunk in diff.hunks:
            start = hunk.old_start - 1
            actual = lines[start:start + hunk.old_len]
            assert actual == hunk.old_text, (
                f"hunk at line {hunk.old_start} no longer matches the installed module; "
                "regenerate upstream/rerank.patch against the current retrieval.py")


def test_doc_covers_failure_modes_and_rollback(doc_text):
    lowered = doc_text.lower()
    for topic in ("timeout", "rollback", "endpoint down", "latency", "fusion", "reserve"):
        assert topic in lowered, f"proposal does not discuss {topic!r}"


def test_doc_states_the_patch_is_not_applied(doc_text):
    assert "not applied" in doc_text.lower()


def test_doc_presents_both_alternatives(doc_text):
    lowered = doc_text.lower()
    assert "two alternatives" in lowered
    assert "candidate window only" in lowered
    assert "rerank` hook + `candidate_window" in doc_text


def test_doc_contains_no_real_hostnames_or_addresses(doc_text):
    bad_ips = [ip for ip in re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", doc_text)
               if not ip.startswith("192.0.2.")]
    assert not bad_ips, f"non-documentation IP(s) in proposal: {bad_ips}"

    ipv6 = re.findall(r"(?<![\w:])(?:[0-9a-fA-F]{1,4}:){2,}[0-9a-fA-F]{0,4}(?![\w:])", doc_text)
    assert not ipv6, f"IPv6 address(es) in proposal: {ipv6}"

    hosts = re.findall(r"\b[a-z0-9][a-z0-9-]*\.(?:local|lan|internal|home|corp|com|net|org|io|dev|ai)\b",
                       doc_text, flags=re.IGNORECASE)
    assert not hosts, f"hostname(s) in proposal: {hosts}"

    emails = re.findall(r"[\w.+-]+@[\w.-]+\.\w+", doc_text)
    assert not emails, f"email address(es) in proposal: {emails}"

    urls = re.findall(r"https?://[^\s)`]+", doc_text)
    assert all(re.match(r"https?://192\.0\.2\.\d+", url) for url in urls), urls
