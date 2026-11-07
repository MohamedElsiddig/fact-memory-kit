"""Fail if tests or integrations hardcode a developer machine's paths.

Rule: every scanned file under ``tests/`` and ``integrations/`` must be free of
the host-path patterns listed in ``PATTERNS`` below — an absolute home-directory
path or the installed plugin location. CI runs on a clean ``ubuntu-latest``
runner with no Hermes/holographic provider installed, so a host-bound path makes
the suite fail there. Use ``$HOME``, ``pathlib.Path(__file__).resolve()``,
``tmp_path`` or another temporary directory, or neutral example paths instead.

The forbidden strings are assembled from fragments below so that this file
never flags itself.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("tests", "integrations")
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "node_modules",
}

# Fragments keep the literals out of this source file, which lives under tests/.
_FRAGMENTS = (
    ("/", "home", "/"),
    ("/", "Users", "/"),
    ("~", "/.hermes/plugins"),
)
PATTERNS = tuple("".join(parts) for parts in _FRAGMENTS)

# 192.0.2.0/24 is reserved for documentation (RFC 5737) and is never a host
# path; neutralise it before matching so an example address cannot trip us up.
_DOC_IP = re.compile(r"\b192\.0\.2\.\d{1,3}\b")


def _iter_files():
    for root_name in SCAN_ROOTS:
        root = REPO / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if SKIP_DIRS.intersection(path.parts):
                continue
            yield path


def _read_text(path):
    """Return decoded text, or ``None`` for binary/unreadable files."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None  # binary file (e.g. a stray .pyc)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _violations():
    found = []
    for path in _iter_files():
        text = _read_text(path)
        if text is None:
            continue
        rel = path.relative_to(REPO)
        for lineno, line in enumerate(text.splitlines(), 1):
            scan_line = _DOC_IP.sub("192.0.2.0", line)
            for pattern in PATTERNS:
                if pattern in scan_line:
                    found.append(
                        f"{rel}:{lineno}: contains {pattern!r}: {line.strip()}"
                    )
    return found


def test_no_host_specific_paths():
    found = _violations()
    assert not found, (
        "Host-specific paths found in tests/ or integrations/ (use $HOME, "
        "pathlib.Path(__file__).resolve(), tmp_path/temp dirs, or neutral "
        "example paths instead):\n  " + "\n  ".join(found)
    )
