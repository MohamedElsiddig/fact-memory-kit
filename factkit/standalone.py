"""Runtime shims for running the maintenance CLI outside the Hermes gateway.

The holographic provider's ``store.py`` imports two helpers that only exist
inside a Hermes install. When factkit runs from the CLI (no gateway process)
those imports fail, so minimal stand-ins are registered before the store module
is loaded.

This module is deliberately NOT vendored into the Hermes plugin: the gateway
already provides the real helpers, and publishing rules forbid a plugin from
rebinding core modules.
"""
from __future__ import annotations

import sqlite3
import sys
import types


def _wal(conn, **_kw):
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error:
        pass


def _add_column(conn, table, column, decl):
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def ensure() -> None:
    """Register the two helpers if the running environment lacks them."""
    try:
        import hermes_state_wal  # noqa: F401
    except ImportError:
        mod = types.ModuleType("hermes_state_wal")
        mod.apply_wal_with_fallback = _wal
        sys.modules["hermes_state_wal"] = mod

    try:
        from hermes_cli.sqlite_util import add_column_if_missing  # noqa: F401
    except ImportError:
        pkg = types.ModuleType("hermes_cli")
        pkg.__path__ = []  # namespace only, no discovery
        util = types.ModuleType("hermes_cli.sqlite_util")
        util.add_column_if_missing = _add_column
        sys.modules.setdefault("hermes_cli", pkg)
        sys.modules["hermes_cli.sqlite_util"] = util
