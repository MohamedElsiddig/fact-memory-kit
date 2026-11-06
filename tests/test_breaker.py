"""Circuit-breaker tests for the factkit Hermes plugin.

The plugin lives outside this repo, so it is loaded by path as package
``fkplugin`` (with the plugin directory as the search location) to make its
relative ``from .factkit import holo`` imports resolve. No store is opened,
no embedder is contacted: prefetch is always patched.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from unittest.mock import Mock

import pytest

PLUGIN = "/home/melsiddig/.hermes/plugins/factkit/__init__.py"
PLUGIN_DIR = "/home/melsiddig/.hermes/plugins/factkit"

QUERY = "what are the specs of my atlas server"
BREAKER_SECONDS = 300.0


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "fkplugin", PLUGIN, submodule_search_locations=[PLUGIN_DIR]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["fkplugin"] = module
    spec.loader.exec_module(module)
    return module


m = _load_plugin()
holo = importlib.import_module("fkplugin.factkit.holo")


@pytest.fixture(autouse=True)
def breaker(monkeypatch):
    """Fixed config and a clean breaker around every test.

    Pinning ``_CFG`` keeps the tests independent of the real config.yaml
    (which carries the embed endpoint) while still exercising the real
    ``_num``/``_enabled``/``_limit`` code paths.
    """
    monkeypatch.setattr(
        m,
        "_CFG",
        {
            "prefetch": True,
            "limit": 3,
            "min_score": 0.55,
            "embed_timeout_ms": 1500,
            "breaker_seconds": BREAKER_SECONDS,
        },
    )
    m._breaker_reset()
    yield m._BREAKER
    m._breaker_reset()


def _trip(times=1):
    for _ in range(times):
        m._breaker_trip()


# ------------------------------------------------------------------ breaker
def test_three_consecutive_failures_open_breaker():
    started = time.time()
    _trip(2)
    assert m._BREAKER["fails"] == 2
    assert not m._breaker_open()          # two failures are not enough

    _trip()
    assert m._BREAKER["fails"] == 3
    assert m._breaker_open()
    # open window is breaker_seconds (300 s) from the moment of the third trip
    assert started + BREAKER_SECONDS - 1 <= m._BREAKER["until"] <= time.time() + BREAKER_SECONDS


def test_reset_closes_breaker_and_zeroes_fails():
    _trip(3)
    assert m._breaker_open()

    m._breaker_reset()
    assert not m._breaker_open()
    assert m._BREAKER == {"fails": 0, "until": 0.0}


def test_open_breaker_never_attempts_prefetch(monkeypatch):
    _trip(3)
    assert m._breaker_open()

    prefetch = Mock(return_value=[])
    monkeypatch.setattr(holo, "prefetch", prefetch)

    # documented hook call shape
    result = m.on_pre_llm_call(
        message=QUERY, conversation_history=[], user_message=QUERY
    )
    assert result == []
    assert prefetch.call_count == 0
    assert m._BREAKER["fails"] == 3       # still open, not re-tripped
    assert m._breaker_open()

    # the same call without ``user_message`` must also stay inert
    assert m.on_pre_llm_call(message=QUERY, conversation_history=[]) == []
    assert prefetch.call_count == 0


def test_success_after_two_failures_resets_counter():
    _trip(2)
    assert m._BREAKER["fails"] == 2
    assert not m._breaker_open()

    m._breaker_reset()                    # what a successful prefetch does

    _trip()
    assert m._BREAKER["fails"] == 1
    assert not m._breaker_open()


# ------------------------------------------------------- hook path sanity
def test_prefetch_runs_when_breaker_closed(monkeypatch):
    """Guards the test above: the hook really reaches prefetch when closed."""
    monkeypatch.setattr(m, "_store", lambda: object())
    prefetch = Mock(return_value=[])
    monkeypatch.setattr(holo, "prefetch", prefetch)

    result = m.on_pre_llm_call(
        message=QUERY, conversation_history=[], user_message=QUERY
    )
    assert result == []
    assert prefetch.call_count == 1
    assert m._BREAKER["fails"] == 0
