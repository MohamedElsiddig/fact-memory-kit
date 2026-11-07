"""Tests for the vendored-plugin drift checks (install.sh --check, plugin-status)."""
from __future__ import annotations

import json
import re
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INSTALL = REPO / "integrations" / "hermes" / "factkit" / "install.sh"


def _run(args, env: dict | None = None):
    full = dict(os.environ)
    full.pop("FACTKIT_CHECK_VERBOSE", None)
    full.update(env or {})
    return subprocess.run(args, cwd=REPO, env=full, text=True, capture_output=True)


def _install(tmp_path: Path):
    env = {"HERMES_PLUGINS": str(tmp_path / "plugins")}
    res = _run(["bash", str(INSTALL)], env=env)
    assert res.returncode == 0, res.stderr
    return tmp_path / "plugins" / "factkit", env


def _status(env: dict, *extra: str):
    return _run([sys.executable, "-m", "factkit.cli", "plugin-status", *extra], env=env)


def test_check_reports_ok_for_fresh_install(tmp_path):
    _, env = _install(tmp_path)
    res = _run(["bash", str(INSTALL), "--check"], env=env)
    assert res.returncode == 0
    assert res.stdout.strip() == "OK"


def test_check_never_writes_for_missing_install(tmp_path):
    missing = tmp_path / "absent"
    res = _run(["bash", str(INSTALL), "--check"], env={"HERMES_PLUGINS": str(missing)})
    assert res.returncode == 1
    assert "DRIFT factkit/holo.py" in res.stdout
    assert not missing.exists()


def test_check_names_the_drifted_file_and_reinstall_restores(tmp_path):
    dest, env = _install(tmp_path)
    holo = dest / "factkit" / "holo.py"
    original = holo.read_bytes()
    holo.write_bytes(original + b"\n# drift probe\n")

    res = _run(["bash", str(INSTALL), "--check"], env=env)
    assert res.returncode == 1
    assert res.stdout.splitlines() == ["DRIFT factkit/holo.py"]

    _run(["bash", str(INSTALL)], env=env)
    assert holo.read_bytes() == original
    assert _run(["bash", str(INSTALL), "--check"], env=env).returncode == 0


def test_plugin_status_matches_clean_install(tmp_path):
    _, env = _install(tmp_path)
    res = _status(env)
    assert res.returncode == 0
    manifest_version = re.search(r"^version:\s*(\S+)", (REPO / "integrations/hermes/factkit/plugin.yaml").read_text(), re.M).group(1)
    assert f"version  {manifest_version}" in res.stdout
    assert "verdict  MATCH" in res.stdout


def test_plugin_status_reports_drift_as_json(tmp_path):
    dest, env = _install(tmp_path)
    with (dest / "factkit" / "holo.py").open("a", encoding="utf-8") as handle:
        handle.write("\n# drift probe\n")

    res = _status(env, "--json")
    assert res.returncode == 1
    payload = json.loads(res.stdout)
    assert payload["verdict"] == "DRIFT"
    statuses = {row["name"]: row["status"] for row in payload["files"]}
    assert statuses["factkit/holo.py"] == "drift"
    assert statuses["factkit/embed.py"] == "match"


def test_plugin_status_missing_install_exits_two(tmp_path):
    env = {"HERMES_PLUGINS": str(tmp_path / "absent")}
    res = _status(env)
    assert res.returncode == 2
    assert "not installed" in res.stderr
    assert "Traceback" not in res.stderr


DIST_OWNED = ("update.yaml", "README.md", "LICENSE", "config.yaml.example")


def test_check_ignores_dist_owned_extras(tmp_path):
    dest, env = _install(tmp_path)
    for name in DIST_OWNED:
        (dest / name).write_text("# dist-owned\n", encoding="utf-8")

    res = _run(["bash", str(INSTALL), "--check"], env=env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert res.stdout.strip() == "OK"


def test_plugin_status_ignores_dist_owned_extras(tmp_path):
    dest, env = _install(tmp_path)
    for name in DIST_OWNED:
        (dest / name).write_text("# dist-owned\n", encoding="utf-8")

    res = _status(env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "verdict  MATCH" in res.stdout

    payload = json.loads(_status(env, "--json").stdout)
    assert payload["verdict"] == "MATCH"
    names = {row["name"] for row in payload["files"]}
    assert not names & set(DIST_OWNED)


def test_unknown_extra_still_reports_drift(tmp_path):
    dest, env = _install(tmp_path)
    (dest / "rogue.txt").write_text("not managed\n", encoding="utf-8")

    res = _run(["bash", str(INSTALL), "--check"], env=env)
    assert res.returncode == 1
    assert "DRIFT rogue.txt" in res.stdout

    res = _status(env)
    assert res.returncode == 1
    assert "verdict  DRIFT" in res.stdout

    payload = json.loads(_status(env, "--json").stdout)
    statuses = {row["name"]: row["status"] for row in payload["files"]}
    assert statuses["rogue.txt"] == "extra"


def test_verbose_lists_ignored_dist_owned_extras(tmp_path):
    dest, env = _install(tmp_path)
    (dest / "update.yaml").write_text("# dist-owned\n", encoding="utf-8")

    res = _run(
        ["bash", str(INSTALL), "--check"], env={**env, "FACTKIT_CHECK_VERBOSE": "1"}
    )
    assert res.returncode == 0
    assert "IGNORED update.yaml" in res.stdout
    assert res.stdout.strip().splitlines()[-1] == "OK"

    res = _status({**env, "FACTKIT_CHECK_VERBOSE": "1"}, "--json")
    payload = json.loads(res.stdout)
    assert payload["verdict"] == "MATCH"
    assert "update.yaml" in payload["ignored"]
