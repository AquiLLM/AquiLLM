"""Storage infrastructure is opt-in and cannot change serving or sidecar defaults."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from .compose_render_test_support import render_compose_with_reviewed_env
from .test_kv_storage_plan import layout, limits, env

ROOT = Path(__file__).resolve().parents[3]
FILE = ROOT / "deploy/compose/kv-storage.yml"


@pytest.mark.parametrize("mode,services", [
    ("off", set()), ("kv-local", {"lmcache", "kv-storage-preflight"}),
    ("kv-mooncake", {"lmcache", "mooncake", "kv-storage-preflight"}),
])
def test_compose_profile_isolation(mode, services):
    assert FILE.exists(), "storage Compose missing"
    result = render_compose_with_reviewed_env((FILE,), profile=mode)
    assert set(result["services"]) == services
    if services:
        assert result["services"]["lmcache"]["ipc"] == "shareable"
        assert result["services"]["kv-storage-preflight"]["ipc"] == "service:lmcache"
        assert result["services"]["lmcache"]["gpus"]
        assert "env_file" not in result["services"]["lmcache"]
        assert not result["services"]["lmcache"].get("ports")


def test_planning_cli_emits_real_json(layout, limits, tmp_path):
    paths = []
    for name, data in (("layout", layout), ("limits", limits)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data))
        paths += [f"--{name}", str(path)]
    result = subprocess.run([sys.executable, str(ROOT / "deploy/scripts/kv_storage_plan.py"), *paths],
                            env={**os.environ, **env()}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["chunk_tokens"] == 32


def test_capacity_planning_reports_storage_and_pager_independently():
    sys.path.insert(0, str(ROOT / "deploy/scripts"))
    config = importlib.import_module("kv_cache_config")
    profile = config._resolve({"KV_CACHE_STORAGE_MODE": "mooncake", "KV_CACHE_EXECUTION_MODE": "paged"}, [], planning=True)
    assert profile.get("storage_readiness") == "blocked-on-mixed-group-registration"
    assert profile.get("active_pager") == "not-implemented"


def test_infrastructure_uses_validated_plan_not_mutated_argv(layout, limits):
    sys.path.insert(0, str(ROOT / "deploy/scripts"))
    planner = importlib.import_module("kv_storage_plan")
    plan = planner.plan_storage(env(), layout, limits)
    plan["server_argv"] = ["arbitrary-command"]
    preflight = importlib.import_module("kv_storage_preflight")
    assert hasattr(preflight, "reconstruct_plan"), "safe plan revalidation missing"
    restored = preflight.reconstruct_plan(plan)
    assert restored["server_argv"][:2] == ["lmcache", "server"]
    assert restored["status"] == "experimental-not-launchable"


def write_operator_utf8_json(path, data):
    payload = json.dumps(data)
    if os.name == "nt":
        # Exercise the runbook's real Windows PowerShell 5.1 producer.
        result = subprocess.run([
            "powershell.exe", "-NoProfile", "-Command",
            "$env:KV_TEST_JSON | Set-Content -LiteralPath $env:KV_TEST_JSON_PATH -Encoding utf8; $PSVersionTable.PSVersion.ToString()",
        ], env={**os.environ, "KV_TEST_JSON": payload, "KV_TEST_JSON_PATH": str(path)},
            capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().startswith("5.1.")
    else:
        # Preserve coverage on Linux CI with the same UTF-8 BOM bytes.
        path.write_text(payload, encoding="utf-8-sig")
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


@pytest.mark.parametrize("bom_input", ["layout", "limits"])
def test_planner_cli_accepts_powershell_utf8_inputs(layout, limits, tmp_path, bom_input):
    arguments = []
    for name, data in (("layout", layout), ("limits", limits)):
        path = tmp_path / f"{name}.json"
        if name == bom_input:
            write_operator_utf8_json(path, data)
        else:
            path.write_text(json.dumps(data), encoding="utf-8")
        arguments += [f"--{name}", str(path)]
    result = subprocess.run([sys.executable, str(ROOT / "deploy/scripts/kv_storage_plan.py"), *arguments],
                            env={**os.environ, **env()}, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["chunk_tokens"] == 32


@pytest.mark.parametrize("flag", ["--plan", "--server-plan", "--master-plan"])
def test_preflight_cli_reads_powershell_json_before_identity_gate(layout, limits, tmp_path, flag):
    sys.path.insert(0, str(ROOT / "deploy/scripts"))
    planner = importlib.import_module("kv_storage_plan")
    plan = planner.plan_storage(env(), layout, limits)
    plan["layout"]["identity"]["model"] = "incorrect-model"
    path = tmp_path / "plan.json"
    write_operator_utf8_json(path, plan)
    result = subprocess.run([sys.executable, str(ROOT / "deploy/scripts/kv_storage_preflight.py"), flag, str(path)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 64, result.stderr
    assert "layout identity must match" in result.stderr
    assert "Traceback" not in result.stderr
