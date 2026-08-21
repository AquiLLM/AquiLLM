from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "task21_hybrid_failure_bundle.py"
CLI = REPO / "scripts" / "task21_hybrid_evidence_cli.py"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.mark.parametrize(
    "value",
    (
        "postgresql://aquillm:canary%40pw@db:5432/aquillm",
        "postgres://aquillm:canary%2Fpw@db/aquillm",
        "memgraph://projection:canary%3Apw@graph:7687",
        "bolt+s://projection:canary%23pw@graph:7687",
        "https://client:canary%2Btoken@example.test/v1",
        "http://client:canary-secret@example.test/v1",
    ),
)
def test_url_credentials_are_redacted_without_encoded_canary_leakage(value):
    module = _module("task21_hybrid_failure_bundle", SCRIPT)

    redacted = module.redact_log(f"backend={value}")

    assert "canary" not in redacted
    assert "<redacted>@" in redacted


class FakeRunner:
    def __init__(self, module, outputs=None):
        self.module = module
        self.outputs = outputs or {}
        self.calls = []

    def run(self, arguments, **_kwargs):
        command = tuple(arguments)
        self.calls.append(command)
        return self.outputs.get(command, self.module.CommandResult("", "", 0))


def test_command_result_preserves_signal_return_code():
    module = _module("task21_hybrid_failure_bundle", SCRIPT)

    assert module.CommandResult("", "terminated", -15).returncode == -15


def test_capture_records_failed_commands_instead_of_converting_them_to_empty_success():
    module = _module("task21_hybrid_failure_bundle", SCRIPT)
    prefix = ("docker", "compose", "-p", "aquillm-task21-failed")
    outputs = {
        prefix + ("config", "--no-interpolate"): module.CommandResult(
            "", "daemon password=canary-secret", 17
        )
    }
    runner = FakeRunner(module, outputs)

    captured = module.capture_runtime(runner, prefix)
    state = json.loads(captured.members["runtime/state.json"])

    assert captured.complete is False
    assert state["command_errors"][0]["returncode"] == 17
    assert "canary-secret" not in json.dumps(state)


@pytest.mark.parametrize("failed_operation", ("down", "listing"))
def test_cleanup_requires_successful_down_and_three_successful_zero_samples(
    failed_operation,
):
    module = _module("task21_hybrid_failure_bundle", SCRIPT)
    prefix = ("docker", "compose", "-p", "aquillm-task21-cleanup")
    label = "com.docker.compose.project=aquillm-task21-cleanup"
    down = prefix + ("down", "--volumes", "--remove-orphans")
    containers = ("docker", "ps", "-aq", "--filter", f"label={label}")
    outputs = {}
    outputs[down if failed_operation == "down" else containers] = module.CommandResult(
        "", "cleanup denied", 9
    )
    runner = FakeRunner(module, outputs)

    cleanup = module.cleanup_runtime(runner, prefix, project_label=label)

    assert cleanup["complete"] is False
    assert len(cleanup["zero_samples"]) == 3
    assert cleanup["command_errors"]
    assert cleanup["command_errors"][0]["returncode"] == 9


def test_cli_success_requires_capture_cleanup_and_clean_source():
    module = _module("task21_hybrid_failure_bundle", SCRIPT)
    cli = _module("task21_hybrid_evidence_cli", CLI)
    captured = module.CapturedRuntime({}, {}, "a" * 64, complete=True)
    cleanup = {"complete": True}
    source = {"commit": "b" * 40, "clean": True}

    assert cli._evidence_exit_code(captured, cleanup, source) == 0
    assert cli._evidence_exit_code(captured, {"complete": False}, source) == 2
    assert cli._evidence_exit_code(captured, cleanup, source | {"clean": False}) == 2


def test_evidence_source_status_includes_untracked_files():
    module = _module("task21_hybrid_failure_bundle", SCRIPT)
    cli = _module("task21_hybrid_evidence_cli", CLI)
    status = ("git", "status", "--porcelain", "--untracked-files=normal")
    runner = FakeRunner(
        module,
        {
            ("git", "rev-parse", "HEAD"): module.CommandResult("c" * 40, "", 0),
            status: module.CommandResult("?? post-preflight.txt\n", "", 0),
        },
    )

    assert cli._source(runner) == {"commit": "c" * 40, "clean": False}
    assert status in runner.calls
