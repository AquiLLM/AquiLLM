from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from task21_hybrid_live_trace_support import valid_trace

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "task21_hybrid_failure_bundle.py"
CLI = REPO / "scripts" / "task21_hybrid_evidence_cli.py"
SHELL = REPO / "scripts" / "run_task21_hybrid_cloud_eval.sh"


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
        "endpoints=https://public.test/path,postgresql://user:canary@db/a",
        "endpoint=https://public.test/path?next=bolt://user:canary@graph:7687",
        "endpoint=https://public.test/redirect/https://inner:canary@host/path",
        "endpoint=https://public.test/path#next=bolt://user:canary@graph:7687",
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


@pytest.mark.parametrize(
    ("service_output", "inspect_state", "expected_error"),
    (
        ("", None, "service_missing"),
        ("container", ("exited", 0), "service_not_running"),
        ("container", ("running", 137), "service_exit_unsuccessful"),
    ),
)
def test_capture_requires_every_fixed_service_running_with_successful_exit(
    service_output,
    inspect_state,
    expected_error,
):
    module = _module("task21_hybrid_failure_bundle", SCRIPT)
    prefix = ("docker", "compose", "-p", "aquillm-task21-runtime")
    outputs = {prefix + ("config", "--no-interpolate"): module.CommandResult(
        "services: {}\n", "", 0
    )}
    for index, service in enumerate(module.SERVICES, start=1):
        container = f"{index:x}" * 64
        image = f"{(index + 7) % 16:x}" * 64
        outputs[prefix + ("ps", "--all", "--quiet", service)] = (
            module.CommandResult(container, "", 0)
        )
        outputs[module.inspect_command(container)] = module.CommandResult(
            f"{container}\tsha256:{image}\trunning\t0", "", 0
        )
    service = module.SERVICES[0]
    container = "1" * 64
    if not service_output:
        outputs[prefix + ("ps", "--all", "--quiet", service)] = (
            module.CommandResult("", "", 0)
        )
    else:
        status, exit_code = inspect_state
        outputs[module.inspect_command(container)] = module.CommandResult(
            f"{container}\tsha256:{'8' * 64}\t{status}\t{exit_code}", "", 0
        )

    captured = module.capture_runtime(FakeRunner(module, outputs), prefix)
    state = json.loads(captured.members["runtime/state.json"])

    assert captured.complete is False
    assert state[service]["capture_error"] == expected_error


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


def _publish_fixture(module, tmp_path):
    artifacts = {}
    bodies = {
        "arm_results": '{"arms":{}}',
        "timings": json.dumps(
            {
                "elapsed_ms": 1.0,
                "finished_ns": 2,
                "original_exit_code": 0,
                "started_ns": 1,
                "status": "passed",
            }
        ),
        "projection_checksums": json.dumps(
            {"generation_key": "a" * 64, "projection_checksum": "b" * 64}
        ),
        "live_trace": '{"schema":"task21-hybrid-live-trace-v1"}',
    }
    for name, body in bodies.items():
        artifacts[name] = tmp_path / f"{name}.json"
        artifacts[name].write_text(body, encoding="utf-8")
    return {
        "output_root": tmp_path / "evidence",
        "run_id": "d" * 32,
        "captured": module.CapturedRuntime({}, {}, "c" * 64),
        "artifacts": artifacts,
        "cleanup_proof": {"complete": True},
        "source": {"commit": "e" * 40, "clean": True},
        "expected_source_commit": "e" * 40,
        "claim_scope": "cloud",
        "signing_key": b"task23-signing-key",
        "signing_key_version": "task23-v2",
    }


def test_publisher_rejects_not_completed_timings(tmp_path):
    module = _module("task21_hybrid_failure_bundle", SCRIPT)
    arguments = _publish_fixture(module, tmp_path)
    arguments["artifacts"]["timings"].write_text(
        '{"status":"not_completed"}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="timings"):
        module.publish_bundle(**arguments)


def test_publisher_rejects_post_capture_source_commit_change(tmp_path):
    module = _module("task21_hybrid_failure_bundle", SCRIPT)
    arguments = _publish_fixture(module, tmp_path)
    arguments["source"] = {"commit": "f" * 40, "clean": True}

    with pytest.raises(ValueError, match="source commit"):
        module.publish_bundle(**arguments)


@pytest.mark.parametrize("mutation", ("invalid", "run_id", "source_commit"))
def test_publisher_validates_and_binds_live_trace_identity(tmp_path, mutation):
    module = _module("task21_hybrid_failure_bundle", SCRIPT)
    arguments = _publish_fixture(module, tmp_path)
    if mutation != "invalid":
        contract = _module(
            "apps.knowledge_graph.evals.task21_hybrid_live_trace",
            REPO
            / "aquillm"
            / "apps"
            / "knowledge_graph"
            / "evals"
            / "task21_hybrid_live_trace.py",
        )
        payload = valid_trace()
        payload["run_id"] = arguments["run_id"]
        payload["source_commit"] = arguments["expected_source_commit"]
        if mutation == "run_id":
            payload["run_id"] = "0" * 32
        else:
            payload["source_commit"] = "0" * 40
        arguments["artifacts"]["live_trace"].write_bytes(
            contract.canonical_live_trace_bytes(payload)
        )

    with pytest.raises(ValueError, match="live trace"):
        module.publish_bundle(**arguments)


def test_shell_gates_timing_write_and_passes_expected_source_commit():
    source = SHELL.read_text(encoding="utf-8")

    assert "timings_status" in source
    assert '--expected-source-commit "$SOURCE_COMMIT"' in source
    assert "--live-trace-output" in source
    assert "--live-trace" in source
