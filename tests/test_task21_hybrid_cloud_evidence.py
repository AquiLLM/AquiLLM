from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from task21_hybrid_live_trace_support import valid_trace

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "task21_hybrid_failure_bundle.py"
SHELL = REPO / "scripts" / "run_task21_hybrid_cloud_eval.sh"


def _module():
    spec = importlib.util.spec_from_file_location(
        "task21_hybrid_failure_bundle", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class FakeCommandResult:
    stdout: str
    stderr: str = ""
    returncode: int = 0

    @property
    def succeeded(self):
        return self.returncode == 0


class FakeRunner:
    def __init__(self, outputs=None):
        self.outputs = outputs or {}
        self.calls = []

    def run(self, arguments, **_kwargs):
        command = tuple(arguments)
        self.calls.append(command)
        result = self.outputs.get(command, "")
        return result if hasattr(result, "succeeded") else FakeCommandResult(result)


def _capture_outputs(module, prefix):
    outputs = {prefix + ("config", "--no-interpolate"): "services: {}\n"}
    for index, service in enumerate(module.SERVICES, start=1):
        container = f"{index:x}" * 64
        image = f"{(index + 7) % 16:x}" * 64
        outputs[prefix + ("ps", "--all", "--quiet", service)] = container
        outputs[module.inspect_command(container)] = (
            f"{container}\tsha256:{image}\trunning\t0"
        )
        outputs[prefix + ("logs", "--no-color", "--tail", "200", service)] = (
            f"{service}=healthy PASSWORD=do-not-retain"
        )
    return outputs


def test_cloud_shell_is_provider_neutral_fail_closed_and_uses_fixed_entrypoints():
    source = SHELL.read_text(encoding="utf-8")

    assert "set -euo pipefail" in source
    assert "--require-clean-head" in source
    assert "task21-hybrid-cloud-evidence-v1" in source
    assert "TASK21_EVIDENCE_SIGNING_KEY" in source
    assert "apps.knowledge_graph.evals.task21_hybrid_eval" in source
    assert "task21_hybrid_failure_bundle.py" in source
    assert "task21_hybrid_live_observations" in source
    assert "task21_hybrid_observation_attestation.py" in source
    assert "TASK21_HYBRID_OBSERVATIONS" not in source
    assert "CREATE ROLE aquillm_projection_source" in source
    assert "CREATE ROLE aquillm_projection_state" in source
    assert "worker_knowledge_graph_projection" in source
    assert source.count("git status --porcelain=v1 --untracked-files=normal") >= 2
    assert "original_exit_code" in source
    assert "timings_status" in source
    assert '--expected-source-commit "$SOURCE_COMMIT"' in source
    assert not {"gcloud", "aws ", "az "}.intersection(source.splitlines())
    assert 'eval "' not in source


def test_capture_precedes_cleanup_and_preserves_bounded_redacted_service_logs():
    module = _module()
    prefix = ("docker", "compose", "-p", "aquillm-task21-a", "-f", "compose.yml")
    runner = FakeRunner(_capture_outputs(module, prefix))

    captured = module.capture_runtime(runner, prefix)
    cleanup = module.cleanup_runtime(
        runner,
        prefix,
        project_label="com.docker.compose.project=aquillm-task21-a",
    )

    down = runner.calls.index(prefix + ("down", "--volumes", "--remove-orphans"))
    capture_calls = [
        index
        for index, call in enumerate(runner.calls)
        if call[: len(prefix)] == prefix
        and call[len(prefix) : len(prefix) + 1] in {("ps",), ("logs",), ("config",)}
    ]
    assert max(capture_calls) < down
    assert len(cleanup["zero_samples"]) == 3
    assert all(
        not value for sample in cleanup["zero_samples"] for value in sample.values()
    )
    for service in module.SERVICES:
        log = captured.members[f"logs/{service}.log"].decode()
        assert "do-not-retain" not in log and "<redacted>" in log
        assert len(log.encode()) <= module.MAX_LOG_BYTES


def test_capture_uses_only_allowlisted_inspect_fields_and_fixed_services():
    module = _module()
    prefix = ("docker", "compose", "-p", "aquillm-task21-b")
    runner = FakeRunner(_capture_outputs(module, prefix))

    captured = module.capture_runtime(runner, prefix)

    assert tuple(captured.images) == module.SERVICES
    assert all(value.startswith("sha256:") for value in captured.images.values())
    inspect_calls = [call for call in runner.calls if call[:2] == ("docker", "inspect")]
    assert len(inspect_calls) == len(module.SERVICES)
    assert all(call == module.inspect_command(call[-1]) for call in inspect_calls)
    assert all("Config.Env" not in " ".join(call) for call in inspect_calls)


def test_publish_is_canonical_signed_0600_and_never_overwrites(tmp_path):
    module = _module()
    arm_results = tmp_path / "arms.json"
    timings = tmp_path / "timings.json"
    projections = tmp_path / "projections.json"
    live_trace = tmp_path / "live-trace.json"
    arm_results.write_text('{"arms":[]}', encoding="utf-8")
    timings.write_text(
        json.dumps(
            {
                "elapsed_ms": 12.5,
                "finished_ns": 13_500_000,
                "original_exit_code": 0,
                "started_ns": 1_000_000,
                "status": "passed",
            }
        ),
        encoding="utf-8",
    )
    projections.write_text(
        json.dumps({"generation_key": "a" * 64, "projection_checksum": "b" * 64}),
        encoding="utf-8",
    )
    trace_contract = importlib.import_module(
        "apps.knowledge_graph.evals.task21_hybrid_live_trace"
    )
    trace_payload = valid_trace()
    trace_payload["run_id"] = "e" * 32
    trace_payload["source_commit"] = "f" * 40
    live_trace.write_bytes(trace_contract.canonical_live_trace_bytes(trace_payload))
    captured = module.CapturedRuntime(
        members={
            "runtime/config.redacted.yml": b"services: {}\n",
            "runtime/state.json": b"{}",
            "logs/web.log": b"failed before cleanup",
        },
        images={"web": "sha256:" + "c" * 64},
        config_sha256="d" * 64,
    )
    cleanup = {"zero_samples": [{"containers": "", "volumes": ""}] * 3}
    key = b"task23-signing-key"

    destination = module.publish_bundle(
        output_root=tmp_path / "artifacts" / "task21-hybrid-cloud",
        run_id="e" * 32,
        captured=captured,
        artifacts={
            "arm_results": arm_results,
            "timings": timings,
            "projection_checksums": projections,
            "live_trace": live_trace,
        },
        cleanup_proof=cleanup,
        source={"commit": "f" * 40, "clean": True},
        expected_source_commit="f" * 40,
        claim_scope="cloud",
        signing_key=key,
        signing_key_version="task23-v1",
    )

    assert destination == tmp_path / "artifacts" / "task21-hybrid-cloud" / ("e" * 32)
    manifest = json.loads((destination / "bundle.json").read_text(encoding="utf-8"))
    signature = manifest.pop("signature")
    assert manifest["schema"] == "task21-hybrid-cloud-evidence-v1"
    assert manifest["source"] == {"clean": True, "commit": "f" * 40}
    assert manifest["images"] == {"web": "sha256:" + "c" * 64}
    assert signature["key_version"] == "task23-v1"
    assert (
        signature["value"]
        == hmac.new(
            key, module.canonical_json_bytes(manifest), hashlib.sha256
        ).hexdigest()
    )
    for member in manifest["members"]:
        path = destination / member["path"]
        assert member["size"] == path.stat().st_size
        assert member["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        if os.name != "nt":
            assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        module.publish_bundle(
            output_root=destination.parent,
            run_id="e" * 32,
            captured=captured,
            artifacts={
                "arm_results": arm_results,
                "timings": timings,
                "projection_checksums": projections,
                "live_trace": live_trace,
            },
            cleanup_proof=cleanup,
            source={"commit": "f" * 40, "clean": True},
            expected_source_commit="f" * 40,
            claim_scope="cloud",
            signing_key=key,
            signing_key_version="task23-v1",
        )


@pytest.mark.parametrize(
    ("text", "secret"),
    (
        ("Authorization: Bearer abc123", "abc123"),
        ("API_KEY=abcdef", "abcdef"),
        ("password: hunter2", "hunter2"),
    ),
)
def test_log_redaction_removes_secret_shapes_and_enforces_byte_cap(text, secret):
    module = _module()
    redacted = module.redact_log((text + "\n") * 20, max_bytes=96)
    assert secret not in redacted
    assert len(redacted.encode("utf-8")) <= 96
