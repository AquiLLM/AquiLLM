#!/usr/bin/env python3
"""Capture, clean, and immutably publish Task21 cloud evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass

try:
    from scripts import task21_hybrid_evidence_publish as _publisher
except ImportError:
    import task21_hybrid_evidence_publish as _publisher

SCHEMA = _publisher.SCHEMA
publish_bundle = _publisher.publish_bundle

SERVICES = (
    "web",
    "db",
    "redis",
    "memgraph_knowledge_graph",
    "knowledge_graph_query_extractor",
    "worker_knowledge_graph_projection",
    "worker_knowledge_graph",
    "vllm_embed",
    "vllm_rerank",
)
MAX_LOG_BYTES = 65_536
_HEX64 = re.compile(r"[0-9a-f]{64}")
_INSPECT_FORMAT = "{{.Id}}\t{{.Image}}\t{{.State.Status}}\t{{.State.ExitCode}}"
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(password|secret|token|api[_-]?key)(\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_AUTHORIZATION = re.compile(r"(?i)\bauthorization\s*:\s*(?:bearer\s+)?[^\s,;]+")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def redact_log(value: str, *, max_bytes: int = MAX_LOG_BYTES) -> str:
    if type(value) is not str or type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("log redaction inputs are invalid")
    redacted = _AUTHORIZATION.sub("Authorization: <redacted>", value)
    redacted = _SECRET_ASSIGNMENT.sub(r"\1\2<redacted>", redacted)
    bounded = redacted.encode("utf-8")[:max_bytes]
    return bounded.decode("utf-8", errors="ignore")


def inspect_command(container_id: str) -> tuple[str, ...]:
    if _HEX64.fullmatch(container_id) is None:
        raise ValueError("container id must be exact lowercase hexadecimal")
    return ("docker", "inspect", "--format", _INSPECT_FORMAT, container_id)


class CommandRunner:
    def run(self, arguments, *, check=True, timeout=1_800):
        try:
            result = subprocess.run(
                tuple(arguments),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            if check:
                raise RuntimeError("evidence command timed out") from error
            return error.stdout if isinstance(error.stdout, str) else ""
        if check and result.returncode:
            raise RuntimeError(
                f"evidence command failed ({result.returncode}): "
                + " ".join(tuple(arguments)[:4])
            )
        return result.stdout


@dataclass(frozen=True, slots=True)
class CapturedRuntime:
    members: dict[str, bytes]
    images: dict[str, str]
    config_sha256: str


def capture_runtime(runner, compose_prefix) -> CapturedRuntime:
    prefix = tuple(compose_prefix)
    raw_config = runner.run(prefix + ("config", "--no-interpolate"), check=False)
    redacted_config = redact_log(raw_config).encode("utf-8")
    members = {"runtime/config.redacted.yml": redacted_config}
    images: dict[str, str] = {}
    states: dict[str, object] = {}
    for service in SERVICES:
        container = runner.run(
            prefix + ("ps", "--all", "--quiet", service), check=False
        ).strip()
        state: dict[str, object] = {"service": service, "present": bool(container)}
        if container:
            if _HEX64.fullmatch(container) is None:
                state["capture_error"] = "invalid_container_id"
            else:
                fields = (
                    runner.run(inspect_command(container), check=False)
                    .strip()
                    .split("\t")
                )
                if len(fields) != 4 or fields[0] != container:
                    state["capture_error"] = "inspect_state_unavailable"
                else:
                    image, status, exit_code = fields[1:]
                    if re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
                        state["capture_error"] = "invalid_image_digest"
                    else:
                        try:
                            exact_exit = int(exit_code)
                        except ValueError:
                            state["capture_error"] = "invalid_exit_code"
                        else:
                            state.update(
                                container_id=container,
                                image=image,
                                status=status,
                                exit_code=exact_exit,
                            )
                            images[service] = image
        states[service] = state
        log = runner.run(
            prefix + ("logs", "--no-color", "--tail", "200", service),
            check=False,
        )
        members[f"logs/{service}.log"] = redact_log(log).encode("utf-8")
    members["runtime/state.json"] = canonical_json_bytes(states) + b"\n"
    return CapturedRuntime(
        members,
        images,
        hashlib.sha256(raw_config.encode("utf-8")).hexdigest(),
    )


def cleanup_runtime(runner, compose_prefix, *, project_label: str) -> dict[str, object]:
    if not re.fullmatch(
        r"com\.docker\.compose\.project=[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", project_label
    ):
        raise ValueError("project label is invalid")
    prefix = tuple(compose_prefix)
    runner.run(prefix + ("down", "--volumes", "--remove-orphans"), check=False)
    commands = {
        "containers": ("docker", "ps", "-aq", "--filter", f"label={project_label}"),
        "volumes": (
            "docker",
            "volume",
            "ls",
            "-q",
            "--filter",
            f"label={project_label}",
        ),
        "networks": (
            "docker",
            "network",
            "ls",
            "-q",
            "--filter",
            f"label={project_label}",
        ),
    }
    samples = tuple(
        {
            name: runner.run(command, check=False).strip()
            for name, command in commands.items()
        }
        for _ in range(3)
    )
    return {
        "project_label": project_label,
        "zero_samples": samples,
        "complete": not any(value for sample in samples for value in sample.values()),
    }


def main() -> int:
    from task21_hybrid_evidence_cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
