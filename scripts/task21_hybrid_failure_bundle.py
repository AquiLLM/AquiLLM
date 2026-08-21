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
_CREDENTIAL_URL = re.compile(
    r"(?i)\b(?:postgres(?:ql)?|memgraph|bolt(?:\+s|\+ssc)?|neo4j(?:\+s|\+ssc)?|https?)"
    r"://[^\s,\"'<>?&;]+"
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _redact_url_userinfo(match: re.Match[str]) -> str:
    value = match.group(0)
    prefix, remainder = value.split("://", 1)
    boundary = min(
        (index for marker in "/?#" if (index := remainder.find(marker)) >= 0),
        default=len(remainder),
    )
    authority, suffix = remainder[:boundary], remainder[boundary:]
    if "@" not in authority:
        return value
    host = authority.rsplit("@", 1)[1]
    return f"{prefix}://<redacted>@{host}{suffix}"


def redact_log(value: str, *, max_bytes: int = MAX_LOG_BYTES) -> str:
    if type(value) is not str or type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("log redaction inputs are invalid")
    redacted = _CREDENTIAL_URL.sub(_redact_url_userinfo, value)
    redacted = _AUTHORIZATION.sub("Authorization: <redacted>", redacted)
    redacted = _SECRET_ASSIGNMENT.sub(r"\1\2<redacted>", redacted)
    bounded = redacted.encode("utf-8")[:max_bytes]
    return bounded.decode("utf-8", errors="ignore")


def inspect_command(container_id: str) -> tuple[str, ...]:
    if _HEX64.fullmatch(container_id) is None:
        raise ValueError("container id must be exact lowercase hexadecimal")
    return ("docker", "inspect", "--format", _INSPECT_FORMAT, container_id)


@dataclass(frozen=True, slots=True)
class CommandResult:
    stdout: str
    stderr: str
    returncode: int

    def __post_init__(self) -> None:
        if type(self.stdout) is not str or type(self.stderr) is not str:
            raise TypeError("command output must be exact text")
        if type(self.returncode) is not int:
            raise ValueError("command return code is invalid")

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


class CommandRunner:
    def run(self, arguments, *, timeout=1_800) -> CommandResult:
        try:
            result = subprocess.run(
                tuple(arguments),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout if isinstance(error.stdout, str) else ""
            stderr = error.stderr if isinstance(error.stderr, str) else ""
            return CommandResult(stdout, stderr or "command timed out", 124)
        return CommandResult(result.stdout, result.stderr, result.returncode)


def _command_error(operation: str, result) -> dict[str, object]:
    return {
        "operation": operation,
        "returncode": result.returncode,
        "stderr": redact_log(result.stderr, max_bytes=4_096),
    }


@dataclass(frozen=True, slots=True)
class CapturedRuntime:
    members: dict[str, bytes]
    images: dict[str, str]
    config_sha256: str
    complete: bool = True


def capture_runtime(runner, compose_prefix) -> CapturedRuntime:
    prefix = tuple(compose_prefix)
    config_result = runner.run(prefix + ("config", "--no-interpolate"))
    raw_config = config_result.stdout
    redacted_config = redact_log(raw_config).encode("utf-8")
    members = {"runtime/config.redacted.yml": redacted_config}
    images: dict[str, str] = {}
    states: dict[str, object] = {}
    errors = []
    if not config_result.succeeded:
        errors.append(_command_error("compose_config", config_result))
    for service in SERVICES:
        ps_result = runner.run(prefix + ("ps", "--all", "--quiet", service))
        container = ps_result.stdout.strip() if ps_result.succeeded else ""
        state: dict[str, object] = {"service": service, "present": bool(container)}
        if not ps_result.succeeded:
            state["capture_error"] = "compose_ps_failed"
            errors.append(_command_error(f"compose_ps:{service}", ps_result))
        elif not container:
            state["capture_error"] = "service_missing"
        else:
            if _HEX64.fullmatch(container) is None:
                state["capture_error"] = "invalid_container_id"
            else:
                inspect_result = runner.run(inspect_command(container))
                fields = inspect_result.stdout.strip().split("\t")
                if not inspect_result.succeeded:
                    state["capture_error"] = "inspect_command_failed"
                    errors.append(_command_error(f"inspect:{service}", inspect_result))
                elif len(fields) != 4 or fields[0] != container:
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
                            if status != "running":
                                state["capture_error"] = "service_not_running"
                            elif exact_exit != 0:
                                state["capture_error"] = "service_exit_unsuccessful"
        states[service] = state
        log_result = runner.run(
            prefix + ("logs", "--no-color", "--tail", "200", service)
        )
        log = log_result.stdout
        if not log_result.succeeded:
            errors.append(_command_error(f"compose_logs:{service}", log_result))
            log += f"\ncommand_error={log_result.stderr}"
        members[f"logs/{service}.log"] = redact_log(log).encode("utf-8")
    states["command_errors"] = errors
    members["runtime/state.json"] = canonical_json_bytes(states) + b"\n"
    capture_complete = not errors and not any(
        isinstance(state, dict) and "capture_error" in state
        for state in states.values()
    )
    return CapturedRuntime(
        members,
        images,
        hashlib.sha256(raw_config.encode("utf-8")).hexdigest(),
        complete=capture_complete,
    )


def cleanup_runtime(runner, compose_prefix, *, project_label: str) -> dict[str, object]:
    if not re.fullmatch(
        r"com\.docker\.compose\.project=[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", project_label
    ):
        raise ValueError("project label is invalid")
    prefix = tuple(compose_prefix)
    errors = []
    down = runner.run(prefix + ("down", "--volumes", "--remove-orphans"))
    if not down.succeeded:
        errors.append(_command_error("compose_down", down))
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
    samples = []
    for sample_index in range(3):
        sample = {}
        for name, command in commands.items():
            result = runner.run(command)
            sample[name] = result.stdout.strip()
            if not result.succeeded:
                errors.append(
                    _command_error(f"cleanup_{name}:{sample_index + 1}", result)
                )
        samples.append(sample)
    zero_samples = tuple(samples)
    return {
        "project_label": project_label,
        "down_returncode": down.returncode,
        "zero_samples": zero_samples,
        "command_errors": errors,
        "complete": not errors
        and not any(value for sample in zero_samples for value in sample.values()),
    }


def main() -> int:
    from task21_hybrid_evidence_cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
