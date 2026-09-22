"""Render Compose contracts without consulting the operator's .env or Docker daemon."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml


def render_compose_with_reviewed_env(
    compose_files: tuple[Path, ...],
    *,
    profile: str,
    environment_overrides: dict[str, str] | None = None,
) -> dict:
    if type(compose_files) is not tuple or not compose_files:
        raise TypeError("compose_files must be a nonempty exact tuple of Paths")
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is unavailable")
    # Only retain client/OS variables; ambient model/API settings must not leak
    # into a defaults contract. Service env files are redirected below because
    # older Compose still stats them when --no-env-resolution is requested.
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "HOME",
        "USERPROFILE",
        "TEMP",
        "TMP",
        "DOCKER_CONFIG",
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PATHEXT",
        "COMSPEC",
    }
    environment = {
        key: value for key, value in os.environ.items() if key.upper() in allowed
    }
    with TemporaryDirectory(prefix="aquillm-compose-contract-") as directory:
        env_file = Path(directory) / "reviewed.env"
        values = {
            key: "contract-test-only"
            for key in (
                "POSTGRES_NAME",
                "POSTGRES_USER",
                "POSTGRES_PASSWORD",
                "STORAGE_ACCESS_KEY",
                "STORAGE_SECRET_KEY",
            )
        }
        values.update(environment_overrides or {})
        env_file.write_text(
            "\n".join(f"{key}={value}" for key, value in values.items()),
            encoding="utf-8",
        )
        command = [
            docker,
            "compose",
            "--env-file",
            str(env_file),
            "--project-directory",
            str(compose_files[0].resolve().parent),
        ]
        for index, path in enumerate(compose_files):
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            for service in config.get("services", {}).values():
                if "env_file" in service:
                    service["env_file"] = [str(env_file)]
            # Preserve all other values and interpolate them in Compose itself.
            # --project-directory keeps relative builds/mounts anchored to the
            # original first Compose file, just as with multiple real -f files.
            reviewed_compose = Path(directory) / f"compose-{index}.yml"
            reviewed_compose.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            command.extend(("-f", str(reviewed_compose)))
        command.extend(
            ("--profile", profile, "config", "--format", "json", "--no-env-resolution")
        )
        result = subprocess.run(
            command,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)
