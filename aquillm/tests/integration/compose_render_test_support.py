"""Hermetic Docker Compose rendering helpers for integration contracts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from tests.integration.task21_compose_test_support import (
    DOCKER_CLIENT_ENV,
    env_file_shim,
    reviewed_env,
)


def render_compose_with_reviewed_env(
    compose_files: tuple[Path, ...],
    *,
    profile: str,
    environment_overrides: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Render Compose using only a disposable reviewed service environment."""

    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is unavailable")
    version = subprocess.run(
        [docker, "compose", "version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if version.returncode != 0:
        pytest.skip("Docker Compose is unavailable")
    if (
        type(compose_files) is not tuple
        or not compose_files
        or any(not isinstance(path, Path) for path in compose_files)
    ):
        raise TypeError("compose_files must be a nonempty exact tuple of Paths")
    with TemporaryDirectory(prefix="aquillm-compose-test-") as directory:
        temporary = Path(directory)
        env_file = reviewed_env(temporary, dict(environment_overrides or {}))
        shims = tuple(
            env_file_shim(temporary, env_file, compose_file)
            for compose_file in compose_files
        )
        command = [docker, "compose", "--env-file", str(env_file)]
        for compose_file in (*compose_files, *shims):
            command.extend(("-f", str(compose_file)))
        command.extend(
            ("--profile", profile, "config", "--format", "json", "--no-env-resolution")
        )
        environment = {
            name: value
            for name, value in os.environ.items()
            if name in DOCKER_CLIENT_ENV
        }
        environment["TASK21_ENV_FILE"] = str(env_file)
        result = subprocess.run(
            command,
            cwd=compose_files[0].resolve().parents[2],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


__all__ = ["render_compose_with_reviewed_env"]
