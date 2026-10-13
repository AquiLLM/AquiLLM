"""Runtime environment contracts used by graph management commands."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def test_pytest_runner_is_an_explicit_test_environment() -> None:
    from django.conf import settings

    assert settings.TESTING is True


@pytest.mark.parametrize(
    ("configured", "expected"),
    ((None, "5432"), ("55432", "55432"), ("0", "5432"), ("invalid", "5432")),
)
def test_postgres_port_setting_has_a_bounded_environment_override(
    configured: str | None,
    expected: str,
) -> None:
    repository_root = __import__("pathlib").Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
    if configured is None:
        environment.pop("POSTGRES_PORT", None)
    else:
        environment["POSTGRES_PORT"] = configured
    environment["DJANGO_DEBUG"] = "1"
    environment["PYTHONPATH"] = str(repository_root / "aquillm")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from aquillm import settings; "
                "print(settings.DATABASES['default']['PORT'])"
            ),
        ],
        cwd=repository_root / "aquillm",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected
