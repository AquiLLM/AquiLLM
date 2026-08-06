"""Pytest plugin enforcing and auditing offline socket policy."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from .network import NetworkAttempts, deny_network

_AUDIT_ENV = "AQUILLM_OFFLINE_NETWORK_ATTEMPTS_FILE"
_guard = None
_attempts: NetworkAttempts | None = None


def pytest_sessionstart(session) -> None:
    """Enter the socket guard before test collection and fixture setup."""
    del session
    global _attempts, _guard
    _guard = deny_network(allow_loopback=True)
    _attempts = _guard.__enter__()


def pytest_sessionfinish(session, exitstatus) -> None:
    """Persist the audit and make every attempted outbound connection fatal."""
    del exitstatus
    global _attempts, _guard
    attempts = _attempts or NetworkAttempts()
    if _guard is not None:
        _guard.__exit__(None, None, None)
    audit_path = os.environ.get(_AUDIT_ENV)
    if audit_path:
        Path(audit_path).write_text(
            json.dumps(
                {"total": attempts.total, "details": attempts.details},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if attempts.total:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    _guard = None
    _attempts = None
