"""CPU-only contract tests for the PowerShell deployment verifier."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VERIFIER = REPOSITORY_ROOT / "scripts" / "verify_nemotron_asr.ps1"


def test_profile_self_test_covers_failure_policy_and_image_provenance() -> None:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is unavailable")

    completed = subprocess.run(
        [powershell, "-NoProfile", "-File", str(VERIFIER), "-SelfTest"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "profile failure classification" in completed.stdout
    assert "profile image provenance" in completed.stdout


def test_profile_requires_full_pass_unless_measurement_is_opted_in() -> None:
    source = VERIFIER.read_text(encoding="utf-8")

    assert "[switch]$AllowIncompleteProfile" in source
    assert "if ($AllowIncompleteProfile -and" in source
    assert 'throw "Required profile verification failed' in source
    assert '"capacity", "timeout"' in source


def test_profile_preflight_probes_both_fresh_images_with_explicit_entrypoints() -> None:
    source = VERIFIER.read_text(encoding="utf-8")

    assert "docker run --rm --entrypoint python3 $ProfileImage" in source
    assert "m.version('vllm') == '0.21.0'" in source
    assert '"profile-generic-image-probe.txt"' in source
    assert '"--entrypoint", "python3", $Image' in source
