"""Standalone evaluation needs standard Git, never the workstation wrapper."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from apps.chat.evals.evidence_quality_review import dirty_checkout


def git_environment():
    git = shutil.which("git")
    assert git, "standard Git is required by evaluation provenance"
    environment = dict(os.environ)
    environment["PATH"] = str(Path(git).parent)
    assert shutil.which("rtk", path=environment["PATH"]) is None
    for key in list(environment):
        if key == "SECRET_KEY" or key.startswith(
            ("DJANGO_", "POSTGRES_", "OPENAI_", "GEMINI_", "ANTHROPIC_", "GOOGLE_")
        ):
            environment.pop(key)
    return environment


@pytest.mark.parametrize("kind", ["quality", "operational"])
def test_standalone_reports_and_rescore_without_rtk(tmp_path, kind):
    environment = git_environment()
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    runner = Path(__file__).parents[1] / f"evals/run_evidence_{kind}_eval.py"
    arguments = (
        ["--split", "development", "--require-activation"]
        if kind == "quality"
        else ["--profile", "pilot", "--repetition", "1"]
    )
    report = tmp_path / "original.json"
    command = [
        sys.executable,
        str(runner),
        "--backend",
        "fixture",
        "--mode",
        "combined",
        *arguments,
    ]
    result = subprocess.run(
        [*command, "--report", str(report)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (2 if kind == "quality" else 0), result.stderr
    original = json.loads(report.read_text(encoding="utf-8"))
    assert original["revision"] == git
    assert original["dirty"] is dirty
    assert original["activation_eligible"] is False
    assert all(row["code_revision"] == git for row in original["observations"])
    # Saved observations retain original provenance without inspecting this checkout.
    environment["PATH"] = ""
    rescored = tmp_path / "rescored.json"
    result = subprocess.run(
        [*command, "--observations", str(report), "--report", str(rescored)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (2 if kind == "quality" else 0), result.stderr
    saved = json.loads(rescored.read_text(encoding="utf-8"))
    assert saved["revision"] == git and saved["dirty"] is dirty
    assert saved["activation_eligible"] is False
    assert saved["observations"] == original["observations"]
    unavailable = tmp_path / "unavailable.json"
    result = subprocess.run(
        [*command, "--report", str(unavailable)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "FileNotFoundError" in result.stderr
    assert not unavailable.exists()


def test_dirty_provenance_uses_raw_git_and_does_not_invent_clean(monkeypatch, tmp_path):
    environment = git_environment()
    monkeypatch.setenv("PATH", environment["PATH"])
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "--quiet"], check=True)
    assert dirty_checkout() is False
    (tmp_path / "untracked.txt").write_text("untracked source", encoding="utf-8")
    assert dirty_checkout() is True
    monkeypatch.setenv("PATH", "")
    with pytest.raises(FileNotFoundError):
        dirty_checkout()
