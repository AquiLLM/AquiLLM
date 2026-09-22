"""Checked-in vLLM profile and shell portability contracts."""
from pathlib import Path


def test_checked_in_profiles_keep_protected_runner_and_dtype_out_of_extra_args():
    repo_root = Path(__file__).resolve().parents[3]
    env_lines = (repo_root / ".env.example").read_text(encoding="utf-8").splitlines()
    active_extra_lines = [
        line
        for line in env_lines
        if line and not line.startswith("#") and "VLLM_EXTRA_ARGS=" in line
    ]

    assert active_extra_lines
    assert all(
        "--dtype" not in line and "--runner" not in line for line in active_extra_lines
    )
    assert "VLLM_DTYPE=float16" in env_lines
    assert "OCR_VLLM_DTYPE=float16" in env_lines

    profile = (repo_root / "scripts" / "verify_nemotron_asr.ps1").read_text(
        encoding="utf-8"
    )
    for prefix in (
        "PROFILE_MAIN_EXTRA_ARGS=",
        "PROFILE_EMBED_EXTRA_ARGS=",
        "PROFILE_RERANK_EXTRA_ARGS=",
    ):
        line = next(row for row in profile.splitlines() if row.startswith(prefix))
        assert "--dtype" not in line
        assert "--runner" not in line
    assert profile.count("VLLM_RUNNER: pooling") >= 2
    assert profile.count("VLLM_DTYPE: float16") >= 3


def test_repository_forces_shell_scripts_to_lf():
    repo_root = Path(__file__).resolve().parents[3]
    attributes_path = repo_root / ".gitattributes"

    assert attributes_path.exists(), ".gitattributes must define shell line endings"
    attributes = attributes_path.read_text(encoding="utf-8").splitlines()
    assert "*.sh text eol=lf" in attributes
