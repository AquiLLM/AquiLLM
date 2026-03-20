"""Integration tests for Docker Compose service definitions."""

from pathlib import Path


def test_compose_files_define_ocr_and_transcribe_services():
    repo_root = Path(__file__).resolve().parents[3]
    compose_files = [
        repo_root / "deploy" / "compose" / "base.yml",
        repo_root / "deploy" / "compose" / "development.yml",
        repo_root / "deploy" / "compose" / "production.yml",
    ]

    for compose_file in compose_files:
        contents = compose_file.read_text(encoding="utf-8")
        assert "vllm_ocr:" in contents
        assert "vllm_transcribe:" in contents
