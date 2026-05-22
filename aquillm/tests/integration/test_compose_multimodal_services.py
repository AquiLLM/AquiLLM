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


def test_ocr_sidecar_is_not_in_default_vllm_profile():
    repo_root = Path(__file__).resolve().parents[3]
    compose_files = [
        repo_root / "deploy" / "compose" / "base.yml",
        repo_root / "deploy" / "compose" / "development.yml",
        repo_root / "deploy" / "compose" / "production.yml",
    ]

    for compose_file in compose_files:
        contents = compose_file.read_text(encoding="utf-8")
        ocr_service = contents.split("\n  vllm_ocr:", 1)[1].split("\n  vllm_transcribe:", 1)[0]
        transcribe_service = contents.split("\n  vllm_transcribe:", 1)[1].split("\n  vllm_embed:", 1)[0]

        assert "- ocr-sidecar" in ocr_service
        assert "- vllm\n" not in ocr_service
        assert "      vllm:\n        condition: service_healthy" in transcribe_service
