"""Integration tests for Docker Compose service definitions."""

from pathlib import Path

import pytest


@pytest.mark.parametrize("compose_name", ["base.yml", "development.yml", "production.yml"])
def test_sidecar_defaults_fit_the_shared_gpu_profile(compose_name):
    repo_root = Path(__file__).resolve().parents[3]
    contents = (repo_root / "deploy/compose" / compose_name).read_text(encoding="utf-8")
    example = (repo_root / ".env.example").read_text(encoding="utf-8")
    allocations = {
        "VLLM_GPU_MEMORY_UTILIZATION": "0.45",
        "TRANSCRIBE_VLLM_GPU_MEMORY_UTILIZATION": "0.08",
        "MEM0_EMBED_GPU_MEMORY_UTILIZATION": "0.12",
        "APP_RERANK_GPU_MEMORY_UTILIZATION": "0.15",
        "OCR_VLLM_GPU_MEMORY_UTILIZATION": "0.15",
    }
    for name, fraction in allocations.items():
        assert f"{name}={fraction}" in example.splitlines()
        if name != "VLLM_GPU_MEMORY_UTILIZATION" or compose_name != "base.yml":
            assert "${" + name + ":-" + fraction + "}" in contents
    assert sum(map(float, allocations.values())) < 1.0
    assert "${MEM0_EMBED_MAX_MODEL_LEN:-2048}" in contents
    assert "${APP_RERANK_MAX_MODEL_LEN:-1024}" in contents


def test_chat_context_defaults_match_with_environment_overrides():
    repo_root = Path(__file__).resolve().parents[3]
    example = (repo_root / ".env.example").read_text(encoding="utf-8")
    assert "VLLM_MAX_MODEL_LEN=131072" in example.splitlines()
    for compose_name in ("development.yml", "production.yml"):
        contents = (repo_root / "deploy/compose" / compose_name).read_text(encoding="utf-8")
        assert "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-131072}" in contents


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
