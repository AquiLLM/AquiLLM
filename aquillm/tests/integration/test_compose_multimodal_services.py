"""Integration tests for Docker Compose service definitions."""

from pathlib import Path

import pytest

from tests.integration.compose_render_test_support import (
    render_compose_with_reviewed_env,
)


@pytest.mark.parametrize(
    "compose_name", ["base.yml", "development.yml", "production.yml"]
)
@pytest.mark.parametrize("direct", [None, "0"])
def test_rendered_defaults_preserve_production_and_direct_retrieval(
    compose_name, direct
):
    root = Path(__file__).resolve().parents[3]
    config = render_compose_with_reviewed_env(
        (root / "deploy/compose" / compose_name,),
        profile="vllm",
        environment_overrides={} if direct is None else {"RAG_DIRECT_ENABLED": direct},
    )
    services = config["services"]
    assert services["web"]["environment"]["RAG_DIRECT_ENABLED"] == (direct or "1")
    for service, allocation in [
        ("vllm_transcribe", "0.08"),
        ("vllm_embed", "0.12"),
        ("vllm_rerank", "0.15"),
    ]:
        assert (
            services[service]["environment"]["VLLM_GPU_MEMORY_UTILIZATION"]
            == allocation
        )
    assert (
        services["vllm_transcribe"]["environment"]["VLLM_MODEL"]
        == "openai/whisper-large-v3-turbo"
    )
    assert (
        services["vllm_transcribe"]["build"]["dockerfile"]
        == "deploy/docker/vllm/Dockerfile"
    )
    for name, service in services.items():
        if "knowledge_graph" in name:
            assert "knowledge-graph" in service.get("profiles", [])
    for service in ("worker", "worker_memory_promotion"):
        if service in services:
            assert "&& exec " in " ".join(services[service]["command"])


@pytest.mark.parametrize(
    "compose_name", ["base.yml", "development.yml", "production.yml"]
)
def test_nemotron_override_is_complete_despite_existing_whisper_environment(
    compose_name,
):
    root = Path(__file__).resolve().parents[3]
    config = render_compose_with_reviewed_env(
        (
            root / "deploy/compose" / compose_name,
            root / "deploy/compose/nemotron-asr.yml",
        ),
        profile="vllm",
        environment_overrides={
            "TRANSCRIBE_VLLM_MODEL": "openai/whisper-large-v3-turbo",
            "TRANSCRIBE_VLLM_GPU_MEMORY_UTILIZATION": "0.08",
            "TRANSCRIBE_VLLM_EXTRA_ARGS": "--max-num-batched-tokens 1500",
        },
    )
    services = config["services"]
    transcribe = services["vllm_transcribe"]
    env = transcribe["environment"]
    assert (
        transcribe["build"]["dockerfile"] == "deploy/docker/vllm/Dockerfile.transcribe"
    )
    assert env["VLLM_MODEL"] == "nvidia/nemotron-3.5-asr-streaming-0.6b"
    assert env["VLLM_REVISION"] == "f3d333391852ba876df169dcc9ba902d25b6ab0b"
    assert env["VLLM_GPU_MEMORY_UTILIZATION"] == "0.20"
    assert env["VLLM_MAX_MODEL_LEN"] == "50000"
    assert env["VLLM_DTYPE"] == "float32"
    assert env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] == "1"
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert env["VLLM_SERVICE_KIND"] == "transcribe"
    assert (
        env["VLLM_EXTRA_ARGS"]
        == "--enforce-eager --max-num-seqs 1 --max-num-batched-tokens 50000 --generation-config /opt/aquillm/nemotron-generation-config"
    )
    for app in ("web", "worker"):
        assert (
            services[app]["environment"]["INGEST_TRANSCRIBE_MODEL"]
            == env["VLLM_SERVED_MODEL_NAME"]
        )


def test_test_compose_storage_dependency_resolves():
    root = Path(__file__).resolve().parents[3]
    config = render_compose_with_reviewed_env(
        (root / "deploy/compose/test.yml",), profile="*"
    )
    assert "storage_test" in config["services"]["web_test"]["depends_on"]
    assert "storage_test" in config["services"]


@pytest.mark.parametrize(
    "compose_name", ["base.yml", "development.yml", "production.yml"]
)
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
        contents = (repo_root / "deploy/compose" / compose_name).read_text(
            encoding="utf-8"
        )
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
        ocr_service = contents.split("\n  vllm_ocr:", 1)[1].split(
            "\n  vllm_transcribe:", 1
        )[0]
        transcribe_service = contents.split("\n  vllm_transcribe:", 1)[1].split(
            "\n  vllm_embed:", 1
        )[0]

        assert "- ocr-sidecar" in ocr_service
        assert "- vllm\n" not in ocr_service
        assert "      vllm:\n        condition: service_healthy" in transcribe_service


def test_no_gpu_compose_keeps_hosted_whisper_model():
    contents = (Path(__file__).resolve().parents[3] / "deploy" / "compose" / "no_gpu_dev.yml").read_text(
        encoding="utf-8"
    )
    assert "INGEST_TRANSCRIBE_MODEL: whisper-1" in contents


def test_start_script_recovers_transcribe_args_by_service_kind_and_gates_revision():
    contents = (Path(__file__).resolve().parents[3] / "deploy" / "scripts" / "vllm_start.sh").read_text(
        encoding="utf-8"
    )

    service_kind = '[ "${VLLM_SERVICE_KIND:-}" = "transcribe" ]'
    task_recovery = 'case "${VLLM_TASK:-}" in'
    runner_recovery = '[ "${VLLM_RUNNER:-}" = "pooling" ]'
    model_recovery = 'case "${VLLM_MODEL:-}" in'
    assert service_kind in contents
    assert contents.index(service_kind) < contents.index(task_recovery)
    assert contents.index(service_kind) < contents.index(runner_recovery)
    assert contents.index(service_kind) < contents.index(model_recovery)
    assert "*whisper*|*Whisper*) export VLLM_EXTRA_ARGS=" not in contents
    assert 'export VLLM_EXTRA_ARGS="${TRANSCRIBE_VLLM_EXTRA_ARGS}"' in contents
    assert 'export VLLM_EXTRA_ARGS="${_DEFAULT_TRANSCRIBE_VLLM_EXTRA_ARGS}"' in contents

    assert (
        'if [ -n "${VLLM_REVISION:-}" ] && supports_arg "--revision"; then' in contents
    )
    assert 'cmd+=(--revision "${VLLM_REVISION}")' in contents
    assert 'score) export VLLM_EXTRA_ARGS="${APP_RERANK_VLLM_EXTRA_ARGS:-}"' in contents
    assert 'export VLLM_EXTRA_ARGS="${MEM0_EMBED_VLLM_EXTRA_ARGS:-}"' in contents
    assert 'export VLLM_EXTRA_ARGS="${OCR_VLLM_EXTRA_ARGS}"' in contents

    unset_block = contents.split("# Avoid vLLM env validation warnings", 1)[1].split(
        'echo "Starting vLLM', 1
    )[0]
    assert "VLLM_REVISION" in unset_block
    assert "VLLM_SERVICE_KIND" in unset_block
    assert "VLLM_ALLOW_LONG_MAX_MODEL_LEN" not in unset_block
    assert "VLLM_USE_V2_MODEL_RUNNER" not in unset_block
