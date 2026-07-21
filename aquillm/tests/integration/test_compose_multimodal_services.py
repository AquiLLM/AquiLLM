"""Integration tests for Docker Compose service definitions."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GPU_COMPOSE_FILES = [
    REPO_ROOT / "deploy" / "compose" / "base.yml",
    REPO_ROOT / "deploy" / "compose" / "development.yml",
    REPO_ROOT / "deploy" / "compose" / "production.yml",
]


def _service_block(contents: str, service: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(service)}:\r?\n(?P<body>.*?)(?=^  [A-Za-z0-9_]+:|\Z)",
        contents,
    )
    assert match is not None, f"service {service!r} is not defined"
    return match.group("body")


def test_compose_files_define_ocr_and_transcribe_services():
    for compose_file in GPU_COMPOSE_FILES:
        contents = compose_file.read_text(encoding="utf-8")
        assert "vllm_ocr:" in contents
        assert "vllm_transcribe:" in contents


def test_ocr_sidecar_is_not_in_default_vllm_profile():
    for compose_file in GPU_COMPOSE_FILES:
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


def test_only_transcribe_sidecars_use_the_dedicated_asr_image():
    for compose_file in GPU_COMPOSE_FILES:
        contents = compose_file.read_text(encoding="utf-8")
        assert "dockerfile: deploy/docker/vllm/Dockerfile.transcribe" in _service_block(
            contents, "vllm_transcribe"
        )
        for service in ("vllm", "vllm_ocr", "vllm_embed", "vllm_rerank"):
            assert "Dockerfile.transcribe" not in _service_block(contents, service)


def test_transcribe_sidecars_share_exact_nemotron_defaults():
    expected_lines = {
        "VLLM_MODEL=${TRANSCRIBE_VLLM_MODEL:-nvidia/nemotron-3.5-asr-streaming-0.6b}",
        "VLLM_REVISION=${TRANSCRIBE_VLLM_REVISION-f3d333391852ba876df169dcc9ba902d25b6ab0b}",
        "VLLM_SERVED_MODEL_NAME=${TRANSCRIBE_VLLM_SERVED_MODEL_NAME:-nemotron-3.5-asr-streaming-0.6b}",
        "VLLM_TOKENIZER=${TRANSCRIBE_VLLM_TOKENIZER:-nvidia/nemotron-3.5-asr-streaming-0.6b}",
        "VLLM_TENSOR_PARALLEL_SIZE=${TRANSCRIBE_VLLM_TENSOR_PARALLEL_SIZE:-1}",
        "VLLM_GPU_MEMORY_UTILIZATION=${TRANSCRIBE_VLLM_GPU_MEMORY_UTILIZATION:-0.20}",
        "VLLM_MAX_MODEL_LEN=${TRANSCRIBE_VLLM_MAX_MODEL_LEN:-50000}",
        "VLLM_DTYPE=${TRANSCRIBE_VLLM_DTYPE:-auto}",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN=${TRANSCRIBE_VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}",
        "VLLM_TRUST_REMOTE_CODE=${TRANSCRIBE_VLLM_TRUST_REMOTE_CODE:-1}",
        "VLLM_SERVICE_KIND=transcribe",
        "VLLM_USE_V2_MODEL_RUNNER=0",
        '"VLLM_EXTRA_ARGS=${TRANSCRIBE_VLLM_EXTRA_ARGS}"',
    }

    for compose_file in GPU_COMPOSE_FILES:
        transcribe = _service_block(
            compose_file.read_text(encoding="utf-8"), "vllm_transcribe"
        )
        for expected in expected_lines:
            assert expected in transcribe, f"{compose_file.name}: missing {expected}"
        assert "bitsandbytes" not in transcribe
        assert "turboquant" not in transcribe.lower()


def test_env_example_defines_nemotron_and_complete_whisper_rollback():
    contents = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    active = contents.split("# Dedicated transcription vLLM service", 1)[1].split(
        "# Whisper rollback", 1
    )[0]
    rollback = contents.split("# Whisper rollback", 1)[1].split("# App Reranker", 1)[0]

    expected_active = {
        "TRANSCRIBE_VLLM_MODEL=nvidia/nemotron-3.5-asr-streaming-0.6b",
        "TRANSCRIBE_VLLM_REVISION=f3d333391852ba876df169dcc9ba902d25b6ab0b",
        "TRANSCRIBE_VLLM_SERVED_MODEL_NAME=nemotron-3.5-asr-streaming-0.6b",
        "TRANSCRIBE_VLLM_TOKENIZER=nvidia/nemotron-3.5-asr-streaming-0.6b",
        "TRANSCRIBE_VLLM_TENSOR_PARALLEL_SIZE=1",
        "TRANSCRIBE_VLLM_GPU_MEMORY_UTILIZATION=0.20",
        "TRANSCRIBE_VLLM_MAX_MODEL_LEN=50000",
        "TRANSCRIBE_VLLM_DTYPE=auto",
        "TRANSCRIBE_VLLM_ALLOW_LONG_MAX_MODEL_LEN=1",
        "TRANSCRIBE_VLLM_TRUST_REMOTE_CODE=1",
        'TRANSCRIBE_VLLM_EXTRA_ARGS="--enforce-eager --max-num-seqs 1 '
        "--max-num-batched-tokens 50000 --generation-config "
        '/opt/aquillm/nemotron-generation-config"',
    }
    for expected in expected_active:
        assert expected in active
    assert "bitsandbytes" not in active
    assert "turboquant" not in active.lower()

    expected_rollback = {
        "# TRANSCRIBE_VLLM_MODEL=openai/whisper-large-v3-turbo",
        "# TRANSCRIBE_VLLM_REVISION=",
        "# TRANSCRIBE_VLLM_SERVED_MODEL_NAME=whisper-large-v3-turbo",
        "# TRANSCRIBE_VLLM_TOKENIZER=openai/whisper-large-v3-turbo",
        "# TRANSCRIBE_VLLM_TENSOR_PARALLEL_SIZE=1",
        "# TRANSCRIBE_VLLM_GPU_MEMORY_UTILIZATION=0.08",
        "# TRANSCRIBE_VLLM_MAX_MODEL_LEN=448",
        "# TRANSCRIBE_VLLM_DTYPE=float16",
        "# TRANSCRIBE_VLLM_ALLOW_LONG_MAX_MODEL_LEN=0",
        "# TRANSCRIBE_VLLM_TRUST_REMOTE_CODE=1",
        "# INGEST_TRANSCRIBE_MODEL=whisper-large-v3-turbo",
        "--quantization bitsandbytes --load-format bitsandbytes",
        "--model-loader-extra-config '{\\\"load_in_8bit\\\":true}'",
        "--max-num-seqs 1 --max-num-batched-tokens 448",
        '--limit-mm-per-prompt \'{\\"audio\\":{\\"count\\":1,\\"length\\":30}}\'',
    }
    for expected in expected_rollback:
        assert expected in rollback
    assert "generation-config" not in rollback

    assert contents.count("\nINGEST_TRANSCRIBE_MODEL=") == 1
    assert "INGEST_TRANSCRIBE_MODEL=nemotron-3.5-asr-streaming-0.6b" in contents
    assert contents.count("\nINGEST_TRANSCRIBE_LANGUAGE=") == 1


def test_no_gpu_compose_keeps_hosted_whisper_model():
    contents = (REPO_ROOT / "deploy" / "compose" / "no_gpu_dev.yml").read_text(
        encoding="utf-8"
    )
    assert "INGEST_TRANSCRIBE_MODEL: whisper-1" in contents


def test_start_script_recovers_transcribe_args_by_service_kind_and_gates_revision():
    contents = (REPO_ROOT / "deploy" / "scripts" / "vllm_start.sh").read_text(
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
