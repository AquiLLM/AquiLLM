"""Static contract tests for the pinned Nemotron transcription image."""

from __future__ import annotations

import json
import re
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _markdown_section(document: str, heading: str) -> str:
    """Return one Markdown section without coupling tests to prose wrapping."""
    lines = document.splitlines(keepends=True)
    in_fence = False
    start: int | None = None
    level: int | None = None
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^(?P<marks>#+)\s+(?P<title>.*?)\s*$", line)
        if match is None:
            continue
        if start is None and match.group("title") == heading:
            start = index + 1
            level = len(match.group("marks"))
        elif start is not None and len(match.group("marks")) <= level:
            return "".join(lines[start:index])
    assert start is not None, f"missing Markdown heading: {heading}"
    return "".join(lines[start:])


def test_transcription_image_is_pinned_and_self_contained():
    root = _repo_root()
    dockerfile = (root / "deploy/docker/vllm/Dockerfile.transcribe").read_text(
        encoding="utf-8"
    )

    assert "FROM vllm/vllm-openai:v0.21.0" in dockerfile
    assert "transformers==5.13.0" in dockerfile
    transformers_install = (
        "RUN python3 -m pip install --no-cache-dir --upgrade "
        '"transformers==5.13.0" soundfile'
    )
    assert (
        f"{transformers_install}\n"
        'RUN python3 -m pip install --no-cache-dir "librosa==0.11.0"'
    ) in dockerfile
    assert "libsndfile1" in dockerfile
    assert "ffmpeg" in dockerfile
    assert "soundfile" in dockerfile
    assert "deploy/vllm_plugins/nemotron_asr" in dockerfile
    assert "pip wheel" in dockerfile
    assert "pip install" in dockerfile
    assert "--no-index" in dockerfile
    assert "python3 -m pip check" in dockerfile
    assert "/opt/aquillm/nemotron-generation-config" in dockerfile
    assert (
        "COPY ./deploy/docker/vllm/probe_nemotron_plugin.py /probe_nemotron_plugin.py"
        in dockerfile
    )
    assert "deploy/scripts/vllm_start.sh" in dockerfile
    assert "deploy/scripts/parse_vllm_extra_args.py" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert 'ENTRYPOINT ["/vllm_start.sh"]' in dockerfile
    assert "ENV VLLM_EXTRA_ARGS" not in dockerfile


def test_generation_config_has_the_required_rnnt_ids():
    config_path = (
        _repo_root()
        / "deploy/docker/vllm/nemotron_generation_config/generation_config.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["eos_token_id"] == 13087
    assert config["decoder_start_token_id"] == 13087
    assert config["pad_token_id"] == 0


def test_probe_checks_the_pinned_runtime_and_plugin_contract():
    probe = (_repo_root() / "deploy/docker/vllm/probe_nemotron_plugin.py").read_text(
        encoding="utf-8"
    )

    for required in (
        "0.21.0",
        "5.13.0",
        "0.11.0",
        "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "f3d333391852ba876df169dcc9ba902d25b6ab0b",
        "TemporaryDirectory",
        "AutoProcessor.from_pretrained",
        "NemotronAsrStreamingFeatureExtractor",
        "sampling_rate",
        "np.zeros(16_000, dtype=np.float32)",
        'language="auto"',
        'return_tensors="pt"',
        "(1, 101, 128)",
        "(1, 101)",
        'endswith((".safetensors", ".bin", ".pt"))',
        "VLLM_USE_V2_MODEL_RUNNER",
        "vllm.general_plugins",
        "load_general_plugins",
        "Nemotron3_5AsrForRNNT",
        "Nemotron3_5AsrProcessor",
        "supports_multimodal",
        "supports_transcription",
        "is_attention_free",
        "DefaultModelState",
        "SchedulerConfig",
        "max_model_len=50_000",
        "max_num_batched_tokens=50_000",
        "max_num_seqs=1",
        "GenerationConfig",
        "local_files_only=True",
        "tokenizers",
        "safetensors",
    ):
        assert required in probe


def test_standard_images_remain_nemotron_free():
    root = _repo_root()
    for filename in ("Dockerfile", "Dockerfile.genesis"):
        contents = (root / "deploy/docker/vllm" / filename).read_text(encoding="utf-8")
        assert "Nemotron" not in contents
        assert "aquillm-vllm-nemotron-asr" not in contents
        assert "transformers==5.13.0" not in contents


def test_dockerignore_excludes_disposable_virtual_environments():
    ignored = (_repo_root() / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert ".venv*/" in ignored
    assert "venv/" in ignored


def test_readme_documents_the_nemotron_operator_contract():
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    section = " ".join(
        _markdown_section(readme, "Local GPU ASR (Nemotron 3.5)").split()
    )

    for required in (
        "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "served as `nemotron-3.5-asr-streaming-0.6b`",
        "pinned at revision `f3d333391852ba876df169dcc9ba902d25b6ab0b`",
        "POST /v1/audio/transcriptions",
        ".text",
        "batch/offline only",
        "390 seconds",
        "--max-num-seqs 1",
        "no concurrency promise",
        "dtype float32",
        "FP32",
        "VLLM_USE_V2_MODEL_RUNNER=0",
        "--enforce-eager",
        "INGEST_TRANSCRIBE_LANGUAGE",
        "automatic language detection",
        "en-US",
        "NEMOTRON_ASR_ALLOW_ADAPTATION_LANGUAGES=1",
        "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/"
        "blob/f3d333391852ba876df169dcc9ba902d25b6ab0b/README.md",
        "https://openmdw.ai/license/1-1/",
        "distinct from the AquiLLM source license",
        "does not redistribute them",
        "--env-file .env -f deploy/compose/base.yml",
        "--profile vllm",
        "--no-deps --wait --wait-timeout 900",
        "http://localhost:8000/health",
        "http://127.0.0.1:8005/v1/models",
        "http://127.0.0.1:8005/v1/audio/transcriptions",
        "tests/fixtures/audio/librispeech_1272-128104-0000.flac",
        "0.20",
        "features, activations, and runtime overhead",
        "no_gpu_dev",
        "whisper-1",
        "development",
        "does not publish host port 8005",
    ):
        assert required in section

    for outside_release in (
        "translations",
        "diarization",
        "word timestamps",
        "verbose",
        "realtime WebSockets",
    ):
        assert outside_release.lower() in section.lower()


def test_readme_documents_a_complete_environment_only_whisper_rollback():
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")
    rollback = " ".join(_markdown_section(readme, "Whisper rollback").split())

    for setting in (
        "TRANSCRIBE_VLLM_MODEL=openai/whisper-large-v3-turbo",
        "TRANSCRIBE_VLLM_REVISION=",
        "TRANSCRIBE_VLLM_SERVED_MODEL_NAME=whisper-large-v3-turbo",
        "TRANSCRIBE_VLLM_TOKENIZER=openai/whisper-large-v3-turbo",
        "TRANSCRIBE_VLLM_TENSOR_PARALLEL_SIZE=1",
        "TRANSCRIBE_VLLM_GPU_MEMORY_UTILIZATION=0.08",
        "TRANSCRIBE_VLLM_MAX_MODEL_LEN=448",
        "TRANSCRIBE_VLLM_DTYPE=float16",
        "TRANSCRIBE_VLLM_ALLOW_LONG_MAX_MODEL_LEN=0",
        "TRANSCRIBE_VLLM_TRUST_REMOTE_CODE=1",
        "--quantization bitsandbytes",
        "--load-format bitsandbytes",
        '\\"load_in_8bit\\":true',
        "--max-num-seqs 1",
        "--max-num-batched-tokens 448",
        '\\"audio\\":{\\"count\\":1,\\"length\\":30}',
        "INGEST_TRANSCRIBE_MODEL=whisper-large-v3-turbo",
        "--force-recreate vllm_transcribe",
        "--force-recreate web worker",
        "without rebuilding",
        "not automatic",
        "not a second resident model",
    ):
        assert setting in rollback

    assert "--generation-config" not in rollback
    assert "TRANSCRIBE_VLLM_ALLOW_LONG_MAX_MODEL_LEN=1" not in rollback


def test_whisperx_docs_use_the_configurable_asr_baseline():
    root = _repo_root()
    paths = (
        root / "docs/specs/2026-03-30-whisperx-transcription-design.md",
        root / "docs/roadmap/plans/pending/"
        "2026-03-30-whisperx-transcription-implementation.md",
    )
    stale_phrases = (
        "serving a whisper-family model",
        "existing whisper (vllm) deployment",
        "vllm_transcribe serves whisper",
        "vllm whisper (v1)",
        "existing **vllm whisper**",
        "enhancer for the existing whisper (vllm) stack",
    )

    for path in paths:
        contents = path.read_text(encoding="utf-8").lower()
        assert "configured openai-compatible asr" in contents
        assert "nemotron" in contents and "default" in contents
        assert "whisper rollback" in contents
        assert "ingest_transcribe_provider=openai" in contents
        for stale in stale_phrases:
            assert stale not in contents


def test_roadmap_tracks_the_optional_whisperx_enhancement():
    roadmap = (_repo_root() / "docs/roadmap/roadmap-status.md").read_text(
        encoding="utf-8"
    )
    row = next(
        line
        for line in roadmap.splitlines()
        if line.startswith("| Optional WhisperX enhancement ")
    )

    assert "docs/specs/2026-03-30-whisperx-transcription-design.md" in row
    assert (
        "docs/roadmap/plans/pending/2026-03-30-whisperx-transcription-implementation.md"
        in row
    )
    assert "**Not started**" in row
    assert "Nemotron default" in row
    assert "Whisper rollback" in row
    assert "without changing the baseline contract" in row
