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


def test_operator_contract_documents_optional_activation_and_rollback():
    guide = (_repo_root() / "deploy/NEMOTRON_ASR.md").read_text(encoding="utf-8")
    for required in (
        "explicit opt-in",
        "0.08",
        "0.20",
        "0.45",
        "0.12",
        "0.15",
        "131072",
        "--no-deps --wait --wait-timeout 900",
        "INGEST_TRANSCRIBE_LANGUAGE",
        "390 seconds",
        "no concurrency promise",
        "Whisper rollback",
        "disabled",
    ):
        assert required in guide
