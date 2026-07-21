"""Static contract tests for the pinned Nemotron transcription image."""

from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_transcription_image_is_pinned_and_self_contained():
    root = _repo_root()
    dockerfile = (root / "deploy/docker/vllm/Dockerfile.transcribe").read_text(
        encoding="utf-8"
    )

    assert "FROM vllm/vllm-openai:v0.21.0" in dockerfile
    assert "transformers==5.13.0" in dockerfile
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
