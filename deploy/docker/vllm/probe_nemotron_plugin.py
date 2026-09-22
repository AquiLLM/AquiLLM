#!/usr/bin/env python3
"""Fail-fast runtime contract probe for the pinned Nemotron ASR image."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

VLLM_VERSION = "0.21.0"
TRANSFORMERS_VERSION = "5.13.0"
LIBROSA_VERSION = "0.11.0"
ARCHITECTURE = "Nemotron3_5AsrForRNNT"
PLUGIN_NAME = "aquillm_nemotron_asr"
PLUGIN_TARGET = "aquillm_vllm_nemotron_asr:register"
MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_REVISION = "f3d333391852ba876df169dcc9ba902d25b6ab0b"
DEFAULT_GENERATION_CONFIG = Path("/opt/aquillm/nemotron-generation-config")


def _require_version(distribution: str, expected: str) -> None:
    actual = importlib.metadata.version(distribution)
    print(f"{distribution}={actual}")
    if actual != expected:
        raise AssertionError(f"{distribution} must be {expected}; found {actual}")


def _print_runtime_versions() -> None:
    _require_version("vllm", VLLM_VERSION)
    _require_version("transformers", TRANSFORMERS_VERSION)
    _require_version("librosa", LIBROSA_VERSION)
    for distribution in ("torch", "tokenizers", "safetensors"):
        print(f"{distribution}={importlib.metadata.version(distribution)}")

    import torch

    print(f"torch_cuda={torch.version.cuda}")


def _assert_plugin_registration() -> None:
    from vllm import envs

    if envs.VLLM_USE_V2_MODEL_RUNNER:
        raise AssertionError("Nemotron ASR requires VLLM_USE_V2_MODEL_RUNNER=0")

    entry_points = importlib.metadata.entry_points(group="vllm.general_plugins")
    matching = [entry for entry in entry_points if entry.name == PLUGIN_NAME]
    if len(matching) != 1:
        raise AssertionError(
            f"expected exactly one {PLUGIN_NAME!r} vllm.general_plugins entry point; "
            f"found {len(matching)}"
        )
    if matching[0].value != PLUGIN_TARGET:
        raise AssertionError(
            f"{PLUGIN_NAME} must target {PLUGIN_TARGET}; found {matching[0].value}"
        )

    from vllm.plugins import load_general_plugins

    load_general_plugins()
    from vllm.model_executor.models.registry import ModelRegistry

    registered = ModelRegistry.models.get(ARCHITECTURE)
    if registered is None:
        raise AssertionError(f"vLLM plugin did not register {ARCHITECTURE}")
    module_name = getattr(registered, "module_name", None)
    class_name = getattr(registered, "class_name", None)
    if (module_name, class_name) != (
        "aquillm_vllm_nemotron_asr.model",
        ARCHITECTURE,
    ):
        raise AssertionError(
            f"{ARCHITECTURE} registered to {(module_name, class_name)!r}, "
            "not the local lazy target"
        )
    model_info = registered.inspect_model_cls()
    for name in (
        "supports_multimodal",
        "supports_transcription",
        "supports_transcription_only",
        "is_attention_free",
    ):
        if getattr(model_info, name, False) is not True:
            raise AssertionError(f"lazy model inspection did not report {name}=True")


def _assert_model_contract() -> None:
    from aquillm_vllm_nemotron_asr.model import Nemotron3_5AsrForRNNT as VllmNemotron
    from transformers import Nemotron3_5AsrForRNNT, Nemotron3_5AsrProcessor
    from vllm.model_executor.models.interfaces import (
        is_attention_free,
        supports_multimodal,
        supports_transcription,
    )

    # Imports above deliberately prove the checkpoint-facing HF types exist.
    assert Nemotron3_5AsrForRNNT and Nemotron3_5AsrProcessor
    for predicate in (supports_multimodal, supports_transcription, is_attention_free):
        if not predicate(VllmNemotron):
            raise AssertionError(f"wrapper is missing {predicate.__name__}")

    # Call the exact V1 task-discovery implementation without constructing a
    # model or downloading checkpoint weights.
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    state = DefaultModelState.__new__(DefaultModelState)
    state.model = VllmNemotron
    tasks = state.get_supported_generation_tasks()
    if tasks != ("transcription",):
        raise AssertionError(
            f"expected transcription-only task discovery; found {tasks}"
        )


def _assert_processor_contract() -> None:
    from transformers import AutoProcessor, Nemotron3_5AsrProcessor
    from transformers.models.nemotron_asr_streaming import (
        NemotronAsrStreamingFeatureExtractor,
    )

    with TemporaryDirectory(prefix="aquillm-nemotron-processor-") as cache_directory:
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_directory,
        )
        if type(processor) is not Nemotron3_5AsrProcessor:
            raise AssertionError(
                "AutoProcessor must load the exact Nemotron3_5AsrProcessor class; "
                f"found {type(processor)!r}"
            )
        feature_extractor = processor.feature_extractor
        if type(feature_extractor) is not NemotronAsrStreamingFeatureExtractor:
            raise AssertionError(
                "Nemotron processor must load the exact "
                "NemotronAsrStreamingFeatureExtractor class; "
                f"found {type(feature_extractor)!r}"
            )
        if feature_extractor.sampling_rate != 16_000:
            raise AssertionError(
                "Nemotron feature extractor sampling_rate must be 16000; "
                f"found {feature_extractor.sampling_rate}"
            )

        inputs = processor(
            np.zeros(16_000, dtype=np.float32),
            sampling_rate=16_000,
            language="auto",
            return_tensors="pt",
        )
        expected_shapes = {
            "input_features": (1, 101, 128),
            "attention_mask": (1, 101),
        }
        for name, expected_shape in expected_shapes.items():
            actual_shape = tuple(inputs[name].shape)
            if actual_shape != expected_shape:
                raise AssertionError(
                    f"processor {name} shape must be {expected_shape}; "
                    f"found {actual_shape}"
                )

        forbidden_paths = [
            path
            for path in Path(cache_directory).rglob("*")
            if path.name.lower().endswith((".safetensors", ".bin", ".pt"))
        ]
        if forbidden_paths:
            raise AssertionError(
                "processor probe downloaded checkpoint weights: "
                + ", ".join(str(path) for path in forbidden_paths)
            )


def _assert_scheduler_contract() -> None:
    from vllm.config import SchedulerConfig

    scheduler = SchedulerConfig(
        max_model_len=50_000,
        is_encoder_decoder=True,
        is_multimodal_model=True,
        max_num_batched_tokens=50_000,
        max_num_seqs=1,
    )
    expected_budgets = {
        "max_num_batched_tokens": 50_000,
        "max_num_encoder_input_tokens": 50_000,
        "encoder_cache_size": 50_000,
        "max_num_seqs": 1,
    }
    for name, expected in expected_budgets.items():
        actual = getattr(scheduler, name)
        if actual != expected:
            raise AssertionError(
                f"SchedulerConfig.{name} must be {expected}; found {actual}"
            )
    if scheduler.enable_chunked_prefill:
        raise AssertionError(
            "encoder-decoder SchedulerConfig must disable chunked prefill"
        )
    if not scheduler.disable_chunked_mm_input:
        raise AssertionError(
            "encoder-decoder SchedulerConfig must disable chunked MM input"
        )


def _assert_generation_config(config_directory: Path) -> None:
    from transformers import GenerationConfig

    config_file = config_directory / "generation_config.json"
    if not config_file.is_file():
        raise AssertionError(f"generation config file is missing: {config_file}")
    config = GenerationConfig.from_pretrained(config_directory, local_files_only=True)
    for name, expected in (
        ("eos_token_id", 13087),
        ("decoder_start_token_id", 13087),
        ("pad_token_id", 0),
    ):
        actual = getattr(config, name)
        if actual != expected:
            raise AssertionError(
                f"generation config {name} must be {expected}; found {actual}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation-config",
        type=Path,
        default=DEFAULT_GENERATION_CONFIG,
        help="vLLM --generation-config directory (the deployed default)",
    )
    args = parser.parse_args()

    _print_runtime_versions()
    _assert_plugin_registration()
    _assert_model_contract()
    _assert_processor_contract()
    _assert_scheduler_contract()
    _assert_generation_config(args.generation_config)
    print("Nemotron ASR vLLM plugin probe passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
