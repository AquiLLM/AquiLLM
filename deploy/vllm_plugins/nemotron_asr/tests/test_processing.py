"""Pinned vLLM 0.21 processor contracts for the Nemotron ASR plugin."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from aquillm_vllm_nemotron_asr.processing import (  # noqa: E402
    MAX_AUDIO_DURATION_S,
    SAMPLE_RATE,
    NemotronDummyInputsBuilder,
    NemotronMultiModalProcessor,
    NemotronProcessingInfo,
)
from vllm.multimodal.inputs import MultiModalFieldConfig  # noqa: E402


class _ProcessorContext:
    def __init__(self, *, max_position_embeddings: int = 5000) -> None:
        self.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                encoder_config=SimpleNamespace(
                    max_position_embeddings=max_position_embeddings
                )
            )
        )
        self.calls: list[tuple[object, dict[str, object], dict[str, object]]] = []

    def get_hf_config(self):
        return self.model_config.hf_config

    def get_hf_processor(self, **kwargs: object) -> object:
        return object()

    def call_hf_processor(self, processor, data, kwargs):
        self.calls.append((processor, data, kwargs))
        return {
            "input_features": torch.zeros((1, 4, 128), dtype=torch.float32),
            "attention_mask": torch.ones((1, 4), dtype=torch.bool),
            "prompt_ids": torch.tensor([1], dtype=torch.long),
            "num_lookahead_tokens": 3,
        }


@pytest.fixture
def processing_info() -> NemotronProcessingInfo:
    return NemotronProcessingInfo(_ProcessorContext())


def _processor_without_constructor(info: NemotronProcessingInfo):
    """Exercise the vLLM override without needing a full parser context."""
    processor = object.__new__(NemotronMultiModalProcessor)
    processor.info = info
    return processor


def test_processing_info_uses_mono_16khz_one_audio_and_checkpoint_bound(
    processing_info: NemotronProcessingInfo,
) -> None:
    parser = processing_info.get_data_parser()

    assert parser.audio_resampler.target_sr == SAMPLE_RATE
    assert parser.target_channels == 1
    assert processing_info.get_supported_mm_limits() == {"audio": 1}
    assert processing_info.get_num_audio_tokens() == 5000


def test_dummy_audio_represents_full_390_seconds_without_transcript_replay(
    processing_info: NemotronProcessingInfo,
) -> None:
    builder = NemotronDummyInputsBuilder(processing_info)

    dummy = builder.get_dummy_mm_data(5000, {"audio": 1}, {})

    assert builder.get_dummy_text({"audio": 1}) == ""
    assert len(dummy["audio"]) == 1
    assert dummy["audio"][0].shape == (MAX_AUDIO_DURATION_S * SAMPLE_RATE,)


def test_hf_processor_receives_audio_only_auto_language_and_normalizes_lookahead(
    processing_info: NemotronProcessingInfo,
) -> None:
    processor = _processor_without_constructor(processing_info)
    audio = object()

    outputs = processor._call_hf_processor(
        prompt="ignored encoder placeholder",
        mm_data={"audios": [audio]},
        mm_kwargs={},
        tok_kwargs={"truncation": False},
    )

    context = processing_info.ctx
    _hf_processor, data, kwargs = context.calls[-1]
    assert data == {"audio": [audio]}
    assert "text" not in data
    assert kwargs == {"sampling_rate": SAMPLE_RATE, "language": "auto"}
    assert set(outputs) >= {
        "input_features",
        "attention_mask",
        "prompt_ids",
        "num_lookahead_tokens",
    }
    assert outputs["num_lookahead_tokens"].dtype is torch.long
    assert outputs["num_lookahead_tokens"].tolist() == [3]


def test_all_nemotron_hf_fields_are_batched_audio(
    processing_info: NemotronProcessingInfo,
) -> None:
    processor = _processor_without_constructor(processing_info)

    fields = processor._get_mm_fields_config({}, {})

    expected = MultiModalFieldConfig.batched("audio")
    assert fields == {
        "input_features": expected,
        "attention_mask": expected,
        "prompt_ids": expected,
        "num_lookahead_tokens": expected,
    }


def test_encoder_prompt_and_audio_replacement_use_checkpoint_bound(
    processing_info: NemotronProcessingInfo,
) -> None:
    processor = _processor_without_constructor(processing_info)

    assert processor.create_encoder_prompt("unused", None) == [0]
    (update,) = processor._get_prompt_updates(None, {}, {})
    assert update.modality == "audio"
    assert update.target == [0]
    assert update.replacement == [0] * 5000
