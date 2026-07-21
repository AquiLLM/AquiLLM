"""Pinned vLLM 0.21 transcription protocol checks for the ASR model shell."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from aquillm_vllm_nemotron_asr.model import Nemotron3_5AsrForRNNT  # noqa: E402
from vllm.config import SpeechToTextConfig  # noqa: E402
from vllm.config.speech_to_text import SpeechToTextParams  # noqa: E402
from vllm.model_executor.models.interfaces import (  # noqa: E402
    is_attention_free,
    supports_multimodal,
    supports_transcription,
)
from vllm.sampling_params import SamplingParams  # noqa: E402


def _params(**overrides: object) -> SpeechToTextParams:
    values = {
        "audio": object(),
        "stt_config": SpeechToTextConfig(sample_rate=16_000),
        "model_config": SimpleNamespace(),
        "language": None,
        "hotwords": None,
        "task_type": "transcribe",
        "request_prompt": "",
        "to_language": None,
    }
    values.update(overrides)
    return SpeechToTextParams(**values)


def test_protocol_advertises_transcription_multimodal_and_attention_free() -> None:
    assert isinstance(Nemotron3_5AsrForRNNT, type)
    assert supports_transcription(Nemotron3_5AsrForRNNT)
    assert supports_multimodal(Nemotron3_5AsrForRNNT)
    assert is_attention_free(Nemotron3_5AsrForRNNT)
    assert Nemotron3_5AsrForRNNT.supports_transcription_only is True


def test_production_bare_languages_cover_every_model_locale() -> None:
    supported = Nemotron3_5AsrForRNNT.supported_languages

    assert {
        locale.split("-", 1)[0]
        for locale in (
            "en-US en-GB es-US es-ES fr-FR fr-CA it-IT pt-BR pt-PT nl-NL de-DE "
            "tr-TR ru-RU ar-AR hi-IN ja-JP ko-KR vi-VN uk-UA pl-PL sv-SE cs-CZ "
            "nb-NO da-DK bg-BG fi-FI hr-HR sk-SK zh-CN hu-HU ro-RO et-EE"
        ).split()
    } <= supported.keys()
    assert Nemotron3_5AsrForRNNT.validate_language("en") == "en-US"
    assert Nemotron3_5AsrForRNNT.validate_language(None) == "auto"


def test_speech_to_text_config_is_batch_only_and_rejects_translation() -> None:
    config = Nemotron3_5AsrForRNNT.get_speech_to_text_config(
        SimpleNamespace(), "transcribe"
    )

    assert config == SpeechToTextConfig(
        sample_rate=16_000,
        max_audio_clip_s=390,
        min_energy_split_window_size=None,
    )
    with pytest.raises(ValueError, match="transcribe"):
        Nemotron3_5AsrForRNNT.get_speech_to_text_config(SimpleNamespace(), "translate")


def test_generation_prompt_has_exact_encoder_decoder_nesting_and_auto_language() -> (
    None
):
    prompt = Nemotron3_5AsrForRNNT.get_generation_prompt(_params())

    # vLLM prompt annotations are TypedDict schemas, represented by dicts.
    assert isinstance(prompt, dict)
    assert set(prompt) == {"encoder_prompt", "decoder_prompt"}
    assert isinstance(prompt["encoder_prompt"], dict)
    assert isinstance(prompt["decoder_prompt"], dict)
    assert prompt["encoder_prompt"]["prompt"] == ""
    assert prompt["encoder_prompt"]["multi_modal_data"] == {
        "audio": (prompt["encoder_prompt"]["multi_modal_data"]["audio"][0], 16_000)
    }
    assert prompt["encoder_prompt"]["mm_processor_kwargs"] == {"language": "auto"}
    assert prompt["decoder_prompt"]["prompt_token_ids"] == [13087]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("request_prompt", "guide me", "prompt"),
        ("hotwords", "vLLM", "hotwords"),
        ("task_type", "translate", "transcribe"),
        ("to_language", "fr", "translation"),
    ],
)
def test_generation_prompt_rejects_unsupported_request_controls(
    field: str, value: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        Nemotron3_5AsrForRNNT.get_generation_prompt(_params(**{field: value}))


def test_post_process_output_only_cleans_text_and_preserves_repetition() -> None:
    assert (
        Nemotron3_5AsrForRNNT.post_process_output("  hello hello <en-US>  ")
        == "hello hello"
    )
    assert Nemotron3_5AsrForRNNT.post_process_output("again again") == "again again"


def test_vllm_default_detokenization_skips_special_ids_before_text_cleanup() -> None:
    # 13087 is the tokenizer pad / model terminal token. The serving path
    # supplies decoded text to post_process_output, so special-ID removal must
    # happen in vLLM detokenization rather than this text-only method.
    assert SamplingParams().skip_special_tokens is True
    assert Nemotron3_5AsrForRNNT.post_process_output("repeat repeat") == "repeat repeat"


def test_model_blank_is_decoder_start_not_hf_processor_blank() -> None:
    config = SimpleNamespace(blank_token_id=13087, decoder_start_token_id=13087)
    processor = SimpleNamespace(blank_token_id=13088)

    assert Nemotron3_5AsrForRNNT.decoder_start_token_id(config, processor) == 13087
