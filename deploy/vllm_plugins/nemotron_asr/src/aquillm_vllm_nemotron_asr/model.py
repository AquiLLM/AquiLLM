"""vLLM protocol shell for Nemotron RNNT transcription.

Task 5 intentionally contains no checkpoint construction or inference logic;
Task 6 fills those weight-bearing methods while preserving this public protocol.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import (
    ExplicitEncoderDecoderPrompt,
    PromptType,
    TextPrompt,
    TokensPrompt,
)
from vllm.model_executor.models.interfaces import (
    IsAttentionFree,
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsTranscription,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .decoding import BLANK_TOKEN_ID
from .languages import (
    PRODUCTION_LOCALES,
    adaptation_languages_enabled,
    normalize_language,
)
from .processing import (
    MAX_AUDIO_DURATION_S,
    SAMPLE_RATE,
    NemotronDummyInputsBuilder,
    NemotronMultiModalProcessor,
    NemotronProcessingInfo,
)

_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nb": "Norwegian Bokmal",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sv": "Swedish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}
_PRODUCTION_LANGUAGE_CODES = {locale.split("-", 1)[0] for locale in PRODUCTION_LOCALES}


class _SupportedLanguages(dict[str, str]):
    """Expose model ISO keys while satisfying vLLM 0.21's legacy ``no`` map.

    vLLM validates ``set(supported_languages)`` at class creation against its
    internal language table, which calls Norwegian Bokmal ``no``. The model
    and public transcription API correctly use ISO-639-1 ``nb`` instead, so
    ``.keys()`` retains ``nb`` for vLLM's request-facing language surface.
    """

    def __iter__(self):
        for code in super().keys():
            yield "no" if code == "nb" else code


@MULTIMODAL_REGISTRY.register_processor(
    NemotronMultiModalProcessor,
    info=NemotronProcessingInfo,
    dummy_inputs=NemotronDummyInputsBuilder,
)
class Nemotron3_5AsrForRNNT(
    nn.Module,
    SupportsTranscription,
    SupportsMultiModal,
    IsAttentionFree,
):
    """Discovery-safe protocol implementation for the batch-only ASR model."""

    supports_transcription_only = True
    supported_languages = _SupportedLanguages(
        {code: _LANGUAGE_NAMES[code] for code in sorted(_PRODUCTION_LANGUAGE_CODES)}
    )

    @classmethod
    def validate_language(cls, language: str | None) -> str:
        return normalize_language(
            language,
            allow_adaptation=adaptation_languages_enabled(),
        )

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        if task_type != "transcribe":
            raise ValueError("Nemotron supports only the 'transcribe' task")
        return SpeechToTextConfig(
            sample_rate=SAMPLE_RATE,
            max_audio_clip_s=MAX_AUDIO_DURATION_S,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        if stt_params.task_type != "transcribe":
            raise ValueError("Nemotron supports only the 'transcribe' task")
        if stt_params.request_prompt:
            raise ValueError("Nemotron transcription does not support request_prompt")
        if stt_params.hotwords:
            raise ValueError("Nemotron transcription does not support hotwords")
        if stt_params.to_language:
            raise ValueError("Nemotron transcription does not support translation")

        locale = cls.validate_language(stt_params.language)
        return ExplicitEncoderDecoderPrompt(
            encoder_prompt=TextPrompt(
                prompt="",
                multi_modal_data={
                    "audio": (stt_params.audio, stt_params.stt_config.sample_rate)
                },
                mm_processor_kwargs={"language": locale},
            ),
            decoder_prompt=TokensPrompt(prompt_token_ids=[BLANK_TOKEN_ID]),
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int:
        return int(model_config.hf_config.encoder_config.max_position_embeddings)

    @classmethod
    def post_process_output(cls, text: str) -> str:
        """Remove a residual trailing locale tag after vLLM detokenization."""
        return re.sub(r"\s*<[a-z]{2}-[A-Z]{2}>\s*$", "", text).strip()

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return None
        raise ValueError("Nemotron accepts only audio multimodal inputs")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        raise NotImplementedError("Nemotron runtime embedding is implemented in Task 6")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Nemotron runtime embedding is implemented in Task 6")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        raise NotImplementedError("Nemotron runtime forward is implemented in Task 6")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Nemotron runtime logits are implemented in Task 6")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        raise NotImplementedError("Nemotron runtime loading is implemented in Task 6")
