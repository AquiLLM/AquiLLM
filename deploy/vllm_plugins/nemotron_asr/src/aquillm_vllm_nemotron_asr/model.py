"""vLLM protocol shell for Nemotron RNNT transcription.

Task 5 intentionally contains no checkpoint construction or inference logic;
Task 6 fills those weight-bearing methods while preserving this public protocol.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import cast

import torch
from torch import nn
from transformers import Nemotron3_5AsrForRNNT as HfNemotron3_5AsrForRNNT
from transformers.models.nemotron3_5_asr.generation_nemotron3_5_asr import (
    Nemotron3_5AsrRNNTDecoderCache,
)
from vllm import envs as vllm_envs
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

from .decoding import BLANK_TOKEN_ID, greedy_rnnt_decode
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
from .state import ReplayState

_REQUIRED_MAX_MODEL_LEN = 50_000
_VOCAB_SIZE = 13_088
_CHECKPOINT_PREFIXES = (
    "encoder.",
    "decoder.",
    "encoder_projector.",
    "prompt_projector.",
    "joint.",
)
_WEIGHT_NAME_ALLOWLIST: frozenset[str] = frozenset()


class _TorchRNNTAdapter:
    """Torch/HF bridge for the framework-independent RNN-T decoder loop.

    Transformers 5.13's decoder mutates a
    ``Nemotron3_5AsrRNNTDecoderCache`` in place.  Passing that same cache to
    the next call gives its exact initial-blank/nonblank/later-blank behavior.
    """

    def __init__(self, model: Nemotron3_5AsrForRNNT) -> None:
        self._model = model

    def predict(
        self,
        frame: object,
        previous_token: int,
        cache: object | None,
    ) -> tuple[int, object]:
        frame_tensor = cast(torch.Tensor, frame)
        decoder_cache = cast(Nemotron3_5AsrRNNTDecoderCache | None, cache)
        if decoder_cache is None:
            decoder_cache = Nemotron3_5AsrRNNTDecoderCache(self._model.config)

        decoder_input_ids = torch.tensor(
            [[previous_token]], dtype=torch.long, device=frame_tensor.device
        )
        decoder_hidden_states = self._model.decoder(
            decoder_input_ids, cache=decoder_cache
        )
        logits = self._model.joint(
            encoder_hidden_states=frame_tensor.reshape(1, 1, 1, -1),
            decoder_hidden_states=decoder_hidden_states[:, None, :, :],
        ).squeeze(2)
        predicted = int(logits.reshape(-1, logits.shape[-1]).argmax(dim=-1).item())
        return predicted, decoder_cache


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
        # vLLM 0.21 instantiates both endpoint handlers for every
        # transcription-capable model. The compatibility wrapper rejects real
        # translation requests before generation, but translation handler
        # construction itself must remain startup-safe.
        if task_type not in {"transcribe", "translate"}:
            raise ValueError(f"Nemotron does not recognize task type {task_type!r}")
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
        self._validate_runtime_config(vllm_config)
        self.config = vllm_config.model_config.hf_config
        # Deliberately construct from config, rather than from_pretrained: vLLM
        # owns checkpoint loading.  Rehome only these modules so their parameter
        # names stay checkpoint-identical (``encoder.*``, never ``model.*``).
        hf_model = HfNemotron3_5AsrForRNNT(self.config)
        self.encoder = hf_model.encoder
        self.decoder = hf_model.decoder
        self.encoder_projector = hf_model.encoder_projector
        self.prompt_projector = hf_model.prompt_projector
        self.joint = hf_model.joint
        self.replay_state = ReplayState()

    @staticmethod
    def _validate_runtime_config(vllm_config: VllmConfig) -> None:
        scheduler = vllm_config.scheduler_config
        model = vllm_config.model_config
        parallel = vllm_config.parallel_config
        if vllm_envs.VLLM_USE_V2_MODEL_RUNNER:
            raise ValueError("Nemotron ASR does not support the V2 model runner")
        if scheduler.max_num_seqs != 1:
            raise ValueError("Nemotron ASR requires scheduler max_num_seqs=1")
        if not model.enforce_eager:
            raise ValueError("Nemotron ASR requires enforce_eager=True")
        if parallel.tensor_parallel_size != 1:
            raise ValueError("Nemotron ASR requires tensor_parallel_size=1")
        if model.max_model_len != _REQUIRED_MAX_MODEL_LEN:
            raise ValueError(
                f"Nemotron ASR requires max_model_len={_REQUIRED_MAX_MODEL_LEN}"
            )
        if scheduler.max_num_batched_tokens != _REQUIRED_MAX_MODEL_LEN:
            raise ValueError(
                "Nemotron ASR requires "
                f"max_num_batched_tokens={_REQUIRED_MAX_MODEL_LEN}"
            )
        if scheduler.max_num_encoder_input_tokens != _REQUIRED_MAX_MODEL_LEN:
            raise ValueError(
                "Nemotron ASR requires "
                f"max_num_encoder_input_tokens={_REQUIRED_MAX_MODEL_LEN}"
            )
        if model.hf_config.vocab_size != _VOCAB_SIZE:
            raise ValueError(f"Nemotron ASR requires vocab_size={_VOCAB_SIZE}")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        input_features = cast(torch.Tensor, kwargs["input_features"])
        attention_mask = cast(torch.Tensor | None, kwargs.get("attention_mask"))
        prompt_ids = cast(torch.Tensor | None, kwargs.get("prompt_ids"))
        num_lookahead_tokens = kwargs.get("num_lookahead_tokens")

        encoder_kwargs: dict[str, object] = {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }
        if num_lookahead_tokens is not None:
            if isinstance(num_lookahead_tokens, torch.Tensor):
                lookahead_values = num_lookahead_tokens.reshape(-1)
                if lookahead_values.numel() == 0 or not torch.equal(
                    lookahead_values, lookahead_values[:1].expand_as(lookahead_values)
                ):
                    raise ValueError(
                        "Nemotron ASR requires a uniform num_lookahead_tokens value"
                    )
                num_lookahead_tokens = int(lookahead_values[0].item())
            encoder_kwargs["num_lookahead_tokens"] = num_lookahead_tokens
        encoder_outputs = self.encoder(**encoder_kwargs)
        hidden_states = encoder_outputs.last_hidden_state
        if prompt_ids is None:
            prompt_ids = torch.full(
                (hidden_states.shape[0],),
                self.config.default_prompt_id,
                dtype=torch.long,
                device=hidden_states.device,
            )
        prompt_ids = prompt_ids.to(hidden_states.device)
        one_hot = torch.nn.functional.one_hot(
            prompt_ids, num_classes=self.config.num_prompts
        ).to(hidden_states.dtype)
        one_hot = one_hot[:, None, :].expand(-1, hidden_states.shape[1], -1)
        pooled = self.encoder_projector(
            self.prompt_projector(torch.cat([hidden_states, one_hot], dim=-1))
        )

        post_encoder_mask = getattr(encoder_outputs, "attention_mask", None)
        if post_encoder_mask is None:
            post_encoder_mask = attention_mask
        if post_encoder_mask is None:
            lengths = [pooled.shape[1]] * pooled.shape[0]
        else:
            lengths = [int(length) for length in post_encoder_mask.sum(dim=-1).tolist()]
        return [pooled[index, :length] for index, length in enumerate(lengths)]

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # This model has a fixed one-token decoder prompt.  The V1 runner only
        # needs a 2-D active-token embedding carrier; RNNT work happens in the
        # encoder prefill, not in a decoder attention stack.
        hidden_size = int(getattr(self.config, "decoder_hidden_size", 1))
        return torch.zeros(
            (input_ids.numel(), hidden_size),
            dtype=torch.float32,
            device=input_ids.device,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if encoder_outputs is not None:
            if len(encoder_outputs) != 1:
                raise ValueError("Nemotron ASR supports one encoder output per request")
            transcript = greedy_rnnt_decode(
                [encoder_outputs[0]],
                _TorchRNNTAdapter(self),
            )
            # This executes for both actual current 0.21 encoder prefills and a
            # future cached encoder tensor path.  Never retain a stale replay.
            self.replay_state.replace_real(transcript)
        else:
            # vLLM profiling has no audio.  It may seed a placeholder but can
            # never overwrite a completed real request.
            self.replay_state.replace_profiling(())

        forced_ids = self.replay_state.forced_ids(
            (positions.to(dtype=torch.long) + 1).tolist()
        )
        # V1's logits selector indexes hidden_states[logits_indices], therefore
        # retain an explicit feature axis instead of a scalar vector.
        return torch.tensor(forced_ids, dtype=torch.long, device=positions.device).view(
            -1, 1
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forced_ids = hidden_states.to(dtype=torch.long).reshape(-1)
        vocab_size = _VOCAB_SIZE
        if forced_ids.numel() and (
            forced_ids.min().item() < 0 or forced_ids.max().item() >= vocab_size
        ):
            raise ValueError("forced token ID is outside the Nemotron vocabulary")
        logits = torch.full(
            (forced_ids.numel(), vocab_size),
            float("-inf"),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        if forced_ids.numel():
            logits.scatter_(1, forced_ids[:, None], 0.0)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.models.utils import default_weight_loader

        parameters = dict(self.named_parameters())
        required_names = set(parameters) - _WEIGHT_NAME_ALLOWLIST
        loaded_names: set[str] = set()

        for checkpoint_name, weight in weights:
            if not checkpoint_name.startswith(_CHECKPOINT_PREFIXES):
                raise ValueError(f"unknown checkpoint parameter {checkpoint_name!r}")
            if checkpoint_name not in parameters:
                if checkpoint_name in _WEIGHT_NAME_ALLOWLIST:
                    continue
                raise ValueError(f"unknown checkpoint parameter {checkpoint_name!r}")
            if checkpoint_name in loaded_names:
                raise ValueError(f"duplicate checkpoint parameter {checkpoint_name!r}")
            default_weight_loader(parameters[checkpoint_name], weight)
            loaded_names.add(checkpoint_name)

        missing = required_names - loaded_names
        if missing:
            raise ValueError(
                "missing checkpoint parameters: " + ", ".join(sorted(missing))
            )
        return loaded_names
