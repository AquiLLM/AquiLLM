"""vLLM 0.21 multimodal preprocessing for Nemotron 3.5 streaming ASR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.renderers import TokenizeParams

SAMPLE_RATE = 16_000
MAX_AUDIO_DURATION_S = 390


class NemotronProcessingInfo(BaseProcessingInfo):
    """Static preprocessing facts for the pinned Nemotron checkpoint."""

    def get_default_tok_params(self) -> TokenizeParams:
        # Nemotron's HF processor owns audio feature and language-prompt setup.
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=SAMPLE_RATE, target_channels=1)

    @property
    def skip_prompt_length_check(self) -> bool:
        return True

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_num_audio_tokens(self) -> int:
        """Return the checkpoint encoder's fixed positional upper bound."""
        config = self.get_hf_config()
        return int(config.encoder_config.max_position_embeddings)


class NemotronDummyInputsBuilder(BaseDummyInputsBuilder[NemotronProcessingInfo]):
    """Create one maximum-duration waveform without constructing decoder replay."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # The encoder accepts audio features only; decoder replay is unrelated.
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        audio_overrides = mm_options.get("audio")
        return {
            "audio": self._get_dummy_audios(
                length=MAX_AUDIO_DURATION_S * SAMPLE_RATE,
                num_audios=mm_counts.get("audio", 0),
                overrides=audio_overrides,
            )
        }


class NemotronMultiModalProcessor(EncDecMultiModalProcessor[NemotronProcessingInfo]):
    """Translate vLLM audio payloads to the Nemotron HF processor contract."""

    skip_decoder_start_token = True

    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        return True

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        return [0]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> dict[str, Any]:
        """Call Nemotron's audio-only processor without decoder text injection."""
        audios = mm_data.get("audios", ())
        if not isinstance(audios, Sequence):
            raise ValueError("Nemotron expects a sequence of audio items")
        language = mm_kwargs.get("language") or "auto"
        processor = self.info.get_hf_processor(**mm_kwargs)
        outputs = self.info.ctx.call_hf_processor(
            processor,
            {"audio": audios},
            {"sampling_rate": SAMPLE_RATE, "language": language},
        )

        # Transformers 5.13 emits this as one Python scalar. vLLM field parsing
        # needs an item-aligned batch tensor even though this release permits one.
        lookahead = outputs["num_lookahead_tokens"]
        lookahead_tensor = torch.as_tensor(lookahead, dtype=torch.long)
        if lookahead_tensor.ndim == 0:
            lookahead_tensor = torch.full(
                (len(audios),), int(lookahead_tensor.item()), dtype=torch.long
            )
        elif lookahead_tensor.ndim != 1 or lookahead_tensor.numel() != len(audios):
            raise ValueError("num_lookahead_tokens must provide one value per audio")
        outputs["num_lookahead_tokens"] = lookahead_tensor

        # EncDecMultiModalProcessor removes this encoder-only placeholder before
        # it parses multimodal fields. It is deliberately not supplied to HF.
        outputs["input_ids"] = torch.zeros((len(audios), 1), dtype=torch.long)
        return outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: object,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "input_features": MultiModalFieldConfig.batched("audio"),
            "attention_mask": MultiModalFieldConfig.batched("audio"),
            "prompt_ids": MultiModalFieldConfig.batched("audio"),
            "num_lookahead_tokens": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        return [
            PromptReplacement(
                modality="audio",
                target=[0],
                replacement=[0] * self.info.get_num_audio_tokens(),
            )
        ]
