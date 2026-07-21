"""Pinned-Transformers parity checks for the framework-independent RNNT core.

This suite is deliberately container-only: the dedicated transcription image owns
Torch and the exact Transformers release.  The host test environment skips it.
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aquillm_vllm_nemotron_asr.decoding import BLANK_TOKEN_ID, greedy_rnnt_decode


@dataclass
class _ScriptedAdapter:
    predictions: deque[tuple[int, object]]

    def predict(
        self, frame: object, previous_emitted_token: int, cache: object | None
    ) -> tuple[int, object]:
        del frame, previous_emitted_token, cache
        return self.predictions.popleft()


@pytest.mark.container
def test_pinned_nemotron_generate_matches_the_pure_rnnt_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real pinned generation mixin with tiny deterministic modules.

    Task 8 runs this inside the transcription image.  It intentionally invokes
    ``Nemotron3_5AsrForRNNT.generate`` instead of duplicating the Hugging Face
    generation algorithm in the test.
    """

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.13.0":
        pytest.skip("requires the pinned Transformers 5.13.0 transcription image")

    try:
        from transformers import Nemotron3_5AsrConfig, Nemotron3_5AsrForRNNT
        from transformers.models.nemotron3_5_asr import modeling_nemotron3_5_asr
    except ImportError:
        pytest.skip(
            "Nemotron 3.5 ASR classes are unavailable in this Transformers build"
        )

    hidden_size = 4
    vocab_size = BLANK_TOKEN_ID + 1
    scripted_tokens = (91, BLANK_TOKEN_ID, 91, BLANK_TOKEN_ID)

    class TinyEncoder(torch.nn.Module):
        def forward(self, input_features, attention_mask=None, **kwargs):
            del kwargs
            return SimpleNamespace(
                last_hidden_state=input_features,
                attention_mask=attention_mask,
                hidden_states=None,
                attentions=None,
                past_key_values=None,
                padding_cache=None,
            )

    class TinyDecoder(torch.nn.Module):
        """Use the pinned cache object's real blank/nonblank update semantics."""

        def forward(self, input_ids, cache=None):
            decoder_output = torch.zeros(
                input_ids.shape[0],
                input_ids.shape[1],
                hidden_size,
                device=input_ids.device,
            )
            if cache is None:
                return decoder_output

            blank_mask = input_ids[:, -1] == BLANK_TOKEN_ID
            if cache.is_initialized and blank_mask.all():
                return cache.cache

            was_initialized = cache.is_initialized
            if not was_initialized:
                cache.lazy_initialization(decoder_output)
            states = torch.zeros(
                1, input_ids.shape[0], hidden_size, device=input_ids.device
            )
            cache.update(
                decoder_output,
                states,
                states,
                mask=~blank_mask if was_initialized else None,
            )
            return cache.cache

    class ScriptedJoint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._tokens = deque(scripted_tokens)

        def forward(self, decoder_hidden_states, encoder_hidden_states):
            del decoder_hidden_states
            token = self._tokens.popleft()
            logits = torch.full(
                (*encoder_hidden_states.shape[:-1], vocab_size),
                float("-inf"),
                device=encoder_hidden_states.device,
            )
            logits[..., token] = 0
            return logits

    # Avoid allocating the real FastConformer while still constructing the real
    # Nemotron class and invoking its inherited Transformers generation path.
    monkeypatch.setattr(
        modeling_nemotron3_5_asr.AutoModel,
        "from_config",
        staticmethod(lambda config: TinyEncoder()),
    )
    config = Nemotron3_5AsrConfig(
        vocab_size=vocab_size,
        decoder_hidden_size=hidden_size,
        num_decoder_layers=1,
        num_prompts=2,
        prompt_intermediate_size=hidden_size,
        default_prompt_id=0,
        decoder_start_token_id=BLANK_TOKEN_ID,
        eos_token_id=None,
    )
    config.encoder_config.hidden_size = hidden_size
    model = Nemotron3_5AsrForRNNT(config).eval()
    model.encoder = TinyEncoder()
    model.decoder = TinyDecoder()
    model.joint = ScriptedJoint()

    generated = (
        model.generate(
            input_features=torch.zeros((1, 2, hidden_size)),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            prompt_ids=torch.zeros((1,), dtype=torch.long),
            do_sample=False,
            max_new_tokens=len(scripted_tokens),
            return_dict_in_generate=True,
        )
        .sequences[0]
        .tolist()
    )
    transformers_ids = [token for token in generated if token != BLANK_TOKEN_ID]

    pure_adapter = _ScriptedAdapter(
        deque((token, object()) for token in scripted_tokens)
    )
    assert transformers_ids == greedy_rnnt_decode(
        [["frame-0", "frame-1"]], pure_adapter
    )
