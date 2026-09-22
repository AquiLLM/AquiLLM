"""Framework-independent greedy RNN-T decoding primitives.

The adapter owns framework-specific encoder, decoder, joint-network, and cache
objects.  Keeping this control flow free of Torch and Transformers makes its
state transitions directly testable on the host.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

BLANK_TOKEN_ID = 13087
MAX_SYMBOLS_PER_FRAME = 10


class RNNTDecoderAdapter(Protocol):
    """Predict one RNN-T token for an encoder frame.

    ``candidate_cache`` is the effective decoder cache after processing
    ``previous_token``.  For pinned Transformers 5.13 Nemotron, the initial
    blank initializes that cache, later blank inputs retain it, and nonblank
    inputs update it.  The predicted token always becomes the following decoder
    input, including when it is blank.
    """

    def predict(
        self,
        frame: object,
        previous_token: int,
        cache: object | None,
    ) -> tuple[int, object]:
        """Return the next token and cache after the current decoder input."""


def valid_frame_lengths_from_attention_mask(
    attention_mask: Sequence[Sequence[object]],
) -> list[int]:
    """Return each sample's valid encoder-frame count from its 0/1 mask.

    The model's post-encoder masks are valid-frame prefixes, so their sum is
    the crop length used by the decoder.  This intentionally accepts ordinary
    Python nested sequences instead of requiring a tensor framework.
    """

    return [sum(bool(value) for value in sample_mask) for sample_mask in attention_mask]


def greedy_rnnt_decode(
    encoder_frames: Sequence[Sequence[object]],
    adapter: RNNTDecoderAdapter,
    *,
    attention_mask: Sequence[Sequence[object]] | None = None,
    blank_token_id: int = BLANK_TOKEN_ID,
    max_symbols_per_frame: int = MAX_SYMBOLS_PER_FRAME,
) -> list[int]:
    """Decode exactly one utterance with Transformers-compatible RNN-T control flow.

    This release intentionally supports a batch size of one because the vLLM
    wrapper stores one replay sequence.  Blank emissions advance the encoder
    frame and are never emitted; nonblank tokens remain on the current frame
    until a blank or the per-frame symbol limit advances it.
    """

    if len(encoder_frames) != 1:
        raise ValueError("greedy RNNT decoding currently requires batch size 1")
    if max_symbols_per_frame < 1:
        raise ValueError("max_symbols_per_frame must be at least 1")

    frames = encoder_frames[0]
    valid_frames = len(frames)
    if attention_mask is not None:
        valid_lengths = valid_frame_lengths_from_attention_mask(attention_mask)
        if len(valid_lengths) != 1:
            raise ValueError("attention_mask must have batch size 1")
        valid_frames = valid_lengths[0]
        if valid_frames > len(frames):
            raise ValueError("attention_mask has more valid frames than encoder_frames")

    emitted: list[int] = []
    frame_index = 0
    symbols_on_frame = 0
    previous_token = blank_token_id
    cache: object | None = None

    while frame_index < valid_frames:
        token, candidate_cache = adapter.predict(
            frames[frame_index],
            previous_token,
            cache,
        )
        is_blank = token == blank_token_id
        cache = candidate_cache
        previous_token = token

        if not is_blank:
            emitted.append(token)
            symbols_on_frame += 1

        if is_blank or symbols_on_frame >= max_symbols_per_frame:
            frame_index += 1
            symbols_on_frame = 0

    return emitted
