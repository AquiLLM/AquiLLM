from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aquillm_vllm_nemotron_asr.decoding import (
    BLANK_TOKEN_ID,
    MAX_SYMBOLS_PER_FRAME,
    greedy_rnnt_decode,
    valid_frame_lengths_from_attention_mask,
)


@dataclass
class ScriptedAdapter:
    """A deterministic stand-in for the model's decoder and joint network."""

    predictions: deque[tuple[int, object]]
    calls: list[tuple[object, int, object | None]] = field(default_factory=list)

    def predict(
        self,
        frame: object,
        previous_emitted_token: int,
        cache: object | None,
    ) -> tuple[int, object]:
        self.calls.append((frame, previous_emitted_token, cache))
        return self.predictions.popleft()


def scripted_adapter(*predictions: tuple[int, object]) -> ScriptedAdapter:
    return ScriptedAdapter(deque(predictions))


def test_blank_advances_a_frame_without_emitting_a_token() -> None:
    adapter = scripted_adapter(
        (BLANK_TOKEN_ID, "initial-blank-cache"),
        (BLANK_TOKEN_ID, "second-blank-cache"),
    )

    assert greedy_rnnt_decode([["frame-0", "frame-1"]], adapter) == []
    assert [call[0] for call in adapter.calls] == ["frame-0", "frame-1"]


def test_nonblank_stays_on_the_current_frame() -> None:
    adapter = scripted_adapter(
        (42, "after-initial-blank"),
        (BLANK_TOKEN_ID, "after-token-42"),
    )

    assert greedy_rnnt_decode([["frame-0"]], adapter) == [42]
    assert [call[0] for call in adapter.calls] == ["frame-0", "frame-0"]


def test_repeated_nonblank_ids_are_not_ctc_collapsed() -> None:
    adapter = scripted_adapter(
        (77, "after-initial-blank"),
        (77, "after-first-77"),
        (BLANK_TOKEN_ID, "after-second-77"),
    )

    assert greedy_rnnt_decode([["frame-0"]], adapter) == [77, 77]


def test_cache_tracks_the_current_decoder_input() -> None:
    adapter = scripted_adapter(
        # The decoder processes the initial blank and initializes cache state.
        (BLANK_TOKEN_ID, "initialized-by-initial-blank"),
        # A subsequent blank reuses the initialized cache while predicting a token.
        (11, "initialized-by-initial-blank"),
        # That token becomes the next decoder input and updates cache state.
        (BLANK_TOKEN_ID, "updated-by-token-11"),
    )

    assert greedy_rnnt_decode([["frame-0", "frame-1"]], adapter) == [11]
    assert adapter.calls == [
        ("frame-0", BLANK_TOKEN_ID, None),
        ("frame-1", BLANK_TOKEN_ID, "initialized-by-initial-blank"),
        ("frame-1", 11, "initialized-by-initial-blank"),
    ]


def test_tenth_nonblank_is_emitted_before_the_symbol_cap_advances() -> None:
    adapter = scripted_adapter(
        *[(token, f"cache-{token}") for token in range(1, MAX_SYMBOLS_PER_FRAME + 1)],
        (BLANK_TOKEN_ID, "frame-1-blank"),
    )

    assert greedy_rnnt_decode([["frame-0", "frame-1"]], adapter) == list(
        range(1, MAX_SYMBOLS_PER_FRAME + 1)
    )
    assert [call[0] for call in adapter.calls] == [
        "frame-0"
    ] * MAX_SYMBOLS_PER_FRAME + ["frame-1"]


def test_attention_mask_valid_lengths_exclude_padded_frames() -> None:
    attention_mask = [[1, 1, 0, 0]]
    assert valid_frame_lengths_from_attention_mask(attention_mask) == [2]

    adapter = scripted_adapter(
        (BLANK_TOKEN_ID, "first"),
        (BLANK_TOKEN_ID, "second"),
    )
    assert (
        greedy_rnnt_decode(
            [["frame-0", "frame-1", "padded-2", "padded-3"]],
            adapter,
            attention_mask=attention_mask,
        )
        == []
    )
    assert [call[0] for call in adapter.calls] == ["frame-0", "frame-1"]


def test_batch_size_other_than_one_fails_fast() -> None:
    adapter = scripted_adapter()

    with pytest.raises(ValueError, match="batch size 1"):
        greedy_rnnt_decode([["first"], ["second"]], adapter)


def test_empty_utterance_has_no_transcript_tokens() -> None:
    adapter = scripted_adapter()

    assert greedy_rnnt_decode([[]], adapter) == []
    assert adapter.calls == []
