from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aquillm_vllm_nemotron_asr.decoding import BLANK_TOKEN_ID
from aquillm_vllm_nemotron_asr.state import ReplayState


def test_real_prefill_atomically_replaces_every_previous_snapshot() -> None:
    state = ReplayState()
    state.replace_real([10, 11])
    state.replace_real([20])

    assert state.forced_ids([1, 2, 3]) == [20, BLANK_TOKEN_ID, BLANK_TOKEN_ID]


def test_positions_map_transcript_then_terminal_blank() -> None:
    state = ReplayState()
    state.replace_real([10, 11, 12])

    assert state.forced_ids([1, 2, 3, 4, 99]) == [
        10,
        11,
        12,
        BLANK_TOKEN_ID,
        BLANK_TOKEN_ID,
    ]


def test_empty_transcript_terminates_at_first_generated_position() -> None:
    state = ReplayState()
    state.replace_real([])

    assert state.forced_ids([1, 2]) == [BLANK_TOKEN_ID, BLANK_TOKEN_ID]


def test_first_and_last_transcript_tokens_are_not_skipped() -> None:
    state = ReplayState()
    state.replace_real([101, 202])

    assert state.forced_ids([1, 2]) == [101, 202]


def test_profiling_state_never_survives_a_real_prefill() -> None:
    state = ReplayState()
    state.replace_profiling([1, 2])
    state.replace_real([9])

    assert state.forced_ids([1, 2]) == [9, BLANK_TOKEN_ID]


def test_duplicate_audio_prefill_replaces_state() -> None:
    state = ReplayState()
    state.replace_real([31, 32])
    state.replace_real([41, 42, 43])

    assert state.forced_ids([1, 2, 3, 4]) == [41, 42, 43, BLANK_TOKEN_ID]


def test_cached_tensor_prefill_replaces_state() -> None:
    state = ReplayState()
    state.replace_real([51, 52])
    state.replace_real([61])

    assert state.forced_ids([1, 2]) == [61, BLANK_TOKEN_ID]


def test_reset_after_abort_prevents_stale_state_leakage() -> None:
    state = ReplayState()
    state.replace_real([71, 72])
    state.reset()

    with pytest.raises(RuntimeError, match="initialized"):
        state.forced_ids([1])

    state.replace_real([81])
    assert state.forced_ids([1, 2]) == [81, BLANK_TOKEN_ID]


def test_requested_positions_preserve_duplicates_order_and_gaps() -> None:
    state = ReplayState()
    state.replace_real([90, 91, 92])

    assert state.forced_ids([3, 1, 3, 5, 2]) == [
        92,
        90,
        92,
        BLANK_TOKEN_ID,
        91,
    ]


def test_repeated_forward_calls_are_idempotent() -> None:
    state = ReplayState()
    state.replace_real([100, 101])

    assert state.forced_ids([2, 1, 3]) == [101, 100, BLANK_TOKEN_ID]
    assert state.forced_ids([2, 1, 3]) == [101, 100, BLANK_TOKEN_ID]


def test_48750_tokens_plus_prompt_and_terminal_fit_within_50000_positions() -> None:
    state = ReplayState()
    transcript = list(range(48_750))
    state.replace_real(transcript)

    assert state.forced_ids([1, 48_750, 48_751]) == [
        transcript[0],
        transcript[-1],
        BLANK_TOKEN_ID,
    ]
    assert 1 + len(transcript) + 1 <= 50_000


def test_lookup_before_initialization_fails_clearly() -> None:
    with pytest.raises(RuntimeError, match="initialized"):
        ReplayState().forced_ids([1])


@pytest.mark.parametrize("position", [0, -1, True, 1.0])
def test_nonpositive_or_noninteger_positions_fail_clearly(position: object) -> None:
    state = ReplayState()
    state.replace_real([1])

    with pytest.raises((TypeError, ValueError), match="position"):
        state.forced_ids([position])  # type: ignore[list-item]


@pytest.mark.parametrize("tokens", [[True], [1.0], ["1"]])
def test_noninteger_or_boolean_token_ids_are_rejected(tokens: list[object]) -> None:
    with pytest.raises(TypeError, match="token"):
        ReplayState().replace_real(tokens)  # type: ignore[arg-type]
