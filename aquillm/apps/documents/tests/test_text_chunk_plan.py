from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from apps.documents.services.text_chunk_plan import TextChunkSpec, plan_text_chunks


CHUNK_SIZE = 2048
OVERLAP = 384
PITCH = CHUNK_SIZE - OVERLAP


def historical_slicing_oracle(text: str) -> list[tuple[str, int, int, int]]:
    last_character = len(text) - 1
    return [
        (
            text[PITCH * index : min((PITCH * index) + CHUNK_SIZE, last_character + 1)],
            PITCH * index,
            min((PITCH * index) + CHUNK_SIZE, last_character + 1),
            index,
        )
        for index in range(last_character // PITCH + 1)
    ]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (2048, -1), (2048, 2048), (2048, 2049)],
)
def test_plan_text_chunks_rejects_invalid_configuration(chunk_size, overlap):
    with pytest.raises(ValueError):
        plan_text_chunks("text", chunk_size=chunk_size, overlap=overlap)


@pytest.mark.parametrize("text_length", [0, 1, 1663, 1664, 2047, 2048, 2049, 10000])
def test_plan_text_chunks_matches_historical_slicing(text_length):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    text = "".join(alphabet[index % len(alphabet)] for index in range(text_length))

    specs = plan_text_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    actual = [
        (spec.content, spec.start_position, spec.end_position, spec.chunk_number)
        for spec in specs
    ]

    assert actual == historical_slicing_oracle(text)
    assert [spec.chunk_number for spec in specs] == list(range(len(specs)))
    if specs:
        assert specs[-1].end_position == text_length
        covered_positions = {
            position
            for spec in specs
            for position in range(spec.start_position, spec.end_position)
        }
        assert covered_positions == set(range(text_length))
    else:
        assert text == ""


def test_text_chunk_spec_is_immutable():
    spec = TextChunkSpec(content="abc", start_position=0, end_position=3, chunk_number=0)

    with pytest.raises(FrozenInstanceError):
        spec.content = "changed"
