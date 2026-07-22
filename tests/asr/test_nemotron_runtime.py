"""Longer live-runtime state, duration, and cancellation checks."""

from __future__ import annotations

import asyncio
import io
import os
import unicodedata
import wave
from pathlib import Path

import httpx
import pytest

_BASE_URL = os.environ.get("ASR_BASE_URL", "").strip()
_RUNTIME_ENABLED = os.environ.get("RUN_ASR_RUNTIME") == "1" and bool(_BASE_URL)

pytestmark = [
    pytest.mark.asr_runtime,
    pytest.mark.skipif(
        not _RUNTIME_ENABLED,
        reason="set RUN_ASR_RUNTIME=1 and a nonblank ASR_BASE_URL",
    ),
]

MODEL_ID = "nemotron-3.5-asr-streaming-0.6b"
_API_ROOT = _BASE_URL.rstrip("/")
_OPENAI_BASE_URL = _API_ROOT if _API_ROOT.endswith("/v1") else f"{_API_ROOT}/v1"
_TRANSCRIPTIONS_URL = f"{_OPENAI_BASE_URL}/audio/transcriptions"
_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "audio"
_UTTERANCE_IDS = ("1272-128104-0000", "1272-128104-0001")
_AUTH = {"Authorization": "Bearer EMPTY"}


def _fixture(utterance_id: str) -> tuple[bytes, str]:
    stem = _FIXTURE_ROOT / f"librispeech_{utterance_id}"
    return stem.with_suffix(".flac").read_bytes(), stem.with_suffix(
        ".txt"
    ).read_text().strip()


def _canonical_lexical(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "ʼ": "'"}))
    lexical = "".join(
        character
        if character.isalnum() or character.isspace() or character == "'"
        else " "
        for character in text
    )
    return " ".join(lexical.split())


def _mono_pcm16_wav(duration_s: int) -> bytes:
    """Create deterministic, low-amplitude 16 kHz PCM without Python samples."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x01\x00" * (duration_s * 16_000))
    return output.getvalue()


async def _transcribe_fixture(client: httpx.AsyncClient, utterance_id: str) -> str:
    audio, _ = _fixture(utterance_id)
    response = await client.post(
        _TRANSCRIPTIONS_URL,
        data={"model": MODEL_ID},
        files={"file": (f"{utterance_id}.flac", audio, "audio/flac")},
    )
    assert response.status_code == 200, response.text
    return response.json()["text"]


@pytest.mark.parametrize(
    ("duration_s", "expected_status"), [(389, 200), (390, 200), (391, 400)]
)
async def test_audio_duration_boundary_is_not_truncated(
    duration_s: int, expected_status: int
) -> None:
    wav_bytes = _mono_pcm16_wav(duration_s)
    async with httpx.AsyncClient(headers=_AUTH, timeout=900.0) as client:
        response = await client.post(
            _TRANSCRIPTIONS_URL,
            data={"model": MODEL_ID},
            files={"file": (f"boundary-{duration_s}.wav", wav_bytes, "audio/wav")},
        )

    assert response.status_code == expected_status, response.text
    if expected_status == 400:
        assert response.json()["error"]["param"] == "duration_s"


async def test_sequential_and_duplicate_requests_rebuild_their_own_transcript() -> None:
    first_id, second_id = _UTTERANCE_IDS
    _, first_expected = _fixture(first_id)
    _, second_expected = _fixture(second_id)

    async with httpx.AsyncClient(headers=_AUTH, timeout=420.0) as client:
        actual = [
            await _transcribe_fixture(client, utterance_id)
            for utterance_id in (first_id, second_id, first_id, first_id)
        ]

    expected_first = _canonical_lexical(first_expected)
    expected_second = _canonical_lexical(second_expected)
    assert [_canonical_lexical(text) for text in actual] == [
        expected_first,
        expected_second,
        expected_first,
        expected_first,
    ]
    assert expected_first != expected_second


async def test_cancelled_request_does_not_poison_a_fresh_client_request() -> None:
    request_started = asyncio.Event()

    async def signal_request_start(request: httpx.Request) -> None:
        request_started.set()

    async def consume_entire_stream(client: httpx.AsyncClient) -> None:
        async with client.stream(
            "POST",
            _TRANSCRIPTIONS_URL,
            data={"model": MODEL_ID},
            files={
                "file": (
                    "cancel.wav",
                    _mono_pcm16_wav(390),
                    "audio/wav",
                )
            },
        ) as response:
            await response.aread()

    async with httpx.AsyncClient(
        headers=_AUTH,
        timeout=900.0,
        event_hooks={"request": [signal_request_start]},
    ) as cancelling_client:
        request_task = asyncio.create_task(consume_entire_stream(cancelling_client))
        await asyncio.wait_for(request_started.wait(), timeout=10.0)
        await asyncio.sleep(0.05)
        assert not request_task.done()
        request_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request_task

    follow_up_id = _UTTERANCE_IDS[1]
    _, expected = _fixture(follow_up_id)
    async with httpx.AsyncClient(headers=_AUTH, timeout=420.0) as fresh_client:
        actual = await _transcribe_fixture(fresh_client, follow_up_id)

    assert _canonical_lexical(actual) == _canonical_lexical(expected)
