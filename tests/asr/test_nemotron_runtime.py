"""Longer live-runtime state, duration, and cancellation checks."""

from __future__ import annotations

import asyncio
import io
import os
import unicodedata
import wave
from collections.abc import AsyncIterator
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


def _word_error_rate(actual: str, reference: str) -> float:
    actual_words = _canonical_lexical(actual).split()
    reference_words = _canonical_lexical(reference).split()
    assert reference_words
    previous = list(range(len(actual_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current = [reference_index]
        for actual_index, actual_word in enumerate(actual_words, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[actual_index] + 1,
                    previous[actual_index - 1] + (actual_word != reference_word),
                )
            )
        previous = current
    return previous[-1] / len(reference_words)


def _assert_acceptable_transcript(actual: str, reference: str) -> None:
    assert _canonical_lexical(actual)
    assert _word_error_rate(actual, reference) <= 0.25


def _mono_pcm16_wav(duration_s: int) -> bytes:
    """Create deterministic, low-amplitude 16 kHz PCM without Python samples."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x01\x00" * (duration_s * 16_000))
    return output.getvalue()


class _CompletionAwareUpload(httpx.AsyncByteStream):
    """Signal only after the transport has consumed the final body chunk."""

    def __init__(
        self,
        body: bytes,
        upload_complete: asyncio.Event,
        *,
        chunk_size: int = 64 * 1024,
    ) -> None:
        self._body = body
        self._upload_complete = upload_complete
        self._chunk_size = chunk_size

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for offset in range(0, len(self._body), self._chunk_size):
            yield self._body[offset : offset + self._chunk_size]
        # httpcore requests the next chunk only after its send of the preceding
        # chunk completes, so reaching here proves the final chunk was consumed.
        self._upload_complete.set()

    async def aclose(self) -> None:
        return None


def _multipart_upload(
    filename: str,
    content: bytes,
    content_type: str,
    upload_complete: asyncio.Event,
) -> tuple[_CompletionAwareUpload, dict[str, str]]:
    boundary = "aquillm-nemotron-cancellation-boundary"
    opening = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="model"\r\n'
        "\r\n"
        f"{MODEL_ID}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n"
        "\r\n"
    ).encode("ascii")
    closing = f"\r\n--{boundary}--\r\n".encode("ascii")
    body = opening + content + closing
    return _CompletionAwareUpload(body, upload_complete), {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }


async def _transcribe_fixture(
    client: httpx.AsyncClient, utterance_id: str, *, language: str | None = None
) -> str:
    audio, _ = _fixture(utterance_id)
    data = {"model": MODEL_ID}
    if language is not None:
        data["language"] = language
    response = await client.post(
        _TRANSCRIPTIONS_URL,
        data=data,
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
        # The explicit locale must be request-local: every later call omits the
        # field and therefore exercises automatic language selection.
        actual = [await _transcribe_fixture(client, first_id, language="en")]
        actual.extend(
            [
                await _transcribe_fixture(client, utterance_id)
                for utterance_id in (second_id, first_id, first_id)
            ]
        )

    normalized = [_canonical_lexical(text) for text in actual]
    assert normalized[0] == normalized[2] == normalized[3]
    assert normalized[0] != normalized[1]
    for text in (actual[0], actual[2], actual[3]):
        _assert_acceptable_transcript(text, first_expected)
    _assert_acceptable_transcript(actual[1], second_expected)


async def test_cancelled_request_does_not_poison_a_fresh_client_request() -> None:
    upload_complete = asyncio.Event()
    upload, upload_headers = _multipart_upload(
        "cancel.wav",
        _mono_pcm16_wav(390),
        "audio/wav",
        upload_complete,
    )

    async def consume_entire_stream(client: httpx.AsyncClient) -> None:
        async with client.stream(
            "POST",
            _TRANSCRIPTIONS_URL,
            content=upload,
            headers=upload_headers,
        ) as response:
            await response.aread()

    async with httpx.AsyncClient(
        headers=_AUTH,
        timeout=900.0,
    ) as cancelling_client:
        request_task = asyncio.create_task(consume_entire_stream(cancelling_client))
        await asyncio.wait_for(upload_complete.wait(), timeout=30.0)
        assert not request_task.done()
        request_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request_task

    follow_up_id = _UTTERANCE_IDS[1]
    _, expected = _fixture(follow_up_id)
    async with httpx.AsyncClient(headers=_AUTH, timeout=420.0) as fresh_client:
        actual = await _transcribe_fixture(fresh_client, follow_up_id)

    _assert_acceptable_transcript(actual, expected)
