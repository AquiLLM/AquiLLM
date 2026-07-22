"""OpenAI-compatible HTTP contract for the live Nemotron ASR service."""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

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
_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "audio"
_UTTERANCE_IDS = ("1272-128104-0000", "1272-128104-0001")


def _fixture(utterance_id: str) -> tuple[Path, str]:
    stem = _FIXTURE_ROOT / f"librispeech_{utterance_id}"
    return stem.with_suffix(".flac"), stem.with_suffix(".txt").read_text().strip()


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


@pytest.fixture(scope="module")
def openai_client():
    with OpenAI(
        base_url=_OPENAI_BASE_URL,
        api_key="EMPTY",
        max_retries=0,
        timeout=420.0,
    ) as client:
        yield client


@pytest.fixture(scope="module")
def raw_client():
    with httpx.Client(
        headers={"Authorization": "Bearer EMPTY"},
        timeout=420.0,
    ) as client:
        yield client


def _sdk_transcribe(
    client: OpenAI, utterance_id: str, *, language: str | None = None
) -> str:
    audio_path, _ = _fixture(utterance_id)
    options: dict[str, object] = {
        "model": MODEL_ID,
        "file": (audio_path.name, audio_path.read_bytes(), "audio/flac"),
    }
    if language is not None:
        options["language"] = language
    return client.audio.transcriptions.create(**options).text  # type: ignore[arg-type]


def _raw_transcription(
    client: httpx.Client,
    *,
    fields: dict[str, str] | None = None,
    utterance_id: str = _UTTERANCE_IDS[0],
) -> httpx.Response:
    audio_path, _ = _fixture(utterance_id)
    data = {"model": MODEL_ID, **(fields or {})}
    return client.post(
        f"{_OPENAI_BASE_URL}/audio/transcriptions",
        data=data,
        files={"file": (audio_path.name, audio_path.read_bytes(), "audio/flac")},
    )


def _assert_error_param(response: httpx.Response, expected: str) -> None:
    assert response.status_code == 400, response.text
    assert response.headers["content-type"].split(";", 1)[0] == "application/json"
    assert response.json()["error"]["param"] == expected


def test_models_exposes_only_the_configured_nemotron_alias(
    openai_client: OpenAI,
) -> None:
    models = openai_client.models.list()

    assert [model.id for model in models.data] == [MODEL_ID]


@pytest.mark.parametrize("utterance_id", _UTTERANCE_IDS)
def test_sdk_default_response_has_text_and_omits_language(
    openai_client: OpenAI, utterance_id: str
) -> None:
    _, expected = _fixture(utterance_id)

    actual = _sdk_transcribe(openai_client, utterance_id)

    assert isinstance(actual, str)
    assert _canonical_lexical(actual) == _canonical_lexical(expected)


@pytest.mark.parametrize("utterance_id", _UTTERANCE_IDS)
def test_sdk_explicit_english_matches_each_fixture(
    openai_client: OpenAI, utterance_id: str
) -> None:
    _, expected = _fixture(utterance_id)

    actual = _sdk_transcribe(openai_client, utterance_id, language="en")

    assert _canonical_lexical(actual) == _canonical_lexical(expected)


def test_raw_text_response_is_plain_text_without_a_json_wrapper(
    raw_client: httpx.Client,
) -> None:
    _, expected = _fixture(_UTTERANCE_IDS[0])

    response = _raw_transcription(raw_client, fields={"response_format": "text"})

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].split(";", 1)[0] == "text/plain"
    assert _canonical_lexical(response.text) == _canonical_lexical(expected)


@pytest.mark.parametrize(
    ("fields", "error_param"),
    [
        ({"prompt": "MISTER"}, "prompt"),
        ({"temperature": "0.5"}, "temperature"),
        ({"response_format": "verbose_json"}, "response_format"),
        ({"response_format": "srt"}, "response_format"),
        ({"response_format": "vtt"}, "response_format"),
        ({"timestamp_granularities[]": "word"}, "timestamp_granularities"),
        ({"timestamp_granularities[]": "segment"}, "timestamp_granularities"),
        ({"language": "not-a-language"}, "language"),
    ],
)
def test_sdk_surface_rejections_have_stable_error_parameters(
    raw_client: httpx.Client, fields: dict[str, str], error_param: str
) -> None:
    _assert_error_param(_raw_transcription(raw_client, fields=fields), error_param)


@pytest.mark.parametrize(
    ("fields", "error_param"),
    [
        ({"hotwords": "MISTER"}, "hotwords"),
        ({"use_beam_search": "true"}, "use_beam_search"),
        ({"n": "2"}, "n"),
        ({"stream": "true"}, "stream"),
        ({"stream_include_usage": "true"}, "stream_include_usage"),
        (
            {"stream_continuous_usage_stats": "true"},
            "stream_continuous_usage_stats",
        ),
        ({"to_language": "fr"}, "to_language"),
        ({"vllm_xargs": '{"test":true}'}, "vllm_xargs"),
        ({"max_completion_tokens": "8"}, "max_completion_tokens"),
    ],
)
def test_vllm_only_multipart_rejections_have_stable_error_parameters(
    raw_client: httpx.Client, fields: dict[str, str], error_param: str
) -> None:
    _assert_error_param(_raw_transcription(raw_client, fields=fields), error_param)


def test_translation_endpoint_rejects_the_task_type(raw_client: httpx.Client) -> None:
    audio_path, _ = _fixture(_UTTERANCE_IDS[0])
    response = raw_client.post(
        f"{_OPENAI_BASE_URL}/audio/translations",
        data={"model": MODEL_ID},
        files={"file": (audio_path.name, audio_path.read_bytes(), "audio/flac")},
    )

    _assert_error_param(response, "task_type")
