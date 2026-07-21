"""Tests for media transcription provider selection and configuration."""

import pytest

from aquillm.ingestion import media


def test_openai_client_defaults_to_dedicated_transcribe_service(monkeypatch):
    monkeypatch.delenv("INGEST_TRANSCRIBE_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("INGEST_TRANSCRIBE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(media, "OpenAI", FakeOpenAI)

    media._openai_client()

    assert captured["base_url"] == "http://vllm_transcribe:8000/v1"
    assert captured["api_key"] == "EMPTY"


def test_transcribe_rejects_unconfigured_provider(monkeypatch):
    monkeypatch.delenv("INGEST_TRANSCRIBE_PROVIDER", raising=False)
    try:
        media.transcribe_media_bytes(b"audio-bytes", "sample.wav")
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as exc:
        assert "No supported transcription provider configured" in str(exc)


@pytest.mark.parametrize(
    ("configured_language", "expected_kwargs"),
    [
        (None, {"model": "asr-model"}),
        ("", {"model": "asr-model"}),
        ("  \t\n", {"model": "asr-model"}),
        ("  en-US  ", {"model": "asr-model", "language": "en-US"}),
    ],
)
def test_transcribe_passes_optional_configured_language(
    monkeypatch, configured_language, expected_kwargs
):
    monkeypatch.setenv("INGEST_TRANSCRIBE_PROVIDER", "openai")
    monkeypatch.setenv("INGEST_TRANSCRIBE_MODEL", "asr-model")
    if configured_language is None:
        monkeypatch.delenv("INGEST_TRANSCRIBE_LANGUAGE", raising=False)
    else:
        monkeypatch.setenv("INGEST_TRANSCRIBE_LANGUAGE", configured_language)

    captured = {}

    class FakeTranscriptions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type("Transcription", (), {"text": " hello "})()

    class FakeClient:
        audio = type("Audio", (), {"transcriptions": FakeTranscriptions()})()

    monkeypatch.setattr(media, "_openai_client", lambda: FakeClient())

    assert media.transcribe_media_bytes(b"audio-bytes", "sample.wav") == "hello"
    assert set(captured) == {"file", *expected_kwargs}
    assert {
        key: value for key, value in captured.items() if key != "file"
    } == expected_kwargs
    assert captured["file"].name == "sample.wav"
