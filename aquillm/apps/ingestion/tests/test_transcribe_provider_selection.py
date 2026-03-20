"""Tests for media transcription provider selection and configuration."""

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
