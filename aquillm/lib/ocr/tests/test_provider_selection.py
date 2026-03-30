"""Tests for OCR provider selection and configuration."""

from types import SimpleNamespace

from lib.ocr import (
    extract_text_from_image,
    extract_text_with_tesseract,
    extract_text_with_qwen,
    extract_text_with_gemini,
)
from lib.ocr.config import get_qwen_config


def test_qwen_provider_defaults_to_dedicated_service(monkeypatch):
    monkeypatch.delenv("APP_OCR_QWEN_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("APP_OCR_QWEN_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    base_url, api_key, _model, _timeout = get_qwen_config()

    assert base_url == "http://vllm_ocr:8000/v1"
    assert api_key == "EMPTY"


def test_local_provider_does_not_call_gemini(monkeypatch):
    monkeypatch.setenv("APP_OCR_PROVIDER", "local")
    monkeypatch.setattr(
        "lib.ocr.extract_text_with_tesseract",
        lambda *_args, **_kwargs: {"extracted_text": "local text"},
    )
    monkeypatch.setattr(
        "lib.ocr.extract_text_with_gemini",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("gemini should not run")),
    )

    result = extract_text_from_image(b"fake-bytes")

    assert result["extracted_text"] == "local text"


def test_auto_provider_falls_back_to_gemini(monkeypatch):
    monkeypatch.setenv("APP_OCR_PROVIDER", "auto")
    monkeypatch.setattr(
        "lib.ocr.extract_text_with_qwen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("qwen failed")),
    )
    monkeypatch.setattr(
        "lib.ocr.extract_text_with_tesseract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("local failed")),
    )
    monkeypatch.setattr(
        "lib.ocr.extract_text_with_gemini",
        lambda *_args, **_kwargs: {"extracted_text": "gemini text"},
    )

    result = extract_text_from_image(b"fake-bytes")

    assert result["extracted_text"] == "gemini text"


def test_qwen_provider_uses_qwen_only(monkeypatch):
    monkeypatch.setenv("APP_OCR_PROVIDER", "qwen")
    monkeypatch.setattr(
        "lib.ocr.extract_text_with_qwen",
        lambda *_args, **_kwargs: {"extracted_text": "qwen text"},
    )
    monkeypatch.setattr(
        "lib.ocr.extract_text_with_tesseract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("local should not run")),
    )
    monkeypatch.setattr(
        "lib.ocr.extract_text_with_gemini",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("gemini should not run")),
    )

    result = extract_text_from_image(b"fake-bytes")

    assert result["extracted_text"] == "qwen text"


def test_auto_provider_falls_back_to_local_when_qwen_fails(monkeypatch):
    monkeypatch.setenv("APP_OCR_PROVIDER", "auto")
    monkeypatch.setattr(
        "lib.ocr.extract_text_with_qwen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("qwen failed")),
    )
    monkeypatch.setattr(
        "lib.ocr.extract_text_with_tesseract",
        lambda *_args, **_kwargs: {"extracted_text": "local text"},
    )
    monkeypatch.setattr(
        "lib.ocr.extract_text_with_gemini",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("gemini should not run")),
    )

    result = extract_text_from_image(b"fake-bytes")

    assert result["extracted_text"] == "local text"


def test_local_provider_returns_clear_error(monkeypatch):
    monkeypatch.setenv("APP_OCR_PROVIDER", "local")
    monkeypatch.setattr(
        "lib.ocr.extract_text_with_tesseract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("install tesseract")),
    )

    try:
        extract_text_from_image(b"fake-bytes")
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "OCR processing failed: install tesseract" in str(exc)


def test_qwen3_5_ocr_disables_thinking(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="plain transcription"))]
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("APP_OCR_QWEN_MODEL", "qwen3.5-ocr")
    monkeypatch.setenv("OCR_VLLM_MODEL", "Qwen/Qwen3.5-4B")
    monkeypatch.setattr("lib.ocr.qwen.OpenAI", FakeOpenAI)
    monkeypatch.setattr("lib.ocr.qwen.resize_image_for_ocr", lambda content: content)
    monkeypatch.setattr("lib.ocr.qwen.get_image_mime_type", lambda _content: "image/png")

    result = extract_text_with_qwen(b"fake-image")

    assert result["extracted_text"] == "plain transcription"
    assert captured["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
