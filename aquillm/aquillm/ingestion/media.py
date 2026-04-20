import io
import structlog
from os import getenv

from openai import BadRequestError, OpenAI

logger = structlog.stdlib.get_logger(__name__)


def _provider() -> str:
    return (getenv("INGEST_TRANSCRIBE_PROVIDER") or "").strip().lower()


def _openai_client() -> OpenAI:
    base_url = (
        getenv("INGEST_TRANSCRIBE_OPENAI_BASE_URL")
        or getenv("OPENAI_BASE_URL")
        or "http://vllm_transcribe:8000/v1"
    ).strip()
    api_key = (getenv("INGEST_TRANSCRIBE_OPENAI_API_KEY") or getenv("OPENAI_API_KEY") or "").strip()
    if base_url:
        return OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    return OpenAI(api_key=api_key)


def transcribe_media_bytes(data: bytes, filename: str) -> str:
    provider = _provider()
    if provider not in ("openai",):
        raise RuntimeError(
            "No supported transcription provider configured. "
            "Set INGEST_TRANSCRIBE_PROVIDER=openai and corresponding API/base URL env vars."
        )

    model = (getenv("INGEST_TRANSCRIBE_MODEL") or "gpt-4o-mini-transcribe").strip()
    client = _openai_client()
    file_obj = io.BytesIO(data)
    # OpenAI client expects a name to infer media type.
    file_obj.name = filename  # type: ignore[attr-defined]
    try:
        response = client.audio.transcriptions.create(model=model, file=file_obj)
    except BadRequestError as exc:
        raise RuntimeError(
            "Transcription request failed. Verify that INGEST_TRANSCRIBE_OPENAI_BASE_URL points "
            "to a model endpoint that supports audio transcription and that INGEST_TRANSCRIBE_MODEL "
            "matches the served model name."
        ) from exc
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Transcription provider returned empty text.")
    return text.strip()
