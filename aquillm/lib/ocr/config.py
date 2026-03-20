"""
OCR configuration from environment variables.
"""

import os
from typing import Callable, Optional


def get_ocr_provider() -> str:
    """Get configured OCR provider."""
    provider = (os.getenv("APP_OCR_PROVIDER") or "auto").strip().lower()
    if provider not in {"auto", "qwen", "local", "gemini"}:
        return "auto"
    return provider


def get_qwen_config() -> tuple[str, str, str, int]:
    """Get Qwen OCR configuration: (base_url, api_key, model, timeout_seconds)."""
    base_url = (
        os.getenv("APP_OCR_QWEN_BASE_URL")
        or os.getenv("VLLM_BASE_URL")
        or "http://vllm_ocr:8000/v1"
    ).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    api_key = os.getenv("APP_OCR_QWEN_API_KEY") or os.getenv("VLLM_API_KEY") or "EMPTY"
    model = (
        os.getenv("APP_OCR_QWEN_MODEL")
        or os.getenv("VLLM_SERVED_MODEL_NAME")
        or os.getenv("VLLM_MODEL")
        or "qwen3.5:27b-q8_0"
    )
    timeout_raw = (os.getenv("APP_OCR_QWEN_TIMEOUT_SECONDS") or "120").strip()
    try:
        timeout_seconds = int(timeout_raw)
    except Exception:
        timeout_seconds = 120
    if timeout_seconds <= 0:
        timeout_seconds = 120
    return base_url, api_key, model, timeout_seconds


def get_gemini_api_key() -> Optional[str]:
    """Get Gemini API key if configured."""
    return os.getenv("GEMINI_API_KEY")


# Optional usage tracker callback for external integration
_usage_logger: Optional[Callable[[str, int, int], None]] = None


def set_usage_logger(logger: Optional[Callable[[str, int, int], None]]) -> None:
    """Set an external usage logger callback for API cost tracking."""
    global _usage_logger
    _usage_logger = logger


def log_usage(operation_type: str, input_tokens: int, output_tokens: int) -> None:
    """Log API usage if a logger is configured."""
    if _usage_logger is not None:
        _usage_logger(operation_type, input_tokens, output_tokens)


__all__ = [
    'get_ocr_provider',
    'get_qwen_config',
    'get_gemini_api_key',
    'set_usage_logger',
    'log_usage',
]
