"""
Embedding system configuration from environment variables.
"""

import structlog
import re
from os import getenv
from typing import Any

logger = structlog.stdlib.get_logger(__name__)

_CONTEXT_LIMIT_RE = re.compile(
    r"maximum input length of\s*(\d+)\s*tokens|context length is only\s*(\d+)\s*tokens",
    flags=re.IGNORECASE,
)


def _env_int(name: str, default: int) -> int:
    """Parse integer from environment with validation."""
    raw = (getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return value if value > 0 else default


def get_local_embed_config() -> tuple[str, str, str]:
    """Get local embedding configuration: (base_url, api_key, model)."""
    base_url = (
        getenv("APP_EMBED_BASE_URL")
        or getenv("MEM0_EMBED_BASE_URL")
        or getenv("VLLM_BASE_URL")
        or "http://vllm:8000/v1"
    ).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    api_key = (
        getenv("APP_EMBED_API_KEY")
        or getenv("MEM0_EMBED_API_KEY")
        or getenv("VLLM_API_KEY")
        or "EMPTY"
    )
    model = (
        getenv("APP_EMBED_MODEL")
        or getenv("MEM0_EMBED_MODEL")
        or "Qwen/Qwen3-Embedding-4B"
    )
    return base_url, api_key, model


def get_target_dims() -> int:
    """Get target embedding dimensions from config."""
    target_raw = getenv("APP_EMBED_DIMS", "1024").strip()
    try:
        target_dims = int(target_raw)
    except Exception:
        target_dims = 1024
    if target_dims <= 0:
        target_dims = 1024
    return target_dims


def max_embed_input_chars() -> int:
    """Get optional explicit hard cap for local embedding inputs."""
    explicit = _env_int("APP_EMBED_MAX_INPUT_CHARS", 0)
    return explicit if explicit > 0 else 0


def is_context_limit_error(exc: Exception) -> bool:
    """Check if an exception indicates a context length limit error."""
    message = str(exc).lower()
    return (
        ("input_tokens" in message and "context length" in message)
        or "maximum input length" in message
    )


def extract_context_limit_tokens(exc: Exception) -> int | None:
    """Extract token limit from context limit error message."""
    match = _CONTEXT_LIMIT_RE.search(str(exc))
    if not match:
        return None
    for group in match.groups():
        if group:
            try:
                return int(group)
            except Exception:
                return None
    return None


__all__ = [
    'get_local_embed_config',
    'get_target_dims',
    'max_embed_input_chars',
    'is_context_limit_error',
    'extract_context_limit_tokens',
    '_env_int',
]
