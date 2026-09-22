"""Validated local vLLM settings and bounded generation limits."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

_DEFAULT_MAX_CHUNKS = 32
_DEFAULT_MAX_CHARACTERS = 48_000
_DEFAULT_TIMEOUT_SECONDS = 180


class SchemaGenerationConfigurationError(ValueError):
    """The local-only generation configuration is unsafe or malformed."""


@dataclass(frozen=True, slots=True)
class SchemaGenerationConfig:
    base_url: str
    api_key: str
    model: str
    max_chunks: int
    max_characters: int
    timeout_seconds: int


def _positive_env_int(name: str, default: int) -> int:
    value = (os.environ.get(name) or "").strip()
    if not value:
        return default
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise SchemaGenerationConfigurationError(
            f"{name} must be a positive integer"
        ) from exc
    if parsed <= 0:
        raise SchemaGenerationConfigurationError(f"{name} must be a positive integer")
    return parsed


def _enabled_from_environment() -> bool:
    return (os.environ.get("KG_SCHEMA_GENERATION_ENABLED") or "0").strip() == "1"


def load_schema_generation_config() -> SchemaGenerationConfig:
    """Load bounded settings and reject every endpoint outside the vLLM service."""

    raw_url = (os.environ.get("VLLM_BASE_URL") or "http://vllm:8000/v1").strip()
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SchemaGenerationConfigurationError("VLLM_BASE_URL must be an HTTP(S) URL")
    if parsed.hostname.lower() != "vllm":
        raise SchemaGenerationConfigurationError(
            "VLLM_BASE_URL host must equal the configured Docker service host"
        )
    normalized_path = parsed.path.rstrip("/")
    if not normalized_path:
        normalized_path = "/v1"
    elif normalized_path != "/v1":
        raise SchemaGenerationConfigurationError(
            "VLLM_BASE_URL must use the /v1 API path"
        )
    base_url = f"{parsed.scheme}://{parsed.netloc}{normalized_path}"
    model = (os.environ.get("VLLM_SERVED_MODEL_NAME") or "qwen3.5:27b").strip()
    return SchemaGenerationConfig(
        base_url=base_url,
        api_key=(os.environ.get("VLLM_API_KEY") or "EMPTY").strip() or "EMPTY",
        model=model,
        max_chunks=min(
            _positive_env_int("KG_SCHEMA_GENERATION_MAX_CHUNKS", _DEFAULT_MAX_CHUNKS),
            _DEFAULT_MAX_CHUNKS,
        ),
        max_characters=min(
            _positive_env_int(
                "KG_SCHEMA_GENERATION_MAX_CHARACTERS", _DEFAULT_MAX_CHARACTERS
            ),
            _DEFAULT_MAX_CHARACTERS,
        ),
        timeout_seconds=_positive_env_int(
            "KG_SCHEMA_GENERATION_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS
        ),
    )
