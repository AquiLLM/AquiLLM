"""Shared helpers for the chat WebSocket consumer (env, text, UUID parsing, images)."""
from __future__ import annotations

import structlog
from os import getenv

from lib.llm.utils.images import resize_image_data_url_for_llm
from lib.tools.documents.ids import clean_and_parse_doc_id

logger = structlog.stdlib.get_logger(__name__)


def env_int(name: str, default: int) -> int:
    raw = getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default
    return value if value > 0 else default


CHAT_MAX_FUNC_CALLS = env_int("CHAT_MAX_FUNC_CALLS", 5)
CHAT_MAX_TOKENS = env_int("CHAT_MAX_TOKENS", 2048)
TOOL_CHUNK_CHAR_LIMIT = env_int("TOOL_CHUNK_CHAR_LIMIT", 1500)
MAX_IMAGES_PER_TOOL_RESULT = env_int("MAX_IMAGES_PER_TOOL_RESULT", 1)
LLM_IMAGE_MAX_DIMENSION = env_int("LLM_IMAGE_MAX_DIMENSION", 384)
LLM_IMAGE_MAX_BYTES = env_int("LLM_IMAGE_MAX_BYTES", 50_000)


def truncate_tool_text(text: str) -> str:
    if len(text) <= TOOL_CHUNK_CHAR_LIMIT:
        return text
    return text[:TOOL_CHUNK_CHAR_LIMIT] + "\n...[truncated for context window]..."


def resize_image_for_llm_context(
    image_data_url: str,
    max_dimension: int | None = None,
    max_bytes: int | None = None,
) -> str | None:
    return resize_image_data_url_for_llm(
        image_data_url,
        max_dimension=max_dimension or LLM_IMAGE_MAX_DIMENSION,
        max_bytes=max_bytes or LLM_IMAGE_MAX_BYTES,
    )


__all__ = [
    "CHAT_MAX_FUNC_CALLS",
    "CHAT_MAX_TOKENS",
    "TOOL_CHUNK_CHAR_LIMIT",
    "MAX_IMAGES_PER_TOOL_RESULT",
    "LLM_IMAGE_MAX_DIMENSION",
    "LLM_IMAGE_MAX_BYTES",
    "clean_and_parse_doc_id",
    "env_int",
    "resize_image_for_llm_context",
    "truncate_tool_text",
]
