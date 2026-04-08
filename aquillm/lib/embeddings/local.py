"""
Local OpenAI-compatible embedding provider.
"""

import structlog
from typing import Any

from openai import OpenAI

from .config import (
    get_local_embed_config,
    get_target_dims,
    allow_embed_dimensions_override,
    max_embed_input_chars,
    is_context_limit_error,
    extract_context_limit_tokens,
    _env_int,
)

logger = structlog.stdlib.get_logger(__name__)

_LOCAL_OPENAI_CLIENT: OpenAI | None = None
_LOCAL_OPENAI_CLIENT_CFG: tuple[str, str] | None = None


def _get_local_openai_client(base_url: str, api_key: str) -> OpenAI:
    """Get or create a cached OpenAI client for local embedding."""
    global _LOCAL_OPENAI_CLIENT, _LOCAL_OPENAI_CLIENT_CFG
    cfg = (base_url, api_key)
    if _LOCAL_OPENAI_CLIENT is None or _LOCAL_OPENAI_CLIENT_CFG != cfg:
        _LOCAL_OPENAI_CLIENT = OpenAI(base_url=base_url, api_key=api_key)
        _LOCAL_OPENAI_CLIENT_CFG = cfg
    return _LOCAL_OPENAI_CLIENT


def _shrink_text_for_retry(text: str) -> str:
    """Shrink text for context limit retry."""
    if len(text) <= 128:
        return text
    next_len = max(128, int(len(text) * 0.8))
    if next_len >= len(text):
        next_len = len(text) - 1
    return text[:next_len]


def _dims_kwargs() -> dict:
    """Return dimensions kwarg for OpenAI API if APP_EMBED_DIMS is set."""
    if not allow_embed_dimensions_override():
        return {}
    dims = get_target_dims()
    return {"dimensions": dims} if dims else {}


def _embed_local_with_context_retry(client: OpenAI, model: str, query: Any) -> list[float]:
    """Embed with automatic retry on context limit errors."""
    dims_kw = _dims_kwargs()
    if not isinstance(query, str):
        response = client.embeddings.create(
            model=model,
            input=query,
            **dims_kw,
        )
        return response.data[0].embedding

    max_retries = _env_int("APP_EMBED_CONTEXT_RETRIES", 6)
    candidate = query
    char_cap = max_embed_input_chars()
    if char_cap > 0 and len(candidate) > char_cap:
        candidate = candidate[:char_cap]

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.embeddings.create(
                model=model,
                input=candidate,
                **dims_kw,
            )
            return response.data[0].embedding
        except Exception as exc:
            last_exc = exc
            if not is_context_limit_error(exc):
                raise
            limit_tokens = extract_context_limit_tokens(exc)
            if limit_tokens:
                reserve = _env_int("APP_EMBED_TOKEN_RESERVE", 16)
                token_based_cap = max(128, limit_tokens - reserve)
                next_candidate = candidate[:token_based_cap] if len(candidate) > token_based_cap else _shrink_text_for_retry(candidate)
            else:
                next_candidate = _shrink_text_for_retry(candidate)
            if next_candidate == candidate:
                break
            token_note = f" (model limit={limit_tokens} tokens)" if limit_tokens else ""
            logger.warning(
                "obs.embed.local_error",
                chars_before=len(candidate),
                chars_after=len(next_candidate),
                attempt=attempt,
                max_retries=max_retries,
                token_note=token_note,
            )
            candidate = next_candidate
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Local embedding failed without an exception detail.")


def get_embedding_via_local_openai(query: Any) -> list[float]:
    """Get embedding via local OpenAI-compatible endpoint."""
    base_url, api_key, model = get_local_embed_config()
    client = _get_local_openai_client(base_url, api_key)
    return _embed_local_with_context_retry(client, model, query)


def get_embeddings_via_local_openai(queries: list[Any]) -> list[list[float]]:
    """Get batch embeddings via local OpenAI-compatible endpoint."""
    if not queries:
        return []
    base_url, api_key, model = get_local_embed_config()
    client = _get_local_openai_client(base_url, api_key)
    char_cap = max_embed_input_chars()
    prepared_queries = [
        (query[:char_cap] if char_cap > 0 and isinstance(query, str) else query)
        for query in queries
    ]
    dims_kw = _dims_kwargs()
    try:
        response = client.embeddings.create(
            model=model,
            input=prepared_queries,
            **dims_kw,
        )
        return [item.embedding for item in response.data]
    except Exception as exc:
        if not is_context_limit_error(exc):
            raise
        logger.warning(
            "obs.embed.local_dim_error",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return [_embed_local_with_context_retry(client, model, query) for query in queries]


__all__ = [
    'get_embedding_via_local_openai',
    'get_embeddings_via_local_openai',
]
