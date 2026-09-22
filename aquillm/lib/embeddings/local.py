"""
Local OpenAI-compatible embedding provider.
"""

from typing import Any

import structlog
from openai import OpenAI

from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields

from .config import (
    _env_int,
    allow_embed_dimensions_override,
    extract_context_limit_tokens,
    get_local_embed_config,
    get_target_dims,
    is_context_limit_error,
    max_embed_input_chars,
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


def _embed_local_with_context_retry(
    client: OpenAI, model: str, query: Any
) -> list[float]:
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
                next_candidate = (
                    candidate[:token_based_cap]
                    if len(candidate) > token_based_cap
                    else _shrink_text_for_retry(candidate)
                )
            else:
                next_candidate = _shrink_text_for_retry(candidate)
            if next_candidate == candidate:
                break
            logger.warning(
                "obs.embed.input_retry_truncate",
                **retrieval_log_fields(
                    reason=RetrievalLogReason.INVALID_REQUEST,
                    count=0,
                    elapsed_ms=0.0,
                ),
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
            "obs.embed.batch_context_retry",
            **retrieval_log_fields(
                reason=RetrievalLogReason.INVALID_REQUEST,
                count=0,
                elapsed_ms=0.0,
            ),
        )
        return [
            _embed_local_with_context_retry(client, model, query) for query in queries
        ]


def get_strict_indexed_embeddings_via_local_openai(
    queries: list[str],
) -> list[tuple[int, list[float]]]:
    """Embed an exact durable batch once and preserve provider indices.

    Unlike the availability-oriented helpers above, this seam never truncates,
    retries with transformed text, or falls back to another provider.
    """

    if type(queries) is not list or any(type(query) is not str for query in queries):
        raise ValueError("strict embedding inputs must be an exact list of strings")
    if not queries:
        return []
    base_url, api_key, model = get_local_embed_config()
    client = _get_local_openai_client(base_url, api_key)
    response = client.embeddings.create(
        model=model,
        input=queries,
        dimensions=1024,
    )
    response_model = getattr(response, "model", None)
    if type(response_model) is not str or response_model != model:
        raise RuntimeError(
            "Local embedding response model identity differs from configured model"
        )
    data = response.data
    if not isinstance(data, (list, tuple)):
        raise RuntimeError("Local embedding endpoint returned invalid indexed data")
    indexed: list[tuple[int, list[float]]] = []
    for item in data:
        index = getattr(item, "index", None)
        vector = getattr(item, "embedding", None)
        if type(index) is not int or not isinstance(vector, (list, tuple)):
            raise RuntimeError("Local embedding response lacks index/vector binding")
        indexed.append((index, list(vector)))
    return indexed


__all__ = [
    "get_embedding_via_local_openai",
    "get_embeddings_via_local_openai",
    "get_strict_indexed_embeddings_via_local_openai",
]
