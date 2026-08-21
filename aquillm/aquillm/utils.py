"""
Embedding utilities with Django app integration.

This module provides the main embedding interface for the application,
integrating lib/embeddings with Django app configuration for Cohere fallback.
"""

from hashlib import sha256
from math import isfinite
from os import getenv
from typing import Any
from urllib.parse import urlsplit

import structlog
from django.apps import apps

# Import from lib/embeddings for pure Python operations
from lib.embeddings import (
    fit_embedding_dims,
    get_embedding_via_cohere,
    get_embedding_via_local_openai,
    get_embeddings_via_cohere,
    get_embeddings_via_local_openai,
    get_multimodal_embedding_via_vllm_pooling,
    get_strict_indexed_embeddings_via_local_openai,
)
from lib.embeddings.config import get_local_embed_config, get_target_dims
from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields

logger = structlog.stdlib.get_logger(__name__)


def _strict_embedding_endpoint_digest(base_url: str) -> str:
    """Hash a canonical, credential-free endpoint identity for build audits."""

    if (
        type(base_url) is not str
        or base_url != base_url.strip()
        or not base_url
        or len(base_url) > 2_048
        or any(character in base_url for character in "\x00\r\n")
    ):
        raise RuntimeError("Configured embedding endpoint must be a safe URL")
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("Configured embedding endpoint must be a valid URL") from exc
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if (
        scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            "Configured embedding endpoint must be credential-free HTTP(S) URL"
        )
    host = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    path = parsed.path.rstrip("/")
    canonical = f"{scheme}://{host}{path}"
    return sha256(canonical.encode("utf-8")).hexdigest()


def get_multimodal_embedding(
    prompt: str,
    image_data_url: str,
    input_type: str = "search_document",
) -> list[float]:
    """
    Get an embedding for multimodal (text + image) input.

    First attempts native vLLM multimodal embedding, then falls back to
    text-only embedding of the prompt if multimodal is not supported.

    Args:
        prompt: Text description/caption for the image
        image_data_url: Base64 data URL of the image (data:image/...;base64,...)
        input_type: Embedding type (search_document, search_query, etc.)

    Returns:
        Embedding vector fitted to APP_EMBED_DIMS
    """
    if input_type not in (
        "search_document",
        "search_query",
        "classification",
        "clustering",
    ):
        raise ValueError(f"bad input type to embedding call: {input_type}")

    try:
        embedding = get_multimodal_embedding_via_vllm_pooling(prompt, image_data_url)
        if embedding:
            logger.info(
                "obs.core.multimodal_embedding_generated",
                **retrieval_log_fields(
                    reason=RetrievalLogReason.COMPLETED,
                    count=0,
                    elapsed_ms=0.0,
                ),
            )
            return fit_embedding_dims(embedding)
    except Exception:
        logger.debug(
            "obs.core.multimodal_embedding_failed",
            **retrieval_log_fields(
                reason=RetrievalLogReason.UPSTREAM_UNAVAILABLE,
                count=0,
                elapsed_ms=0.0,
            ),
        )

    logger.debug(
        "obs.core.multimodal_embedding_unavailable",
        **retrieval_log_fields(
            reason=RetrievalLogReason.EMBEDDING_UNAVAILABLE,
            count=0,
            elapsed_ms=0.0,
        ),
    )
    return get_embedding(prompt, input_type=input_type)


def get_embedding(query: Any, input_type: str = "search_query"):
    """Get embedding for a single query, with Cohere fallback."""
    if input_type not in (
        "search_document",
        "search_query",
        "classification",
        "clustering",
    ):
        raise ValueError(f"bad input type to embedding call: {input_type}")

    try:
        return fit_embedding_dims(get_embedding_via_local_openai(query))
    except Exception:
        logger.warning(
            "obs.core.embedding_local_fallback",
            **retrieval_log_fields(
                reason=RetrievalLogReason.EMBEDDING_UNAVAILABLE,
                count=0,
                elapsed_ms=0.0,
            ),
        )

    if not isinstance(query, str):
        raise RuntimeError(
            "All embedding providers failed: local provider rejected non-text "
            "embedding payload "
            "and Cohere fallback only supports text."
        )
    try:
        cohere_client = apps.get_app_config("aquillm").cohere_client
        return fit_embedding_dims(
            get_embedding_via_cohere(cohere_client, query, input_type)
        )
    except Exception as exc:
        raise RuntimeError("All embedding providers failed") from exc


def get_embeddings(
    queries: list[Any], input_type: str = "search_query"
) -> list[list[float]]:
    """Get embeddings for multiple queries, with Cohere fallback."""
    if input_type not in (
        "search_document",
        "search_query",
        "classification",
        "clustering",
    ):
        raise ValueError(f"bad input type to embedding call: {input_type}")
    if not queries:
        return []
    try:
        return [
            fit_embedding_dims(emb) for emb in get_embeddings_via_local_openai(queries)
        ]
    except Exception:
        logger.warning(
            "obs.core.embedding_batch_local_fallback",
            **retrieval_log_fields(
                reason=RetrievalLogReason.EMBEDDING_UNAVAILABLE,
                count=0,
                elapsed_ms=0.0,
            ),
        )
    if not all(isinstance(q, str) for q in queries):
        raise RuntimeError(
            "All embedding providers failed: local provider rejected non-text "
            "embedding payloads "
            "and Cohere fallback only supports text."
        )
    try:
        cohere_client = apps.get_app_config("aquillm").cohere_client
        text_queries: list[str] = [q for q in queries if isinstance(q, str)]
        return [
            fit_embedding_dims(emb)
            for emb in get_embeddings_via_cohere(
                cohere_client, text_queries, input_type
            )
        ]
    except Exception as exc:
        raise RuntimeError("All embedding providers failed") from exc


def strict_index_embedding_signature() -> str:
    """Return the auditable local-only signature used by durable KG builds."""

    base_url, _api_key, model = get_local_embed_config()
    dimensions = get_target_dims()
    if dimensions != 1024:
        raise RuntimeError("Durable KG entity embeddings require APP_EMBED_DIMS=1024")
    revision = getenv("APP_EMBED_MODEL_REVISION")
    if not revision:
        raise RuntimeError(
            "APP_EMBED_MODEL_REVISION must name an immutable model revision or digest"
        )
    for value, label in ((model, "embedding model"), (revision, "model revision")):
        if (
            type(value) is not str
            or value != value.strip()
            or not value
            or len(value) > 256
            or any(character in value for character in "\x00\r\n")
        ):
            raise RuntimeError(f"Configured {label} must be a safe nonempty token")
    endpoint_digest = _strict_embedding_endpoint_digest(base_url)
    return (
        f"local-openai:{model}@{revision}:endpoint={endpoint_digest}:"
        f"dims={dimensions}:prep=kg-entity-v1:"
        "max_chars=8192:batch=64"
    )


def get_strict_index_embeddings(
    queries: list[str],
    *,
    expected_model_signature: str,
) -> tuple[list[tuple[int, list[float]]], str]:
    """Embed one durable index batch locally with no cross-provider fallback.

    The ordinary query interface intentionally falls back to Cohere for
    availability. A durable graph build cannot do that because one artifact
    must never silently mix providers or model revisions.
    """

    if type(queries) is not list or any(type(query) is not str for query in queries):
        raise ValueError(
            "strict index embedding inputs must be an exact list of strings"
        )
    actual_signature = strict_index_embedding_signature()
    if expected_model_signature != actual_signature:
        raise RuntimeError(
            "Configured embedding provider/model signature drift detected"
        )
    if not queries:
        return [], actual_signature
    if any(len(query) > 8_192 for query in queries):
        raise ValueError("strict index embedding input exceeds max_chars=8192")
    indexed_vectors = get_strict_indexed_embeddings_via_local_openai(queries)
    if not isinstance(indexed_vectors, (list, tuple)) or len(indexed_vectors) != len(
        queries
    ):
        raise RuntimeError("Local embedding endpoint returned an invalid batch count")
    indices = tuple(item[0] for item in indexed_vectors)
    if len(set(indices)) != len(indices) or set(indices) != set(range(len(queries))):
        raise RuntimeError("Local embedding endpoint returned invalid provider indices")
    vectors: list[tuple[int, list[float]]] = []
    for index, raw_vector in indexed_vectors:
        if not isinstance(raw_vector, (list, tuple)) or len(raw_vector) != 1024:
            raise RuntimeError(
                "Local embedding endpoint returned an invalid 1024-d vector"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in raw_vector
        ):
            raise RuntimeError("Local embedding endpoint returned an invalid vector")
        vectors.append((index, [float(value) for value in raw_vector]))
    return vectors, actual_signature


__all__ = [
    "get_embedding",
    "get_embeddings",
    "get_strict_index_embeddings",
    "strict_index_embedding_signature",
    "get_multimodal_embedding",
]
