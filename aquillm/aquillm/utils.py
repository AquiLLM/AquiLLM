"""
Embedding utilities with Django app integration.

This module provides the main embedding interface for the application,
integrating lib/embeddings with Django app configuration for Cohere fallback.
"""

import structlog
from typing import Any

from django.apps import apps

# Import from lib/embeddings for pure Python operations
from lib.embeddings import (
    get_embedding_via_local_openai,
    get_embeddings_via_local_openai,
    get_embedding_via_cohere,
    get_embeddings_via_cohere,
    get_multimodal_embedding_via_vllm_pooling,
    fit_embedding_dims,
)

logger = structlog.stdlib.get_logger(__name__)


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
    if input_type not in ("search_document", "search_query", "classification", "clustering"):
        raise ValueError(f"bad input type to embedding call: {input_type}")
    
    try:
        embedding = get_multimodal_embedding_via_vllm_pooling(prompt, image_data_url)
        if embedding:
            logger.info("obs.embed.multimodal_success")
            return fit_embedding_dims(embedding)
    except Exception as exc:
        logger.debug("obs.embed.multimodal_fallback", error_type=type(exc).__name__, error=str(exc))
    
    logger.debug("obs.embed.multimodal_fallback")
    return get_embedding(prompt, input_type=input_type)


def get_embedding(query: Any, input_type: str = "search_query"):
    """Get embedding for a single query, with Cohere fallback."""
    if input_type not in ("search_document", "search_query", "classification", "clustering"):
        raise ValueError(f"bad input type to embedding call: {input_type}")

    try:
        return fit_embedding_dims(get_embedding_via_local_openai(query))
    except Exception as exc:
        logger.warning("obs.embed.local_fallback", error_type=type(exc).__name__, error=str(exc))

    if not isinstance(query, str):
        raise RuntimeError(
            "All embedding providers failed: local provider rejected non-text embedding payload "
            "and Cohere fallback only supports text."
        )
    try:
        cohere_client = apps.get_app_config("aquillm").cohere_client
        return fit_embedding_dims(get_embedding_via_cohere(cohere_client, query, input_type))
    except Exception as exc:
        raise RuntimeError(f"All embedding providers failed: {exc}") from exc


def get_embeddings(queries: list[Any], input_type: str = "search_query") -> list[list[float]]:
    """Get embeddings for multiple queries, with Cohere fallback."""
    if input_type not in ("search_document", "search_query", "classification", "clustering"):
        raise ValueError(f"bad input type to embedding call: {input_type}")
    if not queries:
        return []
    try:
        return [fit_embedding_dims(emb) for emb in get_embeddings_via_local_openai(queries)]
    except Exception as exc:
        logger.warning("obs.embed.batch_fallback", error_type=type(exc).__name__, error=str(exc))
    if not all(isinstance(q, str) for q in queries):
        raise RuntimeError(
            "All embedding providers failed: local provider rejected non-text embedding payloads "
            "and Cohere fallback only supports text."
        )
    try:
        cohere_client = apps.get_app_config("aquillm").cohere_client
        text_queries: list[str] = [q for q in queries if isinstance(q, str)]
        return [fit_embedding_dims(emb) for emb in get_embeddings_via_cohere(cohere_client, text_queries, input_type)]
    except Exception as exc:
        raise RuntimeError(f"All embedding providers failed: {exc}") from exc


__all__ = [
    'get_embedding',
    'get_embeddings',
    'get_multimodal_embedding',
]
