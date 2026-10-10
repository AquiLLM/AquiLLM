"""
Cohere embedding provider.

Note: Requires a Cohere client to be passed in. The Django app config provides
this via apps.get_app_config("aquillm").cohere_client.
"""

from typing import Any

from .utils import EmbeddingContractError, validate_embedding


def _response_vectors(response: Any, count: int) -> list[list[float]]:
    vectors = getattr(response, "embeddings", None)
    if not isinstance(vectors, (list, tuple)) or len(vectors) != count:
        raise EmbeddingContractError("Embedding response count differs from input")
    for vector in vectors:
        validate_embedding(vector)
    return [list(vector) for vector in vectors]


def get_embedding_via_cohere(
    cohere_client: Any, query: str, input_type: str
) -> list[float]:
    """Get embedding via Cohere API."""
    if cohere_client is None:
        raise RuntimeError("Cohere client not configured")
    response = cohere_client.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type=input_type,
    )
    return _response_vectors(response, 1)[0]


def get_embeddings_via_cohere(
    cohere_client: Any, queries: list[str], input_type: str
) -> list[list[float]]:
    """Get batch embeddings via Cohere API."""
    if not queries:
        return []
    if cohere_client is None:
        raise RuntimeError("Cohere client not configured")
    response = cohere_client.embed(
        texts=queries,
        model="embed-english-v3.0",
        input_type=input_type,
    )
    return _response_vectors(response, len(queries))


__all__ = [
    "get_embedding_via_cohere",
    "get_embeddings_via_cohere",
]
