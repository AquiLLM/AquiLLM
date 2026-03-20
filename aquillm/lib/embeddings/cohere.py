"""
Cohere embedding provider.

Note: Requires a Cohere client to be passed in. The Django app config provides
this via apps.get_app_config("aquillm").cohere_client.
"""

from typing import Any


def get_embedding_via_cohere(cohere_client: Any, query: str, input_type: str) -> list[float]:
    """Get embedding via Cohere API."""
    if cohere_client is None:
        raise RuntimeError("Cohere client not configured")
    response = cohere_client.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type=input_type,
    )
    return response.embeddings[0]


def get_embeddings_via_cohere(cohere_client: Any, queries: list[str], input_type: str) -> list[list[float]]:
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
    return response.embeddings


__all__ = [
    'get_embedding_via_cohere',
    'get_embeddings_via_cohere',
]
