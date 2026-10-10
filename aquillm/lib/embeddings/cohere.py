"""
Cohere embedding provider.

Note: Requires a Cohere client to be passed in. The Django app config provides
this via apps.get_app_config("aquillm").cohere_client.
"""

from typing import Any

from .provenance import EmbeddingResult, make_result
from .utils import EmbeddingContractError, validate_embedding
from .errors import REQUEST_TIMEOUT_SECONDS


def _response_vectors(response: Any, count: int) -> list[list[float]]:
    vectors = getattr(response, "embeddings", None)
    if not isinstance(vectors, (list, tuple)) or len(vectors) != count:
        raise EmbeddingContractError("Embedding response count differs from input")
    for vector in vectors:
        validate_embedding(vector)
    return [list(vector) for vector in vectors]


def get_embedding_result_via_cohere(
    cohere_client: Any, query: str, input_type: str
) -> EmbeddingResult:
    """Get embedding via Cohere API."""
    if cohere_client is None:
        raise RuntimeError("Cohere client not configured")
    response = cohere_client.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type=input_type,
        request_options={"timeout_in_seconds": REQUEST_TIMEOUT_SECONDS, "max_retries": 0},
        batching=False,
    )
    return _cohere_result(
        _response_vectors(response, 1)[0], query, input_type, response
    )


def get_embedding_results_via_cohere(
    cohere_client: Any, queries: list[str], input_type: str
) -> list[EmbeddingResult]:
    """Get batch embeddings via Cohere API."""
    if not queries:
        return []
    if cohere_client is None:
        raise RuntimeError("Cohere client not configured")
    results = []
    # Match the SDK's 96-input cap, but stop before dispatching more on failure.
    for start in range(0, len(queries), 96):
        batch = queries[start:start + 96]
        response = cohere_client.embed(
            texts=batch,
            model="embed-english-v3.0",
            input_type=input_type,
            request_options={"timeout_in_seconds": REQUEST_TIMEOUT_SECONDS, "max_retries": 0},
            batching=False,
        )
        results.extend(
            _cohere_result(vector, query, input_type, response)
            for vector, query in zip(_response_vectors(response, len(batch)), batch)
        )
    return results


__all__ = [
    "get_embedding_via_cohere",
    "get_embeddings_via_cohere",
]


def _cohere_result(vector, query, input_type, response):
    return make_result(
        vector,
        provider="cohere",
        route="cohere-embed",
        role=input_type,
        prepared_input=query,
        original_input=query,
        model="embed-english-v3.0",
        response_model=getattr(response, "model", None),
    )


def get_embedding_via_cohere(
    cohere_client: Any, query: str, input_type: str
) -> list[float]:
    return get_embedding_result_via_cohere(cohere_client, query, input_type).vector


def get_embeddings_via_cohere(
    cohere_client: Any, queries: list[str], input_type: str
) -> list[list[float]]:
    return [
        result.vector
        for result in get_embedding_results_via_cohere(
            cohere_client, queries, input_type
        )
    ]
