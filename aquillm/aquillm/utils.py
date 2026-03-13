
import logging
from os import getenv

from django.apps import apps
from openai import OpenAI

logger = logging.getLogger(__name__)


def _fit_embedding_dims(embedding: list[float]) -> list[float]:
    """
    Fit embedding vectors to the pgvector schema dimension.
    Existing DB columns are vector(1024), so pad/truncate as needed.
    """
    target_raw = getenv("APP_EMBED_DIMS", "1024").strip()
    try:
        target_dims = int(target_raw)
    except Exception:
        target_dims = 1024
    if target_dims <= 0:
        target_dims = 1024

    current = len(embedding)
    if current == target_dims:
        return embedding
    if current > target_dims:
        logger.warning(
            "Embedding dims (%d) exceed APP_EMBED_DIMS (%d); truncating.",
            current,
            target_dims,
        )
        return embedding[:target_dims]
    logger.warning(
        "Embedding dims (%d) below APP_EMBED_DIMS (%d); zero-padding.",
        current,
        target_dims,
    )
    return embedding + [0.0] * (target_dims - current)


def _get_local_embed_config() -> tuple[str, str, str]:
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
        or "nomic-ai/nomic-embed-text-v1.5"
    )
    return base_url, api_key, model


def _get_embedding_via_local_openai(query: str) -> list[float]:
    base_url, api_key, model = _get_local_embed_config()
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.embeddings.create(
        model=model,
        input=query,
    )
    return response.data[0].embedding


def _get_embedding_via_cohere(query: str, input_type: str) -> list[float]:
    cohere_client = apps.get_app_config("aquillm").cohere_client
    if cohere_client is None:
        raise RuntimeError("Cohere client not configured")
    response = cohere_client.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type=input_type,
    )
    return response.embeddings[0]


def get_embedding(query: str, input_type: str = "search_query"):
    if input_type not in ("search_document", "search_query", "classification", "clustering"):
        raise ValueError(f"bad input type to embedding call: {input_type}")

    # Prefer self-hosted OpenAI-compatible embeddings first.
    try:
        return _fit_embedding_dims(_get_embedding_via_local_openai(query))
    except Exception as exc:
        logger.warning("Local embed request failed; trying Cohere fallback. Error: %s", exc)

    # Fallback for legacy deployments still using Cohere.
    try:
        return _fit_embedding_dims(_get_embedding_via_cohere(query, input_type))
    except Exception as exc:
        raise RuntimeError(f"All embedding providers failed: {exc}") from exc
