"""Capability cache records for the local reranker HTTP client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from django.core.cache import cache

from apps.documents.services import rag_cache


class RerankCapability(TypedDict):
    endpoint: str
    shape: str


_SHAPES = {
    "rerank_documents",
    "score_batch_text_pairs",
    "score_single_text_pair",
}


def rerank_capability_cache_key(base_url: str, model: str) -> str:
    return rag_cache.stable_cache_key("rrcap", base_url.rstrip("/"), model)


def rerank_capability_ttl() -> int:
    from django.conf import settings

    return int(getattr(settings, "RAG_RERANK_CAPABILITY_TTL_SECONDS", 900))


def get_cached_rerank_capability(
    base_url: str, model: str
) -> RerankCapability | None:
    if not rag_cache._rag_enabled():
        return None
    val = rag_cache.cache_get(rerank_capability_cache_key(base_url, model))
    if (
        isinstance(val, dict)
        and isinstance(val.get("endpoint"), str)
        and val.get("endpoint")
        and val.get("shape") in _SHAPES
    ):
        rag_cache._log_hit_miss(rag_cache._MET_RERANK_CAP, True)
        return {"endpoint": val["endpoint"], "shape": val["shape"]}
    if isinstance(val, str) and val:
        # Read endpoint-only records during rolling deployments.
        rag_cache._log_hit_miss(rag_cache._MET_RERANK_CAP, True)
        return {"endpoint": val, "shape": "rerank_documents"}
    rag_cache._log_hit_miss(rag_cache._MET_RERANK_CAP, False)
    return None


def set_cached_rerank_capability(
    base_url: str, model: str, capability: Mapping[str, str]
) -> None:
    endpoint = capability.get("endpoint")
    shape = capability.get("shape")
    if not endpoint or shape not in _SHAPES:
        return
    rag_cache.cache_set(
        rerank_capability_cache_key(base_url, model),
        {"endpoint": endpoint, "shape": shape},
        rerank_capability_ttl(),
    )


def delete_cached_rerank_capability(base_url: str, model: str) -> None:
    if not rag_cache._rag_enabled():
        return
    try:
        cache.delete(rerank_capability_cache_key(base_url, model))
    except Exception:
        return


__all__ = [
    "RerankCapability",
    "delete_cached_rerank_capability",
    "get_cached_rerank_capability",
    "rerank_capability_cache_key",
    "rerank_capability_ttl",
    "set_cached_rerank_capability",
]
