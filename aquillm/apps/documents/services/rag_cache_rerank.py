"""Capability cache records for the local reranker HTTP client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from django.core.cache import cache

from apps.documents.services import rag_cache


def _current_candidates(scope, candidate_ids):
    from django.db.models.functions import MD5

    from apps.documents.models import TextChunk

    with scope.database():
        scope.validate()
        rows = (
            scope.queryset(
                TextChunk.objects.using(scope.using).filter(
                    pk__in=candidate_ids,
                )
            )
            .annotate(_revision=MD5("content"))
            .values_list(
                "pk",
                "doc_id",
                "chunk_number",
                "_revision",
                "representation_generation",
            )
        )
        current = {row[0]: tuple(row[1:]) for row in rows}
        return all(
            pk in current and current[pk] == scope.bodies.get(pk)
            for pk in candidate_ids
        )


def get_scoped_rerank_result(signature, candidate_ids, top_k, model):
    from .chunk_search_scope import current_scope

    scope = current_scope()
    if scope is None:
        return rag_cache.get_cached_rerank_result(
            signature, candidate_ids, top_k, model
        )
    key = scope.cache_key()
    if key is None or not _current_candidates(scope, candidate_ids):
        return None
    rows = rag_cache.get_cached_rerank_result(key, candidate_ids, top_k, model)
    if (
        not _current_candidates(scope, candidate_ids)
        or scope.cache_key() != key
        or not isinstance(rows, list)
        or len(rows) > top_k
        or any(type(pk) is not int or pk not in candidate_ids for pk in rows)
        or len(set(rows)) != len(rows)
    ):
        return None
    return rows


def set_scoped_rerank_result(signature, candidate_ids, top_k, model, ranked_ids):
    from .chunk_search_scope import current_scope

    scope = current_scope()
    if scope is None:
        return rag_cache.set_cached_rerank_result(
            signature, candidate_ids, top_k, model, ranked_ids
        )
    key = scope.cache_key()
    if (
        key is not None
        and _current_candidates(scope, candidate_ids)
        and scope.cache_key() == key
    ):
        rag_cache.set_cached_rerank_result(key, candidate_ids, top_k, model, ranked_ids)


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


def get_cached_rerank_capability(base_url: str, model: str) -> RerankCapability | None:
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
