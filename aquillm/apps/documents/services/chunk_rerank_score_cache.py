"""Versioned scored-result cache; serialized values contain no source text."""

from __future__ import annotations

from uuid import UUID

from apps.documents.services import rag_cache
from apps.documents.services.chunk_rerank_results import (
    PassageScore,
    RerankScoreSet,
    ScoredRerankResult,
    fingerprint_pool,
    score_identity_for_chunk,
    validate_score_set,
)


def scored_result_cache_key(
    *,
    query_fingerprint: str,
    scorer_fingerprint: str,
    scoring_kind: str,
    candidate_identities: tuple[tuple[int, str, str], ...],
    preparation_policy_fingerprint: str,
) -> str:
    if any(
        type(pk) is not int or pk <= 0 or not source or not pair
        for pk, source, pair in candidate_identities
    ):
        raise ValueError("invalid cache candidate identity")
    if len({pk for pk, _, _ in candidate_identities}) != len(candidate_identities):
        raise ValueError("duplicate cache candidate identity")
    return rag_cache.stable_cache_key(
        "rrscore:v2",
        query_fingerprint,
        scorer_fingerprint,
        scoring_kind,
        fingerprint_pool(candidate_identities),
        preparation_policy_fingerprint,
    )


def get_scored_result(key: str) -> RerankScoreSet | None:
    if not key.startswith("rrscore:v2:"):
        return None
    data = rag_cache.cache_get(key)
    if not isinstance(data, dict) or data.get("schema_version") != "v2":
        return None
    try:
        scores = tuple(
            PassageScore(pk, UUID(doc_id), number, source, pair, value)
            for pk, doc_id, number, source, pair, value in data["scores"]
        )
        return RerankScoreSet(
            data["schema_version"],
            data["query_fingerprint"],
            data["scorer_fingerprint"],
            data["pool_fingerprint"],
            data["scoring_kind"],
            data["status"],
            tuple(data["candidate_order"]),
            scores,
        )
    except (KeyError, TypeError, ValueError):
        return None


def set_scored_result(
    key: str, result: RerankScoreSet, *, timeout_seconds: int
) -> None:
    if not key.startswith("rrscore:v2:") or result.schema_version != "v2":
        return
    rag_cache.cache_set(
        key,
        {
            "schema_version": result.schema_version,
            "query_fingerprint": result.query_fingerprint,
            "scorer_fingerprint": result.scorer_fingerprint,
            "pool_fingerprint": result.pool_fingerprint,
            "scoring_kind": result.scoring_kind,
            "status": result.status,
            "candidate_order": result.candidate_order,
            "scores": tuple(
                (
                    score.chunk_pk,
                    str(score.document_id),
                    score.chunk_number,
                    score.source_fingerprint,
                    score.effective_pair_fingerprint,
                    score.value,
                )
                for score in result.scores
            ),
        },
        timeout_seconds,
    )


def reusable_scored_result(
    *,
    key: str,
    chunks: list,
    canonical_identities: tuple[tuple[int, str, str], ...],
    query_fingerprint: str,
    scorer_fingerprint: str,
    pool_fingerprint: str,
    top_k: int,
) -> ScoredRerankResult | None:
    """Return a cache hit only after current identity and scope validation."""
    cached = get_scored_result(key)
    if cached is None:
        return None
    try:
        validate_score_set(
            cached,
            authorized_identities=tuple(
                score_identity_for_chunk(chunk, effective_pair_fingerprint=identity[2])
                for chunk, identity in zip(chunks, canonical_identities)
            ),
            expected_query_fingerprint=query_fingerprint,
            expected_scorer_fingerprint=scorer_fingerprint,
        )
    except ValueError:
        return None
    if cached.pool_fingerprint != pool_fingerprint or cached.status != "complete":
        return None
    positions = {chunk.pk: index for index, chunk in enumerate(chunks)}
    ranked = sorted(
        cached.scores, key=lambda score: (-score.value, positions[score.chunk_pk])
    )
    return ScoredRerankResult(tuple(score.chunk_pk for score in ranked[:top_k]), cached)


def window_cache_key(scorer_fingerprint, plan_fingerprint):
    return rag_cache.stable_cache_key(
        "rrwindow:v1", scorer_fingerprint, plan_fingerprint
    )


def get_window_result(key, *, budget=None):
    from .chunk_rerank_score_transport import deserialize_score_set

    if not rag_cache._rag_enabled() or not key.startswith("rrwindow:v1:"):
        return None
    from .bounded_rag_cache import cache_operation

    result = deserialize_score_set(
        cache_operation("get", key, budget=budget)
        if budget
        else rag_cache.cache_get(key)
    )
    return result if result and result.schema_version == "v3-window" else None


def set_window_result(key, result, *, budget):
    from .chunk_rerank_score_transport import serialize_score_set

    if (
        not rag_cache._rag_enabled()
        or budget is None
        or not key.startswith("rrwindow:v1:")
        or result.schema_version != "v3-window"
        or result.status != "complete"
    ):
        return False
    from .bounded_rag_cache import cache_operation

    return bool(
        cache_operation(
            "set",
            key,
            serialize_score_set(result),
            timeout=rag_cache.rerank_result_ttl(),
            budget=budget,
        )
    )


__all__ = [
    "scored_result_cache_key",
    "get_scored_result",
    "set_scored_result",
    "reusable_scored_result",
]
