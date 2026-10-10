"""Compatibility helpers using bounded one-attempt pointwise transport."""

from math import isfinite

import requests as requests  # Compatibility patch surface for provider tests.

from .chunk_rerank_budget import trim_rerank_pair
from .chunk_rerank_local_scored_http import _score_one_document as _score_pair
from .chunk_rerank_local_scored_http import _score_single_pool
from .chunk_rerank_parse import parse_single_score


def _is_complete_finite_scoring(pairs, candidate_count):
    return (
        len(pairs) == candidate_count
        and all(
            type(i) is int
            and 0 <= i < candidate_count
            and type(v) in (int, float)
            and isfinite(float(v))
            for i, v in pairs
        )
        and {i for i, _ in pairs} == set(range(candidate_count))
    )


def _score_one_document(
    *, parse_score=parse_single_score, trim_pair=trim_rerank_pair, **kwargs
):
    result = _score_pair(**kwargs)
    return result[:2] if result is not None else None


def _score_documents_concurrently(
    *,
    endpoint,
    query,
    documents,
    headers,
    timeout,
    max_workers,
    model_name,
    pair_token_limit,
    reserve_tokens,
    score_one=_score_one_document,
    turn_budget=None,
    rerank_state=None,
):
    results = _score_single_pool(
        endpoint,
        [(query, doc) for doc in documents],
        headers,
        timeout,
        model_name,
        pair_token_limit,
        reserve_tokens,
        turn_budget=turn_budget,
        rerank_state=rerank_state,
        max_workers=max_workers,
    )
    return [(i, value) for i, value, _ in results]


def _rank_complete_scores(pairs, chunks_list, top_k):
    if not _is_complete_finite_scoring(pairs, len(chunks_list)):
        return []
    return [
        chunks_list[i].pk
        for i, _ in sorted(pairs, key=lambda item: (-item[1], item[0]))[:top_k]
    ]
