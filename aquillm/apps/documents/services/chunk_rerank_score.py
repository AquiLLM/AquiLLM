"""Concurrent single-pair scoring for the local reranker client."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from math import isfinite
from typing import Any

import requests

from apps.documents.services.chunk_rerank_budget import trim_rerank_pair
from apps.documents.services.chunk_rerank_parse import (
    parse_score_results,
    parse_single_score,
)


def batch_score_payloads(
    model_name: str,
    query: str,
    documents: list[str],
    multimodal_documents: list[Any],
    has_multimodal_documents: bool,
) -> tuple[dict[str, Any], ...]:
    payloads: tuple[dict[str, Any], ...] = (
        {"model": model_name, "text_1": query, "text_2": documents},
        {"text_1": query, "text_2": documents},
        {"model": model_name, "query": query, "documents": documents},
    )
    if has_multimodal_documents:
        payloads += (
            {
                "model": model_name,
                "query": [{"type": "text", "text": query}],
                "documents": multimodal_documents,
            },
        )
    return payloads


def score_one_document(
    *,
    endpoint: str,
    index: int,
    query: str,
    document: str,
    headers: dict[str, str],
    timeout: int,
    model_name: str,
    pair_token_limit: int,
    reserve_tokens: int,
) -> tuple[int, float] | None:
    try:
        adaptive_reserve = min(
            max(0, pair_token_limit - 2),
            max(reserve_tokens + 256, pair_token_limit // 2),
        )
        response = requests.post(
            endpoint,
            headers=headers,
            json={
                "model": model_name,
                "text_1": query,
                "text_2": document,
                # The server tokenizer and score template decide the final fit.
                "truncate_prompt_tokens": pair_token_limit,
                "truncation_side": "right",
            },
            timeout=timeout,
        )
        if response.status_code == 400 and document:
            retry_reserve = min(
                max(0, pair_token_limit - 2),
                max(adaptive_reserve + 256, (pair_token_limit * 3) // 4),
            )
            retry_query, retry_document = trim_rerank_pair(
                query,
                document,
                pair_token_limit,
                retry_reserve,
            )
            if (retry_query, retry_document) != (query, document):
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json={
                        "model": model_name,
                        "text_1": retry_query,
                        "text_2": retry_document,
                        "truncate_prompt_tokens": pair_token_limit,
                        "truncation_side": "right",
                    },
                    timeout=timeout,
                )
        if response.status_code >= 400:
            return None
        score = parse_single_score(response.json())
        if type(score) not in (int, float) or not isfinite(float(score)):
            return None
        return index, float(score)
    except Exception:
        return None


def score_documents_concurrently(
    *,
    endpoint: str,
    query: str,
    documents: list[str],
    headers: dict[str, str],
    timeout: int,
    max_workers: int,
    model_name: str,
    pair_token_limit: int,
    reserve_tokens: int,
) -> list[tuple[int, float]]:
    if not documents:
        return []
    workers = min(max(1, max_workers), len(documents))
    scores: list[tuple[int, float]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                score_one_document,
                endpoint=endpoint,
                index=index,
                query=query,
                document=document,
                headers=headers,
                timeout=timeout,
                model_name=model_name,
                pair_token_limit=pair_token_limit,
                reserve_tokens=reserve_tokens,
            )
            for index, document in enumerate(documents)
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                scores.append(result)
    return scores


def rank_complete_scores(
    pairs: list[tuple[int, float]], chunks_list, top_k: int
) -> list[int]:
    candidate_count = len(chunks_list)
    if not (
        len(pairs) == candidate_count
        and all(
            type(index) is int
            and 0 <= index < candidate_count
            and type(score) in (int, float)
            and isfinite(float(score))
            for index, score in pairs
        )
        and {index for index, _score in pairs} == set(range(candidate_count))
    ):
        return []
    return [
        chunks_list[index].pk
        for index, _score in sorted(pairs, key=lambda item: (-item[1], item[0]))[:top_k]
    ]


def parse_batch_scores(body, chunks_list, top_k: int) -> list[int]:
    return rank_complete_scores(parse_score_results(body), chunks_list, top_k)


__all__ = [
    "batch_score_payloads",
    "parse_batch_scores",
    "rank_complete_scores",
    "score_documents_concurrently",
    "score_one_document",
]
