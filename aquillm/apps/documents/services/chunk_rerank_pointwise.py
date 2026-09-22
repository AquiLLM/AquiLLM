"""Bounded pointwise HTTP scoring and validation for local rerankers."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import isfinite

import requests

from .chunk_rerank_budget import trim_rerank_pair
from .chunk_rerank_parse import parse_single_score


def _is_complete_finite_scoring(
    pairs: list[tuple[int, float]],
    candidate_count: int,
) -> bool:
    return (
        len(pairs) == candidate_count
        and all(
            type(index) is int
            and 0 <= index < candidate_count
            and type(score) in (int, float)
            and isfinite(float(score))
            for index, score in pairs
        )
        and {index for index, _score in pairs} == set(range(candidate_count))
    )


def _score_one_document(
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
    parse_score=parse_single_score,
    trim_pair=trim_rerank_pair,
) -> tuple[int, float] | None:
    try:
        request_query = query
        request_document = document
        adaptive_reserve = min(
            max(0, pair_token_limit - 2),
            max(reserve_tokens + 256, pair_token_limit // 2),
        )
        response = requests.post(
            endpoint,
            headers=headers,
            json={
                "model": model_name,
                "text_1": request_query,
                "text_2": request_document,
                # Let vLLM make the final fit with the model's actual tokenizer
                # and score template. The local cl100k estimate can otherwise
                # undercount a Qwen pair by enough to turn 1024 into 1025.
                "truncate_prompt_tokens": pair_token_limit,
                "truncation_side": "right",
            },
            timeout=timeout,
        )
        if response.status_code == 400 and request_document:
            retry_reserve = min(
                max(0, pair_token_limit - 2),
                max(adaptive_reserve + 256, (pair_token_limit * 3) // 4),
            )
            retry_query, retry_document = trim_pair(
                request_query,
                request_document,
                pair_token_limit,
                retry_reserve,
            )
            if (retry_query, retry_document) != (
                request_query,
                request_document,
            ):
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
        score = parse_score(response.json())
        if type(score) not in (int, float) or not isfinite(float(score)):
            return None
        return index, float(score)
    except Exception:
        return None


def _score_documents_concurrently(
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
    score_one=_score_one_document,
) -> list[tuple[int, float]]:
    if not documents:
        return []
    workers = min(max(1, max_workers), len(documents))
    scores: list[tuple[int, float]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                score_one,
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


def _rank_complete_scores(
    pairs: list[tuple[int, float]],
    chunks_list,
    top_k: int,
) -> list[int]:
    if not _is_complete_finite_scoring(pairs, len(chunks_list)):
        return []
    return [
        chunks_list[index].pk
        for index, _score in sorted(pairs, key=lambda item: item[1], reverse=True)[
            :top_k
        ]
    ]
