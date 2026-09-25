from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from math import isfinite

import requests

from apps.documents.services.chunk_rerank_budget import trim_rerank_pair
from apps.documents.services.chunk_rerank_config import rerank_score_concurrency
from apps.documents.services.chunk_rerank_parse import parse_single_score
from apps.documents.services.chunk_rerank_results import order_scored_pairs


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
) -> tuple[int, float, tuple[str, str]] | None:
    """Keep the successful pair, including the 400 retry's stricter trim."""
    successful_pair = (query, document)

    def send(pair):
        return requests.post(
            endpoint,
            headers=headers,
            json={
                "model": model_name,
                "text_1": pair[0],
                "text_2": pair[1],
                "truncate_prompt_tokens": pair_token_limit,
                "truncation_side": "right",
            },
            timeout=timeout,
        )

    try:
        response = send(successful_pair)
        if response.status_code == 400 and document:
            adaptive = min(
                max(0, pair_token_limit - 2),
                max(reserve_tokens + 256, pair_token_limit // 2),
            )
            retry_reserve = min(
                max(0, pair_token_limit - 2),
                max(adaptive + 256, (pair_token_limit * 3) // 4),
            )
            retry_pair = trim_rerank_pair(
                query, document, pair_token_limit, retry_reserve
            )
            if retry_pair != successful_pair:
                response = send(retry_pair)
                successful_pair = retry_pair
        if response.status_code >= 400:
            return None
        value = parse_single_score(response.json())
        if type(value) not in (int, float) or not isfinite(float(value)):
            return None
        return index, float(value), successful_pair
    except Exception:
        return None


def _score_single_pool(
    endpoint, pairs, headers, timeout, model_name, pair_limit, reserve
):
    workers = min(6, rerank_score_concurrency(), len(pairs))
    if not workers:
        return []
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _score_one_document,
                endpoint=endpoint,
                index=index,
                query=query,
                document=document,
                headers=headers,
                timeout=timeout,
                model_name=model_name,
                pair_token_limit=pair_limit,
                reserve_tokens=reserve,
            )
            for index, (query, document) in enumerate(pairs)
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    return results


def _complete_pairs(indexed, candidate_ids):
    try:
        ordered = order_scored_pairs(tuple(indexed), candidate_ids)
    except (TypeError, ValueError):
        return None
    return tuple(pk for pk, _value in ordered)


def _rerank_scores(body):
    if not isinstance(body, dict):
        return []
    raw = body.get("results", body.get("data", []))
    if not isinstance(raw, list):
        return []
    indexed = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        value = item.get("relevance_score", item.get("score"))
        if (
            type(index) is int
            and type(value) in (int, float)
            and isfinite(float(value))
        ):
            indexed.append((index, float(value)))
    return indexed
