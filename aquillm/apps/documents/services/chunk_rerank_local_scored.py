"""Scored local reranking; each result records the successful HTTP input."""

from __future__ import annotations

import requests

from apps.documents.services import rag_cache
from apps.documents.services.chunk_rerank_budget import trim_rerank_pair
from apps.documents.services.chunk_rerank_config import (
    rerank_api_key,
    rerank_base_url,
    rerank_doc_char_limit,
    rerank_model,
    rerank_model_is_qwen3_vl,
    rerank_model_revision,
    rerank_pair_token_limit,
    rerank_template_reserve_tokens,
    rerank_timeout_seconds,
    rerank_tokenizer,
    rerank_tokenizer_revision,
)
from apps.documents.services.chunk_rerank_local_scored_http import (
    _complete_pairs,
    _rerank_scores,
    _score_single_pool,
)
from apps.documents.services.chunk_rerank_parse import (
    parse_rerank_results,
    parse_score_results,
)
from apps.documents.services.chunk_rerank_payload import rerank_document_payload
from apps.documents.services.chunk_rerank_results import (
    PassageScore,
    RerankScoreSet,
    ScoredRerankResult,
    fingerprint_pair,
    fingerprint_pool,
    fingerprint_text,
)
from apps.documents.services.chunk_rerank_score_cache import (
    reusable_scored_result,
    scored_result_cache_key,
    set_scored_result,
)


def rerank_via_local_vllm_scored(
    model_cls, query: str, chunks_list, top_k: int
) -> ScoredRerankResult:
    """Use one provider response per attempt; rank-only responses stay explicit."""
    chunks = list(chunks_list)
    candidate_ids = tuple(chunk.pk for chunk in chunks)
    query_fp = fingerprint_text(query)
    if not chunks:
        return ScoredRerankResult(
            (),
            RerankScoreSet("v2", query_fp, "", "", "rank_only", "unavailable", (), ()),
        )
    base_v1, model_name = rerank_base_url(), rerank_model()
    root = base_v1[:-3] if base_v1.endswith("/v1") else base_v1
    char_limit = rerank_doc_char_limit()
    pair_limit = rerank_pair_token_limit()
    reserve = rerank_template_reserve_tokens()
    pairs = [
        trim_rerank_pair(query, chunk.content[:char_limit], pair_limit, reserve)
        for chunk in chunks
    ]
    identities = tuple(
        (chunk.pk, fingerprint_text(chunk.content), fingerprint_pair(*pair))
        for chunk, pair in zip(chunks, pairs)
    )
    pool_fp = fingerprint_pool(identities)
    policy_fp = fingerprint_text(
        f"scored-v2:cl100k:{char_limit}:{pair_limit}:{reserve}"
    )
    capability = rag_cache.get_cached_rerank_capability(base_v1, model_name)
    pinned = bool(rerank_model_revision())

    def scorer_fp(endpoint, shape):
        return fingerprint_text(
            f"local:{endpoint}:{shape}:{model_name}:{rerank_model_revision()}:"
            f"{rerank_tokenizer()}:{rerank_tokenizer_revision()}:{policy_fp}"
        )

    def key(endpoint, shape):
        kind = "listwise" if shape == "rerank_documents" else "pointwise"
        return scored_result_cache_key(
            query_fingerprint=query_fp,
            scorer_fingerprint=scorer_fp(endpoint, shape),
            scoring_kind=kind,
            candidate_identities=identities,
            preparation_policy_fingerprint=policy_fp,
        )

    if pinned and capability:
        cached = reusable_scored_result(
            key=key(capability["endpoint"], capability["shape"]),
            chunks=chunks,
            canonical_identities=identities,
            query_fingerprint=query_fp,
            scorer_fingerprint=scorer_fp(capability["endpoint"], capability["shape"]),
            pool_fingerprint=pool_fp,
            top_k=top_k,
        )
        if cached is not None:
            return cached

    headers = {"Content-Type": "application/json"}
    api_key = rerank_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    timeout = rerank_timeout_seconds()

    def finish(ranked_ids, endpoint, shape, indexed=(), successful_pairs=None):
        kind = "listwise" if shape == "rerank_documents" else "pointwise"
        complete_order = (
            _complete_pairs(indexed, candidate_ids) if successful_pairs else None
        )
        scores = ()
        if complete_order is not None:
            values = {index: float(value) for index, value in indexed}
            try:
                scores = tuple(
                    PassageScore(
                        chunk.pk,
                        chunk.doc_id,
                        chunk.chunk_number,
                        fingerprint_text(chunk.content),
                        fingerprint_pair(*successful_pairs[index]),
                        values[index],
                    )
                    for index, chunk in enumerate(chunks)
                )
            except (KeyError, AttributeError):
                scores = ()
        status = "complete" if scores else "unavailable"
        result = ScoredRerankResult(
            tuple(complete_order[:top_k] if scores else ranked_ids),
            RerankScoreSet(
                "v2",
                query_fp,
                scorer_fp(endpoint, shape),
                pool_fp,
                kind if scores else "rank_only",
                status,
                candidate_ids,
                scores,
            ),
        )
        rag_cache.set_cached_rerank_capability(
            base_v1, model_name, {"endpoint": endpoint, "shape": shape}
        )
        if (
            pinned
            and scores
            and all(
                score.effective_pair_fingerprint == identities[index][2]
                for index, score in enumerate(scores)
            )
        ):
            set_scored_result(
                key(endpoint, shape),
                result.score_set,
                timeout_seconds=rag_cache.rerank_result_ttl(),
            )
        return result

    # Listwise input must have one shared query. Shape probing preserves the
    # existing adapter's text and multimodal variants.
    same_query = len({prepared_query for prepared_query, _ in pairs}) == 1
    rerank_endpoints = [f"{root}/rerank", f"{root}/v2/rerank", f"{base_v1}/rerank"]
    if capability and capability["shape"] == "rerank_documents":
        rerank_endpoints.insert(0, capability["endpoint"])
    if same_query and not rerank_model_is_qwen3_vl():
        documents = [document for _, document in pairs]
        payloads = [
            {
                "model": model_name,
                "query": pairs[0][0],
                "documents": documents,
                "top_n": len(chunks),
            },
            {
                "model": model_name,
                "query": pairs[0][0],
                "documents": [{"text": doc} for doc in documents],
                "top_n": len(chunks),
            },
            {"query": pairs[0][0], "documents": documents, "top_n": len(chunks)},
        ]
        multimodal = [rerank_document_payload(chunk) for chunk in chunks]
        if any(isinstance(item, list) for item in multimodal):
            prepared = []
            for item, document in zip(multimodal, documents):
                prepared.append(
                    [
                        {"type": "text", "text": document}
                        if part.get("type") == "text"
                        else part
                        for part in item
                    ]
                    if isinstance(item, list)
                    else document
                )
            payloads.append(
                {
                    "model": model_name,
                    "query": pairs[0][0],
                    "documents": prepared,
                    "top_n": len(chunks),
                }
            )
        for endpoint in dict.fromkeys(rerank_endpoints):
            for payload in payloads:
                try:
                    response = requests.post(
                        endpoint, headers=headers, json=payload, timeout=timeout
                    )
                    if response.status_code >= 400:
                        continue
                    body = response.json()
                    ranked = parse_rerank_results(body, chunks)
                    if ranked:
                        effective = (
                            dict(enumerate(pairs))
                            if payload is not payloads[-1]
                            or not any(isinstance(item, list) for item in multimodal)
                            else None
                        )
                        return finish(
                            ranked,
                            endpoint,
                            "rerank_documents",
                            _rerank_scores(body),
                            effective,
                        )
                except Exception:
                    continue

    score_endpoints = [f"{root}/score", f"{base_v1}/score"]
    if capability and capability["shape"] in (
        "score_batch_text_pairs",
        "score_single_text_pair",
    ):
        score_endpoints.insert(0, capability["endpoint"])
    for endpoint in dict.fromkeys(score_endpoints):
        if same_query and not rerank_model_is_qwen3_vl():
            documents = [document for _, document in pairs]
            payloads = [
                {"model": model_name, "text_1": pairs[0][0], "text_2": documents},
                {"text_1": pairs[0][0], "text_2": documents},
                {"model": model_name, "query": pairs[0][0], "documents": documents},
            ]
            for payload in payloads:
                try:
                    response = requests.post(
                        endpoint, headers=headers, json=payload, timeout=timeout
                    )
                    if response.status_code >= 400:
                        continue
                    indexed = parse_score_results(response.json())
                    ranked = _complete_pairs(indexed, candidate_ids)
                    if ranked is not None:
                        return finish(
                            ranked[:top_k],
                            endpoint,
                            "score_batch_text_pairs",
                            indexed,
                            dict(enumerate(pairs)),
                        )
                except Exception:
                    continue
        results = _score_single_pool(
            endpoint, pairs, headers, timeout, model_name, pair_limit, reserve
        )
        indexed = [(index, value) for index, value, _pair in results]
        ranked = _complete_pairs(indexed, candidate_ids)
        if ranked is not None:
            return finish(
                ranked[:top_k],
                endpoint,
                "score_single_text_pair",
                indexed,
                {index: pair for index, _value, pair in results},
            )
    return ScoredRerankResult(
        (),
        RerankScoreSet(
            "v2",
            query_fp,
            scorer_fp(base_v1, "unavailable"),
            pool_fp,
            "rank_only",
            "unavailable",
            candidate_ids,
            (),
        ),
    )
