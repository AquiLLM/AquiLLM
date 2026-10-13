"""Bound local scored requests to one verified protocol and the caller's ledger."""

import requests as requests  # Compatibility patch surface for provider tests.

from . import rag_cache
from .chunk_rerank_budget import trim_rerank_documents, trim_rerank_pair
from .chunk_rerank_config import (
    rerank_base_url,
    rerank_doc_char_limit,
    rerank_headers,
    rerank_model,
    rerank_model_revision,
    rerank_pair_token_limit,
    rerank_score_concurrency,
    rerank_template_reserve_tokens,
    rerank_timeout_seconds,
    rerank_tokenizer,
    rerank_tokenizer_revision,
)
from .chunk_rerank_local_scored_http import score_bound_pairs
from .chunk_rerank_results import (
    PassageScore,
    RerankScoreSet,
    ScoredRerankResult,
    fingerprint_pair,
    fingerprint_pool,
    fingerprint_text,
    rank_only_result_for_chunks,
)
from .chunk_rerank_score_cache import (
    reusable_scored_result,
    scored_result_cache_key,
    set_scored_result,
)
from .rerank_request_lifetime import rerank_state_for
from .rerank_transport_policy import RerankProtocolV1, resolve_rerank_protocol


def rerank_via_local_vllm_scored(
    model_cls,
    query,
    chunks_list,
    top_k,
    *,
    turn_budget=None,
    window_scorer=None,
    protocol_binding=None,
    rerank_state=None,
    _preparation=None,
):
    chunks = list(chunks_list)
    state = rerank_state_for(turn_budget, rerank_state)
    turn_budget = state.budget

    def unavailable():
        return rank_only_result_for_chunks(query, chunks, top_k=top_k)

    if not chunks or not state.can_admit():
        return unavailable()
    from .chunk_rerank_window_acquisition import dispatch_windowed

    windowed = dispatch_windowed(query, chunks, top_k, turn_budget, window_scorer)
    if windowed is not None:
        return windowed
    base, model_name = rerank_base_url(), rerank_model()
    char_limit, pair_limit, reserve = (
        rerank_doc_char_limit(),
        rerank_pair_token_limit(),
        rerank_template_reserve_tokens(),
    )
    if _preparation:
        base, model_name, char_limit, pair_limit, reserve = _preparation
    binding = protocol_binding or resolve_rerank_protocol(
        base, model_name, budget=turn_budget
    )
    if not isinstance(binding, RerankProtocolV1) or not binding.live(model_name):
        return unavailable()
    documents = [chunk.content[:char_limit] for chunk in chunks]
    if not turn_budget.reserve_text(
        sum(len(query) + len(doc) for doc in documents), kind="tokenized"
    ):
        return unavailable()
    if binding.payload_kind == "rerank_documents":
        shared_query, documents = trim_rerank_documents(
            query, documents, pair_limit, reserve
        )
        pairs = tuple((shared_query, doc) for doc in documents)
    else:
        pairs = tuple(
            trim_rerank_pair(query, doc, pair_limit, reserve) for doc in documents
        )
    identities = tuple(
        (chunk.pk, fingerprint_text(chunk.content), fingerprint_pair(*pair))
        for chunk, pair in zip(chunks, pairs)
    )
    query_fp, pool_fp = fingerprint_text(query), fingerprint_pool(identities)
    policy_fp = fingerprint_text(
        f"scored-v2:cl100k:{char_limit}:{pair_limit}:{reserve}"
    )
    scorer_fp = fingerprint_text(
        f"local:{binding.endpoint}:{binding.payload_kind}:{model_name}:"
        f"{rerank_model_revision()}:{rerank_tokenizer()}:{rerank_tokenizer_revision()}:{policy_fp}"
    )
    kind = "listwise" if binding.payload_kind == "rerank_documents" else "pointwise"
    key = scored_result_cache_key(
        query_fingerprint=query_fp,
        scorer_fingerprint=scorer_fp,
        scoring_kind=kind,
        candidate_identities=identities,
        preparation_policy_fingerprint=policy_fp,
    )
    if rerank_model_revision():
        cached = reusable_scored_result(
            key=key,
            chunks=chunks,
            canonical_identities=identities,
            query_fingerprint=query_fp,
            scorer_fingerprint=scorer_fp,
            pool_fingerprint=pool_fp,
            top_k=top_k,
        )
        if cached is not None and state.can_admit():
            return cached
    values = score_bound_pairs(
        binding,
        pairs,
        state=state,
        headers=rerank_headers(),
        timeout=rerank_timeout_seconds(),
        pair_limit=pair_limit,
        max_workers=rerank_score_concurrency(),
    )
    if values is None or not state.can_admit():
        return unavailable()
    candidate_ids = tuple(chunk.pk for chunk in chunks)
    scores = tuple(
        PassageScore(
            chunk.pk,
            chunk.doc_id,
            chunk.chunk_number,
            fingerprint_text(chunk.content),
            fingerprint_pair(*pair),
            value,
        )
        for chunk, pair, value in zip(chunks, pairs, values)
    )
    ranked = tuple(
        candidate_ids[i]
        for i in sorted(range(len(chunks)), key=lambda i: (-values[i], i))
    )
    result = ScoredRerankResult(
        ranked[:top_k],
        RerankScoreSet(
            "v2", query_fp, scorer_fp, pool_fp, kind, "complete", candidate_ids, scores
        ),
    )
    if rerank_model_revision():
        state.publish(
            lambda: set_scored_result(
                key, result.score_set, timeout_seconds=rag_cache.rerank_result_ttl()
            )
        )
    return result if state.can_admit() else unavailable()
