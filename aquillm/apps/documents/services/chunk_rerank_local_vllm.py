"""Legacy queryset adapter over the same bounded, verified local transport."""

import requests as requests  # Compatibility patch surface for provider tests.
import structlog

from lib.retrieval_redaction import RetrievalLogReason, retrieval_log_fields

from . import chunk_rerank_pointwise as pointwise
from . import rag_cache
from .chunk_rerank_budget import trim_rerank_pair
from .chunk_rerank_config import (
    rerank_base_url,
    rerank_doc_char_limit,
    rerank_model,
    rerank_pair_token_limit,
    rerank_template_reserve_tokens,
)
from .chunk_rerank_config import rerank_model_is_qwen3_vl as rerank_model_is_qwen3_vl
from .chunk_rerank_local_scored import rerank_via_local_vllm_scored
from .chunk_rerank_parse import ordered_queryset_from_ids, parse_single_score
from .chunk_rerank_parse import parse_rerank_results as parse_rerank_results
from .chunk_rerank_payload import rerank_document_payload as rerank_document_payload
from .rag_cache_rerank import get_scoped_rerank_result, set_scoped_rerank_result
from .rerank_request_lifetime import rerank_state_for

logger = structlog.stdlib.get_logger(__name__)


class _StrictCompleteScoringCapability:
    __slots__ = ()


_STRICT_COMPLETE_SCORING = _StrictCompleteScoringCapability()
_is_complete_finite_scoring = pointwise._is_complete_finite_scoring


def _score_one_document(**kwargs):
    return pointwise._score_one_document(
        **kwargs, parse_score=parse_single_score, trim_pair=trim_rerank_pair
    )


def _score_documents_concurrently(**kwargs):
    return pointwise._score_documents_concurrently(
        **kwargs, score_one=_score_one_document
    )


def rerank_via_local_vllm(
    model_cls,
    query,
    chunks_list,
    top_k,
    *,
    _complete_scoring_capability=None,
    turn_budget=None,
    protocol_binding=None,
    rerank_state=None,
):
    if (
        _complete_scoring_capability is not None
        and _complete_scoring_capability is not _STRICT_COMPLETE_SCORING
    ):
        raise PermissionError("complete local scoring requires its private capability")
    strict = _complete_scoring_capability is _STRICT_COMPLETE_SCORING
    state = rerank_state_for(turn_budget, rerank_state)
    chunks = list(chunks_list)
    model_name = rerank_model()
    candidate_ids = tuple(chunk.pk for chunk in chunks)
    signature = rag_cache.query_signature_for_rerank(query)
    if not strict and state.can_admit():
        cached = get_scoped_rerank_result(signature, candidate_ids, top_k, model_name)
        if cached and state.can_admit():
            logger.info(
                "obs.rag.rerank_cache_hit",
                **retrieval_log_fields(
                    reason=RetrievalLogReason.COMPLETED, count=0, elapsed_ms=0.0
                ),
            )
            return ordered_queryset_from_ids(model_cls, cached)
    result = rerank_via_local_vllm_scored(
        model_cls,
        query,
        chunks,
        top_k,
        turn_budget=state.budget,
        protocol_binding=protocol_binding,
        rerank_state=state,
        _preparation=(
            rerank_base_url(),
            model_name,
            rerank_doc_char_limit(),
            rerank_pair_token_limit(),
            rerank_template_reserve_tokens(),
        ),
    )
    if strict and result.score_set.status != "complete":
        return ()
    if result.score_set.status == "complete" and not strict:
        state.publish(
            lambda: set_scoped_rerank_result(
                signature, candidate_ids, top_k, model_name, list(result.ranked_ids)
            )
        )
    return ordered_queryset_from_ids(model_cls, result.ranked_ids)
