"""Direct RAG retrieval, selection, and synthesis orchestration."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

import structlog
from channels.db import database_sync_to_async

from apps.chat.services.manual_search_turn import run_manual_search_turn
from apps.chat.services.rag_config import (
    direct_rag_candidate_top_k,
    direct_rag_max_queries,
    direct_rag_top_k,
    evidence_selection_config,
    is_direct_rag_enabled,
    rag_preservation_config,
)
from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.services.rag_intent import classify_chat_message
from apps.chat.services.rag_metrics import log_direct_rag_turn
from apps.chat.services.rag_pipeline_messages import (
    _append_retrieval_messages,
    _has_prior_vector_search,
    _latest_user_message,
)
from apps.chat.services.rag_query import build_retrieval_queries
from apps.chat.services.rag_retrieval import merge_ranked_tool_results
from apps.chat.services.rag_selection_coordinator import (
    coordinate_selection,
    selection_metric_fields,
)
from apps.chat.services.rag_selection_coordinator import (
    prepare_selection_turn as _prepare_selection_sync,
)
from apps.chat.services.rag_selection_coordinator import (
    revalidate_selection_turn as _revalidate_selection_sync,
)
from apps.chat.services.rag_selection_coordinator import (
    selection_question as _selection_question,
)
from apps.chat.services.rag_source_continuity_turn import continuity_candidates
from apps.chat.services.rag_source_synthesis import LIMITED_MESSAGE
from apps.chat.services.rag_synthesis import synthesize_from_evidence
from apps.chat.services.tool_wiring.documents import vector_search_tool
from apps.documents.services.source_loading import current_source_runtime
from lib.llm.providers.request_observability import (
    new_correlation_id,
    observability_scope,
)
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage

logger = structlog.stdlib.get_logger(__name__)
_SEARCH_SCOPE = "selected documents"
_SELECT_COLLECTIONS_MESSAGE = (
    "I can search your documents, but no collections are selected for this chat. "
    "Please select one or more collections in the collection picker and ask again."
)

DirectRagOutcome = Literal["handled", "skipped"]


def _run_vector_search(consumer: Any, query: str, top_k: int) -> dict:
    """Execute vector_search synchronously via the existing tool factory.

    Kept as a module-level function so it runs on a sync DB thread and can be
    monkeypatched in tests.
    """
    tool = vector_search_tool(consumer.user, consumer.col_ref)
    return dict(tool(search_string=query, top_k=top_k))


async def run_direct_rag_turn(
    consumer: Any,
    llm_if: Any,
    convo: Conversation,
    *,
    stream_func: Any = None,
) -> DirectRagOutcome:
    """Handle an obvious document question through the deterministic RAG pipeline.

    Returns ``"handled"`` when the turn was fully answered here (caller must skip
    the normal tool loop), or ``"skipped"`` to let the existing spin run.
    """
    manual_outcome = await run_manual_search_turn(
        consumer,
        llm_if,
        convo,
        stream_func=stream_func,
    )
    if manual_outcome == "handled":
        return "handled"
    if not is_direct_rag_enabled():
        return "skipped"

    user_message = _latest_user_message(convo)
    if user_message is None:
        return "skipped"

    t_start = time.perf_counter()
    correlation_id = new_correlation_id()

    collection_ids = list(getattr(consumer.col_ref, "collections", []) or [])

    t_intent_start = time.perf_counter()
    prior_vector_search = _has_prior_vector_search(convo)
    intent = classify_chat_message(
        user_message.content or "",
        selected_collection_ids=collection_ids,
        prior_tools=["vector_search"] if prior_vector_search else None,
    )
    t_intent_end = time.perf_counter()

    if not intent.requires_rag or intent.requires_local_tools:
        return "skipped"

    if not collection_ids:
        consumer.convo = convo + [
            AssistantMessage(
                content=_SELECT_COLLECTIONS_MESSAGE, stop_reason="end_turn"
            )
        ]
        return "handled"

    preservation = rag_preservation_config()
    if preservation.error or (
        preservation.evidence_text_mode == "source" and current_source_runtime() is None
    ):
        consumer.convo = convo + [
            AssistantMessage(content=LIMITED_MESSAGE, stop_reason="end_turn")
        ]
        return "handled"
    try:
        t_query_start = time.perf_counter()
        continuity_result, continuity_notice = await continuity_candidates(
            convo,
            user_message.content or "",
            user=consumer.user,
            selected_scope=collection_ids,
            enabled=preservation.followup_evidence_enabled,
        )
        if continuity_notice:
            consumer.convo = convo + [
                AssistantMessage(content=continuity_notice, stop_reason="end_turn")
            ]
            return "handled"
        queries = build_retrieval_queries(
            convo,
            user_message.content or "",
            max_queries=direct_rag_max_queries(),
        )
        query = queries[0]
        t_query_end = time.perf_counter()

        top_k = direct_rag_top_k()
        candidate_top_k = direct_rag_candidate_top_k()
        selection_config = evidence_selection_config()

        t_retrieval_start = time.perf_counter()
        search_async = database_sync_to_async(
            _run_vector_search,
            thread_sensitive=False,
        )
        search_outcomes = await asyncio.gather(
            *(
                search_async(consumer, search_query, candidate_top_k)
                for search_query in queries
            ),
            return_exceptions=True,
        )
        search_results = [
            outcome for outcome in search_outcomes if isinstance(outcome, dict)
        ]
        failed_query_count = len(search_outcomes) - len(search_results)
        if continuity_result is not None:
            search_results.insert(0, continuity_result)
        if not search_results:
            first_error = next(
                (
                    outcome
                    for outcome in search_outcomes
                    if isinstance(outcome, BaseException)
                ),
                RuntimeError("all direct-RAG retrieval queries failed"),
            )
            raise first_error
        if failed_query_count:
            logger.warning(
                "obs.rag.partial_retrieval_failure",
                failed_count=failed_query_count,
                total_count=len(search_outcomes),
            )
        raw_result = (
            {}
            if preservation.active
            else merge_ranked_tool_results(search_results, limit=top_k)
        )
        t_retrieval_end = time.perf_counter()
        retrieval_diagnostics = raw_result.get("_retrieval_diagnostics")
        if not isinstance(retrieval_diagnostics, dict):
            retrieval_diagnostics = {}

        t_evidence_start = time.perf_counter()
        packet = (
            None
            if preservation.active
            else build_evidence_packet(
                raw_result, query=query, search_scope=_SEARCH_SCOPE
            )
        )
        selection_turn = None
        if preservation.active or selection_config.mode in ("shadow", "adaptive"):
            (
                selection_turn,
                selected_packet,
                selected_result,
            ) = await coordinate_selection(
                consumer,
                search_results,
                query,
                _selection_question(convo, user_message.content or ""),
                selection_config,
                top_k,
                prepare_fn=_prepare_selection_sync,
                revalidate_fn=_revalidate_selection_sync,
                request_conversation=convo,
                llm_if=llm_if,
            )
            if selected_packet is not None:
                packet, raw_result = selected_packet, selected_result
        t_evidence_end = time.perf_counter()

        working_convo = _append_retrieval_messages(convo, query, raw_result, top_k)

        t_synthesis_start = time.perf_counter()
        with observability_scope(correlation_id, "direct_synthesis"):
            result_convo = await synthesize_from_evidence(
                llm_if, working_convo, packet, stream_func=stream_func
            )
        t_synthesis_end = time.perf_counter()

        t_persistence_start = time.perf_counter()
        consumer.convo = result_convo
        t_persistence_end = time.perf_counter()

        _ms = lambda a, b: (b - a) * 1000.0  # noqa: E731
        log_direct_rag_turn(
            correlation_id=correlation_id,
            intent_ms=_ms(t_intent_start, t_intent_end),
            query_ms=_ms(t_query_start, t_query_end),
            retrieval_ms=_ms(t_retrieval_start, t_retrieval_end),
            evidence_ms=_ms(t_evidence_start, t_evidence_end),
            synthesis_ms=_ms(t_synthesis_start, t_synthesis_end),
            persistence_ms=_ms(t_persistence_start, t_persistence_end),
            total_ms=_ms(t_start, t_persistence_end),
            retrieved_count=int(raw_result.get("retrieved_count", 0) or 0),
            retained_count=len(packet.chunks),
            retrieval_status=packet.retrieval_status,
            graph_ms=retrieval_diagnostics.get("graph_ms"),
            graph_seed_count=retrieval_diagnostics.get("graph_seed_count"),
            graph_candidate_count=retrieval_diagnostics.get("graph_candidate_count"),
            graph_status=retrieval_diagnostics.get("graph_status"),
            graph_algorithm_signature=retrieval_diagnostics.get(
                "graph_algorithm_signature"
            ),
            graph_version_signature=retrieval_diagnostics.get(
                "graph_version_signature"
            ),
            **selection_metric_fields(selection_config, selection_turn, packet),
        )
        logger.info(
            "obs.rag.direct_turn_handled",
            correlation_id=correlation_id,
            retrieved_count=int(raw_result.get("retrieved_count", 0) or 0),
            retained_count=len(packet.chunks),
            retrieval_status=packet.retrieval_status,
        )
        return "handled"
    except Exception as exc:
        logger.warning(
            "obs.rag.direct_turn_failed",
            correlation_id=correlation_id,
            error_type=type(exc).__name__,
        )
        if preservation.active:
            consumer.convo = convo + [
                AssistantMessage(content=LIMITED_MESSAGE, stop_reason="end_turn")
            ]
            return "handled"
        return "skipped"


__all__ = ["DirectRagOutcome", "run_direct_rag_turn"]
