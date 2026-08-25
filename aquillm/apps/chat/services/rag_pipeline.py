"""Direct RAG pipeline orchestration (backend-driven, deterministic).

When ``RAG_DIRECT_ENABLED`` is on and the latest user turn is an obvious document
question, this path retrieves evidence *before* asking the model anything. It runs
retrieval directly (no LLM tool-selection round trip), packages the evidence, and
then hands a post-tool conversation to :mod:`rag_synthesis` for the final answer.

Failures fail open: any retrieval/synthesis exception returns ``"skipped"`` with
``consumer.convo`` untouched so the normal tool loop can still run.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Literal

import structlog
from channels.db import database_sync_to_async

from apps.chat.services.rag_config import (
    direct_rag_max_queries,
    direct_rag_top_k,
    is_direct_rag_enabled,
)
from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.services.rag_intent import classify_chat_message
from apps.chat.services.rag_metrics import log_direct_rag_turn
from apps.chat.services.rag_query import build_retrieval_queries
from apps.chat.services.rag_retrieval import merge_ranked_tool_results
from apps.chat.services.rag_synthesis import synthesize_from_evidence
from apps.chat.services.tool_wiring.documents import vector_search_tool
from lib.llm.providers import image_context as imgctx
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage

logger = structlog.stdlib.get_logger(__name__)

_SEARCH_SCOPE = "selected documents"
_SELECT_COLLECTIONS_MESSAGE = (
    "I can search your documents, but no collections are selected for this chat. "
    "Please select one or more collections in the collection picker and ask again."
)

DirectRagOutcome = Literal["handled", "skipped"]


def _latest_user_message(convo: Conversation) -> UserMessage | None:
    if len(convo) == 0:
        return None
    last = convo[-1]
    return last if isinstance(last, UserMessage) else None


def _run_vector_search(consumer: Any, query: str, top_k: int) -> dict:
    """Execute vector_search synchronously via the existing tool factory.

    Kept as a module-level function so it runs on a sync DB thread and can be
    monkeypatched in tests.
    """
    tool = vector_search_tool(consumer.user, consumer.col_ref)
    return dict(tool(search_string=query, top_k=top_k))


def _append_retrieval_messages(
    convo: Conversation, query: str, raw_result: dict, top_k: int
) -> Conversation:
    """Append a synthetic tool-call + tool-result so synthesis sees a post-tool turn."""
    arguments = {"search_string": query, "top_k": top_k}
    assistant_tool_call = AssistantMessage(
        content="",
        stop_reason="tool_use",
        tool_call_id=str(uuid.uuid4()),
        tool_call_name="vector_search",
        tool_call_input=arguments,
    )
    tool_message = ToolMessage(
        tool_name="vector_search",
        for_whom="assistant",
        content=imgctx.serialize_tool_result_for_llm(raw_result),
        arguments=arguments,
        result_dict=raw_result,
    )
    return convo + [assistant_tool_call, tool_message]


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
    if not is_direct_rag_enabled():
        return "skipped"

    user_message = _latest_user_message(convo)
    if user_message is None:
        return "skipped"

    t_start = time.perf_counter()

    collection_ids = list(getattr(consumer.col_ref, "collections", []) or [])

    t_intent_start = time.perf_counter()
    intent = classify_chat_message(
        user_message.content or "", selected_collection_ids=collection_ids
    )
    t_intent_end = time.perf_counter()

    if not intent.requires_rag or intent.requires_local_tools or intent.is_retry:
        return "skipped"

    if not collection_ids:
        consumer.convo = convo + [
            AssistantMessage(content=_SELECT_COLLECTIONS_MESSAGE, stop_reason="end_turn")
        ]
        return "handled"

    try:
        t_query_start = time.perf_counter()
        queries = build_retrieval_queries(
            convo,
            user_message.content or "",
            max_queries=direct_rag_max_queries(),
        )
        query = queries[0]
        t_query_end = time.perf_counter()

        top_k = direct_rag_top_k()

        t_retrieval_start = time.perf_counter()
        search_async = database_sync_to_async(
            _run_vector_search,
            thread_sensitive=False,
        )
        search_outcomes = await asyncio.gather(
            *(search_async(consumer, search_query, top_k) for search_query in queries),
            return_exceptions=True,
        )
        search_results = [
            outcome for outcome in search_outcomes if isinstance(outcome, dict)
        ]
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
        failed_query_count = len(search_outcomes) - len(search_results)
        if failed_query_count:
            logger.warning(
                "direct_rag_partial_retrieval_failure failed=%d total=%d",
                failed_query_count,
                len(search_outcomes),
            )
        raw_result = merge_ranked_tool_results(search_results, limit=top_k)
        t_retrieval_end = time.perf_counter()
        retrieval_diagnostics = raw_result.get("_retrieval_diagnostics")
        if not isinstance(retrieval_diagnostics, dict):
            retrieval_diagnostics = {}

        t_evidence_start = time.perf_counter()
        packet = build_evidence_packet(
            raw_result, query=query, search_scope=_SEARCH_SCOPE
        )
        t_evidence_end = time.perf_counter()

        working_convo = _append_retrieval_messages(convo, query, raw_result, top_k)

        t_synthesis_start = time.perf_counter()
        result_convo = await synthesize_from_evidence(
            llm_if, working_convo, packet, stream_func=stream_func
        )
        t_synthesis_end = time.perf_counter()

        consumer.convo = result_convo

        _ms = lambda a, b: (b - a) * 1000.0  # noqa: E731
        log_direct_rag_turn(
            intent_ms=_ms(t_intent_start, t_intent_end),
            query_ms=_ms(t_query_start, t_query_end),
            retrieval_ms=_ms(t_retrieval_start, t_retrieval_end),
            evidence_ms=_ms(t_evidence_start, t_evidence_end),
            synthesis_ms=_ms(t_synthesis_start, t_synthesis_end),
            total_ms=_ms(t_start, t_synthesis_end),
            retrieved_count=int(raw_result.get("retrieved_count", 0) or 0),
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
        )
        logger.info(
            "direct_rag_turn_handled retrieved=%d retained=%d status=%s",
            int(raw_result.get("retrieved_count", 0) or 0),
            len(packet.chunks),
            packet.retrieval_status,
        )
        return "handled"
    except Exception:
        logger.exception("direct_rag_turn_failed; falling back to tool loop")
        return "skipped"


__all__ = ["DirectRagOutcome", "run_direct_rag_turn"]
