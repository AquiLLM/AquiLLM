"""Direct RAG pipeline orchestration (backend-driven, deterministic).

When ``RAG_DIRECT_ENABLED`` is on and the latest user turn is an obvious document
question, this path retrieves evidence *before* asking the model anything. It runs
retrieval directly (no LLM tool-selection round trip), packages the evidence, and
then hands a post-tool conversation to :mod:`rag_synthesis` for the final answer.

Failures fail open: any retrieval/synthesis exception returns ``"skipped"`` with
``consumer.convo`` untouched so the normal tool loop can still run.
"""
from __future__ import annotations

import uuid
from typing import Any, Literal

import structlog
from asgiref.sync import sync_to_async

from lib.llm.providers import image_context as imgctx
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage

from apps.chat.services.rag_config import direct_rag_top_k, is_direct_rag_enabled
from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.services.rag_intent import classify_chat_message
from apps.chat.services.rag_query import build_retrieval_query
from apps.chat.services.rag_synthesis import synthesize_from_evidence
from apps.chat.services.tool_wiring.documents import vector_search_tool

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

    collection_ids = list(getattr(consumer.col_ref, "collections", []) or [])
    intent = classify_chat_message(
        user_message.content or "", selected_collection_ids=collection_ids
    )
    if not intent.requires_rag or intent.requires_local_tools or intent.is_retry:
        return "skipped"

    if not collection_ids:
        consumer.convo = convo + [
            AssistantMessage(content=_SELECT_COLLECTIONS_MESSAGE, stop_reason="end_turn")
        ]
        return "handled"

    try:
        query = build_retrieval_query(convo, user_message.content or "")
        top_k = direct_rag_top_k()
        raw_result = await sync_to_async(_run_vector_search, thread_sensitive=True)(
            consumer, query, top_k
        )
        packet = build_evidence_packet(
            raw_result, query=query, search_scope=_SEARCH_SCOPE
        )
        working_convo = _append_retrieval_messages(convo, query, raw_result, top_k)
        result_convo = await synthesize_from_evidence(
            llm_if, working_convo, packet, stream_func=stream_func
        )
        consumer.convo = result_convo
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
