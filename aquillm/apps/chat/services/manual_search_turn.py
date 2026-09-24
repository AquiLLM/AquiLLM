"""Execute manual search commands through the existing authorized retrieval tools."""

from __future__ import annotations

from uuid import uuid4

import structlog
from channels.db import database_sync_to_async

from apps.chat.services.manual_search_commands import (
    ManualSearchError,
    document_uuid,
    parse_manual_search,
    resolve_search_document,
    resolve_search_query,
)
from apps.chat.services.rag_config import direct_rag_top_k
from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.services.rag_synthesis import synthesize_from_evidence
from apps.chat.services.tool_wiring.documents import (
    search_single_document_tool,
    vector_search_tool,
)
from apps.collections.models import Collection
from apps.collections.models import collection as collection_models
from lib.llm.providers.image_context import serialize_tool_result_for_llm
from lib.llm.providers.request_observability import (
    new_correlation_id,
    observability_scope,
)
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage

logger = structlog.stdlib.get_logger(__name__)


def load_search_documents(consumer):
    """List only documents the user can view in the currently selected collections."""
    collections = Collection.objects.filter(
        id__in=consumer.col_ref.collections
    ).filter_by_user_perm(consumer.user, "VIEW")
    return [
        doc
        for model in collection_models._get_document_types()
        for doc in model.objects.filter(collection__in=collections).only("id", "title")
    ]


def _retrieve(consumer, command, query, prior_messages):
    arguments = {"search_string": query, "top_k": direct_rag_top_k()}
    if command.command == "collection":
        if not consumer.col_ref.collections:
            raise ManualSearchError(
                "Select one or more collections before using /collection."
            )
        name, scope = "vector_search", "selected documents"
        tool = vector_search_tool(consumer.user, consumer.col_ref)
    else:
        # Exact IDs are authorized by the tool even when no collection is selected.
        documents = (
            [] if document_uuid(command.document) else load_search_documents(consumer)
        )
        arguments["doc_id"] = resolve_search_document(
            command.document, prior_messages, documents
        )
        name, scope = "search_single_document", "requested document"
        tool = search_single_document_tool(consumer.user, consumer.col_ref)
    raw_result = dict(tool(**arguments))
    if raw_result.get("exception"):
        raise ManualSearchError(
            "I couldn't search the requested documents. Check your "
            "document or "
            "collection selection and access, then try again."
        )
    return name, scope, arguments, raw_result


def _reply(consumer, convo, text):
    consumer.convo = convo + [AssistantMessage(content=text, stop_reason="end_turn")]
    return "handled"


async def run_manual_search_turn(consumer, llm_if, convo, *, stream_func=None):
    """Handle explicit commands even on errors or when automatic RAG is disabled."""
    if not convo.messages or not isinstance(convo[-1], UserMessage):
        return "skipped"
    try:
        command = parse_manual_search(convo[-1].content or "")
    except ManualSearchError as exc:
        return _reply(consumer, convo, str(exc))
    if command is None:
        return "skipped"

    from apps.chat.services.rag_config import rag_preservation_config
    from apps.chat.services.rag_manual_preservation import run_manual_preservation

    if rag_preservation_config().active:
        return await run_manual_preservation(
            consumer, llm_if, convo, command, stream_func=stream_func
        )

    working_convo = convo
    correlation_id = new_correlation_id()
    try:
        query = resolve_search_query(command, convo.messages[:-1])
        name, scope, arguments, raw = await database_sync_to_async(
            _retrieve,
            thread_sensitive=False,
        )(consumer, command, query, convo.messages[:-1])
        working_convo = convo + [
            AssistantMessage(
                content="",
                stop_reason="tool_use",
                tool_call_id=str(uuid4()),
                tool_call_name=name,
                tool_call_input=arguments,
            ),
            ToolMessage(
                tool_name=name,
                for_whom="assistant",
                arguments=arguments,
                content=serialize_tool_result_for_llm(raw),
                result_dict=raw,
            ),
        ]
        packet = build_evidence_packet(raw, query=query, search_scope=scope)
        with observability_scope(correlation_id, "direct_synthesis"):
            consumer.convo = await synthesize_from_evidence(
                llm_if,
                working_convo,
                packet,
                stream_func=stream_func,
            )
        return "handled"
    except ManualSearchError as exc:
        return _reply(consumer, working_convo, str(exc))
    except Exception as exc:
        logger.warning(
            "obs.rag.manual_search_failed",
            correlation_id=correlation_id,
            command=command.command,
            error_type=type(exc).__name__,
        )
        return _reply(
            consumer,
            working_convo,
            "I couldn't complete that search. Please try the command again.",
        )
