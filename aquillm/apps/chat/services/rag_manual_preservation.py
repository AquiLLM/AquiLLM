"""Explicit manual scope shares the original acquisition/selection ledger."""

from types import SimpleNamespace

from channels.db import database_sync_to_async

from apps.chat.services.manual_search_commands import (
    ManualSearchError,
    document_uuid,
    resolve_search_document,
    resolve_search_query,
)
from apps.chat.services.rag_config import direct_rag_candidate_top_k
from apps.chat.services.rag_pipeline_messages import _append_retrieval_messages
from apps.chat.services.rag_preservation_turn import finish_preservation
from apps.chat.services.rag_source_synthesis import LIMITED_MESSAGE
from apps.chat.services.tool_wiring.source_documents import (
    explicit_document_source_scope,
)
from apps.documents.services.source_loading import (
    current_source_runtime,
    source_runtime_scope,
)
from lib.llm.turn_context import bounded_retrieval, check_turn_active
from lib.llm.types.messages import AssistantMessage


def _manual_acquire(consumer, command, query, history):
    from apps.chat.services.manual_search_turn import load_search_documents
    from apps.chat.services.tool_wiring.documents import (
        search_single_document_tool,
        vector_search_tool,
    )

    if command.command == "collection":
        if not consumer.col_ref.collections:
            raise ManualSearchError(
                "Select one or more collections before using /collection."
            )
        raw = dict(
            vector_search_tool(consumer.user, consumer.col_ref)(
                search_string=query, top_k=direct_rag_candidate_top_k()
            )
        )
        return raw, current_source_runtime()
    documents = (
        [] if document_uuid(command.document) else load_search_documents(consumer)
    )
    document_id = resolve_search_document(command.document, history, documents)
    with explicit_document_source_scope(consumer.user, document_id) as child:
        raw = dict(
            search_single_document_tool(consumer.user, consumer.col_ref)(
                doc_id=document_id,
                search_string=query,
                top_k=direct_rag_candidate_top_k(),
            )
        )
        return raw, child


async def run_manual_preservation(consumer, llm, convo, command, *, stream_func):
    runtime = current_source_runtime()
    try:
        if runtime is None:
            raise ValueError("missing manual ledger")
        query = resolve_search_query(command, convo.messages[:-1])
        raw, scope = await bounded_retrieval(
            database_sync_to_async(_manual_acquire, thread_sensitive=False)(
                consumer, command, query, convo.messages[:-1]
            ),
            runtime.budget,
        )
        if raw.get("exception"):
            raise ManualSearchError(
                "I couldn't search the requested documents. Check your "
                "document selection and access."
            )
        selected = getattr(
            scope.authorization, "selected_collection_ids", consumer.col_ref.collections
        )
        view = SimpleNamespace(
            user=consumer.user, col_ref=SimpleNamespace(collections=list(selected))
        )
        working = _append_retrieval_messages(
            convo, query, raw, direct_rag_candidate_top_k()
        )
        with source_runtime_scope(scope):
            result = await finish_preservation(
                view,
                llm,
                working,
                question=query,
                initial_results=(raw,),
                initial_acquired=True,
                iterative=False,
                stream_func=stream_func,
            )
    except ManualSearchError as exc:
        result = convo + [AssistantMessage(content=str(exc), stop_reason="end_turn")]
    except Exception:
        result = convo + [
            AssistantMessage(content=LIMITED_MESSAGE, stop_reason="end_turn")
        ]
    check_turn_active()
    consumer.convo = result
    return "handled"
