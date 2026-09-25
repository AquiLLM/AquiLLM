"""One action admission boundary shared by direct, manual and model tools."""

from apps.documents.services.source_loading import current_source_runtime
from lib.llm.turn_context import current_turn

from .rag_coverage import AcquisitionAction

RETRIEVAL_TOOLS = {
    "vector_search",
    "search_single_document",
    "more_context",
    "whole_document",
}


def admit_tool_action(kind, query="", document_id=None, chunk_id=None):
    runtime = current_source_runtime()
    if runtime is None:
        return True
    runtime.budget.check_active()
    action = AcquisitionAction(
        kind, query, str(document_id) if document_id else None, chunk_id
    )
    return runtime.budget.reserve_action(action.signature)


def limited_action_result():
    return {
        "result": [],
        "retrieval_status": "context_limited",
        "retrieval_message": (
            "Retrieval allowance exhausted; available evidence may "
            "be partial. This is not proof that supporting "
            "information is absent."
        ),
    }


def allow_targeted_tool(name, arguments):
    import json

    from lib.llm.providers.tool_budget import ToolCallObservation

    state = current_turn()
    if state is None or state.policy is None:
        return True
    if sum(state.policy.state.tool_name_call_counts.values()) >= state.max_func_calls:
        return False
    return state.policy.observe_tool_call(
        ToolCallObservation(name, json.dumps(arguments, sort_keys=True))
    ).should_continue


async def execute_action(consumer, action, *, top_k, vector_runner=None):
    from channels.db import database_sync_to_async

    from apps.chat.services.tool_wiring.documents import (
        more_context_tool,
        search_single_document_tool,
        vector_search_tool,
    )

    if action.kind == "vector":
        name, arguments = (
            "vector_search",
            {"search_string": action.query, "top_k": top_k},
        )
        tool = vector_search_tool(consumer.user, consumer.col_ref)
    elif action.kind == "document":
        name, arguments = (
            "search_single_document",
            {
                "doc_id": action.document_id,
                "search_string": action.query,
                "top_k": top_k,
            },
        )
        tool = search_single_document_tool(consumer.user, consumer.col_ref)
    else:
        name, arguments = (
            "more_context",
            {"chunk_id": action.chunk_id, "adjacent_chunks": 1},
        )
        tool = more_context_tool(consumer.user)
    if not allow_targeted_tool(name, arguments):
        return limited_action_result()
    if vector_runner is not None and action.kind == "vector":
        return await database_sync_to_async(vector_runner, thread_sensitive=False)(
            consumer, action.query, top_k
        )
    return dict(await database_sync_to_async(tool, thread_sensitive=False)(**arguments))


def source_views(results):
    from lib.retrieval.evidence import SourceEvidence, fingerprint_source

    runtime = current_source_runtime()
    if runtime is None:
        return ()
    records = {
        r["chunk_id"]: r
        for result in results
        for r in result.get("_source_provenance", ())
    }
    sources = {}
    with runtime.lock:
        for row in runtime.cache.values():
            record = records.get(row.pk)
            revision = fingerprint_source(row.content)
            if record and record["source_fingerprint"] == revision:
                source = SourceEvidence(
                    row.pk, str(row.doc_id), row.chunk_number, revision, row.content
                )
                sources[source.identity] = source
    return tuple(sources.values())
