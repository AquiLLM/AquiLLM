"""STAGED, UNEXECUTED: actual tool/turn; observers never fabricate authority."""
from asgiref.sync import async_to_sync
from channels.db import database_sync_to_async
from django.db import connections

from apps.chat.services.rag_turn import preservation_turn
from apps.chat.services.tool_wiring.documents import search_single_document_tool
from apps.documents.services.source_loading import current_source_runtime
from lib.llm.turn_context import current_turn


def modes(monkeypatch, mode):
    for name, value in {
        "RAG_EVIDENCE_TEXT_MODE": "source" if mode == "source" else "legacy",
        "RAG_RERANK_TEXT_MODE": "legacy", "RAG_DOCUMENT_CAPACITY_MODE": "legacy",
        "RAG_FOLLOWUP_EVIDENCE_ENABLED": "0", "RAG_ITERATIVE_RETRIEVAL_ENABLED": "0",
        "RAG_VOCABULARY_QUERY_ENABLED": "1" if mode == "query" else "0",
        "RAG_QUERY_REWRITE_ENABLED": "0", "RAG_EVIDENCE_SELECTION_MODE": "legacy",
        "RAG_RERANK_SHADOW_SCORING_ENABLED": "0",
        "TOOL_SEARCH_COMPACT_PAYLOAD": "0",
    }.items():
        monkeypatch.setenv(name, value)


async def database(function, *args, **kwargs):
    def work():
        try:
            return function(*args, **kwargs)
        finally:
            connections.close_all()
    return await database_sync_to_async(work, thread_sensitive=False)()


def useful(budget):
    budget.check_active()
    assert budget.can_publish() and budget.remaining_ms() > 1000
    return budget.remaining_ms()


async def tools_in_turn(corpus, order, query, callback=None, *, mode="default", consumer=None):
    consumer = corpus.consumer if consumer is None else consumer
    async with preservation_turn(consumer, max_func_calls=3) as public:
        turn = current_turn()
        runtime = turn._media_source_runtime
        original = runtime._frozen_retrieval_scope
        assert original.runtime is runtime and runtime.budget is turn.budget
        assert {s.document.id for s in original.representation.selections} == {
            doc.id for doc in corpus.documents}
        assert (public is None and current_source_runtime() is None) if mode == "default" else (
            public is runtime and current_source_runtime() is runtime)
        tool = search_single_document_tool(consumer.user, consumer.col_ref,
            authorization_context=runtime.authorization)
        results = []
        for index in order:
            results.append(await database(tool, doc_id=str(corpus.documents[index].id),
                search_string=query, top_k=1))
            assert runtime._frozen_retrieval_scope is original
            assert runtime.budget is turn.budget
            useful(turn.budget)
        if callback:
            await callback(runtime, results)
        return original, results, turn.budget.actions_used


def run_tools(*args, **kwargs):
    return async_to_sync(tools_in_turn)(*args, **kwargs)


def watch_finalization(monkeypatch):
    from apps.documents.services import chunk_search
    from apps.documents.services.chunk_search_scope import current_scope
    original, events = chunk_search.finalize_ranking, []
    def observe(*args, **kwargs):
        scope = current_scope()
        event = {"scope": scope, "budget": scope.runtime.budget,
                 "remaining": useful(scope.runtime.budget)}
        events.append(event)
        try:
            value = original(*args, **kwargs)
        except Exception as error:
            event["error"] = type(error)
            raise
        event["value"] = value
        return value
    monkeypatch.setattr(chunk_search, "finalize_ranking", observe)
    return events


def assert_current_result(result, corpus, index):
    from apps.chat.consumers.utils import truncate_tool_text
    rows = result.get("result", ())
    assert rows and "exception" not in result
    expected = {row.pk: row for row in corpus.rows[index]}
    for row in rows:
        source = expected[row["chunk_id"]]
        assert row["doc_id"] == str(corpus.documents[index].id)
        assert row["text"] == truncate_tool_text(source.content)
        assert row["title"] == corpus.documents[index].title
        assert str(source.pk) in row["citation"]


def snapshot_identity(scope):
    return (scope.representation, scope.dense, scope.space, scope.authorization,
            scope.runtime, scope.runtime.budget)
