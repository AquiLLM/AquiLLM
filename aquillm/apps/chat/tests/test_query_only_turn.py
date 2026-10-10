"""Query-only treatment uses current scope without enabling preservation modes."""
from types import SimpleNamespace

import pytest
import pytest_asyncio
from channels.db import database_sync_to_async

from apps.chat.refs import CollectionsRef
from apps.chat.services import rag_pipeline
from apps.chat.services.rag_config import rag_preservation_config
from apps.collections.tests.test_vocabulary_scope import setup_scope
from apps.documents.models.chunks import TextChunk
from apps.documents.services.source_loading import current_source_runtime, source_query_rows
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, UserMessage


@pytest_asyncio.fixture(autouse=True)
async def close_fixture_worker_connection():
    from django.db import connections
    yield
    await database_sync_to_async(connections.close_all)()


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_query_only_real_turn_keeps_original_question_modes_and_ledger(monkeypatch):
    actor, collection, auth = await database_sync_to_async(setup_scope)()
    doc_id = next(iter(auth.selected_document_ids))
    chunk = (await database_sync_to_async(TextChunk.objects.bulk_create)([
        TextChunk(doc_id=doc_id, chunk_number=0, content="A_12 is a synthetic instrument.",
                  start_position=0, end_position=31)]))[0]
    for key, value in {
        "RAG_DIRECT_ENABLED": "1", "RAG_DIRECT_MAX_QUERIES": "3",
        "RAG_QUERY_REWRITE_ENABLED": "1", "RAG_VOCABULARY_QUERY_ENABLED": "0",
        "RAG_EVIDENCE_TEXT_MODE": "legacy", "RAG_RERANK_TEXT_MODE": "legacy",
        "RAG_DOCUMENT_CAPACITY_MODE": "legacy", "RAG_FOLLOWUP_EVIDENCE_ENABLED": "0",
        "RAG_ITERATIVE_RETRIEVAL_ENABLED": "0", "RAG_EVIDENCE_SELECTION_MODE": "legacy",
        "OPENAI_CONTEXT_LIMIT": "16000",
    }.items():
        monkeypatch.setenv(key, value)
    question = "What is A_12?"
    convo = Conversation(system="sys", messages=[UserMessage(content=question)])
    consumer = SimpleNamespace(user=actor, col_ref=CollectionsRef([collection.pk]), convo=convo)
    searches, captured, ledgers = [], [], []

    def search(_consumer, query, _top_k):
        from apps.chat.services.rag_action_tools import admit_tool_action
        runtime = current_source_runtime()
        assert runtime is not None and admit_tool_action("vector", query)
        ledgers.append(runtime.budget)
        row = source_query_rows(TextChunk.objects.filter(pk=chunk.pk), runtime=runtime)[0]
        searches.append(query)
        return {"result": [{"doc_id": str(doc_id), "chunk_id": chunk.pk, "chunk": 0,
            "title": "Synthetic", "text": row.content, "citation": f"[doc:{doc_id} chunk:{chunk.pk}]"}]}

    async def get_message(**kwargs):
        runtime = current_source_runtime()
        assert runtime is not None and runtime.budget.planner_calls == 1
        return SimpleNamespace(text='{"rewrite":"Explain A_12?"}')

    async def complete(request, *_args, **_kw):
        captured.append(request[-1].result_dict)
        return request + [AssistantMessage(content="A_12 is an instrument " +
            request[-1].result_dict["result"][0]["citation"], stop_reason="end_turn")], "changed"

    original_synthesis = rag_pipeline.synthesize_from_evidence
    async def synthesis(llm, request, packet, **kwargs):
        assert packet.query == question and packet.source_mode is False
        assert packet.source_authorization is current_source_runtime().authorization
        assert not rag_preservation_config().active
        assert current_source_runtime().budget.pairs_used["final"] == 0
        return await original_synthesis(llm, request, packet, **kwargs)

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synthesis)
    assert await rag_pipeline.run_direct_rag_turn(consumer,
        SimpleNamespace(base_args={"model": "synthetic"}, get_message=get_message, complete=complete), convo) == "handled"
    assert searches == [question, "Explain A_12?"]
    assert captured and len({id(v) for v in ledgers}) == 1
    assert ledgers[0].planner_calls == 1 and ledgers[0].actions_used == 2
