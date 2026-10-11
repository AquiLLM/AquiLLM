"""Known source identity survives proposal admission and actual query-only turns."""
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio
from channels.db import database_sync_to_async

from apps.chat.refs import CollectionsRef
from apps.chat.services import rag_pipeline
from apps.chat.services.rag_config import rag_preservation_config
from apps.chat.services.rag_source_continuity import resolve_source_anchors
from apps.collections.tests.test_vocabulary_scope import setup_scope
from apps.documents.models import RawTextDocument
from apps.documents.models.chunks import TextChunk
from apps.documents.services.source_loading import current_source_runtime, source_query_rows
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage


@pytest_asyncio.fixture(autouse=True)
async def close_fixture_worker_connection():
    from django.db import connections
    yield
    await database_sync_to_async(connections.close_all)()


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [False, True])
@pytest.mark.parametrize("rewrite,expected", [
    ("Summarize Blue Dwarfs", ["Summarize Red Giants"]),
    ("Explain Red Giants", ["Summarize Red Giants", "Explain Red Giants"]),
    ("Explain Infrared Giants", ["Summarize Red Giants"]),
    ("Explain Red Giantships", ["Summarize Red Giants"]),
])
async def test_prepared_named_source_constraints_reach_actual_acquisition(
    followup, rewrite, expected, monkeypatch,
):
    actor, collection, auth = await database_sync_to_async(setup_scope)()
    doc_id = next(iter(auth.selected_document_ids))
    await database_sync_to_async(RawTextDocument.objects.filter(pk=doc_id).update)(title="Red Giants")
    text = "Red Giants are synthetic stars."
    chunk = (await database_sync_to_async(TextChunk.objects.bulk_create)([
        TextChunk(doc_id=doc_id, chunk_number=0, content=text,
                  start_position=0, end_position=len(text))]))[0]
    citation = f"[doc:{doc_id} chunk:{chunk.pk}]"
    row = {"doc_id": str(doc_id), "chunk_id": chunk.pk, "chunk": 0,
           "title": "Red Giants", "text": text, "citation": citation}
    question = "Summarize Red Giants"
    convo = Conversation(system="sys", messages=[
        ToolMessage(tool_name="vector_search", for_whom="assistant", content="{}",
                    result_dict={"result": [row]}),
        AssistantMessage(content="Red Giants " + citation, stop_reason="end_turn"),
        UserMessage(content=question),
    ])
    assert resolve_source_anchors(question, convo).chunk_identities == ((chunk.pk, str(doc_id), 0),)
    for key, value in {
        "RAG_DIRECT_ENABLED": "1", "RAG_DIRECT_MAX_QUERIES": "3",
        "RAG_QUERY_REWRITE_ENABLED": "1", "RAG_VOCABULARY_QUERY_ENABLED": "0",
        "RAG_EVIDENCE_TEXT_MODE": "source" if followup else "legacy", "RAG_RERANK_TEXT_MODE": "legacy",
        "RAG_DOCUMENT_CAPACITY_MODE": "legacy", "RAG_FOLLOWUP_EVIDENCE_ENABLED": str(int(followup)),
        "RAG_ITERATIVE_RETRIEVAL_ENABLED": "0", "RAG_EVIDENCE_SELECTION_MODE": "legacy",
        "OPENAI_CONTEXT_LIMIT": "16000",
    }.items():
        monkeypatch.setenv(key, value)
    consumer = SimpleNamespace(user=actor, col_ref=CollectionsRef([collection.pk]), convo=convo)
    searches, ledgers, synthesis_questions = [], [], []

    def search(_consumer, query, _top_k):
        from apps.chat.services.rag_action_tools import admit_tool_action
        runtime = current_source_runtime()
        assert runtime is not None and admit_tool_action("vector", query)
        assert runtime.authorization.selected_collection_ids == frozenset({collection.pk})
        ledgers.append(runtime.budget)
        current = source_query_rows(TextChunk.objects.filter(pk=chunk.pk), runtime=runtime)[0]
        searches.append(query)
        return {"result": [row | {"text": current.content}]}

    async def get_message(**kwargs):
        assert current_source_runtime().budget.planner_calls == 1
        return SimpleNamespace(text=json.dumps({"rewrite": rewrite}))

    async def complete(request, *_args, **_kw):
        return request + [AssistantMessage(content="Red Giants are synthetic stars " + citation,
                                            stop_reason="end_turn")], "changed"

    original_synthesis = rag_pipeline.synthesize_from_evidence
    async def synthesis(llm, request, packet, **kwargs):
        synthesis_questions.append(packet.query)
        assert packet.source_mode is followup
        config = rag_preservation_config()
        assert config.followup_evidence_enabled is followup
        if not followup:
            assert not config.active
        assert len(current_source_runtime().observation["acquisition"].results) == len(expected) + int(followup)
        assert current_source_runtime().budget.pairs_used["final"] == 0
        return await original_synthesis(llm, request, packet, **kwargs)

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synthesis)
    assert await rag_pipeline.run_direct_rag_turn(consumer,
        SimpleNamespace(base_args={"model": "synthetic"}, get_message=get_message, complete=complete), convo) == "handled"
    assert searches == expected
    assert synthesis_questions == [question]
    assert len({id(ledger) for ledger in ledgers}) == 1
    assert ledgers[0].planner_calls == 1 and ledgers[0].actions_used == len(expected)


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [False, True])
@pytest.mark.parametrize("prefix", ["the paper ", ""])
@pytest.mark.parametrize("historical_title,requested_title", [
    (history, request) for history in ("Red", "Red Giants")
    for request in ("Red Giants", "red giants", "Red giants")
] + [
    ("Red", "Red: Giants"), ("Red", "Red/Giants"),
    ("Red: Giants", "Red: Giants"), ("Red/Giants", "Red/Giants"),
    ("Red", "Red"), ("Red: Giants", '"Red: Giants"'), ("Red", '"Red: Giants"'),
    ("Red", "Big Red"), ("Big Red", "Big Red"),
    ("Red", "Red—Giants"), ("Red—Giants", "Red—Giants"),
    ("Stars", "Named Stars"), ("Named Stars", "Named Stars"), ("Stars", 'named "Stars"'),
])
async def test_explicit_longer_title_does_not_borrow_partial_historical_identity(
    followup, prefix, historical_title, requested_title, monkeypatch,
):
    actor, collection, auth = await database_sync_to_async(setup_scope)()
    doc_id = next(iter(auth.selected_document_ids))
    chunk = (await database_sync_to_async(TextChunk.objects.bulk_create)([
        TextChunk(doc_id=doc_id, chunk_number=0, content="A different paper.",
                  start_position=0, end_position=18)]))[0]
    citation = f"[doc:{doc_id} chunk:{chunk.pk}]"
    question = f"Summarize {prefix}{requested_title}"
    rewrite = f"Explain {prefix}{requested_title}"
    convo = Conversation(system="sys", messages=[
        ToolMessage(tool_name="vector_search", for_whom="assistant", content="{}",
            result_dict={"result": [{"doc_id": str(doc_id), "chunk_id": chunk.pk,
                "chunk": 0, "title": historical_title, "text": "A different paper.", "citation": citation}]}),
        AssistantMessage(content=historical_title + " " + citation, stop_reason="end_turn"),
        UserMessage(content=question),
    ])
    for key, value in {
        "RAG_DIRECT_ENABLED": "1", "RAG_QUERY_REWRITE_ENABLED": "1",
        "RAG_VOCABULARY_QUERY_ENABLED": "0", "OPENAI_CONTEXT_LIMIT": "16000",
        "RAG_EVIDENCE_TEXT_MODE": "source" if followup else "legacy",
        "RAG_FOLLOWUP_EVIDENCE_ENABLED": str(int(followup)),
        "RAG_RERANK_TEXT_MODE": "legacy", "RAG_DOCUMENT_CAPACITY_MODE": "legacy",
        "RAG_ITERATIVE_RETRIEVAL_ENABLED": "0", "RAG_EVIDENCE_SELECTION_MODE": "legacy",
    }.items():
        monkeypatch.setenv(key, value)
    searches, identities = [], []

    def search(_consumer, query, _top_k):
        from apps.chat.services.rag_action_tools import admit_tool_action
        assert admit_tool_action("vector", query)
        searches.append(query)
        identities.append(current_source_runtime()._query_proposal_anchors.document_ids)
        return {"result": []}

    async def get_message(**kwargs):
        return SimpleNamespace(text=json.dumps({"rewrite": rewrite}))

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    consumer = SimpleNamespace(user=actor, col_ref=CollectionsRef([collection.pk]), convo=convo)
    assert await rag_pipeline.run_direct_rag_turn(consumer,
        SimpleNamespace(base_args={"model": "synthetic"}, get_message=get_message), convo) == "handled"
    assert searches == [question, rewrite]
    exact_title = (historical_title.casefold() == requested_title.strip('"').casefold()
                   or bool(prefix and requested_title == 'named "Stars"'))
    expected_identity = (str(doc_id),) if exact_title else ()
    assert identities == [expected_identity, expected_identity]
