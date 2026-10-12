"""Original requested collections stay distinct from represented source scope."""
from types import SimpleNamespace
import json

import pytest
from channels.db import database_sync_to_async

from apps.chat.refs import CollectionsRef
from apps.chat.services import rag_pipeline, rag_immutable_images
from apps.chat.services.rag_action_tools import admit_tool_action
from apps.chat.services.rag_turn import preservation_turn
from apps.chat.tests.direct_rag_test_support import _results_payload, _user_convo
from apps.chat.tests.persisted_direct_test_support import (
    close_direct_worker, persisted_direct_case, persisted_conversation,
)
from apps.collections.models import CollectionPermission
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.source_loading import current_source_runtime, source_query_rows
from lib.llm.answer_publication import AnswerPublicationRejected, LIMITED_ANSWER
from lib.llm.turn_context import current_turn
from .cross_lane_publication_support import Provider, publish


def configure(monkeypatch, source_mode):
    for key, value in {
        "RAG_DIRECT_ENABLED": "1", "RAG_DIRECT_MAX_QUERIES": "3",
        "RAG_QUERY_REWRITE_ENABLED": "1", "RAG_VOCABULARY_QUERY_ENABLED": "0",
        "RAG_EVIDENCE_TEXT_MODE": "source" if source_mode else "legacy",
        "RAG_RERANK_TEXT_MODE": "legacy", "RAG_DOCUMENT_CAPACITY_MODE": "legacy",
        "RAG_FOLLOWUP_EVIDENCE_ENABLED": "0", "RAG_ITERATIVE_RETRIEVAL_ENABLED": "0",
        "RAG_EVIDENCE_SELECTION_MODE": "legacy", "OPENAI_CONTEXT_LIMIT": "32768",
        "OPENAI_STREAM_RESPONSES": "0", "CONTEXT_PACKER_ENABLED": "0",
        "TOKEN_EFFICIENCY_ENABLED": "0", "LLM_STREAM_FINAL_ANSWER_ONLY": "1",
        "RAG_APPEND_CITATION_SOURCES": "0",
    }.items():
        monkeypatch.setenv(key, value)


def foreign_row(case):
    document = RawTextDocument(collection=case.collection_map[4], ingested_by=case.user,
        title="Nonselected source", full_text="Nonselected confidential comparison.")
    document.save(dont_rechunk=True)
    chunk = TextChunk.objects.create(doc_id=document.id, chunk_number=0,
        content=document.full_text, start_position=0, end_position=len(document.full_text))
    assert CollectionPermission.objects.filter(user=case.user,
        collection=case.collection_map[4], permission="VIEW").exists()
    return {"doc_id": str(document.id), "chunk_id": chunk.pk, "chunk": 0,
        "title": document.title, "text": chunk.content,
        "citation": f"[doc:{document.id} chunk:{chunk.pk}]"}


def mutate_selection(consumer, case, change):
    if change == "drop_empty":
        consumer.col_ref.collections = case.collections((1, 2))
    elif change == "replace_reference":
        consumer.col_ref = CollectionsRef(case.collections((1, 2, 3)))
    else:
        raise AssertionError(change)


async def reject_all_sinks(case, consumer, db):
    from apps.chat.consumers.chat_publish import end_spin_publish
    from aquillm.message_adapters import save_conversation_to_db
    from lib.llm.providers.final_stream import send_final_answer_stream
    events = []
    async def stream(payload):
        events.append(payload)
    async def send(**kwargs):
        events.append(kwargs)
    message = consumer.convo[-1]
    context = current_turn().answer_publication
    lease = context.lease
    def active():
        assert current_turn().answer_publication is context and context.lease is lease
        context._active()
        assert context.remaining_seconds() > 0 and lease.remaining_seconds() > 0
    active()
    with pytest.raises(AnswerPublicationRejected, match="^scope_changed$"):
        await send_final_answer_stream(stream, message_uuid=str(message.message_uuid),
            content=message.content, stop_reason=message.stop_reason,
            usage=message.usage, message=message)
    active()
    with pytest.raises(AnswerPublicationRejected, match="^scope_changed$"):
        await end_spin_publish(SimpleNamespace(convo=consumer.convo,
            _spin_epoch=len(consumer.convo)-1, send=send), consumer.convo)
    active()
    with pytest.raises(AnswerPublicationRejected, match="^scope_changed$"):
        await database_sync_to_async(save_conversation_to_db)(consumer.convo, db)
    assert events == []
    assert not await database_sync_to_async(db.db_messages.exists)()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_mode", [False, True])
@pytest.mark.parametrize("change", [None, "drop_empty", "replace_reference", "revoke",
                                   "nonselected", "before_sdk", "before_sinks"])
async def test_requested_scope_survives_empty_collections_but_not_mutation(
    monkeypatch, source_mode, change, close_direct_worker,
):
    configure(monkeypatch, source_mode)
    raw = _results_payload()
    case = await persisted_direct_case(raw["result"], collection_labels=(1, 2, 3, 4))
    foreign = await database_sync_to_async(foreign_row)(case)
    db = await persisted_conversation(case, (1, 2, 3))
    question = "Search selected documents for calibration"
    convo = _user_convo(question)
    original_conversation = convo.model_dump()
    consumer = case.consumer(convo, (1, 2, 3))
    requested = tuple(consumer.col_ref.collections)
    citation = case.text("[doc:doc-a chunk:1]")
    provider = Provider([raw["result"][0]["text"] + " " + citation])
    calls, mutations, packets = [], [], []
    original_synthesis = rag_pipeline.synthesize_from_evidence

    async def planner(**kwargs):
        return SimpleNamespace(text=json.dumps({"rewrite": question}))

    def search(actual_consumer, query, top_k):
        assert actual_consumer is consumer and query == question
        assert admit_tool_action("vector", query)
        runtime = current_source_runtime()
        calls.append(runtime)
        assert runtime.authorization.selected_collection_ids == frozenset(case.collections((1,)))
        chunk = case.chunks[("doc-a", 1)]
        assert source_query_rows(TextChunk.objects.filter(pk=chunk.pk), runtime=runtime)[0].content == chunk.content
        if change in {"drop_empty", "replace_reference"}:
            mutate_selection(consumer, case, change)
            mutations.append(change)
        elif change == "revoke":
            deleted, _ = CollectionPermission.objects.filter(user=case.user,
                collection=case.collection_map[1], permission="VIEW").delete()
            assert deleted == 1
            mutations.append(change)
        return {"result": [foreign]} if change == "nonselected" else case.rows(raw)

    async def synthesis(_planner, working, packet, **kwargs):
        assert packet.query == question
        assert packet.source_mode is source_mode
        packets.append(packet)
        return await original_synthesis(provider.llm, working, packet, **kwargs)

    original_dispatch = rag_immutable_images.authorize_media_dispatch
    async def dispatch(packet, *, pre_seal=False):
        if change == "before_sdk" and not pre_seal:
            assert current_turn().answer_publication.packet is packet
            mutate_selection(consumer, case, "drop_empty")
            mutations.append(change)
        return await original_dispatch(packet, pre_seal=pre_seal)

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synthesis)
    monkeypatch.setattr(rag_immutable_images, "authorize_media_dispatch", dispatch)
    async with preservation_turn(consumer, max_func_calls=3) as runtime:
        scope, authority, budget = runtime._frozen_retrieval_scope, runtime.authorization, runtime.budget
        snapshot = scope.representation
        assert tuple(consumer.col_ref.collections) == requested
        assert len(requested) == 3 and len(authority.selected_collection_ids) == 1
        assert await rag_pipeline.run_direct_rag_turn(consumer,
            SimpleNamespace(base_args={}, get_message=planner), convo) == "handled"
        assert calls == [runtime]
        assert current_source_runtime() is runtime and runtime.authorization is authority
        assert runtime._frozen_retrieval_scope is scope and scope.representation is snapshot
        assert runtime.budget is budget and budget.actions_used == 1 and budget.planner_calls == 1
        assert (budget._synthesis.remaining_seconds() > 0 if budget._synthesis
                else budget.remaining_ms() > 0)
        if change in {None, "before_sinks"}:
            assert len(provider.requests) == 1
            assert citation in consumer.convo[-1].content
            assert raw["result"][0]["text"] in str(provider.requests)
            assert foreign["text"] not in str(provider.requests)
            assert consumer.convo[-1]._validated_answer is not None
            assert consumer.convo[-1]._validated_answer.outcome != "limited"
            if change == "before_sinks":
                mutate_selection(consumer, case, "drop_empty")
                await reject_all_sinks(case, consumer, db)
            else:
                await publish(SimpleNamespace(db=db), consumer)
        else:
            assert provider.requests == []
            assert consumer.convo[-1].content == LIMITED_ANSWER
            if change != "nonselected":
                assert mutations == [change]
            assert foreign["text"] not in str(consumer.convo.model_dump())
    assert convo.model_dump() == original_conversation


@pytest.mark.asyncio
async def test_registered_original_parent_never_downgrades_to_standalone(monkeypatch, close_direct_worker):
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    from apps.documents.services.representation_requested_selection import requested_selection
    configure(monkeypatch, False)
    case = await persisted_direct_case(_results_payload()["result"])
    consumer = case.consumer(_user_convo("Search calibration"), (1,))
    async with preservation_turn(consumer, max_func_calls=3) as runtime:
        turn, scope, budget = current_turn(), runtime._frozen_retrieval_scope, runtime.budget
        assert runtime.authorization.selected_collection_ids == frozenset(consumer.col_ref.collections)
        retained = runtime._media_principal_binding
        del runtime._media_principal_binding
        del runtime._media_consumer
        del turn._media_principal_binding
        try:
            assert turn._media_source_runtime is runtime
            with pytest.raises(SourceScopeUnavailable, match="^source_scope_unavailable$"):
                requested_selection(runtime, consumer)
            assert runtime._frozen_retrieval_scope is scope and runtime.budget is budget
            assert budget.actions_used == 0 and budget.planner_calls == 0
        finally:
            runtime._media_principal_binding = turn._media_principal_binding = retained
            runtime._media_consumer = consumer


@pytest.mark.asyncio
async def test_genuine_standalone_runtime_keeps_exact_represented_selection(close_direct_worker):
    from apps.chat.services.rag_turn import create_runtime
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    from apps.documents.services.representation_requested_selection import requested_selection
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits
    case = await persisted_direct_case(_results_payload()["result"], collection_labels=(1, 2))
    consumer = case.consumer(_user_convo("Search calibration"), (1,))
    budget = TurnBudget(TurnLimits())
    runtime = await database_sync_to_async(create_runtime)(consumer, budget)
    assert current_turn() is None and not hasattr(runtime, "_media_principal_binding")
    assert requested_selection(runtime, consumer) == frozenset(case.collections((1,)))
    consumer.col_ref.collections = case.collections((1, 2))
    with pytest.raises(SourceScopeUnavailable, match="^source_scope_unavailable$"):
        requested_selection(runtime, consumer)
    assert runtime.budget is budget and budget.actions_used == 0
