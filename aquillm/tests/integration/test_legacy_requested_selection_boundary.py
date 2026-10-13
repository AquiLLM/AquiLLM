"""Default text publication retains the original requested-collection witness.

Regression: recapturing only current collection IDs after retrieval accepts a
changed empty selection or replacement CollectionsRef as the original request.
Only retrieval/provider transport is synthetic; turn creation, representation
capture, publication binding, synthesis, and the three final sinks remain real.
"""
from types import SimpleNamespace

import pytest

from apps.chat.refs import CollectionsRef
from apps.chat.services import rag_pipeline
from apps.chat.services.manual_search_turn import run_manual_search_turn
from apps.chat.services.rag_action_tools import admit_tool_action
from apps.chat.services.rag_config import rag_preservation_config
from apps.chat.services.rag_turn import preservation_turn
from apps.chat.services.retrieval_query_runtime import query_proposals_enabled
from apps.chat.tests.direct_rag_test_support import _results_payload, _user_convo
from apps.chat.tests.persisted_direct_test_support import (
    close_direct_worker, persisted_conversation, persisted_direct_case,
)
from apps.documents.services import chunk_search
from apps.documents.services.representation_media_acquisition import media_runtime
from apps.documents.services.source_loading import current_source_runtime
from lib.llm.answer_publication import LIMITED_ANSWER
from lib.llm.turn_context import current_turn

from .cross_lane_publication_support import Provider, publish
from .test_requested_selection_boundary import reject_all_sinks


def configure_default(monkeypatch, settings):
    for key, value in {
        "RAG_DIRECT_ENABLED": "1", "RAG_DIRECT_MAX_QUERIES": "3",
        "RAG_QUERY_REWRITE_ENABLED": "0", "RAG_VOCABULARY_QUERY_ENABLED": "0",
        "RAG_EVIDENCE_TEXT_MODE": "legacy", "RAG_RERANK_TEXT_MODE": "legacy",
        "RAG_DOCUMENT_CAPACITY_MODE": "legacy", "RAG_DOCUMENT_HARD_CAP": "0",
        "RAG_FOLLOWUP_EVIDENCE_ENABLED": "0", "RAG_ITERATIVE_RETRIEVAL_ENABLED": "0",
        "RAG_EVIDENCE_SELECTION_MODE": "legacy", "RAG_RERANK_SHADOW_SCORING_ENABLED": "0",
        "OPENAI_CONTEXT_LIMIT": "32768", "OPENAI_STREAM_RESPONSES": "0",
        "CONTEXT_PACKER_ENABLED": "0", "TOKEN_EFFICIENCY_ENABLED": "0",
        "LLM_STREAM_FINAL_ANSWER_ONLY": "1", "RAG_APPEND_CITATION_SOURCES": "0",
    }.items():
        monkeypatch.setenv(key, value)
    settings.KG_OVERLAY_ENABLED = False
    settings.RAG_CACHE_ENABLED = False
    settings.DOCUMENT_EMBEDDING_SPACE_KEY = None
    assert rag_preservation_config().active is False
    assert query_proposals_enabled() is False


def change_requested_selection(consumer, case, change):
    if change == "drop_empty":
        consumer.col_ref.collections = case.collections((1, 2))
    elif change == "replace_reference":
        consumer.col_ref = CollectionsRef(case.collections((1, 2, 3)))
    else:
        raise AssertionError(change)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [None, "drop_empty", "replace_reference"])
async def test_default_direct_rejects_selection_change_before_publication_binding(
    monkeypatch, settings, change, close_direct_worker,
):
    configure_default(monkeypatch, settings)
    raw = _results_payload()
    case = await persisted_direct_case(raw["result"], collection_labels=(1, 2, 3))
    db = await persisted_conversation(case, (1, 2, 3))
    convo = _user_convo("Search selected documents for calibration")
    consumer = case.consumer(convo, (1, 2, 3))
    source = raw["result"][0]["text"]
    citation = case.text("[doc:doc-a chunk:1]")
    provider = Provider([source + " " + citation])
    calls = []

    def search(actual_consumer, query, top_k):
        assert actual_consumer is consumer
        assert current_source_runtime() is None
        assert admit_tool_action("vector", query)
        runtime = media_runtime()
        calls.append(runtime)
        assert runtime.authorization.selected_collection_ids == frozenset(case.collections((1,)))
        if change is not None:
            # The original turn and representation are already captured; the
            # legacy publication binding has not yet been constructed.
            change_requested_selection(consumer, case, change)
        return case.rows(raw)

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    async with preservation_turn(consumer, max_func_calls=3):
        runtime, turn = media_runtime(), current_turn()
        binding = runtime._media_principal_binding
        snapshot = runtime._frozen_retrieval_scope.representation
        assert binding.requested_collection_ids == tuple(sorted(case.collections((1, 2, 3))))
        assert await rag_pipeline.run_direct_rag_turn(consumer, provider.llm, convo) == "handled"
        assert calls == [runtime]
        assert media_runtime() is runtime and current_turn() is turn
        assert runtime._media_principal_binding is binding
        assert runtime._frozen_retrieval_scope.representation is snapshot
        assert turn.budget.actions_used == 1 and turn.budget.planner_calls == 0
        if change is None:
            assert len(provider.requests) == 1
            assert citation in consumer.convo[-1].content
            assert consumer.convo[-1]._validated_answer.outcome == "partial"
        else:
            assert provider.requests == [], "changed original selection reached the answer SDK"
            assert consumer.convo[-1].content == LIMITED_ANSWER
        stored, events = await publish(SimpleNamespace(db=db), consumer)
        if change is None:
            assert source in stored and citation in stored
        else:
            for text in (stored, *(text for _, text in events)):
                assert text == LIMITED_ANSWER
                assert source not in text and citation not in text


@pytest.mark.asyncio
async def test_default_direct_rejects_same_id_reference_replacement_at_final_sinks(
    monkeypatch, settings, close_direct_worker,
):
    configure_default(monkeypatch, settings)
    raw = _results_payload()
    case = await persisted_direct_case(raw["result"], collection_labels=(1, 2, 3))
    db = await persisted_conversation(case, (1, 2, 3))
    convo = _user_convo("Search selected documents for calibration")
    consumer = case.consumer(convo, (1, 2, 3))
    citation = case.text("[doc:doc-a chunk:1]")
    provider = Provider([raw["result"][0]["text"] + " " + citation])

    def search(actual_consumer, query, top_k):
        assert actual_consumer is consumer and admit_tool_action("vector", query)
        return case.rows(raw)

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    async with preservation_turn(consumer, max_func_calls=3):
        assert await rag_pipeline.run_direct_rag_turn(consumer, provider.llm, convo) == "handled"
        assert len(provider.requests) == 1 and citation in consumer.convo[-1].content
        original = consumer.col_ref
        change_requested_selection(consumer, case, "replace_reference")
        assert consumer.col_ref is not original and consumer.col_ref.collections == original.collections
        await reject_all_sinks(case, consumer, db)


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_labels", [(1, 2, 3), (2, 3)])
async def test_default_explicit_document_keeps_registered_child_authority(
    monkeypatch, settings, parent_labels, close_direct_worker,
):
    """Original requested selection is distinct from an explicit child's scope."""
    configure_default(monkeypatch, settings)
    # External embedding I/O is unavailable; authorized lexical source loading
    # and manual child creation remain real.
    monkeypatch.setattr(chunk_search, "load_query_embedding", lambda *args: None)
    raw = _results_payload()
    case = await persisted_direct_case(raw["result"], collection_labels=(1, 2, 3))
    document = case.documents["doc-a"]
    db = await persisted_conversation(case, parent_labels)
    convo = _user_convo(f"/search {document.id} calibration")
    consumer = case.consumer(convo, parent_labels)
    original_ref = consumer.col_ref
    requested = tuple(consumer.col_ref.collections)
    citation = case.text("[doc:doc-a chunk:1]")
    provider = Provider([raw["result"][0]["text"] + " " + citation])

    async with preservation_turn(consumer, max_func_calls=3):
        parent, turn = media_runtime(), current_turn()
        assert await run_manual_search_turn(consumer, provider.llm, convo) == "handled"
        assert len(provider.requests) == 1 and citation in consumer.convo[-1].content
        assert consumer.col_ref is original_ref and tuple(consumer.col_ref.collections) == requested
        assert media_runtime() is parent and turn._media_source_runtime is parent
        context = turn.answer_publication
        child_record = context._manual_record
        assert child_record is not None and child_record.parent is parent
        assert child_record.runtime is not parent and child_record.runtime.budget is turn.budget
        assert child_record.authority.selected_document_ids == frozenset((document.id,))
        assert child_record.authority.selected_collection_ids == frozenset(case.collections((1,)))
        stored, _ = await publish(SimpleNamespace(db=db), consumer)
        assert citation in stored and raw["result"][0]["text"] in stored
