"""Default turns capture original authority without enabling preservation routing."""
from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync, sync_to_async

from apps.chat.services.rag_answer_publication import _freeze_legacy_packet, bind_packet_publication
from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.services.rag_turn import preservation_turn
from apps.documents.services.chunk_search_scope import freeze_search_scope
from apps.documents.services.source_loading import current_source_runtime
from lib.llm.turn_context import current_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage
from .representation_support import retained as retained, upload as upload, activate
from .test_representation_tool_scope import pair as pair

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("swap", [False, True])
def test_default_text_turn_captures_before_acquisition_and_keeps_legacy_route(retained, monkeypatch, swap):
    from apps.chat.services import rag_synthesis
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    for key, value in {"RAG_EVIDENCE_TEXT_MODE": "legacy", "RAG_RERANK_TEXT_MODE": "legacy",
                       "RAG_DOCUMENT_CAPACITY_MODE": "legacy", "RAG_FOLLOWUP_EVIDENCE_ENABLED": "0",
                       "RAG_ITERATIVE_RETRIEVAL_ENABLED": "0"}.items():
        monkeypatch.setenv(key, value)
    doc, output, chunk = retained
    consumer = SimpleNamespace(user=doc.ingested_by, col_ref=SimpleNamespace(collections=[doc.collection_id]))
    calls = []
    async def forbidden(*args, **kwargs):
        raise AssertionError("private runtime changed default synthesis routing")
    monkeypatch.setattr(rag_synthesis, "revalidate_source_packet", forbidden)
    monkeypatch.setattr(rag_synthesis, "complete_source_request", forbidden)
    citation = f"[doc:{doc.id} chunk:{chunk.pk}]"
    result = {"result": [{"doc_id": str(doc.id), "chunk_id": chunk.pk,
                         "chunk": 0, "text": chunk.content, "citation": citation}]}
    class Provider:
        async def complete(self, convo, *args, **kwargs):
            calls.append(convo)
            return convo + [AssistantMessage(content=chunk.content + " " + citation, stop_reason="stop")], "stop"

    async def run():
        async with preservation_turn(consumer, max_func_calls=1) as public_runtime:
            assert public_runtime is None and current_source_runtime() is None
            turn = current_turn()
            runtime = turn._media_source_runtime
            snapshot = runtime._frozen_retrieval_scope.representation
            assert {item.document.id for item in snapshot.selections} == {doc.id}
            if swap:
                assert await sync_to_async(activate)(doc, output, text=chunk.content) == "activated"
                with pytest.raises(SourceScopeUnavailable):
                    await sync_to_async(freeze_search_scope)((doc,), using="default", authorization=runtime.authorization)
                assert runtime._frozen_retrieval_scope.representation is snapshot
                return
            scope = await sync_to_async(freeze_search_scope)((doc,), using="default", authorization=runtime.authorization)
            assert scope.representation is snapshot
            packet = build_evidence_packet(result, query="table", search_scope="selected")
            assert not packet.source_mode
            assert await sync_to_async(_freeze_legacy_packet)(consumer, packet, (doc.collection_id,), turn.budget)
            bind_packet_publication(packet)
            convo = Conversation(system="s", messages=[ToolMessage(tool_name="vector_search", for_whom="assistant",
                content="result", result_dict=result)])
            answer = await rag_synthesis.synthesize_from_evidence(Provider(), convo, packet, max_tokens=256)
            assert citation in answer[-1].content
            assert current_source_runtime() is None and not packet.source_mode
    async_to_sync(run)()
    assert len(calls) == (0 if swap else 1)


@pytest.mark.parametrize("concurrent", [False, True])
def test_private_default_subset_searches_share_original_scope(pair, concurrent):
    from concurrent.futures import ThreadPoolExecutor
    from django.db import close_old_connections
    from lib.llm.turn_context import TurnContext, bind_turn
    from apps.documents.services.representation_media_acquisition import initialize_media_runtime
    a, b, _, _, _, original = pair
    turn = TurnContext(original.budget)
    consumer = SimpleNamespace(user=a.ingested_by, col_ref=SimpleNamespace(collections=[a.collection_id]))
    runtime = initialize_media_runtime(consumer, turn)
    captured = runtime._frozen_retrieval_scope.representation
    assert {row.document.id for row in captured.selections} == {a.id, b.id}
    def read(document):
        close_old_connections()
        try:
            with bind_turn(turn):
                assert current_source_runtime() is None
                scope = freeze_search_scope((document,), using="default", authorization=runtime.authorization)
                assert scope.representation is captured
                assert scope.requested_document_ids == {document.id}
                assert current_source_runtime() is None
                return scope
        finally:
            close_old_connections()
    if concurrent:
        with ThreadPoolExecutor(max_workers=2) as pool:
            left, right = tuple(pool.map(read, (a, b)))
    else:
        left, right = read(a), read(b)
    assert left.representation is right.representation


@pytest.mark.parametrize("change", [None, "pointer", "foreign_scope"])
def test_default_hydration_reuses_original_private_scope(pair, change):
    from dataclasses import replace
    from apps.chat.services.rag_selection_hydration import hydrate_pool_rows
    from apps.documents.services.representation_media_acquisition import initialize_media_runtime
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    from lib.llm.turn_context import TurnContext, bind_turn
    a, _, output, chunk, _, foreign = pair
    turn = TurnContext(foreign.budget)
    runtime = initialize_media_runtime(SimpleNamespace(user=a.ingested_by,
        col_ref=SimpleNamespace(collections=[a.collection_id])), turn)
    scope = runtime._frozen_retrieval_scope
    row = {"doc_id": str(a.id), "chunk_id": chunk.pk, "chunk": 0, "text": chunk.content,
           "citation": f"[doc:{a.id} chunk:{chunk.pk}]"}
    if change == "pointer":
        assert activate(a, output, text=chunk.content) == "activated"
    elif change == "foreign_scope":
        runtime._frozen_retrieval_scope = replace(scope, runtime=foreign)
    with bind_turn(turn):
        if change:
            with pytest.raises(SourceScopeUnavailable):
                hydrate_pool_rows((row,), runtime.authorization)
        else:
            assert hydrate_pool_rows((row,), runtime.authorization)[0].chunk.pk == chunk.pk
        assert current_source_runtime() is None
    assert runtime._frozen_retrieval_scope.representation is scope.representation
