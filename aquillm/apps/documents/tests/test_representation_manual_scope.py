"""Human-selected child authority cannot widen automatic or sibling acquisition."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import TextChunk
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_manual_scope import (
    create_manual_document_runtime, private_manual_acquisition,
)
from apps.documents.services.representation_media_acquisition import initialize_media_runtime, media_runtime
from apps.documents.services.representation_tool_scope import tool_scope
from apps.documents.services.source_loading import source_runtime_scope, current_source_runtime
from apps.chat.services.tool_wiring.source_documents import document_metadata
from lib.llm.turn_context import TurnContext, bind_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, UserMessage
from .representation_support import retained as retained, upload as upload
from .test_representation_tool_scope import pair as pair

pytestmark = pytest.mark.django_db(transaction=True)


def outside_parent(pair):
    a, b, _, _, chunk, original = pair
    other = Collection.objects.create(name="Outside picker")
    permission = CollectionPermission.objects.create(user=a.ingested_by, collection=other, permission="VIEW")
    b.collection = other
    b.save(dont_rechunk=True, update_fields=["collection"])
    turn = TurnContext(original.budget)
    consumer = SimpleNamespace(user=a.ingested_by, col_ref=SimpleNamespace(collections=[a.collection_id]))
    parent = initialize_media_runtime(consumer, turn)
    return a, b, chunk, permission, turn, consumer, parent


@pytest.mark.parametrize("source_mode", [False, True])
def test_child_is_exact_eager_frozen_and_never_becomes_parent(pair, source_mode):
    from concurrent.futures import ThreadPoolExecutor
    from django.db import close_old_connections
    a, b, _, _, turn, _, parent = outside_parent(pair)
    snapshot = parent._frozen_retrieval_scope.representation
    with bind_turn(turn), source_runtime_scope(parent) if source_mode else nullcontext():
        with pytest.raises(SourceScopeUnavailable):
            document_metadata(b.id)
        child = create_manual_document_runtime(a.ingested_by, b.id, parent=parent)
        assert child.budget is parent.budget and child.lock is parent.lock
        assert {s.document.id for s in child._frozen_retrieval_scope.representation.selections} == {b.id}
        def read(manual):
            close_old_connections()
            try:
                with bind_turn(turn):
                    context = source_runtime_scope(child if manual else parent) if source_mode else (
                        private_manual_acquisition(child) if manual else nullcontext())
                    with context:
                        doc = document_metadata(b.id if manual else a.id)
                        assert "full_text" in doc.get_deferred_fields()
                        return doc.id, media_runtime()
            finally:
                close_old_connections()
        with ThreadPoolExecutor(max_workers=2) as pool:
            left, right = tuple(pool.map(read, (False, True)))
        assert left == (a.id, parent) and right == (b.id, child)
        assert parent._frozen_retrieval_scope.representation is snapshot
        assert media_runtime() is parent
    with pytest.raises(SourceScopeUnavailable):
        with private_manual_acquisition(child):
            pass


@pytest.mark.parametrize("change", [None, "permission", "pointer"])
@pytest.mark.parametrize("source_mode", [False, True])
def test_actual_human_search_outside_picker_uses_child_and_final_sinks(pair, monkeypatch, change, source_mode):
    from apps.chat.services.manual_search_turn import run_manual_search_turn
    from apps.chat.services import rag_immutable_images, rag_synthesis
    from apps.documents.services import chunk_search
    from apps.documents.services.representation_snapshot_build import build_canonical_snapshot_representation
    from apps.chat.consumers.chat_publish import end_spin_publish
    from aquillm.message_adapters import save_conversation_to_db
    from aquillm.models import WSConversation
    from lib.llm.providers.final_stream import send_final_answer_stream
    from lib.llm.answer_publication import LIMITED_ANSWER
    from .representation_support import reference
    a, b, chunk, permission, turn, consumer, parent = outside_parent(pair)
    for key in ("RAG_EVIDENCE_TEXT_MODE", "RAG_RERANK_TEXT_MODE", "RAG_DOCUMENT_CAPACITY_MODE"):
        monkeypatch.setenv(key, "legacy")
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", "0")
    monkeypatch.setenv("RAG_ITERATIVE_RETRIEVAL_ENABLED", "0")
    if source_mode:
        monkeypatch.setenv("RAG_DOCUMENT_CAPACITY_MODE", "budgeted")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "65536")
    # Exercise real lexical SQL and packing, with unavailable external embedding.
    monkeypatch.setattr(chunk_search, "load_query_embedding", lambda *args: None)
    calls, events = [], []
    citation = f"[doc:{b.id} chunk:{chunk.pk}]"
    class Provider:
        async def complete(self, request, *args, **kwargs):
            calls.append(request)
            assert (current_source_runtime() is not None) is source_mode
            assert media_runtime() is not parent and media_runtime().budget is parent.budget
            return request + [AssistantMessage(content=chunk.content + " " + citation, stop_reason="stop")], "stop"
    original = rag_immutable_images.authorize_media_dispatch
    async def mutate(*args, **kwargs):
        from asgiref.sync import sync_to_async
        if change == "permission":
            await sync_to_async(permission.delete)()
        elif change == "pointer":
            await sync_to_async(build_canonical_snapshot_representation)(reference(b), using="default")
        return await original(*args, **kwargs)
    if change:
        monkeypatch.setattr(rag_immutable_images, "authorize_media_dispatch", mutate)
    async def send(**kwargs):
        events.append(kwargs)
    async def stream(payload):
        events.append(payload)
    conversation = Conversation(system="s", messages=[UserMessage(content=f"/search {b.id} second source")])
    consumer.convo = conversation
    db = WSConversation.objects.create(owner=consumer.user)
    snapshot = parent._frozen_retrieval_scope.representation
    with bind_turn(turn), source_runtime_scope(parent) if source_mode else nullcontext():
        assert async_to_sync(run_manual_search_turn)(consumer, Provider(), conversation) == "handled"
        assert parent._frozen_retrieval_scope.representation is snapshot
        assert media_runtime() is parent
        assert (current_source_runtime() is not None) is source_mode
        answer = consumer.convo[-1]
        assert answer.content == LIMITED_ANSWER if change else citation in answer.content
        async_to_sync(send_final_answer_stream)(stream, message_uuid=str(answer.message_uuid),
            content=answer.content, stop_reason="stop", usage=0, message=answer)
        async_to_sync(end_spin_publish)(SimpleNamespace(convo=consumer.convo, _spin_epoch=0, send=send), consumer.convo)
        save_conversation_to_db(consumer.convo, db)
    assert events and db.db_messages.exists()
