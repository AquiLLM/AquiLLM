"""Original identity survives human scope exit and all publication boundaries."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync

from apps.chat.services.manual_search_turn import run_manual_search_turn
from apps.chat.services import rag_immutable_images
from apps.documents.services.source_loading import source_runtime_scope
from lib.llm.answer_publication import AnswerPublicationRejected, LIMITED_ANSWER
from lib.llm.turn_context import bind_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage, ToolMessage
from .principal_scope_support import principals as principals, change_principal, sdk_for

pytestmark = pytest.mark.django_db(transaction=True)


def sinks(case, *, rejected, limited=True):
    from apps.chat.consumers.chat_publish import end_spin_publish
    from aquillm.message_adapters import save_conversation_to_db
    from aquillm.models import WSConversation
    from lib.llm.providers.final_stream import send_final_answer_stream
    events = []
    async def stream(payload):
        events.append(payload)
    async def send(**kwargs):
        events.append(kwargs)
    db = WSConversation.objects.create(owner_id=case.turn._media_principal_binding.principal_pk)
    convo = case.consumer.convo
    message = convo[-1]
    for call in (
        lambda: async_to_sync(send_final_answer_stream)(stream, message_uuid=str(message.message_uuid),
            content=message.content, stop_reason="stop", usage=0, message=message),
        lambda: async_to_sync(end_spin_publish)(SimpleNamespace(convo=convo, _spin_epoch=0, send=send), convo),
        lambda: save_conversation_to_db(convo, db),
    ):
        if rejected:
            with pytest.raises(AnswerPublicationRejected):
                call()
        else:
            call()
    assert len(events) == (0 if rejected else 2)
    assert db.db_messages.exists() is not rejected
    if not rejected and limited:
        assert convo[-1].content == LIMITED_ANSWER
        persisted = str(list(db.db_messages.values()))
        for document, chunk in zip(case.documents, case.chunks):
            for forbidden in (chunk.content, f"[doc:{document.id} chunk:{chunk.pk}]"):
                assert forbidden not in str(events)
                assert forbidden not in persisted


def tamper(case, change):
    context = case.turn.answer_publication
    if change in ("different_user", "same_pk_object", "original_pk"):
        change_principal(case, change)
    elif change == "binding_deleted":
        del context.packet._publication_principal_binding
    elif change == "binding_replaced":
        from dataclasses import replace
        context.packet._publication_principal_binding = replace(context._principal_binding)
    elif change == "registry_deleted":
        del case.turn._manual_publication_bindings[id(context.packet)]
    elif change == "registry_replaced":
        from dataclasses import replace
        case.turn._manual_publication_bindings[id(context.packet)] = (
            context.packet, context, replace(context._manual_record))


@pytest.mark.parametrize("source_mode", [False, True])
@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("change", ["different_user", "same_pk_object", "original_pk",
    "binding_deleted", "binding_replaced", "registry_deleted", "registry_replaced"])
def test_manual_original_identity_before_sdk_and_at_actual_sinks(principals, monkeypatch, source_mode, late, change):
    case = principals
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source" if source_mode else "legacy")
    document, chunk = case.documents[1], case.chunks[1]
    provider, create, citation = sdk_for(document, chunk)
    conversation = Conversation(system="s", messages=[UserMessage(content=f"/search {document.id} requested source")])
    case.consumer.convo = conversation
    original_pk = case.alice.pk
    original = rag_immutable_images.authorize_media_dispatch
    mutations = []
    async def changed(packet, **kwargs):
        if getattr(case.turn.answer_publication, "packet", None) is packet:
            tamper(case, change)
            mutations.append(change)
        return await original(packet, **kwargs)
    if not late:
        monkeypatch.setattr(rag_immutable_images, "authorize_media_dispatch", changed)
    try:
        with bind_turn(case.turn), source_runtime_scope(case.parent) if source_mode else nullcontext():
            assert async_to_sync(run_manual_search_turn)(case.consumer, provider, conversation) == "handled"
            if late:
                create.assert_called_once()
                assert citation in case.consumer.convo[-1].content
                tamper(case, change)
            else:
                create.assert_not_called()
                assert mutations == [change]
                assert case.consumer.convo[-1].content == LIMITED_ANSWER
            sinks(case, rejected=late)
            assert case.parent._frozen_retrieval_scope.representation is case.snapshot
    finally:
        case.alice.pk = original_pk
        case.consumer.user = case.alice


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("change", ["different_user", "same_pk_object", "binding_deleted", "binding_replaced"])
def test_default_text_original_identity_is_checked_without_media(principals, monkeypatch, late, change):
    from apps.chat.services.rag_answer_publication import bind_legacy_answer_packet
    from apps.chat.services.rag_evidence import build_evidence_packet
    from apps.chat.services.rag_synthesis import synthesize_from_evidence
    case = principals
    document, chunk = case.documents[0], case.chunks[0]
    provider, create, citation = sdk_for(document, chunk)
    raw = {"result": [{"doc_id": str(document.id), "chunk_id": chunk.pk,
        "chunk": 0, "text": chunk.content, "citation": citation}]}
    case.consumer.convo = Conversation(system="s", messages=[UserMessage(content="Explain source"),
        ToolMessage(tool_name="vector_search", for_whom="assistant", content=str(raw), result_dict=raw)])
    packet = build_evidence_packet(raw, query="source", search_scope="selected")
    original = rag_immutable_images.authorize_media_dispatch
    async def changed(packet):
        tamper(case, change)
        return await original(packet)
    if not late:
        monkeypatch.setattr(rag_immutable_images, "authorize_media_dispatch", changed)
    with bind_turn(case.turn):
        async_to_sync(bind_legacy_answer_packet)(case.consumer, packet)
        assert case.turn.answer_publication._principal_binding is case.parent._media_principal_binding
        case.consumer.convo = async_to_sync(synthesize_from_evidence)(provider, case.consumer.convo, packet)
        if late:
            create.assert_called_once()
            assert citation in case.consumer.convo[-1].content
            tamper(case, change)
        else:
            create.assert_not_called()
            assert case.consumer.convo[-1].content == LIMITED_ANSWER
        sinks(case, rejected=late)


def test_actual_source_inner_dispatch_rejects_deleted_bound_context(principals, monkeypatch):
    case = principals
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    document, chunk = case.documents[1], case.chunks[1]
    provider, create, citation = sdk_for(document, chunk)
    conversation = Conversation(system="s", messages=[UserMessage(content=f"/search {document.id} requested source")])
    case.consumer.convo = conversation
    original = rag_immutable_images.authorize_media_dispatch
    removed = []
    async def changed(packet, *, pre_seal=False):
        if not pre_seal:
            context = case.turn.answer_publication
            assert context is not None and context.packet is packet
            assert context._principal_binding is case.parent._media_principal_binding
            assert context.lease is case.turn.budget._synthesis
            removed.append(context)
            case.turn.answer_publication = None
        return await original(packet, pre_seal=pre_seal)
    monkeypatch.setattr(rag_immutable_images, "authorize_media_dispatch", changed)
    with bind_turn(case.turn), source_runtime_scope(case.parent):
        assert async_to_sync(run_manual_search_turn)(case.consumer, provider, conversation) == "handled"
        create.assert_not_called()
        assert len(removed) == 1
        assert case.consumer.convo[-1].content == LIMITED_ANSWER
        assert citation not in str(case.consumer.convo.model_dump())
        assert chunk.content not in str(case.consumer.convo.model_dump())
        assert case.parent._frozen_retrieval_scope.representation is case.snapshot


@pytest.mark.parametrize("change", [None, "deleted", "replaced", "classification_deleted"])
def test_source_child_cannot_lose_factory_record_before_first_retention(principals, monkeypatch, change):
    from dataclasses import replace
    from apps.chat.services import rag_synthesis
    from apps.documents.services.source_loading import current_source_runtime
    case = principals
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    document, chunk = case.documents[1], case.chunks[1]
    provider, create, citation = sdk_for(document, chunk)
    conversation = Conversation(system="s", messages=[UserMessage(content=f"/search {document.id} requested source")])
    case.consumer.convo = conversation
    original = rag_synthesis.finalize_source_support
    mutations = []
    async def changed(packet):
        child = current_source_runtime()
        assert child is not case.parent and child.budget is case.turn.budget
        assert child.authorization.selected_document_ids == frozenset((document.id,))
        assert case.turn.answer_publication is None
        record = case.turn._manual_representation_children[id(child)]
        assert record.runtime is child and record.parent is case.parent
        assert child._manual_factory_record is record
        assert not hasattr(packet, "_publication_manual_record")
        case.turn.budget.check_active()
        if change in ("deleted", "classification_deleted"):
            del case.turn._manual_representation_children[id(child)]
            if change == "classification_deleted":
                del child._manual_factory_record
        elif change == "replaced":
            case.turn._manual_representation_children[id(child)] = replace(record)
        mutations.append(record)
        return await original(packet)
    monkeypatch.setattr(rag_synthesis, "finalize_source_support", changed)
    with bind_turn(case.turn), source_runtime_scope(case.parent):
        assert async_to_sync(run_manual_search_turn)(case.consumer, provider, conversation) == "handled"
        assert len(mutations) == 1
        if change is None:
            create.assert_called_once()
            assert chunk.content in str(create.call_args.kwargs["messages"])
            assert citation in case.consumer.convo[-1].content
            assert case.turn.answer_publication._manual_record is mutations[0]
        else:
            create.assert_not_called()
            assert case.consumer.convo[-1].content == LIMITED_ANSWER
            assert citation not in str(case.consumer.convo.model_dump())
        sinks(case, rejected=False, limited=change is not None)
        assert case.parent._frozen_retrieval_scope.representation is case.snapshot
