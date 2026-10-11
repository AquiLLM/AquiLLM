"""Verified captured pixels travel only in the current request and fence every sink."""
import base64
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync
from django.contrib.auth.models import User

from apps.chat.services.rag_answer_publication import _freeze_legacy_packet, bind_packet_publication
from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.services.rag_evidence_handoff import prepare_evidence_handoff
from apps.chat.services.rag_publication_images import capture_publication_images
from apps.collections.models import CollectionPermission
from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
from apps.documents.services.representation_activation import build_document_representation
from apps.documents.services.representation_caption import caption_chunker_key
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from aquillm.message_adapters import save_conversation_to_db
from aquillm.models import WSConversation
from lib.llm.answer_publication import AnswerPublicationRejected, bind_answer
from lib.llm.turn_context import TurnContext, bind_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from lib.tools.search.vector_search import pack_chunk_search_results
from .representation_support import reference, upload as upload
from .test_representation_media import figure as figure

pytestmark = pytest.mark.django_db(transaction=True)


def prepare(figure, source_mode):
    from apps.documents.services.representation_media_acquisition import initialize_media_runtime
    doc, output, _ = figure
    assert build_document_representation(reference(doc), extraction_revision_id=output.extraction_revision_id,
        chunker_key=caption_chunker_key(), using="default") == "activated"
    doc.refresh_from_db()
    viewer = User.objects.create_user(username="media-viewer")
    permission = CollectionPermission.objects.create(user=viewer, collection=doc.collection, permission="VIEW")
    authority = build_selected_scope_authorization_context(principal=viewer,
        selected_collection_ids=(doc.collection_id,), selected_documents=(doc,))
    turn = TurnContext(TurnBudget(TurnLimits()))
    consumer = SimpleNamespace(user=viewer, col_ref=SimpleNamespace(collections=[doc.collection_id]))
    runtime = SourceRuntime(turn.budget, authority) if source_mode else initialize_media_runtime(consumer, turn)
    return doc, output, viewer, permission, turn, runtime


@pytest.mark.parametrize("source_mode", [False, True])
@pytest.mark.parametrize("change", [None, "pointer", "path", "bytes", "binding", "permission"])
def test_immutable_image_provider_transport_and_actual_publication_sinks(figure, monkeypatch, source_mode, change):
    from apps.documents.services.representation_media_acquisition import chunk_media_url
    from apps.chat.consumers.chat_publish import end_spin_publish
    from lib.llm.providers.final_stream import send_final_answer_stream
    monkeypatch.setenv("LLM_TOOL_INLINE_IMAGES", "1")
    doc, output, viewer, permission, turn, runtime = prepare(figure, source_mode)
    chunk = doc.chunks.get()
    db = WSConversation.objects.create(owner=viewer)
    events = []
    async def stream(payload):
        events.append(payload)
    async def send(**kwargs):
        events.append(kwargs)
    with bind_turn(turn), source_runtime_scope(runtime) if source_mode else nullcontext():
        result = pack_chunk_search_results((chunk,), titles_by_doc_id={doc.id: doc.title},
            docs_by_doc_id={doc.id: doc}, truncate=lambda value: value, image_modality="image",
            include_images=True, compact_items=False, media_url_resolver=chunk_media_url)
        url = f"/aquillm/document_image/{doc.id}/?chunk_id={chunk.pk}"
        assert result["result"][0]["image_url"] == url
        packet = build_evidence_packet(result, query="figure caption", search_scope="selected")
        assert _freeze_legacy_packet(SimpleNamespace(user=viewer), packet, (doc.collection_id,), turn.budget)
        async_to_sync(capture_publication_images)(packet)
        context = bind_packet_publication(packet)
        convo = Conversation(system="s", messages=[ToolMessage(tool_name="vector_search", for_whom="assistant",
            content="tool result", result_dict=result)])
        persisted, request = prepare_evidence_handoff(convo, packet)
        assert "_images" not in persisted[-1].result_dict
        parts = request[-1].render()["content"]
        data_url = next(part["image_url"]["url"] for part in parts if part["type"] == "image_url")
        assert base64.b64decode(data_url.split(",", 1)[1]) == b"immutable-image-pixels"
        monkeypatch.delenv("LLM_TOOL_INLINE_IMAGES")
        _, disabled = prepare_evidence_handoff(convo, packet)
        assert not disabled[-1].has_images() and "_images" not in disabled[-1].result_dict
        monkeypatch.setenv("LLM_TOOL_INLINE_IMAGES", "1")
        citation = f"[doc:{doc.id} chunk:{chunk.pk}]"
        answer = async_to_sync(context.finalize)(chunk.content + " " + citation + f"\n![Figure]({url})")
        assert answer.outcome != "limited" and url in answer.content
        message = bind_answer(AssistantMessage(content=answer.content, stop_reason="stop"), answer, context)
        delivered = Conversation(system="s", messages=[message])
        if change == "pointer":
            assert build_document_representation(reference(doc), extraction_revision_id=output.extraction_revision_id,
                chunker_key=caption_chunker_key(), using="default") == "activated"
        elif change == "path":
            doc.image_file = "replacement.png"
            doc.save(dont_rechunk=True, update_fields=["image_file"])
        elif change == "bytes":
            object.__setattr__(packet._media_bindings[0], "data", b"substituted")
        elif change == "binding":
            packet._media_bindings = tuple(list(packet._media_bindings))
        elif change == "permission":
            permission.delete()
        calls = (
            lambda: async_to_sync(send_final_answer_stream)(stream, message_uuid=str(message.message_uuid),
                content=message.content, stop_reason="stop", usage=0, message=message),
            lambda: async_to_sync(end_spin_publish)(SimpleNamespace(convo=delivered, _spin_epoch=0, send=send), delivered),
            lambda: save_conversation_to_db(delivered, db),
        )
        for call in calls:
            if change:
                with pytest.raises(AnswerPublicationRejected):
                    call()
            else:
                call()
    assert len(events) == (0 if change else 2)
    assert db.db_messages.count() == (0 if change else 1)
