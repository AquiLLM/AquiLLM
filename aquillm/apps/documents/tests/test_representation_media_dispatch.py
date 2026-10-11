"""Actual provider boundaries retain captured pixels and the original issuer."""
import base64
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync, sync_to_async

from apps.chat.services.rag_answer_publication import _freeze_legacy_packet, bind_packet_publication
from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.services.rag_publication_images import capture_publication_images
from apps.documents.services.representation_activation import build_document_representation
from apps.documents.services.representation_caption import caption_chunker_key
from apps.documents.services.representation_media_acquisition import chunk_media_url
from apps.documents.services.source_loading import source_runtime_scope
from lib.llm.answer_publication import LIMITED_ANSWER
from lib.llm.turn_context import bind_turn, current_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage
from lib.tools.search.vector_search import pack_chunk_search_results
from .representation_support import reference, upload as upload
from .test_representation_media import figure as figure
from .test_representation_media_publication import prepare

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("source_mode", [False, True])
@pytest.mark.parametrize("change", [None, "permission", "pointer", "artifact"])
def test_actual_dispatch_rechecks_after_handoff_and_retains_issuer(figure, monkeypatch, source_mode, change):
    from apps.chat.services import rag_synthesis, rag_immutable_images
    from apps.ingestion.models import SourceArtifact
    from aquillm.models import WSConversation
    from aquillm.message_adapters import save_conversation_to_db
    from apps.chat.consumers.chat_publish import end_spin_publish
    from lib.llm.providers.final_stream import send_final_answer_stream
    from apps.documents.services.representation_media_acquisition import acquire_chunk_media
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    monkeypatch.setenv("LLM_TOOL_INLINE_IMAGES", "1")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "65536")
    doc, output, viewer, permission, turn, runtime = prepare(figure, source_mode)
    chunk = doc.chunks.get()
    calls, original_contexts = [], []
    transcript = WSConversation.objects.create(owner=viewer)
    events = []
    async def stream(payload):
        events.append(payload)
    async def send(**kwargs):
        events.append(kwargs)
    citation = f"[doc:{doc.id} chunk:{chunk.pk}]"
    def mutate():
        if change == "permission":
            permission.delete()
        elif change == "pointer":
            assert build_document_representation(reference(doc), extraction_revision_id=output.extraction_revision_id,
                chunker_key=caption_chunker_key(), using="default") == "activated"
        elif change == "artifact":
            # The separate media artifact is JSON-bound; loss of its registration
            # must invalidate dispatch even though already captured bytes survive.
            SourceArtifact.objects.get(sha256=output.payload["media_artifact"]["sha256"]).delete()
    class Provider:
        async def complete(self, request, *args, **kwargs):
            calls.append(request)
            context = current_turn().answer_publication
            assert context is original_contexts[0]
            assert context.lease is turn.budget._synthesis
            parts = request[-1].render()["content"]
            url = next(p["image_url"]["url"] for p in parts if p["type"] == "image_url")
            assert base64.b64decode(url.split(",", 1)[1]) == b"immutable-image-pixels"
            return request + [AssistantMessage(content=chunk.content + " " + citation, stop_reason="stop")], "stop"
    if source_mode:
        original = rag_synthesis.complete_source_request
        async def delayed_source(*args, **kwargs):
            # This boundary is after rag_synthesis's outer media authorization.
            original_contexts.append(current_turn().answer_publication)
            await sync_to_async(mutate)()
            return await original(*args, **kwargs)
        monkeypatch.setattr(rag_synthesis, "complete_source_request", delayed_source)
    else:
        original = rag_immutable_images.authorize_media_dispatch
        async def delayed_default(packet):
            original_contexts.append(current_turn().answer_publication)
            await sync_to_async(mutate)()
            return await original(packet)
        monkeypatch.setattr(rag_immutable_images, "authorize_media_dispatch", delayed_default)
    with bind_turn(turn), source_runtime_scope(runtime) if source_mode else nullcontext():
        result = pack_chunk_search_results((chunk,), titles_by_doc_id={doc.id: doc.title},
            docs_by_doc_id={doc.id: doc}, truncate=lambda value: value, image_modality="image",
            include_images=True, compact_items=False, media_url_resolver=chunk_media_url)
        packet = build_evidence_packet(result, query="caption", search_scope="selected")
        assert _freeze_legacy_packet(SimpleNamespace(user=viewer), packet, (doc.collection_id,), turn.budget)
        async_to_sync(capture_publication_images)(packet)
        if not source_mode:
            bind_packet_publication(packet)
        convo = Conversation(system="s", messages=[ToolMessage(tool_name="vector_search", for_whom="assistant",
            content="result", result_dict=result)])
        answer = async_to_sync(rag_synthesis.synthesize_from_evidence)(Provider(), convo, packet, max_tokens=256)
        assert original_contexts and current_turn().answer_publication is original_contexts[0]
        if change:
            assert not calls and answer[-1].content == LIMITED_ANSWER
            assert answer[-2].result_dict == {"result": [], "retrieval_status": "context_limited"}
        else:
            assert len(calls) == 1 and citation in answer[-1].content
        with pytest.raises(SourceScopeUnavailable, match="media_acquisition_closed"):
            acquire_chunk_media(chunk)
        message = answer[-1]
        async_to_sync(send_final_answer_stream)(stream, message_uuid=str(message.message_uuid),
            content=message.content, stop_reason="stop", usage=0, message=message)
        async_to_sync(end_spin_publish)(SimpleNamespace(convo=answer, _spin_epoch=0, send=send), answer)
        save_conversation_to_db(answer, transcript)
        assert all("_images" not in getattr(m, "result_dict", {}) for m in answer.messages)
    assert len(events) == 2 and transcript.db_messages.count() == 2
