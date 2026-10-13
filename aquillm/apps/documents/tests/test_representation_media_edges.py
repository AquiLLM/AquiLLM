"""Storage corruption and sealed-phase limits do not create fresh image authority."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync
from django.core.files.storage import default_storage
from django.test import Client

from apps.chat.services.rag_answer_publication import _freeze_legacy_packet, bind_packet_publication
from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.services.rag_publication_images import capture_publication_images
from apps.documents.services.representation_media_acquisition import chunk_media_url
from apps.documents.services.source_loading import source_runtime_scope
from lib.llm.answer_publication import AnswerPublicationRejected
from lib.llm.turn_context import bind_turn
from lib.retrieval.synthesis_budget import seal_synthesis
from lib.tools.search.vector_search import pack_chunk_search_results
from .representation_support import upload as upload
from .test_representation_media import figure as figure
from .test_representation_media_publication import prepare

pytestmark = pytest.mark.django_db(transaction=True)


def pack(doc, chunk):
    return pack_chunk_search_results((chunk,), titles_by_doc_id={doc.id: doc.title},
        docs_by_doc_id={doc.id: doc}, truncate=lambda value: value, image_modality="image",
        include_images=True, compact_items=False, media_url_resolver=chunk_media_url)


@pytest.mark.parametrize("source_mode", [False, True])
def test_corruption_before_acquisition_never_uses_mutable_fallback(figure, monkeypatch, source_mode):
    from lib.tools.search import vector_search
    doc, output, _, _, turn, runtime = prepare(figure, source_mode)
    key = output.payload["media_artifact"]["storage_key"]
    with default_storage.open(key, "wb") as target:
        target.write(b"corruption")
    def forbidden(*args, **kwargs):
        raise AssertionError("mutable image fallback")
    monkeypatch.setattr(vector_search, "_doc_has_renderable_image", forbidden)
    with bind_turn(turn), source_runtime_scope(runtime) if source_mode else nullcontext():
        result = pack(doc, doc.chunks.get())
        assert not result["result"][0].get("image_url")
        assert not getattr(runtime, "_media_captures", {})


@pytest.mark.parametrize("source_mode", [False, True])
@pytest.mark.parametrize("change", ["storage", "foreign_lease", "expired_lease", "collection_ref"])
def test_sealed_media_checks_exact_lease_and_never_reloads_pixels(figure, monkeypatch, source_mode, change):
    doc, output, viewer, _, turn, runtime = prepare(figure, source_mode)
    chunk = doc.chunks.get()
    with bind_turn(turn), source_runtime_scope(runtime) if source_mode else nullcontext():
        packet = build_evidence_packet(pack(doc, chunk), query="caption", search_scope="selected")
        assert _freeze_legacy_packet(SimpleNamespace(user=viewer), packet, (doc.collection_id,), turn.budget)
        async_to_sync(capture_publication_images)(packet)
        context = bind_packet_publication(packet)
        context.lease = seal_synthesis(turn.budget, "captured request", calls=1, output_tokens=256,
            timeout_seconds=60, authority=packet.source_authorization)
        if change == "storage":
            with default_storage.open(output.payload["media_artifact"]["storage_key"], "wb") as target:
                target.write(b"corruption after capture")
            client = Client()
            client.force_login(viewer)
            assert client.get(packet.image_urls[0]).status_code == 404
        elif change == "foreign_lease":
            context.lease = replace(context.lease)
        elif change == "expired_lease":
            object.__setattr__(context.lease, "deadline", turn.budget._clock() - 1)
        elif not source_mode:
            runtime._media_consumer.col_ref = SimpleNamespace(collections=[])
        else:
            runtime.authorization = replace(runtime.authorization, selected_document_ids=frozenset())
        def forbidden(*args, **kwargs):
            raise AssertionError("publication reopened image storage")
        monkeypatch.setattr(default_storage, "open", forbidden)
        if change == "storage":
            async_to_sync(context._authorized)()
            assert packet._media_bindings[0].data == b"immutable-image-pixels"
        else:
            with pytest.raises((AnswerPublicationRejected, ValueError)):
                async_to_sync(context._authorized)()


def test_default_whole_document_preserves_selected_captured_image_in_request(figure, monkeypatch):
    from apps.chat.services.tool_wiring.documents import whole_document_tool
    from apps.chat.services.rag_publication_legacy import legacy_tool_handoff
    from lib.llm.types.conversation import Conversation
    from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
    monkeypatch.setenv("LLM_TOOL_INLINE_IMAGES", "1")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "65536")
    doc, _, viewer, _, turn, _ = prepare(figure, False)
    chunk = doc.chunks.get()
    citation = f"[doc:{doc.id} chunk:{chunk.pk}]"
    calls = []
    class Provider:
        async def token_count(self, *args):
            return 64
        async def complete(self, request, *args, **kwargs):
            calls.append(request[-1].get_images())
            assert request[-1].has_images()
            return request + [AssistantMessage(content=chunk.content + " " + citation, stop_reason="stop")], "stop"
    provider = Provider()
    convo = Conversation(system="s", messages=[UserMessage(content="Show the image")])
    consumer = turn._media_source_runtime._media_consumer
    consumer.llm_if, consumer.convo = provider, convo
    with bind_turn(turn):
        result = whole_document_tool(viewer, SimpleNamespace(chat=consumer), consumer.col_ref)(
            doc_id=str(doc.id), include_images=True)
        assert result["result"]["image_url"].endswith(f"?chunk_id={chunk.pk}")
        convo = convo + [ToolMessage(tool_name="whole_document", for_whom="assistant", content="result",
            arguments={"doc_id": str(doc.id)}, result_dict=result)]
        answer, _ = async_to_sync(legacy_tool_handoff)(consumer, provider, convo, 256, None)
        assert calls and calls[0] and citation in answer[-1].content
