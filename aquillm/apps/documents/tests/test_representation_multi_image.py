"""Actual SDK image parts carry exact selected source attribution beside pixels."""
import base64
from copy import copy
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync
from django.contrib.auth.models import User

from apps.chat.services.rag_answer_publication import _freeze_legacy_packet, bind_packet_publication
from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.services.rag_publication_images import capture_publication_images
from apps.chat.services.rag_synthesis import synthesize_from_evidence
from apps.collections.models import CollectionPermission
from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
from apps.documents.models import DocumentFigure, RawTextDocument, TextChunk
from apps.documents.services.representation_activation import build_document_representation
from apps.documents.services.representation_caption import caption_chunker_key
from apps.documents.services.representation_media_acquisition import chunk_media_url, initialize_media_runtime
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
from lib.llm.providers.openai import OpenAIInterface
from lib.llm.turn_context import TurnContext, bind_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from lib.tools.search.vector_search import pack_chunk_search_results
from .representation_support import reference, upload as upload

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def images(upload, monkeypatch):
    from aquillm.ingestion import parsers
    original = parsers.extract_text_payloads
    def extract(*args, **kwargs):
        outputs = original(*args, **kwargs)
        if str(args[0]).endswith(".zip"):
            for index, parent_payload in enumerate(tuple(outputs)):
                outputs.append(replace(parent_payload, title="Figure", full_text="Identical caption",
                    normalized_type="document_figure", modality="image", media_bytes=f"pixels-{index}".encode(),
                    media_content_type="image/png", metadata={"figure_index": index, "source_format": "pdf"}))
        return outputs
    monkeypatch.setattr(parsers, "extract_text_payloads", extract)
    item = upload((("parent.txt", b"Independent text row"), ("second.txt", b"Second parent source")))
    run_ingest_uploaded_file(item.pk)
    figures = []
    for output in item.outputs.filter(output_kind="figure"):
        assert output.status == "success", output.error_code
        doc = DocumentFigure.objects.get(pk=output.document_pkid)
        assert build_document_representation(reference(doc), extraction_revision_id=output.extraction_revision_id,
            chunker_key=caption_chunker_key(), using="default") == "activated"
        doc.refresh_from_db()
        figures.append((doc, doc.chunks.get(), f"pixels-{output.payload['figure_index']}".encode()))
    assert len(figures) == 2
    output = item.outputs.filter(output_kind="text").order_by("pk").first()
    parent = RawTextDocument.objects.get(pk=output.document_pkid)
    text = TextChunk.objects.bulk_create([TextChunk(doc_id=parent.id, content=parent.full_text,
        chunk_number=0, start_position=0, end_position=len(parent.full_text))])[0]
    figures.sort(key=lambda row: str(row[0].id), reverse=True)
    return figures, parent, text


@pytest.mark.parametrize("source_mode", [False, True])
def test_two_images_and_interleaved_text_keep_exact_sdk_source_mapping(images, monkeypatch, source_mode):
    monkeypatch.setenv("LLM_TOOL_INLINE_IMAGES", "1")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "65536")
    monkeypatch.setenv("RAG_EVIDENCE_TOKEN_BUDGET", "12000")
    figures, parent, text = images
    viewer = User.objects.create_user(username="two-image-viewer")
    documents = (figures[0][0], parent, figures[1][0])
    chunks = (figures[0][1], text, figures[1][1])
    collections = tuple(sorted({doc.collection_id for doc in documents}))
    for collection_id in collections:
        CollectionPermission.objects.create(user=viewer, collection_id=collection_id, permission="VIEW")
    turn = TurnContext(TurnBudget(TurnLimits()))
    authority = build_selected_scope_authorization_context(principal=viewer,
        selected_collection_ids=collections, selected_documents=documents)
    assert authority.selected_document_ids == frozenset(doc.id for doc in documents)
    runtime = SourceRuntime(turn.budget, authority) if source_mode else initialize_media_runtime(
        SimpleNamespace(user=viewer, col_ref=SimpleNamespace(collections=list(collections))), turn)
    calls = []
    answer = "Identical caption " + " ".join(f"[doc:{doc.id} chunk:{chunk.pk}]" for doc, chunk, _ in figures)
    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer, tool_calls=None),
            finish_reason="stop")], usage=None)
    provider = OpenAIInterface(SimpleNamespace(base_url="http://fake-models:8000/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))), "test")
    with bind_turn(turn), source_runtime_scope(runtime) if source_mode else nullcontext():
        result = pack_chunk_search_results(chunks, titles_by_doc_id={doc.id: doc.title for doc in documents},
            docs_by_doc_id={doc.id: doc for doc in documents}, truncate=lambda value: value,
            image_modality="image", include_images=True, compact_items=False, media_url_resolver=chunk_media_url)
        packet = build_evidence_packet(result, query="compare images", search_scope="selected")
        assert _freeze_legacy_packet(SimpleNamespace(user=viewer), packet, collections, turn.budget)
        async_to_sync(capture_publication_images)(packet)
        from apps.chat.services.rag_immutable_images import request_images
        from apps.documents.services.representation_contracts import SourceScopeUnavailable
        for invalid_rows in (packet.chunks[:-1], packet.chunks + [packet.chunks[0]]):
            invalid = copy(packet)
            invalid.chunks = invalid_rows
            with pytest.raises(SourceScopeUnavailable, match="media_capture_changed"):
                request_images(invalid)
        if not source_mode:
            bind_packet_publication(packet)
        convo = Conversation(system="s", messages=[ToolMessage(tool_name="vector_search", for_whom="assistant",
            content="selected", result_dict=result)])
        persisted = async_to_sync(synthesize_from_evidence)(provider, convo, packet, max_tokens=256)
        assert len(calls) == 1
        assert all("_images" not in getattr(message, "result_dict", {}) for message in persisted.messages)
    parts = [part for message in calls[0]["messages"] if isinstance(message.get("content"), list)
             for part in message["content"]]
    pixels = [(i, base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))
              for i, part in enumerate(parts) if part["type"] == "image_url"]
    assert [data for _, data in pixels] == [row[2] for row in figures]
    for (position, _), (doc, chunk, _), evidence_index in zip(pixels, figures, (1, 3)):
        assert parts[position + 1]["type"] == "text"
        label = parts[position + 1]["text"]
        assert f"[Image {evidence_index}:" in label
        assert doc.title in label
        assert f"[doc:{doc.id} chunk:{chunk.pk}]" in label
