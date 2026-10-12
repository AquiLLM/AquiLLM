"""Real retained-I3 media and original runtime for image-selection publication tests."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync
from django.contrib.auth.models import User

from apps.chat.services.rag_answer_publication import _freeze_legacy_packet, bind_packet_publication
from apps.chat.services.rag_evidence import build_evidence_packet
from apps.chat.services.rag_publication_images import capture_publication_images
from apps.collections.models import CollectionPermission
from apps.documents.models import DocumentFigure, RawTextDocument, TextChunk
from apps.documents.services.representation_activation import build_document_representation
from apps.documents.services.representation_caption import caption_chunker_key
from apps.documents.services.representation_media_acquisition import chunk_media_url, initialize_media_runtime
from apps.documents.tests.representation_support import reference
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
from lib.llm.tests.publication_fixture_support import UNCERTAINTY
from lib.llm.turn_context import TurnContext, bind_turn
from lib.retrieval.synthesis_budget import seal_synthesis
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from lib.tools.search.vector_search import pack_chunk_search_results


@pytest.fixture
def image_case(transactional_db, upload, monkeypatch):
    from aquillm.ingestion import parsers
    original = parsers.extract_text_payloads
    captions = ("Figure 1. Galaxy redshift distribution", "Figure 2. Example galaxies in grizy bands",
                "Figure 3. Deep galaxy images")
    def extract(*args, **kwargs):
        outputs = original(*args, **kwargs)
        if str(args[0]).endswith(".zip"):
            for index, parent in enumerate(tuple(outputs)):
                outputs.append(replace(parent, title=f"Figure {index + 1}", full_text=captions[index],
                    normalized_type="document_figure", modality="image", media_bytes=f"pixels-{index}".encode(),
                    media_content_type="image/png", metadata={"figure_index": index, "source_format": "pdf"}))
        return outputs
    monkeypatch.setattr(parsers, "extract_text_payloads", extract)
    parents = (b"Calibration uses flat fields.", b"Calibration uses dark frames.",
               b"Calibration uses bias frames.")
    item = upload(tuple((f"parent-{i}.txt", text) for i, text in enumerate(parents)))
    run_ingest_uploaded_file(item.pk)
    figures = []
    for output in item.outputs.filter(output_kind="figure").order_by("pk"):
        assert output.status == "success", output.error_code
        doc = DocumentFigure.objects.get(pk=output.document_pkid)
        assert build_document_representation(reference(doc), extraction_revision_id=output.extraction_revision_id,
            chunker_key=caption_chunker_key(), using="default") == "activated"
        doc.refresh_from_db()
        chunk = doc.chunks.get()
        assert chunk.representation_generation_id is not None
        figures.append((doc, chunk))
    assert len(figures) == 3
    output = item.outputs.filter(output_kind="text").order_by("pk").first()
    parent = RawTextDocument.objects.get(pk=output.document_pkid)
    text = TextChunk.objects.create(doc_id=parent.id, content=parent.full_text, chunk_number=0,
                                   start_position=0, end_position=len(parent.full_text))
    viewer = User.objects.create_user(username="image-selection-viewer")
    collection_ids = sorted({doc.collection_id for doc, _ in figures} | {parent.collection_id})
    for collection_id in collection_ids:
        CollectionPermission.objects.create(user=viewer, collection_id=collection_id, permission="VIEW")
    return SimpleNamespace(figures=figures, text=(parent, text), user=viewer, collection_ids=collection_ids)


def citation(source):
    doc, chunk = source
    return f"[doc:{doc.id} chunk:{chunk.pk}]"


def image_url(source):
    doc, chunk = source
    return f"/aquillm/document_image/{doc.id}/?chunk_id={chunk.pk}"


def attributed(sources):
    # Literal expected A2 framing. Figure captions have two sentences.
    quotes = []
    for _, chunk in sources:
        lines = chunk.content.replace(". ", ".\n").splitlines()
        quotes.append("\n".join(f"> {line} [doc:{chunk.doc_id} chunk:{chunk.pk}]" for line in lines))
    return UNCERTAINTY + "\n\n" + "\n\n".join(quotes)


@contextmanager
def captured_publication(case, *, text_only=False):
    sources = [case.text] if text_only else case.figures
    turn = TurnContext(TurnBudget(TurnLimits()))
    consumer = SimpleNamespace(user=case.user, col_ref=SimpleNamespace(collections=case.collection_ids))
    with bind_turn(turn):
        runtime = initialize_media_runtime(consumer, turn)
        original = runtime._frozen_retrieval_scope
        result = pack_chunk_search_results([chunk for _, chunk in sources],
            titles_by_doc_id={doc.id: doc.title for doc, _ in sources},
            docs_by_doc_id={doc.id: doc for doc, _ in sources}, truncate=lambda value: value,
            image_modality="image", include_images=not text_only, compact_items=False,
            media_url_resolver=chunk_media_url)
        packet = build_evidence_packet(result, query="Explain selected evidence", search_scope="selected")
        assert _freeze_legacy_packet(consumer, packet, case.collection_ids, turn.budget)
        selection = packet.selection
        async_to_sync(capture_publication_images)(packet)
        assert runtime._frozen_retrieval_scope is original
        if not text_only:
            assert len(packet._media_bindings) == 3
            assert set(packet.image_urls) == {image_url(source) for source in case.figures}
        context = bind_packet_publication(packet)
        assert turn.answer_publication is context
        assert context._representation is original.representation
        assert context._principal_runtime is runtime
        assert context._principal_binding is runtime._media_principal_binding
        assert context._authority is runtime.authorization
        assert context.packet.selection is selection
        assert tuple(c.chunk_id for c in selection.candidates) == tuple(chunk.pk for _, chunk in sources)
        if text_only:
            assert packet.image_urls == []
            assert context._media_bindings is None and context._media_runtime is None
        else:
            assert context._media_runtime is runtime
        lease = seal_synthesis(turn.budget, "image-selection", calls=5, output_tokens=20000,
                               timeout_seconds=turn.budget.remaining_ms() / 1000)
        context.lease = lease
        try:
            yield result, context
        finally:
            context.close()
            turn.budget.close("test_complete")
