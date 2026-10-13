"""Real native extraction/activation observations; no handwritten structural rows."""
from types import SimpleNamespace

import pytest
from django.apps import apps

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import DocumentRepresentationState, RawTextDocument
from apps.documents.services.representation_activation import build_document_representation
from apps.documents.services.representation_inputs import configured_chunker_key, extraction_input
from apps.documents.tasks.chunking import create_chunks
from apps.documents.tests.representation_support import reference
from aquillm.models import WSConversation
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
from .cross_lane_graph_route_support import configure_graph_route
from .structural_native_files import document_bytes, scrambled_pdf_bytes, BODY


def observe_source(doc, *, extraction=None):
    from hashlib import sha256
    from lib.embeddings.provenance import input_digest
    doc.refresh_from_db()
    generation = DocumentRepresentationState.objects.get(document_id=doc.id).active_generation
    chunks = tuple(doc.chunks.order_by('chunk_number'))
    assert chunks and all(row.representation_generation_id == generation.pk for row in chunks)
    assert all(row.content == doc.full_text[row.start_position:row.end_position] for row in chunks)
    assert all(row.search_vector is not None for row in chunks)
    entries = {entry['number']: entry for entry in generation.manifest['chunks']}
    for row in chunks:
        entry = entries[row.chunk_number]
        assert entry['source_text'] == row.content
        assert entry['source_sha256'] == sha256(row.content.encode()).hexdigest()
        assert entry['input_sha256'] == input_digest(entry['embedding_text'])
    if extraction is not None:
        assert generation.input_kind == 'extraction' and doc.full_text == extraction.canonical_text
    else:
        assert generation.input_kind == 'canonical_snapshot'
        assert generation.extraction_quality == 'partial'
        assert all(row.source_kind == 'unknown' and row.section_path == [] for row in chunks)
    return SimpleNamespace(doc=doc, user=doc.ingested_by, generation=generation,
        chunks=chunks, extraction=extraction, db=WSConversation.objects.create(owner=doc.ingested_by))


def detached(source):
    from hashlib import sha256
    from apps.chat.evals.structural_order_observation import row_identity
    document_id = str(source.doc.id)
    rows = [{'doc_id': document_id, 'chunk_id': row.pk, 'generation': str(row.representation_generation_id),
        'start': row.start_position, 'end': row.end_position, 'text': row.content,
        'source_sha256': sha256(row.content.encode()).hexdigest(), 'kind': row.source_kind,
        'section_path': row.section_path} for row in source.chunks]
    sources = {document_id: {'text': source.doc.full_text, 'generation': str(source.generation.pk),
        'extraction': source.extraction, 'revision_sha256': source.generation.source_revision_sha256,
        'observed_chunks': tuple(row_identity(row) for row in rows)}}
    return rows, sources


@pytest.fixture
def native_source(upload, transport, monkeypatch, settings):
    configure_graph_route(monkeypatch, settings, transport[0])
    settings.KG_OVERLAY_ENABLED = False
    monkeypatch.setenv('KG_BUILD_ENABLED', 'false')
    monkeypatch.setenv('RAG_DIRECT_TOP_K', '3')

    def make(kind):
        if kind == 'unknown':
            seed = make('body')
            collection = Collection.objects.create(name='Canonical-only structural control')
            CollectionPermission.objects.create(user=seed.user, collection=collection, permission='EDIT')
            doc = RawTextDocument.objects.create(title='Canonical-only source', full_text=BODY,
                collection=collection, ingested_by=seed.user)
            assert create_chunks.run(str(doc.id), doc.full_text_hash, doc._meta.label_lower, doc.pkid) == 'activated'
            return observe_source(doc)
        name = 'scrambled.pdf' if kind == 'scrambled' else kind + '.docx'
        data = scrambled_pdf_bytes() if kind == 'scrambled' else document_bytes(late=kind == 'late')
        item = upload(((name, data),))
        run_ingest_uploaded_file(item.pk)
        output = item.outputs.get(status='success', output_kind='text')
        doc = apps.get_model(output.document_model).objects.get(pk=output.document_pkid)
        _, revision, extraction = extraction_input(reference(doc), output.extraction_revision_id, using='default')
        assert build_document_representation(reference(doc), extraction_revision_id=revision.pk,
            chunker_key=configured_chunker_key(), using='default') == 'activated'
        result = observe_source(doc, extraction=extraction)
        result.output, result.raw_bytes = output, data
        assert result.generation.extraction_revision_id == revision.pk
        return result
    return make
