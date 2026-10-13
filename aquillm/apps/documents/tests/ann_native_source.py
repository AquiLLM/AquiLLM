"""Native paragraph geometry through real I3 and current I5 representation publication."""
from hashlib import sha256
from io import BytesIO

from django.apps import apps
from docx import Document

from apps.documents.models import DocumentRepresentationState
from apps.documents.services.representation_activation import build_document_representation
from apps.documents.services.representation_inputs import configured_chunker_key, extraction_input
from apps.documents.tests.representation_support import reference
from apps.ingestion.models import IngestionBatch
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
from lib.embeddings.provenance import input_digest


def document(upload, actor, collection, texts, title):
    word = Document()
    for text in texts:
        word.add_paragraph(text.removesuffix('\n'))
    stream = BytesIO()
    word.save(stream)
    raw = stream.getvalue()
    item = upload(((title + '.docx', raw),))
    item.batch = IngestionBatch.objects.create(user=actor, collection=collection)
    item.save(update_fields=['batch'])
    assert collection.user_can_edit(actor)
    run_ingest_uploaded_file(item.pk)
    outcomes = list(item.outputs.values_list("status", "output_kind", "error_code"))
    assert any(status == "success" and kind == "text" for status, kind, _ in outcomes), outcomes
    output = item.outputs.get(status='success', output_kind='text')
    doc = apps.get_model(output.document_model).objects.get(pk=output.document_pkid)
    _, revision, extraction = extraction_input(reference(doc), output.extraction_revision_id, using='default')
    assert extraction.parser_key == 'native-docx-ordered-v2' and extraction.quality == 'complete'
    assert len(extraction.elements) == len(texts) and all(e.kind == 'paragraph' for e in extraction.elements)
    assert build_document_representation(reference(doc), extraction_revision_id=revision.pk,
        chunker_key=configured_chunker_key(), using='default') == 'activated'
    doc.refresh_from_db()
    generation = DocumentRepresentationState.objects.get(document_id=doc.id).active_generation
    chunks = tuple(doc.chunks.order_by('chunk_number'))
    assert len(chunks) == len(texts)
    expected = [text.removesuffix('\n') + ('\n' if i + 1 < len(texts) else '')
                for i, text in enumerate(texts)]
    assert [row.content for row in chunks] == expected
    assert doc.full_text == extraction.canonical_text == ''.join(expected)
    for row, element, entry in zip(chunks, extraction.elements, generation.manifest['chunks'], strict=True):
        assert row.representation_generation_id == generation.pk
        assert (row.start_position, row.end_position) == (element.start, element.end)
        assert row.content == doc.full_text[element.start:element.end] == entry['source_text']
        assert entry['embedding_text'] == row.content
        assert entry['source_sha256'] == sha256(row.content.encode()).hexdigest()
        assert entry['input_sha256'] == input_digest(entry['embedding_text'])
    assert generation.extraction_revision_id == revision.pk
    assert extraction.artifact.sha256 == sha256(raw).hexdigest()
    provenance = {'artifact_sha256': sha256(raw).hexdigest(), 'artifact_bytes': raw,
        'extraction_revision_id': revision.pk, 'extraction_sha256': revision.manifest_sha256,
        'representation_key': str(generation.pk), 'representation_sha256': generation.manifest_sha256}
    return doc, chunks, provenance
