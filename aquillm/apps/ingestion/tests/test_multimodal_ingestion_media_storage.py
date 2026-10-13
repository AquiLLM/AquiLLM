"""Media bytes remain distinct from text and exact source-owned figure parents."""
from dataclasses import replace
from hashlib import sha256
import pytest
from apps.documents.models import DocumentFigure, ImageUploadDocument, MediaUploadDocument, RawTextDocument
from apps.ingestion.models import SourceArtifact
from aquillm.tasks import ingest_uploaded_file_task
from .output_support import upload

pytestmark = pytest.mark.django_db(transaction=True)


def test_ingest_task_dual_saves_media_and_text(upload, monkeypatch):
    monkeypatch.setattr('aquillm.ingestion.parsers.extract_text_from_image',
        lambda *a, **k: {'extracted_text': 'diagram text', 'provider': 'fake', 'model': 'ocr'})
    monkeypatch.setattr('aquillm.ingestion.parsers.transcribe_media_bytes', lambda *a, **k: 'meeting transcript')
    item = upload((('whiteboard.png', b'image-pixels'), ('meeting.mp3', b'audio-samples'),
                   ('project-notes.txt', b'plain text notes')))
    ingest_uploaded_file_task.run(item.pk)
    item.refresh_from_db()
    assert item.status == 'partial'  # Existing OCR/transcription has no native structure.
    assert len(item.document_ids) == 3
    image, audio = ImageUploadDocument.objects.get(), MediaUploadDocument.objects.get()
    assert image.image_file.read() == b'image-pixels'
    assert audio.media_file.read() == b'audio-samples'
    assert image.full_text == 'diagram text' and audio.full_text == 'meeting transcript'
    assert RawTextDocument.objects.get().full_text == 'plain text notes'
    assert all(row['status'] == 'success' for row in item.parser_metadata['outputs'])
    assert all(row['source_occurrence_id'] for row in item.parser_metadata['outputs'])


def figures(monkeypatch, captions):
    from aquillm.ingestion import parsers
    original = parsers.extract_text_payloads
    def extract(*args, **kwargs):
        outputs = original(*args, **kwargs)
        if str(args[0]).endswith('.zip'):
            for index, caption in enumerate(captions):
                outputs.append(replace(outputs[0], normalized_type='document_figure', modality='image',
                    title=f'Figure {index}', full_text=caption, media_bytes=f'pixels-{index}'.encode(),
                    metadata={'figure_index': index, 'source_format': 'pptx'}))
        return outputs
    monkeypatch.setattr(parsers, 'extract_text_payloads', extract)


def test_no_text_figures_keep_artifacts_and_explicit_failures(upload, monkeypatch):
    figures(monkeypatch, ['', ''])
    item = upload((('slides.txt', b'slide deck text'),))
    ingest_uploaded_file_task.run(item.pk)
    item.refresh_from_db()
    assert item.status == 'partial'
    assert RawTextDocument.objects.count() == 1 and not DocumentFigure.objects.exists()
    assert [row['error_code'] for row in item.parser_metadata['outputs']] == [None, 'empty_extraction', 'empty_extraction']
    for index in range(2):
        assert SourceArtifact.objects.filter(sha256=sha256(f'pixels-{index}'.encode()).hexdigest()).exists()


def test_duplicate_figure_text_is_collision_without_fabricated_retrieval_text(upload, monkeypatch):
    figures(monkeypatch, ['Repeated extracted caption', 'Repeated extracted caption'])
    item = upload((('slides.txt', b'slide deck text'),))
    ingest_uploaded_file_task.run(item.pk)
    item.refresh_from_db()
    assert item.status == 'partial'
    parent, figure = RawTextDocument.objects.get(), DocumentFigure.objects.get()
    assert figure.parent_document.id == parent.id
    assert figure.full_text == 'Repeated extracted caption'
    assert item.parser_metadata['outputs'][-1]['error_code'] == 'content_hash_collision'
    assert len(item.document_ids) == 2


def test_distinct_figures_publish_two_exact_children(upload, monkeypatch):
    figures(monkeypatch, ['first caption', 'second caption'])
    item = upload((('slides.txt', b'slide deck text'),))
    ingest_uploaded_file_task.run(item.pk)
    item.refresh_from_db()
    assert item.status == 'success' and len(item.document_ids) == 3
    parent = RawTextDocument.objects.get()
    children = list(DocumentFigure.objects.all())
    assert len(children) == 2
    assert len({child.collection_id for child in children}) == 1
    assert all(child.parent_document.id == parent.id for child in children)
    assert children[0].collection.parent_id == parent.collection_id
