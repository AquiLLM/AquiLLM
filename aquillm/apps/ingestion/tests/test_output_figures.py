"""Figure retries retain exact parent output identity and separate media bytes."""
from dataclasses import replace
import pytest
from apps.documents.models import DocumentFigure, RawTextDocument
from .output_support import upload

pytestmark = pytest.mark.django_db(transaction=True)


def figure_parser(monkeypatch):
    from aquillm.ingestion import parsers
    original = parsers.extract_text_payloads
    def extract(*args, **kwargs):
        outputs = original(*args, **kwargs)
        # Only append at the outer worker call, not recursively per ZIP member.
        if str(args[0]).endswith('.zip'):
            outputs.append(replace(outputs[0], title='Figure', full_text='figure caption',
                normalized_type='document_figure', modality='image', media_bytes=b'figure-pixels',
                metadata={'figure_index': 0, 'source_format': 'pdf'}))
        return outputs
    monkeypatch.setattr(parsers, 'extract_text_payloads', extract)


def test_figure_creation_after_parent_checkpoint_retry(upload, monkeypatch):
    from aquillm import task_ingest_uploaded as task
    from apps.ingestion.models import IngestionOutput, IngestionBatch, SourceArtifact
    figure_parser(monkeypatch)
    item = upload((('one.txt', b'parent source'),))
    original = task.publish_output
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise SystemExit('parent committed')
    monkeypatch.setattr(task, 'publish_output', crash)
    with pytest.raises(SystemExit):
        task.run_ingest_uploaded_file(item.pk)
    parent = RawTextDocument.objects.get()
    monkeypatch.setattr(task, 'publish_output', original)
    task.run_ingest_uploaded_file(item.pk)
    figure = DocumentFigure.objects.get()
    assert figure.parent_document.pkid == parent.pkid
    assert figure.parent_document.id == parent.id
    parent_output, figure_output = list(IngestionOutput.objects.order_by('pk'))
    assert figure_output.parent_output_id == parent_output.pk
    assert figure_output.extraction_revision_id == parent_output.extraction_revision_id
    assert figure_output.payload['media_artifact']['sha256'] != figure_output.payload['source_artifact']['sha256']
    assert figure.image_file.read() == b'figure-pixels'
    count = SourceArtifact.objects.count()
    IngestionBatch.objects.filter(pk=item.batch_id).delete()
    assert not IngestionOutput.objects.exists()
    assert SourceArtifact.objects.count() == count


def test_figure_parent_source_cannot_be_rebound(upload, monkeypatch):
    from django.db import DatabaseError, connection, transaction
    from apps.ingestion.models import IngestionOutput
    from apps.ingestion.services.output_manifest import prepare_output_manifest
    figure_parser(monkeypatch)
    item = upload()
    prepare_output_manifest(item.pk, using='default')
    parent1, parent2, figure = list(IngestionOutput.objects.order_by('pk'))
    assert figure.parent_output_id == parent1.pk
    with pytest.raises(DatabaseError), transaction.atomic(), connection.cursor() as cursor:
        cursor.execute('UPDATE aquillm_ingestionoutput SET parent_output_id=%s WHERE id=%s', [parent2.pk, figure.pk])


def test_unmatched_figure_revision_is_explicit_failure_and_cannot_reset(upload, monkeypatch):
    from aquillm.ingestion import parsers
    from apps.ingestion.models import IngestionOutput
    from apps.ingestion.services.output_manifest import prepare_output_manifest
    from apps.ingestion.services.output_ledger import reset_output
    original = parsers.extract_text_payloads
    def extract(*args, **kwargs):
        outputs = original(*args, **kwargs)
        if str(args[0]).endswith('.zip'):
            outputs.append(replace(outputs[0], full_text='figure caption', extraction=None,
                normalized_type='document_figure', modality='image', media_bytes=b'pixels',
                metadata={'figure_index': 0, 'source_format': 'pdf'}))
        return outputs
    monkeypatch.setattr(parsers, 'extract_text_payloads', extract)
    item = upload((('one.txt', b'parent source'),))
    prepare_output_manifest(item.pk, using='default')
    figure = IngestionOutput.objects.get(output_kind='figure')
    assert figure.status == 'error' and figure.error_code == 'parent_output_unavailable'
    assert not reset_output(figure.output_key, generation=figure.generation,
        extraction_revision_id=figure.extraction_revision_id, using='default')
