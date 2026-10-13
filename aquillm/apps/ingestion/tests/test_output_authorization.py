"""Publication and status must use current scope, including legacy rows."""
import json
import pytest
from django.test import RequestFactory
from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import RawTextDocument
from apps.ingestion.views.api.uploads import ingest_uploads_status
from .output_support import upload

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize('mutation', ['revoke', 'move', 'delete'])
def test_scope_change_during_materialization_publishes_nothing(upload, monkeypatch, mutation):
    from apps.ingestion.services.output_manifest import prepare_output_manifest
    from apps.ingestion.services import output_ledger as ledger
    item = upload((('one.txt', b'body'),))
    key, = prepare_output_manifest(item.pk, using='default')
    lease = ledger.claim_output(key, using='default')
    original = ledger.verify_materialized
    def mutate(row):
        original(row)
        if mutation == 'revoke':
            CollectionPermission.objects.all().delete()
        elif mutation == 'move':
            item.batch.collection.move_to(Collection.objects.create(name='Elsewhere'))
        else:
            item.batch.collection.delete()
    monkeypatch.setattr(ledger, 'verify_materialized', mutate)
    result = ledger.publish_output(lease, using='default')
    assert result.status == 'error' and result.document_ref is None
    assert not RawTextDocument.objects.exists()


def test_api_filters_moved_committed_document(upload):
    from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
    item = upload((('one.txt', b'body'),))
    run_ingest_uploaded_file(item.pk)
    elsewhere = Collection.objects.create(name='Elsewhere')
    CollectionPermission.objects.create(user=item.batch.user, collection=elsewhere, permission='EDIT')
    RawTextDocument.objects.get().move_to(elsewhere, actor=item.batch.user)
    request = RequestFactory().get('/')
    request.user = item.batch.user
    response = json.loads(ingest_uploads_status(request, item.batch_id).content)
    assert response['items'][0]['document_ids'] == []
    assert response['items'][0]['parser_metadata']['outputs'][0]['error_code'] == 'source_scope_unavailable'


def test_legacy_terminal_item_is_not_reingested(upload):
    from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
    item = upload((('one.txt', b'body'),))
    document = RawTextDocument.objects.create(title='Legacy', full_text='old', collection=item.batch.collection, ingested_by=item.batch.user)
    item.status, item.document_ids = 'success', [str(document.id)]
    item.save(update_fields=['status', 'document_ids'])
    run_ingest_uploaded_file(item.pk)
    item.refresh_from_db()
    assert not item.manifest_complete
    assert item.document_ids == [str(document.id)]
    assert item.status == 'success' and RawTextDocument.objects.count() == 1
