"""Current authorization and partial document references are API invariants."""
import pytest
import json
from django.test import RequestFactory
from apps.collections.models import CollectionPermission
from apps.ingestion.views.api.uploads import ingest_uploads_status
from .output_support import upload

pytestmark = pytest.mark.django_db(transaction=True)


def test_revoked_owner_cannot_read_batch(upload):
    item = upload()
    CollectionPermission.objects.all().delete()
    request = RequestFactory().get('/')
    request.user = item.batch.user
    assert ingest_uploads_status(request, item.batch_id).status_code == 403


def test_api_derives_partial_ids_and_index_state_from_committed_outputs(upload):
    from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
    from apps.documents.models import RawTextDocument
    item = upload((('one.txt', b'same'), ('two.txt', b'same')))
    run_ingest_uploaded_file(item.pk)
    item.status, item.document_ids, item.parser_metadata = 'success', [], {'outputs': []}
    item.save(update_fields=['status', 'document_ids', 'parser_metadata'])
    request = RequestFactory().get('/')
    request.user = item.batch.user
    body = json.loads(ingest_uploads_status(request, item.batch_id).content)
    assert body['counts']['partial'] == 1
    row = body['items'][0]
    assert row['document_ids'] == [str(RawTextDocument.objects.get().id)]
    outputs = row['parser_metadata']['outputs']
    assert outputs[0]['index_status'] == 'pending'
    assert outputs[1]['error_code'] == 'content_hash_collision'
