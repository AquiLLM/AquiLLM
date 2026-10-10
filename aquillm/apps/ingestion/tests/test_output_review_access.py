"""Review regressions for derived child grants and historical figure identity."""
import json
import pytest
from django.contrib.auth.models import User
from django.core.exceptions import PermissionDenied
from django.test import RequestFactory
from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import DocumentFigure, RawTextDocument, TextChunk
from apps.documents.views.api import chunk_detail
from apps.documents.views.pages import document_image
from apps.ingestion.views.api.uploads import ingest_uploads_status
from aquillm.task_ingest_helpers import figures_subcollection_name
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
from .output_support import upload
from .test_output_figures import figure_parser

pytestmark = pytest.mark.django_db(transaction=True)


def request_for(user):
    request = RequestFactory().get('/')
    request.user = user
    return request


def chunk_for(figure):
    return TextChunk.objects.create(doc_id=figure.id, content=figure.full_text,
        start_position=0, end_position=len(figure.full_text), chunk_number=0, modality='image')


@pytest.mark.parametrize('direct_grant', [True, False])
def test_figure_destination_cannot_widen_text_or_media_readers(upload, monkeypatch, direct_grant):
    figure_parser(monkeypatch)
    item = upload((('one.txt', b'private parent'),))
    outsider = User.objects.create_user(username='child-only-reader')
    child = Collection.objects.create(parent=item.batch.collection, name=figures_subcollection_name('one'))
    if direct_grant:
        CollectionPermission.objects.create(user=outsider, collection=child, permission='VIEW')
        existing = RawTextDocument.objects.create(title='Existing shared text', full_text='already shared',
            collection=child, ingested_by=item.batch.user)
    run_ingest_uploaded_file(item.pk)
    figures = list(DocumentFigure.objects.all())
    outsider_request = request_for(outsider)
    # Exercise the real text and binary serving boundaries if publication occurred.
    for figure in figures:
        chunk = chunk_for(figure)
        text_status = chunk_detail(outsider_request, chunk.pk).status_code
        try:
            media_status = document_image(outsider_request, figure.id).status_code
        except PermissionDenied:
            media_status = 403
        assert (text_status, media_status) == (403, 403)
    if direct_grant:
        assert not figures
        assert CollectionPermission.objects.filter(user=outsider, collection=child).exists()
        from apps.documents.views.pages import get_doc
        assert get_doc(outsider_request, existing.id).full_text == 'already shared'
        item.refresh_from_db()
        assert item.status == 'partial'
        assert item.outputs.get(output_kind='figure').error_code == 'source_scope_unavailable'
    else:
        figure, = figures
        assert figure.collection_id == child.pk
        owner_request = request_for(item.batch.user)
        chunk = TextChunk.objects.get(doc_id=figure.id)
        assert json.loads(chunk_detail(owner_request, chunk.pk).content)['document']['full_text'] == 'figure caption'
        assert document_image(owner_request, figure.id).content == b'figure-pixels'


@pytest.mark.parametrize('writer', ['create', 'repoint'])
def test_child_lock_serializes_concurrent_grant_writes(upload, monkeypatch, writer):
    from concurrent.futures import ThreadPoolExecutor
    from django.db import DatabaseError, close_old_connections, connection, transaction
    figure_parser(monkeypatch)
    item = upload((('one.txt', b'private parent'),))
    child = Collection.objects.create(parent=item.batch.collection, name=figures_subcollection_name('one'))
    outsider = User.objects.create_user(username='concurrent-child-reader')
    elsewhere = Collection.objects.create(name='Elsewhere')
    grant = CollectionPermission.objects.create(user=outsider, collection=elsewhere, permission='VIEW')
    outcome = []
    def write_grant():
        close_old_connections()
        try:
            with transaction.atomic(), connection.cursor() as cursor:
                cursor.execute("SET LOCAL lock_timeout = '200ms'")
                if writer == 'create':
                    CollectionPermission.objects.create(user_id=outsider.pk, collection_id=child.pk, permission='VIEW')
                else:
                    CollectionPermission.objects.filter(pk=grant.pk).update(collection_id=child.pk)
            return 'committed'
        except DatabaseError:
            return 'blocked'
        finally:
            close_old_connections()
    original = DocumentFigure.save
    def competing_write(figure, *args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as pool:
            outcome.append(pool.submit(write_grant).result(timeout=5))
        return original(figure, *args, **kwargs)
    monkeypatch.setattr(DocumentFigure, 'save', competing_write)
    run_ingest_uploaded_file(item.pk)
    assert outcome == ['blocked']
    figure = DocumentFigure.objects.get()
    assert figure.collection_id == child.pk
    assert not child.user_can_view(outsider)
    assert not CollectionPermission.objects.filter(collection=child).exists()


def legacy_figure(item):
    parent = RawTextDocument.objects.create(title='Historical', full_text='historical parent',
        collection=item.batch.collection, ingested_by=item.batch.user)
    child = Collection.objects.create(parent=item.batch.collection, name=figures_subcollection_name(parent.title))
    figure = DocumentFigure(title='Historical figure', full_text='historical figure', collection=child,
        ingested_by=item.batch.user, source_format='pdf', figure_index=0)
    figure.parent_document = parent
    figure.save()
    item.status, item.document_ids = 'success', [str(parent.id), str(figure.id)]
    item.parser_metadata = {'outputs': [{'document_id': identity, 'modality': 'image'} for identity in item.document_ids]}
    item.save(update_fields=['status', 'document_ids', 'parser_metadata'])
    return parent, child, figure


@pytest.mark.parametrize('mutation', ['none', 'move_child', 'move_parent', 'revoke_parent', 'delete_parent', 'unrelated_child'])
def test_legacy_figure_status_requires_exact_current_parent_and_child(upload, mutation):
    item = upload()
    parent, child, figure = legacy_figure(item)
    elsewhere = Collection.objects.create(name='Elsewhere')
    CollectionPermission.objects.create(user=item.batch.user, collection=elsewhere, permission='EDIT')
    if mutation == 'move_child':
        child.move_to(elsewhere)
    elif mutation == 'move_parent':
        # Normal move_to already refuses parents with figures. Simulate an old
        # externally moved row to exercise status revalidation independently.
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute('UPDATE aquillm_rawtextdocument SET collection_id=%s WHERE pkid=%s', [elsewhere.pk, parent.pkid])
    elif mutation == 'revoke_parent':
        CollectionPermission.objects.filter(collection=item.batch.collection).delete()
        CollectionPermission.objects.create(user=item.batch.user, collection=child, permission='VIEW')
    elif mutation == 'delete_parent':
        parent.delete()
    elif mutation == 'unrelated_child':
        child.name = 'Unrelated'
        child.save(update_fields=['name'])
    response = ingest_uploads_status(request_for(item.batch.user), item.batch_id)
    if mutation == 'revoke_parent':
        assert response.status_code == 403
        return
    result = json.loads(response.content)['items'][0]
    if mutation == 'none':
        assert result['document_ids'] == [str(parent.id), str(figure.id)]
        assert [r['document_id'] for r in result['parser_metadata']['outputs']] == result['document_ids']
        assert result['status'] == 'success'
    else:
        assert str(figure.id) not in result['document_ids']
    item.refresh_from_db()
    assert not item.manifest_complete and not item.outputs.exists()
