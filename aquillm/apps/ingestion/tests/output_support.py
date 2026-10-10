"""Synthetic real-PostgreSQL ingestion fixtures; only broker I/O is replaced."""
import io
import zipfile
import pytest
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from apps.collections.models import Collection, CollectionPermission
from apps.ingestion.models import IngestionBatch, IngestionBatchItem


@pytest.fixture
def upload(settings, tmp_path, monkeypatch):
    settings.STORAGES = {'default': {'BACKEND': 'django.core.files.storage.FileSystemStorage',
                                   'OPTIONS': {'location': str(tmp_path)}}}
    monkeypatch.setattr('apps.documents.tasks.chunking.create_chunks.delay', lambda *a, **k: None)
    owner = User.objects.create_user(username='output-owner')
    collection = Collection.objects.create(name='Outputs')
    CollectionPermission.objects.create(user=owner, collection=collection, permission='EDIT')
    batch = IngestionBatch.objects.create(user=owner, collection=collection)

    def make(members=(('first.txt', b'first body'), ('second.txt', b'second body'))):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            for name, body in members:
                archive.writestr(name, body)
        return IngestionBatchItem.objects.create(batch=batch,
            source_file=SimpleUploadedFile('sources.zip', stream.getvalue()),
            original_filename='sources.zip', content_type='application/zip')
    return make
