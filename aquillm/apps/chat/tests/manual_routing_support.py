"""Real original authority for routing tests whose search/provider boundaries are mocked."""
import pytest
import pytest_asyncio
from channels.db import database_sync_to_async
from django.contrib.auth.models import User
from django.db import connections

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import RawTextDocument, TextChunk


@pytest.fixture(autouse=True)
def real_manual_authority(transactional_db, monkeypatch, request):
    user = User.objects.create_user(username="manual-route-owner")
    collection = Collection.objects.create(pk=1, name="Manual route")
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
    body = "Calibration evidence."
    document = RawTextDocument(id=request.module.DOC, title="Paper A", full_text=body,
                               collection=collection, ingested_by=user)
    document.save(dont_rechunk=True)
    TextChunk.objects.bulk_create([TextChunk(pk=1, doc_id=document.id, content=body,
        chunk_number=0, start_position=0, end_position=len(body))])
    original = request.module._consumer
    def consumer(*args, **kwargs):
        value = original(*args, **kwargs)
        value.user = user
        return value
    monkeypatch.setattr(request.module, "_consumer", consumer)


@pytest_asyncio.fixture(autouse=True)
async def close_routing_worker(transactional_db):
    yield
    await database_sync_to_async(connections.close_all)()


@pytest.fixture
def current_figure(transactional_db, figure):
    from apps.documents.services.representation_activation import build_document_representation
    from apps.documents.services.representation_caption import caption_chunker_key
    from apps.documents.tests.representation_support import reference
    document, output, _ = figure
    assert build_document_representation(reference(document), extraction_revision_id=output.extraction_revision_id,
        chunker_key=caption_chunker_key(), using="default") == "activated"
    document.refresh_from_db()
    viewer = User.objects.create_user(username="manual-image-viewer")
    CollectionPermission.objects.create(user=viewer, collection=document.collection, permission="VIEW")
    return viewer, document, document.chunks.get()
