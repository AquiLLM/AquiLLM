"""Persisted original authority for selection orchestration boundary tests."""
import pytest
import pytest_asyncio
from channels.db import database_sync_to_async
from django.contrib.auth.models import User
from django.db import connections

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import RawTextDocument, TextChunk


@pytest.fixture
def selection_owner(transactional_db):
    user = User.objects.create_user(username="selection-route-owner")
    collection = Collection.objects.create(pk=1, name="Selection route")
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
    passages = [f"Passage {i}" for i in range(1, 4)]
    body = "\n".join(passages)
    document = RawTextDocument(id="00000000-0000-0000-0000-000000000001",
        title="Paper", full_text=body, collection=collection, ingested_by=user)
    document.save(dont_rechunk=True)
    offset = 0
    for index, passage in enumerate(passages, 1):
        TextChunk.objects.bulk_create([TextChunk(pk=index, doc_id=document.id,
            chunk_number=index, content=passage, start_position=offset,
            end_position=offset + len(passage))])
        offset += len(passage) + 1
    return user


@pytest_asyncio.fixture
async def selection_worker(selection_owner):
    yield selection_owner
    await database_sync_to_async(connections.close_all)()


@pytest.fixture
def two_selection_queries(monkeypatch):
    """Controlled multi-result composition, not automatic semicolon splitting."""
    from apps.chat.services import rag_pipeline

    def queries(_conversation, question, *, max_queries):
        assert question == "compare papers; and summarize trials"
        assert max_queries == 2
        # Both retain every requested aspect. The production builder intentionally
        # keeps just the original question; its policy is covered separately.
        return [question, "compare papers and summarize trials"]

    monkeypatch.setattr(rag_pipeline, "build_retrieval_queries", queries)
