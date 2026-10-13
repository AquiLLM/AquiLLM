"""Real synthetic owners and coordinates created before a measured direct turn."""
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid4

from channels.db import database_sync_to_async

from apps.chat.refs import CollectionsRef


@dataclass
class PersistedDirectCase:
    user: object
    collection_map: dict
    documents: dict
    chunks: dict
    declared: dict

    def collections(self, labels):
        return [self.collection_map[label].pk for label in labels]

    def consumer(self, convo, labels):
        from unittest.mock import AsyncMock
        return SimpleNamespace(user=self.user, col_ref=CollectionsRef(self.collections(labels)),
                               convo=convo, _send_stream_payload=AsyncMock())

    def text(self, text):
        for (doc, pk), chunk in self.chunks.items():
            text = text.replace(f"[doc:{doc} chunk:{pk}]",
                                f"[doc:{chunk.doc_id} chunk:{chunk.pk}]")
        return text

    def rows(self, payload):
        result = deepcopy(payload)
        mapped = []
        for row in result.get("result", []):
            key = str(row["doc_id"]), row["chunk_id"]
            source = self.declared[key]
            assert row["text"] == source["text"] and row["title"] == source["title"]
            chunk = self.chunks[key]
            mapped.append(dict(row, doc_id=str(chunk.doc_id), chunk_id=chunk.pk,
                               chunk=chunk.chunk_number,
                               citation=f"[doc:{chunk.doc_id} chunk:{chunk.pk}]"))
        result["result"] = mapped
        return result


def _persist(rows, collection_labels):
    from django.contrib.auth import get_user_model
    from apps.collections.models import Collection, CollectionPermission
    from apps.documents.models import RawTextDocument, TextChunk
    from lib.llm.turn_context import current_turn
    from apps.documents.services.source_loading import current_source_runtime
    assert current_turn() is None and current_source_runtime() is None
    assert collection_labels and len(set(collection_labels)) == len(collection_labels)
    user = get_user_model().objects.create_user(username="direct-" + uuid4().hex)
    collections = {label: Collection.objects.create(name=f"Synthetic scope {label}")
                   for label in collection_labels}
    CollectionPermission.objects.bulk_create([
        CollectionPermission(user=user, collection=value, permission="VIEW")
        for value in collections.values()])
    declared, grouped = {}, {}
    for row in rows:
        key = str(row["doc_id"]), row["chunk_id"]
        assert type(key[1]) is int and key[1] > 0
        assert isinstance(row["text"], str) and row["text"]
        if key in declared:
            assert declared[key]["text"] == row["text"] and declared[key]["title"] == row["title"]
            continue
        declared[key] = deepcopy(row)
        grouped.setdefault(key[0], []).append(key)
    documents, chunks = {}, {}
    collection = next(iter(collections.values()))
    for doc, keys in grouped.items():
        body = "\n".join(declared[key]["text"] for key in keys)
        owner = RawTextDocument(collection=collection, ingested_by=user,
                                title=declared[keys[0]]["title"], full_text=body)
        owner.save(dont_rechunk=True)
        documents[doc] = owner
        start = 0
        for number, key in enumerate(keys):
            text = declared[key]["text"]
            chunks[key] = TextChunk.objects.create(doc_id=owner.id, chunk_number=number,
                content=text, start_position=start, end_position=start + len(text))
            start += len(text) + 1
    return PersistedDirectCase(user, collections, documents, chunks, declared)


async def persisted_direct_case(rows, *, collection_labels=(1,)):
    """No late source writes, provider substitutions, or fabricated authority."""
    return await database_sync_to_async(_persist)(tuple(rows), tuple(collection_labels))


async def persisted_conversation(case, labels=()):
    from apps.chat.models import WSConversation
    return await database_sync_to_async(WSConversation.objects.create)(
        owner=case.user, name="Synthetic direct turn", selected_collection_ids=case.collections(labels))



import pytest_asyncio


@pytest_asyncio.fixture
async def close_direct_worker(transactional_db):
    from django.db import connections
    yield
    await database_sync_to_async(connections.close_all)()
