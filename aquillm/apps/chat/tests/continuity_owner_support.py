"""Real canonical owners and lifecycle transitions for continuity integration."""

from types import SimpleNamespace
from uuid import UUID

import pytest
import pytest_asyncio
from channels.db import database_sync_to_async
from django.contrib.auth.models import User
from django.db import connections

from apps.chat.refs import CollectionsRef
from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import RawTextDocument, TextChunk
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage


@pytest.fixture
def continuity_owners(transactional_db, monkeypatch):
    for key, value in {
        "RAG_DIRECT_ENABLED": "1", "RAG_DIRECT_MAX_QUERIES": "1",
        "RAG_EVIDENCE_TEXT_MODE": "source", "RAG_FOLLOWUP_EVIDENCE_ENABLED": "1",
        "RAG_DOCUMENT_CAPACITY_MODE": "budgeted", "OPENAI_CONTEXT_LIMIT": "16000",
    }.items():
        monkeypatch.setenv(key, value)
    user = User.objects.create_user(username="continuity-viewer")
    collection = Collection.objects.create(name="Continuity selected")
    permission = CollectionPermission.objects.create(
        user=user, collection=collection, permission="VIEW",
    )
    documents, chunks = [], []
    for identifier, name, quantity in (
        ("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "Z", 93),
        ("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", "A", 12),
    ):
        body = f"Current {name} {quantity} units. Tail condition: only below 1 Pa."
        document = RawTextDocument(
            id=UUID(identifier), title=f"Current {name} Paper", full_text=body,
            collection=collection, ingested_by=user,
        )
        document.save(dont_rechunk=True)
        chunk = TextChunk.objects.bulk_create([TextChunk(
            doc_id=document.id, chunk_number=0, content=body,
            start_position=0, end_position=len(body),
        )])[0]
        documents.append(document)
        chunks.append(chunk)
    return SimpleNamespace(user=user, collection=collection, permission=permission,
                           documents=tuple(documents), chunks=tuple(chunks))


@pytest_asyncio.fixture(autouse=True)
async def close_continuity_worker(transactional_db):
    yield
    await database_sync_to_async(connections.close_all)()


def historical_rows(owners, kind="paper"):
    return [dict(doc_id=str(doc.id), chunk_id=chunk.pk, chunk=0,
                 title=f"{name} {kind.title()}", text=f"STALE {name} {old} units",
                 citation=f"[doc:{doc.id} chunk:{chunk.pk}]")
            for doc, chunk, name, old in zip(
                owners.documents, owners.chunks, ("Z", "A"), (91, 11), strict=True)]


def conversation(rows, question, kind="paper"):
    return Conversation(system="sys", messages=[
        UserMessage(content=f"Read two {kind}s"),
        ToolMessage(tool_name="vector_search", for_whom="assistant", content="{}",
                    result_dict={"result": list(reversed(rows)),
                                 "retrieved_documents": [r["title"] for r in reversed(rows)]}),
        AssistantMessage(content=f"Z first {rows[0]['citation']}; A second {rows[1]['citation']}.",
                         stop_reason="end_turn"),
        UserMessage(content=question),
    ])


def consumer_for(owners, convo):
    return SimpleNamespace(user=owners.user,
        col_ref=CollectionsRef([owners.collection.pk]), convo=convo)


def change_owner(owners, change):
    document = owners.documents[0]
    if change == "edit":
        document.full_text = "Corrected Z 97 units. Tail condition: only below 1 Pa."
        document.save(dont_rechunk=True)
    elif change == "move":
        document.collection = Collection.objects.create(name="Outside selected scope")
        document.save(dont_rechunk=True, update_fields=["collection"])
    elif change == "delete":
        document.delete()
    elif change == "revoke":
        owners.permission.delete()
    else:
        raise AssertionError("unknown owner transition")


def deleted_citation(owners):
    body = "Deleted appendix source."
    document = RawTextDocument(title="Archived Appendix", full_text=body,
        collection=owners.collection, ingested_by=owners.user)
    document.save(dont_rechunk=True)
    chunk = TextChunk.objects.bulk_create([TextChunk(doc_id=document.id,
        chunk_number=0, content=body, start_position=0, end_position=len(body))])[0]
    row = dict(doc_id=str(document.id), chunk_id=chunk.pk, chunk=0,
               title=document.title, text=body,
               citation=f"[doc:{document.id} chunk:{chunk.pk}]")
    document.delete()
    assert not TextChunk.objects.filter(pk=chunk.pk).exists()
    return row


def rebuild_current(document):
    from apps.documents.services.representation_history import reference
    from apps.documents.services.representation_snapshot_build import (
        build_canonical_snapshot_representation,
    )
    assert build_canonical_snapshot_representation(reference(document), using="default") == "activated"
    document.refresh_from_db()
    return document.chunks.get()
