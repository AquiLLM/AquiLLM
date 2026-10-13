"""Real original acquisition for synthesis/provider lifetime regressions."""
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

from channels.db import database_sync_to_async
from django.db import connections


def _owner(text):
    from django.contrib.auth import get_user_model
    from apps.collections.models import Collection, CollectionPermission
    from apps.documents.models import RawTextDocument, TextChunk

    user = get_user_model().objects.create_user(username="source-synthesis-owner")
    collection = Collection.objects.create(name="Synthesis source")
    permission = CollectionPermission.objects.create(
        user=user, collection=collection, permission="VIEW")
    document = RawTextDocument(title="Consent policy", full_text=text,
                               collection=collection, ingested_by=user)
    document.save(dont_rechunk=True)
    chunk = TextChunk.objects.bulk_create([TextChunk(doc_id=document.id,
        content=text, chunk_number=0, start_position=0, end_position=len(text))])[0]
    return user, collection, permission, chunk


def _acquire(runtime, chunk):
    from apps.documents.models import TextChunk
    from apps.documents.services.source_loading import source_query_rows

    scope = runtime._frozen_retrieval_scope
    rows = source_query_rows(TextChunk.objects.filter(pk=chunk.pk),
                             runtime=runtime, require_existing=True)
    assert len(rows) == 1 and rows[0].content == chunk.content
    assert runtime._frozen_retrieval_scope is scope
    assert runtime.budget.text_used["materialized"] >= len(chunk.content)
    return rows[0]


@asynccontextmanager
async def authorized_source_scope(packet, monkeypatch, convo):
    from apps.chat.refs import CollectionsRef
    from apps.chat.services.rag_turn import preservation_turn
    from apps.documents.services.source_loading import current_source_runtime
    from lib.llm.turn_context import current_turn
    from lib.retrieval.evidence import fingerprint_source

    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    user, collection, permission, chunk = await database_sync_to_async(_owner)(
        packet.chunks[0]["text"])
    consumer = SimpleNamespace(user=user, col_ref=CollectionsRef([collection.pk]),
                               convo=convo)
    try:
        async with preservation_turn(consumer, max_func_calls=3) as runtime:
            turn = current_turn()
            assert current_source_runtime() is runtime
            assert turn._media_source_runtime is runtime
            assert runtime.budget is turn.budget
            scope = runtime._frozen_retrieval_scope
            acquired = await database_sync_to_async(_acquire)(runtime, chunk)
            row = {**packet.chunks[0], "doc_id": str(acquired.doc_id),
                   "chunk_id": acquired.pk, "chunk": acquired.chunk_number,
                   "text": acquired.content,
                   "citation": f"[doc:{acquired.doc_id} chunk:{acquired.pk}]"}
            packet.chunks[:] = [row]
            packet.source_authorization = runtime.authorization
            candidate = replace(packet.selection.candidates[0],
                chunk_id=acquired.pk, doc_id=str(acquired.doc_id),
                chunk_number=acquired.chunk_number, text=acquired.content,
                source_fingerprint=fingerprint_source(acquired.content), row=row)
            packet.selection = replace(packet.selection, candidates=(candidate,))
            yield SimpleNamespace(runtime=runtime, permission=permission)
            assert runtime._frozen_retrieval_scope is scope
    finally:
        await database_sync_to_async(connections.close_all)()
