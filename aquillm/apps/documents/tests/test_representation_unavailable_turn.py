"""No original selection cannot break ordinary chat or authorize automatic tools."""
from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync, sync_to_async
from django.contrib.auth.models import User

from apps.chat.services.rag_turn import preservation_turn
from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import RawTextDocument
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_manual_scope import create_manual_document_runtime
from apps.documents.services.representation_unavailable_parent import UnavailableAcquisitionParent
from apps.documents.services.source_loading import SourceRuntime, current_source_runtime
from lib.llm.turn_context import current_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage
from lib.llm.providers.openai import OpenAIInterface
from .principal_scope_support import principals as principals

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("source_mode", [False, True])
@pytest.mark.parametrize("selection", ["none", "empty", "unauthorized"])
def test_ordinary_sdk_turn_survives_unavailable_original_scope(monkeypatch, source_mode, selection):
    viewer = User.objects.create_user(username="empty-viewer")
    collection = Collection.objects.create(name="Unavailable picker")
    allowed = Collection.objects.create(name="Explicitly accessible")
    CollectionPermission.objects.create(user=viewer, collection=allowed, permission="VIEW")
    doc = RawTextDocument(title="Explicit", full_text="Permitted manual source",
                          collection=allowed, ingested_by=viewer)
    doc.save(dont_rechunk=True)
    if selection == "unauthorized":
        private = RawTextDocument(title="Private", full_text="Not authorized",
                                  collection=collection, ingested_by=viewer)
        private.save(dont_rechunk=True)
    else:
        CollectionPermission.objects.create(user=viewer, collection=collection, permission="VIEW")
    consumer = SimpleNamespace(user=viewer, col_ref=SimpleNamespace(collections=[] if selection == "none" else [collection.pk]))
    for key in ("RAG_RERANK_TEXT_MODE", "RAG_DOCUMENT_CAPACITY_MODE"):
        monkeypatch.setenv(key, "legacy")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source" if source_mode else "legacy")
    monkeypatch.setenv("RAG_ITERATIVE_RETRIEVAL_ENABLED", "0")
    monkeypatch.setenv("RAG_FOLLOWUP_EVIDENCE_ENABLED", "0")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "65536")
    calls, budgets = [], []
    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="Ordinary answer", tool_calls=None), finish_reason="stop")], usage=None)
    provider = OpenAIInterface(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), "gpt-4o")
    def automatic_and_identity(parent):
        from apps.chat.services.tool_wiring.source_documents import document_metadata
        from apps.documents.services.chunk_search_scope import freeze_search_scope
        with pytest.raises(SourceScopeUnavailable):
            document_metadata(doc.id)
        with pytest.raises(SourceScopeUnavailable):
            freeze_search_scope((doc,), using="default")
        replacement = User.objects.get(pk=viewer.pk)
        with pytest.raises(SourceScopeUnavailable):
            create_manual_document_runtime(replacement, doc.id, parent=parent)
        original = consumer.col_ref
        consumer.col_ref = SimpleNamespace(collections=original.collections)
        try:
            with pytest.raises(SourceScopeUnavailable):
                create_manual_document_runtime(viewer, doc.id, parent=parent)
        finally:
            consumer.col_ref = original
    async def run():
        async with preservation_turn(consumer, max_func_calls=1) as runtime:
            turn = current_turn()
            budgets.append(turn.budget)
            parent = turn._media_source_runtime
            assert runtime is None and current_source_runtime() is None
            assert type(parent) is UnavailableAcquisitionParent and not isinstance(parent, SourceRuntime)
            assert parent.authorization is None and not hasattr(parent, "_frozen_retrieval_scope")
            await sync_to_async(automatic_and_identity)(parent)
            result, stop = await provider.complete(Conversation(system="s", messages=[UserMessage(content="Hello")]), max_tokens=64)
            assert result[-1].content == "Ordinary answer" and result[-1].stop_reason == "stop" and stop == "changed"
            assert current_turn() is turn and turn.budget is parent.budget
        assert current_turn() is None and current_source_runtime() is None
    async_to_sync(run)()
    assert len(calls) == 1 and not budgets[0].can_publish()


def _adjacent_case(selection):
    from apps.documents.models import TextChunk
    from apps.documents.services.representation_media_acquisition import initialize_media_runtime
    from lib.llm.turn_context import TurnContext
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    viewer = User.objects.create_user(username="adjacent-unavailable")
    allowed = Collection.objects.create(name="Outside unavailable picker")
    CollectionPermission.objects.create(user=viewer, collection=allowed, permission="VIEW")
    document = RawTextDocument(title="Existing adjacent target", full_text="Private adjacent source",
        collection=allowed, ingested_by=viewer)
    document.save(dont_rechunk=True)
    chunk = TextChunk.objects.bulk_create([TextChunk(doc_id=document.id, content=document.full_text,
        chunk_number=0, start_position=0, end_position=len(document.full_text))])[0]
    picker = Collection.objects.create(name="Unavailable picker")
    if selection == "unauthorized":
        private = RawTextDocument(title="Unselected", full_text="Denied source",
            collection=picker, ingested_by=viewer)
        private.save(dont_rechunk=True)
    else:
        CollectionPermission.objects.create(user=viewer, collection=picker, permission="VIEW")
    consumer = SimpleNamespace(user=viewer, col_ref=SimpleNamespace(
        collections=[] if selection == "none" else [picker.pk]))
    turn = TurnContext(TurnBudget(TurnLimits()))
    parent = initialize_media_runtime(consumer, turn)
    assert type(parent) is UnavailableAcquisitionParent
    return viewer, turn, chunk


@pytest.mark.parametrize("selection", ["none", "empty", "unauthorized"])
@pytest.mark.parametrize("source_mode", [False, True])
@pytest.mark.parametrize("reader", ["metadata", "rows"])
def test_unavailable_parent_adjacent_helpers_reject_before_database(monkeypatch, source_mode, reader, selection):
    from django.db import connection
    from django.test.utils import CaptureQueriesContext
    from apps.documents.models import TextChunk
    from apps.chat.services.tool_wiring.current_document_rows import current_chunk_metadata, current_document_rows
    from lib.llm.turn_context import bind_turn

    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source" if source_mode else "legacy")
    _, turn, chunk = _adjacent_case(selection)
    with bind_turn(turn), CaptureQueriesContext(connection) as queries:
        with pytest.raises(SourceScopeUnavailable, match="^source_scope_unavailable$"):
            if reader == "metadata":
                current_chunk_metadata(chunk.pk)
            else:
                current_document_rows(TextChunk.objects.filter(pk=chunk.pk))
    assert len(queries) == 0


@pytest.mark.parametrize("selection", ["none", "empty", "unauthorized"])
@pytest.mark.parametrize("source_mode", [False, True])
@pytest.mark.parametrize("debug", [False, True])
def test_real_adjacent_tool_unavailable_parent_is_typed_and_fact_free(monkeypatch, source_mode, debug, selection):
    from apps.chat.services.tool_wiring.adjacent_documents import more_context_tool
    from lib.llm.decorators import tool as decorator
    from lib.llm.turn_context import bind_turn

    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source" if source_mode else "legacy")
    monkeypatch.setattr(decorator, "DEBUG", debug)
    viewer, turn, chunk = _adjacent_case(selection)
    with bind_turn(turn):
        if debug:
            with pytest.raises(SourceScopeUnavailable, match="^source_scope_unavailable$"):
                more_context_tool(viewer)(chunk_id=chunk.pk, adjacent_chunks=1)
        else:
            assert more_context_tool(viewer)(chunk_id=chunk.pk, adjacent_chunks=1) == {
                "exception": "source_scope_unavailable"}


def test_standalone_public_text_runtime_remains_supported_without_media_identity(principals, monkeypatch):
    from apps.chat.services.tool_wiring.current_document_rows import current_chunk_metadata, current_document_rows
    from apps.documents.models import TextChunk
    from apps.documents.services.representation_tool_scope import tool_scope
    from apps.documents.services.source_loading import source_runtime_scope
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits
    case = principals
    runtime = SourceRuntime(TurnBudget(TurnLimits()), case.parent.authorization)
    assert current_turn() is None and not hasattr(runtime, "_media_principal_binding")
    with source_runtime_scope(runtime):
        tool_scope(runtime, using="default")
        metadata = current_chunk_metadata(case.chunks[0].pk)
        assert metadata.pk == case.chunks[0].pk and "content" in metadata.get_deferred_fields()
        rows = current_document_rows(TextChunk.objects.filter(pk=metadata.pk))
        assert tuple(row.pk for row in rows) == (metadata.pk,)
        assert rows[0].content == case.chunks[0].content


def test_genuine_no_turn_adjacent_legacy_and_source_missing_runtime_controls(principals, monkeypatch):
    from apps.chat.services.tool_wiring.adjacent_documents import _central_chunk
    from apps.documents.services.source_loading import SourcePreparationLimited
    assert current_turn() is None and current_source_runtime() is None
    chunk = principals.chunks[0]
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "legacy")
    assert _central_chunk(chunk.pk).pk == chunk.pk
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    with pytest.raises(SourcePreparationLimited):
        _central_chunk(chunk.pk)
