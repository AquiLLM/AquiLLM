"""Merged query-only routing preserves unavailable parents and human children."""
from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync
from channels.db import database_sync_to_async
from django.contrib.auth.models import User

from apps.chat.services.manual_search_turn import run_manual_search_turn
from apps.chat.services.rag_config import rag_preservation_config
from apps.chat.services.rag_turn import preservation_turn
from apps.chat.services.retrieval_query_runtime import query_proposals_enabled
from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_manual_scope import private_manual_acquisition
from apps.documents.services.representation_media_acquisition import media_runtime
from apps.documents.services.representation_unavailable_parent import UnavailableAcquisitionParent
from apps.documents.services.source_loading import current_source_runtime
from lib.llm.turn_context import current_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage
from .principal_scope_support import sdk_for
from .test_representation_principal_publication import sinks
from .test_representation_unavailable_turn import (
    test_ordinary_sdk_turn_survives_unavailable_original_scope as _ordinary_unavailable,
)

pytestmark = pytest.mark.django_db(transaction=True)


def _flags(monkeypatch, query_only):
    for name in ("RAG_EVIDENCE_TEXT_MODE", "RAG_RERANK_TEXT_MODE",
                 "RAG_DOCUMENT_CAPACITY_MODE", "RAG_EVIDENCE_SELECTION_MODE"):
        monkeypatch.setenv(name, "legacy")
    for name in ("RAG_FOLLOWUP_EVIDENCE_ENABLED", "RAG_ITERATIVE_RETRIEVAL_ENABLED",
                 "RAG_VOCABULARY_QUERY_ENABLED"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("RAG_QUERY_REWRITE_ENABLED", str(int(query_only)))
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "65536")
    assert query_proposals_enabled() is query_only
    assert not rag_preservation_config().active


@pytest.mark.parametrize("selection", ["none", "empty", "unauthorized"])
def test_query_only_ordinary_sdk_reuses_real_unavailable_controls(monkeypatch, selection):
    # Default and source-mode cases already exist in the imported module.
    # This runs their actual lifetime/SDK/automatic-denial control with only R4 on.
    _flags(monkeypatch, True)
    _ordinary_unavailable(monkeypatch, source_mode=False, selection=selection)
    assert query_proposals_enabled() and not rag_preservation_config().active


def _owner(selection):
    viewer = User.objects.create_user(username="query-manual-viewer")
    picker = Collection.objects.create(name="Unavailable original picker")
    allowed = Collection.objects.create(name="Explicit human owner")
    CollectionPermission.objects.create(user=viewer, collection=allowed, permission="VIEW")
    document = RawTextDocument(title="Human source", full_text="Requested manual source is precise.",
        collection=allowed, ingested_by=viewer)
    document.save(dont_rechunk=True)
    chunk = TextChunk.objects.bulk_create([TextChunk(doc_id=document.id,
        content=document.full_text, chunk_number=0, start_position=0,
        end_position=len(document.full_text))])[0]
    if selection == "unauthorized":
        private = RawTextDocument(title="Denied picker", full_text="Other private source.",
            collection=picker, ingested_by=viewer)
        private.save(dont_rechunk=True)
        assert not picker.user_can_view(viewer)
    else:
        CollectionPermission.objects.create(user=viewer, collection=picker, permission="VIEW")
    assert allowed.user_can_view(viewer)
    consumer = SimpleNamespace(user=viewer, col_ref=SimpleNamespace(
        collections=[] if selection == "none" else [picker.pk]))
    return SimpleNamespace(consumer=consumer, documents=[document], chunks=[chunk])


@pytest.mark.parametrize("query_only", [False, True])
@pytest.mark.parametrize("selection", ["none", "empty", "unauthorized"])
def test_real_human_search_from_unavailable_parent_keeps_private_original_lifetime(
    monkeypatch, query_only, selection,
):
    from apps.chat.services.tool_wiring.documents import search_single_document_tool
    from apps.chat.services.tool_wiring.source_documents import document_metadata
    from apps.documents.services import chunk_search
    from apps.documents.services.chunk_search_scope import freeze_search_scope
    from lib.llm.decorators import tool as decorator

    _flags(monkeypatch, query_only)
    monkeypatch.setattr(chunk_search, "load_query_embedding", lambda *args: None)
    monkeypatch.setattr(decorator, "DEBUG", False)
    case = _owner(selection)
    document, chunk = case.documents[0], case.chunks[0]
    provider, create, citation = sdk_for(document, chunk)
    response = create.return_value
    observed = []
    conversation = Conversation(system="s", messages=[UserMessage(
        content=f"/search {document.id} requested manual source")])
    case.consumer.convo = conversation

    def automatic_denied():
        with pytest.raises(SourceScopeUnavailable):
            document_metadata(document.id)
        with pytest.raises(SourceScopeUnavailable):
            freeze_search_scope((document,), using="default")
        result = search_single_document_tool(case.consumer.user, case.consumer.col_ref)(
            doc_id=str(document.id), search_string="requested manual source")
        assert result.get("exception") and not result.get("result")
        assert chunk.content not in str(result) and citation not in str(result)

    async def sdk(**kwargs):
        turn = current_turn()
        context = turn.answer_publication
        record = context._manual_record
        child = media_runtime()
        assert turn is case.turn and child is record.runtime
        assert record.parent is case.parent and child.budget is case.parent.budget is turn.budget
        assert record.principal_binding is case.parent._media_principal_binding
        assert child.authorization.selected_document_ids == frozenset((document.id,))
        assert child.authorization.reauthorization_capability._principal is case.consumer.user
        assert context.packet.source_authorization is child.authorization
        assert context.packet.source_mode is False
        assert context.lease is turn.budget._synthesis and context.lease.budget is turn.budget
        context.lease.check_active()
        assert current_source_runtime() is None and not rag_preservation_config().active
        assert query_proposals_enabled() is query_only and turn.budget.planner_calls == 0
        assert chunk.content in str(kwargs["messages"])
        observed.append(child)
        return response
    create.side_effect = sdk

    async def run():
        async with preservation_turn(case.consumer, max_func_calls=3) as runtime:
            case.turn = current_turn()
            case.parent = case.turn._media_source_runtime
            assert runtime is None and current_source_runtime() is None
            assert type(case.parent) is UnavailableAcquisitionParent
            assert case.parent.authorization is None
            assert not hasattr(case.parent, "_frozen_retrieval_scope")
            assert case.parent._media_principal_binding.consumer is case.consumer
            await database_sync_to_async(automatic_denied)()
            create.assert_not_called()
            assert await run_manual_search_turn(case.consumer, provider, conversation) == "handled"
            create.assert_called_once()
            assert len(observed) == 1 and citation in case.consumer.convo[-1].content
            assert media_runtime() is case.parent and current_source_runtime() is None
            assert not hasattr(case.parent, "_frozen_retrieval_scope")
            await database_sync_to_async(sinks)(case, rejected=False, limited=False)
        assert current_turn() is None and current_source_runtime() is None
        assert not case.parent.budget.can_publish()
        with pytest.raises(SourceScopeUnavailable):
            with private_manual_acquisition(observed[0]):
                pass
    async_to_sync(run)()
