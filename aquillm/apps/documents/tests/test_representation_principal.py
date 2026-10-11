"""Parsed human search cannot replace the original turn's principal identity."""
from contextlib import nullcontext

import pytest
from asgiref.sync import async_to_sync

from apps.chat.services.manual_search_turn import run_manual_search_turn
from apps.documents.services.representation_media_acquisition import media_runtime
from apps.documents.services.source_loading import current_source_runtime, source_runtime_scope
from lib.llm.turn_context import bind_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage
from .principal_scope_support import principals as principals, change_principal, sdk_for

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("source_mode", [False, True])
@pytest.mark.parametrize("change", [None, "different_user", "same_pk_object", "original_pk"])
def test_actual_manual_sdk_requires_original_populated_parent_principal(principals, monkeypatch, source_mode, change):
    case = principals
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source" if source_mode else "legacy")
    target = 2 if change == "different_user" else 1
    document, chunk = case.documents[target], case.chunks[target]
    assert not case.documents[2].collection.user_can_view(case.alice)
    assert not case.documents[0].collection.user_can_view(case.bob)
    provider, create, citation = sdk_for(document, chunk)
    conversation = Conversation(system="s", messages=[UserMessage(
        content=f"/search {document.id} requested source")])
    case.consumer.convo = conversation
    original_pk = case.alice.pk
    try:
        with bind_turn(case.turn), source_runtime_scope(case.parent) if source_mode else nullcontext():
            change_principal(case, change)
            assert async_to_sync(run_manual_search_turn)(case.consumer, provider, conversation) == "handled"
            if change:
                # Explicit SDK count detects calls even when synthesis swallows an error.
                create.assert_not_called()
                assert chunk.content not in str(case.consumer.convo.model_dump())
                assert citation not in str(case.consumer.convo.model_dump())
                assert not getattr(case.turn, "_manual_representation_children", {})
            else:
                create.assert_called_once()
                assert chunk.content in str(create.call_args.kwargs["messages"])
                assert citation in case.consumer.convo[-1].content
            assert case.parent._frozen_retrieval_scope.representation is case.snapshot
            assert media_runtime() is case.parent
            assert (current_source_runtime() is not None) is source_mode
    finally:
        case.alice.pk = original_pk
        case.consumer.user = case.alice


def test_reinitialization_cannot_replace_consumer_with_same_principal(principals):
    from types import SimpleNamespace
    from django.db import connection
    from django.test.utils import CaptureQueriesContext
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    from apps.documents.services.representation_media_acquisition import initialize_media_runtime
    case = principals
    replacement = SimpleNamespace(user=case.consumer.user, col_ref=case.consumer.col_ref)
    with bind_turn(case.turn), CaptureQueriesContext(connection) as queries:
        with pytest.raises(SourceScopeUnavailable):
            initialize_media_runtime(replacement, case.turn)
    assert len(queries) == 0
    assert case.turn._media_source_runtime is case.parent
    assert case.parent._frozen_retrieval_scope.representation is case.snapshot
