"""Document metadata, whole-document and adjacency never bypass source admission."""

import re
from dataclasses import replace
from types import SimpleNamespace

import pytest
from django.contrib.auth.models import User
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.chat.refs import CollectionsRef
from apps.collections.models import Collection, CollectionPermission
from apps.collections.services.django_retrieval_authorization import (
    build_selected_scope_authorization_context,
)
from apps.documents.models import RawTextDocument, TextChunk
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.fixture
def docs(db):
    user = User.objects.create_user(username="source-tools")
    collection = Collection.objects.create(name="sources")
    CollectionPermission.objects.bulk_create(
        [CollectionPermission(user=user, collection=collection, permission="VIEW")]
    )
    doc = RawTextDocument.objects.bulk_create(
        [
            RawTextDocument(
                title="Doc",
                full_text="never hydrate this document body " * 10000,
                full_text_hash="a" * 64,
                collection=collection,
                ingested_by=user,
            )
        ]
    )[0]
    chunks = TextChunk.objects.bulk_create(
        [
            TextChunk(
                doc_id=doc.id,
                chunk_number=i,
                start_position=i * 100,
                end_position=(i + 1) * 100,
                content=f"Aspect {i}. Qualified tail: only below 1 Pa.",
            )
            for i in range(3)
        ]
    )
    auth = build_selected_scope_authorization_context(
        principal=user,
        selected_collection_ids=(collection.pk,),
        selected_documents=(doc,),
    )
    return user, doc, chunks, auth


def test_source_document_metadata_defers_full_text(docs, monkeypatch):
    from apps.chat.services.tool_wiring.source_documents import (
        document_metadata,
        selected_document_metadata,
    )
    from apps.documents.services.source_loading import (
        SourceRuntime,
        source_runtime_scope,
    )

    user, doc, _, auth = docs
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    with (
        source_runtime_scope(SourceRuntime(TurnBudget(TurnLimits()), auth)),
        CaptureQueriesContext(connection) as queries,
    ):
        selected = selected_document_metadata(user, CollectionsRef([doc.collection_id]))
        single = document_metadata(doc.id)
    assert len(selected) == 1
    assert "full_text" in selected[0].get_deferred_fields()
    assert "full_text" in single.get_deferred_fields()
    assert not any(
        re.search(r'(?:SELECT |, )"[^"]+"\."full_text"', q["sql"]) for q in queries
    )


@pytest.mark.parametrize("materialized", [10, 1000])
def test_source_whole_document_is_bounded_and_never_claims_prefix_complete(
    docs, monkeypatch, materialized
):
    from apps.chat.services.tool_wiring.documents import whole_document_tool
    from apps.documents.services.source_loading import (
        SourceRuntime,
        source_runtime_scope,
    )

    user, doc, chunks, auth = docs
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "16000")
    conversation = Conversation(
        system="sys", messages=[UserMessage(content="Read the document")]
    )
    chat = SimpleNamespace(
        chat=SimpleNamespace(llm_if=SimpleNamespace(), convo=conversation)
    )
    runtime = SourceRuntime(
        TurnBudget(replace(TurnLimits(), materialized_codepoints=materialized)), auth
    )
    with source_runtime_scope(runtime), CaptureQueriesContext(connection) as queries:
        result = whole_document_tool(user, chat, CollectionsRef([doc.collection_id]))(
            doc_id=str(doc.id)
        )
    assert not any(
        re.search(r'(?:SELECT |, )"[^"]+"\."full_text"', q["sql"]) for q in queries
    )
    if materialized == 10:
        assert result["retrieval_status"] == "context_limited"
    else:
        assert all(c.content in result["result"] for c in chunks)
        assert len(result["citation_chunks"]) == 3


def test_source_adjacent_chunks_share_materialization_allowance(docs, monkeypatch):
    from apps.chat.services.tool_wiring.documents import more_context_tool
    from apps.documents.services.source_loading import (
        SourceRuntime,
        source_runtime_scope,
    )

    user, doc, chunks, auth = docs
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    runtime = SourceRuntime(
        TurnBudget(replace(TurnLimits(), materialized_codepoints=10)), auth
    )
    with source_runtime_scope(runtime), CaptureQueriesContext(connection) as queries:
        result = more_context_tool(user)(chunk_id=chunks[1].pk, adjacent_chunks=1)
    assert result["retrieval_status"] == "context_limited"
    assert runtime.budget.text_used["materialized"] == 0
    assert not any(
        re.search(r'(?:SELECT |, )"[^"]+"\."full_text"', q["sql"]) for q in queries
    )


def test_explicit_document_scope_shares_budget_and_restores_selected_scope(
    docs, monkeypatch
):
    from apps.chat.services.tool_wiring.source_documents import (
        explicit_document_source_scope,
    )
    from apps.documents.services.source_loading import (
        SourceRuntime,
        current_source_runtime,
        source_runtime_scope,
    )

    user, doc, chunks, auth = docs
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    parent = SourceRuntime(TurnBudget(TurnLimits()), auth)
    with source_runtime_scope(parent):
        with explicit_document_source_scope(user, doc.id) as explicit:
            assert explicit.budget is parent.budget
            assert explicit.cache is parent.cache
            assert explicit.authorization.selected_document_ids == frozenset((doc.id,))
            assert current_source_runtime() is explicit
        assert current_source_runtime() is parent


async def test_source_manual_turn_is_limited_until_shared_handoff_is_bound(monkeypatch):
    from unittest.mock import Mock

    from apps.chat.services import manual_search_turn
    from apps.chat.tests.test_manual_search_turn import _consumer

    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    retrieve = Mock(side_effect=AssertionError("must not enter legacy acquisition"))
    monkeypatch.setattr(manual_search_turn, "_retrieve", retrieve)
    consumer = _consumer("/collection question")
    assert (
        await manual_search_turn.run_manual_search_turn(
            consumer, object(), consumer.convo
        )
        == "handled"
    )
    assert "limits" in consumer.convo[-1].content
    retrieve.assert_not_called()
