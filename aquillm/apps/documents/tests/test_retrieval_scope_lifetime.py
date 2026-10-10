"""The default tool path inherits one deadline and one authorization snapshot."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
from django.db import connections
from django.test import override_settings

from apps.chat.services.retrieval_authorization import (
    resolve_document_retrieval_authorization,
)
from apps.collections.models import CollectionPermission
from apps.documents.models import TextChunk
from apps.documents.services import embedding_index_store as store
from apps.documents.services.chunk_search import text_chunk_search
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.tests.test_embedding_generation import legacy as legacy
from apps.documents.tests.test_embedding_generation import stage
from apps.documents.tests.test_embedding_generation_query import runtime_for
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.llm.turn_context import TurnContext, bind_turn

pytestmark = pytest.mark.django_db(transaction=True)


@override_settings(KG_OVERLAY_ENABLED=False, RAG_EVIDENCE_SELECTION_MODE="fixed")
def test_default_graph_disabled_authorization_and_dense_positive(legacy, transport):
    _, lease, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(lease, using="default")
    doc = legacy[0]
    permission = CollectionPermission.objects.create(
        user=doc.ingested_by, collection=doc.collection, permission="VIEW"
    )
    runtime, _ = runtime_for(legacy)
    with (
        override_settings(DOCUMENT_EMBEDDING_SPACE_KEY=transport[0].space_key),
        bind_turn(TurnContext(runtime.budget)),
    ):
        auth = resolve_document_retrieval_authorization(
            doc.ingested_by,
            SimpleNamespace(collections=[doc.collection_id]),
            [doc],
            None,
        )
        assert auth is not None
        result = text_chunk_search(
            TextChunk, "unrelated", 2, [doc], authorization_context=auth
        )
        assert result[0][0].pk == TextChunk.objects.get().pk
        permission.delete()
        with pytest.raises(SourceScopeUnavailable):
            text_chunk_search(
                TextChunk, "unrelated", 2, [doc], authorization_context=auth
            )


def test_invalid_authorization_cannot_enter_automatic_turn(legacy):
    runtime, _ = runtime_for(legacy)
    with bind_turn(TurnContext(runtime.budget)), pytest.raises(SourceScopeUnavailable):
        text_chunk_search(
            TextChunk, "source", 2, [legacy[0]], authorization_context=None
        )


def test_changed_authorization_cannot_create_second_scope(legacy, transport):
    from dataclasses import replace

    from apps.documents.services.chunk_search_lifetime import enter_search

    runtime, _ = runtime_for(legacy)
    with bind_turn(TurnContext(runtime.budget)):
        lifetime, first = enter_search(
            TextChunk, [legacy[0]], runtime.authorization, "source", False
        )
        lifetime.close()
        other = replace(runtime.authorization, authorization_context_signature="c" * 64)
        with pytest.raises(SourceScopeUnavailable):
            enter_search(TextChunk, [legacy[0]], other, "proposal", False)


def test_concurrent_queries_share_one_frozen_snapshot_and_budget(
    legacy, transport, monkeypatch
):
    from apps.documents.services import chunk_search_scope as scopes
    from apps.documents.services.chunk_search_lifetime import enter_search

    _, lease, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(lease, using="default")
    runtime, _ = runtime_for(legacy)
    turn = TurnContext(runtime.budget)
    original = scopes.freeze_active_representations
    captures = []
    simultaneous_queries = Barrier(2)

    def freeze(*args, **kwargs):
        result = original(*args, **kwargs)
        captures.append(result)
        return result

    monkeypatch.setattr(scopes, "freeze_active_representations", freeze)

    def run(query):
        try:
            with bind_turn(turn):
                lifetime, scope = enter_search(
                    TextChunk, [legacy[0]], runtime.authorization, query, False
                )
                with lifetime:
                    assert scope.runtime.budget is runtime.budget
                    simultaneous_queries.wait(timeout=5)
                    return scope.representation, scope.dense
        finally:
            connections.close_all()

    with (
        override_settings(DOCUMENT_EMBEDDING_SPACE_KEY=transport[0].space_key),
        ThreadPoolExecutor(2) as workers,
    ):
        values = list(workers.map(run, ("source", "proposal")))
    assert len(captures) == 1
    assert values[0][0] is values[1][0]
    assert values[0][1] is values[1][1]
