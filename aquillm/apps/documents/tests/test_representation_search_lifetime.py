"""Actual search lifetime preserves private default authority and original charging."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from apps.documents.models import TextChunk, DocumentRepresentationGeneration
from apps.documents.services.chunk_search_lifetime import enter_search
from apps.documents.services.chunk_search_scoped_rows import scoped_rows, dense_rows
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_media_acquisition import initialize_media_runtime
from apps.documents.services.source_loading import current_source_runtime
from lib.llm.turn_context import TurnContext, bind_turn
from lib.embeddings.tests.test_document_provider import transport as transport
from .representation_support import retained as retained, upload as upload, activate, reference
from .test_representation_tool_scope import pair as pair

pytestmark = pytest.mark.django_db(transaction=True)


def private_turn(a, budget):
    turn = TurnContext(budget)
    runtime = initialize_media_runtime(SimpleNamespace(user=a.ingested_by,
        col_ref=SimpleNamespace(collections=[a.collection_id])), turn)
    return turn, runtime


@pytest.mark.parametrize("reverse", [False, True])
def test_actual_search_entry_reuses_original_scope_and_ledger(pair, reverse):
    a, b, _, old, second, seed = pair
    turn, runtime = private_turn(a, seed.budget)
    snapshot = runtime._frozen_retrieval_scope.representation
    expected = {a.id: old.pk, b.id: second.pk}
    with bind_turn(turn):
        for document in ((b, a) if reverse else (a, b)):
            lifetime, scope = enter_search(TextChunk, (document,), runtime.authorization, "source", False)
            with lifetime:
                assert scope.runtime is runtime and scope.representation is snapshot
                assert current_source_runtime() is None
                rows = scoped_rows(scope.queryset(TextChunk.objects.all()), scope, lexical=True)
                assert [row.pk for row in rows] == [expected[document.id]]
                assert scope.requested_document_ids == {document.id}
        assert runtime.budget.sources_used == 2
        assert runtime.budget.text_used["materialized"] >= len(old.content) + len(second.content)
        assert not hasattr(turn, "_retrieval_source_runtime")


def test_actual_search_entry_cannot_refresh_changed_original_pointer(pair):
    a, _, output, old, _, seed = pair
    turn, runtime = private_turn(a, seed.budget)
    assert activate(a, output, text=old.content) == "activated"
    with bind_turn(turn), pytest.raises(SourceScopeUnavailable):
        lifetime, _ = enter_search(TextChunk, (a,), runtime.authorization, "table", False)
        lifetime.close()


def test_initialized_private_search_preserves_missing_authorization_rejection(pair):
    a, _, _, _, _, seed = pair
    turn, _ = private_turn(a, seed.budget)
    with bind_turn(turn), pytest.raises(SourceScopeUnavailable):
        lifetime, _ = enter_search(TextChunk, (a,), None, "table", False)
        lifetime.close()


def test_original_generation_freeze_never_hydrates_canonical_or_manifest(pair, monkeypatch):
    a, _, output, _, _, seed = pair
    assert activate(a, output) == "activated"
    a.refresh_from_db()
    original = DocumentRepresentationGeneration.from_db
    def check(db, field_names, values):
        assert not {"canonical_text", "manifest"}.intersection(field_names)
        return original(db, field_names, values)
    monkeypatch.setattr(DocumentRepresentationGeneration, "from_db", check)
    private_turn(a, seed.budget)


@pytest.mark.parametrize("change", [None, "pointer", "foreign_snapshot"])
def test_actual_default_dense_search_uses_verified_generation_without_public_runtime(
    pair, transport, settings, monkeypatch, change,
):
    from dataclasses import replace
    from apps.documents.services import embedding_index_store as store
    from apps.documents.services.representation_store import freeze_active_representations
    from apps.documents.tasks.embedding_index import rebuild_document_embeddings, build_embedding_generation
    from lib.embeddings.document_provider import observe_query_embedding
    from lib.embeddings import config
    a, _, output, _, _, seed = pair
    assert activate(a, output) == "activated"
    a.refresh_from_db()
    snapshot = freeze_active_representations((reference(a),), using="default")
    space = transport[0]
    store.persist_embedding_space(space, using="default")
    with patch.object(build_embedding_generation, "delay"):
        key = rebuild_document_embeddings(snapshot.selections[0], space_key=space.space_key, using="default")
    assert build_embedding_generation.run(key, using="default", timeout_ms=120000) == "activated"
    settings.DOCUMENT_EMBEDDING_SPACE_KEY = space.space_key
    monkeypatch.setattr(config, "get_local_embed_config", lambda: ("http://fake-models:8000/v1",))
    turn, runtime = private_turn(a, seed.budget)
    captured = runtime._frozen_retrieval_scope.representation
    observation = observe_query_embedding("table", space=space, budget=runtime.budget)
    with bind_turn(turn):
        lifetime, scope = enter_search(TextChunk, (a,), runtime.authorization, "table", False)
        with lifetime:
            assert current_source_runtime() is None
            if change == "pointer":
                assert activate(a, output, parser="parser-v3") == "activated"
            elif change == "foreign_snapshot":
                scope.representation = replace(captured)
            if change:
                with pytest.raises(SourceScopeUnavailable):
                    dense_rows(TextChunk, observation, scope, 10)
            else:
                rows = dense_rows(TextChunk, observation, scope, 10)
                assert {row.pk for row in rows} == set(a.chunks.values_list("pk", flat=True))
                assert rows and runtime.budget.sources_used == len(rows)
            assert current_source_runtime() is None
