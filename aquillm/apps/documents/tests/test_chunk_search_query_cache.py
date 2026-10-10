"""Actual query observation cache bindings, using synthetic PostgreSQL scope."""

from copy import deepcopy

import pytest
from django.core.cache import cache
from django.test import override_settings

from apps.documents.models import TextChunk
from apps.documents.services import embedding_index_store as store
from apps.documents.services import rag_cache
from apps.documents.services.chunk_search import text_chunk_search
from apps.documents.services.chunk_search_lifetime import enter_search
from apps.documents.services.hybrid_graph_lifecycle import query_embedding
from apps.documents.services.source_loading import source_runtime_scope
from apps.documents.tests.test_embedding_generation import legacy as legacy
from apps.documents.tests.test_embedding_generation import stage
from apps.documents.tests.test_embedding_generation_query import runtime_for
from lib.embeddings.document_binding import serialize_query_observation
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.llm.turn_context import TurnContext, bind_turn

pytestmark = pytest.mark.django_db(transaction=True)


def setup(legacy, transport):
    _, lease, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(lease, using="default")
    return runtime_for(legacy)


@override_settings(RAG_CACHE_ENABLED=True)
def test_same_query_restores_private_binding_and_reuses_one_call(legacy, transport):
    runtime, _ = setup(legacy, transport)
    space, calls = transport
    cache.clear()
    calls.clear()
    with (
        override_settings(DOCUMENT_EMBEDDING_SPACE_KEY=space.space_key),
        source_runtime_scope(runtime),
    ):
        lifetime, scope = enter_search(
            TextChunk, [legacy[0]], runtime.authorization, "source", False
        )
        with lifetime:
            first = query_embedding("source")
            second = query_embedding("source")
            assert first is not second
            assert serialize_query_observation(
                first, space=space
            ) == serialize_query_observation(second, space=space)
            assert len(calls) == 1
            assert scope.query_digest == first.prepared_input_sha256


@pytest.mark.parametrize(
    "mutation", ["raw", "bool", "extra", "space", "query", "binding"]
)
@override_settings(RAG_CACHE_ENABLED=True)
def test_malformed_or_relabelled_cache_is_a_miss(
    legacy, transport, monkeypatch, mutation
):
    runtime, _ = setup(legacy, transport)
    space, calls = transport
    with (
        override_settings(DOCUMENT_EMBEDDING_SPACE_KEY=space.space_key),
        source_runtime_scope(runtime),
    ):
        lifetime, _ = enter_search(
            TextChunk, [legacy[0]], runtime.authorization, "source", False
        )
        with lifetime:
            first = query_embedding("source")
            payload = deepcopy(serialize_query_observation(first, space=space))
            if mutation == "raw":
                payload = [1.0] * 1024
            elif mutation == "bool":
                payload["vector"][0] = True
            elif mutation == "extra":
                payload["extra"] = "untrusted"
            elif mutation == "space":
                payload["compatibility_key"] = "b" * 64
            elif mutation == "query":
                payload["prepared_input_sha256"] = "b" * 64
            else:
                payload["serving_binding"]["verification_sha256"] = "b" * 64
            monkeypatch.setattr(rag_cache, "cache_get", lambda _: payload)
            before = len(calls)
            assert query_embedding("source").compatibility_key == space.space_key
            assert len(calls) == before + 1


def test_default_turn_reuses_original_budget_and_freezes_once(
    legacy, transport, monkeypatch
):
    runtime, _ = setup(legacy, transport)
    from apps.documents.services import chunk_search_scope as scopes

    real = scopes.freeze_dense_generations
    captured = []

    def freeze(*args, **kwargs):
        result = real(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(scopes, "freeze_dense_generations", freeze)
    with (
        override_settings(DOCUMENT_EMBEDDING_SPACE_KEY=transport[0].space_key),
        bind_turn(TurnContext(runtime.budget)),
    ):
        for query in ("source", "source revised"):
            result = text_chunk_search(
                TextChunk,
                query,
                2,
                [legacy[0]],
                authorization_context=runtime.authorization,
            )
            assert result[0][0].pk == TextChunk.objects.get().pk
        assert len(captured) == 1
        from lib.llm.turn_context import current_turn

        assert current_turn()._retrieval_source_runtime.budget is runtime.budget


@override_settings(DOCUMENT_EMBEDDING_SPACE_KEY="")
def test_missing_profile_keeps_lexical_and_never_calls_raw_provider(
    legacy, monkeypatch
):
    import aquillm.utils

    monkeypatch.setattr(
        aquillm.utils, "get_embedding", lambda *_: pytest.fail("raw route")
    )
    result = text_chunk_search(TextChunk, "source", 2, [legacy[0]])
    assert result[0] == ()
    assert result[2][0].pk == TextChunk.objects.get().pk
