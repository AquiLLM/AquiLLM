"""PostgreSQL candidate admission uses one source and embedding generation."""

import pytest
from django.test import override_settings

from apps.documents.models import TextChunk
from apps.documents.services.chunk_search_candidates import (
    collect_hybrid_candidate_snapshot,
)
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.tests.test_embedding_generation import legacy as legacy
from apps.documents.tests.test_embedding_generation import stage
from apps.documents.tests.test_embedding_generation_query import runtime_for
from lib.embeddings.tests.test_document_provider import transport as transport

pytestmark = pytest.mark.django_db(transaction=True)


def collect(legacy, observation=None, query="source"):
    return collect_hybrid_candidate_snapshot(
        TextChunk,
        query,
        2,
        [legacy[0]],
        query_embedding=observation,
    )


def test_raw_same_shaped_vector_cannot_enter_dense_but_lexical_survives(legacy):
    TextChunk.objects.update(embedding=[1.0] * 1024)
    snapshot = collect(legacy, [1.0] * 1024)
    assert snapshot.vector_chunk_ids == ()
    assert snapshot.trigram_chunk_ids == (TextChunk.objects.get().pk,)


def test_retired_representation_never_enters_lexical(legacy):
    from uuid import uuid4

    TextChunk.objects.update(representation_generation=uuid4())
    assert collect(legacy).baseline_candidates == ()
    assert collect(legacy, query='"source"').exact_chunk_ids == ()


def test_ambiguous_private_owner_rejects_lexical_without_disclosure(legacy):
    from apps.collections.models import Collection
    from apps.documents.models import ImageUploadDocument

    doc = legacy[0]
    collision = ImageUploadDocument(
        id=doc.id,
        full_text=doc.full_text,
        title="private secret owner",
        collection=Collection.objects.create(name="private"),
        ingested_by=doc.ingested_by,
    )
    collision.save(dont_rechunk=True)
    with pytest.raises(SourceScopeUnavailable, match="^source_scope_unavailable$"):
        collect(legacy)


def test_compatible_dense_order_survives_body_hydration(legacy, transport, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    from apps.documents.services import embedding_index_store as store
    from apps.documents.services.chunk_search import text_chunk_search
    from apps.documents.services.source_loading import source_runtime_scope
    from lib.embeddings import document_provider as provider

    doc, ref = legacy
    doc.full_text = "source second"
    doc.save(dont_rechunk=True)
    legacy = (doc, replace(ref, source_hash=doc.full_text_hash))
    TextChunk.objects.bulk_create(
        [
            TextChunk(
                doc_id=legacy[0].id,
                content="second",
                start_position=7,
                end_position=13,
                chunk_number=1,
            )
        ]
    )
    _, lease, _, _ = stage(legacy, transport)
    assert store.activate_embedding_generation(lease, using="default") == "activated"
    expected = tuple(
        TextChunk.objects.order_by("-chunk_number").values_list("pk", flat=True)
    )
    monkeypatch.setattr(
        provider,
        "_client",
        lambda: SimpleNamespace(
            embeddings=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    model="synthetic",
                    data=[SimpleNamespace(index=0, embedding=[2.0] * 1024)],
                )
            )
        ),
    )
    runtime, _ = runtime_for(legacy)
    with (
        override_settings(DOCUMENT_EMBEDDING_SPACE_KEY=transport[0].space_key),
        source_runtime_scope(runtime),
    ):
        result = text_chunk_search(
            TextChunk,
            "unrelated",
            2,
            [legacy[0]],
            authorization_context=runtime.authorization,
        )
    assert tuple(row.pk for row in result[0]) == expected


def test_pointer_swap_after_rank_never_joins_new_generation(
    legacy, transport, monkeypatch
):
    from apps.documents.services import embedding_index_query as rows
    from apps.documents.services import embedding_index_store as store
    from apps.documents.services.chunk_search import text_chunk_search
    from apps.documents.services.source_loading import source_runtime_scope

    _, first, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(first, using="default")
    _, second, _, _ = stage(legacy, transport)
    runtime, _ = runtime_for(legacy)
    rank = rows.rank_compatible_chunks

    def swapping(**kwargs):
        result = rank(**kwargs)
        assert result
        store.activate_embedding_generation(second, using="default")
        return result

    monkeypatch.setattr(rows, "rank_compatible_chunks", swapping)
    with (
        override_settings(DOCUMENT_EMBEDDING_SPACE_KEY=transport[0].space_key),
        source_runtime_scope(runtime),
    ):
        result = text_chunk_search(
            TextChunk,
            "unrelated",
            2,
            [legacy[0]],
            authorization_context=runtime.authorization,
        )
        assert result[0] == ()
        assert runtime._frozen_retrieval_scope.dense.generation_keys == (
            first.generation_key,
        )


@pytest.mark.parametrize("mutation", ["source", "chunk", "revoke", "profile"])
def test_changed_authority_at_final_handoff_never_returns_stale_dense(
    legacy, transport, monkeypatch, mutation
):
    from django.conf import settings

    from apps.documents.models import RawTextDocument
    from apps.documents.services import chunk_search
    from apps.documents.services import embedding_index_store as store
    from apps.documents.services.source_loading import source_runtime_scope

    _, first, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(first, using="default")
    runtime, policy = runtime_for(legacy)
    original = chunk_search._fallback_rerank

    def mutate(model, rows, k):
        if mutation == "source":
            RawTextDocument.objects.filter(pk=legacy[0].pk).update(full_text="changed")
        elif mutation == "chunk":
            TextChunk.objects.update(content="changed")
        elif mutation == "profile":
            settings.DOCUMENT_EMBEDDING_SPACE_KEY = "f" * 64
        else:
            policy.allowed = False
        return original(model, rows, k)

    monkeypatch.setattr(chunk_search, "_fallback_rerank", mutate)
    with (
        override_settings(DOCUMENT_EMBEDDING_SPACE_KEY=transport[0].space_key),
        source_runtime_scope(runtime),
    ):
        if mutation in ("source", "revoke"):
            with pytest.raises(SourceScopeUnavailable):
                chunk_search.text_chunk_search(
                    TextChunk,
                    "unrelated",
                    2,
                    [legacy[0]],
                    authorization_context=runtime.authorization,
                )
        else:
            result = chunk_search.text_chunk_search(
                TextChunk,
                "unrelated",
                2,
                [legacy[0]],
                authorization_context=runtime.authorization,
            )
            assert result[2] == []


def test_new_representation_graph_is_explicitly_unavailable():
    from types import SimpleNamespace
    from uuid import uuid4

    from apps.documents.services.chunk_search_lifetime import graph_preflight

    scope = SimpleNamespace(
        representation=SimpleNamespace(
            selections=(SimpleNamespace(generation_key=uuid4()),)
        )
    )
    assert graph_preflight(
        True,
        True,
        (),
        None,
        scope,
        lambda *_: pytest.fail("V1 graph must not acquire new scope"),
    ) == (None, None, "miss", ())
