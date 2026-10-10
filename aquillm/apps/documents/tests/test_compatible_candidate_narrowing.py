"""Requested subsets restrict dense SQL before top-k without recapturing scope."""

from dataclasses import replace
from uuid import uuid4

import pytest
from django.test import override_settings

from apps.collections.services.retrieval_authorization import (
    bind_retrieval_reauthorization_capability,
    freeze_retrieval_authorization_context,
)
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services import embedding_index_query as queries
from apps.documents.services import embedding_index_store as store
from apps.documents.services import representation_store
from apps.documents.services.chunk_search import text_chunk_search
from apps.documents.services.chunk_search_lifetime import enter_search
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.documents.tests.test_embedding_generation import legacy as legacy
from apps.documents.tests.test_embedding_generation import stage
from apps.documents.tests.test_embedding_generation_query import runtime_for
from lib.embeddings.document_provider import observe_query_embedding
from lib.embeddings.tests.test_document_provider import transport as transport

pytestmark = pytest.mark.django_db(transaction=True)


def two_documents(legacy, transport):
    original, ref = legacy
    other = RawTextDocument(
        title="Selected subset",
        full_text="second",
        collection=original.collection,
        ingested_by=original.ingested_by,
    )
    other.save(dont_rechunk=True)
    TextChunk.objects.bulk_create(
        [
            TextChunk(
                doc_id=other.id,
                content="second",
                chunk_number=0,
                start_position=0,
                end_position=6,
            )
        ]
    )
    other_ref = replace(
        ref, id=other.id, pkid=other.pkid, source_hash=other.full_text_hash
    )
    for doc in (legacy, (other, other_ref)):
        _, lease, _, _ = stage(doc, transport)
        store.activate_embedding_generation(lease, using="default")
    runtime, policy = runtime_for(legacy)
    policy.current_authorized_document_scope = lambda **_: (
        (ref.collection_id, ref.id),
        (other_ref.collection_id, other_ref.id),
    )
    principal = object()
    auth = freeze_retrieval_authorization_context(
        principal=principal,
        database_alias="default",
        policy=policy,
        selected_collection_ids=(ref.collection_id,),
        selected_document_ids=(ref.id, other_ref.id),
        reauthorization_capability=bind_retrieval_reauthorization_capability(
            principal=principal, policy=policy
        ),
    )
    return other, other_ref, SourceRuntime(runtime.budget, auth)


def test_requested_subset_filters_before_dense_limit(legacy, transport):
    other, other_ref, runtime = two_documents(legacy, transport)
    representation = representation_store.freeze_active_representations(
        (legacy[1], other_ref), using="default"
    )
    dense = queries.freeze_dense_generations(
        representation, space_key=transport[0].space_key, using="default"
    )
    observation = observe_query_embedding(
        "query", space=transport[0], budget=runtime.budget
    )
    with source_runtime_scope(runtime):
        ranked = queries.rank_compatible_chunks(
            snapshot=representation,
            dense_snapshot=dense,
            query=observation,
            limit=1,
            using="default",
            budget=runtime.budget,
            requested_document_ids=frozenset((other.id,)),
        )
    assert [r.chunk_id for r in ranked] == [TextChunk.objects.get(doc_id=other.id).pk]


@pytest.mark.parametrize(
    "subset", [[], (), {True}, frozenset((True,)), frozenset((uuid4(),))]
)
def test_malformed_or_widened_dense_subset_rejected(legacy, transport, subset):
    frozen, lease, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(lease, using="default")
    runtime, _ = runtime_for(legacy)
    dense = queries.freeze_dense_generations(
        frozen, space_key=transport[0].space_key, using="default"
    )
    observation = observe_query_embedding(
        "query", space=transport[0], budget=runtime.budget
    )
    with source_runtime_scope(runtime), pytest.raises(SourceScopeUnavailable):
        queries.rank_compatible_chunks(
            snapshot=frozen,
            dense_snapshot=dense,
            query=observation,
            limit=1,
            using="default",
            budget=runtime.budget,
            requested_document_ids=subset,
        )


def test_all_then_subset_keeps_exact_original_snapshots_and_cache_partition(
    legacy, transport
):
    other, _, runtime = two_documents(legacy, transport)
    with (
        override_settings(DOCUMENT_EMBEDDING_SPACE_KEY=transport[0].space_key),
        source_runtime_scope(runtime),
    ):
        lifetime, first = enter_search(
            TextChunk, [legacy[0], other], runtime.authorization, "query", False
        )
        with lifetime:
            first_key = first.cache_key()
        lifetime, narrowed = enter_search(
            TextChunk, [other], runtime.authorization, "query", False
        )
        with lifetime:
            assert narrowed.representation is first.representation
            assert narrowed.dense is first.dense
            assert narrowed.runtime.budget is runtime.budget
            assert narrowed.cache_key() != first_key
        result = text_chunk_search(
            TextChunk, "query", 1, [other], authorization_context=runtime.authorization
        )
        assert [r.doc_id for r in result[0]] == [other.id]


def test_initial_subset_cannot_expand_later(legacy, transport):
    other, _, runtime = two_documents(legacy, transport)
    with (
        override_settings(DOCUMENT_EMBEDDING_SPACE_KEY=transport[0].space_key),
        source_runtime_scope(runtime),
    ):
        lifetime, _ = enter_search(
            TextChunk, [other], runtime.authorization, "query", False
        )
        lifetime.close()
        with pytest.raises(SourceScopeUnavailable):
            enter_search(
                TextChunk, [legacy[0], other], runtime.authorization, "query", False
            )
