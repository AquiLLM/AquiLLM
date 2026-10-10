from dataclasses import replace

import pytest
from django.contrib.auth.models import User

from apps.collections.models import Collection
from apps.documents.models import RawTextDocument, ImageUploadDocument, TextChunk
from apps.documents.services.representation_contracts import (
    DocumentRefV1,
    SourceScopeUnavailable,
)
from apps.documents.services import representation_store as representations
from apps.documents.services import embedding_index_store as store
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.embeddings.document_provider import embed_document_batch
from lib.embeddings.document_contracts import build_budget

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def legacy():
    user = User.objects.create(username="dense-owner")
    collection = Collection.objects.create(name="Synthetic dense")
    doc = RawTextDocument(
        title="Synthetic", full_text="source", collection=collection, ingested_by=user
    )
    doc.save(dont_rechunk=True)
    TextChunk.objects.bulk_create(
        [
            TextChunk(
                doc_id=doc.id,
                content="source",
                start_position=0,
                end_position=6,
                chunk_number=0,
            )
        ]
    )
    ref = DocumentRefV1(
        doc._meta.label_lower, doc.pkid, doc.id, collection.pk, doc.full_text_hash
    )
    return doc, ref


def test_legacy_unknown_vectors_remain_unverified(legacy):
    doc, ref = legacy
    frozen = representations.freeze_active_representations((ref,), using="default")
    assert frozen.selections[0].generation_key is None
    assert frozen.selections[0].canonical_sha256 == doc.full_text_hash
    assert list(
        representations.active_chunk_scope(TextChunk.objects.all(), snapshot=frozen)
    )
    assert (
        store.capture_embedding_manifest(frozen.selections[0], using="default")[0][
            "source_sha256"
        ]
        == doc.full_text_hash
    )


def test_foreign_owner_collision_fails_closed_even_when_hash_matches(legacy):
    doc, ref = legacy
    frozen = representations.freeze_active_representations((ref,), using="default")
    other = Collection.objects.create(name="Unauthorized")
    collision = ImageUploadDocument(
        id=doc.id,
        full_text=doc.full_text,
        title="Private",
        collection=other,
        ingested_by=doc.ingested_by,
    )
    collision.save(dont_rechunk=True)
    for operation in (
        lambda: representations.freeze_active_representations((ref,), using="default"),
        lambda: list(
            representations.active_chunk_scope(TextChunk.objects.all(), snapshot=frozen)
        ),
        lambda: representations.validate_representation_snapshot(
            frozen, using="default"
        ),
    ):
        with pytest.raises(SourceScopeUnavailable, match="^source_scope_unavailable$"):
            operation()


def test_nonnull_legacy_representation_is_never_a_wildcard(legacy):
    from uuid import uuid4

    _, ref = legacy
    frozen = representations.freeze_active_representations((ref,), using="default")
    TextChunk.objects.update(representation_generation=uuid4())
    assert not representations.active_chunk_scope(
        TextChunk.objects.all(), snapshot=frozen
    ).exists()
    with pytest.raises(SourceScopeUnavailable):
        representations.validate_representation_snapshot(
            replace(
                frozen,
                selections=(replace(frozen.selections[0], generation_key=uuid4()),),
            ),
            using="default",
        )


def stage(legacy, transport):
    _, ref = legacy
    space, _ = transport
    store.persist_embedding_space(space, using="default")
    frozen = representations.freeze_active_representations((ref,), using="default")
    selection = frozen.selections[0]
    manifest = store.capture_embedding_manifest(selection, using="default", space=space)
    lease = store.begin_embedding_generation(
        selection,
        space_key=space.space_key,
        manifest_sha256=store.embedding_manifest_digest(
            selection, manifest, space_key=space.space_key
        ),
        expected_count=len(manifest),
        using="default",
    )
    rows = list(TextChunk.objects.filter(doc_id=ref.id).order_by("chunk_number"))
    results = embed_document_batch(
        tuple(store.source_chunk(row) for row in rows),
        space=space,
        budget=build_budget(),
    )
    store.stage_embedding_batch(
        lease,
        chunk_ids=tuple(row.pk for row in rows),
        results=results,
        input_sha256=tuple(item["prepared_input_sha256"] for item in manifest),
        using="default",
    )
    return frozen, lease, manifest, results


def test_vector_only_publication_is_idempotent_and_keeps_citations(legacy, transport):
    before = list(
        TextChunk.objects.values(
            "pk",
            "content",
            "start_position",
            "end_position",
            "chunk_number",
            "embedding_provenance",
        )
    )
    frozen, lease, manifest, results = stage(legacy, transport)
    store.stage_embedding_batch(
        lease,
        chunk_ids=(manifest[0]["chunk_id"],),
        results=results,
        input_sha256=(manifest[0]["prepared_input_sha256"],),
        using="default",
    )
    assert store.activate_embedding_generation(lease, using="default") == "activated"
    assert (
        store.activate_embedding_generation(lease, using="default") == "already_active"
    )
    assert (
        list(
            TextChunk.objects.values(
                "pk",
                "content",
                "start_position",
                "end_position",
                "chunk_number",
                "embedding_provenance",
            )
        )
        == before
    )
    legacy[0].refresh_from_db()
    assert legacy[0].ingestion_complete


def test_stale_chunk_source_cannot_activate(legacy, transport):
    _, lease, _, _ = stage(legacy, transport)
    TextChunk.objects.update(content="changed")
    assert store.activate_embedding_generation(lease, using="default") == "stale"


def test_new_generation_can_swap_same_space_pointer(legacy, transport):
    from apps.documents.services.embedding_index_query import (
        freeze_dense_generations,
        validate_dense_snapshot,
    )

    frozen, a, _, _ = stage(legacy, transport)
    assert store.activate_embedding_generation(a, using="default") == "activated"
    captured = freeze_dense_generations(
        frozen, space_key=transport[0].space_key, using="default"
    )
    _, b, _, _ = stage(legacy, transport)
    assert a.generation_key != b.generation_key
    assert store.activate_embedding_generation(b, using="default") == "activated"
    assert validate_dense_snapshot(frozen, captured, using="default") == ()
    fresh = freeze_dense_generations(
        frozen, space_key=transport[0].space_key, using="default"
    )
    assert fresh.manifest_sha256 != captured.manifest_sha256
    assert store.activate_embedding_generation(a, using="default") == "stale"


def test_three_durable_deliveries_and_failure_lease_fence(legacy, transport):
    from django.utils import timezone
    from apps.documents.models.embedding_index import EmbeddingGeneration

    _, initial, _, _ = stage(legacy, transport)
    first = store.claim_embedding_delivery(initial.generation_key, using="default")
    assert first and first.lease_token != initial.lease_token
    assert (
        store.claim_embedding_delivery(initial.generation_key, using="default") is None
    )
    assert not store.fail_embedding_delivery(initial, kind="contract", using="default")
    assert store.fail_embedding_delivery(first, kind="upstream", using="default")
    for index in (2, 3):
        EmbeddingGeneration.objects.filter(pk=initial.generation_key).update(
            next_attempt_at=timezone.now()
        )
        lease = store.claim_embedding_delivery(initial.generation_key, using="default")
        assert store.fail_embedding_delivery(
            lease, kind="upstream", using="default"
        ) == (index < 3)
    assert (
        store.claim_embedding_delivery(initial.generation_key, using="default") is None
    )
    row = EmbeddingGeneration.objects.get(pk=initial.generation_key)
    assert (row.deliveries, row.state, row.failure_kind) == (3, "failed", "upstream")


def test_concurrent_activation_serializes_exact_owner(legacy, transport):
    from concurrent.futures import ThreadPoolExecutor
    from django.db import close_old_connections

    _, lease, _, _ = stage(legacy, transport)

    def activate():
        close_old_connections()
        try:
            return store.activate_embedding_generation(lease, using="default")
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: activate(), range(2))) == [
            "activated",
            "already_active",
        ]


def test_document_delete_removes_its_exact_dense_state(legacy, transport):
    from apps.documents.models.embedding_index import (
        DocumentEmbeddingState,
        EmbeddingGeneration,
        ChunkEmbedding,
    )

    stage(legacy, transport)
    legacy[0].delete()
    assert not DocumentEmbeddingState.objects.exists()
    assert not EmbeddingGeneration.objects.exists()
    assert not ChunkEmbedding.objects.exists()


def test_activation_revalidates_persisted_vector_receipt_pair(legacy, transport):
    from apps.documents.models.embedding_index import ChunkEmbedding

    _, lease, _, _ = stage(legacy, transport)
    ChunkEmbedding.objects.update(vector=[2.0] * 1024)
    assert store.activate_embedding_generation(lease, using="default") == "stale"
