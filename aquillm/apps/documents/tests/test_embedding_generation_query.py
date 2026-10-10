import pytest
from dataclasses import replace
from apps.documents.services import embedding_index_query as query
from apps.documents.services.representation_contracts import RepresentationSnapshotV1
from apps.documents.tests.test_embedding_generation import legacy as legacy, stage
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.embeddings.document_provider import observe_query_embedding
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from apps.documents.services.source_loading import (
    SourceRuntime,
    source_runtime_scope,
    SourcePreparationLimited,
)
from apps.collections.services.retrieval_authorization import (
    OpaquePrincipalReference,
    bind_retrieval_reauthorization_capability,
    freeze_retrieval_authorization_context,
)

pytestmark = pytest.mark.django_db(transaction=True)


def runtime_for(legacy):
    _, ref = legacy

    class Policy:
        policy_version = "v1"
        policy_checksum = "a" * 64
        allowed = True

        def opaque_principal_reference(self, **kwargs):
            return OpaquePrincipalReference("b" * 64)

        def current_authorized_document_scope(self, **kwargs):
            return ((ref.collection_id, ref.id),) if self.allowed else ()

    policy = Policy()
    principal = object()
    auth = freeze_retrieval_authorization_context(
        principal=principal,
        database_alias="default",
        policy=policy,
        selected_collection_ids=(ref.collection_id,),
        selected_document_ids=(ref.id,),
        reauthorization_capability=bind_retrieval_reauthorization_capability(
            principal=principal, policy=policy
        ),
    )
    return SourceRuntime(TurnBudget(TurnLimits()), auth), policy


def rank(frozen, dense, observation, runtime):
    with source_runtime_scope(runtime):
        return query.rank_compatible_chunks(
            snapshot=frozen,
            dense_snapshot=dense,
            query=observation,
            limit=10,
            using="default",
            budget=runtime.budget,
        )


@pytest.mark.django_db
def test_explicit_empty_snapshot_has_stable_digest():
    snapshot = RepresentationSnapshotV1(())
    first = query.freeze_dense_generations(
        snapshot, space_key="a" * 64, using="default"
    )
    second = query.freeze_dense_generations(
        snapshot, space_key="a" * 64, using="default"
    )
    assert first == second
    assert first.generation_keys == ()
    assert len(first.manifest_sha256) == 64


def test_unknown_legacy_vector_excluded_then_verified_sidecar_eligible(
    legacy, transport
):
    from apps.documents.models import TextChunk
    from apps.documents.services import (
        representation_store,
        embedding_index_store as store,
    )

    space, _ = transport
    runtime, _ = runtime_for(legacy)
    observation = observe_query_embedding("query", space=space, budget=runtime.budget)
    TextChunk.objects.update(embedding=[1.0] * 1024)
    frozen = representation_store.freeze_active_representations(
        (legacy[1],), using="default"
    )
    empty = query.freeze_dense_generations(
        frozen, space_key=space.space_key, using="default"
    )
    assert rank(frozen, empty, observation, runtime) == ()
    _, lease, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(lease, using="default")
    dense = query.freeze_dense_generations(
        frozen, space_key=space.space_key, using="default"
    )
    rows = rank(frozen, dense, observation, runtime)
    assert [r.chunk_id for r in rows] == [TextChunk.objects.get().pk]
    assert rank(frozen, empty, observation, runtime) == ()
    assert (
        rank(frozen, dense, replace(observation, compatibility_key=None), runtime) == ()
    )
    assert (
        rank(frozen, dense, replace(observation, compatibility_key="f" * 64), runtime)
        == ()
    )
    assert observation.receipt["observed_identity"]["revision"] is None


def test_foreign_document_mapping_and_duplicate_mapping_are_rejected(legacy, transport):
    from uuid import uuid4
    from apps.documents.services import embedding_index_store as store
    from apps.documents.services.representation_contracts import SourceScopeUnavailable

    frozen, lease, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(lease, using="default")
    dense = query.freeze_dense_generations(
        frozen, space_key=transport[0].space_key, using="default"
    )
    for selections in (
        (replace(dense.selections[0], document_id=uuid4()),),
        dense.selections * 2,
    ):
        with pytest.raises(SourceScopeUnavailable):
            query.dense_generation_snapshot_digest(
                frozen, selections, space_key=dense.space_key
            )


def test_stale_body_and_revoked_authority_never_rank(legacy, transport):
    from apps.documents.models import TextChunk
    from apps.documents.services import embedding_index_store as store
    from apps.documents.services.representation_contracts import SourceScopeUnavailable

    frozen, lease, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(lease, using="default")
    dense = query.freeze_dense_generations(
        frozen, space_key=transport[0].space_key, using="default"
    )
    runtime, policy = runtime_for(legacy)
    observation = observe_query_embedding(
        "query", space=transport[0], budget=runtime.budget
    )
    TextChunk.objects.update(content="tamper")
    assert rank(frozen, dense, observation, runtime) == ()
    policy.allowed = False
    with pytest.raises(SourceScopeUnavailable, match="^source_scope_unavailable$"):
        rank(frozen, dense, observation, runtime)


def test_post_freeze_private_collision_blocks_dense_and_lazy_lexical(legacy, transport):
    from apps.collections.models import Collection
    from apps.documents.models import ImageUploadDocument, TextChunk
    from apps.documents.services import (
        representation_store,
        embedding_index_store as store,
    )
    from apps.documents.services.representation_contracts import SourceScopeUnavailable

    frozen, lease, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(lease, using="default")
    dense = query.freeze_dense_generations(
        frozen, space_key=transport[0].space_key, using="default"
    )
    lexical = representation_store.active_chunk_scope(
        TextChunk.objects.all(), snapshot=frozen
    )
    runtime, _ = runtime_for(legacy)
    observation = observe_query_embedding(
        "query", space=transport[0], budget=runtime.budget
    )
    doc = legacy[0]
    private = Collection.objects.create(name="private")
    collision = ImageUploadDocument(
        id=doc.id,
        full_text=doc.full_text,
        title="private",
        collection=private,
        ingested_by=doc.ingested_by,
    )
    collision.save(dont_rechunk=True)
    assert list(lexical) == []
    with pytest.raises(SourceScopeUnavailable, match="^source_scope_unavailable$"):
        rank(frozen, dense, observation, runtime)


def test_dense_ranking_requires_original_runtime_budget(legacy, transport):
    from apps.documents.services import embedding_index_store as store

    frozen, lease, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(lease, using="default")
    dense = query.freeze_dense_generations(
        frozen, space_key=transport[0].space_key, using="default"
    )
    runtime, _ = runtime_for(legacy)
    observation = observe_query_embedding(
        "query", space=transport[0], budget=runtime.budget
    )
    with pytest.raises(SourcePreparationLimited):
        query.rank_compatible_chunks(
            snapshot=frozen,
            dense_snapshot=dense,
            query=observation,
            limit=10,
            using="default",
            budget=runtime.budget,
        )


def test_a_to_b_pointer_swap_never_publishes_under_a_digest(legacy, transport):
    from apps.documents.services import embedding_index_store as store

    frozen, lease, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(lease, using="default")
    captured = query.freeze_dense_generations(
        frozen, space_key=transport[0].space_key, using="default"
    )
    runtime, _ = runtime_for(legacy)
    observation = observe_query_embedding(
        "query", space=transport[0], budget=runtime.budget
    )
    cached = rank(frozen, captured, observation, runtime)
    assert cached
    _, b, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(b, using="default")
    assert query.validate_dense_snapshot(frozen, captured, using="default") == ()
    assert rank(frozen, captured, observation, runtime) == ()
    fresh = query.freeze_dense_generations(
        frozen, space_key=transport[0].space_key, using="default"
    )
    assert fresh.manifest_sha256 != captured.manifest_sha256
    assert {
        r.embedding_generation_key for r in rank(frozen, fresh, observation, runtime)
    } == {b.generation_key}


def test_profile_b_with_same_model_and_dimensions_is_foreign(legacy, transport):
    from lib.embeddings.tests.test_document_contracts import profile
    from lib.embeddings.document_provider import (
        register_verified_profile,
        engine_manifest_for,
    )
    from apps.documents.services import embedding_index_store as store

    frozen, a, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(a, using="default")
    b_space = profile(compute_precision="bf16")
    register_verified_profile(
        b_space,
        endpoint="http://fake-models:8000/v1",
        deployment_artifact_sha256="c" * 64,
        engine_manifest=engine_manifest_for(b_space),
        query_template="{text}",
        document_template="{text}",
    )
    _, b, _, _ = stage(legacy, (b_space, []))
    store.activate_embedding_generation(b, using="default")
    foreign = query.freeze_dense_generations(
        frozen, space_key=b_space.space_key, using="default"
    )
    runtime, _ = runtime_for(legacy)
    observation = observe_query_embedding(
        "query", space=transport[0], budget=runtime.budget
    )
    assert rank(frozen, foreign, observation, runtime) == ()
