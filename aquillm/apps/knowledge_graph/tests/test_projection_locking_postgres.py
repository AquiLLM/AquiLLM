from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Event

import pytest
from django.db import close_old_connections, connection, transaction
from django.utils import timezone

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="set KG_REQUIRE_POSTGRES_TESTS=1 for forced PostgreSQL race tests",
    ),
]


def _active_artifact(collection):
    from apps.knowledge_graph.models import GraphArtifact

    return GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=str(collection.pk),
        collection_scope=collection,
        status=GraphArtifact.Status.ACTIVE,
        source_hash="1" * 64,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="filter-v1",
        embedding_model_signature=(
            f"test-local:model@rev:endpoint={'e' * 64}:dims=1024:"
            "prep=kg-entity-v1:max_chars=8192:batch=64"
        ),
    )


def _building_projection(collection, artifact, now):
    from apps.knowledge_graph.models import (
        CollectionGraphMembershipState,
        CollectionGraphProjection,
    )
    from apps.knowledge_graph.projection.lifecycle import _projection_versions

    membership = CollectionGraphMembershipState.objects.create(
        collection=collection,
        active_artifact=artifact,
        registry_epoch=1,
        membership_checksum="a" * 64,
        resolver_version=artifact.resolver_version,
        resolution_config_checksum=artifact.resolution_config_checksum,
    )
    schema, projection, key = _projection_versions()
    row = CollectionGraphProjection.objects.create(
        collection=collection,
        collection_pk_snapshot=collection.pk,
        artifact=artifact,
        artifact_pk_snapshot=artifact.pk,
        state=CollectionGraphProjection.State.BUILDING,
        schema_version=schema,
        projection_version=projection,
        identifier_key_version=key,
        membership_epoch=membership.registry_epoch,
        membership_checksum=membership.membership_checksum,
        private_mapping_checksum="b" * 64,
        attempt_count=1,
        lease_owner="race-worker",
        lease_expires_at=now + timedelta(minutes=1),
    )
    return membership, row


def test_projection_ready_and_membership_mutation_share_collection_first_lock_order():
    if connection.vendor != "postgresql":
        pytest.fail("KG_REQUIRE_POSTGRES_TESTS requires PostgreSQL")
    from apps.knowledge_graph.projection.lifecycle import _READY_LOCK_ORDER

    assert _READY_LOCK_ORDER == (
        "collection",
        "active_artifact",
        "membership_state",
        "projection",
    )


def test_concurrent_membership_mutation_fences_ready_without_deadlock():
    if connection.vendor != "postgresql":
        pytest.fail("KG_REQUIRE_POSTGRES_TESTS requires PostgreSQL")
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import (
        CollectionGraphMembershipState,
        CollectionGraphProjection,
        GraphArtifact,
    )
    from apps.knowledge_graph.projection.lifecycle import (
        publish_projection_ready_compare_and_set,
    )
    from apps.knowledge_graph.projection.memgraph_repository import (
        ProjectionValidationV1,
    )
    from apps.knowledge_graph.projection.records import ProjectionCountsV1

    collection = Collection.objects.create(name=f"projection race {uuid.uuid4()}")
    artifact = _active_artifact(collection)
    now = timezone.now()
    membership, row = _building_projection(collection, artifact, now)
    mutation_locked, ready_started = Event(), Event()

    def mutate_membership():
        close_old_connections()
        try:
            with transaction.atomic():
                Collection.objects.select_for_update().get(pk=collection.pk)
                GraphArtifact.objects.select_for_update().get(pk=artifact.pk)
                locked = CollectionGraphMembershipState.objects.select_for_update().get(
                    collection_id=collection.pk
                )
                mutation_locked.set()
                assert ready_started.wait(timeout=10)
                locked.registry_epoch += 1
                locked.membership_checksum = "c" * 64
                locked.save(
                    update_fields=[
                        "registry_epoch",
                        "membership_checksum",
                        "updated_at",
                    ]
                )
        finally:
            close_old_connections()

    def publish_ready():
        close_old_connections()
        try:
            assert mutation_locked.wait(timeout=10)
            ready_started.set()
            validation = ProjectionValidationV1(
                "d" * 64,
                "e" * 64,
                ProjectionCountsV1(0, 0, 0, 0, 0, 0, 0, 0, 0),
                True,
            )
            return publish_projection_ready_compare_and_set(
                projection_id=row.id,
                owner="race-worker",
                validation=validation,
                expected_generation_key=validation.generation_key,
                expected_graph_checksum=validation.validation_checksum,
                expected_private_mapping_checksum="b" * 64,
                now=now,
                using="default",
            )
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=2) as executor:
        mutation = executor.submit(mutate_membership)
        ready = executor.submit(publish_ready)
        mutation.result(timeout=20)
        outcome = ready.result(timeout=20)

    row.refresh_from_db()
    membership.refresh_from_db()
    assert membership.registry_epoch == 2
    assert outcome.published is False
    assert row.state == CollectionGraphProjection.State.SUPERSEDED
    assert row.ready_at is None
