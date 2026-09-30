"""Projection membership compatibility across collection and canonical stages."""

import os
from datetime import timedelta

import pytest
from django.db import connection
from django.utils import timezone

from apps.collections.models import Collection
from apps.knowledge_graph.models import (
    CanonicalEntity,
    CanonicalEntityLink,
    CollectionEntity,
    CollectionGraphMembershipState,
    CollectionGraphProjection,
    OntologyVersion,
)
from apps.knowledge_graph.projection.django_projection_rows import (
    DjangoProjectionOrmLoader,
)
from apps.knowledge_graph.projection.django_projection_source import (
    DjangoProjectionRowSource,
)
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
)
from apps.knowledge_graph.projection.memberships import (
    load_automatic_membership_assignments,
    membership_decision_checksum,
)
from apps.knowledge_graph.projection.records import CollectionGraphProjectionBundleV1
from apps.knowledge_graph.services.ontology import load_ontology_yaml
from apps.knowledge_graph.tests.test_models import _artifact

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="set KG_REQUIRE_POSTGRES_TESTS=1 for PostgreSQL integration tests",
    ),
]

COLLECTION_RESOLVER = "collection-resolution-v1"
CANONICAL_RESOLVER = "canonical-resolution-v1"
CODEC = HmacSha256ProjectionIdentifierCodec(
    b"synthetic-projection-key", key_version="key-v1"
)


@pytest.fixture
def source_rows():
    assert connection.vendor == "postgresql"
    collection = Collection.objects.create(name="synthetic projection resolver")
    ontology = load_ontology_yaml(
        """version: 1.0.0
entity_types:
  - name: concept
    description: A synthetic concept.
    aliases: []
    default_retrieval_weight: 1.0
    default_suppression_policy: never
    default_suppression_threshold: 0.0
relations:
  - name: related_to
    description: A synthetic relation.
    direction: directed
    allowed_head_types: [concept]
    allowed_tail_types: [concept]
"""
    )
    OntologyVersion.objects.create(
        kind="graph",
        version=ontology.version,
        checksum=ontology.checksum,
        status="active",
        metadata={"yaml": ontology.raw_yaml},
    )
    artifact = _artifact(
        scope_type="collection",
        scope_id=collection.pk,
        resolver_version=COLLECTION_RESOLVER,
        ontology_version=ontology.version,
        ontology_checksum=ontology.checksum,
    )
    artifact.save()
    entity = CollectionEntity.objects.create(
        artifact=artifact,
        collection=collection,
        cluster_key="a" * 64,
        label="Synthetic Alpha",
        normalized_label="synthetic alpha",
        entity_type="concept",
        version_signature="version-v1",
        extraction_confidence=0.9,
        resolution_confidence=0.9,
        retrieval_utility=0.5,
    )
    canonical = CanonicalEntity.objects.create(
        identity_key="b" * 64,
        label="Synthetic Alpha",
        normalized_label="synthetic alpha",
        entity_type="concept",
        version_signature="version-v1",
        resolver_version=CANONICAL_RESOLVER,
    )
    link = CanonicalEntityLink.objects.create(
        canonical_entity=canonical,
        collection_entity=entity,
        score=0.9,
        method="exact_name_or_alias",
        reason="synthetic exact match",
        outcome="automatic",
        resolver_version=CANONICAL_RESOLVER,
    )
    artifact.status = "active"
    artifact.save(update_fields=["status"])
    assignments = load_automatic_membership_assignments(
        collection_ids=(collection.pk,),
        using="default",
        batch_size=10,
        codec=CODEC,
    )
    checksum = membership_decision_checksum(assignments)
    CollectionGraphMembershipState.objects.create(
        collection=collection,
        active_artifact=artifact,
        registry_epoch=1,
        membership_checksum=checksum,
        resolver_version=COLLECTION_RESOLVER,
        resolution_config_checksum=artifact.resolution_config_checksum,
    )
    projection = CollectionGraphProjection.objects.create(
        collection=collection,
        collection_pk_snapshot=collection.pk,
        artifact=artifact,
        artifact_pk_snapshot=artifact.pk,
        state="building",
        schema_version="collection-graph-v1",
        projection_version="projection-v1",
        identifier_key_version="key-v1",
        membership_epoch=1,
        membership_checksum=checksum,
        private_mapping_checksum="c" * 64,
        lease_owner="synthetic-test-worker",
        lease_expires_at=timezone.now() + timedelta(minutes=5),
    )
    return entity, canonical, link, projection, checksum


def _source():
    return DjangoProjectionRowSource(
        using="default",
        identifier_key=b"synthetic-projection-key",
        identifier_key_version="key-v1",
    )


def test_distinct_valid_resolvers_preserve_registry_checksum_and_projection(
    source_rows,
):
    entity, canonical, link, projection, checksum = source_rows
    assert link.resolver_version == canonical.resolver_version == CANONICAL_RESOLVER
    assert link.resolver_version != COLLECTION_RESOLVER

    snapshot = DjangoProjectionOrmLoader("default").load(
        projection_id=projection.pk, batch_size=10
    )
    assert snapshot["memberships"] == (
        {
            "entity_id": entity.pk,
            "canonical_entity_id": canonical.pk,
            "outcome": "automatic",
            "status": "active",
            "canonical_status": "active",
            "decision_checksum": link.decision_checksum,
        },
    )
    bundle = _source().load_projection_rows(projection_id=projection.pk, batch_size=10)
    assert (
        type(CollectionGraphProjectionBundleV1(**bundle))
        is CollectionGraphProjectionBundleV1
    )
    assert bundle["generation"].membership_checksum == checksum
    assert len(bundle["automatic_memberships"]) == 1
    assert bundle["automatic_memberships"][0].automatic_membership_key is not None


@pytest.mark.parametrize(
    ("table", "column", "value"),
    (
        ("link", "resolver_version", "stale-canonical-resolution-v1"),
        ("canonical", "entity_type", "different-type"),
        ("canonical", "version_signature", "different-version"),
    ),
)
def test_stale_canonical_identity_cannot_serialize_ready_projection(
    source_rows, table, column, value
):
    entity, canonical, link, projection, _checksum = source_rows
    target = link if table == "link" else canonical
    # Simulate an imported malformed row that bypassed immutable model validation.
    with connection.cursor() as cursor:
        cursor.execute(
            f"UPDATE {target._meta.db_table} SET {column}=%s WHERE id=%s",
            [value, target.pk],
        )
    loader = DjangoProjectionOrmLoader("default")
    assert loader._memberships((entity.pk,), 10) == ()
    with pytest.raises(
        ValueError, match="projection membership audit checksum is stale"
    ):
        _source().load_projection_rows(projection_id=projection.pk, batch_size=10)
