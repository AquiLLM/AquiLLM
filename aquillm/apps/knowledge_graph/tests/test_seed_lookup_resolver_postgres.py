# ruff: noqa: F401,F811
"""Real seed lookup parity across collection and canonical resolver stages."""

import os

import pytest
from django.db import connection

from apps.knowledge_graph.models import CanonicalEntity, CanonicalEntityLink
from apps.knowledge_graph.projection.django_projection_rows import (
    DjangoProjectionOrmLoader,
)
from apps.knowledge_graph.projection.identifiers import (
    ProjectionIdentifierDomain as Domain,
)
from apps.knowledge_graph.projection.memberships import (
    load_automatic_membership_assignments,
    membership_decision_checksum,
)
from apps.knowledge_graph.retrieval.extended_seed_repository import (
    ExtendedSeedRepository,
)
from apps.knowledge_graph.tests.seed_lookup_fixtures import seeds

pytestmark = [
    pytest.mark.django_db(transaction=True, databases="__all__"),
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="requires isolated PostgreSQL",
    ),
]

DISTINCT_RESOLVERS = {
    "collection_resolver": "collection-resolution-v1",
    "canonical_resolver": "canonical-resolution-v1",
}


def _lookup(fixture):
    chunk = fixture.chunks[1]
    return ExtendedSeedRepository().load_seed_identities(
        authority=fixture.authority,
        chunks=((chunk.pk, fixture.document.id),),
        authorization=fixture.authorization,
        codec=fixture.codec,
        max_rows=2,
    )[chunk.pk]


@pytest.mark.parametrize("seeds", (DISTINCT_RESOLVERS,), indirect=True)
def test_valid_distinct_resolvers_match_registry_projection_and_seed_keys(seeds):
    assignments = load_automatic_membership_assignments(
        collection_ids=(seeds.collection.pk,),
        using="default",
        batch_size=10,
        codec=seeds.codec,
    )
    assert membership_decision_checksum(assignments) == seeds.state.membership_checksum
    assert seeds.canonical.resolver_version != seeds.artifact.resolver_version
    memberships = DjangoProjectionOrmLoader("projection_source")._memberships(
        tuple(entity.pk for entity in seeds.entities), 10
    )
    assert len(memberships) == 1
    assert memberships[0]["canonical_entity_id"] == seeds.canonical.pk
    canonical_key = seeds.codec.encode(
        Domain.AUTOMATIC_CANONICAL_IDENTITY, source=seeds.canonical.pk
    ).value
    direct = seeds.direct.canonical_name_matches(
        span=seeds.spans[0], ready=seeds.scope.ready, limit=2
    )
    assert len(direct) == 1
    assert direct[0].component_key == canonical_key
    assert _lookup(seeds) == (canonical_key,)


@pytest.mark.parametrize("seeds", (DISTINCT_RESOLVERS,), indirect=True)
@pytest.mark.parametrize(
    ("table", "column", "value"),
    (
        ("link", "resolver_version", "stale-canonical-resolution-v1"),
        ("canonical", "entity_type", "different-type"),
        ("canonical", "version_signature", "different-version"),
    ),
)
def test_seed_lookup_omits_stale_canonical_identity(seeds, table, column, value):
    target = (
        seeds.canonical
        if table == "canonical"
        else CanonicalEntityLink.objects.get(collection_entity=seeds.entities[0])
    )
    # Simulate a malformed imported row bypassing immutable model validation.
    with connection.cursor() as cursor:
        cursor.execute(
            f"UPDATE {target._meta.db_table} SET {column}=%s WHERE id=%s",
            [value, target.pk],
        )
    entity_key = seeds.codec.encode(
        Domain.ENTITY,
        generation=seeds.projection.generation_key,
        source=seeds.entities[0].pk,
    ).value
    direct = seeds.direct.canonical_name_matches(
        span=seeds.spans[0], ready=seeds.scope.ready, limit=2
    )
    assert len(direct) == 1
    assert direct[0].component_key == entity_key
    assert _lookup(seeds) == (entity_key,)


@pytest.mark.parametrize("seeds", (DISTINCT_RESOLVERS,), indirect=True)
def test_seed_lookup_rejects_ambiguous_active_canonical_links(seeds):
    other = CanonicalEntity.objects.create(
        identity_key="f" * 64,
        label="Alpha",
        normalized_label="alpha",
        entity_type="model",
        resolver_version="canonical-resolution-v2",
    )
    CanonicalEntityLink.objects.create(
        canonical_entity=other,
        collection_entity=seeds.entities[0],
        score=0.9,
        method="exact_name_or_alias",
        reason="synthetic second resolver",
        outcome="automatic",
        resolver_version=other.resolver_version,
    )
    with pytest.raises(ValueError, match="automatic membership is not unique"):
        seeds.direct.canonical_name_matches(
            span=seeds.spans[0], ready=seeds.scope.ready, limit=2
        )
    with pytest.raises(ValueError, match="automatic membership is not unique"):
        _lookup(seeds)
