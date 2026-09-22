# ruff: noqa: F401,F811
"""Real PostgreSQL seed selection and projection identity parity."""

import os
from types import SimpleNamespace
from uuid import uuid4

import pytest
from django.contrib.auth.models import User
from django.db import connection, connections
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from apps.collections.models import Collection, CollectionPermission
from apps.collections.services.django_retrieval_authorization import (
    DjangoCollectionRetrievalPermissionPolicy,
)
from apps.collections.services.retrieval_authorization import (
    bind_retrieval_reauthorization_capability,
    freeze_retrieval_authorization_context,
)
from apps.documents.models import RawTextDocument, TextChunk
from apps.knowledge_graph.models import (
    CanonicalEntity,
    CanonicalEntityLink,
    CollectionArtifactInput,
    CollectionEntity,
    CollectionEntityDocumentLink,
    CollectionGraphMembershipState,
    CollectionGraphProjection,
    DocumentEntity,
    DocumentEntityMention,
    EntityMention,
)
from apps.knowledge_graph.projection.django_projection_rows import (
    DjangoProjectionOrmLoader,
)
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
)
from apps.knowledge_graph.projection.identifiers import (
    ProjectionIdentifierDomain as Domain,
)
from apps.knowledge_graph.projection.memberships import (
    MEMBERSHIP_REGISTRY_GENERATION,
    load_automatic_membership_assignments,
    membership_decision_checksum,
)
from apps.knowledge_graph.projection.projection_encoding import _entities, _memberships
from apps.knowledge_graph.retrieval.direct_seed_contracts import (
    DirectResolutionSpanInputV1,
)
from apps.knowledge_graph.retrieval.direct_seed_repository import (
    DirectSeedRepository,
    DirectSeedScopeV1,
)
from apps.knowledge_graph.retrieval.extended_seed_repository import (
    ExtendedSeedRepository,
)
from apps.knowledge_graph.retrieval.ready_scope import (
    ReadyProjectionAuthorityV1,
    assemble_selected_ready_scope,
)
from apps.knowledge_graph.tests.seed_lookup_fixtures import seeds
from apps.knowledge_graph.tests.test_models import _artifact
from lib.knowledge_graph.query_extractor.contracts import QueryEntitySpanV1

pytestmark = [
    pytest.mark.django_db(transaction=True, databases="__all__"),
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="set KG_REQUIRE_POSTGRES_TESTS=1 for PostgreSQL integration tests",
    ),
]




def _lookup(fixture, chunks, max_rows):
    return ExtendedSeedRepository().load_seed_identities(
        authority=fixture.authority,
        chunks=tuple((chunk.pk, fixture.document.id) for chunk in chunks),
        authorization=fixture.authorization,
        codec=fixture.codec,
        max_rows=max_rows,
    )


@pytest.mark.parametrize("seeds", ("document-coreference-v3",), indirect=True)
def test_direct_alias_uses_document_resolver_and_rejects_stale_mention_link(seeds):
    assert seeds.document_artifact.resolver_version != seeds.artifact.resolver_version
    span = seeds.spans[0]
    name_matches = seeds.direct.canonical_name_matches(
        span=span, ready=seeds.scope.ready, limit=2
    )
    alias_matches = seeds.direct.indexed_alias_matches(
        span=span, ready=seeds.scope.ready, limit=2
    )
    assert len(name_matches) == len(alias_matches) == 1
    assert alias_matches[0].entity_key == name_matches[0].entity_key
    assert alias_matches[0].component_key == name_matches[0].component_key

    # Simulate a legacy malformed/stale row bypassing the model's immutable
    # provenance validation. Matching the collection resolver is insufficient.
    link = DocumentEntityMention.objects.get(document_entity=seeds.document_entities[0])
    with connection.cursor() as cursor:
        cursor.execute(
            f"UPDATE {DocumentEntityMention._meta.db_table} "
            "SET resolver_version=%s WHERE id=%s",
            [seeds.artifact.resolver_version, link.pk],
        )
    assert seeds.direct.indexed_alias_matches(
        span=span, ready=seeds.scope.ready, limit=2
    ) == ()
    assert seeds.direct.canonical_name_matches(
        span=span, ready=seeds.scope.ready, limit=2
    ) == name_matches


def test_real_seed_rows_match_projection_for_representative_and_observation(seeds):
    loader = DjangoProjectionOrmLoader(using="projection_source")
    snapshot = {
        "entities": loader._entities(
            seeds.artifact.pk, seeds.collection.pk, 50, "audit"
        ),
        "memberships": loader._memberships(
            tuple(entity.pk for entity in seeds.entities), 50
        ),
        "artifacts": [
            {
                "id": seeds.artifact.pk,
                "resolver_version": seeds.artifact.resolver_version,
                "resolution_config_checksum": seeds.artifact.resolution_config_checksum,
            }
        ],
    }

    def key(domain, source):
        return seeds.codec.encode(
            domain,
            generation=seeds.projection.generation_key,
            source=source,
        ).value

    generation = seeds.scope.ready.selected_generations[0]
    projected, keys = _entities(
        snapshot,
        key,
        SimpleNamespace(
            generation_key=generation.generation_key,
            artifact_key=generation.active_artifact_key,
            collection_key=generation.collection_key,
        ),
    )
    memberships = _memberships(
        snapshot,
        seeds.codec,
        seeds.projection.generation_key,
        {
            "artifact_id": seeds.artifact.pk,
            "membership_checksum": seeds.state.membership_checksum,
        },
        projected,
        keys,
    )
    expected = {
        row.entity_key: row.automatic_membership_key or row.entity_key
        for row in memberships
    }
    canonical_key = seeds.codec.encode(
        Domain.AUTOMATIC_CANONICAL_IDENTITY, source=seeds.canonical.pk
    ).value
    assert type(seeds.canonical.pk) is int
    assert (
        canonical_key
        != seeds.codec.encode(
            Domain.AUTOMATIC_CANONICAL_IDENTITY, source=seeds.canonical.identity_key
        ).value
    )
    with CaptureQueriesContext(connections["projection_source"]) as queries:
        result = _lookup(seeds, seeds.chunks[:2], max_rows=3)
    assert result == {
        seeds.chunks[0].pk: tuple(
            sorted(expected[keys[entity.pk]] for entity in seeds.entities[:2])
        ),
        seeds.chunks[1].pk: (canonical_key,),
    }
    assert expected[keys[seeds.entities[2].pk]] not in set().union(*result.values())
    seed_sql = [row["sql"] for row in queries if "SELECT DISTINCT" in row["sql"]]
    assert len(seed_sql) == 2
    assert all("LIMIT" in sql and "@>" in sql for sql in seed_sql)
    for span, entity in zip(seeds.spans, seeds.entities[:2], strict=True):
        matches = seeds.direct.canonical_name_matches(
            span=span, ready=seeds.scope.ready, limit=2
        )
        assert len(matches) == 1
        assert matches[0].entity_key == keys[entity.pk]
        assert matches[0].component_key == expected[keys[entity.pk]]
    with pytest.raises(ValueError, match="hard cap"):
        _lookup(seeds, seeds.chunks[:2], max_rows=2)


@pytest.mark.parametrize("invalid", ("revoked", "stale"))
def test_real_seed_lookup_rejects_revoked_permission_and_stale_membership(
    seeds, invalid
):
    assert _lookup(seeds, seeds.chunks[1:2], max_rows=1)
    if invalid == "revoked":
        seeds.permission.delete()
    else:
        seeds.state.registry_epoch += 1
        seeds.state.save(update_fields=["registry_epoch"])
    with pytest.raises(ValueError):
        _lookup(seeds, seeds.chunks[1:2], max_rows=1)
    if invalid == "stale":
        with pytest.raises(ValueError, match="membership state"):
            seeds.direct.canonical_name_matches(
                span=seeds.spans[0], ready=seeds.scope.ready, limit=2
            )
