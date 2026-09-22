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
from apps.knowledge_graph.tests.test_models import _artifact
from lib.knowledge_graph.query_extractor.contracts import QueryEntitySpanV1

pytestmark = [
    pytest.mark.django_db(transaction=True, databases="__all__"),
    pytest.mark.skipif(
        os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
        reason="set KG_REQUIRE_POSTGRES_TESTS=1 for PostgreSQL integration tests",
    ),
]


@pytest.fixture
def seeds():
    assert connection.vendor == "postgresql"
    user = User.objects.create_user(username=f"seed-{uuid4()}")
    collection = Collection.objects.create(name="seed integration")
    permission = CollectionPermission.objects.create(
        user=user, collection=collection, permission="VIEW"
    )
    content = "Alpha Beta Gamma"
    document = RawTextDocument(
        title="seed integration",
        full_text=content,
        collection=collection,
        ingested_by=user,
        full_text_hash=RawTextDocument.hash_fn(content),
    )
    document.save(dont_rechunk=True)
    chunks = tuple(
        TextChunk.objects.create(
            content=content[start:end],
            start_position=start,
            end_position=end,
            chunk_number=number,
            doc_id=document.id,
            embedding=[0.0] * 1024,
        )
        for number, (start, end) in enumerate(((0, 10), (0, 5), (11, 16)))
    )
    artifact = _artifact(
        scope_type="collection", scope_id=collection.pk, ontology_checksum="7" * 64
    )
    artifact.save()
    document_artifact = _artifact(scope_id=document.id, ontology_checksum="7" * 64)
    document_artifact.save()
    document_entities = []
    for index, (label, start, end) in enumerate(
        (("Alpha", 0, 5), ("Beta", 6, 10), ("Gamma", 11, 16))
    ):
        mention = EntityMention.objects.create(
            artifact=document_artifact,
            document_id=document.id,
            chunk=chunks[2 if index == 2 else 0],
            start=start,
            end=end,
            raw_text=label,
            normalized_text=label.lower(),
            entity_type="model",
            position_basis=EntityMention.PositionBasis.DOCUMENT_GLOBAL,
            extraction_confidence=0.9,
            metadata={"observations": [{"chunk_id": chunks[1].pk}]}
            if index == 0
            else {},
        )
        entity = DocumentEntity.objects.create(
            artifact=document_artifact,
            document_id=document.id,
            cluster_key=str(index + 1) * 64,
            label=label,
            normalized_label=label.lower(),
            entity_type="model",
        )
        DocumentEntityMention.objects.create(
            document_entity=entity,
            mention=mention,
            method="root",
            resolver_version=document_artifact.resolver_version,
        )
        document_entities.append(entity)
    document_artifact.status = "active"
    document_artifact.save(update_fields=["status"])
    manifest = CollectionArtifactInput.objects.create(
        artifact=artifact,
        collection=collection,
        document_id=document.id,
        document_artifact=document_artifact,
        source_signature="0" * 64,
        build_signature="0" * 64,
    )
    entities = []
    for index, document_entity in enumerate(document_entities):
        entity = CollectionEntity.objects.create(
            artifact=artifact,
            collection=collection,
            cluster_key=str(index + 4) * 64,
            label=document_entity.label,
            normalized_label=document_entity.normalized_label,
            entity_type="model",
            retrieval_utility=0.5,
            extraction_confidence=0.9,
            resolution_confidence=0.9,
        )
        CollectionEntityDocumentLink.objects.create(
            artifact=artifact,
            manifest_input=manifest,
            document_entity=document_entity,
            collection_entity=entity,
            score=0.9,
            method="exact",
            reason="exact",
            resolver_version=artifact.resolver_version,
            outcome="automatic",
            decision_checksum=str(index + 7) * 64,
        )
        entities.append(entity)
    canonical = CanonicalEntity.objects.create(
        identity_key="a" * 64,
        label="Alpha",
        normalized_label="alpha",
        entity_type="model",
        resolver_version=artifact.resolver_version,
    )
    CanonicalEntityLink.objects.create(
        canonical_entity=canonical,
        collection_entity=entities[0],
        score=0.9,
        method="exact_name_or_alias",
        reason="exact",
        outcome="automatic",
        resolver_version=artifact.resolver_version,
    )
    artifact.status = "active"
    artifact.save(update_fields=["status"])
    codec = HmacSha256ProjectionIdentifierCodec(b"audit-test-key", key_version="key-v1")
    checksum = membership_decision_checksum(
        load_automatic_membership_assignments(
            collection_ids=(collection.pk,),
            using="default",
            batch_size=50,
            codec=codec,
            generation=MEMBERSHIP_REGISTRY_GENERATION,
        )
    )
    state = CollectionGraphMembershipState.objects.create(
        collection=collection,
        active_artifact=artifact,
        registry_epoch=1,
        membership_checksum=checksum,
        resolver_version=artifact.resolver_version,
        resolution_config_checksum=artifact.resolution_config_checksum,
    )
    projection = CollectionGraphProjection.objects.create(
        collection=collection,
        collection_pk_snapshot=collection.pk,
        artifact=artifact,
        artifact_pk_snapshot=artifact.pk,
        state="ready",
        schema_version="collection-graph-v1",
        projection_version="projection-v1",
        identifier_key_version="key-v1",
        membership_epoch=1,
        membership_checksum=checksum,
        graph_checksum="b" * 64,
        snapshot_checksum="d" * 64,
        private_mapping_checksum="c" * 64,
        ready_at=timezone.now(),
    )
    authority = ReadyProjectionAuthorityV1(
        projection_id=projection.pk,
        generation_id=projection.generation_key,
        collection_id=collection.pk,
        artifact_id=artifact.pk,
        **{
            name: getattr(projection, name)
            for name in (
                "schema_version",
                "projection_version",
                "identifier_key_version",
                "membership_epoch",
                "membership_checksum",
                "graph_checksum",
                "private_mapping_checksum",
            )
        },
        **{
            name: getattr(artifact, name)
            for name in (
                "resolver_version",
                "resolution_config_checksum",
                "ontology_version",
                "ontology_checksum",
                "embedding_model_signature",
            )
        },
        documents=((document.id, document_artifact.pk),),
    )
    policy = DjangoCollectionRetrievalPermissionPolicy(b"audit-test-only")
    authorization = freeze_retrieval_authorization_context(
        principal=user,
        database_alias="default",
        policy=policy,
        selected_collection_ids=(collection.pk,),
        selected_document_ids=(document.id,),
        reauthorization_capability=bind_retrieval_reauthorization_capability(
            principal=user,
            policy=policy,
        ),
    )
    scope = assemble_selected_ready_scope(
        authorization=authorization,
        authorities=(authority,),
        codec=codec,
    )
    spans = tuple(
        QueryEntitySpanV1("model", start, end, 1.0) for start, end in ((0, 5), (6, 10))
    )
    direct = DirectSeedRepository(
        scope=DirectSeedScopeV1(
            ready_bundle_checksum=scope.ready.bundle_checksum,
            selected_collection_ids=(collection.pk,),
            selected_artifact_ids=(artifact.pk,),
            selected_document_ids=(document.id,),
            selected_document_artifact_ids=(document_artifact.pk,),
            generation_keys_by_artifact=(
                (artifact.pk, scope.ready.selected_generations[0].generation_key),
            ),
            generation_ids_by_artifact=((artifact.pk, projection.generation_key),),
            ontology_checksum=artifact.ontology_checksum,
            resolver_version=artifact.resolver_version,
        ),
        codec=codec,
        span_inputs=tuple(
            DirectResolutionSpanInputV1(span, content[span.start : span.end])
            for span in spans
        ),
    )
    return SimpleNamespace(**locals())


def _lookup(fixture, chunks, max_rows):
    return ExtendedSeedRepository().load_seed_identities(
        authority=fixture.authority,
        chunks=tuple((chunk.pk, fixture.document.id) for chunk in chunks),
        authorization=fixture.authorization,
        codec=fixture.codec,
        max_rows=max_rows,
    )


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
