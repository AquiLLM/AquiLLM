from __future__ import annotations

import os
import socket

import pytest
from django.conf import settings

from apps.knowledge_graph.models import (
    CollectionEntity,
    CollectionEntityDocumentLink,
    CollectionRelation,
    CollectionRelationEvidence,
    DocumentEntity,
    GraphArtifact,
    RelationMention,
)
from apps.knowledge_graph.models.entities import ResolutionStatus


def _database_is_reachable() -> bool:
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)), timeout=0.2
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _database_is_reachable() and os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
    reason="configured PostgreSQL database is not reachable",
)


def _where_lookups(node):
    for child in node.children:
        if hasattr(child, "children"):
            yield from _where_lookups(child)
        elif hasattr(child, "lhs") and hasattr(child.lhs, "target"):
            yield child


def _exact_filters(queryset, model, field_name):
    return tuple(
        (lookup.lhs.alias, lookup.rhs)
        for lookup in _where_lookups(queryset.query.where)
        if lookup.lookup_name == "exact"
        and lookup.lhs.target.model is model
        and lookup.lhs.target.name == field_name
    )


def _in_filters(queryset, model, field_name):
    return tuple(
        (lookup.lhs.alias, tuple(lookup.rhs))
        for lookup in _where_lookups(queryset.query.where)
        if lookup.lookup_name == "in"
        and lookup.lhs.target.model is model
        and lookup.lhs.target.name == field_name
    )


def test_current_collection_entities_require_active_row_and_artifact():
    current = CollectionEntity.objects.current()

    assert _exact_filters(current, CollectionEntity, "status") == (
        (CollectionEntity._meta.db_table, ResolutionStatus.ACTIVE),
    )
    artifact_filters = _exact_filters(current, GraphArtifact, "status")
    assert len(artifact_filters) == 1
    assert artifact_filters[0][1] == GraphArtifact.Status.ACTIVE


def test_current_collection_links_require_live_automatic_end_to_end_path():
    current = CollectionEntityDocumentLink.objects.current()

    assert _exact_filters(current, CollectionEntityDocumentLink, "status") == (
        (CollectionEntityDocumentLink._meta.db_table, ResolutionStatus.ACTIVE),
    )
    assert _exact_filters(current, CollectionEntityDocumentLink, "outcome") == (
        (
            CollectionEntityDocumentLink._meta.db_table,
            CollectionEntityDocumentLink.Outcome.AUTOMATIC,
        ),
    )
    assert len(_exact_filters(current, DocumentEntity, "status")) == 1
    assert len(_exact_filters(current, CollectionEntity, "status")) == 1

    artifact_filters = _exact_filters(current, GraphArtifact, "status")
    assert len(artifact_filters) == 1
    assert artifact_filters[0][1] == GraphArtifact.Status.ACTIVE
    document_artifact_filters = _in_filters(current, GraphArtifact, "status")
    assert len(document_artifact_filters) == 1
    assert document_artifact_filters[0][1] == (
        GraphArtifact.Status.ACTIVE,
        GraphArtifact.Status.SUPERSEDED,
    )


def test_current_relation_evidence_requires_active_current_relation():
    current = CollectionRelationEvidence.objects.current()
    sql, params = current.query.sql_with_params()

    assert _exact_filters(current, CollectionRelationEvidence, "status") == (
        (
            CollectionRelationEvidence._meta.db_table,
            CollectionRelationEvidence.Status.ACTIVE,
        ),
    )
    assert sql.upper().count("EXISTS") >= 2
    assert "apps_knowledge_graph_collectionrelation" in sql
    assert GraphArtifact.Status.ACTIVE in params
    assert ResolutionStatus.ACTIVE in params

    assert not CollectionRelationEvidence.objects.all().query.where.children
    for status in (
        CollectionRelationEvidence.Status.SUPPRESSED,
        CollectionRelationEvidence.Status.REJECTED,
    ):
        _sql, audit_params = CollectionRelationEvidence.objects.filter(
            status=status
        ).query.sql_with_params()
        assert status in audit_params


@pytest.mark.parametrize(
    "current",
    (
        CollectionRelation.objects.current(),
        CollectionRelationEvidence.objects.current(),
    ),
)
def test_current_relations_and_evidence_require_two_live_automatic_paths(current):
    sql, params = current.query.sql_with_params()

    assert sql.count(CollectionEntityDocumentLink._meta.db_table) >= 2
    assert params.count(CollectionEntityDocumentLink.Outcome.AUTOMATIC) >= 2


@pytest.mark.django_db(transaction=True)
@database_required
def test_current_managers_expose_only_live_graph_state_but_keep_audit_rows():
    from apps.knowledge_graph.tests.test_models import (
        _persist_collection_relation_fixture,
    )

    fixture = _persist_collection_relation_fixture()
    artifact = fixture.collection_artifact
    document_artifact = fixture.document_artifact

    document_artifact.status = GraphArtifact.Status.BUILDING
    document_artifact.save(update_fields=["status"])
    raw = fixture.relation_mention
    hidden_mentions = tuple(
        RelationMention.objects.create(
            artifact=document_artifact,
            document_id=raw.document_id,
            chunk=raw.chunk,
            head=raw.head,
            tail=raw.tail,
            relation_type=raw.relation_type,
            extraction_confidence=raw.extraction_confidence,
            metadata={"audit_status": status},
        )
        for status in (
            CollectionRelationEvidence.Status.SUPPRESSED,
            CollectionRelationEvidence.Status.REJECTED,
        )
    )
    document_artifact.status = GraphArtifact.Status.ACTIVE
    document_artifact.save(update_fields=["status"])
    hidden_evidence = tuple(
        CollectionRelationEvidence.objects.create(
            artifact=artifact,
            relation_mention=mention,
            status=status,
            reason=f"audit_{status}",
            ontology_checksum=artifact.ontology_checksum,
            assembly_config_checksum=artifact.assembly_config_checksum,
        )
        for mention, status in zip(
            hidden_mentions,
            (
                CollectionRelationEvidence.Status.SUPPRESSED,
                CollectionRelationEvidence.Status.REJECTED,
            ),
            strict=True,
        )
    )

    assert CollectionEntity.objects.filter(artifact=artifact).count() == 2
    assert CollectionEntityDocumentLink.objects.filter(artifact=artifact).count() == 2
    assert CollectionRelationEvidence.objects.filter(artifact=artifact).count() == 3
    assert not CollectionEntity.objects.current().filter(artifact=artifact).exists()
    assert (
        not CollectionEntityDocumentLink.objects.current()
        .filter(artifact=artifact)
        .exists()
    )
    assert (
        not CollectionRelationEvidence.objects.current()
        .filter(artifact=artifact)
        .exists()
    )

    artifact.status = GraphArtifact.Status.ACTIVE
    artifact.save(update_fields=["status"])

    assert set(
        CollectionEntity.objects.current()
        .filter(artifact=artifact)
        .values_list("pk", flat=True)
    ) == {fixture.relation.source_id, fixture.relation.target_id}
    assert set(
        CollectionEntityDocumentLink.objects.current()
        .filter(artifact=artifact)
        .values_list("pk", flat=True)
    ) == {fixture.head_mapping.pk, fixture.tail_mapping.pk}
    assert list(
        CollectionRelationEvidence.objects.current()
        .filter(artifact=artifact)
        .values_list("pk", flat=True)
    ) == [fixture.evidence.pk]
    assert set(
        CollectionRelationEvidence.objects.filter(
            status__in=(
                CollectionRelationEvidence.Status.SUPPRESSED,
                CollectionRelationEvidence.Status.REJECTED,
            )
        ).values_list("pk", flat=True)
    ) == {item.pk for item in hidden_evidence}

    document_artifact.status = GraphArtifact.Status.SUPERSEDED
    document_artifact.save(update_fields=["status"])
    assert (
        CollectionEntityDocumentLink.objects.current()
        .filter(artifact=artifact)
        .exists()
    )
    assert CollectionRelation.objects.current().filter(artifact=artifact).exists()
    assert (
        CollectionRelationEvidence.objects.current().filter(artifact=artifact).exists()
    )

    document_artifact.status = GraphArtifact.Status.STALE
    document_artifact.save(update_fields=["status"])
    assert (
        not CollectionEntityDocumentLink.objects.current()
        .filter(artifact=artifact)
        .exists()
    )
    assert CollectionEntity.objects.current().filter(artifact=artifact).exists()
    assert not CollectionRelation.objects.current().filter(artifact=artifact).exists()
    assert (
        not CollectionRelationEvidence.objects.current()
        .filter(artifact=artifact)
        .exists()
    )

    artifact.status = GraphArtifact.Status.SUPERSEDED
    artifact.save(update_fields=["status"])
    assert not CollectionEntity.objects.current().filter(artifact=artifact).exists()
    assert (
        not CollectionRelationEvidence.objects.current()
        .filter(artifact=artifact)
        .exists()
    )
    assert CollectionRelationEvidence.objects.filter(artifact=artifact).count() == 3
