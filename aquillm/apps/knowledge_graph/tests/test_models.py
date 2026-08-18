from __future__ import annotations

import socket
import uuid

import pytest
from django.apps import apps
from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.db.models import CheckConstraint, UniqueConstraint
from pgvector.django import VectorField

from apps.documents.models import TextChunk
from apps.knowledge_graph.models import (
    CanonicalEntity,
    CanonicalEntityLink,
    CollectionEntity,
    CollectionEntityDocumentLink,
    CollectionRelation,
    CollectionRelationEvidence,
    DocumentEntity,
    DocumentEntityMention,
    EntityMention,
    GraphArtifact,
    GraphBuildRun,
    OntologyVersion,
    RelationMention,
)

DOCUMENT_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
COLLECTION_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")


def _database_is_reachable():
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)), timeout=0.2
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _database_is_reachable(),
    reason="configured PostgreSQL database is not reachable",
)


def _constraint(model, name, constraint_type):
    return next(
        constraint
        for constraint in model._meta.constraints
        if constraint.name == name and isinstance(constraint, constraint_type)
    )


def _index_fields(model):
    return {tuple(index.fields) for index in model._meta.indexes}


def _artifact(**overrides):
    values = {
        "scope_type": GraphArtifact.ScopeType.DOCUMENT,
        "scope_id": DOCUMENT_ID,
        "status": GraphArtifact.Status.BUILDING,
        "source_hash": "a" * 64,
        "ontology_version": "ontology-v1",
        "extractor_version": "extractor-v1",
        "resolver_version": "resolver-v1",
        "filter_policy_version": "filter-v1",
    }
    values.update(overrides)
    return GraphArtifact(**values)


def _chunk(*, modality=TextChunk.Modality.TEXT, number=0):
    return TextChunk.objects.create(
        content="Aquilla evaluates MMLU.",
        start_position=0,
        end_position=24,
        chunk_number=number,
        modality=modality,
        doc_id=DOCUMENT_ID,
        embedding=[0.0] * 1024,
    )


def _mention(artifact, chunk, **overrides):
    values = {
        "artifact": artifact,
        "document_id": DOCUMENT_ID,
        "chunk": chunk,
        "start": 0,
        "end": 7,
        "position_basis": EntityMention.PositionBasis.DOCUMENT_GLOBAL,
        "raw_text": "Aquilla",
        "normalized_text": "aquilla",
        "entity_type": "model",
        "extraction_confidence": 0.9,
    }
    values.update(overrides)
    return EntityMention.objects.create(**values)


def test_app_is_registered_with_domain_app_label():
    config = apps.get_app_config("apps_knowledge_graph")

    assert config.name == "apps.knowledge_graph"
    assert "apps.knowledge_graph" in settings.INSTALLED_APPS


def test_graph_artifact_has_scope_lifecycle_identity_constraints_and_indexes():
    assert GraphArtifact._meta.get_field("scope_id").get_internal_type() == "UUIDField"
    assert (
        GraphArtifact._meta.get_field("filter_policy_version").get_internal_type()
        == "CharField"
    )
    assert {value for value, _ in GraphArtifact.ScopeType.choices} == {
        "document",
        "collection",
    }
    assert {
        "building",
        "active",
        "failed",
        "stale",
        "superseded",
    }.issubset({value for value, _ in GraphArtifact.Status.choices})

    active = _constraint(
        GraphArtifact, "kg_one_active_artifact_per_scope", UniqueConstraint
    )
    identity = _constraint(
        GraphArtifact, "kg_artifact_build_identity", UniqueConstraint
    )
    assert tuple(active.fields) == ("scope_type", "scope_id")
    assert active.condition is not None
    assert tuple(identity.fields) == (
        "scope_type",
        "scope_id",
        "source_hash",
        "ontology_version",
        "extractor_version",
        "resolver_version",
        "filter_policy_version",
    )
    assert {("scope_type", "scope_id", "status"), ("source_hash",)} <= _index_fields(
        GraphArtifact
    )


def test_graph_build_run_and_ontology_are_typed_and_audit_safe():
    artifact_field = GraphBuildRun._meta.get_field("artifact")
    assert artifact_field.null is True
    assert artifact_field.remote_field.on_delete.__name__ == "SET_NULL"
    assert {"pending", "running", "succeeded", "failed", "cancelled"} <= {
        value for value, _ in GraphBuildRun.Status.choices
    }
    assert {"extraction", "resolution", "filtering", "persistence", "complete"} <= {
        value for value, _ in GraphBuildRun.Stage.choices
    }
    assert GraphBuildRun._meta.get_field("stats").get_internal_type() == "JSONField"
    assert GraphBuildRun._meta.get_field("timings").get_internal_type() == "JSONField"
    assert _constraint(GraphBuildRun, "kg_build_run_attempt_positive", CheckConstraint)

    assert {"draft", "active", "superseded", "rejected"} == {
        value for value, _ in OntologyVersion.Status.choices
    }
    assert _constraint(
        OntologyVersion, "kg_ontology_kind_version_unique", UniqueConstraint
    )


def test_entity_mention_has_typed_provenance_constraints_and_indexes():
    chunk_field = EntityMention._meta.get_field("chunk")
    assert chunk_field.remote_field.model is TextChunk
    assert chunk_field.remote_field.on_delete.__name__ == "CASCADE"
    assert (
        EntityMention._meta.get_field("document_id").get_internal_type() == "UUIDField"
    )
    assert (
        EntityMention._meta.get_field("content_object_id").get_internal_type()
        == "UUIDField"
    )
    assert (
        EntityMention._meta.get_field(
            "content_object_type"
        ).remote_field.on_delete.__name__
        == "PROTECT"
    )
    assert {value for value, _ in EntityMention.PositionBasis.choices} == {
        "document_global",
        "chunk_content",
    }
    assert _constraint(EntityMention, "kg_mention_nonempty_span", CheckConstraint)
    assert _constraint(EntityMention, "kg_mention_confidence_range", CheckConstraint)
    assert {
        ("document_id", "chunk"),
        ("normalized_text", "entity_type"),
    } <= _index_fields(EntityMention)


@pytest.mark.parametrize(
    "start,end,confidence",
    [(-1, 1, 0.5), (1, 1, 0.5), (2, 1, 0.5), (0, 1, -0.01), (0, 1, 1.01)],
)
def test_entity_mention_rejects_invalid_span_or_confidence(start, end, confidence):
    mention = EntityMention(
        artifact=_artifact(),
        document_id=DOCUMENT_ID,
        chunk=TextChunk(modality=TextChunk.Modality.TEXT, doc_id=DOCUMENT_ID),
        start=start,
        end=end,
        position_basis=EntityMention.PositionBasis.DOCUMENT_GLOBAL,
        raw_text="Aquilla",
        normalized_text="aquilla",
        entity_type="model",
        extraction_confidence=confidence,
    )

    with pytest.raises(ValidationError):
        mention.clean()


def test_entity_mention_requires_document_global_positions_for_text_chunks():
    mention = EntityMention(
        artifact=_artifact(),
        document_id=DOCUMENT_ID,
        chunk=TextChunk(modality=TextChunk.Modality.TEXT, doc_id=DOCUMENT_ID),
        start=0,
        end=7,
        position_basis=EntityMention.PositionBasis.CHUNK_CONTENT,
        raw_text="Aquilla",
        normalized_text="aquilla",
        entity_type="model",
        extraction_confidence=0.9,
    )

    with pytest.raises(ValidationError, match="document_global"):
        mention.clean()


def test_text_entity_mention_rejects_image_content_object_provenance():
    mention = EntityMention(
        artifact=_artifact(),
        document_id=DOCUMENT_ID,
        chunk=TextChunk(modality=TextChunk.Modality.TEXT, doc_id=DOCUMENT_ID),
        start=0,
        end=7,
        position_basis=EntityMention.PositionBasis.DOCUMENT_GLOBAL,
        raw_text="Aquilla",
        normalized_text="aquilla",
        entity_type="model",
        extraction_confidence=0.9,
        content_object_type=ContentType(
            pk=1,
            app_label="apps_documents",
            model="documentfigure",
        ),
        content_object_id=DOCUMENT_ID,
    )

    with pytest.raises(ValidationError, match="must not include image provenance"):
        mention.clean()


def test_entity_mention_requires_chunk_positions_and_content_object_for_image_chunks():
    image_chunk = TextChunk(modality=TextChunk.Modality.IMAGE, doc_id=DOCUMENT_ID)
    wrong_basis = EntityMention(
        artifact=_artifact(),
        document_id=DOCUMENT_ID,
        chunk=image_chunk,
        start=0,
        end=7,
        position_basis=EntityMention.PositionBasis.DOCUMENT_GLOBAL,
        raw_text="Aquilla",
        normalized_text="aquilla",
        entity_type="model",
        extraction_confidence=0.9,
        content_object_type=ContentType(
            pk=1,
            app_label="apps_documents",
            model="documentfigure",
        ),
        content_object_id=DOCUMENT_ID,
    )
    missing_object = EntityMention(
        artifact=_artifact(),
        document_id=DOCUMENT_ID,
        chunk=image_chunk,
        start=0,
        end=7,
        position_basis=EntityMention.PositionBasis.CHUNK_CONTENT,
        raw_text="Aquilla",
        normalized_text="aquilla",
        entity_type="model",
        extraction_confidence=0.9,
    )

    with pytest.raises(ValidationError, match="chunk_content"):
        wrong_basis.clean()
    with pytest.raises(ValidationError, match="content object"):
        missing_object.clean()


def test_entity_mention_rejects_unrelated_image_content_object_types():
    mention = EntityMention(
        artifact=_artifact(),
        document_id=DOCUMENT_ID,
        chunk=TextChunk(modality=TextChunk.Modality.IMAGE, doc_id=DOCUMENT_ID),
        start=0,
        end=7,
        position_basis=EntityMention.PositionBasis.CHUNK_CONTENT,
        raw_text="Aquilla",
        normalized_text="aquilla",
        entity_type="model",
        extraction_confidence=0.9,
        content_object_type=ContentType(pk=9, app_label="auth", model="user"),
        content_object_id=DOCUMENT_ID,
    )

    with pytest.raises(ValidationError, match="Image content object"):
        mention.clean()


def test_resolved_entities_are_explicitly_owned_and_only_resolved_nodes_have_vectors():
    assert (
        DocumentEntity._meta.get_field("artifact").remote_field.model is GraphArtifact
    )
    assert (
        DocumentEntity._meta.get_field("document_id").get_internal_type() == "UUIDField"
    )
    assert DocumentEntityMention._meta.get_field("mention").unique is True
    assert (
        DocumentEntityMention._meta.get_field(
            "document_entity"
        ).remote_field.on_delete.__name__
        == "CASCADE"
    )
    assert (
        DocumentEntityMention._meta.get_field("mention").remote_field.on_delete.__name__
        == "CASCADE"
    )

    collection_vector = CollectionEntity._meta.get_field("embedding")
    canonical_vector = CanonicalEntity._meta.get_field("embedding")
    assert isinstance(collection_vector, VectorField)
    assert isinstance(canonical_vector, VectorField)
    assert collection_vector.dimensions == canonical_vector.dimensions == 1024
    assert not any(
        isinstance(field, VectorField) for field in EntityMention._meta.fields
    )
    assert not any(
        isinstance(field, VectorField) for field in DocumentEntity._meta.fields
    )
    assert {("collection_id", "entity_type", "normalized_label")} <= _index_fields(
        CollectionEntity
    )
    assert {("entity_type", "normalized_label")} <= _index_fields(CanonicalEntity)


def test_associations_preserve_rejections_and_constrain_scores_and_targets():
    for model in (CollectionEntityDocumentLink, CanonicalEntityLink):
        assert {"active", "suppressed", "rejected", "superseded"} <= {
            value for value, _ in model.Status.choices
        }
        assert any(
            isinstance(item, CheckConstraint) for item in model._meta.constraints
        )

    assert _constraint(
        CollectionEntityDocumentLink,
        "kg_doc_collection_entity_link_unique",
        UniqueConstraint,
    )
    assert _constraint(
        CanonicalEntityLink, "kg_collection_canonical_link_unique", UniqueConstraint
    )
    assert {
        ("status", "collection_entity"),
        ("status", "document_entity"),
    } <= _index_fields(CollectionEntityDocumentLink)
    assert {
        ("status", "canonical_entity"),
        ("status", "collection_entity"),
    } <= _index_fields(CanonicalEntityLink)


def test_relation_models_have_versioned_uniqueness_evidence_and_indexes():
    assert (
        RelationMention._meta.get_field("chunk").remote_field.on_delete.__name__
        == "CASCADE"
    )
    assert RelationMention._meta.get_field("head").remote_field.model is EntityMention
    assert RelationMention._meta.get_field("tail").remote_field.model is EntityMention
    assert _constraint(
        CollectionRelation, "kg_collection_relation_unique", UniqueConstraint
    )
    assert _constraint(
        CollectionRelationEvidence, "kg_relation_evidence_unique", UniqueConstraint
    )
    assert {("artifact", "source", "target", "relation_type")} <= _index_fields(
        CollectionRelation
    )


def test_collection_relation_rejects_cross_artifact_or_cross_collection_endpoints():
    source = CollectionEntity(
        pk=10,
        artifact_id=1,
        collection_id=COLLECTION_ID,
        label="Aquilla",
        normalized_label="aquilla",
        entity_type="model",
    )
    target = CollectionEntity(
        pk=11,
        artifact_id=2,
        collection_id=uuid.uuid4(),
        label="MMLU",
        normalized_label="mmlu",
        entity_type="benchmark",
    )
    relation = CollectionRelation(
        artifact_id=1,
        source=source,
        target=target,
        relation_type="evaluates_on",
        support_count=1,
        confidence=0.8,
    )

    with pytest.raises(ValidationError, match="artifact|collection"):
        relation.clean()


def test_relation_evidence_rejects_a_different_relation_type():
    relation = CollectionRelation(pk=1, relation_type="evaluates_on")
    mention = RelationMention(pk=2, relation_type="trained_on")
    evidence = CollectionRelationEvidence(
        relation=relation,
        relation_mention=mention,
    )

    with pytest.raises(ValidationError, match="relation type"):
        evidence.clean()


def _unsaved_relation_evidence():
    collection_artifact = _artifact(
        pk=1,
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=COLLECTION_ID,
    )
    document_artifact = _artifact(pk=2, source_hash="b" * 64)
    source = CollectionEntity(
        pk=10,
        artifact=collection_artifact,
        collection_id=COLLECTION_ID,
        label="Aquilla",
        normalized_label="aquilla",
        entity_type="model",
    )
    target = CollectionEntity(
        pk=11,
        artifact=collection_artifact,
        collection_id=COLLECTION_ID,
        label="MMLU",
        normalized_label="mmlu",
        entity_type="benchmark",
    )
    head = EntityMention(pk=20, artifact=document_artifact, document_id=DOCUMENT_ID)
    tail = EntityMention(pk=21, artifact=document_artifact, document_id=DOCUMENT_ID)
    relation_mention = RelationMention(
        pk=30,
        artifact=document_artifact,
        document_id=DOCUMENT_ID,
        head=head,
        tail=tail,
        relation_type="evaluates_on",
    )
    relation = CollectionRelation(
        pk=40,
        artifact=collection_artifact,
        source=source,
        target=target,
        relation_type="evaluates_on",
    )
    return CollectionRelationEvidence(
        relation=relation,
        relation_mention=relation_mention,
    )


def test_relation_evidence_rejects_swapped_endpoint_mappings(monkeypatch):
    evidence = _unsaved_relation_evidence()
    mapped_pairs = {
        (evidence.relation_mention.head_id, evidence.relation.target_id),
        (evidence.relation_mention.tail_id, evidence.relation.source_id),
    }
    monkeypatch.setattr(
        evidence,
        "_endpoint_has_active_mapping",
        lambda mention_id, entity_id: (mention_id, entity_id) in mapped_pairs,
        raising=False,
    )

    with pytest.raises(ValidationError, match="head|tail|mapped"):
        evidence.clean()


def test_relation_evidence_rejects_unmapped_endpoint(monkeypatch):
    evidence = _unsaved_relation_evidence()
    monkeypatch.setattr(
        evidence,
        "_endpoint_has_active_mapping",
        lambda _mention_id, _entity_id: False,
    )

    with pytest.raises(ValidationError, match="head|tail|mapped"):
        evidence.clean()


def test_relation_evidence_rejects_endpoint_from_other_artifact_or_collection(
    monkeypatch,
):
    evidence = _unsaved_relation_evidence()
    evidence.relation.target.artifact_id = 999
    evidence.relation.target.collection_id = uuid.uuid4()
    monkeypatch.setattr(
        evidence,
        "_endpoint_has_active_mapping",
        lambda _mention_id, _entity_id: True,
        raising=False,
    )

    with pytest.raises(ValidationError, match="artifact|collection"):
        evidence.clean()


def test_relation_evidence_accepts_separate_document_artifact_when_actively_mapped(
    monkeypatch,
):
    evidence = _unsaved_relation_evidence()
    monkeypatch.setattr(
        evidence,
        "_endpoint_has_active_mapping",
        lambda _mention_id, _entity_id: True,
        raising=False,
    )

    evidence.clean()


@pytest.mark.django_db(transaction=True)
@database_required
def test_database_enforces_one_active_artifact_and_version_identity():
    first = _artifact(status=GraphArtifact.Status.ACTIVE)
    first.save()

    with pytest.raises(IntegrityError), transaction.atomic():
        _artifact(status=GraphArtifact.Status.ACTIVE, source_hash="b" * 64).save()
    with pytest.raises(IntegrityError), transaction.atomic():
        _artifact(status=GraphArtifact.Status.FAILED).save()


@pytest.mark.django_db
@database_required
def test_artifact_build_identity_is_immutable_after_creation():
    artifact = _artifact()
    artifact.save()
    artifact.extractor_version = "extractor-v2"

    with pytest.raises(ValidationError, match="immutable"):
        artifact.save()


@pytest.mark.django_db(transaction=True)
@database_required
def test_deleting_chunk_cascades_evidence_but_retains_completed_build_audit():
    artifact = _artifact(status=GraphArtifact.Status.ACTIVE)
    artifact.save()
    run = GraphBuildRun.objects.create(
        artifact=artifact,
        stage=GraphBuildRun.Stage.COMPLETE,
        status=GraphBuildRun.Status.SUCCEEDED,
        attempt=1,
        stats={"mentions": 2, "relations": 1},
        timings={"total_ms": 12.5},
    )
    chunk = _chunk()
    head = _mention(artifact, chunk)
    tail = _mention(
        artifact,
        chunk,
        start=18,
        end=22,
        raw_text="MMLU",
        normalized_text="mmlu",
        entity_type="benchmark",
    )
    relation_mention = RelationMention.objects.create(
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=chunk,
        head=head,
        tail=tail,
        relation_type="evaluates_on",
        extraction_confidence=0.8,
    )
    document_entity = DocumentEntity.objects.create(
        artifact=artifact,
        document_id=DOCUMENT_ID,
        label="Aquilla",
        normalized_label="aquilla",
        entity_type="model",
    )
    mention_link = DocumentEntityMention.objects.create(
        document_entity=document_entity,
        mention=head,
    )

    chunk.delete()

    assert not EntityMention.objects.filter(pk__in=(head.pk, tail.pk)).exists()
    assert not RelationMention.objects.filter(pk=relation_mention.pk).exists()
    assert not DocumentEntityMention.objects.filter(pk=mention_link.pk).exists()
    run.refresh_from_db()
    assert run.status == GraphBuildRun.Status.SUCCEEDED
    assert run.stats == {"mentions": 2, "relations": 1}


@pytest.mark.django_db(transaction=True)
@database_required
def test_deleting_resolution_entities_or_links_preserves_raw_mentions():
    artifact = _artifact()
    artifact.save()
    chunk = _chunk()
    mention = _mention(artifact, chunk)
    document_entity = DocumentEntity.objects.create(
        artifact=artifact,
        document_id=DOCUMENT_ID,
        label="Aquilla",
        normalized_label="aquilla",
        entity_type="model",
    )
    link = DocumentEntityMention.objects.create(
        document_entity=document_entity,
        mention=mention,
    )

    document_entity.delete()

    assert EntityMention.objects.filter(pk=mention.pk).exists()
    assert not DocumentEntityMention.objects.filter(pk=link.pk).exists()

    replacement_entity = DocumentEntity.objects.create(
        artifact=artifact,
        document_id=DOCUMENT_ID,
        label="Aquilla replacement",
        normalized_label="aquilla-replacement",
        entity_type="model",
    )
    replacement_link = DocumentEntityMention.objects.create(
        document_entity=replacement_entity,
        mention=mention,
    )

    replacement_link.delete()

    assert EntityMention.objects.filter(pk=mention.pk).exists()


@pytest.mark.django_db(transaction=True)
@database_required
def test_relation_evidence_preserves_each_unique_support_and_cascades_with_mention(
    django_assert_num_queries,
):
    collection_artifact = _artifact(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=COLLECTION_ID,
        status=GraphArtifact.Status.ACTIVE,
    )
    collection_artifact.save()
    document_artifact = _artifact(status=GraphArtifact.Status.ACTIVE)
    document_artifact.save()
    chunk = _chunk()
    head = _mention(document_artifact, chunk)
    tail = _mention(
        document_artifact,
        chunk,
        start=18,
        end=22,
        raw_text="MMLU",
        normalized_text="mmlu",
        entity_type="benchmark",
    )
    mention = RelationMention.objects.create(
        artifact=document_artifact,
        document_id=DOCUMENT_ID,
        chunk=chunk,
        head=head,
        tail=tail,
        relation_type="evaluates_on",
        extraction_confidence=0.8,
    )
    head_document_entity = DocumentEntity.objects.create(
        artifact=document_artifact,
        document_id=DOCUMENT_ID,
        label="Aquilla",
        normalized_label="aquilla",
        entity_type="model",
    )
    tail_document_entity = DocumentEntity.objects.create(
        artifact=document_artifact,
        document_id=DOCUMENT_ID,
        label="MMLU",
        normalized_label="mmlu",
        entity_type="benchmark",
    )
    DocumentEntityMention.objects.create(
        document_entity=head_document_entity,
        mention=head,
    )
    DocumentEntityMention.objects.create(
        document_entity=tail_document_entity,
        mention=tail,
    )
    source = CollectionEntity.objects.create(
        artifact=collection_artifact,
        collection_id=COLLECTION_ID,
        label="Aquilla",
        normalized_label="aquilla",
        entity_type="model",
    )
    target = CollectionEntity.objects.create(
        artifact=collection_artifact,
        collection_id=COLLECTION_ID,
        label="MMLU",
        normalized_label="mmlu",
        entity_type="benchmark",
    )
    relation = CollectionRelation.objects.create(
        artifact=collection_artifact,
        source=source,
        target=target,
        relation_type="evaluates_on",
        support_count=1,
        confidence=0.8,
    )
    head_mapping = CollectionEntityDocumentLink.objects.create(
        document_entity=head_document_entity,
        collection_entity=source,
        score=0.9,
        method="exact",
        resolver_version=collection_artifact.resolver_version,
    )
    CollectionEntityDocumentLink.objects.create(
        document_entity=tail_document_entity,
        collection_entity=target,
        score=0.9,
        method="exact",
        resolver_version=collection_artifact.resolver_version,
    )
    evidence = CollectionRelationEvidence(
        relation=relation,
        relation_mention=mention,
    )

    head_mapping.status = CollectionEntityDocumentLink.Status.SUPPRESSED
    head_mapping.save(update_fields=["status"])
    with pytest.raises(ValidationError, match="Head mention"):
        evidence.clean()
    head_mapping.status = CollectionEntityDocumentLink.Status.ACTIVE
    head_mapping.save(update_fields=["status"])
    with django_assert_num_queries(2):
        evidence.clean()
    evidence.save()

    with pytest.raises(IntegrityError), transaction.atomic():
        CollectionRelationEvidence.objects.create(
            relation=relation,
            relation_mention=mention,
        )

    mention.delete()

    assert not CollectionRelationEvidence.objects.filter(pk=evidence.pk).exists()
    assert CollectionRelation.objects.filter(pk=relation.pk).exists()
