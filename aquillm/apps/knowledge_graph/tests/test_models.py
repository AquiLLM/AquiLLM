from __future__ import annotations

import os
import socket
import uuid
from datetime import timedelta
from math import inf, nan
from types import SimpleNamespace

import pytest
from django.apps import apps
from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import CheckConstraint, UniqueConstraint
from django.db.models.deletion import RestrictedError
from django.utils import timezone
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


def _postgres_test_action(*, available: bool, required: bool) -> str:
    if available:
        return "run"
    return "fail" if required else "skip"


_POSTGRES_AVAILABLE = _database_is_reachable()
_POSTGRES_REQUIRED = os.environ.get(
    "KG_REQUIRE_POSTGRES_TESTS", ""
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@pytest.fixture(scope="session", autouse=True)
def _enforce_required_postgres():
    if (
        _postgres_test_action(
            available=_POSTGRES_AVAILABLE,
            required=_POSTGRES_REQUIRED,
        )
        == "fail"
    ):
        pytest.fail(
            "KG_REQUIRE_POSTGRES_TESTS=1 but configured PostgreSQL is unavailable"
        )


database_required = pytest.mark.skipif(
    _postgres_test_action(
        available=_POSTGRES_AVAILABLE,
        required=_POSTGRES_REQUIRED,
    )
    == "skip",
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


def test_graph_models_expose_validated_persistence_path():
    for model in (
        GraphArtifact,
        GraphBuildRun,
        EntityMention,
        DocumentEntity,
        CollectionEntity,
        CollectionRelationEvidence,
    ):
        assert callable(getattr(model(), "validate_for_persistence"))


def test_invalid_normal_save_create_and_bulk_create_fail_before_database_access():
    invalid = _artifact(status="not-a-status")
    with pytest.raises(ValidationError, match="status"):
        invalid.save()
    with pytest.raises(ValidationError, match="status"):
        GraphArtifact.objects.create(
            scope_type=GraphArtifact.ScopeType.DOCUMENT,
            scope_id=DOCUMENT_ID,
            status="not-a-status",
            source_hash="a" * 64,
            ontology_version="ontology-v1",
            extractor_version="extractor-v1",
            resolver_version="resolver-v1",
            filter_policy_version="filter-v1",
        )
    with pytest.raises(ValidationError, match="status"):
        GraphArtifact.objects.bulk_create([invalid])


def test_graph_artifact_bulk_mutation_rejects_build_identity_fields_before_database():
    artifact = _artifact(pk=1)

    with pytest.raises(ValidationError, match="immutable"):
        GraphArtifact.objects.filter(pk=1).update(source_hash="b" * 64)
    with pytest.raises(ValidationError, match="immutable"):
        GraphArtifact.objects.bulk_update([artifact], ["filter_policy_version"])


def test_graph_build_run_queryset_rejects_artifact_reassignment():
    run = GraphBuildRun(pk=1, artifact_id=1)

    with pytest.raises(ValidationError, match="immutable"):
        GraphBuildRun.objects.filter(pk=1).update(artifact_id=2)
    with pytest.raises(ValidationError, match="immutable"):
        GraphBuildRun.objects.bulk_update([run], ["artifact"])


@pytest.mark.django_db(transaction=True)
@database_required
def test_graph_artifact_bulk_update_allows_lifecycle_status_changes():
    artifact = _artifact()
    artifact.save()
    GraphArtifact.objects.filter(pk=artifact.pk).update(
        status=GraphArtifact.Status.FAILED
    )
    artifact.refresh_from_db()
    assert artifact.status == GraphArtifact.Status.FAILED


@pytest.mark.django_db(transaction=True)
@database_required
@pytest.mark.parametrize("model", [DocumentEntity, CollectionEntity])
def test_identifier_first_database_uniqueness(model):
    artifact = _artifact(
        scope_type=(
            GraphArtifact.ScopeType.DOCUMENT
            if model is DocumentEntity
            else GraphArtifact.ScopeType.COLLECTION
        ),
        scope_id=DOCUMENT_ID if model is DocumentEntity else COLLECTION_ID,
    )
    artifact.save()
    ownership = (
        {"document_id": DOCUMENT_ID}
        if model is DocumentEntity
        else {"collection_id": COLLECTION_ID}
    )
    common = {"artifact": artifact, "entity_type": "model", **ownership}
    model.objects.create(
        **common,
        label="First",
        normalized_label="first",
        identifier="stable-id",
    )
    with pytest.raises(ValidationError, match="identifier"):
        model.objects.create(
            **common,
            label="Different label",
            normalized_label="different-label",
            identifier="stable-id",
        )
    model.objects.create(
        **common,
        label="First",
        normalized_label="first",
        identifier="other-id",
    )


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


@pytest.mark.parametrize(
    ("scope_type", "status"),
    [
        (GraphArtifact.ScopeType.COLLECTION, GraphArtifact.Status.BUILDING),
        (GraphArtifact.ScopeType.DOCUMENT, GraphArtifact.Status.FAILED),
        (GraphArtifact.ScopeType.DOCUMENT, GraphArtifact.Status.STALE),
        (GraphArtifact.ScopeType.DOCUMENT, GraphArtifact.Status.SUPERSEDED),
    ],
)
def test_entity_mention_rejects_ineligible_source_artifact(scope_type, status):
    mention = EntityMention(
        artifact=_artifact(scope_type=scope_type, status=status),
        document_id=DOCUMENT_ID,
        chunk=TextChunk(modality=TextChunk.Modality.TEXT, doc_id=DOCUMENT_ID),
        start=0,
        end=7,
        position_basis=EntityMention.PositionBasis.DOCUMENT_GLOBAL,
        raw_text="Aquilla",
        normalized_text="aquilla",
        entity_type="model",
        extraction_confidence=0.9,
    )

    with pytest.raises(ValidationError, match="artifact"):
        mention.clean()


def test_image_entity_mention_requires_existing_exact_document_subtype(monkeypatch):
    expected_document = SimpleNamespace(
        id=DOCUMENT_ID,
        _meta=SimpleNamespace(
            app_label="apps_documents",
            model_name="documentfigure",
        ),
    )
    wrong_document = SimpleNamespace(
        id=DOCUMENT_ID,
        _meta=SimpleNamespace(
            app_label="apps_documents",
            model_name="imageuploaddocument",
        ),
    )
    content_type = ContentType(
        pk=1,
        app_label="apps_documents",
        model="documentfigure",
    )
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
        content_object_type=content_type,
        content_object_id=DOCUMENT_ID,
    )
    monkeypatch.setattr(
        mention,
        "_resolve_image_content_object",
        lambda: (wrong_document, expected_document),
        raising=False,
    )

    with pytest.raises(ValidationError, match="exact|subtype"):
        mention.clean()


def test_image_entity_mention_resolves_document_by_public_uuid(monkeypatch):
    expected_document = SimpleNamespace(id=DOCUMENT_ID)
    calls = {}
    content_type = ContentType(
        pk=1,
        app_label="apps_documents",
        model="documentfigure",
    )
    monkeypatch.setattr(
        content_type,
        "get_object_for_this_type",
        lambda **kwargs: calls.update(kwargs) or expected_document,
    )
    monkeypatch.setattr(
        TextChunk,
        "document",
        property(lambda _chunk: expected_document),
    )
    mention = EntityMention(
        chunk=TextChunk(doc_id=DOCUMENT_ID),
        content_object_type=content_type,
        content_object_id=DOCUMENT_ID,
    )

    target, document = mention._resolve_image_content_object()

    assert target is document is expected_document
    assert calls == {"id": DOCUMENT_ID}


@pytest.mark.parametrize("invalid", [True, False, nan, inf, -inf])
@pytest.mark.parametrize(
    "factory",
    [
        lambda value: EntityMention(extraction_confidence=value),
        lambda value: RelationMention(extraction_confidence=value),
        lambda value: CollectionRelation(confidence=value),
        lambda value: CollectionEntityDocumentLink(score=value),
        lambda value: CanonicalEntityLink(score=value),
    ],
)
def test_confidence_and_scores_reject_bool_and_nonfinite_values(factory, invalid):
    instance = factory(invalid)

    with pytest.raises(ValidationError, match="confidence|score"):
        instance.validate_for_persistence()


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
    for field_name in ("head_mapping", "tail_mapping"):
        mapping_field = CollectionRelationEvidence._meta.get_field(field_name)
        assert mapping_field.null is False
        assert mapping_field.remote_field.model is CollectionEntityDocumentLink
        assert mapping_field.remote_field.on_delete.__name__ == "RESTRICT"
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


def test_identifier_first_conditional_uniqueness_and_normalization():
    assert (
        _constraint(
            DocumentEntity,
            "kg_document_entity_identifier_unique",
            UniqueConstraint,
        ).condition
        is not None
    )
    assert (
        _constraint(
            DocumentEntity,
            "kg_document_entity_cluster_unique",
            UniqueConstraint,
        ).condition
        is None
    )
    assert (
        _constraint(
            CollectionEntity,
            "kg_collection_entity_identifier_unique",
            UniqueConstraint,
        ).condition
        is not None
    )
    assert (
        _constraint(
            CollectionEntity,
            "kg_collection_entity_label_fallback",
            UniqueConstraint,
        ).condition
        is not None
    )

    entity = DocumentEntity(
        artifact=_artifact(),
        document_id=DOCUMENT_ID,
        cluster_key="1" * 64,
        label="Aquilla",
        normalized_label="aquilla",
        entity_type="model",
        identifier=" model:1 ",
    )
    entity.clean()
    assert entity.identifier == "model:1"

    entity.identifier = "   "
    with pytest.raises(ValidationError, match="identifier"):
        entity.clean()


@pytest.mark.parametrize(
    ("model", "field_name"),
    [
        (EntityMention, "normalized_text"),
        (DocumentEntity, "normalized_label"),
        (CollectionEntity, "normalized_label"),
        (CanonicalEntity, "normalized_label"),
    ],
)
def test_indexed_normalized_values_reject_oversize_input(model, field_name):
    field = model._meta.get_field(field_name)
    assert field.max_length == 512
    with pytest.raises(ValidationError, match="512"):
        field.clean("x" * 513, model())


def test_central_choice_and_build_identity_db_constraints_are_declared():
    expected = {
        GraphArtifact: {
            "kg_artifact_scope_valid",
            "kg_artifact_status_valid",
            "kg_artifact_source_hash_nonempty",
            "kg_artifact_ontology_ver_nonempty",
            "kg_artifact_extractor_ver_nonempty",
            "kg_artifact_resolver_ver_nonempty",
            "kg_artifact_filter_ver_nonempty",
        },
        GraphBuildRun: {
            "kg_build_kind_valid",
            "kg_build_scope_valid",
            "kg_build_stage_valid",
            "kg_build_status_valid",
            "kg_build_snapshot_nonempty",
        },
        OntologyVersion: {"kg_ontology_kind_valid", "kg_ontology_status_valid"},
        EntityMention: {"kg_mention_position_basis_valid"},
        DocumentEntity: {
            "kg_document_entity_status_valid",
            "kg_document_cluster_key_valid",
        },
        DocumentEntityMention: {
            "kg_document_mention_status_valid",
            "kg_document_mention_method_valid",
            "kg_document_mention_resolver_nonempty",
        },
        CollectionEntity: {"kg_collection_entity_status_valid"},
        CanonicalEntity: {"kg_canonical_entity_status_valid"},
        CollectionEntityDocumentLink: {"kg_doc_collection_link_status_valid"},
        CanonicalEntityLink: {"kg_canonical_link_status_valid"},
        CollectionRelation: {"kg_collection_relation_status_valid"},
    }
    for model, names in expected.items():
        actual = {constraint.name for constraint in model._meta.constraints}
        assert names <= actual


def test_graph_build_run_populates_and_freezes_artifact_identity_snapshot(monkeypatch):
    artifact = _artifact(pk=10)
    run = GraphBuildRun(artifact=artifact)
    monkeypatch.setattr(GraphArtifact.objects, "get", lambda **_kwargs: artifact)

    run.populate_artifact_snapshot()

    assert run.build_kind == GraphBuildRun.BuildKind.DOCUMENT
    assert run.scope_type == artifact.scope_type
    assert run.scope_id == artifact.scope_id
    assert run.source_hash == artifact.source_hash
    assert run.filter_policy_version == artifact.filter_policy_version
    assert "scope_id" in run._IMMUTABLE_FIELDS
    assert "filter_policy_version" in run._IMMUTABLE_FIELDS


def test_graph_build_run_snapshot_ignores_mutated_cached_artifact(monkeypatch):
    cached = _artifact(pk=10, source_hash="c" * 64)
    fresh = _artifact(pk=10, source_hash="f" * 64)
    run = GraphBuildRun(artifact=cached)
    monkeypatch.setattr(GraphArtifact.objects, "get", lambda **_kwargs: fresh)

    run.populate_artifact_snapshot()

    assert run.source_hash == fresh.source_hash
    assert run.source_hash != cached.source_hash


@pytest.mark.parametrize("model", [EntityMention, RelationMention])
def test_raw_evidence_querysets_reject_update_and_bulk_update(model):
    instance = model(pk=1, created_at=timezone.now())
    field = "raw_text" if model is EntityMention else "relation_type"

    with pytest.raises(ValidationError, match="immutable"):
        model.objects.filter(pk=1).update(**{field: "rewritten"})
    with pytest.raises(ValidationError, match="immutable"):
        model.objects.bulk_update([instance], [field])
    with pytest.raises(ValidationError, match="immutable"):
        model.objects.filter(pk=1).update(created_at=timezone.now())
    with pytest.raises(ValidationError, match="immutable"):
        model.objects.bulk_update([instance], ["created_at"])


def test_raw_evidence_declares_complete_immutable_fields_and_aliases():
    assert {
        "artifact",
        "artifact_id",
        "document_id",
        "chunk",
        "chunk_id",
        "start",
        "end",
        "position_basis",
        "raw_text",
        "normalized_text",
        "entity_type",
        "extraction_confidence",
        "content_object_type",
        "content_object_type_id",
        "content_object_id",
        "metadata",
        "created_at",
    } <= set(EntityMention._IMMUTABLE_FIELDS)
    assert {
        "artifact",
        "artifact_id",
        "document_id",
        "chunk",
        "chunk_id",
        "head",
        "head_id",
        "tail",
        "tail_id",
        "relation_type",
        "extraction_confidence",
        "metadata",
        "created_at",
    } <= set(RelationMention._IMMUTABLE_FIELDS)


@pytest.mark.parametrize("model", [DocumentEntity, CollectionEntity])
def test_resolved_identity_querysets_reject_identity_rewrites(model):
    instance = model(pk=1)

    with pytest.raises(ValidationError, match="immutable"):
        model.objects.filter(pk=1).update(identifier="replacement")
    with pytest.raises(ValidationError, match="immutable"):
        model.objects.bulk_update([instance], ["normalized_label"])


@pytest.mark.parametrize(
    ("model", "scope_field"),
    [
        (DocumentEntity, "document_id"),
        (CollectionEntity, "collection_id"),
    ],
)
def test_resolved_entity_querysets_reject_cross_scope_ownership_rewrites(
    model,
    scope_field,
):
    instance = model(pk=1, artifact_id=2)

    with pytest.raises(ValidationError, match="immutable"):
        model.objects.filter(pk=1).update(**{scope_field: uuid.uuid4()})
    with pytest.raises(ValidationError, match="immutable"):
        model.objects.bulk_update([instance], ["artifact"])

    immutable = set(model._QUERYSET_IMMUTABLE_FIELDS)
    assert {"artifact", "artifact_id", scope_field} <= immutable


def test_postgres_test_gate_can_be_made_required():
    assert _postgres_test_action(available=False, required=False) == "skip"
    assert _postgres_test_action(available=False, required=True) == "fail"
    assert _postgres_test_action(available=True, required=True) == "run"


def _unsaved_relation_evidence():
    collection_artifact = _artifact(
        pk=1,
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=COLLECTION_ID,
    )
    document_artifact = _artifact(
        pk=2,
        source_hash="b" * 64,
        status=GraphArtifact.Status.ACTIVE,
    )
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
    head_document_entity = DocumentEntity(
        pk=50,
        artifact=document_artifact,
        document_id=DOCUMENT_ID,
        cluster_key="1" * 64,
        label="Aquilla",
        normalized_label="aquilla",
        entity_type="model",
    )
    tail_document_entity = DocumentEntity(
        pk=51,
        artifact=document_artifact,
        document_id=DOCUMENT_ID,
        cluster_key="2" * 64,
        label="MMLU",
        normalized_label="mmlu",
        entity_type="benchmark",
    )
    head_mapping = CollectionEntityDocumentLink(
        pk=60,
        document_entity=head_document_entity,
        collection_entity=source,
        score=0.9,
        method="exact",
        resolver_version=collection_artifact.resolver_version,
    )
    tail_mapping = CollectionEntityDocumentLink(
        pk=61,
        document_entity=tail_document_entity,
        collection_entity=target,
        score=0.9,
        method="exact",
        resolver_version=collection_artifact.resolver_version,
    )
    return CollectionRelationEvidence(
        relation=relation,
        relation_mention=relation_mention,
        head_mapping=head_mapping,
        tail_mapping=tail_mapping,
    )


def test_relation_evidence_rejects_swapped_endpoint_mappings(monkeypatch):
    evidence = _unsaved_relation_evidence()
    evidence.head_mapping, evidence.tail_mapping = (
        evidence.tail_mapping,
        evidence.head_mapping,
    )
    monkeypatch.setattr(
        evidence,
        "_endpoint_membership_is_active",
        lambda _mapping, _mention: True,
        raising=False,
    )

    with pytest.raises(ValidationError, match="head|tail|mapped"):
        evidence.clean()


def test_relation_evidence_rejects_unmapped_endpoint(monkeypatch):
    evidence = _unsaved_relation_evidence()
    monkeypatch.setattr(
        evidence,
        "_endpoint_membership_is_active",
        lambda _mapping, _mention: False,
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
        "_endpoint_membership_is_active",
        lambda _mapping, _mention: True,
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
        "_endpoint_membership_is_active",
        lambda _mapping, _mention: True,
        raising=False,
    )

    evidence.clean()


@pytest.mark.django_db(transaction=True)
@database_required
def test_database_enforces_one_active_artifact_and_version_identity():
    first = _artifact(status=GraphArtifact.Status.ACTIVE)
    first.save()

    with pytest.raises(ValidationError), transaction.atomic():
        _artifact(status=GraphArtifact.Status.ACTIVE, source_hash="b" * 64).save()
    with pytest.raises(ValidationError), transaction.atomic():
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
        cluster_key="1" * 64,
        label="Aquilla",
        normalized_label="aquilla",
        entity_type="model",
    )
    mention_link = DocumentEntityMention.objects.create(
        document_entity=document_entity,
        mention=head,
        method=DocumentEntityMention.Method.SINGLETON,
        resolver_version=artifact.resolver_version,
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
        cluster_key="1" * 64,
        label="Aquilla",
        normalized_label="aquilla",
        entity_type="model",
    )
    link = DocumentEntityMention.objects.create(
        document_entity=document_entity,
        mention=mention,
        method=DocumentEntityMention.Method.SINGLETON,
        resolver_version=artifact.resolver_version,
    )

    document_entity.delete()

    assert EntityMention.objects.filter(pk=mention.pk).exists()
    assert not DocumentEntityMention.objects.filter(pk=link.pk).exists()

    replacement_entity = DocumentEntity.objects.create(
        artifact=artifact,
        document_id=DOCUMENT_ID,
        cluster_key="2" * 64,
        label="Aquilla replacement",
        normalized_label="aquilla-replacement",
        entity_type="model",
    )
    replacement_link = DocumentEntityMention.objects.create(
        document_entity=replacement_entity,
        mention=mention,
        method=DocumentEntityMention.Method.SINGLETON,
        resolver_version=artifact.resolver_version,
    )

    replacement_link.delete()

    assert EntityMention.objects.filter(pk=mention.pk).exists()


@pytest.mark.django_db(transaction=True)
@database_required
def test_raw_evidence_instance_save_rejects_rewrites():
    artifact = _artifact(status=GraphArtifact.Status.ACTIVE)
    artifact.save()
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
    relation = RelationMention.objects.create(
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=chunk,
        head=head,
        tail=tail,
        relation_type="evaluates_on",
        extraction_confidence=0.8,
    )

    head.raw_text = "rewritten"
    with pytest.raises(ValidationError, match="immutable"):
        head.save()
    head.refresh_from_db()
    head.created_at += timedelta(seconds=1)
    with pytest.raises(ValidationError, match="immutable"):
        head.save()
    relation.relation_type = "rewritten"
    with pytest.raises(ValidationError, match="immutable"):
        relation.save()
    relation.refresh_from_db()
    relation.created_at += timedelta(seconds=1)
    with pytest.raises(ValidationError, match="immutable"):
        relation.save()


@pytest.mark.django_db(transaction=True)
@database_required
def test_graph_build_run_uses_fresh_snapshot_and_artifact_delete_sets_null():
    artifact = _artifact()
    artifact.save()
    original_hash = artifact.source_hash
    artifact.source_hash = "c" * 64

    run = GraphBuildRun.objects.create(artifact=artifact)

    assert run.source_hash == original_hash
    artifact_id = artifact.pk
    GraphArtifact.objects.get(pk=artifact_id).delete()
    run.refresh_from_db()
    assert run.artifact_id is None
    assert run.source_hash == original_hash


@pytest.mark.django_db(transaction=True)
@database_required
def test_collection_artifact_delete_collects_restricted_mappings_and_evidence():
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
    relation_mention = RelationMention.objects.create(
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
        cluster_key="1" * 64,
        label="Aquilla",
        normalized_label="aquilla",
        entity_type="model",
    )
    tail_document_entity = DocumentEntity.objects.create(
        artifact=document_artifact,
        document_id=DOCUMENT_ID,
        cluster_key="2" * 64,
        label="MMLU",
        normalized_label="mmlu",
        entity_type="benchmark",
    )
    DocumentEntityMention.objects.create(
        document_entity=head_document_entity,
        mention=head,
        method=DocumentEntityMention.Method.SINGLETON,
        resolver_version=document_artifact.resolver_version,
    )
    DocumentEntityMention.objects.create(
        document_entity=tail_document_entity,
        mention=tail,
        method=DocumentEntityMention.Method.SINGLETON,
        resolver_version=document_artifact.resolver_version,
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
    tail_mapping = CollectionEntityDocumentLink.objects.create(
        document_entity=tail_document_entity,
        collection_entity=target,
        score=0.9,
        method="exact",
        resolver_version=collection_artifact.resolver_version,
    )
    evidence = CollectionRelationEvidence.objects.create(
        relation=relation,
        relation_mention=relation_mention,
        head_mapping=head_mapping,
        tail_mapping=tail_mapping,
    )
    mapping_ids = (head_mapping.pk, tail_mapping.pk)

    with pytest.raises(RestrictedError):
        head_mapping.delete()
    collection_artifact.delete()

    assert not CollectionRelationEvidence.objects.filter(pk=evidence.pk).exists()
    assert not CollectionEntityDocumentLink.objects.filter(pk__in=mapping_ids).exists()
    assert not CollectionRelation.objects.filter(pk=relation.pk).exists()
    assert RelationMention.objects.filter(pk=relation_mention.pk).exists()


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
        cluster_key="1" * 64,
        label="Aquilla",
        normalized_label="aquilla",
        entity_type="model",
    )
    tail_document_entity = DocumentEntity.objects.create(
        artifact=document_artifact,
        document_id=DOCUMENT_ID,
        cluster_key="2" * 64,
        label="MMLU",
        normalized_label="mmlu",
        entity_type="benchmark",
    )
    DocumentEntityMention.objects.create(
        document_entity=head_document_entity,
        mention=head,
        method=DocumentEntityMention.Method.SINGLETON,
        resolver_version=document_artifact.resolver_version,
    )
    DocumentEntityMention.objects.create(
        document_entity=tail_document_entity,
        mention=tail,
        method=DocumentEntityMention.Method.SINGLETON,
        resolver_version=document_artifact.resolver_version,
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
    tail_mapping = CollectionEntityDocumentLink.objects.create(
        document_entity=tail_document_entity,
        collection_entity=target,
        score=0.9,
        method="exact",
        resolver_version=collection_artifact.resolver_version,
    )
    evidence = CollectionRelationEvidence(
        relation=relation,
        relation_mention=mention,
        head_mapping=head_mapping,
        tail_mapping=tail_mapping,
    )

    head_mapping.status = CollectionEntityDocumentLink.Status.SUPPRESSED
    head_mapping.save(update_fields=["status"])
    with pytest.raises(ValidationError, match="active"):
        evidence.clean()
    head_mapping.status = CollectionEntityDocumentLink.Status.ACTIVE
    head_mapping.save(update_fields=["status"])
    with django_assert_num_queries(2):
        evidence.clean()
    evidence.save()

    with pytest.raises(ValidationError), transaction.atomic():
        CollectionRelationEvidence.objects.create(
            relation=relation,
            relation_mention=mention,
            head_mapping=head_mapping,
            tail_mapping=tail_mapping,
        )

    mention.delete()

    assert not CollectionRelationEvidence.objects.filter(pk=evidence.pk).exists()
    assert CollectionRelation.objects.filter(pk=relation.pk).exists()
