from __future__ import annotations

import uuid
from dataclasses import replace

import pytest
from django.apps import apps
from django.core.exceptions import ValidationError
from django.db.models import CheckConstraint, UniqueConstraint
from test_collection_resolution import (
    _EMBEDDING_SIGNATURE,
    _document_entity,
    _ontology,
    _session,
    _snapshot,
)

from apps.knowledge_graph.resolution.collection import (
    resolve_collection_entities,
)


def test_kg_schema_uses_real_collection_fk_typed_scores_and_manifest():
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import (
        CollectionEntity,
        CollectionEntityDocumentLink,
        DocumentEntity,
        GraphArtifact,
        GraphBuildRun,
    )

    collection_input = apps.get_model("apps_knowledge_graph", "CollectionArtifactInput")
    assert GraphArtifact._meta.get_field("scope_id").get_internal_type() == "CharField"
    assert GraphBuildRun._meta.get_field("scope_id").get_internal_type() == "CharField"
    for model in (GraphArtifact, GraphBuildRun):
        signature = model._meta.get_field("embedding_model_signature")
        assert signature.blank is True
        for checksum_field in (
            "ontology_checksum",
            "filter_policy_checksum",
            "resolution_config_checksum",
        ):
            field = model._meta.get_field(checksum_field)
            assert field.max_length == 64
            assert field.editable is False
    assert (
        CollectionEntity._meta.get_field("collection").remote_field.model is Collection
    )
    assert (
        collection_input._meta.get_field("collection").remote_field.model is Collection
    )
    assert (
        collection_input._meta.get_field("document_artifact").remote_field.model
        is GraphArtifact
    )
    assert collection_input._meta.get_field("membership_signature").max_length == 64
    assert (
        CollectionEntityDocumentLink._meta.get_field("artifact").remote_field.model
        is GraphArtifact
    )
    assert (
        CollectionEntityDocumentLink._meta.get_field(
            "manifest_input"
        ).remote_field.model
        is collection_input
    )
    assert DocumentEntity._meta.get_field("resolution_confidence").null is False
    for name in (
        "cluster_key",
        "version_signature",
        "extraction_confidence",
        "resolution_confidence",
        "retrieval_utility",
        "promotion_confidence",
        "filter_reason",
        "embedding_model_signature",
        "embedding_input_hash",
    ):
        assert CollectionEntity._meta.get_field(name) is not None


def test_scope_identity_and_embedding_signature_have_conditional_db_checks():
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    artifact_constraint_names = {
        constraint.name
        for constraint in GraphArtifact._meta.constraints
        if isinstance(constraint, CheckConstraint)
    }
    run_constraint_names = {
        constraint.name
        for constraint in GraphBuildRun._meta.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert "kg_artifact_typed_scope_id" in artifact_constraint_names
    assert "kg_artifact_embedding_signature_scope" in artifact_constraint_names
    assert "kg_run_typed_scope_id" in run_constraint_names
    assert "kg_run_embedding_signature_scope" in run_constraint_names


def test_polymorphic_scope_ids_canonicalize_document_uuid_and_collection_pk():
    from apps.knowledge_graph.models import GraphArtifact

    common = {
        "status": GraphArtifact.Status.BUILDING,
        "source_hash": "a" * 64,
        "ontology_version": "ontology-v1",
        "extractor_version": "extractor-v1",
        "resolver_version": "resolver-v1",
        "filter_policy_version": "filter-v1",
    }
    document_id = uuid.UUID("11111111-1111-4111-8111-111111111111")
    document = GraphArtifact(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=document_id,
        embedding_model_signature="",
        **common,
    )
    collection = GraphArtifact(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=7,
        embedding_model_signature=_EMBEDDING_SIGNATURE,
        **common,
    )

    document.prepare_for_persistence()
    collection.prepare_for_persistence()

    assert document.scope_id == str(document_id)
    assert collection.scope_id == "7"
    document.clean()
    collection.clean()


@pytest.mark.parametrize(
    "scope_type, scope_id, signature",
    [
        ("document", "not-a-uuid", ""),
        (
            "collection",
            "007",
            "local:model@rev:dims=1024:prep=kg-entity-v1:max_chars=8192:batch=64",
        ),
        (
            "collection",
            "0",
            "local:model@rev:dims=1024:prep=kg-entity-v1:max_chars=8192:batch=64",
        ),
        ("document", str(uuid.uuid4()), "must-be-empty"),
        (
            "collection",
            "7",
            "local:model@rev:dims=1024:prep=kg-entity-v1:max_chars=8192:batch=64",
        ),
        ("collection", "7", ""),
    ],
)
def test_invalid_typed_scope_or_embedding_signature_is_rejected(
    scope_type, scope_id, signature
):
    from apps.knowledge_graph.models import GraphArtifact

    artifact = GraphArtifact(
        scope_type=scope_type,
        scope_id=scope_id,
        status=GraphArtifact.Status.BUILDING,
        source_hash="a" * 64,
        ontology_version=_ontology().version,
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="filter-v1",
        embedding_model_signature=signature,
    )

    with pytest.raises(ValidationError, match="scope|embedding"):
        artifact.clean()


def test_document_link_has_explicit_outcomes_component_scores_and_auto_uniqueness():
    from apps.knowledge_graph.models import CollectionEntityDocumentLink

    fields = {field.name for field in CollectionEntityDocumentLink._meta.fields}
    assert {
        "artifact",
        "manifest_input",
        "outcome",
        "identifier_score",
        "alias_score",
        "embedding_similarity",
        "neighborhood_agreement",
        "candidate_rank",
        "decision_checksum",
    } <= fields
    assert any(
        isinstance(constraint, UniqueConstraint)
        and constraint.name == "kg_one_auto_collection_assignment"
        and constraint.condition is not None
        for constraint in CollectionEntityDocumentLink._meta.constraints
    )


def test_collection_cluster_key_is_stable_across_database_ids_and_rebuild_artifacts():
    first_entity = _document_entity(
        10,
        "Atlas",
        document_cluster_key="c" * 64,
    )
    second_entity = _document_entity(
        20,
        "Atlas",
        document_cluster_key="c" * 64,
    )
    first_session, _ = _session({})
    second_session, _ = _session({})

    first = resolve_collection_entities(
        _snapshot(),
        (first_entity,),
        _ontology(),
        embedding_session=first_session,
    )
    second = resolve_collection_entities(
        replace(_snapshot(), destination_artifact_id=102),
        (second_entity,),
        _ontology(),
        embedding_session=second_session,
    )

    assert first.clusters[0].cluster_key == second.clusters[0].cluster_key
