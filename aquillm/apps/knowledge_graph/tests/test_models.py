from __future__ import annotations

import importlib
import inspect
import os
import socket
import uuid
from datetime import timedelta
from io import StringIO
from math import inf, nan
from types import SimpleNamespace

import pytest
from django.apps import apps
from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError
from django.core.management import call_command
from django.db import migrations, models, transaction
from django.db.models import CheckConstraint, UniqueConstraint
from django.db.models.deletion import ProtectedError
from django.utils import timezone
from pgvector.django import VectorField

from apps.documents.models import DESCENDED_FROM_DOCUMENT, TextChunk
from apps.knowledge_graph.models import (
    CanonicalEntity,
    CanonicalEntityLink,
    CollectionArtifactInput,
    CollectionEntity,
    CollectionEntityDocumentLink,
    CollectionRelation,
    CollectionRelationEvidence,
    DocumentEntity,
    DocumentEntityMention,
    EntityMention,
    GraphArtifact,
    GraphBuildRun,
    GraphRebuildRequest,
    OntologyVersion,
    RelationMention,
)

DOCUMENT_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
COLLECTION_ID = 22
COLLECTION_EMBEDDING_SIGNATURE = (
    f"test-local:model@rev:endpoint={'e' * 64}:dims=1024:"
    "prep=kg-entity-v1:max_chars=8192:batch=64"
)


def _request_snapshot(
    *,
    document_id: uuid.UUID = DOCUMENT_ID,
    collection_id: int = COLLECTION_ID,
    source_hash: str = "a" * 64,
    model_label: str | None = None,
) -> dict[str, object]:
    return {
        "document_id": str(document_id),
        "document_pkid": 1,
        "model_label": model_label or DESCENDED_FROM_DOCUMENT[0]._meta.label_lower,
        "collection_id": collection_id,
        "source_hash": source_hash,
    }


def _rebuild_request(**overrides) -> GraphRebuildRequest:
    values = {
        "id": uuid.uuid4(),
        "scope_type": GraphRebuildRequest.ScopeType.DOCUMENT,
        "scope_id": str(DOCUMENT_ID),
        "requested_documents": [_request_snapshot()],
        "document_count": 1,
        "status": GraphRebuildRequest.Status.RUNNING,
    }
    values.update(overrides)
    values.setdefault(
        "collection_count",
        1 if values["scope_type"] == GraphRebuildRequest.ScopeType.COLLECTION else 0,
    )
    return GraphRebuildRequest(**values)


def test_rebuild_request_snapshot_requires_exact_scope_and_concrete_model() -> None:
    wrong_document = _rebuild_request(scope_id=str(uuid.uuid4()))
    with pytest.raises(ValidationError, match="match request scope"):
        wrong_document.prepare_for_persistence()

    wrong_collection = _rebuild_request(
        scope_type=GraphRebuildRequest.ScopeType.COLLECTION,
        scope_id=str(COLLECTION_ID + 1),
    )
    with pytest.raises(ValidationError, match="match collection scope"):
        wrong_collection.prepare_for_persistence()

    unsupported = _rebuild_request(
        requested_documents=[_request_snapshot(model_label="apps_documents.document")]
    )
    with pytest.raises(ValidationError, match="scalar is invalid"):
        unsupported.prepare_for_persistence()


def test_rebuild_request_outcomes_and_error_code_are_private_and_bounded() -> None:
    request = _rebuild_request(
        completed_document_count=1,
        terminal_failure_count=1,
    )
    with pytest.raises(ValidationError, match="outcomes exceed"):
        request.prepare_for_persistence()

    request = _rebuild_request(error_code="forged\nprivate")
    with pytest.raises(ValidationError, match="private identifier"):
        request.prepare_for_persistence()


def test_rebuild_request_queryset_rejects_lifecycle_and_snapshot_rewrites() -> None:
    for update in (
        {"scope_id": str(uuid.uuid4())},
        {"status": GraphRebuildRequest.Status.SUCCEEDED},
        {"terminal_failure_count": 1},
    ):
        with pytest.raises(ValidationError, match="immutable"):
            GraphRebuildRequest.objects.all().update(**update)


@pytest.mark.django_db
def test_terminal_resnapshot_marker_only_resolves_to_bounded_final_codes() -> None:
    request = _rebuild_request(
        status=GraphRebuildRequest.Status.PARTIAL,
        error_code="resnapshot_pending",
        completed_at=timezone.now(),
    )
    request.save()

    request.error_code = "scope_deleted"
    request.save(update_fields=["error_code", "updated_at"])

    request.refresh_from_db()
    request.error_code = "forged_terminal_rewrite"
    with pytest.raises(ValidationError, match="Terminal rebuild state is immutable"):
        request.save(update_fields=["error_code", "updated_at"])


def test_rebuild_occurrences_require_exact_request_marker_and_scope() -> None:
    request = _rebuild_request()
    artifact = _artifact(
        rebuild_request=request,
        evaluation_only=False,
    )
    artifact.clean()

    artifact.scope_id = str(uuid.uuid4())
    with pytest.raises(ValidationError, match="outside its request"):
        artifact.clean()

    evaluation = _rebuild_request(
        scope_type=GraphRebuildRequest.ScopeType.COLLECTION,
        scope_id=str(COLLECTION_ID),
        requested_documents=[_request_snapshot()],
        evaluation_only=True,
        expected_aggregate_signature="b" * 64,
    )
    eval_artifact = _artifact(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=str(COLLECTION_ID),
        source_hash="b" * 64,
        status=GraphArtifact.Status.ACTIVE,
        rebuild_request=evaluation,
        evaluation_only=True,
    )
    with pytest.raises(ValidationError, match="cannot become current"):
        eval_artifact.clean()


def test_task18_request_constraints_cover_eval_and_exact_occurrences() -> None:
    artifact_constraints = {item.name for item in GraphArtifact._meta.constraints}
    run_constraints = {item.name for item in GraphBuildRun._meta.constraints}
    request_constraints = {item.name for item in GraphRebuildRequest._meta.constraints}

    assert {
        "kg_artifact_eval_noncurrent",
        "kg_artifact_request_scope_unique",
    } <= artifact_constraints
    assert {"kg_build_eval_noncurrent", "kg_run_request_scope_unique"} <= (
        run_constraints
    )
    assert {
        "kg_rebuild_outcomes_bounded",
        "kg_rebuild_scoped_success_artifact",
        "kg_rebuild_error_code_safe",
    } <= request_constraints


def test_task18_migrations_separate_atomic_schema_from_retryable_live_changes() -> None:
    schema_migration = importlib.import_module(
        "apps.knowledge_graph.migrations.0005_graph_rebuild_request"
    )
    live_migration = importlib.import_module(
        "apps.knowledge_graph.migrations.0006_graph_rebuild_live_indexes"
    )

    assert schema_migration.Migration.atomic is True
    assert live_migration.Migration.atomic is False
    assert live_migration.Migration.dependencies == [
        ("apps_knowledge_graph", "0005_graph_rebuild_request")
    ]
    assert all(
        isinstance(operation, migrations.SeparateDatabaseAndState)
        for operation in live_migration.Migration.operations
    )

    live_request_fields = []
    for operation in schema_migration.Migration.operations:
        if not isinstance(operation, migrations.SeparateDatabaseAndState):
            continue
        for database_operation in operation.database_operations:
            if (
                isinstance(database_operation, migrations.AddField)
                and database_operation.name == "rebuild_request"
            ):
                live_request_fields.append(database_operation.field)
    assert len(live_request_fields) == 2
    assert all(
        not isinstance(field, models.ForeignKey) and field.db_index is False
        for field in live_request_fields
    )

    install_source = inspect.getsource(live_migration._install_check_constraint)
    foreign_key_source = inspect.getsource(live_migration._install_foreign_key)
    index_source = inspect.getsource(live_migration._create_index)
    assert {
        "kg_art_rebuild_req_idx",
        "kg_run_rebuild_req_idx",
        "kg_art_terminal_idx",
        "kg_art_superseded_idx",
        "kg_run_terminal_idx",
        "kg_run_scope_gen_idx",
        "kg_artifact_request_scope_unique",
        "kg_run_request_scope_unique",
    } == {row[1] for row in live_migration._LIVE_INDEXES}
    assert "NOT VALID" in install_source
    assert "VALIDATE CONSTRAINT" in install_source
    assert "FOREIGN KEY" in foreign_key_source
    assert "NOT VALID" in foreign_key_source
    assert "VALIDATE CONSTRAINT" in foreign_key_source
    assert "indisvalid" in index_source
    assert "CONCURRENTLY" in index_source


def test_manifest_source_status_is_owner_and_evaluation_partition_aware():
    from apps.knowledge_graph.models.inputs import _manifest_source_is_eligible

    production_request = uuid.uuid4()
    evaluation_request = uuid.uuid4()
    production_build = SimpleNamespace(
        status=GraphArtifact.Status.BUILDING,
        evaluation_only=False,
        rebuild_request_id=production_request,
    )
    production_active = SimpleNamespace(
        status=GraphArtifact.Status.ACTIVE,
        evaluation_only=False,
        rebuild_request_id=production_request,
    )
    evaluation_build = SimpleNamespace(
        status=GraphArtifact.Status.BUILDING,
        evaluation_only=True,
        rebuild_request_id=evaluation_request,
    )
    active_source = SimpleNamespace(
        status=GraphArtifact.Status.ACTIVE,
        evaluation_only=False,
        rebuild_request_id=production_request,
    )
    superseded_source = SimpleNamespace(
        status=GraphArtifact.Status.SUPERSEDED,
        evaluation_only=False,
        rebuild_request_id=production_request,
    )
    evaluation_source = SimpleNamespace(
        status=GraphArtifact.Status.SUPERSEDED,
        evaluation_only=True,
        rebuild_request_id=evaluation_request,
    )

    assert _manifest_source_is_eligible(production_build, active_source)
    assert not _manifest_source_is_eligible(production_build, superseded_source)
    assert _manifest_source_is_eligible(production_active, active_source)
    assert _manifest_source_is_eligible(production_active, superseded_source)
    assert not _manifest_source_is_eligible(production_active, evaluation_source)
    assert _manifest_source_is_eligible(evaluation_build, evaluation_source)
    evaluation_source.rebuild_request_id = uuid.uuid4()
    assert not _manifest_source_is_eligible(evaluation_build, evaluation_source)


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


@pytest.mark.django_db(transaction=True)
@database_required
def test_task18_live_table_sql_is_deferred_to_online_migration() -> None:
    schema_output = StringIO()
    call_command(
        "sqlmigrate",
        "apps_knowledge_graph",
        "0005",
        stdout=schema_output,
    )
    schema_statements = tuple(
        statement.strip() for statement in schema_output.getvalue().split(";")
    )
    for table_name in (
        "apps_knowledge_graph_graphartifact",
        "apps_knowledge_graph_graphbuildrun",
    ):
        live_statements = tuple(
            statement for statement in schema_statements if table_name in statement
        )
        assert any(
            'ADD COLUMN "rebuild_request_id" uuid NULL' in statement
            for statement in live_statements
        )
        assert not any(
            "FOREIGN KEY" in statement and "rebuild_request_id" in statement
            for statement in live_statements
        )
        assert not any(
            "CREATE INDEX" in statement and "rebuild_request" in statement
            for statement in live_statements
        )


@pytest.mark.django_db(transaction=True)
@database_required
def test_success_activation_audit_survives_document_artifact_deletion() -> None:
    from apps.knowledge_graph.models.artifacts import _activation_audit_values

    completed_at = timezone.now() - timedelta(days=31)
    request = _rebuild_request()
    request.save()
    artifact = _artifact(
        rebuild_request=request,
        build_key="b" * 64,
        build_generation=1,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        status=GraphArtifact.Status.SUPERSEDED,
        activated_at=completed_at,
        completed_at=completed_at,
        superseded_at=completed_at,
    )
    artifact.save()
    run = GraphBuildRun.objects.create(
        artifact=artifact,
        rebuild_request=request,
        stage=GraphBuildRun.Stage.SUPERSEDED,
        status=GraphBuildRun.Status.CANCELLED,
        attempt=1,
        stage_marker={"stage_sequence": ["active", "superseded"]},
        finished_at=completed_at,
    )
    request.status = GraphRebuildRequest.Status.SUCCEEDED
    request.completed_document_count = 1
    for field, value in _activation_audit_values(artifact, run).items():
        setattr(request, field, value)
    request.completed_at = completed_at
    request.save()
    expected_audit = (
        artifact.pk,
        run.pk,
        artifact.build_key,
        artifact.build_generation,
        artifact.source_hash,
        request.activated_occurrence_signature,
    )

    GraphArtifact.objects.filter(pk=artifact.pk).delete()

    request.refresh_from_db()
    assert (
        request.activated_artifact_pk,
        request.activated_run_pk,
        request.activated_build_key,
        request.activated_build_generation,
        request.activated_source_hash,
        request.activated_occurrence_signature,
    ) == expected_audit
    request._validate_success_activation()


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
    values["scope_id"] = str(values["scope_id"])
    if "collection_scope_id" not in overrides:
        values["collection_scope_id"] = None
        if values["scope_type"] == GraphArtifact.ScopeType.COLLECTION:
            try:
                values["collection_scope_id"] = int(values["scope_id"])
            except ValueError:
                values["collection_scope_id"] = COLLECTION_ID
    if "embedding_model_signature" not in overrides:
        values["embedding_model_signature"] = (
            COLLECTION_EMBEDDING_SIGNATURE
            if values["scope_type"] == GraphArtifact.ScopeType.COLLECTION
            else ""
        )
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


def test_activated_collection_artifact_history_cannot_be_rewritten_or_deleted():
    artifact = _artifact(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=COLLECTION_ID,
        activated_at=timezone.now(),
    )

    with pytest.raises(ValidationError, match="immutable|activation"):
        GraphArtifact.objects.filter(pk=artifact.pk).update(activated_at=None)
    with pytest.raises(ValidationError, match="activation"):
        artifact.delete()

    building = _artifact(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=COLLECTION_ID,
    )
    with pytest.raises(ValidationError, match="activation"):
        building.delete()
    assert "if self.pk:" in inspect.getsource(GraphArtifact.clean)
    assert "self.pk and not self._state.adding" not in inspect.getsource(
        GraphArtifact.clean
    )


@pytest.mark.django_db(transaction=True)
@database_required
def test_explicit_pk_instance_cannot_clear_persisted_activation_history():
    from apps.collections.models import Collection

    Collection.objects.create(
        pk=COLLECTION_ID,
        name=f"activation-history-{uuid.uuid4()}",
    )
    artifact = _artifact(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=COLLECTION_ID,
        status=GraphArtifact.Status.SUPERSEDED,
        activated_at=timezone.now(),
    )
    artifact.save()
    values = GraphArtifact._base_manager.filter(pk=artifact.pk).values().get()
    values["activated_at"] = None
    replacement = GraphArtifact(**values)

    assert replacement._state.adding
    with pytest.raises(ValidationError, match="activation"):
        replacement.save()

    artifact.refresh_from_db()
    assert artifact.activated_at is not None


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


def test_collection_promotion_confidence_preserves_missing_evidence():
    field = CollectionEntity._meta.get_field("promotion_confidence")

    assert field.null is True
    entity = CollectionEntity(
        extraction_confidence=0.9,
        resolution_confidence=0.8,
        retrieval_utility=0.7,
        promotion_confidence=None,
    )
    assert entity._raw_validation_errors() == {}


@pytest.mark.django_db(transaction=True)
@database_required
@pytest.mark.parametrize("model", [DocumentEntity, CollectionEntity])
def test_identifier_first_database_uniqueness(model):
    if model is CollectionEntity:
        from apps.collections.models import Collection

        Collection.objects.create(pk=COLLECTION_ID, name="KG uniqueness")
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
    document_fields = {"cluster_key": "1" * 64, "version_signature": "v1"}
    if model is CollectionEntity:
        document_fields.update(
            extraction_confidence=0.9,
            resolution_confidence=0.9,
            retrieval_utility=0.5,
            promotion_confidence=0.0,
        )
    model.objects.create(
        **common,
        **document_fields,
        label="First",
        normalized_label="first",
        identifier="stable-id",
    )
    with pytest.raises(ValidationError, match="identifier"):
        model.objects.create(
            **common,
            **{
                **document_fields,
                "cluster_key": "2" * 64,
            },
            label="Different label",
            normalized_label="different-label",
            identifier="stable-id",
        )
    model.objects.create(
        **common,
        **{
            **document_fields,
            "cluster_key": "3" * 64,
        },
        label="First",
        normalized_label="first",
        identifier="other-id",
    )
    if model is DocumentEntity:
        model.objects.create(
            **common,
            cluster_key="4" * 64,
            version_signature="v2",
            label="First v2",
            normalized_label="first v2",
            identifier="stable-id",
        )
        model.objects.create(
            **common,
            cluster_key="5" * 64,
            version_signature="",
            label="First versionless",
            normalized_label="first-versionless",
            identifier="stable-id",
        )


def test_graph_artifact_has_scope_lifecycle_identity_constraints_and_indexes():
    assert GraphArtifact._meta.get_field("scope_id").get_internal_type() == "CharField"
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
    occurrence = _constraint(
        GraphArtifact, "kg_artifact_build_occurrence", UniqueConstraint
    )
    assert tuple(active.fields) == ("scope_type", "scope_id")
    assert active.condition is not None
    assert tuple(occurrence.fields) == (
        "scope_type",
        "scope_id",
        "build_key",
        "build_generation",
    )
    assert _constraint(
        GraphArtifact, "kg_artifact_assembly_identity_scope", CheckConstraint
    )
    assert _constraint(
        GraphArtifact, "kg_artifact_collection_scope_xor", CheckConstraint
    )
    assert {("scope_type", "scope_id", "status"), ("source_hash",)} <= _index_fields(
        GraphArtifact
    )


def test_collection_graph_ownership_is_explicit_immutable_and_non_cascading():
    collection_scope = GraphArtifact._meta.get_field("collection_scope")

    assert collection_scope.null is True
    assert collection_scope.remote_field.on_delete is models.DO_NOTHING
    assert "collection_scope" in GraphArtifact._IMMUTABLE_FIELDS
    assert "collection_scope_id" in GraphArtifact._IMMUTABLE_FIELDS
    for model in (CollectionArtifactInput, CollectionEntity):
        collection = model._meta.get_field("collection")
        assert collection.null is False
        assert collection.remote_field.on_delete is models.DO_NOTHING


def test_graph_artifact_collection_scope_auto_binds_and_matches_typed_scope():
    auto_bound = _artifact(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=COLLECTION_ID,
        collection_scope_id=None,
    )
    auto_bound.prepare_for_persistence()
    assert auto_bound.collection_scope_id == COLLECTION_ID

    mismatched = _artifact(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=COLLECTION_ID,
        collection_scope_id=COLLECTION_ID + 1,
    )
    with pytest.raises(ValidationError, match="match the canonical scope ID"):
        mismatched.prepare_for_persistence()

    document = _artifact(collection_scope_id=COLLECTION_ID)
    with pytest.raises(ValidationError, match="cannot own a collection scope"):
        document.prepare_for_persistence()


def test_collection_graph_scope_rejects_values_outside_signed_bigint():
    from apps.knowledge_graph.models.artifacts import canonical_graph_scope_id

    assert canonical_graph_scope_id("collection", 2**63 - 1) == str(2**63 - 1)
    with pytest.raises(ValidationError, match="signed bigint"):
        canonical_graph_scope_id("collection", 2**63)


def test_artifact_assembly_identity_is_typed_by_scope_and_policy_addressed():
    from apps.knowledge_graph.graph.assembly import (
        AssemblyConfig,
        assembly_config_checksum,
    )
    from apps.knowledge_graph.models import (
        ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM,
        ASSEMBLY_NOT_APPLICABLE_VERSION,
    )

    document = _artifact()
    document.prepare_for_persistence()
    assert document.assembly_version == ASSEMBLY_NOT_APPLICABLE_VERSION
    assert document.assembly_config_checksum == ASSEMBLY_NOT_APPLICABLE_CONFIG_CHECKSUM

    collection = _artifact(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=1,
    )
    collection.prepare_for_persistence()
    config = AssemblyConfig()
    assert collection.assembly_version == config.version
    assert collection.assembly_config_checksum == assembly_config_checksum(config)

    forged_document = _artifact(
        assembly_version=config.version,
        assembly_config_checksum=assembly_config_checksum(config),
    )
    with pytest.raises(ValidationError, match="not-applicable"):
        forged_document.prepare_for_persistence()


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
    assert _constraint(GraphBuildRun, "kg_run_assembly_identity_scope", CheckConstraint)
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
    assert {
        ("artifact", "collection", "entity_type", "normalized_label")
    } <= _index_fields(CollectionEntity)
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
        assert mapping_field.null is True
        assert mapping_field.remote_field.model is CollectionEntityDocumentLink
        assert mapping_field.remote_field.on_delete.__name__ == "RESTRICT"
    assert {
        ("artifact", "source", "target", "relation_type"),
        ("artifact", "target", "source", "relation_type"),
    } <= _index_fields(CollectionRelation)


def test_collection_relation_rejects_cross_artifact_or_cross_collection_endpoints():
    artifact = _artifact(
        pk=1,
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=COLLECTION_ID,
    )
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
        collection_id=COLLECTION_ID + 1,
        label="MMLU",
        normalized_label="mmlu",
        entity_type="benchmark",
    )
    relation = CollectionRelation(
        artifact=artifact,
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


def test_relation_mentions_cannot_be_appended_to_active_document_artifacts():
    relation_mention = _unsaved_relation_evidence().relation_mention
    relation_mention.pk = None

    with pytest.raises(ValidationError, match="building"):
        relation_mention.clean()


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
    assert not any(
        constraint.name == "kg_collection_entity_label_fallback"
        for constraint in CollectionEntity._meta.constraints
    )
    assert (
        _constraint(
            CollectionEntity,
            "kg_collection_entity_cluster_unique",
            UniqueConstraint,
        ).condition
        is None
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
    "version_signature",
    [
        None,
        1,
        True,
        "V1",
        " v1",
        "v 1",
        "v1\x00",
        "v1\n",
        "é1",
        "_v1",
        "x" * 129,
    ],
)
def test_document_entity_rejects_noncanonical_version_signatures(version_signature):
    entity = DocumentEntity(
        artifact=_artifact(),
        document_id=DOCUMENT_ID,
        cluster_key="1" * 64,
        label="Aquilla v1",
        normalized_label="aquilla v1",
        version_signature=version_signature,
        entity_type="model",
        identifier="repository:github.com/example/aquilla",
    )

    with pytest.raises(ValidationError, match="version_signature"):
        entity.clean()


@pytest.mark.parametrize("version_signature", ["", "v1", "3.1+8b+instruct", "rc2"])
def test_document_entity_accepts_canonical_version_signatures(version_signature):
    entity = DocumentEntity(
        artifact=_artifact(),
        document_id=DOCUMENT_ID,
        cluster_key="1" * 64,
        label="Aquilla v1",
        normalized_label="aquilla v1",
        version_signature=version_signature,
        entity_type="model",
        identifier="repository:github.com/example/aquilla",
    )

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
            "kg_document_version_signature_valid",
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
        ontology_checksum="7" * 64,
    )
    collection_artifact.prepare_for_persistence()
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
    manifest = CollectionArtifactInput(
        pk=55,
        artifact=collection_artifact,
        collection_id=COLLECTION_ID,
        document_id=DOCUMENT_ID,
        document_artifact=document_artifact,
        source_signature="5" * 64,
        membership_signature="6" * 64,
        build_signature="7" * 64,
    )
    head_mapping = CollectionEntityDocumentLink(
        pk=60,
        artifact=collection_artifact,
        manifest_input=manifest,
        document_entity=head_document_entity,
        collection_entity=source,
        score=0.9,
        method="exact",
        resolver_version=collection_artifact.resolver_version,
        outcome=CollectionEntityDocumentLink.Outcome.AUTOMATIC,
    )
    tail_mapping = CollectionEntityDocumentLink(
        pk=61,
        artifact=collection_artifact,
        manifest_input=manifest,
        document_entity=tail_document_entity,
        collection_entity=target,
        score=0.9,
        method="exact",
        resolver_version=collection_artifact.resolver_version,
        outcome=CollectionEntityDocumentLink.Outcome.AUTOMATIC,
    )
    return CollectionRelationEvidence(
        artifact=collection_artifact,
        relation=relation,
        relation_mention=relation_mention,
        head_mapping=head_mapping,
        tail_mapping=tail_mapping,
        ontology_checksum=collection_artifact.ontology_checksum,
        assembly_config_checksum=collection_artifact.assembly_config_checksum,
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


def test_relation_evidence_rejects_mismatched_assembly_identity():
    evidence = _unsaved_relation_evidence()
    evidence.assembly_config_checksum = "f" * 64

    with pytest.raises(ValidationError, match="assembly checksum"):
        evidence.clean()


def test_collection_artifact_children_cannot_be_deleted_directly():
    evidence = _unsaved_relation_evidence()
    rows = (
        evidence.head_mapping.manifest_input,
        evidence.relation.source,
        evidence.head_mapping,
        evidence.relation,
        evidence,
    )

    for row in rows:
        with pytest.raises(ValidationError, match="deleted directly"):
            row.delete()


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
    evidence.relation.target.collection_id = COLLECTION_ID + 1
    monkeypatch.setattr(
        evidence,
        "_endpoint_membership_is_active",
        lambda _mapping, _mention: True,
        raising=False,
    )

    with pytest.raises(ValidationError, match="artifact|collection"):
        evidence.clean()


def test_relation_evidence_rejects_mapping_row_from_other_collection_artifact(
    monkeypatch,
):
    evidence = _unsaved_relation_evidence()
    evidence.head_mapping.artifact_id = 999
    monkeypatch.setattr(
        evidence,
        "_endpoint_membership_is_active",
        lambda _mapping, _mention: True,
        raising=False,
    )

    with pytest.raises(ValidationError, match="artifact"):
        evidence.clean()


def test_relation_evidence_rejects_nonautomatic_or_wrong_manifest_mapping(
    monkeypatch,
):
    evidence = _unsaved_relation_evidence()
    evidence.head_mapping.outcome = CollectionEntityDocumentLink.Outcome.CANDIDATE
    evidence.tail_mapping.manifest_input.artifact_id = 999
    monkeypatch.setattr(
        evidence,
        "_endpoint_membership_is_active",
        lambda _mapping, _mention: True,
        raising=False,
    )

    with pytest.raises(ValidationError, match="automatic|manifest"):
        evidence.clean()


def test_rejected_relation_evidence_can_retain_raw_mention_without_mappings():
    evidence = _unsaved_relation_evidence()
    evidence.relation = None
    evidence.head_mapping = None
    evidence.tail_mapping = None
    evidence.status = CollectionRelationEvidence.Status.REJECTED
    evidence.reason = "missing_active_mapping"

    evidence.clean()


def test_collection_relation_requires_at_least_one_real_support():
    relation = _unsaved_relation_evidence().relation
    relation.pk = None
    relation.status = CollectionRelation.Status.SUPPRESSED
    relation.support_count = 0
    relation.confidence = 0.5

    with pytest.raises(ValidationError, match="support"):
        relation.clean()


def test_new_relation_building_checks_use_adding_state_not_primary_key_presence():
    relation_source = inspect.getsource(CollectionRelation.clean)
    evidence_source = inspect.getsource(CollectionRelationEvidence.clean)

    assert "self._state.adding" in relation_source
    assert "self._state.adding" in evidence_source
    assert "and not self.pk" not in relation_source
    assert "and not self.pk" not in evidence_source


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
    artifact = _artifact(status=GraphArtifact.Status.BUILDING)
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
        method=DocumentEntityMention.Method.ROOT,
        resolver_version=artifact.resolver_version,
    )
    artifact.status = GraphArtifact.Status.ACTIVE
    artifact.save(update_fields=["status"])

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
        method=DocumentEntityMention.Method.ROOT,
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
        method=DocumentEntityMention.Method.ROOT,
        resolver_version=artifact.resolver_version,
    )

    replacement_link.delete()

    assert EntityMention.objects.filter(pk=mention.pk).exists()


@pytest.mark.django_db(transaction=True)
@database_required
def test_raw_evidence_instance_save_rejects_rewrites():
    artifact = _artifact(status=GraphArtifact.Status.BUILDING)
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
    artifact.status = GraphArtifact.Status.ACTIVE
    artifact.save(update_fields=["status"])

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
def _persist_collection_relation_fixture():
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument

    user = User.objects.create_user(
        username=f"kg-model-{uuid.uuid4()}", password="unused"
    )
    collection = Collection.objects.create(
        pk=COLLECTION_ID,
        name=f"KG model fixture {uuid.uuid4()}",
    )
    document = RawTextDocument(
        id=DOCUMENT_ID,
        title="Aquilla",
        full_text="Aquilla evaluates MMLU.",
        collection=collection,
        ingested_by=user,
        full_text_hash=RawTextDocument.hash_fn("Aquilla evaluates MMLU."),
    )
    document.save(dont_rechunk=True)
    collection_artifact = _artifact(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=COLLECTION_ID,
        status=GraphArtifact.Status.BUILDING,
        ontology_checksum="7" * 64,
    )
    collection_artifact.save()
    document_artifact = _artifact(status=GraphArtifact.Status.BUILDING)
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
        method=DocumentEntityMention.Method.ROOT,
        resolver_version=document_artifact.resolver_version,
    )
    DocumentEntityMention.objects.create(
        document_entity=tail_document_entity,
        mention=tail,
        method=DocumentEntityMention.Method.ROOT,
        resolver_version=document_artifact.resolver_version,
    )
    document_artifact.status = GraphArtifact.Status.ACTIVE
    document_artifact.save(update_fields=["status"])
    manifest = CollectionArtifactInput.objects.create(
        artifact=collection_artifact,
        collection=collection,
        document_id=DOCUMENT_ID,
        document_artifact=document_artifact,
        source_signature="0" * 64,
        build_signature="0" * 64,
    )
    source = CollectionEntity.objects.create(
        artifact=collection_artifact,
        collection_id=COLLECTION_ID,
        cluster_key="3" * 64,
        label="Aquilla",
        normalized_label="aquilla",
        entity_type="model",
        extraction_confidence=0.9,
        resolution_confidence=0.9,
        retrieval_utility=0.5,
        promotion_confidence=0.0,
    )
    target = CollectionEntity.objects.create(
        artifact=collection_artifact,
        collection_id=COLLECTION_ID,
        cluster_key="4" * 64,
        label="MMLU",
        normalized_label="mmlu",
        entity_type="benchmark",
        extraction_confidence=0.9,
        resolution_confidence=0.9,
        retrieval_utility=0.5,
        promotion_confidence=0.0,
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
        artifact=collection_artifact,
        manifest_input=manifest,
        document_entity=head_document_entity,
        collection_entity=source,
        score=0.9,
        method="exact",
        resolver_version=collection_artifact.resolver_version,
        outcome=CollectionEntityDocumentLink.Outcome.AUTOMATIC,
        decision_checksum="5" * 64,
        reason="exact",
    )
    tail_mapping = CollectionEntityDocumentLink.objects.create(
        artifact=collection_artifact,
        manifest_input=manifest,
        document_entity=tail_document_entity,
        collection_entity=target,
        score=0.9,
        method="exact",
        resolver_version=collection_artifact.resolver_version,
        outcome=CollectionEntityDocumentLink.Outcome.AUTOMATIC,
        decision_checksum="6" * 64,
        reason="exact",
    )
    evidence = CollectionRelationEvidence.objects.create(
        artifact=collection_artifact,
        relation=relation,
        relation_mention=relation_mention,
        head_mapping=head_mapping,
        tail_mapping=tail_mapping,
        ontology_checksum=collection_artifact.ontology_checksum,
        assembly_config_checksum=collection_artifact.assembly_config_checksum,
    )
    return SimpleNamespace(
        collection_artifact=collection_artifact,
        document_artifact=document_artifact,
        relation_mention=relation_mention,
        relation=relation,
        head_mapping=head_mapping,
        tail_mapping=tail_mapping,
        evidence=evidence,
    )


@pytest.mark.django_db(transaction=True)
@database_required
def test_collection_artifact_and_child_direct_deletion_is_refused():
    fixture = _persist_collection_relation_fixture()

    with pytest.raises(ValidationError, match="deleted directly"):
        fixture.head_mapping.delete()
    fixture.collection_artifact.scope_type = GraphArtifact.ScopeType.DOCUMENT
    with pytest.raises(ValidationError, match="activation"):
        fixture.collection_artifact.delete()

    assert CollectionRelationEvidence.objects.filter(pk=fixture.evidence.pk).exists()
    assert (
        CollectionEntityDocumentLink.objects.filter(
            pk__in=(fixture.head_mapping.pk, fixture.tail_mapping.pk)
        ).count()
        == 2
    )
    assert CollectionRelation.objects.filter(pk=fixture.relation.pk).exists()
    assert RelationMention.objects.filter(pk=fixture.relation_mention.pk).exists()


@pytest.mark.django_db(transaction=True)
@database_required
def test_relation_evidence_preserves_each_unique_support_and_protects_raw_mention(
    django_assert_num_queries,
):
    fixture = _persist_collection_relation_fixture()
    mention = fixture.relation_mention
    relation = fixture.relation
    head_mapping = fixture.head_mapping
    tail_mapping = fixture.tail_mapping
    evidence = fixture.evidence

    head_mapping.status = CollectionEntityDocumentLink.Status.SUPPRESSED
    with pytest.raises(ValidationError, match="immutable"):
        head_mapping.save(update_fields=["status"])
    with pytest.raises(ValidationError, match="active"):
        evidence.clean()
    head_mapping.refresh_from_db()
    # Refreshing the mapping deliberately clears its three cached FK inputs;
    # validation reloads those plus immutable state and two endpoint memberships.
    with django_assert_num_queries(6):
        evidence.clean()
    evidence.save()

    with pytest.raises(ValidationError), transaction.atomic():
        CollectionRelationEvidence.objects.create(
            artifact=fixture.collection_artifact,
            relation=relation,
            relation_mention=mention,
            head_mapping=head_mapping,
            tail_mapping=tail_mapping,
            ontology_checksum=fixture.collection_artifact.ontology_checksum,
            assembly_config_checksum=(
                fixture.collection_artifact.assembly_config_checksum
            ),
        )

    with pytest.raises(ProtectedError):
        mention.delete()

    assert CollectionRelationEvidence.objects.filter(pk=evidence.pk).exists()
    assert CollectionRelation.objects.filter(pk=relation.pk).exists()
