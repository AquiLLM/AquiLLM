from __future__ import annotations

import importlib
import inspect
import os
import socket
import uuid

import pytest
from django.conf import settings
from django.db import IntegrityError, migrations, models, transaction
from django.db.models import Q, UniqueConstraint
from django.db.models.deletion import Collector

from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

DOCUMENT_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
COLLECTION_ID = 17


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


def _constraint(model, name: str) -> UniqueConstraint:
    return next(
        constraint
        for constraint in model._meta.constraints
        if constraint.name == name and isinstance(constraint, UniqueConstraint)
    )


def _artifact(
    *,
    build_key: str,
    build_generation: int,
    orchestration_version: int = GraphArtifact.OrchestrationVersion.SCOPED_V1,
    scope_type: str = GraphArtifact.ScopeType.DOCUMENT,
    scope_id: object = DOCUMENT_ID,
    status: str = GraphArtifact.Status.BUILDING,
) -> GraphArtifact:
    values = {
        "scope_type": scope_type,
        "scope_id": scope_id,
        "build_key": build_key,
        "build_generation": build_generation,
        "orchestration_version": orchestration_version,
        "status": status,
        "source_hash": build_key,
        "ontology_version": "ontology-v1",
        "extractor_version": "extractor-v1",
        "resolver_version": "resolver-v1",
        "filter_policy_version": "filter-v1",
    }
    if scope_type == GraphArtifact.ScopeType.COLLECTION:
        values.update(
            {
                "embedding_model_signature": (
                    f"test-local:model@rev:endpoint={'e' * 64}:dims=1024:"
                    "prep=kg-entity-v1:max_chars=8192:batch=64"
                ),
                "assembly_version": "collection-assembly-v1",
                "assembly_config_checksum": "f" * 64,
            }
        )
    artifact = GraphArtifact(**values)
    artifact.prepare_for_persistence()
    return artifact


def _detached_run(
    *,
    build_key: str,
    build_generation: int,
    orchestration_version: int = GraphArtifact.OrchestrationVersion.SCOPED_V1,
) -> GraphBuildRun:
    run = GraphBuildRun(
        build_key=build_key,
        build_generation=build_generation,
        orchestration_version=orchestration_version,
        build_kind=GraphBuildRun.BuildKind.DOCUMENT,
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=DOCUMENT_ID,
        source_hash=build_key,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="filter-v1",
        stage=GraphBuildRun.Stage.QUEUED,
        status=GraphBuildRun.Status.PENDING,
    )
    run.prepare_for_persistence()
    return run


def _save_without_model_validation(instance) -> None:
    """Exercise database constraints rather than full_clean() uniqueness checks."""

    models.Model.save(instance, force_insert=True)


def test_scoped_generation_constraints_exclude_the_build_key():
    artifact_constraint = _constraint(
        GraphArtifact, "kg_artifact_scope_generation_unique"
    )
    run_constraint = _constraint(GraphBuildRun, "kg_run_scope_generation_unique")

    assert tuple(artifact_constraint.fields) == (
        "scope_type",
        "scope_id",
        "build_generation",
    )
    assert artifact_constraint.condition == Q(
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1
    )
    assert tuple(run_constraint.fields) == (
        "build_kind",
        "scope_type",
        "scope_id",
        "build_generation",
    )
    assert run_constraint.condition == Q(
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1
    )


def test_migration_adds_scoped_generation_constraints_after_correspondence_backfill():
    migration = importlib.import_module(
        "apps.knowledge_graph.migrations.0002_graph_build_run_stages"
    )
    constraints = {
        operation.constraint.name: operation.constraint
        for operation in migration.Migration.operations
        if isinstance(operation, migrations.AddConstraint)
    }
    operation_positions = {
        operation.constraint.name: position
        for position, operation in enumerate(migration.Migration.operations)
        if isinstance(operation, migrations.AddConstraint)
    }
    backfill_position = next(
        position
        for position, operation in enumerate(migration.Migration.operations)
        if isinstance(operation, migrations.RunPython)
    )

    assert tuple(constraints["kg_artifact_scope_generation_unique"].fields) == (
        "scope_type",
        "scope_id",
        "build_generation",
    )
    assert tuple(constraints["kg_run_scope_generation_unique"].fields) == (
        "build_kind",
        "scope_type",
        "scope_id",
        "build_generation",
    )
    assert (
        backfill_position < operation_positions["kg_artifact_scope_generation_unique"]
    )
    assert backfill_position < operation_positions["kg_run_scope_generation_unique"]
    backfill_source = inspect.getsource(migration.populate_build_keys)
    assert "scope = (artifact.scope_type, artifact.scope_id)" in backfill_source
    assert "run.build_generation = run.artifact.build_generation" in backfill_source


@pytest.mark.django_db(transaction=True)
@database_required
def test_artifact_database_rejects_different_keys_sharing_scoped_generation():
    first = _artifact(build_key="a" * 64, build_generation=1)
    duplicate_generation = _artifact(build_key="b" * 64, build_generation=1)
    _save_without_model_validation(first)

    with pytest.raises(IntegrityError), transaction.atomic():
        _save_without_model_validation(duplicate_generation)


@pytest.mark.django_db(transaction=True)
@database_required
def test_run_database_rejects_different_keys_sharing_scoped_generation():
    first = _detached_run(build_key="a" * 64, build_generation=1)
    duplicate_generation = _detached_run(build_key="b" * 64, build_generation=1)
    _save_without_model_validation(first)

    with pytest.raises(IntegrityError), transaction.atomic():
        _save_without_model_validation(duplicate_generation)


@pytest.mark.django_db(transaction=True)
@database_required
def test_legacy_rows_can_reuse_a_generation_across_different_keys():
    legacy = GraphArtifact.OrchestrationVersion.LEGACY
    artifacts = (
        _artifact(
            build_key="a" * 64,
            build_generation=1,
            orchestration_version=legacy,
        ),
        _artifact(
            build_key="b" * 64,
            build_generation=1,
            orchestration_version=legacy,
        ),
    )
    runs = (
        _detached_run(
            build_key="c" * 64,
            build_generation=1,
            orchestration_version=legacy,
        ),
        _detached_run(
            build_key="d" * 64,
            build_generation=1,
            orchestration_version=legacy,
        ),
    )

    for row in (*artifacts, *runs):
        _save_without_model_validation(row)

    assert GraphArtifact.objects.filter(orchestration_version=legacy).count() == 2
    assert GraphBuildRun.objects.filter(orchestration_version=legacy).count() == 2


@pytest.mark.django_db(transaction=True)
@database_required
def test_distinct_scoped_generations_allocate_and_activate_in_order():
    from apps.collections.models import Collection
    from apps.knowledge_graph.graph.assembly import _swap_active_collection_artifact
    from apps.knowledge_graph.services.builds import _next_build_generation

    Collection.objects.create(pk=COLLECTION_ID, name=f"generation {uuid.uuid4()}")
    prior = _artifact(
        build_key="a" * 64,
        build_generation=1,
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=COLLECTION_ID,
        status=GraphArtifact.Status.ACTIVE,
    )
    prior.save()
    prior_run = GraphBuildRun.objects.create(
        artifact=prior,
        stage=GraphBuildRun.Stage.ACTIVE,
        status=GraphBuildRun.Status.SUCCEEDED,
    )
    candidate = _artifact(
        build_key="b" * 64,
        build_generation=2,
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=COLLECTION_ID,
    )
    candidate.save()
    candidate_run = GraphBuildRun.objects.create(
        artifact=candidate,
        stage=GraphBuildRun.Stage.VALIDATING,
        status=GraphBuildRun.Status.RUNNING,
    )

    assert (prior_run.build_key, prior_run.build_generation) == (
        prior.build_key,
        prior.build_generation,
    )
    assert (candidate_run.build_key, candidate_run.build_generation) == (
        candidate.build_key,
        candidate.build_generation,
    )
    assert _next_build_generation((prior, candidate)) == 3
    with transaction.atomic():
        _swap_active_collection_artifact(
            artifact=candidate,
            run=candidate_run,
            scope_artifacts=(prior, candidate),
        )

    prior.refresh_from_db()
    prior_run.refresh_from_db()
    candidate.refresh_from_db()
    candidate_run.refresh_from_db()
    assert prior.status == GraphArtifact.Status.SUPERSEDED
    assert prior_run.stage == GraphBuildRun.Stage.SUPERSEDED
    assert candidate.status == GraphArtifact.Status.ACTIVE
    assert candidate_run.stage == GraphBuildRun.Stage.ACTIVE


@pytest.mark.django_db(transaction=True)
@database_required
@pytest.mark.parametrize(
    ("scope_type", "scope_id"),
    (
        (GraphArtifact.ScopeType.DOCUMENT, DOCUMENT_ID),
        (GraphArtifact.ScopeType.COLLECTION, COLLECTION_ID),
    ),
)
def test_generation_allocation_survives_detached_document_and_collection_audit_runs(
    scope_type,
    scope_id,
):
    from apps.collections.models import Collection
    from apps.knowledge_graph.services.builds import (
        _lock_latest_scope_run,
        _next_build_generation,
    )

    if scope_type == GraphArtifact.ScopeType.COLLECTION:
        Collection.objects.create(pk=scope_id, name=f"generation {uuid.uuid4()}")
    artifact = _artifact(
        build_key="d" * 64,
        build_generation=9,
        scope_type=scope_type,
        scope_id=scope_id,
    )
    artifact.save()
    audit = GraphBuildRun.objects.create(
        artifact=artifact,
        stage=GraphBuildRun.Stage.QUEUED,
        status=GraphBuildRun.Status.PENDING,
    )
    collector = Collector(using="default")
    collector.collect((artifact,))
    collector.delete()
    audit.refresh_from_db()
    assert audit.artifact_id is None

    with transaction.atomic():
        locked = _lock_latest_scope_run(scope_type, scope_id)
        assert tuple(row.pk for row in locked) == (audit.pk,)
        assert _next_build_generation((), locked) == 10
