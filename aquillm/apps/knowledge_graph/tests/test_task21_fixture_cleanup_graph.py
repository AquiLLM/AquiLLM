from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest
from django.core.management import CommandError
from django.db import connection
from django.test import override_settings
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from apps.knowledge_graph.tests.task21_fixture_test_support import (
    MODEL_SIGNATURE,
    cleanup,
    create_document_eval_artifact,
    create_eval_request,
    database_counts,
    seed,
    strict_eval_environment,
)

_STRICT_EVAL_ENVIRONMENT = strict_eval_environment


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_cleanup_discovers_adversarial_run_rebound_to_fixture_artifact(
    tmp_path, monkeypatch
) -> None:
    from apps.knowledge_graph.models import (
        GraphArtifact,
        GraphBuildRun,
        GraphRebuildRequest,
    )

    manifest_path = tmp_path / "fixture.json"
    payload, _output, _observed = seed(manifest_path, monkeypatch)
    request, documents = create_eval_request(
        payload, status=GraphRebuildRequest.Status.FAILED
    )
    fixture_artifact = create_document_eval_artifact(
        request, documents[0], status=GraphArtifact.Status.FAILED
    )
    foreign_scope = str(uuid4())
    foreign_artifact = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=foreign_scope,
        status=GraphArtifact.Status.FAILED,
        source_hash="f" * 64,
        ontology_version="foreign-ontology",
        extractor_version="foreign-extractor",
        resolver_version="foreign-resolver",
        filter_policy_version="foreign-filter",
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        completed_at=timezone.now(),
    )
    foreign_run = GraphBuildRun.objects.create(
        artifact=foreign_artifact,
        stage=GraphBuildRun.Stage.FAILED,
        status=GraphBuildRun.Status.FAILED,
        started_at=timezone.now(),
        finished_at=timezone.now(),
    )
    with connection.cursor() as cursor:
        table = connection.ops.quote_name(GraphBuildRun._meta.db_table)
        cursor.execute(
            f"UPDATE {table} SET artifact_id = %s WHERE id = %s",
            [fixture_artifact.pk, foreign_run.pk],
        )
    with CaptureQueriesContext(connection) as captured:
        with pytest.raises(CommandError, match="graph"):
            cleanup(manifest_path, payload)
    run_queries = [
        query["sql"].upper()
        for query in captured.captured_queries
        if GraphBuildRun._meta.db_table.upper() in query["sql"].upper()
        and query["sql"].lstrip().upper().startswith("SELECT")
    ]
    assert any('."ARTIFACT_ID" IN' in sql for sql in run_queries)
    foreign_run.refresh_from_db()
    assert foreign_run.artifact_id == fixture_artifact.pk


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_cleanup_rejects_running_eval_request(tmp_path, monkeypatch) -> None:
    from apps.knowledge_graph.models import GraphRebuildRequest

    manifest_path = tmp_path / "fixture.json"
    payload, _output, _observed = seed(manifest_path, monkeypatch)
    request, _documents = create_eval_request(
        payload, status=GraphRebuildRequest.Status.RUNNING
    )
    before = database_counts()
    with pytest.raises(CommandError, match="graph"):
        cleanup(manifest_path, payload)
    assert database_counts() == before
    assert GraphRebuildRequest.objects.filter(pk=request.pk).exists()


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
@pytest.mark.parametrize("error_code", ("resnapshot_pending", "resnapshot_churn"))
def test_cleanup_rejects_reconcilable_partial_request(
    tmp_path, monkeypatch, error_code
) -> None:
    from apps.knowledge_graph.models import GraphRebuildRequest

    manifest_path = tmp_path / "fixture.json"
    payload, _output, _observed = seed(manifest_path, monkeypatch)
    request, _documents = create_eval_request(
        payload,
        status=GraphRebuildRequest.Status.PARTIAL,
        error_code=error_code,
    )
    with pytest.raises(CommandError, match="graph"):
        cleanup(manifest_path, payload)
    assert GraphRebuildRequest.objects.filter(pk=request.pk).exists()


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_cleanup_rejects_pending_publication(tmp_path, monkeypatch) -> None:
    from django.db import connection

    from apps.knowledge_graph.models import GraphRebuildRequest

    manifest_path = tmp_path / "fixture.json"
    payload, _output, _observed = seed(manifest_path, monkeypatch)
    request, _documents = create_eval_request(
        payload, status=GraphRebuildRequest.Status.FAILED
    )
    with connection.cursor() as cursor:
        table = connection.ops.quote_name(GraphRebuildRequest._meta.db_table)
        cursor.execute(
            f"UPDATE {table} SET document_publication_state = %s WHERE id = %s",
            [GraphRebuildRequest.PublicationState.PENDING, request.pk],
        )
    with pytest.raises(CommandError, match="graph"):
        cleanup(manifest_path, payload)
    assert GraphRebuildRequest.objects.filter(pk=request.pk).exists()


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_cleanup_rejects_building_eval_artifact(tmp_path, monkeypatch) -> None:
    from apps.knowledge_graph.models import GraphArtifact, GraphRebuildRequest

    manifest_path = tmp_path / "fixture.json"
    payload, _output, _observed = seed(manifest_path, monkeypatch)
    request, documents = create_eval_request(
        payload, status=GraphRebuildRequest.Status.FAILED
    )
    artifact = create_document_eval_artifact(
        request, documents[0], status=GraphArtifact.Status.BUILDING
    )
    before = database_counts()
    with pytest.raises(CommandError, match="graph"):
        cleanup(manifest_path, payload)
    assert database_counts() == before
    assert GraphArtifact.objects.filter(pk=artifact.pk).exists()


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_cleanup_rejects_live_eval_run(tmp_path, monkeypatch) -> None:
    from apps.knowledge_graph.models import (
        GraphArtifact,
        GraphBuildRun,
        GraphRebuildRequest,
    )

    manifest_path = tmp_path / "fixture.json"
    payload, _output, _observed = seed(manifest_path, monkeypatch)
    request, documents = create_eval_request(
        payload, status=GraphRebuildRequest.Status.FAILED
    )
    artifact = create_document_eval_artifact(
        request, documents[0], status=GraphArtifact.Status.FAILED
    )
    run = GraphBuildRun.objects.create(
        artifact=artifact,
        rebuild_request=request,
        evaluation_only=True,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        stage=GraphBuildRun.Stage.EXTRACTING,
        status=GraphBuildRun.Status.RUNNING,
        lease_owner="task21-live-lease",
        lease_generation=1,
        lease_expires_at=timezone.now() + timedelta(minutes=5),
        started_at=timezone.now(),
    )
    with pytest.raises(CommandError, match="graph"):
        cleanup(manifest_path, payload)
    assert GraphBuildRun.objects.filter(pk=run.pk).exists()


@pytest.mark.django_db(transaction=True)
@override_settings(DEBUG=True)
def test_cleanup_accepts_exact_terminal_eval_occurrence_and_preserves_audit(
    tmp_path, monkeypatch
) -> None:
    from apps.knowledge_graph.models import (
        GraphArtifact,
        GraphBuildRun,
        GraphRebuildRequest,
    )
    from apps.knowledge_graph.models.artifacts import _activation_audit_values

    manifest_path = tmp_path / "fixture.json"
    payload, _output, _observed = seed(manifest_path, monkeypatch)
    request, documents = create_eval_request(
        payload, status=GraphRebuildRequest.Status.RUNNING
    )
    completed_at = timezone.now()
    artifact = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=request.scope_id,
        collection_scope_id=int(request.scope_id),
        status=GraphArtifact.Status.SUPERSEDED,
        source_hash=request.expected_aggregate_signature,
        ontology_version="task21-test-ontology",
        extractor_version="task21-test-extractor",
        resolver_version="task21-test-resolver",
        filter_policy_version="task21-test-filter",
        embedding_model_signature=MODEL_SIGNATURE,
        assembly_version="task21-test-assembly",
        assembly_config_checksum="e" * 64,
        rebuild_request=request,
        evaluation_only=True,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        completed_at=completed_at,
        superseded_at=completed_at,
    )
    run = GraphBuildRun.objects.create(
        artifact=artifact,
        rebuild_request=request,
        evaluation_only=True,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        stage=GraphBuildRun.Stage.SUPERSEDED,
        status=GraphBuildRun.Status.CANCELLED,
        stage_marker={"evaluation_completed": True},
        started_at=completed_at,
        finished_at=completed_at,
    )
    for field, value in _activation_audit_values(artifact, run).items():
        setattr(request, field, value)
    request.status = GraphRebuildRequest.Status.SUCCEEDED
    request.completed_document_count = len(documents)
    request.completed_collection_count = 1
    request.document_publication_state = GraphRebuildRequest.PublicationState.PUBLISHED
    request.collection_publication_state = (
        GraphRebuildRequest.PublicationState.PUBLISHED
    )
    request.completed_at = completed_at
    request.save()
    cleanup(manifest_path, payload)
    assert GraphRebuildRequest.objects.filter(pk=request.pk).exists()
    assert not GraphArtifact.objects.filter(pk=artifact.pk).exists()
    run.refresh_from_db()
    assert run.artifact_id is None
    assert run.lease_owner == "" and run.lease_expires_at is None
