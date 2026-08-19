from __future__ import annotations

import os
import socket
import uuid
from datetime import timedelta
from inspect import getsource
from types import SimpleNamespace

import pytest
from django.core.management import call_command
from django.test import override_settings
from django.utils import timezone


def _database_is_reachable() -> bool:
    from django.conf import settings

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


def test_retention_config_defaults_and_invalid_values_are_conservative() -> None:
    from lib.knowledge_graph.config import load_retention_settings

    defaults = load_retention_settings({})
    assert defaults.retention_days == 30
    assert defaults.keep_superseded == 2
    assert load_retention_settings({"KG_ARTIFACT_RETENTION_DAYS": "0"}) == defaults
    assert load_retention_settings({"KG_ARTIFACT_KEEP_SUPERSEDED": "-1"}) == defaults


def test_prune_command_defaults_to_dry_run(monkeypatch, capsys) -> None:
    calls: list[dict[str, object]] = []

    def fake_prune(**kwargs):
        calls.append(kwargs)
        from apps.knowledge_graph.services.pruning import PruneReport

        return PruneReport(dry_run=True, artifact_ids=(), run_ids=())

    monkeypatch.setattr(
        "apps.knowledge_graph.management.commands.prune_knowledge_graph.prune_graph_artifacts",
        fake_prune,
    )
    call_command("prune_knowledge_graph")
    assert calls == [{"execute": False}]


def test_prune_report_is_bounded_and_contains_only_ids_counts() -> None:
    from apps.knowledge_graph.services.pruning import PruneReport

    report = PruneReport(dry_run=True, artifact_ids=(9, 4), run_ids=(12,))
    assert report.artifact_ids == (4, 9)
    assert report.run_ids == (12,)
    assert report.artifact_count == 2
    assert report.run_count == 1
    with pytest.raises(ValueError, match="bounded"):
        PruneReport(dry_run=True, artifact_ids=tuple(range(1, 10_002)), run_ids=())


@override_settings(KG_ARTIFACT_RETENTION_DAYS=30, KG_ARTIFACT_KEEP_SUPERSEDED=2)
def test_pruning_policy_uses_strict_older_than_boundary() -> None:
    from apps.knowledge_graph.services.pruning import pruning_boundary

    now = timezone.now()
    assert pruning_boundary(now=now) == now - timedelta(days=30)


def test_prune_plan_uses_one_deterministic_total_row_budget() -> None:
    from apps.knowledge_graph.services.pruning import (
        _ArtifactCandidate,
        _plan_pruning_candidates,
        _RunCandidate,
    )

    now = timezone.now()
    artifacts = (
        _ArtifactCandidate(row_id=8, prune_before=now - timedelta(days=40)),
        _ArtifactCandidate(row_id=3, prune_before=now - timedelta(days=41)),
    )
    runs = (
        _RunCandidate(row_id=13, prune_before=now - timedelta(days=50)),
        _RunCandidate(row_id=11, prune_before=now - timedelta(days=60)),
    )

    plan = _plan_pruning_candidates(
        artifact_candidates=tuple(reversed(artifacts)),
        run_candidates=tuple(reversed(runs)),
        batch_size=3,
    )

    # Artifact removal is planned first so detached audit rows cannot consume the
    # window needed to make progress on artifact retention. Runs share the one
    # remaining row slot rather than receiving a second batch-sized allowance.
    assert plan.artifact_ids == (3, 8)
    assert plan.run_ids == (11,)
    assert len(plan.artifact_ids) + len(plan.run_ids) == 3
    assert plan == _plan_pruning_candidates(
        artifact_candidates=artifacts,
        run_candidates=runs,
        batch_size=3,
    )


def test_newest_run_is_generation_high_water_not_a_prune_candidate() -> None:
    from apps.knowledge_graph.services.pruning import (
        _run_has_clear_lease,
        _run_is_generation_high_water,
    )

    assert _run_is_generation_high_water(newer_scope_run_exists=False) is True
    assert _run_is_generation_high_water(newer_scope_run_exists=True) is False
    assert _run_has_clear_lease(SimpleNamespace(lease_owner="", lease_expires_at=None))
    assert not _run_has_clear_lease(
        SimpleNamespace(lease_owner="worker", lease_expires_at=timezone.now())
    )


def test_artifact_retention_age_requires_exact_terminal_run_and_uses_later_time() -> (
    None
):
    from apps.knowledge_graph.services.pruning import _artifact_retention_age

    now = timezone.now()
    old = now - timedelta(days=40)
    identity = {
        "scope_type": "document",
        "scope_id": str(uuid.uuid4()),
        "rebuild_request_id": None,
        "evaluation_only": False,
        "build_key": "a" * 64,
        "build_generation": 4,
        "orchestration_version": 1,
        "source_hash": "b" * 64,
        "ontology_version": "ontology-v1",
        "extractor_version": "extractor-v1",
        "resolver_version": "resolver-v1",
        "filter_policy_version": "filter-v1",
        "embedding_model_signature": "",
        "ontology_checksum": "c" * 64,
        "filter_policy_checksum": "d" * 64,
        "resolution_config_checksum": "e" * 64,
        "assembly_version": "not-applicable-v1",
        "assembly_config_checksum": "f" * 64,
    }
    artifact = SimpleNamespace(
        status="failed",
        completed_at=now,
        superseded_at=None,
        **identity,
    )
    run = SimpleNamespace(
        build_kind="document",
        stage="failed",
        status="failed",
        finished_at=old,
        lease_owner="",
        lease_expires_at=None,
        **identity,
    )

    assert _artifact_retention_age(artifact, (run,)) == now
    artifact.completed_at = old
    run.finished_at = now
    assert _artifact_retention_age(artifact, (run,)) == now
    run.finished_at = old
    run.ontology_checksum = "0" * 64
    assert _artifact_retention_age(artifact, (run,)) is None
    run.ontology_checksum = identity["ontology_checksum"]
    run.lease_owner = "worker-claim"
    assert _artifact_retention_age(artifact, (run,)) is None
    assert _artifact_retention_age(artifact, ()) is None


def test_pruning_queries_and_guards_are_bounded_before_materialization() -> None:
    from apps.knowledge_graph.services import pruning

    source = getsource(pruning)
    assert "_MAX_QUERY_PREDICATE_IDS = 5_000" in source
    assert "[: batch_size + 1]" in source
    assert "select_for_update" in source
    assert "_newer_scope_run_exists" in source
    assert ".filter(_newer_scope_run_exists=True)" in source
    assert "rebuild_request__status__in" in source
    assert "parent_request__status__in" in source
    assert "activated_artifact" not in source
    assert source.count(".filter(_nonterminal_request_filter())") >= 2
    assert "updated_at__lt" not in source
    assert "superseded_at" in source
    assert "finished_at__lt" in source
    assert 'lease_owner=""' in source
    assert "lease_expires_at__isnull=True" in source
    assert "build_generation__gt=OuterRef" in source
    assert "created_at__gt=OuterRef" not in source
    assert '"finished_at", "build_kind", "scope_id", "build_generation", "pk"' in source


def test_pruning_candidate_queries_compile_with_bounded_predicates() -> None:
    from apps.knowledge_graph.services.pruning import (
        _candidate_artifact_queryset,
        _candidate_run_queryset,
    )

    boundary = timezone.now() - timedelta(days=30)
    artifact_sql, artifact_params = _candidate_artifact_queryset(
        boundary=boundary
    ).query.sql_with_params()
    run_sql, run_params = _candidate_run_queryset(
        tuple(range(1, 1_002)), boundary=boundary
    ).query.sql_with_params()

    assert "EXISTS" in artifact_sql.upper()
    assert "GREATEST" in artifact_sql.upper()
    assert "EXISTS" in run_sql.upper()
    assert len(artifact_params) < 5_000
    assert len(run_params) < 5_000


def test_pruning_service_is_not_gated_by_build_feature_flag(monkeypatch) -> None:
    from apps.knowledge_graph.services import pruning

    monkeypatch.setenv("KG_BUILD_ENABLED", "0")
    calls: list[tuple[object, int]] = []
    monkeypatch.setattr(
        pruning,
        "_build_prune_plan",
        lambda *, boundary, batch_size: (
            calls.append((boundary, batch_size))
            or pruning._PrunePlan(artifact_ids=(), run_ids=())
        ),
    )

    report = pruning.prune_graph_artifacts(execute=False, batch_size=7)

    assert len(calls) == 1
    assert calls[0][1] == 7
    assert report == pruning.PruneReport(
        dry_run=True,
        artifact_ids=(),
        run_ids=(),
    )


def test_pruning_task_is_low_priority_and_calls_execute(monkeypatch) -> None:
    from apps.knowledge_graph import tasks

    calls: list[dict[str, object]] = []
    monkeypatch.setenv("KG_BUILD_ENABLED", "1")
    monkeypatch.setattr(
        "apps.knowledge_graph.services.pruning.prune_graph_artifacts",
        lambda **kwargs: calls.append(kwargs) or {"artifact_count": 0},
    )
    tasks.prune_graph_artifacts_task.run()
    assert tasks.prune_graph_artifacts_task.priority == 9
    assert calls == [{"execute": True}]


@pytest.mark.django_db(transaction=True)
@database_required
@override_settings(KG_ARTIFACT_RETENTION_DAYS=30, KG_ARTIFACT_KEEP_SUPERSEDED=0)
def test_execute_prunes_only_terminal_graph_rows_and_preserves_sources(
    admin_user,
) -> None:
    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services.pruning import prune_graph_artifacts

    collection = Collection.objects.create(name=f"prune-{uuid.uuid4()}")
    text = "source rows survive graph retention"
    document = RawTextDocument(
        title="retained document",
        full_text=text,
        full_text_hash=RawTextDocument.hash_fn(text),
        collection=collection,
        ingested_by=admin_user,
        ingestion_complete=True,
    )
    RawTextDocument.objects.bulk_create([document])
    chunk = TextChunk.objects.create(
        doc_id=document.id,
        content=text,
        start_position=0,
        end_position=len(text),
        chunk_number=0,
        modality=TextChunk.Modality.TEXT,
        embedding=[0.0] * 1024,
    )
    artifact = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=document.id,
        build_key="a" * 64,
        build_generation=1,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        status=GraphArtifact.Status.FAILED,
        source_hash=document.full_text_hash,
        ontology_version="research-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="unfiltered-v1",
        completed_at=timezone.now() - timedelta(days=31),
    )
    older_run = GraphBuildRun.objects.create(
        artifact=artifact,
        stage=GraphBuildRun.Stage.FAILED,
        status=GraphBuildRun.Status.FAILED,
        attempt=1,
        finished_at=timezone.now() - timedelta(days=31),
    )
    newest_artifact = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=document.id,
        build_key="b" * 64,
        build_generation=2,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        status=GraphArtifact.Status.FAILED,
        source_hash=document.full_text_hash,
        ontology_version="research-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="unfiltered-v1",
        completed_at=timezone.now() - timedelta(days=31),
    )
    newest_run = GraphBuildRun.objects.create(
        artifact=newest_artifact,
        stage=GraphBuildRun.Stage.FAILED,
        status=GraphBuildRun.Status.FAILED,
        attempt=1,
        finished_at=timezone.now() - timedelta(days=31),
    )
    preview = prune_graph_artifacts(execute=False, batch_size=10)
    report = prune_graph_artifacts(execute=True, batch_size=10)

    assert (
        report.artifact_ids
        == preview.artifact_ids
        == tuple(sorted((artifact.pk, newest_artifact.pk)))
    )
    assert report.run_ids == preview.run_ids == (older_run.pk,)
    assert not GraphArtifact.objects.filter(
        pk__in=(artifact.pk, newest_artifact.pk)
    ).exists()
    assert not GraphBuildRun.objects.filter(pk=older_run.pk).exists()
    newest_run.refresh_from_db()
    assert newest_run.artifact_id is None
    assert RawTextDocument.objects.filter(pk=document.pk).exists()
    assert TextChunk.objects.filter(pk=chunk.pk).exists()

    # Direct and duplicate service delivery are idempotent, while the newest
    # detached occurrence remains the generation allocator's high-water mark.
    repeated = prune_graph_artifacts(execute=True, batch_size=10)
    assert repeated.artifact_ids == ()
    assert repeated.run_ids == ()
    assert GraphBuildRun.objects.filter(pk=newest_run.pk).exists()


@pytest.mark.django_db(transaction=True)
@database_required
@override_settings(KG_ARTIFACT_RETENTION_DAYS=30, KG_ARTIFACT_KEEP_SUPERSEDED=2)
def test_retention_keeps_newest_superseded_generations_and_recent_terminal(
    admin_user,
) -> None:
    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services.pruning import prune_graph_artifacts

    collection = Collection.objects.create(name=f"retention-order-{uuid.uuid4()}")
    text = "generation ordered graph retention"
    document = RawTextDocument(
        title="retention generations",
        full_text=text,
        full_text_hash=RawTextDocument.hash_fn(text),
        collection=collection,
        ingested_by=admin_user,
        ingestion_complete=True,
    )
    RawTextDocument.objects.bulk_create([document])
    old = timezone.now() - timedelta(days=31)
    artifacts = []
    runs = []
    for generation in (1, 2, 3):
        artifact = GraphArtifact.objects.create(
            scope_type=GraphArtifact.ScopeType.DOCUMENT,
            scope_id=document.id,
            build_key=f"{generation:064x}",
            build_generation=generation,
            orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
            status=GraphArtifact.Status.SUPERSEDED,
            source_hash=document.full_text_hash,
            ontology_version="research-v1",
            extractor_version="extractor-v1",
            resolver_version="resolver-v1",
            filter_policy_version="unfiltered-v1",
            completed_at=old,
            superseded_at=old,
        )
        run = GraphBuildRun.objects.create(
            artifact=artifact,
            stage=GraphBuildRun.Stage.SUPERSEDED,
            status=GraphBuildRun.Status.CANCELLED,
            attempt=1,
            finished_at=old,
        )
        artifacts.append(artifact)
        runs.append(run)
    recent = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=document.id,
        build_key=f"{4:064x}",
        build_generation=4,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        status=GraphArtifact.Status.FAILED,
        source_hash=document.full_text_hash,
        ontology_version="research-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="unfiltered-v1",
        completed_at=timezone.now(),
    )
    recent_run = GraphBuildRun.objects.create(
        artifact=recent,
        stage=GraphBuildRun.Stage.FAILED,
        status=GraphBuildRun.Status.FAILED,
        attempt=1,
        finished_at=old,
    )

    preview = prune_graph_artifacts(execute=False, batch_size=10)

    assert preview.artifact_ids == (artifacts[0].pk,)
    assert preview.run_ids == (runs[0].pk,)
    assert recent.pk not in preview.artifact_ids
    assert recent_run.pk not in preview.run_ids


@pytest.mark.django_db(transaction=True)
@database_required
@override_settings(KG_ARTIFACT_RETENTION_DAYS=30, KG_ARTIFACT_KEEP_SUPERSEDED=0)
def test_terminal_success_activation_audit_survives_artifact_pruning() -> None:
    from django.contrib.auth import get_user_model

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.models import (
        GraphArtifact,
        GraphBuildRun,
        GraphRebuildRequest,
    )
    from apps.knowledge_graph.models.artifacts import _activation_audit_values
    from apps.knowledge_graph.services.pruning import prune_graph_artifacts

    user = get_user_model().objects.create_user(username=f"kg-pruning-{uuid.uuid4()}")
    collection = Collection.objects.create(name=f"retained-request-{uuid.uuid4()}")
    text = "successful rebuild audit remains inspectable"
    document = RawTextDocument(
        title="successful rebuild artifact",
        full_text=text,
        full_text_hash=RawTextDocument.hash_fn(text),
        collection=collection,
        ingested_by=user,
        ingestion_complete=True,
    )
    RawTextDocument.objects.bulk_create([document])
    old = timezone.now() - timedelta(days=31)
    request = GraphRebuildRequest.objects.create(
        scope_type=GraphRebuildRequest.ScopeType.DOCUMENT,
        scope_id=document.id,
        requested_documents=[
            {
                "document_id": str(document.id),
                "document_pkid": document.pk,
                "model_label": document._meta.label_lower,
                "collection_id": collection.pk,
                "source_hash": document.full_text_hash,
            }
        ],
        status=GraphRebuildRequest.Status.RUNNING,
        document_count=1,
        started_at=old,
    )
    artifact = GraphArtifact.objects.create(
        rebuild_request=request,
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=document.id,
        build_key="a" * 64,
        build_generation=1,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        status=GraphArtifact.Status.SUPERSEDED,
        source_hash=document.full_text_hash,
        ontology_version="research-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="unfiltered-v1",
        activated_at=old,
        completed_at=old,
        superseded_at=old,
    )
    run = GraphBuildRun.objects.create(
        artifact=artifact,
        rebuild_request=request,
        stage=GraphBuildRun.Stage.SUPERSEDED,
        status=GraphBuildRun.Status.CANCELLED,
        stage_marker={"stage_sequence": ["active", "superseded"]},
        attempt=1,
        finished_at=old,
    )
    request.status = GraphRebuildRequest.Status.SUCCEEDED
    request.completed_document_count = 1
    for field_name, value in _activation_audit_values(artifact, run).items():
        setattr(request, field_name, value)
    request.completed_at = old
    request.save()

    preview = prune_graph_artifacts(execute=False, batch_size=10)
    report = prune_graph_artifacts(execute=True, batch_size=10)

    assert artifact.pk in preview.artifact_ids
    assert artifact.pk in report.artifact_ids
    assert not GraphArtifact.objects.filter(pk=artifact.pk).exists()
    request.refresh_from_db()
    assert request.activated_artifact_pk == artifact.pk
    assert request.activated_run_pk == run.pk
    assert request.activated_build_key == artifact.build_key
    assert request.activated_build_generation == artifact.build_generation
    assert request.activated_source_hash == artifact.source_hash
    assert len(request.activated_occurrence_signature) == 64
