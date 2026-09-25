from __future__ import annotations

import inspect
import os
import socket
import uuid
from datetime import timedelta
from types import SimpleNamespace

import pytest
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from apps.knowledge_graph.graph import assembly
from apps.knowledge_graph.models import (
    GraphArtifact,
    GraphBuildRun,
    GraphRebuildRequest,
)
from apps.knowledge_graph.models.inputs import collection_input_source_signature


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


def _embedding_signature() -> str:
    return (
        f"test-local:model@rev:endpoint={'e' * 64}:dims=1024:"
        "prep=kg-entity-v1:max_chars=8192:batch=64"
    )


def _collection_artifact(
    collection,
    *,
    source_digit: str,
    status: str,
    build_generation: int = 1,
    orchestration_version: int = GraphArtifact.OrchestrationVersion.LEGACY,
    rebuild_request=None,
    evaluation_only: bool = False,
    activated_at=None,
    completed_at=None,
    superseded_at=None,
):
    return GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=collection.pk,
        status=status,
        build_generation=build_generation,
        orchestration_version=orchestration_version,
        rebuild_request=rebuild_request,
        evaluation_only=evaluation_only,
        source_hash=source_digit * 64,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="filter-v1",
        embedding_model_signature=_embedding_signature(),
        activated_at=activated_at,
        completed_at=completed_at,
        superseded_at=superseded_at,
    )


def test_document_source_signature_changes_for_every_immutable_input_identity_field():
    document_id = uuid.uuid4()
    values = {
        "pk": 17,
        "build_key": "0" * 64,
        "source_hash": "1" * 64,
        "ontology_version": "ontology-v1",
        "extractor_version": "extractor-v1",
        "resolver_version": "resolver-v1",
        "filter_policy_version": "filter-v1",
        "embedding_model_signature": "embed-v1",
        "ontology_checksum": "2" * 64,
        "filter_policy_checksum": "3" * 64,
        "resolution_config_checksum": "4" * 64,
        "assembly_version": "not-applicable",
        "assembly_config_checksum": "5" * 64,
    }

    def signature(**overrides):
        artifact = SimpleNamespace(**{**values, **overrides})
        return collection_input_source_signature(
            collection_id=23,
            document_id=document_id,
            document_artifact=artifact,
            membership_signature="6" * 64,
        )

    baseline = signature()
    changes = {
        "pk": 18,
        "build_key": "9" * 64,
        "source_hash": "a" * 64,
        "ontology_version": "ontology-v2",
        "extractor_version": "extractor-v2",
        "resolver_version": "resolver-v2",
        "filter_policy_version": "filter-v2",
        "embedding_model_signature": "embed-v2",
        "ontology_checksum": "b" * 64,
        "filter_policy_checksum": "c" * 64,
        "resolution_config_checksum": "d" * 64,
        "assembly_version": "other-assembly",
        "assembly_config_checksum": "e" * 64,
    }

    unchanged = {
        field
        for field, value in changes.items()
        if signature(**{field: value}) == baseline
    }
    assert unchanged == set()


def test_supersession_has_a_distinct_immutable_lifecycle_timestamp():
    field_names = {field.name for field in GraphArtifact._meta.fields}
    activation_source = inspect.getsource(assembly._swap_active_collection_artifact)

    assert "superseded_at" in field_names
    assert "superseded_at" in GraphArtifact._QUERYSET_IMMUTABLE_FIELDS
    assert "previous.completed_at =" not in activation_source
    assert "previous.superseded_at =" in activation_source

    with pytest.raises(ValidationError, match="immutable"):
        GraphArtifact.objects.filter(pk=1).update(superseded_at=timezone.now())


@pytest.mark.django_db(transaction=True)
@database_required
def test_persisted_supersession_timestamp_cannot_be_cleared_or_rewritten():
    from apps.collections.models import Collection

    collection = Collection.objects.create(name="immutable supersession")
    activated_at = timezone.now()
    superseded_at = timezone.now()
    artifact = _collection_artifact(
        collection,
        source_digit="1",
        status=GraphArtifact.Status.SUPERSEDED,
        activated_at=activated_at,
        completed_at=activated_at,
        superseded_at=superseded_at,
    )
    values = GraphArtifact._base_manager.filter(pk=artifact.pk).values().get()
    values["superseded_at"] = None
    replacement = GraphArtifact(**values)

    assert replacement._state.adding
    with pytest.raises(ValidationError, match="activation history"):
        replacement.save()

    artifact.refresh_from_db()
    assert artifact.superseded_at == superseded_at


@pytest.mark.django_db(transaction=True)
@database_required
def test_validation_failure_preserves_the_prior_active_artifact():
    from apps.collections.models import Collection

    collection = Collection.objects.create(name="validation rollback")
    completed_at = timezone.now()
    prior = _collection_artifact(
        collection,
        source_digit="1",
        status=GraphArtifact.Status.ACTIVE,
        activated_at=completed_at,
        completed_at=completed_at,
    )
    candidate = _collection_artifact(
        collection,
        source_digit="2",
        status=GraphArtifact.Status.BUILDING,
    )
    run = GraphBuildRun.objects.create(
        artifact=candidate,
        stage=GraphBuildRun.Stage.PERSISTENCE,
        status=GraphBuildRun.Status.RUNNING,
    )

    with pytest.raises(assembly.CollectionGraphAssemblyError):
        assembly.activate_collection_graph(
            collection.pk,
            run.pk,
            candidate.source_hash,
        )

    prior.refresh_from_db()
    candidate.refresh_from_db()
    assert prior.status == GraphArtifact.Status.ACTIVE
    assert prior.completed_at == completed_at
    assert prior.superseded_at is None
    assert candidate.status == GraphArtifact.Status.BUILDING


@pytest.mark.django_db(transaction=True)
@database_required
@pytest.mark.parametrize(
    "source_status",
    (GraphArtifact.Status.STALE, GraphArtifact.Status.SUPERSEDED),
)
def test_stale_or_superseded_document_contributor_is_rejected(source_status):
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument

    user = User.objects.create_user(username=f"source-{source_status}")
    collection = Collection.objects.create(name=f"source {source_status}")
    text = "immutable contributor"
    document = RawTextDocument(
        title="Contributor",
        full_text=text,
        collection=collection,
        ingested_by=user,
        full_text_hash=RawTextDocument.hash_fn(text),
    )
    document.save(dont_rechunk=True)
    source = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=document.id,
        status=source_status,
        source_hash=document.full_text_hash,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="filter-v1",
    )
    manifest = (SimpleNamespace(document_artifact_id=source.pk),)

    with (
        transaction.atomic(),
        pytest.raises(assembly.CollectionGraphAssemblyError, match="snapshot changed"),
    ):
        assembly._validate_locked_manifest(
            collection,
            SimpleNamespace(),
            manifest,
            "a" * 64,
            assembly.AssemblyConfig(),
        )


@pytest.mark.django_db(transaction=True)
@database_required
def test_document_moved_to_another_collection_is_rejected_as_a_contributor():
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument

    user = User.objects.create_user(username="moved-source")
    original = Collection.objects.create(name="original")
    destination = Collection.objects.create(name="destination")
    text = "moving contributor"
    document = RawTextDocument(
        title="Contributor",
        full_text=text,
        collection=original,
        ingested_by=user,
        full_text_hash=RawTextDocument.hash_fn(text),
    )
    document.save(dont_rechunk=True)
    source = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=document.id,
        status=GraphArtifact.Status.ACTIVE,
        source_hash=document.full_text_hash,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="filter-v1",
    )
    RawTextDocument.objects.filter(pk=document.pk).update(collection=destination)
    manifest = (SimpleNamespace(document_artifact_id=source.pk),)

    with (
        transaction.atomic(),
        pytest.raises(assembly.CollectionGraphAssemblyError, match="snapshot changed"),
    ):
        assembly._validate_locked_manifest(
            original,
            SimpleNamespace(),
            manifest,
            "a" * 64,
            assembly.AssemblyConfig(),
        )


@pytest.mark.django_db(transaction=True)
@database_required
def test_newer_candidate_supersedes_without_rewriting_completion_and_is_idempotent():
    from apps.collections.models import Collection

    swap = getattr(assembly, "_swap_active_collection_artifact", None)
    assert callable(swap)
    collection = Collection.objects.create(name="newer wins")
    original_completed_at = timezone.now()
    prior = _collection_artifact(
        collection,
        source_digit="1",
        status=GraphArtifact.Status.ACTIVE,
        activated_at=original_completed_at,
        completed_at=original_completed_at,
    )
    candidate = _collection_artifact(
        collection,
        source_digit="2",
        status=GraphArtifact.Status.BUILDING,
    )
    run = GraphBuildRun.objects.create(
        artifact=candidate,
        stage=GraphBuildRun.Stage.PERSISTENCE,
        status=GraphBuildRun.Status.RUNNING,
    )

    with transaction.atomic():
        swap(artifact=candidate, run=run, scope_artifacts=(prior, candidate))

    prior.refresh_from_db()
    candidate.refresh_from_db()
    run.refresh_from_db()
    assert prior.status == GraphArtifact.Status.SUPERSEDED
    assert prior.completed_at == original_completed_at
    assert prior.superseded_at is not None
    assert candidate.status == GraphArtifact.Status.ACTIVE
    first_timestamps = (
        candidate.activated_at,
        candidate.completed_at,
        candidate.superseded_at,
    )

    with transaction.atomic():
        swap(artifact=candidate, run=run, scope_artifacts=(prior, candidate))
    candidate.refresh_from_db()
    assert (
        candidate.activated_at,
        candidate.completed_at,
        candidate.superseded_at,
    ) == first_timestamps


@pytest.mark.django_db(transaction=True)
@database_required
def test_older_candidate_cannot_activate_after_a_newer_winner():
    from apps.collections.models import Collection

    swap = getattr(assembly, "_swap_active_collection_artifact", None)
    assert callable(swap)
    collection = Collection.objects.create(name="older loses")
    candidate = _collection_artifact(
        collection,
        source_digit="1",
        status=GraphArtifact.Status.BUILDING,
        build_generation=1,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
    )
    won_at = timezone.now()
    newer = _collection_artifact(
        collection,
        source_digit="2",
        status=GraphArtifact.Status.SUPERSEDED,
        build_generation=2,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        activated_at=won_at,
        completed_at=won_at,
    )
    run = GraphBuildRun.objects.create(
        artifact=candidate,
        stage=GraphBuildRun.Stage.VALIDATING,
        status=GraphBuildRun.Status.RUNNING,
    )

    with (
        transaction.atomic(),
        pytest.raises(assembly.CollectionGraphAssemblyError, match="newer"),
    ):
        swap(artifact=candidate, run=run, scope_artifacts=(candidate, newer))

    candidate.refresh_from_db()
    assert candidate.status == GraphArtifact.Status.BUILDING


@pytest.mark.django_db(transaction=True)
@database_required
def test_candidate_fence_cannot_be_crowded_by_evaluation_occurrences():
    from apps.collections.models import Collection

    collection = Collection.objects.create(name="evaluation does not crowd production")
    candidate = _collection_artifact(
        collection,
        source_digit="1",
        status=GraphArtifact.Status.BUILDING,
        build_generation=1,
    )
    run = GraphBuildRun.objects.create(
        artifact=candidate,
        stage=GraphBuildRun.Stage.PERSISTENCE,
        status=GraphBuildRun.Status.RUNNING,
    )
    for generation, digit in ((2, "2"), (3, "3")):
        request = GraphRebuildRequest.objects.create(
            scope_type=GraphRebuildRequest.ScopeType.COLLECTION,
            scope_id=str(collection.pk),
            requested_documents=[],
            expected_aggregate_signature=digit * 64,
            evaluation_only=True,
            collection_count=1,
        )
        _collection_artifact(
            collection,
            source_digit=digit,
            status=GraphArtifact.Status.BUILDING,
            build_generation=generation,
            rebuild_request=request,
            evaluation_only=True,
        )
    won_at = timezone.now()
    blocker = _collection_artifact(
        collection,
        source_digit="4",
        status=GraphArtifact.Status.SUPERSEDED,
        build_generation=4,
        activated_at=won_at,
        completed_at=won_at,
    )
    failed = _collection_artifact(
        collection,
        source_digit="5",
        status=GraphArtifact.Status.FAILED,
        build_generation=5,
    )

    with transaction.atomic():
        _collection, locked_candidate, _run, scope_artifacts = (
            assembly._locked_candidate(collection.pk, run.pk)
        )

    assert locked_candidate.pk == candidate.pk
    assert tuple(row.pk for row in scope_artifacts) == (candidate.pk, blocker.pk)
    assert assembly._newer_activation_exists(locked_candidate, scope_artifacts)
    assert not assembly._newer_activation_exists(locked_candidate, (failed,))


@pytest.mark.django_db(transaction=True)
@database_required
def test_concurrent_older_and_newer_candidates_leave_the_newer_active(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from django.db import close_old_connections

    from apps.collections.models import Collection

    collection = Collection.objects.create(name="concurrent newer wins")
    older = _collection_artifact(
        collection,
        source_digit="1",
        status=GraphArtifact.Status.BUILDING,
        build_generation=1,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
    )
    newer = _collection_artifact(
        collection,
        source_digit="2",
        status=GraphArtifact.Status.BUILDING,
        build_generation=2,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
    )
    runs = tuple(
        GraphBuildRun.objects.create(
            artifact=artifact,
            stage=GraphBuildRun.Stage.VALIDATING,
            status=GraphBuildRun.Status.RUNNING,
            lease_owner=f"activation-worker-{artifact.pk}",
            lease_generation=1,
            lease_expires_at=timezone.now() + timedelta(minutes=5),
        )
        for artifact in (older, newer)
    )
    barrier = Barrier(2)

    monkeypatch.setattr(
        assembly,
        "_validate_locked_complete_artifact",
        lambda **_kwargs: object(),
    )

    def activate(run, source_hash):
        close_old_connections()
        try:
            barrier.wait(timeout=10)
            assembly.activate_collection_graph(
                collection.pk,
                run.pk,
                source_hash,
                lease_owner=run.lease_owner,
                lease_generation=run.lease_generation,
            )
            return "activated"
        except assembly.CollectionGraphAssemblyError:
            return "newer_already_won"
        finally:
            close_old_connections()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(
            executor.submit(activate, run, artifact.source_hash)
            for run, artifact in zip(runs, (older, newer), strict=True)
        )
        outcomes = tuple(future.result(timeout=30) for future in futures)

    older.refresh_from_db()
    newer.refresh_from_db()
    assert "activated" in outcomes
    assert newer.status == GraphArtifact.Status.ACTIVE
    assert older.status in {
        GraphArtifact.Status.BUILDING,
        GraphArtifact.Status.SUPERSEDED,
    }


@pytest.mark.django_db(transaction=True)
@database_required
def test_injected_swap_failure_rolls_back_prior_supersession(monkeypatch):
    from apps.collections.models import Collection

    swap = getattr(assembly, "_swap_active_collection_artifact", None)
    assert callable(swap)
    collection = Collection.objects.create(name="swap rollback")
    original_completed_at = timezone.now()
    prior = _collection_artifact(
        collection,
        source_digit="1",
        status=GraphArtifact.Status.ACTIVE,
        activated_at=original_completed_at,
        completed_at=original_completed_at,
    )
    candidate = _collection_artifact(
        collection,
        source_digit="2",
        status=GraphArtifact.Status.BUILDING,
    )
    run = GraphBuildRun.objects.create(
        artifact=candidate,
        stage=GraphBuildRun.Stage.PERSISTENCE,
        status=GraphBuildRun.Status.RUNNING,
    )

    def fail_candidate_save(*_args, **_kwargs):
        raise RuntimeError("injected swap failure")

    monkeypatch.setattr(candidate, "save", fail_candidate_save)
    with pytest.raises(RuntimeError, match="injected swap failure"):
        with transaction.atomic():
            swap(artifact=candidate, run=run, scope_artifacts=(prior, candidate))

    prior.refresh_from_db()
    candidate.refresh_from_db()
    assert prior.status == GraphArtifact.Status.ACTIVE
    assert prior.completed_at == original_completed_at
    assert prior.superseded_at is None
    assert candidate.status == GraphArtifact.Status.BUILDING
