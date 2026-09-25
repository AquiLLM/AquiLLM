"""Current request lineage progress must not count replaced attempts twice."""

import uuid

import pytest
from django.utils import timezone

from apps.collections.models import Collection
from apps.knowledge_graph.models import (
    GraphArtifact,
    GraphBuildRun,
    GraphRebuildRequest,
)
from apps.knowledge_graph.models.artifacts import _activation_audit_values
from apps.knowledge_graph.services.inspection import inspect_graph_state

pytestmark = pytest.mark.django_db


def _request(*, scope_id="1", documents=0, status="running", **kwargs):
    Collection.objects.get_or_create(pk=int(scope_id), defaults={"name": "Progress"})
    now = timezone.now()
    snapshots = [
        {
            "document_id": str(uuid.UUID(int=index + 1, version=4)),
            "document_pkid": index + 1,
            "model_label": "apps_documents.rawtextdocument",
            "collection_id": int(scope_id),
            "source_hash": "a" * 64,
        }
        for index in range(documents)
    ]
    return GraphRebuildRequest.objects.create(
        scope_type="collection",
        scope_id=scope_id,
        requested_documents=snapshots,
        document_count=documents,
        collection_count=1,
        status=status,
        expected_aggregate_signature="a" * 64,
        started_at=now,
        completed_at=now if status in {"partial", "failed"} else None,
        document_publication_state="published",
        **kwargs,
    )


def _artifact(request, *, scope_type="document", status="active", generation=1):
    now = timezone.now()
    return GraphArtifact.objects.create(
        scope_type=scope_type,
        scope_id=(
            request.scope_id
            if scope_type == "collection"
            else request.requested_documents[0]["document_id"]
        ),
        status=status,
        source_hash="a" * 64,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="filter-v1",
        build_key=f"{generation:064x}",
        build_generation=generation,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        rebuild_request=request,
        activated_at=now,
        completed_at=now,
        superseded_at=now if status == "superseded" else None,
        embedding_model_signature=(
            "test:endpoint="
            + "b" * 64
            + ":dims=1024:prep=kg-entity-v1:max_chars=8192:batch=64"
            if scope_type == "collection"
            else ""
        ),
    )


def _succeed(request):
    artifact = _artifact(request, scope_type="collection")
    run = GraphBuildRun.objects.create(
        artifact=artifact,
        rebuild_request=request,
        build_kind="collection",
        stage="active",
        status="succeeded",
        attempt=1,
        finished_at=timezone.now(),
    )
    request.status = "succeeded"
    request.expected_aggregate_signature = "a" * 64
    request.completed_document_count = request.document_count
    request.completed_collection_count = 1
    request.completed_at = timezone.now()
    for field, value in _activation_audit_values(artifact, run).items():
        setattr(request, field, value)
    request.save()


def test_all_progress_uses_only_effective_child_leaves_and_current_artifacts():
    root = GraphRebuildRequest.objects.create(
        scope_type="all",
        status="partial",
        completed_at=timezone.now(),
        enumeration_high_water=5,
        enumeration_complete=True,
        expected_child_count=5,
        collection_count=5,
        completed_collection_count=1,
        failed_collection_count=4,
        document_publication_state="not_applicable",
    )
    original = _request(
        parent_request=root, documents=3, status="partial", completed_document_count=3
    )
    replacement = _request(
        predecessor_request=original,
        lineage_root=original,
        documents=3,
        status="partial",
        completed_document_count=3,
    )
    current = _request(
        predecessor_request=replacement,
        lineage_root=original,
        documents=4,
        completed_document_count=2,
        terminal_failure_count=1,
    )
    _request(parent_request=root, scope_id="2", documents=2, status="queued")
    _request(
        parent_request=root,
        scope_id="3",
        documents=1,
        status="partial",
        completed_document_count=1,
        error_code="resnapshot_pending",
    )
    _request(
        parent_request=root,
        scope_id="4",
        documents=2,
        status="failed",
        terminal_failure_count=2,
    )
    success = _request(parent_request=root, scope_id="5", documents=1)
    _succeed(success)
    _request(scope_id="99", documents=20)
    _artifact(original, status="superseded")
    _artifact(current, generation=2)

    result = inspect_graph_state(request_id=root.pk)

    assert result["status"] == "partial"
    assert result["document_count"] == 0
    assert result["failed_collection_count"] == 4
    assert result["progress"] == {
        "document_count": 10,
        "completed_document_count": 4,
        "failed_document_count": 3,
        "collection_count": 5,
        "collection_status_counts": {
            "queued": 1,
            "running": 1,
            "succeeded": 1,
            "partial": 1,
            "failed": 1,
        },
        "resnapshot_pending_count": 1,
        "resnapshot_churn_count": 0,
        "effective_request_count": 5,
        "enumeration_complete": True,
        "live_status": "running",
        "active_document_artifact_count": 1,
        "active_collection_artifact_count": 1,
        "historical_activation_count": 3,
    }


def test_scoped_progress_follows_successor_and_treats_resnapshot_as_pending():
    original = _request(documents=3, status="partial", completed_document_count=2)
    current = _request(
        predecessor_request=original,
        lineage_root=original,
        documents=1,
        status="partial",
        error_code="resnapshot_pending",
    )

    result = inspect_graph_state(request_id=original.pk)

    assert result["effective_request_id"] == str(current.pk)
    assert result["progress"]["document_count"] == 1
    assert result["progress"]["completed_document_count"] == 0
    assert result["progress"]["effective_request_count"] == 1
    assert result["progress"]["resnapshot_pending_count"] == 1
    assert result["progress"]["live_status"] == "pending"


def test_incomplete_enumeration_never_reports_completed_progress():
    root = GraphRebuildRequest.objects.create(
        scope_type="all",
        status="running",
        enumeration_high_water=5,
        document_publication_state="not_applicable",
    )

    result = inspect_graph_state(request_id=root.pk)

    assert result["progress"]["enumeration_complete"] is False
    assert result["progress"]["live_status"] == "running"
    assert result["progress"]["effective_request_count"] == 0


def test_unscoped_inspection_has_no_request_progress():
    assert inspect_graph_state()["progress"] is None


@pytest.mark.parametrize("scope", ["all", "collection"])
def test_reconcilable_churn_leaf_remains_pending_without_counting_predecessors(scope):
    root = GraphRebuildRequest.objects.create(
        scope_type="all",
        status="running",
        enumeration_high_water=1,
        enumeration_complete=True,
        expected_child_count=1,
        collection_count=1,
        document_publication_state="not_applicable",
    )
    original = _request(
        parent_request=root,
        documents=1,
        status="partial",
        error_code="resnapshot_churn",
    )
    _request(
        predecessor_request=original,
        lineage_root=original,
        documents=1,
        status="partial",
        error_code="resnapshot_churn",
    )

    progress = inspect_graph_state(
        request_id=root.pk if scope == "all" else original.pk
    )["progress"]

    assert progress["live_status"] == "pending"
    assert progress["resnapshot_pending_count"] == 1
    assert progress["resnapshot_churn_count"] == 1
    assert progress["effective_request_count"] == 1


@pytest.mark.parametrize(
    "status", ["queued", "running", "partial", "failed", "succeeded"]
)
def test_scoped_live_status_reports_current_outcome(status):
    request = _request(status="running" if status == "succeeded" else status)
    if status == "succeeded":
        _succeed(request)

    progress = inspect_graph_state(request_id=request.pk)["progress"]

    assert progress["live_status"] == status
    assert progress["collection_status_counts"][status] == 1
