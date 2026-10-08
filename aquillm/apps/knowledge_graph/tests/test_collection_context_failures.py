import uuid
from unittest.mock import Mock

import pytest
from django.utils import timezone


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize(
    ("failure_kind", "expected_code", "refreshes"),
    [
        ("capacity", "collection_context_invalid", 0),
        ("typed_capacity", "collection_entity_limit", 0),
        ("unexpected", "collection_context_failed", 0),
        ("stale", "resnapshot_pending", 1),
    ],
)
def test_collection_context_failure_only_resnapshots_stale_sources(
    monkeypatch, failure_kind, expected_code, refreshes
):
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds
    from apps.knowledge_graph.services.collection_context_policy import (
        CollectionCapacityError,
    )

    collection = Collection.objects.create(name="Context classification")
    request = GraphRebuildRequest.objects.create(
        id=uuid.uuid4(),
        scope_type="collection",
        scope_id=str(collection.pk),
        requested_documents=[],
        document_count=0,
        collection_count=1,
        document_publication_state="published",
        status="running",
        started_at=timezone.now(),
    )
    error = {
        "capacity": builds.CorruptBuildError("collection context entity cap exceeded"),
        "typed_capacity": CollectionCapacityError("entity"),
        "unexpected": ValueError("private provider detail must not be recorded"),
        "stale": builds.StaleBuildError(
            "collection awaits fresh document graph artifacts"
        ),
    }[failure_kind]
    monkeypatch.setattr(builds, "_collection_context", Mock(side_effect=error))
    successor = Mock(return_value=None)
    monkeypatch.setattr(builds, "_reconcile_rebuild_successor", successor)

    builds.advance_rebuild_request(request.pk)

    request.refresh_from_db()
    assert request.status == "partial"
    assert request.error_code == expected_code
    assert request.failed_collection_count == 1
    assert request.terminal_failure_count == 0
    assert request.completed_document_count == 0
    assert successor.call_count == refreshes
    assert GraphRebuildRequest.objects.count() == 1


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize(
    ("error_type", "expected_code", "refreshes"),
    [
        ("corrupt", "request_snapshot_invalid", 0),
        ("invalid", "request_snapshot_invalid", 0),
        ("capacity", "request_snapshot_failed", 0),
        ("stale", "resnapshot_pending", 1),
        ("missing", "resnapshot_pending", 1),
    ],
)
def test_completion_snapshot_errors_only_retry_real_source_changes(
    monkeypatch, error_type, expected_code, refreshes
):
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import GraphRebuildRequest
    from apps.knowledge_graph.services import builds

    collection = Collection.objects.create(name="Snapshot classification")
    request = GraphRebuildRequest.objects.create(
        scope_type="collection",
        scope_id=str(collection.pk),
        requested_documents=[],
        document_count=0,
        collection_count=1,
        document_publication_state="published",
        status="running",
        started_at=timezone.now(),
    )
    error = {
        "corrupt": builds.CorruptBuildError("request owns surplus document artifacts"),
        "invalid": ValueError("private source detail"),
        "capacity": RuntimeError("rebuild document snapshot exceeds its cap"),
        "stale": builds.StaleBuildError("snapshot changed"),
        "missing": LookupError("collection does not exist"),
    }[error_type]
    monkeypatch.setattr(
        builds, "_lock_request_completion_snapshot", Mock(side_effect=error)
    )
    successor = Mock()
    monkeypatch.setattr(builds, "_reconcile_rebuild_successor", successor)
    context = Mock(side_effect=AssertionError("invalid preflight reached content"))
    monkeypatch.setattr(builds, "_collection_context", context)

    builds.advance_rebuild_request(request.pk)

    request.refresh_from_db()
    assert request.status == "partial"
    assert request.error_code == expected_code
    assert request.failed_collection_count == 1
    assert successor.call_count == refreshes
    context.assert_not_called()
