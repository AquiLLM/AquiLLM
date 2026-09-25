from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone

from apps.collections.models import Collection, CollectionPermission
from apps.knowledge_graph.models import GraphRebuildRequest


@pytest.fixture
def graph_api_users(db):
    viewer = User.objects.create_user(username="graph-viewer")
    editor = User.objects.create_user(username="graph-editor")
    outsider = User.objects.create_user(username="graph-outsider")
    collection = Collection.objects.create(name="Graph API collection")
    CollectionPermission.objects.create(
        user=viewer, collection=collection, permission="VIEW"
    )
    CollectionPermission.objects.create(
        user=editor, collection=collection, permission="EDIT"
    )
    return collection, viewer, editor, outsider


def _url(collection):
    return reverse(
        "api_collection_graph_visualization", kwargs={"col_id": collection.pk}
    )


def _rebuild_url(collection):
    return reverse("api_collection_graph_rebuild", kwargs={"col_id": collection.pk})


@pytest.mark.django_db
def test_graph_visualization_requires_login(client):
    collection = Collection.objects.create(name="Private graph")

    response = client.get(_url(collection))

    assert response.status_code == 302


@pytest.mark.django_db
def test_graph_visualization_enforces_collection_view_permission(
    client, graph_api_users
):
    collection, viewer, _editor, outsider = graph_api_users
    client.force_login(outsider)
    assert client.get(_url(collection)).status_code == 403

    client.force_login(viewer)
    response = client.get(_url(collection))

    assert response.status_code == 200
    assert response.json()["status"] == {
        "state": "empty",
        "error_code": None,
        "request_id": None,
        "updated_at": None,
    }
    assert response.json()["nodes"] == []
    assert response.json()["edges"] == []
    assert response.json()["permissions"] == {"can_rebuild": False}


@pytest.mark.django_db
def test_graph_visualization_reports_latest_partial_build(client, graph_api_users):
    collection, viewer, _editor, _outsider = graph_api_users
    completed_at = timezone.now()
    request = GraphRebuildRequest.objects.create(
        scope_type=GraphRebuildRequest.ScopeType.COLLECTION,
        scope_id=str(collection.pk),
        requested_documents=[],
        status=GraphRebuildRequest.Status.PARTIAL,
        collection_count=1,
        failed_collection_count=1,
        error_code="task_terminal_failure",
        started_at=completed_at,
        completed_at=completed_at,
    )
    client.force_login(viewer)

    response = client.get(_url(collection))

    assert response.status_code == 200
    assert response.json()["status"]["state"] == "partial"
    assert response.json()["status"]["error_code"] == "task_terminal_failure"
    assert response.json()["status"]["request_id"] == str(request.pk)


def _collection_attempt(collection, user, *, failed=False, evaluation=False):
    from apps.collections.tests.test_graph_build_progress import _artifact, _document
    from apps.knowledge_graph.models import (
        GraphArtifact,
        GraphBuildRun,
        OntologyVersion,
    )
    from apps.knowledge_graph.models.inputs import (
        collection_input_source_signature,
        collection_manifest_source_hash,
        document_membership_signature,
    )

    document = _document(collection, user, "Current source")
    source = _artifact(document, "active")
    ontology = OntologyVersion.objects.create(
        kind="graph",
        version=source.ontology_version,
        checksum=source.ontology_checksum,
        status="active",
        metadata={"collection_id": collection.pk},
    )
    source_hash = collection_manifest_source_hash(
        [
            collection_input_source_signature(
                collection_id=collection.pk,
                document_id=document.id,
                document_artifact=source,
                membership_signature=document_membership_signature(document),
            )
        ]
    )
    request = None
    if evaluation:
        request = GraphRebuildRequest.objects.create(
            scope_type="collection",
            scope_id=str(collection.pk),
            requested_documents=[],
            collection_count=1,
            expected_aggregate_signature=source_hash,
            evaluation_only=True,
        )
    artifact = GraphArtifact.objects.create(
        scope_type="collection",
        scope_id=str(collection.pk),
        collection_scope=collection,
        status="failed" if failed else "building",
        source_hash=source_hash,
        ontology_version=source.ontology_version,
        ontology_checksum=source.ontology_checksum,
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="pending-v1",
        embedding_model_signature=(
            f"local:model@revision:endpoint={'e' * 64}:dims=1024:"
            "prep=kg-entity-v1:max_chars=8192:batch=64"
        ),
        orchestration_version=1,
        build_key="a" * 64,
        evaluation_only=evaluation,
        rebuild_request=request,
    )
    run = GraphBuildRun.objects.create(
        artifact=artifact,
        status="failed" if failed else "running",
        stage="failed" if failed else "resolving",
        started_at=timezone.now(),
        finished_at=timezone.now() if failed else None,
        lease_owner="" if failed else "worker",
        lease_generation=0 if failed else 1,
        lease_expires_at=None if failed else timezone.now() + timedelta(minutes=10),
        error_code="private_provider_detail" if failed else "",
    )
    return document, source, ontology, artifact, run


def _partial_request(collection):
    return GraphRebuildRequest.objects.create(
        scope_type="collection",
        scope_id=str(collection.pk),
        requested_documents=[],
        status="partial",
        collection_count=1,
        failed_collection_count=1,
        error_code="task_terminal_failure",
        started_at=timezone.now(),
        completed_at=timezone.now(),
    )


@pytest.mark.django_db
def test_live_collection_retry_overrides_historical_partial_request(
    client, graph_api_users
):
    collection, viewer, _editor, _outsider = graph_api_users
    request = _partial_request(collection)
    _document, _source, _ontology, _artifact, run = _collection_attempt(
        collection, viewer
    )
    client.force_login(viewer)

    payload = client.get(_url(collection)).json()

    assert payload["progress"]["active"] == payload["progress"]["total"] == 1
    assert payload["status"] == {
        "state": "building",
        "error_code": None,
        "request_id": str(request.pk),
        "updated_at": run.started_at.isoformat(),
    }
    assert payload["nodes"] == payload["edges"] == []
    request.refresh_from_db()
    assert request.status == "partial"
    assert request.error_code == "task_terminal_failure"


@pytest.mark.django_db
@pytest.mark.parametrize(
    "invalid", ["source", "membership", "ontology", "evaluation", "expired", "owner"]
)
def test_ineligible_collection_retry_does_not_override_request(
    client, graph_api_users, invalid
):
    collection, viewer, _editor, _outsider = graph_api_users
    request = _partial_request(collection)
    document, _source, ontology, _artifact, run = _collection_attempt(
        collection,
        viewer,
        evaluation=invalid == "evaluation",
    )
    if invalid == "source":
        type(document).objects.filter(pk=document.pk).update(full_text_hash="f" * 64)
    elif invalid == "membership":
        other = Collection.objects.create(name="Moved document")
        type(document).objects.filter(pk=document.pk).update(collection=other)
    elif invalid == "ontology":
        ontology.status = "superseded"
        ontology.save(update_fields=["status"])
    elif invalid == "expired":
        run.lease_expires_at = timezone.now() - timedelta(seconds=1)
        run.save(update_fields=["lease_expires_at"])
    elif invalid == "owner":
        run.lease_owner = ""
        run.lease_expires_at = None
        run.save(update_fields=["lease_owner", "lease_expires_at"])
    client.force_login(viewer)

    payload = client.get(_url(collection)).json()

    assert payload["status"]["state"] == "partial"
    assert payload["status"]["request_id"] == str(request.pk)


@pytest.mark.django_db
def test_automatic_collection_failure_is_terminal_with_all_documents_active(
    client, graph_api_users
):
    collection, viewer, _editor, _outsider = graph_api_users
    _collection_attempt(collection, viewer, failed=True)
    client.force_login(viewer)

    payload = client.get(_url(collection)).json()

    assert payload["progress"]["active"] == payload["progress"]["total"] == 1
    assert payload["status"]["state"] == "failed"
    assert payload["status"]["error_code"] == "collection_build_failed"
    assert "private_provider_detail" not in str(payload)
