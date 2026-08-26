from __future__ import annotations

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


@pytest.mark.django_db(transaction=True)
def test_graph_visualization_returns_current_nodes_edges_and_bounded_evidence(client):
    from apps.knowledge_graph.models import GraphArtifact
    from apps.knowledge_graph.tests.test_models import (
        _persist_collection_relation_fixture,
    )

    fixture = _persist_collection_relation_fixture()
    artifact = fixture.collection_artifact
    artifact.status = GraphArtifact.Status.ACTIVE
    artifact.save(update_fields=["status"])
    collection = artifact.collection_scope
    viewer = User.objects.create_user(username="graph-ready-viewer")
    CollectionPermission.objects.create(
        user=viewer, collection=collection, permission="VIEW"
    )
    client.force_login(viewer)

    response = client.get(_url(collection))

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"]["state"] == "ready"
    assert payload["artifact_id"] == str(artifact.pk)
    assert [(node["label"], node["entity_type"]) for node in payload["nodes"]] == [
        ("Aquilla", "model"),
        ("MMLU", "benchmark"),
    ]
    assert payload["nodes"][0]["evidence"] == [
        {
            "document_id": str(fixture.relation_mention.document_id),
            "chunk_id": fixture.relation_mention.chunk_id,
            "start": fixture.relation_mention.head.start,
            "end": fixture.relation_mention.head.end,
            "excerpt": "Aquilla evaluates MMLU.",
        }
    ]
    assert payload["edges"] == [
        {
            "id": f"relation:{fixture.relation.pk}",
            "source": f"entity:{fixture.relation.source_id}",
            "target": f"entity:{fixture.relation.target_id}",
            "relation_type": "evaluates_on",
            "confidence": 0.8,
            "support_count": 1,
            "evidence": [
                {
                    "document_id": str(fixture.relation_mention.document_id),
                    "chunk_id": fixture.relation_mention.chunk_id,
                    "start": fixture.relation_mention.head.start,
                    "end": fixture.relation_mention.tail.end,
                    "excerpt": "Aquilla evaluates MMLU.",
                }
            ],
        }
    ]


@pytest.mark.django_db
def test_graph_rebuild_requires_edit_and_queues_existing_service(
    client, graph_api_users, monkeypatch
):
    collection, viewer, editor, _outsider = graph_api_users
    calls = []

    def create_rebuild_request(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            pk="11111111-1111-4111-8111-111111111111",
            status="queued",
        )

    monkeypatch.setattr(
        "apps.collections.views.graph_api.create_rebuild_request",
        create_rebuild_request,
    )
    client.force_login(viewer)
    assert client.post(_rebuild_url(collection)).status_code == 403

    client.force_login(editor)
    response = client.post(_rebuild_url(collection))

    assert response.status_code == 202
    assert response.json()["request_id"] == "11111111-1111-4111-8111-111111111111"
    assert calls == [{"scope_type": "collection", "scope_id": collection.pk}]
