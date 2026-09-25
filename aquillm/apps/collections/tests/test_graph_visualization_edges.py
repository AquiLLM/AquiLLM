from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone

from apps.collections.models import Collection, CollectionPermission
from apps.knowledge_graph.models import GraphRebuildRequest

from apps.collections.tests.test_graph_visualization_api import (
    graph_api_users,
    _url,
    _rebuild_url,
    _collection_attempt,
    _partial_request,
)


@pytest.mark.django_db
@pytest.mark.parametrize("newer_source_current", [True, False])
def test_collection_activity_prefers_scope_generation_over_old_key_attempt(
    client, graph_api_users, newer_source_current
):
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    collection, viewer, _editor, _outsider = graph_api_users
    request = _partial_request(collection)
    _document, _source, _ontology, artifact, old_run = _collection_attempt(
        collection, viewer
    )
    GraphBuildRun.objects.filter(pk=old_run.pk).update(attempt=99)
    values = {
        field.attname: getattr(artifact, field.attname)
        for field in GraphArtifact._meta.concrete_fields
        if not field.primary_key and not field.auto_created
    }
    values.update(
        build_generation=artifact.build_generation + 1,
        build_key="b" * 64,
        source_hash=artifact.source_hash if newer_source_current else "f" * 64,
    )
    newer = GraphArtifact.objects.create(**values)
    newer_run = GraphBuildRun.objects.create(
        artifact=newer,
        status="running",
        stage="resolving",
        started_at=timezone.now(),
        lease_owner="new-worker",
        lease_generation=1,
        lease_expires_at=timezone.now() + timedelta(minutes=10),
    )
    client.force_login(viewer)

    payload = client.get(_url(collection)).json()

    assert payload["status"]["request_id"] == str(request.pk)
    if newer_source_current:
        assert payload["status"]["state"] == "building"
        assert payload["status"]["updated_at"] == newer_run.started_at.isoformat()
    else:
        # The newer building generation also fences activation of the older run.
        assert payload["status"]["state"] == "partial"


@pytest.mark.django_db
def test_new_queued_request_takes_precedence_over_old_collection_failure(
    client,
    graph_api_users,
):
    collection, viewer, _editor, _outsider = graph_api_users
    _collection_attempt(collection, viewer, failed=True)
    request = GraphRebuildRequest.objects.create(
        scope_type="collection",
        scope_id=str(collection.pk),
        requested_documents=[],
        collection_count=1,
    )
    client.force_login(viewer)

    payload = client.get(_url(collection)).json()

    assert payload["status"]["state"] == "building"
    assert payload["status"]["error_code"] is None
    assert payload["status"]["request_id"] == str(request.pk)


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


@pytest.mark.django_db(transaction=True)
def test_graph_visualization_prioritizes_relation_endpoints_over_isolated_nodes(
    client, monkeypatch
):
    from apps.collections.services import graph_visualization
    from apps.knowledge_graph.models import CollectionEntity, GraphArtifact
    from apps.knowledge_graph.tests.test_models import (
        _persist_collection_relation_fixture,
    )

    fixture = _persist_collection_relation_fixture()
    artifact = fixture.collection_artifact
    isolated = CollectionEntity.objects.create(
        artifact=artifact,
        collection=artifact.collection_scope,
        cluster_key="f" * 64,
        label="High utility isolated concept",
        normalized_label="high utility isolated concept",
        entity_type="concept",
        extraction_confidence=1.0,
        resolution_confidence=1.0,
        retrieval_utility=1.0,
        promotion_confidence=1.0,
    )
    artifact.status = GraphArtifact.Status.ACTIVE
    artifact.save(update_fields=["status"])
    viewer = User.objects.create_user(username="graph-connected-viewer")
    CollectionPermission.objects.create(
        user=viewer, collection=artifact.collection_scope, permission="VIEW"
    )
    monkeypatch.setattr(graph_visualization, "NODE_LIMIT", 2)
    client.force_login(viewer)

    payload = client.get(_url(artifact.collection_scope)).json()

    assert {node["id"] for node in payload["nodes"]} == {
        f"entity:{fixture.relation.source_id}",
        f"entity:{fixture.relation.target_id}",
    }
    assert f"entity:{isolated.pk}" not in {node["id"] for node in payload["nodes"]}
    assert [edge["id"] for edge in payload["edges"]] == [
        f"relation:{fixture.relation.pk}"
    ]
    assert payload["truncated"] == {"nodes": True, "edges": False}


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
