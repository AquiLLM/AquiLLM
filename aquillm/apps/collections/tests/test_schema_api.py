from __future__ import annotations

import json

import pytest
from django.contrib.auth.models import User
from django.urls import reverse

from apps.collections.models import (
    Collection,
    CollectionPermission,
    CollectionSchemaDraft,
    CollectionSchemaGenerationRun,
    CollectionSchemaVersion,
)
from apps.knowledge_graph.models import OntologyVersion


def _entity(key: str, description: str | None = None) -> dict:
    return {
        "key": key,
        "origin": "collection",
        "change_state": "added",
        "capabilities": {
            "editable_fields": [
                "description",
                "aliases",
                "default_retrieval_weight",
                "default_suppression_policy",
                "default_suppression_threshold",
            ],
            "removable": True,
            "renameable": False,
        },
        "values": {
            "name": key,
            "description": description or f"A {key}.",
            "aliases": [],
            "default_retrieval_weight": 1.0,
            "default_suppression_policy": "never",
            "default_suppression_threshold": 0.0,
        },
    }


def _relation(key: str = "authored_by") -> dict:
    return {
        "key": key,
        "origin": "collection",
        "change_state": "added",
        "capabilities": {
            "editable_fields": [
                "description",
                "direction",
                "allowed_head_types",
                "allowed_tail_types",
            ],
            "removable": True,
            "renameable": False,
        },
        "values": {
            "name": key,
            "description": "Connects a paper to an author.",
            "direction": "directed",
            "allowed_head_types": ["paper"],
            "allowed_tail_types": ["author"],
        },
    }


def _definitions() -> dict:
    return {
        "entities": [_entity("paper"), _entity("author")],
        "relations": [_relation()],
    }


def _ontology_record(version: str, checksum: str) -> OntologyVersion:
    return OntologyVersion.objects.create(
        kind=OntologyVersion.Kind.GRAPH,
        version=version,
        checksum=checksum,
        metadata={},
    )


@pytest.fixture
def schema_users(db):
    viewer = User.objects.create_user(username="schema-viewer")
    editor = User.objects.create_user(username="schema-editor")
    manager = User.objects.create_user(username="schema-manager")
    collection = Collection.objects.create(name="API schema collection")
    CollectionPermission.objects.create(
        user=viewer, collection=collection, permission="VIEW"
    )
    CollectionPermission.objects.create(
        user=editor, collection=collection, permission="EDIT"
    )
    CollectionPermission.objects.create(
        user=manager, collection=collection, permission="MANAGE"
    )
    return collection, viewer, editor, manager


def _request(client, method: str, url: str, *, body=None, revision=None):
    headers = {} if revision is None else {"HTTP_IF_MATCH": str(revision)}
    return getattr(client, method)(
        url,
        data=json.dumps(body) if body is not None else None,
        content_type="application/json",
        **headers,
    )


@pytest.mark.django_db
@pytest.mark.parametrize("method", ["put", "delete"])
def test_replacement_draft_rejects_previous_uuid_at_equal_revision(
    client, schema_users, method
):
    collection, _viewer, editor, _manager = schema_users
    old = CollectionSchemaDraft.objects.create(
        collection=collection, definitions=_definitions(), last_editor=editor
    )
    old_id = str(old.pk)
    old.delete()
    replacement = CollectionSchemaDraft.objects.create(
        collection=collection, definitions=_definitions(), last_editor=editor
    )
    client.force_login(editor)
    response = _request(
        client,
        method,
        reverse(
            "api_collection_schema_entity",
            kwargs={
                "col_id": collection.pk,
                "entity_key": "paper",
            },
        ),
        body={"draft_id": old_id, "values": _entity("paper", "stale")["values"]},
        revision=1,
    )
    assert response.status_code == 409
    assert response.json()["draft_id"] == str(replacement.pk)
    replacement.refresh_from_db()
    assert replacement.revision == 1
    assert replacement.definitions == _definitions()


@pytest.mark.django_db
def test_expired_generation_can_be_restarted_without_reviving_old_worker(
    client,
    schema_users,
    monkeypatch,
    django_capture_on_commit_callbacks,
):
    import uuid
    from datetime import timedelta

    from django.utils import timezone

    from apps.collections.services.schema_generation import collection_source_signature
    from apps.collections.tasks.schema_generation import _claim_run
    from apps.collections.views import schema_api

    collection, _viewer, editor, _manager = schema_users
    expired = CollectionSchemaGenerationRun.objects.create(
        collection=collection,
        requested_by=editor,
        status="running",
        source_signature=collection_source_signature(collection.pk),
        lease_token=uuid.uuid4(),
        lease_expires_at=timezone.now() - timedelta(seconds=1),
    )
    published = []
    monkeypatch.setattr(schema_api, "enqueue_schema_generation", published.append)
    client.force_login(editor)
    with django_capture_on_commit_callbacks(execute=True):
        response = _request(
            client,
            "post",
            reverse("api_collection_schema_generate", kwargs={"col_id": collection.pk}),
            body={},
        )
    assert response.status_code == 202
    assert response.json()["run_id"] != str(expired.pk)
    expired.refresh_from_db()
    assert expired.status == "failed"
    assert expired.lease_token is None
    assert _claim_run(expired.pk) is None
    assert published == [response.json()["run_id"]]


def _validate_and_publish(client, collection, draft):
    validation = _request(
        client,
        "post",
        reverse("api_collection_schema_validate", kwargs={"col_id": collection.pk}),
        body={"draft_id": str(draft.pk), "revision": draft.revision},
    )
    assert validation.status_code == 200
    identity = validation.json()["identity"]
    response = _request(
        client,
        "post",
        reverse("api_collection_schema_publish", kwargs={"col_id": collection.pk}),
        body={
            "draft_id": str(draft.pk),
            "revision": draft.revision,
            "candidate_checksum": identity["candidate_checksum"],
            "validation_result_id": identity["result_id"],
        },
        revision=draft.revision,
    )
    assert response.status_code == 200, response.json()
    return response


@pytest.mark.django_db(transaction=True)
def test_schema_generation_start_is_idempotent_for_same_source(
    client, schema_users, monkeypatch
):
    collection, _viewer, editor, _manager = schema_users
    client.force_login(editor)
    enqueued = []
    monkeypatch.setattr(
        "apps.collections.views.schema_api._locked_collection_source_signature",
        lambda collection_id: "a" * 64,
    )
    monkeypatch.setattr(
        "apps.collections.views.schema_api.enqueue_schema_generation",
        enqueued.append,
    )
    url = reverse("api_collection_schema_generate", kwargs={"col_id": collection.pk})

    first = _request(client, "post", url, body={})
    second = _request(client, "post", url, body={})

    assert first.status_code == second.status_code == 202
    assert first.json()["run_id"] == second.json()["run_id"]
    assert first.json()["status"] == "queued"
    assert first.json()["status_url"] == reverse(
        "api_collection_schema_generation_status",
        kwargs={"col_id": collection.pk, "run_id": first.json()["run_id"]},
    )
    assert CollectionSchemaGenerationRun.objects.count() == 1
    assert enqueued == [first.json()["run_id"]]


@pytest.mark.django_db(transaction=True)
def test_schema_generation_broker_failure_marks_run_retryable(
    client, schema_users, monkeypatch, caplog
):
    collection, _viewer, editor, _manager = schema_users
    client.force_login(editor)
    monkeypatch.setattr(
        "apps.collections.views.schema_api._locked_collection_source_signature",
        lambda collection_id: "a" * 64,
    )
    attempts = []

    def enqueue(run_id):
        attempts.append(run_id)
        if len(attempts) == 1:
            raise ConnectionError("private broker detail")

    monkeypatch.setattr(
        "apps.collections.views.schema_api.enqueue_schema_generation",
        enqueue,
    )
    url = reverse("api_collection_schema_generate", kwargs={"col_id": collection.pk})

    first = _request(client, "post", url, body={})
    first_run = CollectionSchemaGenerationRun.objects.get(pk=first.json()["run_id"])
    second = _request(client, "post", url, body={})

    assert first.status_code == second.status_code == 202
    assert first_run.status == CollectionSchemaGenerationRun.Status.FAILED
    assert first_run.error_code == "local_inference_failed"
    assert first.json()["run_id"] != second.json()["run_id"]
    assert attempts == [first.json()["run_id"], second.json()["run_id"]]
    assert "private broker detail" not in caplog.text
