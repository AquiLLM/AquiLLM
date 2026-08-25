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


@pytest.mark.django_db(transaction=True)
def test_schema_generation_adopts_unchanged_empty_draft(
    client, schema_users, monkeypatch
):
    collection, _viewer, editor, _manager = schema_users
    client.force_login(editor)
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions={"entities": [], "relations": []},
        last_editor=editor,
    )
    monkeypatch.setattr(
        "apps.collections.views.schema_api._locked_collection_source_signature",
        lambda collection_id: "a" * 64,
    )
    enqueued = []
    monkeypatch.setattr(
        "apps.collections.views.schema_api.enqueue_schema_generation",
        enqueued.append,
    )

    response = _request(
        client,
        "post",
        reverse("api_collection_schema_generate", kwargs={"col_id": collection.pk}),
        body={},
    )

    assert response.status_code == 202
    run = CollectionSchemaGenerationRun.objects.get(pk=response.json()["run_id"])
    assert run.base_draft_id == draft.pk
    assert run.base_draft_revision == draft.revision
    assert enqueued == [str(run.pk)]


@pytest.mark.django_db(transaction=True)
def test_schema_generation_rebinds_legacy_active_run_to_empty_draft(
    client, schema_users, monkeypatch
):
    collection, _viewer, editor, _manager = schema_users
    client.force_login(editor)
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions={"entities": [], "relations": []},
        last_editor=editor,
    )
    run = CollectionSchemaGenerationRun.objects.create(
        collection=collection,
        requested_by=editor,
        source_signature="a" * 64,
        base_draft_id=None,
        base_draft_revision=None,
    )
    monkeypatch.setattr(
        "apps.collections.views.schema_api._locked_collection_source_signature",
        lambda collection_id: "a" * 64,
    )
    enqueued = []
    monkeypatch.setattr(
        "apps.collections.views.schema_api.enqueue_schema_generation",
        enqueued.append,
    )

    response = _request(
        client,
        "post",
        reverse("api_collection_schema_generate", kwargs={"col_id": collection.pk}),
        body={},
    )

    run.refresh_from_db()
    assert response.status_code == 202
    assert response.json()["run_id"] == str(run.pk)
    assert run.base_draft_id == draft.pk
    assert run.base_draft_revision == draft.revision
    assert enqueued == [str(run.pk)]


@pytest.mark.django_db
def test_schema_generation_rejects_nonempty_draft_and_changed_active_source(
    client, schema_users, monkeypatch
):
    collection, _viewer, editor, _manager = schema_users
    client.force_login(editor)
    monkeypatch.setattr(
        "apps.collections.views.schema_api._locked_collection_source_signature",
        lambda collection_id: "b" * 64,
    )
    CollectionSchemaGenerationRun.objects.create(
        collection=collection,
        requested_by=editor,
        source_signature="a" * 64,
    )
    url = reverse("api_collection_schema_generate", kwargs={"col_id": collection.pk})

    changed = _request(client, "post", url, body={})
    assert changed.status_code == 409
    assert changed.json() == {"error": "source_changed"}

    CollectionSchemaGenerationRun.objects.all().delete()
    CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions={"entities": [_entity("paper")], "relations": []},
        last_editor=editor,
    )
    draft_conflict = _request(client, "post", url, body={})
    assert draft_conflict.status_code == 409
    assert draft_conflict.json() == {"error": "draft_exists"}


@pytest.mark.django_db
def test_schema_generation_status_is_visible_to_collection_viewer(client, schema_users):
    collection, viewer, editor, _manager = schema_users
    run = CollectionSchemaGenerationRun.objects.create(
        collection=collection,
        requested_by=editor,
        source_signature="a" * 64,
        status=CollectionSchemaGenerationRun.Status.FAILED,
        error_code="no_collection_text",
        statistics={"sample_count": 0},
    )
    client.force_login(viewer)
    url = reverse(
        "api_collection_schema_generation_status",
        kwargs={"col_id": collection.pk, "run_id": run.pk},
    )

    response = client.get(url)

    assert response.status_code == 200
    assert response.json() == {
        "run_id": str(run.pk),
        "status": "failed",
        "error_code": "no_collection_text",
        "statistics": {"sample_count": 0},
    }


@pytest.mark.django_db
def test_workspace_and_entity_mutation_persist_across_requests(client, schema_users):
    collection, _viewer, editor, _manager = schema_users
    client.force_login(editor)
    workspace_url = reverse(
        "api_collection_schema_workspace", kwargs={"col_id": collection.pk}
    )
    draft_url = reverse("api_collection_schema_draft", kwargs={"col_id": collection.pk})
    entity_url = reverse(
        "api_collection_schema_entity",
        kwargs={"col_id": collection.pk, "entity_key": "paper"},
    )

    initial = client.get(workspace_url).json()
    assert initial["published"]["entities"] == []
    assert initial["draft"] is None
    created = _request(client, "post", draft_url, body={}).json()
    draft_id = created["draft"]["draft_id"]
    assert created["draft"]["revision"] == 1

    updated = _request(
        client,
        "put",
        entity_url,
        body={"values": _entity("paper")["values"]},
        revision=1,
    )
    assert updated.status_code == 200
    assert updated.json()["draft"]["revision"] == 2
    reloaded = client.get(workspace_url).json()
    assert reloaded["draft"]["draft_id"] == draft_id
    assert reloaded["draft"]["entities"] == [_entity("paper")]


@pytest.mark.django_db
def test_workspace_capabilities_do_not_advertise_unsupported_renames(
    client, schema_users
):
    collection, _viewer, editor, _manager = schema_users
    definitions = _definitions()
    definitions["entities"][0]["origin"] = "generated"
    definitions["entities"][0]["capabilities"] = {
        "editable_fields": ["description"],
        "removable": True,
        "renameable": True,
    }
    CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions=definitions,
        last_editor=editor,
    )
    client.force_login(editor)

    workspace = client.get(
        reverse("api_collection_schema_workspace", kwargs={"col_id": collection.pk})
    ).json()

    entity = next(
        row for row in workspace["draft"]["entities"] if row["key"] == "paper"
    )
    relation = next(
        row for row in workspace["draft"]["relations"] if row["key"] == "authored_by"
    )
    assert entity["origin"] == "generated"
    assert entity["capabilities"] == {
        "editable_fields": [
            "description",
            "aliases",
            "default_retrieval_weight",
            "default_suppression_policy",
            "default_suppression_threshold",
        ],
        "removable": True,
        "renameable": False,
    }
    assert relation["capabilities"] == {
        "editable_fields": [
            "description",
            "direction",
            "allowed_head_types",
            "allowed_tail_types",
        ],
        "removable": True,
        "renameable": False,
    }


@pytest.mark.django_db
def test_if_match_conflict_returns_current_draft_without_mutating(client, schema_users):
    collection, _viewer, editor, _manager = schema_users
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        revision=4,
        definitions={"entities": [_entity("paper", "Server value")], "relations": []},
        last_editor=editor,
    )
    client.force_login(editor)
    url = reverse(
        "api_collection_schema_entity",
        kwargs={"col_id": collection.pk, "entity_key": "paper"},
    )

    response = _request(
        client,
        "put",
        url,
        body={"values": _entity("paper", "Attempted value")["values"]},
        revision=3,
    )

    assert response.status_code == 409
    assert response.json() == {
        "attempted_revision": 3,
        "current_revision": 4,
        "draft_id": str(draft.pk),
        "definitions": [
            {
                "kind": "entity",
                "key": "paper",
                "fields": [
                    {
                        "field": "description",
                        "server_value": "Server value",
                        "attempted_value": "Attempted value",
                    }
                ],
            }
        ],
    }
    draft.refresh_from_db()
    assert draft.revision == 4


@pytest.mark.django_db
def test_relation_upsert_and_delete_are_revisioned(client, schema_users):
    collection, _viewer, editor, _manager = schema_users
    CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions=_definitions(),
        last_editor=editor,
    )
    client.force_login(editor)
    url = reverse(
        "api_collection_schema_relation",
        kwargs={"col_id": collection.pk, "relation_key": "cites"},
    )
    values = {
        **_relation("cites")["values"],
        "allowed_head_types": ["paper"],
        "allowed_tail_types": ["paper"],
    }

    created = _request(client, "put", url, body={"values": values}, revision=1)
    assert created.status_code == 200
    assert [row["key"] for row in created.json()["draft"]["relations"]] == [
        "authored_by",
        "cites",
    ]
    deleted = _request(client, "delete", url, revision=2)
    assert deleted.status_code == 200
    assert [row["key"] for row in deleted.json()["draft"]["relations"]] == [
        "authored_by"
    ]
    assert deleted.json()["draft"]["revision"] == 3


@pytest.mark.django_db
@pytest.mark.parametrize("body", [{}, {"values": None}])
def test_put_requires_non_null_values_object(client, schema_users, body):
    collection, _viewer, editor, _manager = schema_users
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions={"entities": [_entity("paper")], "relations": []},
        last_editor=editor,
    )
    client.force_login(editor)
    url = reverse(
        "api_collection_schema_entity",
        kwargs={"col_id": collection.pk, "entity_key": "paper"},
    )

    response = _request(client, "put", url, body=body, revision=1)

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_definition"}
    draft.refresh_from_db()
    assert draft.revision == 1
    assert draft.definitions["entities"] == [_entity("paper")]


@pytest.mark.django_db
def test_malformed_json_returns_stable_json_error(client, schema_users):
    collection, _viewer, editor, _manager = schema_users
    CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions={"entities": [_entity("paper")], "relations": []},
        last_editor=editor,
    )
    client.force_login(editor)
    url = reverse(
        "api_collection_schema_entity",
        kwargs={"col_id": collection.pk, "entity_key": "paper"},
    )

    response = client.put(
        url,
        data="{",
        content_type="application/json",
        HTTP_IF_MATCH="1",
    )

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_json"}


@pytest.mark.django_db
@pytest.mark.parametrize("revision", ["1", True, [], {}, 0, -1, None])
def test_validate_revision_requires_exact_positive_int(client, schema_users, revision):
    collection, _viewer, editor, _manager = schema_users
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions=_definitions(),
        last_editor=editor,
    )
    client.force_login(editor)

    response = _request(
        client,
        "post",
        reverse("api_collection_schema_validate", kwargs={"col_id": collection.pk}),
        body={"draft_id": str(draft.pk), "revision": revision},
    )

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_revision"}
    assert CollectionSchemaDraft.objects.filter(pk=draft.pk, revision=1).exists()


@pytest.mark.django_db
@pytest.mark.parametrize("draft_id", [None, "", "not-a-uuid", 1, True, [], {}])
def test_validate_draft_id_requires_uuid_string(client, schema_users, draft_id):
    collection, _viewer, editor, _manager = schema_users
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions=_definitions(),
        last_editor=editor,
    )
    client.force_login(editor)

    response = _request(
        client,
        "post",
        reverse("api_collection_schema_validate", kwargs={"col_id": collection.pk}),
        body={"draft_id": draft_id, "revision": 1},
    )

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_draft_id"}
    assert CollectionSchemaDraft.objects.filter(pk=draft.pk, revision=1).exists()


@pytest.mark.django_db
def test_validation_and_publish_use_exact_draft_identity(client, schema_users):
    collection, _viewer, _editor, manager = schema_users
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions=_definitions(),
        last_editor=manager,
    )
    client.force_login(manager)
    validate_url = reverse(
        "api_collection_schema_validate", kwargs={"col_id": collection.pk}
    )
    publish_url = reverse(
        "api_collection_schema_publish", kwargs={"col_id": collection.pk}
    )

    validation = _request(
        client,
        "post",
        validate_url,
        body={"draft_id": str(draft.pk), "revision": 1},
    )
    assert validation.status_code == 200
    result = validation.json()
    assert result["issues"] == []
    assert result["identity"]["draft_id"] == str(draft.pk)
    assert len(result["identity"]["candidate_checksum"]) == 64

    operation = {
        "draft_id": str(draft.pk),
        "revision": 1,
        "candidate_checksum": result["identity"]["candidate_checksum"],
        "validation_result_id": result["identity"]["result_id"],
    }
    published = _request(client, "post", publish_url, body=operation, revision=1)

    assert published.status_code == 200
    assert published.json()["draft"] is None
    assert published.json()["published"]["version"] == 1
    version = CollectionSchemaVersion.objects.get(collection=collection)
    assert version.checksum == operation["candidate_checksum"]
    assert version.ontology_version_id is not None
    collection.refresh_from_db()
    assert collection.current_schema_version == version


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("body_revision", "header_revision", "error"),
    [
        ("1", 1, "invalid_revision"),
        (True, 1, "invalid_revision"),
        ([], 1, "invalid_revision"),
        ({}, 1, "invalid_revision"),
        (0, 1, "invalid_revision"),
        (1, 2, "revision_mismatch"),
    ],
)
def test_publish_revision_is_exact_positive_int_matching_if_match(
    client, schema_users, body_revision, header_revision, error
):
    collection, _viewer, _editor, manager = schema_users
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions=_definitions(),
        last_editor=manager,
    )
    client.force_login(manager)
    validation = _request(
        client,
        "post",
        reverse("api_collection_schema_validate", kwargs={"col_id": collection.pk}),
        body={"draft_id": str(draft.pk), "revision": 1},
    ).json()["identity"]

    response = _request(
        client,
        "post",
        reverse("api_collection_schema_publish", kwargs={"col_id": collection.pk}),
        body={
            "draft_id": str(draft.pk),
            "revision": body_revision,
            "candidate_checksum": validation["candidate_checksum"],
            "validation_result_id": validation["result_id"],
        },
        revision=header_revision,
    )

    assert response.status_code == 400
    assert response.json() == {"error": error}
    assert CollectionSchemaDraft.objects.filter(pk=draft.pk, revision=1).exists()
    assert not CollectionSchemaVersion.objects.filter(collection=collection).exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("draft_id", [], "invalid_draft_id"),
        ("draft_id", "not-a-uuid", "invalid_draft_id"),
        ("candidate_checksum", {}, "invalid_candidate_checksum"),
        ("validation_result_id", True, "invalid_validation_result_id"),
    ],
)
def test_publish_identity_fields_require_nonempty_strings(
    client, schema_users, field, value, error
):
    collection, _viewer, _editor, manager = schema_users
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions=_definitions(),
        last_editor=manager,
    )
    client.force_login(manager)
    validation = _request(
        client,
        "post",
        reverse("api_collection_schema_validate", kwargs={"col_id": collection.pk}),
        body={"draft_id": str(draft.pk), "revision": 1},
    ).json()["identity"]
    body = {
        "draft_id": str(draft.pk),
        "revision": 1,
        "candidate_checksum": validation["candidate_checksum"],
        "validation_result_id": validation["result_id"],
    }
    body[field] = value

    response = _request(
        client,
        "post",
        reverse("api_collection_schema_publish", kwargs={"col_id": collection.pk}),
        body=body,
        revision=1,
    )

    assert response.status_code == 400
    assert response.json() == {"error": error}
    assert CollectionSchemaDraft.objects.filter(pk=draft.pk, revision=1).exists()


@pytest.mark.django_db
def test_publishing_unchanged_current_schema_is_idempotent(client, schema_users):
    collection, _viewer, _editor, manager = schema_users
    client.force_login(manager)
    first_draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions=_definitions(),
        last_editor=manager,
    )
    first_response = _validate_and_publish(client, collection, first_draft)
    first = CollectionSchemaVersion.objects.get(collection=collection)

    created = _request(
        client,
        "post",
        reverse("api_collection_schema_draft", kwargs={"col_id": collection.pk}),
        body={},
    )
    unchanged_draft = CollectionSchemaDraft.objects.get(
        pk=created.json()["draft"]["draft_id"]
    )
    response = _validate_and_publish(client, collection, unchanged_draft)

    assert first_response.json()["published"]["version"] == 1
    assert response.json()["published"]["version"] == 1
    assert CollectionSchemaVersion.objects.filter(collection=collection).count() == 1
    collection.refresh_from_db()
    assert collection.current_schema_version == first
    assert first.ontology_version.status == OntologyVersion.Status.ACTIVE
    assert (
        OntologyVersion.objects.filter(metadata__collection_id=collection.pk).count()
        == 1
    )


@pytest.mark.django_db
def test_publishing_restored_checksum_reactivates_immutable_history(
    client, schema_users
):
    collection, _viewer, _editor, manager = schema_users
    client.force_login(manager)
    first_draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions=_definitions(),
        last_editor=manager,
    )
    _validate_and_publish(client, collection, first_draft)
    first = CollectionSchemaVersion.objects.get(collection=collection, version=1)

    created = _request(
        client,
        "post",
        reverse("api_collection_schema_draft", kwargs={"col_id": collection.pk}),
        body={},
    )
    second_draft = CollectionSchemaDraft.objects.get(
        pk=created.json()["draft"]["draft_id"]
    )
    mutation = _request(
        client,
        "put",
        reverse(
            "api_collection_schema_entity",
            kwargs={"col_id": collection.pk, "entity_key": "paper"},
        ),
        body={"values": _entity("paper", "Changed paper")["values"]},
        revision=second_draft.revision,
    )
    assert mutation.status_code == 200
    second_draft.refresh_from_db()
    _validate_and_publish(client, collection, second_draft)
    second = CollectionSchemaVersion.objects.get(collection=collection, version=2)

    restored = _request(
        client,
        "post",
        reverse(
            "api_collection_schema_restore",
            kwargs={"col_id": collection.pk, "version_id": first.version},
        ),
        body={},
    )
    assert restored.status_code == 200
    restored_draft = CollectionSchemaDraft.objects.get(collection=collection)
    response = _validate_and_publish(client, collection, restored_draft)

    assert response.json()["published"]["version"] == 1
    assert CollectionSchemaVersion.objects.filter(collection=collection).count() == 2
    assert (
        OntologyVersion.objects.filter(metadata__collection_id=collection.pk).count()
        == 2
    )
    collection.refresh_from_db()
    first.ontology_version.refresh_from_db()
    second.ontology_version.refresh_from_db()
    assert collection.current_schema_version == first
    assert first.ontology_version.status == OntologyVersion.Status.ACTIVE
    assert second.ontology_version.status == OntologyVersion.Status.SUPERSEDED


@pytest.mark.django_db
def test_publish_remains_successful_when_rebuild_publication_fails(
    client,
    schema_users,
    django_capture_on_commit_callbacks,
    monkeypatch,
):
    from apps.knowledge_graph.services import builds

    collection, _viewer, _editor, manager = schema_users
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions=_definitions(),
        last_editor=manager,
    )
    client.force_login(manager)
    validate_url = reverse(
        "api_collection_schema_validate", kwargs={"col_id": collection.pk}
    )
    publish_url = reverse(
        "api_collection_schema_publish", kwargs={"col_id": collection.pk}
    )
    validation = _request(
        client,
        "post",
        validate_url,
        body={"draft_id": str(draft.pk), "revision": 1},
    ).json()
    identity = validation["identity"]

    def unavailable_rebuild(**_kwargs):
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr(builds, "create_rebuild_request", unavailable_rebuild)
    with django_capture_on_commit_callbacks(execute=True):
        response = _request(
            client,
            "post",
            publish_url,
            body={
                "draft_id": str(draft.pk),
                "revision": 1,
                "candidate_checksum": identity["candidate_checksum"],
                "validation_result_id": identity["result_id"],
            },
            revision=1,
        )

    assert response.status_code == 200
    assert CollectionSchemaVersion.objects.filter(collection=collection).exists()


@pytest.mark.django_db
def test_discard_requires_manage_and_exact_revision(client, schema_users):
    collection, _viewer, editor, manager = schema_users
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        revision=2,
        definitions=_definitions(),
        last_editor=editor,
    )
    url = reverse("api_collection_schema_discard", kwargs={"col_id": collection.pk})
    client.force_login(editor)
    assert (
        _request(
            client,
            "post",
            url,
            body={"draft_id": str(draft.pk), "revision": 2},
            revision=2,
        ).status_code
        == 403
    )
    client.force_login(manager)
    discarded = _request(
        client,
        "post",
        url,
        body={"draft_id": str(draft.pk), "revision": 2},
        revision=2,
    )
    assert discarded.status_code == 200
    assert discarded.json()["draft"] is None
    assert not CollectionSchemaDraft.objects.filter(pk=draft.pk).exists()


@pytest.mark.django_db
def test_history_and_diffs_come_from_persisted_versions(client, schema_users):
    collection, viewer, _editor, manager = schema_users
    first = CollectionSchemaVersion.objects.create(
        collection=collection,
        version=1,
        checksum="1" * 64,
        definitions=_definitions(),
        ontology_version=_ontology_record("8.0.0+schema.api.1", "a" * 64),
        published_by=manager,
        summary="Initial",
    )
    changed = _definitions()
    changed["entities"][0] = _entity("paper", "Updated paper")
    CollectionSchemaVersion.objects.create(
        collection=collection,
        version=2,
        checksum="2" * 64,
        definitions=changed,
        ontology_version=_ontology_record("8.0.0+schema.api.2", "b" * 64),
        published_by=manager,
        summary="Updated",
    )
    client.force_login(viewer)
    versions_url = reverse(
        "api_collection_schema_versions", kwargs={"col_id": collection.pk}
    )
    diff_url = reverse(
        "api_collection_schema_version_diff",
        kwargs={"col_id": collection.pk, "version_id": 2},
    )

    history = client.get(versions_url).json()
    assert [row["version"] for row in history["versions"]] == [2, 1]
    assert history["has_more"] is False
    diff = client.get(diff_url).json()
    assert diff == {
        "base_version": first.version,
        "base_checksum": first.checksum,
        "candidate_version": 2,
        "candidate_checksum": "2" * 64,
        "entities": {"added": 0, "changed": 1, "removed": 0},
        "relations": {"added": 0, "changed": 0, "removed": 0},
    }


@pytest.mark.django_db
def test_restore_challenge_protects_existing_draft_then_replaces_it(
    client, schema_users
):
    collection, _viewer, _editor, manager = schema_users
    version = CollectionSchemaVersion.objects.create(
        collection=collection,
        version=1,
        checksum="3" * 64,
        definitions=_definitions(),
        ontology_version=_ontology_record("8.0.0+schema.restore.1", "c" * 64),
        published_by=manager,
    )
    collection.current_schema_version = version
    collection.save(update_fields=("current_schema_version",))
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        revision=3,
        definitions={"entities": [_entity("other")], "relations": []},
        last_editor=manager,
    )
    client.force_login(manager)
    restore_url = reverse(
        "api_collection_schema_restore",
        kwargs={"col_id": collection.pk, "version_id": version.version},
    )
    replace_url = reverse(
        "api_collection_schema_restore_replace", kwargs={"col_id": collection.pk}
    )

    challenged = _request(client, "post", restore_url, body={})
    assert challenged.status_code == 409
    challenge = challenged.json()
    assert challenge["existing_draft_id"] == str(draft.pk)
    replaced = _request(
        client,
        "post",
        replace_url,
        body={
            "version_id": 1,
            "challenge_token": challenge["challenge_token"],
            "existing_draft_revision": 3,
        },
        revision=3,
    )
    assert replaced.status_code == 200
    assert replaced.json()["draft"]["revision"] == 1
    restored_entities = replaced.json()["draft"]["entities"]
    assert [row["key"] for row in restored_entities] == ["author", "paper"]
    assert {row["change_state"] for row in restored_entities} == {"unchanged"}


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("body_revision", "header_revision", "error"),
    [
        ("3", 3, "invalid_revision"),
        (True, 3, "invalid_revision"),
        (2, 3, "revision_mismatch"),
    ],
)
def test_discard_body_revision_is_exact_int_matching_if_match(
    client, schema_users, body_revision, header_revision, error
):
    collection, _viewer, _editor, manager = schema_users
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        revision=3,
        definitions=_definitions(),
        last_editor=manager,
    )
    client.force_login(manager)
    response = _request(
        client,
        "post",
        reverse("api_collection_schema_discard", kwargs={"col_id": collection.pk}),
        body={"draft_id": str(draft.pk), "revision": body_revision},
        revision=header_revision,
    )

    assert response.status_code == 400
    assert response.json() == {"error": error}
    assert CollectionSchemaDraft.objects.filter(pk=draft.pk).exists()


@pytest.mark.django_db
@pytest.mark.parametrize("draft_id", [None, "", "not-a-uuid", 1, True, [], {}])
def test_discard_draft_id_requires_nonempty_string(client, schema_users, draft_id):
    collection, _viewer, _editor, manager = schema_users
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        revision=3,
        definitions=_definitions(),
        last_editor=manager,
    )
    client.force_login(manager)

    response = _request(
        client,
        "post",
        reverse("api_collection_schema_discard", kwargs={"col_id": collection.pk}),
        body={"draft_id": draft_id, "revision": 3},
        revision=3,
    )

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_draft_id"}
    assert CollectionSchemaDraft.objects.filter(pk=draft.pk).exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("body_revision", "header_revision", "error"),
    [
        ("3", 3, "invalid_revision"),
        (True, 3, "invalid_revision"),
        (2, 3, "revision_mismatch"),
    ],
)
def test_restore_replace_revision_is_exact_int_matching_if_match(
    client, schema_users, body_revision, header_revision, error
):
    collection, _viewer, _editor, manager = schema_users
    version = CollectionSchemaVersion.objects.create(
        collection=collection,
        version=1,
        checksum="7" * 64,
        definitions=_definitions(),
        ontology_version=_ontology_record("8.0.0+schema.restore.dto", "8" * 64),
        published_by=manager,
    )
    collection.current_schema_version = version
    collection.save(update_fields=("current_schema_version",))
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        revision=3,
        definitions={"entities": [_entity("other")], "relations": []},
        last_editor=manager,
    )
    client.force_login(manager)
    challenged = _request(
        client,
        "post",
        reverse(
            "api_collection_schema_restore",
            kwargs={"col_id": collection.pk, "version_id": version.version},
        ),
        body={},
    )
    challenge = challenged.json()

    response = _request(
        client,
        "post",
        reverse(
            "api_collection_schema_restore_replace",
            kwargs={"col_id": collection.pk},
        ),
        body={
            "version_id": 1,
            "challenge_token": challenge["challenge_token"],
            "existing_draft_revision": body_revision,
        },
        revision=header_revision,
    )

    assert response.status_code == 400
    assert response.json() == {"error": error}
    draft.refresh_from_db()
    assert draft.revision == 3


@pytest.mark.django_db
@pytest.mark.parametrize("version_id", [None, "", "1", True, [], {}, 0, -1])
def test_restore_replace_version_id_requires_exact_positive_int(
    client, schema_users, version_id
):
    collection, _viewer, _editor, manager = schema_users
    version = CollectionSchemaVersion.objects.create(
        collection=collection,
        version=1,
        checksum="9" * 64,
        definitions=_definitions(),
        ontology_version=_ontology_record("8.0.0+schema.restore.version.dto", "a" * 64),
        published_by=manager,
    )
    collection.current_schema_version = version
    collection.save(update_fields=("current_schema_version",))
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        revision=3,
        definitions={"entities": [_entity("other")], "relations": []},
        last_editor=manager,
    )
    client.force_login(manager)
    challenge = _request(
        client,
        "post",
        reverse(
            "api_collection_schema_restore",
            kwargs={"col_id": collection.pk, "version_id": version.version},
        ),
        body={},
    ).json()

    response = _request(
        client,
        "post",
        reverse(
            "api_collection_schema_restore_replace",
            kwargs={"col_id": collection.pk},
        ),
        body={
            "version_id": version_id,
            "challenge_token": challenge["challenge_token"],
            "existing_draft_revision": 3,
        },
        revision=3,
    )

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_version_id"}
    assert CollectionSchemaDraft.objects.filter(pk=draft.pk, revision=3).exists()


@pytest.mark.django_db
@pytest.mark.parametrize("challenge_token", [None, "", 1, True, [], {}])
def test_restore_replace_challenge_requires_nonempty_string(
    client, schema_users, challenge_token
):
    collection, _viewer, _editor, manager = schema_users
    version = CollectionSchemaVersion.objects.create(
        collection=collection,
        version=1,
        checksum="b" * 64,
        definitions=_definitions(),
        ontology_version=_ontology_record(
            "8.0.0+schema.restore.challenge.dto", "c" * 64
        ),
        published_by=manager,
    )
    collection.current_schema_version = version
    collection.save(update_fields=("current_schema_version",))
    draft = CollectionSchemaDraft.objects.create(
        collection=collection,
        revision=3,
        definitions={"entities": [_entity("other")], "relations": []},
        last_editor=manager,
    )
    client.force_login(manager)

    response = _request(
        client,
        "post",
        reverse(
            "api_collection_schema_restore_replace",
            kwargs={"col_id": collection.pk},
        ),
        body={
            "version_id": 1,
            "challenge_token": challenge_token,
            "existing_draft_revision": 3,
        },
        revision=3,
    )

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_challenge_token"}
    assert CollectionSchemaDraft.objects.filter(pk=draft.pk, revision=3).exists()


@pytest.mark.django_db
def test_view_permission_redacts_shared_draft_and_blocks_mutation(client, schema_users):
    collection, viewer, editor, _manager = schema_users
    CollectionSchemaDraft.objects.create(
        collection=collection,
        definitions=_definitions(),
        last_editor=editor,
    )
    client.force_login(viewer)
    workspace_url = reverse(
        "api_collection_schema_workspace", kwargs={"col_id": collection.pk}
    )
    draft_url = reverse("api_collection_schema_draft", kwargs={"col_id": collection.pk})

    envelope = client.get(workspace_url).json()
    assert envelope["permissions"]["level"] == "VIEW"
    assert envelope["draft"] is None
    assert _request(client, "post", draft_url, body={}).status_code == 403
