from __future__ import annotations

import json

import pytest
from django.contrib.auth.models import User
from django.urls import reverse

from apps.collections.models import (
    Collection,
    CollectionPermission,
    CollectionSchemaDraft,
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
                "name",
                "description",
                "aliases",
                "default_retrieval_weight",
                "default_suppression_policy",
                "default_suppression_threshold",
            ],
            "removable": True,
            "renameable": True,
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
                "name",
                "description",
                "direction",
                "allowed_head_types",
                "allowed_tail_types",
            ],
            "removable": True,
            "renameable": True,
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
