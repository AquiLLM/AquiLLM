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
from apps.collections.tests.test_schema_api import (
    _entity,
    _relation,
    _definitions,
    _ontology_record,
    schema_users,
    _request,
    _validate_and_publish,
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
        body={"draft_id": draft_id, "values": _entity("paper")["values"]},
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
        body={
            "draft_id": str(draft.pk),
            "values": _entity("paper", "Attempted value")["values"],
        },
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
    draft = CollectionSchemaDraft.objects.create(
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

    created = _request(
        client,
        "put",
        url,
        body={"draft_id": str(draft.pk), "values": values},
        revision=1,
    )
    assert created.status_code == 200
    assert [row["key"] for row in created.json()["draft"]["relations"]] == [
        "authored_by",
        "cites",
    ]
    deleted = _request(
        client, "delete", url, body={"draft_id": str(draft.pk)}, revision=2
    )
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

    response = _request(
        client, "put", url, body={"draft_id": str(draft.pk), **body}, revision=1
    )

    assert response.status_code == 400
    assert response.json() == {"error": "invalid_definition"}
    draft.refresh_from_db()
    assert draft.revision == 1
    assert draft.definitions["entities"] == [_entity("paper")]
