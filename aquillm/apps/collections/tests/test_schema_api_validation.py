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
@pytest.mark.parametrize("limit", ["entity_count", "schema_bytes"])
def test_publish_rejects_schema_exceeding_query_transport_limits(
    client, schema_users, limit
):
    collection, _viewer, _editor, manager = schema_users
    definitions = _definitions()
    count = 65 if limit == "entity_count" else 35
    while len(definitions["entities"]) < count:
        row = _entity(f"extra_{len(definitions['entities'])}")
        if limit == "schema_bytes":
            row["values"]["description"] = chr(0x1F600) * 512
        definitions["entities"].append(row)
    draft = CollectionSchemaDraft.objects.create(
        collection=collection, definitions=definitions, last_editor=manager
    )
    client.force_login(manager)
    validation = _request(
        client,
        "post",
        reverse("api_collection_schema_validate", kwargs={"col_id": collection.pk}),
        body={"draft_id": str(draft.pk), "revision": draft.revision},
    )
    assert validation.status_code == 200
    result = validation.json()
    assert result["issues"][0]["code"] == "ontology_invalid"
    published = _request(
        client,
        "post",
        reverse("api_collection_schema_publish", kwargs={"col_id": collection.pk}),
        body={
            "draft_id": str(draft.pk),
            "revision": draft.revision,
            "candidate_checksum": result["identity"]["candidate_checksum"],
            "validation_result_id": result["identity"]["result_id"],
        },
        revision=draft.revision,
    )
    assert published.status_code == 422
    assert published.json()["error"] == "validation_failed"
    assert not CollectionSchemaVersion.objects.filter(collection=collection).exists()
