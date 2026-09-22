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
