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
        body={
            "draft_id": str(second_draft.pk),
            "values": _entity("paper", "Changed paper")["values"],
        },
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
