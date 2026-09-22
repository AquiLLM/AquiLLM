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
