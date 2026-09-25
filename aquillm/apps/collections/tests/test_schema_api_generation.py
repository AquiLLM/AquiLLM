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
