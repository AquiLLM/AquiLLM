"""Behavior regressions for audit findings 3, 5 and 7; no external services."""

import json
import uuid
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import yaml
from celery.exceptions import Retry
from django.test import RequestFactory

from apps.collections.services import schema
from apps.collections.tasks import schema_generation as tasks
from apps.collections.views import schema_api as api
from apps.knowledge_graph.services.ontology import load_ontology_yaml
from apps.knowledge_graph.tests.test_ontology import _document


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def select_for_update(self):
        return self

    def filter(self, **filters):
        def matches(row):
            return all(
                getattr(row, key[:-4]) in value
                if key.endswith("__in")
                else str(getattr(row, key)) == str(value)
                for key, value in filters.items()
            )

        return Rows([row for row in self.rows if matches(row)])

    def first(self):
        return next(iter(self.rows), None)

    def get(self, **filters):
        return self.filter(**filters).first()

    def create(self, **values):
        row = SimpleNamespace(pk=uuid.uuid4(), status="queued", **values)
        self.rows.append(row)
        return row


@pytest.mark.parametrize("retries", [0, 3, 9])
def test_worker_loss_delivery_waits_until_live_lease_can_be_reclaimed(
    monkeypatch, retries
):
    from apps.collections import models

    now = tasks.timezone.now()
    token = uuid.uuid4()
    run = SimpleNamespace(
        id=uuid.uuid4(),
        status="running",
        lease_token=token,
        lease_expires_at=now + timedelta(minutes=9),
    )
    monkeypatch.setattr(
        models, "CollectionSchemaGenerationRun", SimpleNamespace(objects=Rows([run]))
    )
    monkeypatch.setattr(tasks.transaction, "atomic", nullcontext)
    task = tasks.generate_collection_schema_task
    task.push_request(retries=retries, called_directly=False, is_eager=True)
    try:
        with pytest.raises(Retry) as pending:
            task.run(str(run.id))
        assert 539 <= pending.value.when <= 541
        assert run.status == "running"
        assert run.lease_token == token
    finally:
        task.pop_request()


@pytest.mark.parametrize("group", ["entity_types", "relations"])
@pytest.mark.parametrize(
    "name",
    ["x" * 65, "entities", "Not Canonical", "under__score", "trailing_", " leading"],
)
def test_type_names_are_rejected_before_publication_and_provider_use(group, name):
    document = _document()
    document[group][0]["name"] = name
    if group == "entity_types":
        document["relations"][0]["allowed_head_types"] = [name]
    with pytest.raises(ValueError, match="name"):
        load_ontology_yaml(yaml.safe_dump(document))


@pytest.mark.parametrize("values", [{"description": "stale tab"}, None])
def test_stale_editor_cannot_mutate_replacement_draft_at_same_revision(
    monkeypatch, values
):
    collection = SimpleNamespace(pk=1)
    draft = SimpleNamespace(
        pk=uuid.uuid4(),
        collection=collection,
        revision=1,
        definitions={
            "entities": [
                {"key": "paper", "values": {"name": "paper", "description": "restored"}}
            ],
            "relations": [],
        },
        save=lambda **kwargs: None,
    )
    before = deepcopy(draft.definitions)
    monkeypatch.setattr(schema.transaction, "atomic", nullcontext)
    monkeypatch.setattr(
        schema, "Collection", SimpleNamespace(objects=Rows([collection]))
    )
    monkeypatch.setattr(
        schema, "CollectionSchemaDraft", SimpleNamespace(objects=Rows([draft]))
    )
    monkeypatch.setattr(api, "_collection", lambda _: collection)
    monkeypatch.setattr(api, "require_edit", lambda *args: None)
    monkeypatch.setattr(api, "workspace_envelope", lambda *args: {})
    method = "put" if values is not None else "delete"
    request = getattr(RequestFactory(), method)(
        "/",
        data=json.dumps({"draft_id": str(uuid.uuid4()), "values": values}),
        content_type="application/json",
        HTTP_IF_MATCH="1",
    )
    request.user = object()
    response = api._mutate(request, 1, "entity", "paper")
    assert response.status_code == 409
    assert draft.definitions == before
    assert draft.revision == 1


def test_generation_request_recovers_expired_run_with_new_fenced_identity(monkeypatch):
    collection = SimpleNamespace(pk=1)
    now = tasks.timezone.now()
    expired = SimpleNamespace(
        pk=uuid.uuid4(),
        collection=collection,
        status="running",
        source_signature="source",
        base_draft_id=None,
        base_draft_revision=None,
        lease_token=uuid.uuid4(),
        lease_expires_at=now - timedelta(seconds=1),
        save=lambda **kwargs: None,
    )
    runs = Rows([expired])
    statuses = api.CollectionSchemaGenerationRun.Status
    monkeypatch.setattr(
        api,
        "CollectionSchemaGenerationRun",
        SimpleNamespace(objects=runs, Status=statuses),
    )
    monkeypatch.setattr(api, "Collection", SimpleNamespace(objects=Rows([collection])))
    monkeypatch.setattr(api, "CollectionSchemaDraft", SimpleNamespace(objects=Rows([])))
    monkeypatch.setattr(api, "_collection", lambda _: collection)
    monkeypatch.setattr(api, "require_edit", lambda *args: None)
    monkeypatch.setattr(api, "_locked_collection_source_signature", lambda _: "source")
    monkeypatch.setattr(api.transaction, "atomic", nullcontext)
    monkeypatch.setattr(api.transaction, "on_commit", lambda callback: callback())
    published = []
    monkeypatch.setattr(api, "_enqueue_generation_safely", published.append)
    request = RequestFactory().post("/", data="{}", content_type="application/json")
    request.user = SimpleNamespace(is_authenticated=True)
    response = api.schema_generate(request, 1)
    assert response.status_code == 202
    result = json.loads(response.content)
    assert result["run_id"] != str(expired.pk)
    assert result["status"] == "queued"
    assert expired.status == "failed"
    assert expired.lease_token is None
    assert published == [result["run_id"]]


@pytest.mark.parametrize("name", ["x" * 65, "entities"])
def test_invalid_draft_names_return_structured_validation_issues(monkeypatch, name):
    document = _document()
    document["relations"][0]["name"] = name
    definitions = {
        "entities": [
            {"key": row["name"], "values": row} for row in document["entity_types"]
        ],
        "relations": [
            {"key": row["name"], "values": row} for row in document["relations"]
        ],
    }
    collection = SimpleNamespace(pk=1)
    draft = SimpleNamespace(
        pk=uuid.uuid4(),
        collection=collection,
        revision=1,
        definitions=definitions,
        base_version=None,
    )
    monkeypatch.setattr(
        schema, "CollectionSchemaDraft", SimpleNamespace(objects=Rows([draft]))
    )
    monkeypatch.setattr(schema, "_next_version", lambda _: 1)
    result = schema.validate_draft(collection, draft.pk, 1)
    assert result["issues"]
    assert result["issues"][0]["code"] == "ontology_invalid"
    assert result["issues"][0]["severity"] == "error"
    assert "name" in result["issues"][0]["message"]
    assert result["identity"]["draft_id"] == str(draft.pk)


def test_canonical_name_length_boundary_is_accepted():
    document = _document()
    document["relations"][0]["name"] = "x" * 64
    assert "x" * 64 in load_ontology_yaml(yaml.safe_dump(document)).relations


@pytest.mark.parametrize(
    "limit",
    [
        None,
        "entity_count",
        "bytes",
        "entity_description",
        "relation_description",
        "alias_length",
        "alias_count",
    ],
)
def test_query_transport_limits_are_structured_before_publication(monkeypatch, limit):
    document = _document()
    if limit in {"entity_count", "bytes"}:
        count = 65 if limit == "entity_count" else 35
        while len(document["entity_types"]) < count:
            row = deepcopy(document["entity_types"][0])
            row.update(name=f"extra_{len(document['entity_types'])}", aliases=[])
            if limit == "bytes":
                row["description"] = chr(0x1F600) * 512
            document["entity_types"].append(row)
    elif limit == "entity_description":
        document["entity_types"][0]["description"] = "x" * 513
    elif limit == "relation_description":
        document["relations"][0]["description"] = "x" * 513
    elif limit == "alias_length":
        document["entity_types"][0]["aliases"] = ["a" * 129]
    elif limit == "alias_count":
        document["entity_types"][0]["aliases"] = [
            f"alias{index}" for index in range(33)
        ]
    collection = SimpleNamespace(pk=1)
    draft = SimpleNamespace(
        pk=uuid.uuid4(),
        collection=collection,
        revision=1,
        base_version=None,
        definitions={
            "entities": [
                {"key": row["name"], "values": row} for row in document["entity_types"]
            ],
            "relations": [
                {"key": row["name"], "values": row} for row in document["relations"]
            ],
        },
    )
    monkeypatch.setattr(
        schema, "CollectionSchemaDraft", SimpleNamespace(objects=Rows([draft]))
    )
    monkeypatch.setattr(schema, "_next_version", lambda _: 1)
    result = schema.validate_draft(collection, draft.pk, 1)
    if limit is None:
        assert result["issues"] == []
    else:
        assert result["issues"]
        assert result["issues"][0]["code"] == "ontology_invalid"
        assert result["issues"][0]["severity"] == "error"


@pytest.mark.parametrize("method", ["put", "delete"])
def test_definition_mutations_require_draft_uuid(monkeypatch, method):
    monkeypatch.setattr(api, "_collection", lambda _: object())
    monkeypatch.setattr(api, "require_edit", lambda *args: None)
    request = getattr(RequestFactory(), method)(
        "/", data='{"values": {}}', content_type="application/json", HTTP_IF_MATCH="1"
    )
    request.user = object()
    response = api._mutate(request, 1, "entity", "paper")
    assert response.status_code == 400
    assert json.loads(response.content) == {"error": "invalid_draft_id"}


@pytest.mark.parametrize("collection_scoped", [False, True])
def test_activation_revalidates_programmatic_definitions_before_database_access(
    collection_scoped,
):
    from dataclasses import replace

    from apps.knowledge_graph.services import ontology

    valid = load_ontology_yaml(yaml.safe_dump(_document()))
    invalid = ontology._build_definition(
        valid.version,
        valid.entity_types,
        {"entities": replace(valid.relations["authored_by"], name="entities")},
        None,
    )
    with pytest.raises(ontology.OntologyValidationError, match="reserved"):
        if collection_scoped:
            ontology.activate_collection_ontology(1, invalid)
        else:
            ontology.activate_ontology(invalid)
