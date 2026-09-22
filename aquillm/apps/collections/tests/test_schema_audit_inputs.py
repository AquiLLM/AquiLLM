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
