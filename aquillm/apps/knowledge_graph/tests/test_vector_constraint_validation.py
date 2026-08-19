from __future__ import annotations

import os
import socket

import numpy as np
import pytest
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models

from apps.collections.models import Collection
from apps.knowledge_graph.models import (
    CanonicalEntity,
    CollectionEntity,
    GraphArtifact,
)

COLLECTION_ID = 22
EMBEDDING_SIGNATURE = (
    f"test-local:model@rev:endpoint={'e' * 64}:dims=1024:"
    "prep=kg-entity-v1:max_chars=8192:batch=64"
)


def _database_is_reachable() -> bool:
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)), timeout=0.2
        ):
            return True
    except OSError:
        return False


def _postgres_test_action(*, available: bool, required: bool) -> str:
    if available:
        return "run"
    return "fail" if required else "skip"


_POSTGRES_AVAILABLE = _database_is_reachable()
_POSTGRES_REQUIRED = os.environ.get(
    "KG_REQUIRE_POSTGRES_TESTS", ""
).strip().lower() in {"1", "true", "yes", "on"}


@pytest.fixture(scope="session", autouse=True)
def _enforce_required_postgres():
    if (
        _postgres_test_action(
            available=_POSTGRES_AVAILABLE,
            required=_POSTGRES_REQUIRED,
        )
        == "fail"
    ):
        pytest.fail(
            "KG_REQUIRE_POSTGRES_TESTS=1 but configured PostgreSQL is unavailable"
        )


database_required = pytest.mark.skipif(
    _postgres_test_action(
        available=_POSTGRES_AVAILABLE,
        required=_POSTGRES_REQUIRED,
    )
    == "skip",
    reason="configured PostgreSQL database is not reachable",
)


def _artifact() -> GraphArtifact:
    artifact = GraphArtifact(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=str(COLLECTION_ID),
        collection_scope_id=COLLECTION_ID,
        status=GraphArtifact.Status.BUILDING,
        source_hash="a" * 64,
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="filter-v1",
        embedding_model_signature=EMBEDDING_SIGNATURE,
    )
    artifact.save()
    return artifact


def _entity(artifact: GraphArtifact, **overrides) -> CollectionEntity:
    values = {
        "artifact": artifact,
        "collection_id": COLLECTION_ID,
        "cluster_key": "b" * 64,
        "label": "Vector entity",
        "normalized_label": "vector entity",
        "entity_type": "model",
        "identifier": "",
        "extraction_confidence": 0.9,
        "resolution_confidence": 0.8,
        "retrieval_utility": 0.7,
        "promotion_confidence": 0.6,
        "embedding_model_signature": EMBEDDING_SIGNATURE,
        "embedding_input_hash": "c" * 64,
        "embedding": [0.0] * 1024,
    }
    values.update(overrides)
    return CollectionEntity(**values)


@pytest.mark.django_db(transaction=True)
@database_required
def test_collection_vector_constraint_validation_restores_pgvector_array():
    Collection.objects.create(pk=COLLECTION_ID, name="vector constraint validation")
    entity = _entity(_artifact())
    supplied_vector = entity.embedding

    entity.save()

    assert isinstance(supplied_vector, list)
    assert isinstance(entity.embedding, np.ndarray)
    cleaned_vector = entity.embedding
    entity.validate_for_persistence()
    assert entity.embedding is cleaned_vector


@pytest.mark.django_db(transaction=True)
@database_required
def test_collection_vector_immutable_comparison_is_exact_and_unambiguous():
    Collection.objects.create(pk=COLLECTION_ID, name="vector immutable comparison")
    artifact = _artifact()
    stored_values = [0.1] * 1024
    entity = _entity(
        artifact,
        embedding=None,
        embedding_model_signature="",
        embedding_input_hash="",
    )
    entity.save()
    models.QuerySet.update(
        CollectionEntity.objects.filter(pk=entity.pk),
        embedding=stored_values,
        embedding_model_signature=EMBEDDING_SIGNATURE,
        embedding_input_hash="c" * 64,
    )
    entity.refresh_from_db()
    stored_vector = entity.embedding

    entity.embedding = stored_values
    entity.clean()
    assert isinstance(entity.embedding, list)
    entity.embedding = stored_vector
    entity.validate_for_persistence()

    assert entity.embedding is stored_vector
    entity.embedding = np.zeros((2, 512), dtype=np.float32)
    with pytest.raises(ValidationError) as exc_info:
        entity.validate_for_persistence()
    assert exc_info.value.message_dict["embedding"] == ["Graph field is immutable."]

    entity.embedding = [0.2] * 1024
    with pytest.raises(ValidationError) as exc_info:
        entity.validate_for_persistence()
    assert exc_info.value.message_dict["embedding"] == ["Graph field is immutable."]


@pytest.mark.django_db(transaction=True)
@database_required
def test_canonical_vector_constraint_validation_preserves_null_only_constraint():
    canonical = CanonicalEntity(
        identity_key="d" * 64,
        resolver_version="canonical-resolution-v1",
        label="Canonical vector",
        normalized_label="canonical vector",
        entity_type="model",
        version_signature="",
        status=CanonicalEntity.Status.ACTIVE,
        embedding=[0.0] * 1024,
        metadata={},
    )
    canonical.clean_fields()
    cleaned_vector = canonical.embedding

    with pytest.raises(ValidationError) as exc_info:
        canonical.validate_constraints()

    assert "kg_canonical_v1_embedding_null" in str(exc_info.value)
    assert canonical.embedding is cleaned_vector


@pytest.mark.django_db(transaction=True)
@database_required
def test_collection_vector_validation_still_rejects_wrong_dimensions():
    Collection.objects.create(pk=COLLECTION_ID, name="invalid vector dimensions")
    entity = _entity(_artifact(), embedding=[0.0] * 1023)

    with pytest.raises(ValidationError) as exc_info:
        entity.validate_for_persistence()

    assert exc_info.value.message_dict["embedding"] == [
        "Embedding must contain 1024 finite dimensions."
    ]
