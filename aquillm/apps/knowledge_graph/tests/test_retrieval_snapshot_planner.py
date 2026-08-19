"""PostgreSQL planner bounds for the authorized retrieval snapshot."""

from __future__ import annotations

import os
import socket

import pytest
from django.conf import settings

from apps.knowledge_graph.retrieval import expansion


def _database_is_reachable() -> bool:
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)),
            timeout=0.2,
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _database_is_reachable() and os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
    reason="configured PostgreSQL database is not reachable",
)


def _identity(*, embedding_model_signature: str) -> dict[str, object]:
    return {
        "build_key": "a" * 64,
        "build_generation": 1,
        "orchestration_version": 1,
        "source_hash": "b" * 64,
        "ontology_version": "research-v1",
        "extractor_version": "extractor-v1",
        "resolver_version": "resolver-v1",
        "filter_policy_version": "filter-v1",
        "embedding_model_signature": embedding_model_signature,
        "ontology_checksum": "c" * 64,
        "filter_policy_checksum": "d" * 64,
        "resolution_config_checksum": "e" * 64,
        "assembly_version": "not-applicable",
        "assembly_config_checksum": "f" * 64,
    }


def test_artifact_identity_requires_scope_appropriate_embedding_signature() -> None:
    expansion._validate_evaluation_identity_row(
        _identity(embedding_model_signature=""),
        scope_type="document",
    )
    expansion._validate_evaluation_identity_row(
        _identity(embedding_model_signature="embedding:model@revision"),
        scope_type="collection",
    )

    with pytest.raises(expansion._SnapshotMiss):
        expansion._validate_evaluation_identity_row(
            _identity(embedding_model_signature="embedding:model@revision"),
            scope_type="document",
        )
    with pytest.raises(expansion._SnapshotMiss):
        expansion._validate_evaluation_identity_row(
            _identity(embedding_model_signature=""),
            scope_type="collection",
        )


def test_eval_document_identity_differs_only_by_embedding_signature() -> None:
    collection = _identity(embedding_model_signature="embedding:model@revision")
    document = {
        f"document_artifact__{field}": value for field, value in collection.items()
    }
    document["document_artifact__embedding_model_signature"] = ""

    assert expansion._evaluation_document_identity_matches_collection(
        document,
        collection,
    )
    document["document_artifact__ontology_checksum"] = "0" * 64
    assert not expansion._evaluation_document_identity_matches_collection(
        document,
        collection,
    )


@pytest.mark.django_db(transaction=True)
@database_required
def test_postgres_snapshot_limits_join_reordering_locally() -> None:
    from django.db import connection

    with connection.cursor() as cursor:
        cursor.execute("SHOW join_collapse_limit")
        original_limit = cursor.fetchone()

    with expansion.authorized_retrieval_snapshot(timeout_ms=150):
        with connection.cursor() as cursor:
            cursor.execute("SHOW join_collapse_limit")
            assert cursor.fetchone() == ("1",)

    with connection.cursor() as cursor:
        cursor.execute("SHOW join_collapse_limit")
        assert cursor.fetchone() == original_limit
