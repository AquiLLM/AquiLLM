from __future__ import annotations

import pytest
from django.contrib.auth.models import User

from apps.collections.models import Collection
from apps.collections.services.graph_visualization import collection_graph_envelope
from apps.documents.models import RawTextDocument
from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun


def _document(collection, user, label, *, complete=True):
    document = RawTextDocument(
        collection=collection,
        ingested_by=user,
        title=label,
        full_text=label,
        full_text_hash=RawTextDocument.hash_fn(label),
        ingestion_complete=complete,
    )
    document.save(dont_rechunk=True)
    return document


def _artifact(document, status, *, source_hash=None, error_code=""):
    artifact = GraphArtifact.objects.create(
        scope_type="document",
        scope_id=str(document.id),
        source_hash=source_hash or document.full_text_hash,
        ontology_version="test-v1",
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="pending-v1",
        status=status,
    )
    if error_code:
        GraphBuildRun.objects.create(
            artifact=artifact,
            build_kind="document",
            scope_type="document",
            scope_id=artifact.scope_id,
            source_hash=artifact.source_hash,
            ontology_version=artifact.ontology_version,
            extractor_version=artifact.extractor_version,
            resolver_version=artifact.resolver_version,
            filter_policy_version=artifact.filter_policy_version,
            status="failed",
            stage="failed",
            error_code=error_code,
        )
    return artifact


@pytest.mark.django_db
def test_automatic_graph_failures_are_visible_without_explicit_rebuild():
    user = User.objects.create_user(username="progress-viewer")
    collection = Collection.objects.create(name="Bulk graph")
    _artifact(_document(collection, user, "ready"), "active")
    _artifact(
        _document(collection, user, "failed"),
        "failed",
        error_code="extraction_entity_limit",
    )
    _document(collection, user, "waiting")
    _document(collection, user, "ingesting", complete=False)

    result = collection_graph_envelope(collection, user)

    assert result["status"]["state"] == "partial"
    assert result["status"]["error_code"] == "document_builds_failed"
    assert result["progress"] == {
        "total": 4,
        "ingesting": 1,
        "pending": 1,
        "building": 0,
        "active": 1,
        "failed": 1,
        "failures": [{"code": "extraction_entity_limit", "count": 1}],
    }


@pytest.mark.django_db
def test_progress_ignores_old_sources_and_other_collections_and_hides_unknown_codes():
    user = User.objects.create_user(username="progress-isolation")
    collection = Collection.objects.create(name="Current graph")
    other = Collection.objects.create(name="Other graph")
    stale = _document(collection, user, "edited")
    _artifact(stale, "active", source_hash="a" * 64)
    _artifact(
        _document(collection, user, "failed"),
        "failed",
        error_code="private_provider_detail",
    )
    _artifact(
        _document(other, user, "other"), "failed", error_code="extraction_entity_limit"
    )

    result = collection_graph_envelope(collection, user)

    assert result["progress"]["active"] == 0
    assert result["progress"]["pending"] == 1
    assert result["progress"]["failed"] == 1
    assert result["progress"]["failures"] == [
        {"code": "document_build_failed", "count": 1}
    ]
    assert "private_provider_detail" not in str(result)


@pytest.mark.django_db
def test_automatic_pending_documents_report_building():
    user = User.objects.create_user(username="progress-pending")
    collection = Collection.objects.create(name="Pending graph")
    _document(collection, user, "waiting")

    result = collection_graph_envelope(collection, user)

    assert result["status"]["state"] == "building"
    assert result["progress"]["pending"] == 1
