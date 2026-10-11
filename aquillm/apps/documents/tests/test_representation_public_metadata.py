"""Existing public source carriers support text metadata without gaining media authority."""
import re

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.chat.refs import CollectionsRef
from apps.chat.services.tool_wiring.source_documents import can_view_document, document_metadata, selected_document_metadata
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_media_acquisition import acquire_chunk_media
from apps.documents.services.source_loading import source_runtime_scope
from .representation_support import retained as retained, upload as upload, activate
from .test_representation_tool_scope import pair as pair

pytestmark = pytest.mark.django_db(transaction=True)


def test_public_metadata_uses_real_original_scope_without_turn_or_body_read(pair, monkeypatch):
    a, b, _, old, _, runtime = pair
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    with source_runtime_scope(runtime), CaptureQueriesContext(connection) as queries:
        selected = selected_document_metadata(a.ingested_by, CollectionsRef([a.collection_id]))
        one = document_metadata(b.id)
        assert can_view_document(a.ingested_by, one)
        assert {doc.id for doc in selected} == {a.id, b.id}
        assert all("full_text" in doc.get_deferred_fields() for doc in selected)
        assert one.id == b.id and "full_text" in one.get_deferred_fields()
        with pytest.raises(SourceScopeUnavailable):
            acquire_chunk_media(old)
    assert runtime.budget.text_used["materialized"] == 0
    assert not any(re.search(r'(?:SELECT |, )"[^"]+"\."full_text"', row["sql"]) for row in queries)


def test_public_metadata_cannot_refresh_original_pointer(pair, monkeypatch):
    a, _, output, _, _, runtime = pair
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    with source_runtime_scope(runtime):
        assert document_metadata(a.id).id == a.id
        captured = runtime._frozen_retrieval_scope.representation
        assert activate(a, output) == "activated"
        with pytest.raises(SourceScopeUnavailable):
            document_metadata(a.id)
        with pytest.raises(SourceScopeUnavailable):
            can_view_document(a.ingested_by, a)
        assert runtime._frozen_retrieval_scope.representation is captured


def test_public_view_checks_keep_original_subset_and_actual_revocation(pair):
    from apps.collections.models import CollectionPermission
    from apps.documents.models import RawTextDocument
    a, _, _, _, _, runtime = pair
    other = RawTextDocument(title="Unselected", full_text="unselected canonical source",
        collection=a.collection, ingested_by=a.ingested_by)
    other.save(dont_rechunk=True)
    assert other.collection.user_can_view(a.ingested_by)
    with source_runtime_scope(runtime):
        assert can_view_document(a.ingested_by, a)
        assert not can_view_document(a.ingested_by, other)
        CollectionPermission.objects.filter(user=a.ingested_by).delete()
        with pytest.raises(SourceScopeUnavailable, match="source_scope_unavailable"):
            can_view_document(a.ingested_by, a)
