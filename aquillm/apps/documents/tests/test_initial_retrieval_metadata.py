"""Initial selection hydrates coordinates without copying every selected source body."""
import re

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.chat.refs import CollectionsRef
from apps.chat.services.tool_wiring.source_documents import selected_document_metadata
from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import DESCENDED_FROM_DOCUMENT, RawTextDocument
from apps.documents.services import rag_cache

from .representation_owner_predicate_support import legacy_case

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("cached", [False, True])
def test_initial_selected_metadata_defers_bodies_with_same_identity_and_order(cached, settings, monkeypatch):
    case = legacy_case(mixed=True)
    settings.RAG_CACHE_ENABLED = False
    expected = (case.docs if cached else Collection.get_user_accessible_documents(
        case.user, Collection.objects.filter(pk=case.collection.pk)))
    settings.RAG_CACHE_ENABLED = cached
    # Cache I/O is external; keep the real cache-reference hydration and SQL.
    refs = rag_cache.document_refs_from_documents(case.docs)
    monkeypatch.setattr(rag_cache, "get_cached_doc_access_refs", lambda *_: refs)
    monkeypatch.setattr(rag_cache, "set_cached_doc_access_refs", lambda *_: None)
    with CaptureQueriesContext(connection) as queries:
        documents = selected_document_metadata(case.user, CollectionsRef([case.collection.pk]))
        assert [(type(doc), doc.pk, doc.id) for doc in documents] == [
            (type(doc), doc.pk, doc.id) for doc in expected]
        assert {doc.id for doc in documents} == {doc.id for doc in case.docs}
        assert all("full_text" in doc.get_deferred_fields() for doc in documents)
    assert not any(re.search(r'(?:SELECT |, )"[^"]+"\."full_text"', row["sql"]) for row in queries)
    # Same cached refs cannot retain a now-revoked collection.
    CollectionPermission.objects.filter(user=case.user, collection=case.collection).delete()
    assert selected_document_metadata(case.user, CollectionsRef([case.collection.pk])) == []


def test_general_document_list_still_provides_full_sources(settings):
    case = legacy_case()
    settings.RAG_CACHE_ENABLED = False
    documents = Collection.get_user_accessible_documents(case.user, Collection.objects.filter(pk=case.collection.pk))
    assert {doc.id for doc in documents} == {doc.id for doc in case.docs}
    assert all("full_text" not in doc.get_deferred_fields() for doc in documents)


def test_initial_selected_metadata_does_not_hydrate_each_permission_collection(settings):
    case = legacy_case()
    settings.RAG_CACHE_ENABLED = False
    for number in range(12):
        collection = Collection.objects.create(name=f"Unselected permission {number}")
        CollectionPermission.objects.create(user=case.user, collection=collection, permission="VIEW")
    with CaptureQueriesContext(connection) as queries:
        documents = selected_document_metadata(case.user, CollectionsRef([case.collection.pk]))
    assert {doc.id for doc in documents} == {doc.id for doc in case.docs}
    assert len(queries) <= len(DESCENDED_FROM_DOCUMENT) + 1, len(queries)


def test_initial_selection_keeps_direct_scope_despite_inherited_view_permission(settings):
    case = legacy_case()
    settings.RAG_CACHE_ENABLED = False
    child = Collection.objects.create(name="Child selection", parent=case.collection)
    source = "Inherited view must not widen original selected retrieval scope"
    inherited = RawTextDocument.objects.bulk_create([
        RawTextDocument(title="Child document", collection=child, ingested_by=case.user,
                        full_text=source, full_text_hash=RawTextDocument.hash_fn(source))
    ])[0]
    assert child.user_can_view(case.user)
    documents = selected_document_metadata(case.user, CollectionsRef([case.collection.pk, child.pk]))
    assert {doc.id for doc in documents} == {doc.id for doc in case.docs}
    assert inherited.id not in {doc.id for doc in documents}
