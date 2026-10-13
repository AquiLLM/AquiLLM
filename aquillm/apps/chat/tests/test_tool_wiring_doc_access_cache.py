"""Collection document list cache for chat tools (no DB)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from django.test import override_settings

from apps.collections.models import Collection


@override_settings(
    RAG_CACHE_ENABLED=True,
    CACHES={
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "test-document-access-cache",
        },
    },
)
@patch("apps.collections.models.collection._get_document_types")
def test_get_user_accessible_documents_hits_cache_second_call(mock_doc_types):
    fake_model = MagicMock()
    fake_model.objects.filter.return_value = []
    mock_doc_types.return_value = [fake_model]

    perm_qs = MagicMock()
    perm_qs.values_list.return_value = [2, 1]
    col_qs = MagicMock()
    col_qs.filter_by_user_perm.return_value = perm_qs
    user = MagicMock()
    user.id = 42

    Collection.get_user_accessible_documents(user, col_qs)
    Collection.get_user_accessible_documents(user, col_qs)
    assert mock_doc_types.call_count == 1


@pytest.mark.parametrize(
    ("options", "metadata_only"),
    [({}, False), ({"metadata_only": True}, True)],
    ids=["default-full-documents", "metadata-only"],
)
@override_settings(RAG_CACHE_ENABLED=True)
@patch("apps.documents.services.rag_cache.rehydrate_documents_from_refs")
@patch("apps.documents.services.rag_cache.get_cached_doc_access_refs")
def test_cached_document_refs_are_rehydrated_with_fresh_authorized_collection_ids(
    mock_get_cached_refs,
    mock_rehydrate,
    options,
    metadata_only,
):
    cached_refs = [{"model": "RawTextDocument", "pkid": 7}]
    mock_get_cached_refs.return_value = cached_refs
    hydrated_documents = [object()]
    mock_rehydrate.return_value = hydrated_documents

    permitted = MagicMock()
    permitted.values_list.return_value = [9, 3]
    selected = MagicMock()
    selected.filter_by_user_perm.return_value = permitted
    user = MagicMock(id=42)

    result = Collection.get_user_accessible_documents(user, selected, **options)

    assert result is hydrated_documents
    mock_rehydrate.assert_called_once_with(
        cached_refs,
        allowed_collection_ids=(3, 9),
        metadata_only=metadata_only,
    )
