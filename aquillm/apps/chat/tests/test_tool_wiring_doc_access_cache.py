"""Collection document list cache for chat tools (no DB)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import override_settings

from apps.collections.models import Collection


@override_settings(RAG_CACHE_ENABLED=True)
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
