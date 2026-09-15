"""Scope discovery fetches authorized metadata without corpus text."""

from types import SimpleNamespace

from django.db.models.query import QuerySet, ValuesListIterable

from apps.chat.refs import CollectionsRef
from apps.chat.services import manual_search_turn
from apps.collections.models import Collection
from apps.collections.models import collection as collection_models
from apps.documents.models import PDFDocument


def test_document_selection_fetches_only_metadata_in_permission_filtered_scope(
    monkeypatch,
):
    user = object()
    seen_users = []
    statements = []
    doc = SimpleNamespace(id="doc-a", title="Paper A")

    def permitted(queryset, actual_user, perm="VIEW"):
        seen_users.append(actual_user)
        return queryset.filter(pk=7)

    def fetch(queryset):
        if queryset._result_cache is not None:
            return
        if queryset.model is Collection:
            queryset._result_cache = (
                [7] if issubclass(queryset._iterable_class, ValuesListIterable) else []
            )
        elif queryset.model is PDFDocument:
            statements.append(str(queryset.query))
            queryset._result_cache = [doc]
        else:
            raise AssertionError("Unexpected database query")

    monkeypatch.setattr(
        collection_models.CollectionQuerySet, "filter_by_user_perm", permitted
    )
    monkeypatch.setattr(collection_models, "_get_document_types", lambda: [PDFDocument])
    monkeypatch.setattr(QuerySet, "_fetch_all", fetch)
    consumer = SimpleNamespace(user=user, col_ref=CollectionsRef([7, 8]))
    assert manual_search_turn.load_search_documents(consumer) == [doc]
    assert seen_users == [user]
    assert len(statements) == 1
    assert '"full_text"' not in statements[0]
    assert '"full_text_hash"' not in statements[0]
    assert '"title"' in statements[0]
    assert '"collection_id" IN' in statements[0]
    assert '"id" = 7' in statements[0]
