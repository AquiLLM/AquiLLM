"""One original representation lifetime across partial searches and source tools."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from django.db import close_old_connections

from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.chunk_search_scope import freeze_search_scope
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import SourceRuntime, source_query_rows, source_runtime_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from .representation_support import retained as retained, upload as upload, activate

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def pair(retained):
    a, output, old = retained
    with patch("apps.documents.tasks.create_chunks.delay"):
        b = RawTextDocument.objects.create(title="Second", full_text="second source",
            collection=a.collection, ingested_by=a.ingested_by)
    chunk = TextChunk.objects.bulk_create([TextChunk(doc_id=b.id, content=b.full_text,
        start_position=0, end_position=len(b.full_text), chunk_number=0)])[0]
    auth = build_selected_scope_authorization_context(principal=a.ingested_by,
        selected_collection_ids=(a.collection_id,), selected_documents=(a, b))
    return a, b, output, old, chunk, SourceRuntime(TurnBudget(TurnLimits()), auth)


@pytest.mark.parametrize("first", ["search", "tool"])
def test_subset_first_captures_original_selection_once(pair, first):
    a, b, output, old, chunk, runtime = pair
    with source_runtime_scope(runtime):
        if first == "search":
            scope = freeze_search_scope((a,), using="default")
            assert scope.requested_document_ids == frozenset((a.id,))
        else:
            assert [r.pk for r in source_query_rows(TextChunk.objects.filter(doc_id=a.id))] == [old.pk]
        captured = runtime._frozen_retrieval_scope.representation
        assert {s.document.id for s in captured.selections} == {a.id, b.id}
        assert [r.pk for r in source_query_rows(TextChunk.objects.filter(doc_id=b.id))] == [chunk.pk]
        scope = freeze_search_scope((b,), using="default")
        assert scope.representation is captured
        assert scope.requested_document_ids == frozenset((b.id,))
        assert activate(a, output) == "activated"
        with pytest.raises(SourceScopeUnavailable):
            source_query_rows(TextChunk.objects.filter(doc_id=b.id))
        assert runtime._frozen_retrieval_scope.representation is captured


def test_concurrent_tools_share_original_snapshot(pair):
    a, b, _, old, chunk, runtime = pair
    def read(doc_id):
        close_old_connections()
        try:
            with source_runtime_scope(runtime):
                rows = source_query_rows(TextChunk.objects.filter(doc_id=doc_id))
                return [r.pk for r in rows], runtime._frozen_retrieval_scope.representation
        finally:
            close_old_connections()
    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = list(pool.map(read, (a.id, b.id)))
    assert left[0] == [old.pk] and right[0] == [chunk.pk]
    assert left[1] is right[1]


def test_current_source_reader_never_returns_retired_pk(retained):
    doc, output, old = retained
    assert activate(doc, output) == "activated"
    doc.refresh_from_db()
    auth = build_selected_scope_authorization_context(principal=doc.ingested_by,
        selected_collection_ids=(doc.collection_id,), selected_documents=(doc,))
    with source_runtime_scope(SourceRuntime(TurnBudget(TurnLimits()), auth)):
        assert source_query_rows(TextChunk.objects.filter(pk=old.pk)) == ()
        rows = source_query_rows(TextChunk.objects.filter(doc_id=doc.id))
        assert rows and all(row.representation_generation_id for row in rows)


def test_ranked_slice_filters_history_before_offset_without_changing_query(retained):
    from django.apps import apps
    from django.contrib.postgres.search import TrigramSimilarity
    doc, output, old = retained
    config = apps.get_app_config("aquillm")
    text = "table value " * (config.chunk_size // 2)
    assert activate(doc, output, text=text) == "activated"
    doc.refresh_from_db()
    auth = build_selected_scope_authorization_context(principal=doc.ingested_by,
        selected_collection_ids=(doc.collection_id,), selected_documents=(doc,))
    query = TextChunk.objects.filter(doc_id=doc.id).annotate(rank=TrigramSimilarity("content", "table"))
    query = query.order_by("-rank", "pk")[1:3]
    sql = str(query.query)
    with source_runtime_scope(SourceRuntime(TurnBudget(TurnLimits()), auth)):
        actual = source_query_rows(query)
    expected = list(doc.chunks.annotate(rank=TrigramSimilarity("content", "table"))
                    .order_by("-rank", "pk").values_list("pk", flat=True)[1:3])
    assert [row.pk for row in actual] == expected
    assert len(actual) <= 45 and old.pk not in expected
    assert str(query.query) == sql
