"""One original representation lifetime across partial searches and source tools."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from threading import Event, local
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


@pytest.mark.parametrize("pause", ["before_check", "before_lock"])
def test_first_public_read_adopts_concurrent_original_only_under_lock(pair, monkeypatch, pause):
    """A real first installation can happen on either side of the entry check."""
    from apps.documents.services import representation_store, source_deadline

    a, b, _, old, chunk, runtime = pair
    waiting, resume, worker = Event(), Event(), local()
    original_lock = runtime.lock

    def gate():
        if getattr(worker, "pause", False):
            worker.pause = False
            waiting.set()
            assert resume.wait(5), "installer did not release waiting source reader"

    class GatedLock:
        def __enter__(self):
            if pause == "before_lock":
                gate()
            return original_lock.__enter__()

        def __exit__(self, *args):
            return original_lock.__exit__(*args)

    runtime.lock = GatedLock()
    real_deadline = source_deadline.check_source_deadline

    def deadline():
        if pause == "before_check":
            gate()
        return real_deadline()

    monkeypatch.setattr(source_deadline, "check_source_deadline", deadline)

    def read_waiter():
        close_old_connections()
        worker.pause = True
        try:
            with source_runtime_scope(runtime):
                return source_query_rows(TextChunk.objects.filter(doc_id=b.id))
        finally:
            close_old_connections()

    with patch.object(representation_store, "_capture_active_representations",
                      wraps=representation_store._capture_active_representations) as freeze:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(read_waiter)
            try:
                assert waiting.wait(5), "source reader did not reach controlled boundary"
                assert getattr(runtime, "_frozen_retrieval_scope", None) is None
                with source_runtime_scope(runtime):
                    assert [r.pk for r in source_query_rows(TextChunk.objects.filter(doc_id=a.id))] == [old.pk]
                original = runtime._frozen_retrieval_scope
            finally:
                resume.set()
            assert [r.pk for r in pending.result(timeout=5)] == [chunk.pk]
        assert freeze.call_count == 1
    assert runtime._frozen_retrieval_scope is original
    assert {s.document.id for s in original.representation.selections} == {a.id, b.id}
    assert runtime.budget.sources_used == 2


@pytest.mark.parametrize("change", ["replacement", "header", "cancel"])
def test_public_original_captured_before_entry_check_cannot_change(pair, monkeypatch, change):
    from apps.documents.services import source_deadline, source_loading

    a, _, _, old, _, runtime = pair
    with source_runtime_scope(runtime):
        assert [r.pk for r in source_query_rows(TextChunk.objects.filter(doc_id=a.id))] == [old.pk]
        cache = runtime.cache.copy()
        original = runtime._frozen_retrieval_scope
        real_deadline = source_deadline.check_source_deadline
        changed = []

        def deadline():
            if not changed:
                changed.append(True)
                if change == "replacement":
                    runtime._frozen_retrieval_scope = replace(original)
                elif change == "header":
                    original.query_digest = "mutated"
                else:
                    runtime.budget.close("cancelled")
            return real_deadline()

        monkeypatch.setattr(source_deadline, "check_source_deadline", deadline)
        with patch.object(source_loading, "_source_body_rows", wraps=source_loading._source_body_rows) as read:
            with pytest.raises(SourceScopeUnavailable):
                source_query_rows(TextChunk.objects.filter(doc_id=a.id))
            assert read.call_count == 0
        assert runtime.cache == cache
        assert runtime.budget.sources_used == 1


@pytest.mark.parametrize("change", ["replacement", "header", "cancel"])
def test_first_public_initialization_rejects_late_cleanup_changes(pair, monkeypatch, change):
    from apps.documents.services import source_loading

    a, _, _, _, _, runtime = pair
    database = source_loading.bounded_source_database
    changed = []

    @contextmanager
    def cleanup(*args):
        with database(*args):
            yield
        if runtime.budget.sources_used == 1 and not changed:
            changed.append(True)
            original = runtime._frozen_retrieval_scope
            if change == "replacement":
                runtime._frozen_retrieval_scope = replace(original)
            elif change == "header":
                original.query_digest = "mutated"
            else:
                runtime.budget.close("cancelled")

    monkeypatch.setattr(source_loading, "bounded_source_database", cleanup)
    with source_runtime_scope(runtime):
        with pytest.raises(SourceScopeUnavailable):
            source_query_rows(TextChunk.objects.filter(doc_id=a.id))
    assert changed == [True]
    assert runtime.cache == {}
    assert runtime.budget.sources_used == 1


def test_initially_absent_public_holder_requires_tool_scope_validation(pair, monkeypatch):
    from apps.documents.services import source_deadline, source_loading

    a, _, _, _, _, runtime = pair
    foreign = SourceRuntime(TurnBudget(TurnLimits()), runtime.authorization)
    with source_runtime_scope(foreign):
        freeze_search_scope((a,), using="default")
    real_deadline = source_deadline.check_source_deadline

    def deadline():
        runtime._frozen_retrieval_scope = foreign._frozen_retrieval_scope
        return real_deadline()

    monkeypatch.setattr(source_deadline, "check_source_deadline", deadline)
    with source_runtime_scope(runtime):
        with patch.object(source_loading, "_source_body_rows", wraps=source_loading._source_body_rows) as read:
            with pytest.raises(SourceScopeUnavailable):
                source_query_rows(TextChunk.objects.filter(doc_id=a.id))
            assert read.call_count == 0
    assert runtime.cache == {} and runtime.budget.sources_used == 0


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
