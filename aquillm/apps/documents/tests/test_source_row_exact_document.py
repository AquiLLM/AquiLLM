"""Single-body SQL stays narrow; original full-scope authority stays current."""
from uuid import uuid4

import pytest

from apps.documents.models import PDFDocument, TextChunk
from apps.documents.services.chunk_search_scope import freeze_search_scope
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import source_query_rows, source_runtime_scope
from lib.retrieval.operation import operation_scope
from .representation_support import retained as retained, upload as upload, activate
from .test_representation_tool_scope import pair as pair

pytestmark = pytest.mark.django_db(transaction=True)


def body_query(scope, queryset, document_id):
    from apps.documents.services.representation_body_query import exact_document_queryset
    return exact_document_queryset(scope, queryset, document_id)


def captured(pair):
    a, b, _, _, _, runtime = pair
    return freeze_search_scope((a, b), using="default"), runtime


def test_single_body_sql_excludes_unrelated_owner_but_keeps_original_snapshot(pair):
    a, b, _, _, chunk, runtime = pair
    with source_runtime_scope(runtime):
        scope, _ = captured(pair)
        original = scope.representation
        query = body_query(scope, TextChunk.objects.all(), b.id).filter(pk=chunk.pk)
        sql, params = query.query.sql_with_params()
        assert str(b.id) in {str(v) for v in params}
        assert str(a.id) not in {str(v) for v in params}
        assert 'sha256' in sql and 'EXISTS' in sql
        assert [row.pk for row in query] == [chunk.pk]
        assert scope.representation is original
        assert len(original.selections) == 2


def test_real_source_body_reader_uses_exact_metadata_document_hint(pair, monkeypatch):
    from apps.documents.services import representation_store
    _, b, _, _, chunk, runtime = pair
    original = representation_store._active_chunk_predicates
    calls = []
    def observe(queryset, **kwargs):
        result = original(queryset, **kwargs)
        calls.append((kwargs.get('document_id'), kwargs['snapshot']))
        return result
    monkeypatch.setattr(representation_store, '_active_chunk_predicates', observe)
    with source_runtime_scope(runtime):
        scope, _ = captured(pair)
        rows = source_query_rows(TextChunk.objects.filter(pk=chunk.pk))
        assert [row.pk for row in rows] == [chunk.pk]
        assert rows[0].content == b.full_text
        assert any(hint == b.id for hint, _ in calls)
        assert all(snapshot is scope.representation for _, snapshot in calls)
        assert scope.runtime is runtime


@pytest.mark.parametrize('change', ['owner', 'state'])
def test_non_target_drift_still_rejects_single_body_query(pair, change):
    a, b, output, _, chunk, runtime = pair
    with source_runtime_scope(runtime):
        scope, _ = captured(pair)
        if change == 'owner':
            a.full_text = 'changed non-target source'
            a.save()
        else:
            assert activate(a, output) == 'activated'
        with pytest.raises(SourceScopeUnavailable):
            body_query(scope, TextChunk.objects.filter(pk=chunk.pk), b.id)


def collision(doc):
    PDFDocument.objects.bulk_create([PDFDocument(id=doc.id, collection=doc.collection,
        ingested_by=doc.ingested_by, title='Collision', full_text=doc.full_text,
        full_text_hash=doc.full_text_hash)])


@pytest.mark.parametrize('when', ['before', 'after'])
def test_cross_model_collision_rejected_before_or_after_query_construction(pair, when, monkeypatch):
    _, b, _, _, chunk, runtime = pair
    with source_runtime_scope(runtime):
        scope, _ = captured(pair)
        if when == 'before':
            collision(b)
            from apps.documents.services import representation_store
            original_predicates = representation_store._active_chunk_predicates
            predicate_calls = []
            def observe_predicates(*args, **kwargs):
                predicate_calls.append(True)
                return original_predicates(*args, **kwargs)
            monkeypatch.setattr(representation_store, '_active_chunk_predicates', observe_predicates)
            with pytest.raises(ValueError, match='^permission policy returned duplicate scope rows$'):
                body_query(scope, TextChunk.objects.all(), b.id)
            assert predicate_calls == []
        else:
            query = body_query(scope, TextChunk.objects.all(), b.id).filter(pk=chunk.pk)
            collision(b)
            assert list(query) == []


@pytest.mark.parametrize('hint', ['outside', 'string', 'other_selected'])
def test_wrong_hint_cannot_widen_original_requested_documents(pair, hint):
    a, b, _, _, _, runtime = pair
    with source_runtime_scope(runtime):
        scope = freeze_search_scope((b,), using='default')
        value = {'outside': uuid4(), 'string': str(b.id), 'other_selected': a.id}[hint]
        with pytest.raises(SourceScopeUnavailable):
            body_query(scope, TextChunk.objects.all(), value)


@pytest.mark.parametrize('closed', ['budget', 'operation'])
def test_closed_original_lifetime_never_builds_single_body_query(pair, closed):
    _, b, _, _, _, runtime = pair
    with source_runtime_scope(runtime):
        scope, _ = captured(pair)
        with operation_scope(runtime.budget) as operation:
            if closed == 'budget':
                runtime.budget.close('cancelled')
            else:
                operation.live = False
            with pytest.raises(SourceScopeUnavailable):
                body_query(scope, TextChunk.objects.all(), b.id)


def test_lazy_canonical_hash_guard_survives_narrow_sql(pair):
    _, b, _, _, chunk, runtime = pair
    with source_runtime_scope(runtime):
        scope, _ = captured(pair)
        query = body_query(scope, TextChunk.objects.all(), b.id).filter(pk=chunk.pk)
        original_text = b.full_text
        b.full_text = 'changed canonical source through supported owner save'
        b.save()
        assert list(query) == []
        chunk.refresh_from_db()
        assert chunk.content == original_text
        assert chunk.legacy_source_snapshot.canonical_text == original_text
