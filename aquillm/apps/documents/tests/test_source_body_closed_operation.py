"""Real body reads keep fresh boundary fences without per-body full-scope work."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from django.db import connection, transaction
from django.test.utils import CaptureQueriesContext

from apps.collections.models import CollectionPermission
from apps.collections.services.django_retrieval_authorization import (
    build_selected_scope_authorization_context,
)
from apps.documents.models import (
    DocumentRepresentationState,
    PDFDocument,
    RawTextDocument,
    TextChunk,
)
from apps.documents.services.chunk_search_scope import (
    FrozenRetrievalScope,
    freeze_search_scope,
)
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import (
    SourcePreparationLimited,
    SourceRuntime,
    source_query_rows,
    source_runtime_scope,
)
from lib.retrieval.operation import operation_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

from .representation_support import activate
from .representation_support import retained as retained
from .representation_support import upload as upload
from .source_body_test_hooks import install_body_hook

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture
def bodies(retained):
    other, output, old = retained
    docs, chunks = [], []
    with patch("apps.documents.tasks.create_chunks.delay"):
        for number in range(8):
            text = f"Original bounded source {number} states the observed value."
            doc = RawTextDocument.objects.create(title=f"Body {number}", full_text=text,
                collection=other.collection, ingested_by=other.ingested_by)
            chunk = TextChunk.objects.bulk_create([TextChunk(doc_id=doc.id,
                content=text, chunk_number=0, start_position=0, end_position=len(text))])[0]
            docs.append(doc)
            chunks.append(chunk)
    authority = build_selected_scope_authorization_context(principal=other.ingested_by,
        selected_collection_ids=(other.collection_id,), selected_documents=(other, *docs))
    runtime = SourceRuntime(TurnBudget(TurnLimits()), authority)
    return SimpleNamespace(other=other, output=output, old=old, docs=docs,
        chunks=chunks, runtime=runtime, authority=authority)


def query(case, count=8):
    return TextChunk.objects.filter(pk__in=[c.pk for c in case.chunks[:count]]).order_by("pk")


def capture(case):
    scope = freeze_search_scope((case.other, *case.docs), using="default")
    assert scope.runtime is case.runtime and scope.authorization is case.authority
    return scope


@pytest.mark.parametrize("count", [1, 8])
@pytest.mark.parametrize("refresh", [False, True])
def test_actual_body_read_full_scope_work_is_constant(bodies, monkeypatch, count, refresh):
    case = bodies
    original_validate = FrozenRetrievalScope.validate
    calls = []
    def validate(scope):
        result = original_validate(scope)
        calls.append((scope.representation, scope.authorization, scope.runtime))
        return result
    with source_runtime_scope(case.runtime):
        scope = capture(case)
        if refresh:
            assert len(source_query_rows(query(case, count))) == count
        before = case.runtime.budget.text_used["materialized"]
        monkeypatch.setattr(FrozenRetrievalScope, "validate", validate)
        with CaptureQueriesContext(connection) as queries:
            rows = source_query_rows(query(case, count), refresh=refresh)
        assert [row.pk for row in rows] == [c.pk for c in case.chunks[:count]]
        assert [row.content for row in rows] == [d.full_text for d in case.docs[:count]]
        assert all(s is scope.representation and a is case.authority and r is case.runtime
                   for s, a, r in calls)
        assert len(calls) >= 2  # Fresh entry and final authority; never a validated cache.
        assert len(calls) <= 4  # Full-scope work is bounded independently of body count.
        owner_reads = [q for q in queries if "UNION ALL" in q["sql"] and "sha256" in q["sql"]]
        assert 2 <= len(owner_reads) <= 6
        assert case.runtime.budget.text_used["materialized"] - before == sum(len(d.full_text) for d in case.docs[:count])
        assert case.runtime.budget.sources_used == count
        assert case.runtime._frozen_retrieval_scope.representation is scope.representation


def test_cache_reuse_is_copy_only_with_fresh_metadata_and_no_text_recharge(bodies):
    case = bodies
    with source_runtime_scope(case.runtime):
        capture(case)
        first = source_query_rows(query(case))
        charged = case.runtime.budget.text_used.copy()
        first[0].content = "caller-local mutation"
        with CaptureQueriesContext(connection) as queries:
            second = source_query_rows(query(case))
        assert queries.captured_queries
        assert all(left is not right for left, right in zip(first, second))
        assert second[0].content == case.docs[0].full_text
        assert case.runtime.budget.text_used == charged
        assert case.runtime.budget.sources_used == 8


def commit_source_change(case, change):
    """A separate connection commits despite rollback of the rejected reader."""
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_backend_pid()")
            writer_pid = cursor.fetchone()[0]
        assert connection.get_autocommit()
        with transaction.atomic():
            if change == "permission":
                deleted, _ = CollectionPermission.objects.filter(user=case.other.ingested_by,
                    collection=case.other.collection).delete()
                assert deleted > 0
            elif change == "other_owner":
                case.other.full_text = "A supported edit of a non-body owner."
                case.other.save()
            elif change == "other_generation":
                assert activate(case.other, case.output) == "activated"
            else:
                doc = case.docs[0]
                PDFDocument.objects.bulk_create([PDFDocument(id=doc.id,
                    collection=doc.collection, ingested_by=doc.ingested_by,
                    title="Collision", full_text=doc.full_text,
                    full_text_hash=doc.full_text_hash)])
        assert connection.get_autocommit()
        return writer_pid
    finally:
        connection.close()


@pytest.mark.parametrize("change", ["permission", "other_owner", "other_generation", "collision", "budget", "operation"])
def test_change_after_real_body_fetch_blocks_final_return_and_cache_reuse(bodies, monkeypatch, change):
    case = bodies
    fetched = []
    writer_pids = []
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_backend_pid()")
        reader_pid = cursor.fetchone()[0]
    with source_runtime_scope(case.runtime), operation_scope(case.runtime.budget) as operation:
        scope = capture(case)
        original_other = case.other.full_text
        def first(queryset, row):
            if queryset.model is TextChunk and row is not None and not fetched:
                fetched.append((row.pk, row.content))
                if change in {"permission", "other_owner", "other_generation", "collision"}:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        writer_pids.append(pool.submit(commit_source_change, case, change).result(timeout=10))
                elif change == "budget":
                    case.runtime.budget.close("cancelled")
                else:
                    operation.live = False
            return row
        install_body_hook(monkeypatch, first)
        error = ValueError if change == "collision" else SourceScopeUnavailable
        message = "^permission policy returned duplicate scope rows$" if change == "collision" else "^source_scope_unavailable$"
        with pytest.raises(error, match=message):
            source_query_rows(query(case))
        assert len(fetched) == 1 and fetched[0] == (case.chunks[0].pk, case.docs[0].full_text)
        assert case.runtime._frozen_retrieval_scope.representation is scope.representation
        if change in {"other_owner", "other_generation"}:
            case.old.refresh_from_db()
            assert case.old.content == original_other
            assert TextChunk.objects.filter(pk=case.chunks[0].pk,
                content=case.docs[0].full_text).exists()
        if change not in {"budget", "operation"}:
            assert len(writer_pids) == 1 and writer_pids[0] != reader_pid
        if change == "permission":
            assert not CollectionPermission.objects.filter(user=case.other.ingested_by,
                collection=case.other.collection).exists()
        elif change == "other_owner":
            assert RawTextDocument.objects.get(pk=case.other.pk).full_text == "A supported edit of a non-body owner."
        elif change == "other_generation":
            assert DocumentRepresentationState.objects.get(
                model_label=case.other._meta.label_lower, document_pkid=case.other.pk,
                document_id=case.other.id).active_generation_id is not None
        elif change == "collision":
            assert PDFDocument.objects.filter(id=case.docs[0].id).exists()
        second_error = SourcePreparationLimited if change in {"budget", "operation"} else error
        with pytest.raises(second_error):
            source_query_rows(query(case))
