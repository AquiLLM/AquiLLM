"""Closed body reads retain fresh entry/final scope without nested entry sweeps."""
from dataclasses import replace

import pytest
from django.db import connection
from django.db.models.query import QuerySet
from django.test.utils import CaptureQueriesContext

from apps.collections.models import CollectionPermission
from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
from apps.documents.models import EmbeddingGeneration, TextChunk
from apps.documents.services.chunk_search_scope import FrozenRetrievalScope, freeze_search_scope
from apps.documents.services.chunk_search_scoped_rows import scoped_rows
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import SourceRuntime, source_query_rows, source_runtime_scope
from apps.documents.tests.dense_sweep_support import corpus, build
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


def captured(count, mode, transport, monkeypatch, settings):
    case = corpus(count, transport[0], monkeypatch, settings)
    if mode == "ready":
        build(case)
    if mode == "none":
        settings.DOCUMENT_EMBEDDING_SPACE_KEY = None
    authority = build_selected_scope_authorization_context(principal=case.owner,
        selected_collection_ids=(case.collection.pk,), selected_documents=case.docs)
    runtime = SourceRuntime(TurnBudget(TurnLimits()), authority)
    with source_runtime_scope(runtime):
        scope = freeze_search_scope(case.docs, using="default")
    assert (scope.dense is None) is (mode == "none")
    if mode != "none":
        assert bool(scope.dense.generation_keys) is (mode == "ready")
    query = TextChunk.objects.filter(doc_id__in=[doc.id for doc in case.docs]).order_by("pk")
    expected = tuple(query.values_list("pk", "doc_id", "content"))
    assert len(expected) == count
    return case, scope, query, expected


@pytest.mark.parametrize("count", [1, 4])
@pytest.mark.parametrize("mode", ["none", "empty", "ready"])
def test_body_and_cache_reads_use_only_fresh_entry_and_final_sweeps(count, mode, transport, monkeypatch, settings):
    case, scope, query, expected = captured(count, mode, transport, monkeypatch, settings)
    runtime, original = scope.runtime, scope.runtime._frozen_retrieval_scope
    real_validate, validations = FrozenRetrievalScope.validate, []
    def validate(current):
        result = real_validate(current)
        validations.append((current.representation, current.authorization, current.runtime))
        return result
    monkeypatch.setattr(FrozenRetrievalScope, "validate", validate)
    try:
        with source_runtime_scope(runtime):
            previous = None
            for refresh, charge in ((False, True), (False, False), (True, True)):
                validations.clear()
                before = runtime.budget.text_used["materialized"]
                with CaptureQueriesContext(connection) as statements:
                    rows = source_query_rows(query, refresh=refresh)
                assert tuple((r.pk, r.doc_id, r.content) for r in rows) == expected
                assert runtime.budget.sources_used == count
                assert runtime.budget.text_used["materialized"] - before == (
                    sum(len(row[2]) for row in expected) if charge else 0)
                assert runtime._frozen_retrieval_scope is original
                assert all(s is scope.representation and a is runtime.authorization and r is runtime
                           for s, a, r in validations)
                if previous is not None:
                    assert all(left is not right for left, right in zip(rows, previous, strict=True))
                previous = rows
                # Observe real successful bodies first. This is an operation bound,
                # independent of source count and of cold/cache/refresh paths.
                owner_reads = [s for s in statements if "UNION ALL" in s["sql"] and "sha256" in s["sql"]]
                assert len(validations) >= 2
                assert 2 <= len(owner_reads) <= 2
                assert len(validations) <= 2
    finally:
        runtime.budget.close("closed_body_test_finished")


def test_late_dense_withdrawal_is_seen_without_discarding_valid_source_or_adopting_dense(transport, monkeypatch, settings):
    _, scope, query, expected = captured(1, "ready", transport, monkeypatch, settings)
    runtime, original_dense = scope.runtime, scope.dense
    key = original_dense.generation_keys[0]
    real_first, real_validate = QuerySet.first, FrozenRetrievalScope.validate
    fetched, keys = [], []
    def first(queryset):
        row = real_first(queryset)
        if queryset.model is TextChunk and row is not None and not fetched:
            fetched.append((row.pk, row.content))
            # Ordered mutation after a real body fetch; not an independent-connection race.
            assert EmbeddingGeneration.objects.filter(pk=key).update(state="failed") == 1
        return row
    def validate(current):
        result = real_validate(current)
        keys.append(result)
        return result
    monkeypatch.setattr(QuerySet, "first", first)
    monkeypatch.setattr(FrozenRetrievalScope, "validate", validate)
    try:
        with source_runtime_scope(runtime):
            rows = source_query_rows(query)
            assert tuple((r.pk, r.doc_id, r.content) for r in rows) == expected
            assert fetched == [(expected[0][0], expected[0][2])]
            assert key in keys[0] and keys[-1] == ()
            assert scope.dense is original_dense and original_dense.generation_keys == (key,)
            charged = runtime.budget.text_used.copy()
            keys.clear()
            cached = source_query_rows(query)
            assert tuple((r.pk, r.doc_id, r.content) for r in cached) == expected
            assert len(keys) >= 2 and all(value == () for value in keys)
            assert runtime.budget.text_used == charged and runtime.budget.sources_used == 1
            assert cached[0] is not rows[0]
    finally:
        runtime.budget.close("closed_body_test_finished")


def test_late_vocabulary_detachment_blocks_bound_scope_handoff_after_real_body(transport, monkeypatch, settings):
    from apps.chat.services.retrieval_query_binding import ensure_admission
    from apps.chat.services.retrieval_query_runtime import BoundProposal, _frozen_vocabulary
    from apps.collections.tests.test_vocabulary_scope import publish, service, entry
    case, scope, query, expected = captured(1, "empty", transport, monkeypatch, settings)
    runtime = scope.runtime
    CollectionPermission.objects.create(user=case.owner, collection=case.collection, permission="MANAGE")
    _, version = publish(case.owner, case.collection, [entry("source", "instrument")])
    pin = service().attach_vocabulary(actor=case.owner, collection=case.collection,
        version=version, expected_revision=0)
    real_first, fetched = QuerySet.first, []
    def first(queryset):
        row = real_first(queryset)
        if queryset.model is TextChunk and row is not None and not fetched:
            fetched.append((row.pk, row.content))
            pin.delete()
        return row
    try:
        with source_runtime_scope(runtime):
            witness = ensure_admission(runtime)
            vocabulary = _frozen_vocabulary(runtime)
            bound = BoundProposal(runtime, "source", "closed-body-control", vocabulary, None)
            scope = replace(witness.scope, proposal_binding=bound)
            query = scope.queryset(query)
            monkeypatch.setattr(QuerySet, "first", first)
            # Direct body loading has no vocabulary policy. Its existing bound
            # caller must revalidate the original pin before handing any rows out.
            with pytest.raises(ValueError, match="pins changed"):
                scoped_rows(query, scope)
            assert fetched == [(expected[0][0], expected[0][2])]
            assert scope.bodies == {}
            assert runtime._query_vocabulary is vocabulary
            assert runtime._frozen_retrieval_scope is witness.scope
    finally:
        runtime.budget.close("closed_body_test_finished")


def test_closed_metadata_preserves_ranked_slice_and_rejects_combinator(transport, monkeypatch, settings):
    _, scope, query, expected = captured(4, "none", transport, monkeypatch, settings)
    runtime = scope.runtime
    sliced = query.order_by("-pk")[1:3]
    original_sql = str(sliced.query)
    try:
        with source_runtime_scope(runtime):
            rows = source_query_rows(sliced)
            assert tuple((r.pk, r.doc_id, r.content) for r in rows) == tuple(reversed(expected))[1:3]
            assert str(sliced.query) == original_sql
            union = query.order_by().union(query.order_by())
            with pytest.raises(SourceScopeUnavailable, match="source_scope_unavailable"):
                source_query_rows(union)
            assert runtime.budget.sources_used == 2
    finally:
        runtime.budget.close("closed_body_test_finished")
