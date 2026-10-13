"""Private eager source/row boundaries stage positive observations until handoff."""

from dataclasses import replace

import pytest
from django.db import connection
from django.db.models.query import QuerySet
from django.test.utils import CaptureQueriesContext

from apps.collections.models import CollectionPermission
from apps.collections.services.django_retrieval_authorization import (
    build_selected_scope_authorization_context,
)
from apps.documents.models import TextChunk
from apps.documents.services import chunk_search_scoped_rows as materialization
from apps.documents.services.chunk_search_scope import (
    FrozenRetrievalScope,
    freeze_search_scope,
)
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.documents.tests.test_embedding_generation import legacy as legacy
from apps.documents.tests.test_source_metadata_closed_operation import captured
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.retrieval.operation import operation_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db(transaction=True)


def revoke(case):
    CollectionPermission.objects.filter(
        user=case.owner, collection=case.collection
    ).delete()


def assert_unpublished(scope):
    assert scope.bodies == {}
    assert scope.lexical_ids == set()
    assert scope.fts_rows == {}
    assert scope.runtime.cache == {}


@pytest.mark.parametrize("mode", ["none", "ready"])
@pytest.mark.parametrize("lexical", [False, True, "full_text"])
def test_successful_cold_and_cached_rows_have_two_fresh_ends(
    mode, lexical, transport, monkeypatch, settings
):
    case, scope, query, expected = captured(2, mode, transport, monkeypatch, settings)
    assert all(query.values_list("search_vector_binding", flat=True))
    real_validate, checks = FrozenRetrievalScope.validate, []

    def validate(current):
        result = real_validate(current)
        checks.append(current)
        return result

    monkeypatch.setattr(FrozenRetrievalScope, "validate", validate)
    try:
        with source_runtime_scope(scope.runtime):
            previous = None
            for cold in (True, False):
                checks.clear()
                before = scope.runtime.budget.text_used["materialized"]
                with CaptureQueriesContext(connection) as statements:
                    rows = materialization.scoped_rows(query, scope, lexical=lexical)
                assert tuple((r.pk, r.doc_id, r.content) for r in rows) == expected
                assert set(scope.bodies) == {r.pk for r in rows}
                assert len(scope.runtime.cache) == 2
                assert scope.runtime.budget.sources_used == 2
                assert scope.runtime.budget.text_used["materialized"] - before == (
                    sum(len(item[2]) for item in expected) if cold else 0
                )
                if previous:
                    assert all(a is not b for a, b in zip(rows, previous, strict=True))
                previous = rows
                owner_reads = [
                    s
                    for s in statements
                    if "UNION ALL" in s["sql"] and "sha256" in s["sql"]
                ]
                assert len(checks) == len(owner_reads) == 2
    finally:
        scope.runtime.budget.close("scoped_consolidation_finished")


@pytest.mark.parametrize("phase", ["body", "metadata"])
@pytest.mark.parametrize(
    "mutation", ["permission", "other_body", "authority", "witness", "operation"]
)
def test_actual_io_then_mutation_never_publishes_positive_state(
    phase, mutation, transport, monkeypatch, settings
):
    case, scope, query, expected = captured(2, "none", transport, monkeypatch, settings)
    runtime, fired = scope.runtime, []

    def change():
        fired.append(mutation)
        if mutation == "permission":
            revoke(case)
        elif mutation == "other_body":
            type(case.docs[1]).objects.filter(pk=case.docs[1].pk).update(
                full_text="changed"
            )
        elif mutation == "authority":
            runtime.authorization = replace(runtime.authorization)
        elif mutation == "operation":
            operation.live = False
        else:
            runtime._frozen_retrieval_scope = replace(runtime._frozen_retrieval_scope)

    real_first, real_current = QuerySet.first, materialization._current_rows

    def first(queryset):
        row = real_first(queryset)
        if queryset.model is TextChunk and row is not None and not fired:
            assert row.pk == expected[0][0]
            change()
        return row

    def current(*args, **kwargs):
        result = real_current(*args, **kwargs)
        assert len(result[0]) == 2
        if not fired:
            change()
        return result

    monkeypatch.setattr(QuerySet, "first", first if phase == "body" else real_first)
    monkeypatch.setattr(
        materialization,
        "_current_rows",
        current if phase == "metadata" else real_current,
    )
    try:
        with (
            source_runtime_scope(runtime),
            operation_scope(runtime.budget) as operation,
        ):
            with pytest.raises((SourceScopeUnavailable, ValueError)):
                materialization.scoped_rows(query, scope, lexical=True)
            assert fired == [mutation]
            assert runtime.budget.text_used["materialized"] > 0
            assert_unpublished(scope)
    finally:
        runtime.budget.close("scoped_consolidation_finished")


def test_captured_body_fingerprint_is_never_replaced(transport, monkeypatch, settings):
    _, scope, query, expected = captured(1, "none", transport, monkeypatch, settings)
    original = (expected[0][1], 0, "0" * 32, None)
    scope.bodies[expected[0][0]] = original
    try:
        with source_runtime_scope(scope.runtime):
            assert materialization.scoped_rows(query, scope, lexical=True) == ()
            assert scope.bodies == {expected[0][0]: original}
            assert scope.lexical_ids == set()
            assert scope.runtime.cache == {}
    finally:
        scope.runtime.budget.close("scoped_consolidation_finished")


def test_vocabulary_detachment_discards_pending_cache_and_bound_observations(
    transport, monkeypatch, settings
):
    from apps.chat.services.retrieval_query_binding import ensure_admission
    from apps.chat.services.retrieval_query_runtime import (
        BoundProposal,
        _frozen_vocabulary,
    )
    from apps.collections.tests.test_vocabulary_scope import entry, publish, service

    case, scope, query, _ = captured(1, "none", transport, monkeypatch, settings)
    runtime, fetched = scope.runtime, []
    CollectionPermission.objects.create(
        user=case.owner, collection=case.collection, permission="MANAGE"
    )
    _, version = publish(case.owner, case.collection, [entry("source", "instrument")])
    pin = service().attach_vocabulary(
        actor=case.owner,
        collection=case.collection,
        version=version,
        expected_revision=0,
    )
    real_first = QuerySet.first

    def first(queryset):
        row = real_first(queryset)
        if queryset.model is TextChunk and row is not None and not fetched:
            fetched.append(row.pk)
            pin.delete()
        return row

    try:
        with source_runtime_scope(runtime):
            witness = ensure_admission(runtime)
            vocabulary = _frozen_vocabulary(runtime)
            scope = replace(
                witness.scope,
                proposal_binding=BoundProposal(
                    runtime, "source", "closed-body-control", vocabulary, None
                ),
            )
            monkeypatch.setattr(QuerySet, "first", first)
            with pytest.raises(ValueError, match="pins changed"):
                materialization.scoped_rows(query, scope, lexical="full_text")
            assert len(fetched) == 1
            assert_unpublished(scope)
            assert runtime._frozen_retrieval_scope is witness.scope
    finally:
        runtime.budget.close("scoped_consolidation_finished")


@pytest.mark.parametrize("change", ["same_length", "growing"])
def test_body_digest_and_length_races_never_publish_stale_bytes(
    change, legacy, monkeypatch, settings
):
    doc = legacy[0]
    settings.DOCUMENT_EMBEDDING_SPACE_KEY = None
    CollectionPermission.objects.create(
        user=doc.ingested_by, collection=doc.collection, permission="VIEW"
    )
    authority = build_selected_scope_authorization_context(
        principal=doc.ingested_by,
        selected_collection_ids=(doc.collection_id,),
        selected_documents=(doc,),
    )
    runtime = SourceRuntime(TurnBudget(TurnLimits()), authority)
    with source_runtime_scope(runtime):
        scope = freeze_search_scope((doc,), using="default")
    query = TextChunk.objects.filter(doc_id=doc.id)
    expected = tuple(query.values_list("pk", "doc_id", "content"))
    real_first, attempted = QuerySet.first, []

    def first(queryset):
        if queryset.model is TextChunk and not attempted:
            attempted.append(expected[0][0])
            content = "x" * (len(expected[0][2]) + (100 if change == "growing" else 0))
            TextChunk.objects.filter(pk=expected[0][0]).update(content=content)
        return real_first(queryset)

    monkeypatch.setattr(QuerySet, "first", first)
    try:
        with source_runtime_scope(scope.runtime):
            assert materialization.scoped_rows(query, scope, lexical=True) == ()
            assert attempted == [expected[0][0]]
            assert_unpublished(scope)
            assert scope.runtime.budget.text_used["materialized"] == len(expected[0][2])
    finally:
        scope.runtime.budget.close("scoped_consolidation_finished")


def test_dense_generation_restored_after_entry_cannot_acquire_a_vote(
    transport, monkeypatch, settings
):
    from apps.documents.models import EmbeddingGeneration
    from apps.documents.services.embedding_index_query import rank_compatible_chunks
    from lib.embeddings.document_provider import observe_query_embedding

    _, scope, query, _ = captured(1, "ready", transport, monkeypatch, settings)
    key = scope.dense.generation_keys[0]
    real_first, fetched = QuerySet.first, []

    def first(queryset):
        row = real_first(queryset)
        if queryset.model is TextChunk and row is not None and not fetched:
            fetched.append(row.pk)
            EmbeddingGeneration.objects.filter(pk=key).update(state="ready")
        return row

    try:
        with source_runtime_scope(scope.runtime):
            observation = observe_query_embedding(
                "source", space=scope.space, budget=scope.runtime.budget
            )
            ranked = rank_compatible_chunks(
                snapshot=scope.representation,
                dense_snapshot=scope.dense,
                query=observation,
                limit=1,
                using=scope.using,
                budget=scope.runtime.budget,
                requested_document_ids=scope.requested_document_ids,
            )
            assert len(ranked) == 1
            scope.dense_rows.update({r.chunk_id: r for r in ranked})
            EmbeddingGeneration.objects.filter(pk=key).update(state="failed")
            monkeypatch.setattr(QuerySet, "first", first)
            assert materialization.scoped_rows(query, scope) == ()
            assert len(fetched) == 1
            assert_unpublished(scope)
            assert scope.dense.generation_keys == (key,)
    finally:
        scope.runtime.budget.close("scoped_consolidation_finished")
