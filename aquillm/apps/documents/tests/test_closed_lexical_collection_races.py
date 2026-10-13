"""Lexical channels cannot expose earlier positives after a later failed fence."""

from dataclasses import replace

import pytest
from django.db.models.query import QuerySet

from apps.collections.models import CollectionPermission
from apps.documents.models import TextChunk
from apps.documents.services import chunk_search_scoped_rows as rows
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import source_runtime_scope
from apps.documents.tests.test_embedding_generation import legacy as legacy
from apps.documents.tests.test_source_metadata_closed_operation import captured
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.retrieval.operation import operation_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

from .test_closed_lexical_collection_work import collect, configured

pytestmark = pytest.mark.django_db(transaction=True)


def after_exact_metadata(monkeypatch, change):
    original, fired = QuerySet.__iter__, []

    def iterate(queryset):
        result = original(queryset)
        if (
            queryset.model is TextChunk
            and "_source_length" in queryset.query.annotations
            and "_exact_coverage" in queryset.query.annotations
            and not fired
        ):
            fired.append(True)
            change()
        return result

    monkeypatch.setattr(QuerySet, "__iter__", iterate)
    return fired


@pytest.mark.parametrize(
    "mutation", ["permission", "source", "authority", "witness", "budget", "operation"]
)
def test_later_channel_failure_discards_all_new_lexical_positives(
    mutation, transport, monkeypatch, settings
):
    case, scope, _, _ = captured(1, "none", transport, monkeypatch, settings)
    configured(settings)
    runtime, original_budget = scope.runtime, scope.runtime.budget
    replacement = TurnBudget(TurnLimits())

    def change():
        if mutation == "permission":
            CollectionPermission.objects.filter(
                user=case.owner, collection=case.collection
            ).delete()
        elif mutation == "source":
            type(case.docs[0]).objects.filter(pk=case.docs[0].pk).update(
                full_text="changed"
            )
        elif mutation == "authority":
            runtime.authorization = replace(runtime.authorization)
        elif mutation == "witness":
            runtime._frozen_retrieval_scope = replace(runtime._frozen_retrieval_scope)
        elif mutation == "budget":
            runtime.budget = replacement
        else:
            operation.live = False

    fired = after_exact_metadata(monkeypatch, change)
    try:
        with (
            source_runtime_scope(runtime),
            operation_scope(original_budget) as operation,
        ):
            with pytest.raises((SourceScopeUnavailable, ValueError)):
                collect(scope, case.docs)
            assert fired == [True]
            assert scope.bodies == {} and scope.lexical_ids == set()
            assert scope.fts_rows == {} and runtime.cache == {}
            assert original_budget.sources_used == 1
            assert original_budget.text_used["materialized"] > 0
            assert replacement.sources_used == 0
    finally:
        original_budget.close("closed_lexical_collection_finished")
        replacement.close("closed_lexical_collection_finished")


def test_sticky_existing_full_text_withdrawal_survives_later_channel_abort(
    transport, monkeypatch, settings
):
    from apps.documents.services.candidate_rank_fusion import fuse_ranked_channels_v2

    case, scope, query, expected = captured(1, "none", transport, monkeypatch, settings)
    configured(settings)
    try:
        with source_runtime_scope(scope.runtime):
            seeded = rows.scoped_rows(query, scope, lexical="full_text")
            assert tuple((r.pk, r.doc_id, r.content) for r in seeded) == expected
            pk = seeded[0].pk
            scope.fusion_v2 = fuse_ranked_channels_v2(
                {"full_text": ((pk, 1),)}, limit=1, scope_key="a" * 64
            )
            before_bodies, before_cache = (
                scope.bodies.copy(),
                scope.runtime.cache.copy(),
            )

            def change():
                type(case.docs[0]).objects.filter(pk=case.docs[0].pk).update(
                    title="changed"
                )
                CollectionPermission.objects.filter(
                    user=case.owner, collection=case.collection
                ).delete()

            fired = after_exact_metadata(monkeypatch, change)
            with pytest.raises((SourceScopeUnavailable, ValueError)):
                collect(scope, case.docs)
            assert fired == [True]
            assert scope.fts_invalid_ids == {pk} and scope.fusion_v2.candidates == ()
            assert scope.bodies == before_bodies and scope.runtime.cache == before_cache
            assert scope.lexical_ids == set()
    finally:
        scope.runtime.budget.close("closed_lexical_collection_finished")


@pytest.mark.parametrize("replacement", ["SOURCE", "source additional"])
def test_changed_duplicate_body_between_channels_is_never_revived(
    replacement, legacy, monkeypatch, settings
):
    from apps.collections.services.django_retrieval_authorization import (
        build_selected_scope_authorization_context,
    )
    from apps.documents.services.chunk_search_scope import freeze_search_scope
    from apps.documents.services.source_loading import SourceRuntime

    doc = legacy[0]
    configured(settings)
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
    original, fired = QuerySet.__iter__, []

    def iterate(queryset):
        if (
            queryset.model is TextChunk
            and "_source_length" in queryset.query.annotations
            and "_exact_coverage" in queryset.query.annotations
            and not fired
        ):
            fired.append(True)
            assert (
                TextChunk.objects.filter(doc_id=doc.id).update(content=replacement) == 1
            )
        return original(queryset)

    monkeypatch.setattr(QuerySet, "__iter__", iterate)
    try:
        with source_runtime_scope(runtime):
            result = collect(scope, (doc,), query='"source"')
        assert fired == [True]
        assert (
            result.trigram_results
            == result.exact_results
            == result.baseline_candidates
            == ()
        )
        assert runtime.budget.sources_used == 2
        assert runtime.budget.text_used["materialized"] == len("source") + len(
            replacement
        )
        assert scope.bodies == {} and scope.lexical_ids == set() and runtime.cache == {}
    finally:
        runtime.budget.close("closed_lexical_collection_finished")
