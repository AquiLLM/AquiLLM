"""Real initially uncaptured scopes and current anchor metadata boundaries."""
from dataclasses import replace

import pytest
from channels.db import database_sync_to_async
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
from apps.documents.services import chunk_search_scope
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.ingestion.tests.output_support import upload as upload
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from apps.chat.services import retrieval_query_binding as binding, retrieval_query_runtime as query
from apps.chat.services.rag_source_anchor_resolution import SourceAnchors
from .rag_query_treatment_support import (
    QUESTION, controls, mutate, query_corpus as query_corpus, close_query_worker as close_query_worker,
)
from .rag_query_treatment_transport import QueryProvider, score_transport as score_transport
from .rag_query_treatment_observation import new_consumer

pytestmark = pytest.mark.django_db(transaction=True)


def uncaptured(corpus):
    authority = build_selected_scope_authorization_context(principal=corpus.user,
        selected_collection_ids=(corpus.collection.pk,), selected_documents=tuple(corpus.docs))
    assert authority is not None
    runtime = SourceRuntime(TurnBudget(TurnLimits()), authority)
    assert not hasattr(runtime, "_frozen_retrieval_scope")
    return runtime


def anchors(corpus):
    row = corpus.chunks[0]
    return SourceAnchors((str(row.doc_id),), ((row.pk, str(row.doc_id), row.chunk_number),),
                         "explicit_document", (), (str(row.doc_id),))


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [False, True])
async def test_uncaptured_original_query_admission_captures_once_and_searches(
    query_corpus, score_transport, monkeypatch, source,
):
    controls(monkeypatch, "vocabulary", source)
    runtime = await database_sync_to_async(uncaptured)(query_corpus)
    real_freeze, captures = chunk_search_scope.freeze_active_representations, []
    def freeze(*args, **kwargs):
        value = real_freeze(*args, **kwargs)
        captures.append(value)
        return value
    monkeypatch.setattr(chunk_search_scope, "freeze_active_representations", freeze)
    from apps.chat.services.rag_pipeline import _run_vector_search
    try:
        with source_runtime_scope(runtime):
            rows = await query.prepare_query_proposals(QUESTION, llm=QueryProvider(QUESTION).llm,
                                                       runtime=runtime, anchors=anchors(query_corpus))
            original = runtime._frozen_retrieval_scope
            async def search(action):
                return await database_sync_to_async(_run_vector_search, thread_sensitive=False)(
                    new_consumer(query_corpus), action.query, 1)
            result = await query.run_optional_proposal(rows[0], anchors=anchors(query_corpus), run=search)
            assert result["result"] and len(captures) == 1
            assert binding.admission(runtime).scope is original
            assert runtime._frozen_retrieval_scope is original
            assert binding.original_binding(runtime).vocabulary is runtime._query_vocabulary
    finally:
        runtime.budget.close("test_finished")


@pytest.mark.parametrize("change", ["remove", "replace", "authority"])
def test_admitted_scope_removal_or_replacement_never_recaptures(query_corpus, change):
    runtime = uncaptured(query_corpus)
    with source_runtime_scope(runtime):
        witness = binding.ensure_admission(runtime)
        if change == "remove":
            del runtime._frozen_retrieval_scope
        elif change == "replace":
            runtime._frozen_retrieval_scope = replace(witness.scope)
        else:
            runtime.authorization = replace(runtime.authorization)
        with pytest.raises(SourceScopeUnavailable):
            binding.ensure_admission(runtime)
        assert runtime._query_admission is witness


def test_failed_capture_cannot_retry_after_source_becomes_available(query_corpus, monkeypatch):
    runtime = uncaptured(query_corpus)
    from apps.collections.models import CollectionPermission
    from apps.documents.services import representation_tool_scope
    original = representation_tool_scope.tool_scope
    attempts = []
    def capture(*args, **kwargs):
        attempts.append(True)
        return original(*args, **kwargs)
    with source_runtime_scope(runtime), monkeypatch.context() as probe:
        probe.setattr(representation_tool_scope, "tool_scope", capture)
        CollectionPermission.objects.filter(user=query_corpus.user, collection=query_corpus.collection).delete()
        with pytest.raises((SourceScopeUnavailable, PermissionError, ValueError)):
            binding.ensure_admission(runtime)
        CollectionPermission.objects.create(user=query_corpus.user, collection=query_corpus.collection,
                                            permission="MANAGE")
        with pytest.raises(SourceScopeUnavailable):
            binding.ensure_admission(runtime)
    assert attempts == [True] and not hasattr(runtime, "_query_admission")


def test_anchor_metadata_keeps_first_latest_checks_without_body_projection(query_corpus, monkeypatch):
    runtime = uncaptured(query_corpus)
    with source_runtime_scope(runtime):
        witness = binding.ensure_admission(runtime)
        real, checks = witness.scope.validate, []
        def validate():
            checks.append("current")
            return real()
        monkeypatch.setattr(witness.scope, "validate", validate)
        before = dict(runtime.budget._text)
        with CaptureQueriesContext(connection) as captured:
            binding.validate_anchors(runtime, anchors(query_corpus))
        assert checks == ["current", "current"]
        assert runtime.budget._text == before and not runtime.cache
        table = connection.ops.quote_name(query_corpus.chunks[0]._meta.db_table)
        selects = [row["sql"].split(" FROM ", 1)[0] for row in captured
                   if " FROM " + table in row["sql"]]
        assert selects and all(table + '."content"' not in row for row in selects)
        with CaptureQueriesContext(connection) as negative:
            list(type(query_corpus.chunks[0]).objects.values("content")[:1])
        assert any(table + '."content"' in row["sql"].split(" FROM ", 1)[0] for row in negative)


def test_anchor_latest_check_rejects_actual_generation_swap(query_corpus, monkeypatch):
    runtime = uncaptured(query_corpus)
    with source_runtime_scope(runtime):
        witness = binding.ensure_admission(runtime)
        real, mutations = binding.anchor_rows, []
        def rows(*args):
            result = real(*args)
            mutate(query_corpus, "generation")
            mutations.append(True)
            return result
        monkeypatch.setattr(binding, "anchor_rows", rows)
        with pytest.raises(SourceScopeUnavailable):
            binding.validate_anchors(runtime, anchors(query_corpus))
        assert mutations == [True] and runtime._frozen_retrieval_scope is witness.scope


@pytest.mark.parametrize("change", ["wrong_pk", "wrong_uuid", "wrong_number", "duplicate", "overbound"])
def test_exact_anchor_metadata_rejects_mismatched_or_unbounded_coordinates(query_corpus, change):
    runtime, requested = uncaptured(query_corpus), anchors(query_corpus)
    pk, doc, number = requested.chunk_identities[0]
    if change == "wrong_pk":
        values = ((query_corpus.chunks[1].pk, doc, number),)
    elif change == "wrong_uuid":
        values = ((pk, str(query_corpus.docs[1].id), number),)
    elif change == "wrong_number":
        values = ((pk, doc, number + 1),)
    elif change == "duplicate":
        values = ((pk, doc, number),) * 2
    else:
        values = tuple((pk, doc, index) for index in range(46))
    with source_runtime_scope(runtime):
        binding.ensure_admission(runtime)
        with pytest.raises(SourceScopeUnavailable):
            binding.validate_anchors(runtime, replace(requested, chunk_identities=values))


@pytest.mark.asyncio
@pytest.mark.parametrize("source", [False, True])
async def test_initialized_first_admission_cannot_downgrade_after_binding_and_scope_deletion(
    query_corpus, score_transport, monkeypatch, source,
):
    from apps.chat.services.rag_turn import preservation_turn
    from apps.documents.services.representation_media_acquisition import media_runtime
    controls(monkeypatch, "vocabulary", source)
    consumer, provider = new_consumer(query_corpus), QueryProvider(QUESTION)
    async with preservation_turn(consumer, max_func_calls=3):
        runtime = media_runtime()
        principal, frozen = runtime._media_principal_binding, runtime._frozen_retrieval_scope
        assert not hasattr(runtime, "_query_admission_attempt")
        calls, real = [], chunk_search_scope.freeze_active_representations
        def freeze(*args, **kwargs):
            calls.append(True)
            return real(*args, **kwargs)
        monkeypatch.setattr(chunk_search_scope, "freeze_active_representations", freeze)
        del runtime._media_principal_binding
        del runtime._frozen_retrieval_scope
        try:
            with pytest.raises(SourceScopeUnavailable):
                await query.prepare_query_proposals(QUESTION, llm=provider.llm, runtime=runtime)
            assert not calls and not provider.requests and not hasattr(runtime, "_query_admission")
        finally:
            runtime._media_principal_binding, runtime._frozen_retrieval_scope = principal, frozen
