"""Metadata admission precedes every source body fetch, including graph loads."""

from dataclasses import replace

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.chat.tests.test_rag_selection_scoring import DOC, Policy, _authorization
from apps.documents.models.chunks import TextChunk
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def make_chunk(text):
    return TextChunk.objects.bulk_create(
        [
            TextChunk(
                doc_id=DOC,
                chunk_number=0,
                start_position=0,
                end_position=len(text),
                content=text,
            )
        ]
    )[0]


@pytest.mark.django_db
def test_oversized_source_never_materializes_body(monkeypatch):
    from apps.documents.services.source_loading import (
        SourceRuntime,
        source_query_rows,
        source_runtime_scope,
    )

    chunk = make_chunk("x" * 1000)
    runtime = SourceRuntime(
        TurnBudget(replace(TurnLimits(), materialized_codepoints=30)),
        _authorization(Policy()),
    )
    with source_runtime_scope(runtime), CaptureQueriesContext(connection) as queries:
        assert source_query_rows(TextChunk.objects.filter(pk=chunk.pk)) == ()
    assert runtime.budget.text_used["materialized"] == 0
    assert not any(
        'SELECT "aquillm_textchunk"."id", "aquillm_textchunk"."content"' in q["sql"]
        for q in queries
    )


@pytest.mark.django_db
@pytest.mark.parametrize("new_text", ["changed", "a much longer changed source"])
def test_revision_length_race_is_filtered_before_body_read(new_text, monkeypatch):
    from apps.documents.services.source_loading import (
        SourceRuntime,
        source_query_rows,
        source_runtime_scope,
    )

    chunk = make_chunk("initial")
    runtime = SourceRuntime(TurnBudget(TurnLimits()), _authorization(Policy()))

    reserve = runtime.budget.reserve_text

    def changed(size, *, kind):
        accepted = reserve(size, kind=kind)
        TextChunk.objects.filter(pk=chunk.pk).update(content=new_text)
        return accepted

    monkeypatch.setattr(runtime.budget, "reserve_text", changed)

    with source_runtime_scope(runtime):
        assert source_query_rows(TextChunk.objects.filter(pk=chunk.pk)) == ()
    assert runtime.budget.text_used["materialized"] == len("initial")


@pytest.mark.django_db
def test_cached_source_is_reauthorized_and_revision_checked():
    from apps.documents.services.source_loading import (
        SourceRuntime,
        source_query_rows,
        source_runtime_scope,
    )

    chunk = make_chunk("initial")
    policy = Policy()
    runtime = SourceRuntime(TurnBudget(TurnLimits()), _authorization(policy))
    with source_runtime_scope(runtime):
        assert (
            source_query_rows(TextChunk.objects.filter(pk=chunk.pk))[0].content
            == "initial"
        )
        assert (
            source_query_rows(TextChunk.objects.filter(pk=chunk.pk))[0].content
            == "initial"
        )
        assert runtime.budget.text_used["materialized"] == 7
        TextChunk.objects.filter(pk=chunk.pk).update(content="updated")
        assert (
            source_query_rows(TextChunk.objects.filter(pk=chunk.pk))[0].content
            == "updated"
        )
        assert runtime.budget.text_used["materialized"] == 14
        policy.allowed = ()
        assert source_query_rows(TextChunk.objects.filter(pk=chunk.pk)) == ()


def test_source_mode_without_turn_budget_fails_closed(monkeypatch):
    from apps.documents.services.source_loading import (
        SourcePreparationLimited,
        source_query_rows,
    )

    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    with pytest.raises(SourcePreparationLimited):
        source_query_rows(object())


@pytest.mark.django_db
def test_projected_graph_materialization_uses_the_same_body_preflight():
    from apps.documents.services.source_loading import (
        SourceRuntime,
        source_runtime_scope,
    )
    from apps.knowledge_graph.retrieval.ready_materialization import (
        DjangoPrivateChunkMapRepository,
    )

    chunk = make_chunk("x" * 1000)
    runtime = SourceRuntime(
        TurnBudget(replace(TurnLimits(), materialized_codepoints=20)),
        _authorization(Policy()),
    )
    with source_runtime_scope(runtime):
        result = DjangoPrivateChunkMapRepository().load_chunk_objects(
            chunk_predicates=((chunk.pk, DOC, 0),),
            authorized_document_ids=(DOC,),
            database_alias="default",
        )
    assert result == ()
    assert runtime.budget.text_used["materialized"] == 0


def test_independent_authorization_builder_preserves_old_graph_gate(
    monkeypatch, settings
):
    from types import SimpleNamespace

    from django.contrib.auth.models import User

    from apps.collections.services import django_retrieval_authorization as auth

    settings.KG_OVERLAY_ENABLED = False
    monkeypatch.setattr(
        auth.DjangoCollectionRetrievalPermissionPolicy,
        "current_authorized_document_scope",
        lambda *a, **kw: ((1, DOC),),
    )
    kwargs = dict(
        principal=User(pk=1),
        selected_collection_ids=(1,),
        selected_documents=(SimpleNamespace(id=DOC, collection_id=1),),
    )
    assert auth.build_production_retrieval_authorization_context(**kwargs) is None
    independent = auth.build_selected_scope_authorization_context(**kwargs)
    assert independent is not None
    assert independent.selected_document_ids == frozenset((DOC,))
