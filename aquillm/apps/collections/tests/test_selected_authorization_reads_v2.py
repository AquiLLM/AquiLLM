"""Initial and final built-in authorization read only captured selected IDs."""

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.collections.models import Collection
from apps.collections.services.retrieval_authorization import (
    freeze_retrieval_authorization_context,
)
from apps.collections.tests.test_bounded_graph_authorization_v2 import fixture
from apps.documents.models import RawTextDocument, TeXDocument, TextChunk
from apps.documents.services.source_loading import SourceRuntime, source_query_rows
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db


def freeze_again(context):
    capability = context.reauthorization_capability
    return freeze_retrieval_authorization_context(
        principal=capability._principal,
        policy=capability._policy,
        database_alias=context.database_alias,
        selected_collection_ids=context.selected_collection_ids,
        selected_document_ids=context.selected_document_ids,
        reauthorization_capability=capability,
    )


def mutate(race, permission, docs):
    if race == "revoke":
        permission.delete()
    elif race == "move":
        RawTextDocument.objects.filter(pk=docs[0].pk).update(
            collection=Collection.objects.create(name="moved")
        )
    elif race == "duplicate":
        RawTextDocument.objects.filter(pk=docs[1].pk).update(id=docs[0].id)
    elif race == "cross_model":
        doc = docs[0]
        TeXDocument.objects.bulk_create(
            [
                TeXDocument(
                    id=doc.id,
                    collection_id=doc.collection_id,
                    ingested_by_id=doc.ingested_by_id,
                    full_text=doc.full_text,
                    full_text_hash=doc.full_text_hash,
                    title=doc.title,
                )
            ]
        )


def assert_bounded(queries, selected):
    data = [q["sql"] for q in queries if "UNION ALL" in q["sql"]]
    assert len(data) == 1
    assert data[0].rstrip().endswith("LIMIT 2")
    assert str(selected) in data[0]
    assert data[0].count("UNION ALL") == 7


@pytest.mark.parametrize("race", [None, "revoke", "move", "duplicate", "cross_model"])
def test_initial_freeze_bounds_selected_rows_and_rejects_selected_races(race):
    context, permission, docs = fixture()
    mutate(race, permission, docs)
    with CaptureQueriesContext(connection) as queries:
        if race is not None:
            with pytest.raises(ValueError):
                freeze_again(context)
        else:
            same = freeze_again(context)
            assert (
                same.authorization_context_signature
                == context.authorization_context_signature
            )
    if race != "revoke":
        assert_bounded(queries, docs[0].id)


@pytest.mark.parametrize("race", [None, "revoke", "move", "duplicate", "cross_model"])
def test_source_cache_hit_rechecks_bounded_original_scope_and_budget(race):
    context, permission, docs = fixture()
    chunk = TextChunk.objects.bulk_create(
        [
            TextChunk(
                doc_id=docs[0].id,
                chunk_number=0,
                start_position=0,
                end_position=7,
                content="initial",
            )
        ]
    )[0]
    budget = TurnBudget(TurnLimits())
    runtime = SourceRuntime(budget, context)
    queryset = TextChunk.objects.filter(pk=chunk.pk)
    assert source_query_rows(queryset, runtime=runtime)[0].content == "initial"
    mutate(race, permission, docs)
    with CaptureQueriesContext(connection) as queries:
        if race in ("duplicate", "cross_model"):
            with pytest.raises(ValueError):
                source_query_rows(queryset, runtime=runtime)
        else:
            rows = source_query_rows(queryset, runtime=runtime)
            assert bool(rows) is (race is None)
    if race != "revoke":
        assert_bounded(queries, docs[0].id)
    assert runtime.budget is budget and runtime.authorization is context
    assert budget.sources_used == 1 and budget.text_used["materialized"] == 7


def test_unrelated_duplicate_documents_do_not_invalidate_selected_scope():
    context, _, docs = fixture()
    RawTextDocument.objects.filter(pk=docs[2].pk).update(id=docs[1].id)
    assert freeze_again(context).selected_document_ids == context.selected_document_ids
