"""Ranked body reads use one exact eager SQL batch within original authority."""

import re
from unittest.mock import patch

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.collections.services.django_retrieval_authorization import (
    build_selected_scope_authorization_context,
)
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.source_loading import (
    SourceRuntime,
    source_query_rows,
    source_runtime_scope,
)
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

from .representation_support import activate
from .representation_support import retained as retained
from .representation_support import upload as upload
from .test_source_body_closed_operation import bodies as bodies
from .test_source_body_closed_operation import capture, query

pytestmark = pytest.mark.django_db(transaction=True)


def body_statements(statements):
    return [
        item["sql"]
        for item in statements
        if re.search(
            rf'(?:SELECT|,)\s*"{re.escape(TextChunk._meta.db_table)}"\."content"',
            item["sql"].split(" FROM ", 1)[0],
        )
    ]


def mixed_45(case):
    assert activate(case.other, case.output) == "activated"
    active = tuple(
        TextChunk.objects.filter(
            doc_id=case.other.id, representation_generation__isnull=False
        )
    )
    assert len(active) == 1
    with patch("apps.documents.tasks.create_chunks.delay"):
        for number in range(36):
            text = f"Mixed legacy source {number} supplies a measured statement."
            doc = RawTextDocument.objects.create(
                title=f"Mixed {number}",
                full_text=text,
                collection=case.other.collection,
                ingested_by=case.other.ingested_by,
            )
            row = TextChunk.objects.bulk_create(
                [
                    TextChunk(
                        doc_id=doc.id,
                        content=text,
                        chunk_number=0,
                        start_position=0,
                        end_position=len(text),
                    )
                ]
            )[0]
            case.docs.append(doc)
            case.chunks.append(row)
    case.chunks = sorted([*case.chunks, *active], key=lambda row: row.pk)
    authority = build_selected_scope_authorization_context(
        principal=case.other.ingested_by,
        selected_collection_ids=(case.other.collection_id,),
        selected_documents=(case.other, *case.docs),
    )
    case.runtime = SourceRuntime(TurnBudget(TurnLimits()), authority)
    case.authority = authority
    assert len(case.chunks) == 45
    assert {row.representation_generation_id is None for row in case.chunks} == {
        True,
        False,
    }


@pytest.mark.parametrize("count", [1, 8, 45])
@pytest.mark.parametrize("refresh", [False, True])
def test_real_ranked_body_reads_use_one_exact_batch(bodies, count, refresh):
    if count == 45:
        mixed_45(bodies)
    with source_runtime_scope(bodies.runtime):
        capture(bodies)
        ranked = query(bodies, count).order_by("-pk")
        if refresh:
            assert len(source_query_rows(ranked)) == count
        before = bodies.runtime.budget.text_used["materialized"]
        with CaptureQueriesContext(connection) as statements:
            rows = source_query_rows(ranked, refresh=refresh)
        expected = tuple(reversed(bodies.chunks[:count]))
        assert [(row.pk, row.doc_id, row.content) for row in rows] == [
            (row.pk, row.doc_id, row.content) for row in expected
        ]
        assert all("content" not in row.get_deferred_fields() for row in rows)
        assert bodies.runtime.budget.sources_used == count
        assert bodies.runtime.budget.text_used["materialized"] - before == sum(
            len(row.content) for row in expected
        )
        assert len(body_statements(statements)) == 1


def test_45_cached_copies_skip_body_io_and_do_not_recharge(bodies):
    mixed_45(bodies)
    with source_runtime_scope(bodies.runtime):
        capture(bodies)
        ranked = query(bodies, 45).order_by("-pk")
        first = source_query_rows(ranked)
        charged = bodies.runtime.budget.text_used.copy()
        with CaptureQueriesContext(connection) as statements:
            second = source_query_rows(ranked)
        assert len(first) == len(second) == 45
        assert [row.pk for row in first] == [row.pk for row in second]
        assert all(left is not right for left, right in zip(first, second, strict=True))
        assert len(body_statements(statements)) == 0
        assert bodies.runtime.budget.sources_used == 45
        assert bodies.runtime.budget.text_used == charged
