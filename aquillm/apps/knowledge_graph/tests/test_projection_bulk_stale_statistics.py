"""Original bulk contracts with the measured retained-pages/empty-statistics case."""
from uuid import uuid4

import pytest
from django.db import connection

from apps.documents.models import TextChunk
from apps.knowledge_graph.projection.operation_context import current_operation
from apps.knowledge_graph.projection.runtime import load_projection_runtime_settings
from apps.knowledge_graph.tests.test_projection_bulk_postgres import (
    bulk_mapping,  # noqa: F401 -- pytest discovers the unchanged original fixture.
    pytestmark,
    test_bulk_mapping_state_role_and_source_lease_fences_remain_exact as exercise_fences,
    test_complete_sql_source_and_chunk_mapping_have_exact_checksum as exercise_complete,
)


def _chunk_statistics():
    with connection.cursor() as cursor:
        cursor.execute("SELECT reltuples, relpages FROM pg_class "
                       "WHERE oid='public.aquillm_textchunk'::regclass")
        return cursor.fetchone()


@pytest.fixture(params=[64, 10_005], ids=["small-retained-heap", "large-retained-heap"])
def retained_empty_chunk_statistics(request):
    assert connection.vendor == "postgresql"
    assert not TextChunk.objects.exists()
    document = uuid4()
    TextChunk.objects.bulk_create([
        TextChunk(content="statistics fixture", doc_id=document, chunk_number=index,
                  start_position=index * 7, end_position=index * 7 + 7)
        for index in range(request.param)
    ], batch_size=500)
    assert TextChunk.objects.filter(doc_id=document).count() == request.param
    TextChunk.objects.filter(doc_id=document).delete()
    assert not TextChunk.objects.exists()
    # Populate statistics only while empty. Retained heap pages make the later
    # insertion reproduce the observed zero-cardinality planner estimate.
    with connection.cursor() as cursor:
        cursor.execute("ANALYZE public.aquillm_textchunk")
    estimated_rows, retained_pages = _chunk_statistics()
    assert estimated_rows == 0
    assert retained_pages > 0


@pytest.fixture
def stale_bulk_mapping(retained_empty_chunk_statistics, request):
    mapping = request.getfixturevalue("bulk_mapping")
    _, document, values, _ = mapping
    assert len(values) == 10_005
    assert TextChunk.objects.filter(doc_id=document).count() == 10_005
    assert _chunk_statistics()[0] == 0
    assert current_operation() is None
    assert load_projection_runtime_settings().projection_timeout_ms == 5_000
    return mapping


def test_complete_bulk_contract_with_stale_chunk_statistics(stale_bulk_mapping):
    exercise_complete(stale_bulk_mapping)


def test_bulk_role_and_lease_contract_with_stale_chunk_statistics(stale_bulk_mapping):
    exercise_fences(stale_bulk_mapping)
