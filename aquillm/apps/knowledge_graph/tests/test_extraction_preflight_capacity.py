"""Capacity categories must survive the coordinator's normal source preflight."""

from __future__ import annotations

import inspect
import pytest

from apps.documents.models import TextChunk
from apps.knowledge_graph.extraction import pipeline
from apps.knowledge_graph.services import builds


@pytest.mark.django_db
@pytest.mark.parametrize("case", ["count", "characters", "growth_after_count"])
@pytest.mark.parametrize("coordinator_preflight", [False, True])
def test_preflight_capacity_overflow_preserves_exact_category(
    monkeypatch, case, coordinator_preflight
):
    from apps.knowledge_graph.tests.legacy_preflight_support import legacy_owner
    document = legacy_owner(monkeypatch)
    document_id, source_hash = document.id, document.full_text_hash
    monkeypatch.setattr(pipeline, "DOCUMENT_EXTRACTION_V1_MAX_CHUNKS", 2)
    monkeypatch.setattr(pipeline, "DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS", 3)
    calls = []

    class Chunks:
        db = "default"

        def filter(self, **kwargs):
            assert kwargs == {"doc_id": document_id}
            return self

        def only(self, *fields):
            assert "content" in fields
            return self

        def order_by(self, *fields):
            assert fields == ("chunk_number", "pk")
            return self

        def count(self):
            calls.append("count")
            return 3 if case == "count" else 2

        def aggregate(self, **kwargs):
            calls.append("aggregate")
            return {"total_characters": 4 if case == "characters" else 2}

        def iterator(self, *, chunk_size):
            calls.append("iterator")
            assert case == "growth_after_count"
            assert chunk_size == 2
            yield from (object(), object(), object())
            pytest.fail("overflow sentinel must stop further source materialization")

    monkeypatch.setattr(TextChunk, "objects", Chunks())
    with pytest.raises(pipeline.ExtractionCapacityError) as error:
        if coordinator_preflight:
            builds._document_context(document_id, source_hash)
        else:
            pipeline._ordered_chunks(document_id)

    expected = (
        pipeline.ExtractionCapacityCode.CHARACTER_LIMIT
        if case == "characters"
        else pipeline.ExtractionCapacityCode.CHUNK_LIMIT
    )
    assert error.value.code is expected
    assert not isinstance(error.value, pipeline.StaleSourceError)
    assert (
        calls
        == {
            "count": ["count"],
            "characters": ["count", "aggregate"],
            "growth_after_count": ["count", "aggregate", "iterator"],
        }[case]
    )


def test_ordered_chunks_are_counted_and_sized_before_materialization():
    from apps.knowledge_graph.extraction import pipeline

    from apps.knowledge_graph.extraction.representation_chunks import ordered_chunks
    assert "return ordered_chunks(" in inspect.getsource(pipeline._ordered_chunks)
    source = inspect.getsource(ordered_chunks)

    assert "queryset.count()" in source
    assert "DOCUMENT_EXTRACTION_V1_MAX_CHUNKS" in source
    assert "DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS" in source
    assert "queryset.iterator(" in source
    assert "islice(" in source
    assert source.index("queryset.count()") < source.index("queryset.iterator(")
    assert source.index("aggregate(") < source.index("queryset.iterator(")


@pytest.mark.django_db
def test_ordered_chunk_query_loads_only_extraction_fields(monkeypatch):
    from apps.documents import models as document_models
    from apps.knowledge_graph.extraction.pipeline import _ordered_chunks

    calls = []

    class Query:
        db = "default"

        def filter(self, **kwargs):
            calls.append(("filter", kwargs))
            return self

        def order_by(self, *fields):
            calls.append(("order_by", fields))
            return self

        def only(self, *fields):
            calls.append(("only", fields))
            return self

        def count(self):
            calls.append(("count",))
            return 0

        def aggregate(self, **expressions):
            calls.append(("aggregate", tuple(expressions)))
            return {"total_characters": 0}

        def iterator(self, *, chunk_size):
            calls.append(("iterator", chunk_size))
            return iter(())

        def __iter__(self):
            raise AssertionError("ordered chunks must not populate queryset cache")

    from apps.knowledge_graph.tests.legacy_preflight_support import legacy_owner
    document = legacy_owner(monkeypatch)
    monkeypatch.setattr(document_models.TextChunk, "objects", Query())

    assert _ordered_chunks(document.id) == ()
    assert ("count",) in calls
    assert ("aggregate", ("total_characters",)) in calls
    assert ("iterator", 1_000) in calls
    assert (
        "only",
        (
            "pk",
            "doc_id",
            "chunk_number",
            "start_position",
            "end_position",
            "modality",
            "content",
        ),
    ) in calls


@pytest.mark.django_db
def test_ordered_chunk_query_caps_actual_iteration_after_count_drift(monkeypatch):
    from apps.documents import models as document_models
    from apps.knowledge_graph.extraction import pipeline

    consumed = []

    class Query:
        db = "default"

        def filter(self, **_kwargs):
            return self

        def order_by(self, *_fields):
            return self

        def only(self, *_fields):
            return self

        def count(self):
            return 0

        def aggregate(self, **_expressions):
            return {"total_characters": 0}

        def iterator(self, *, chunk_size):
            assert chunk_size == 2
            for value in range(10):
                consumed.append(value)
                yield value

        def __iter__(self):
            raise AssertionError("ordered chunks must use a bounded iterator")

    from apps.knowledge_graph.tests.legacy_preflight_support import legacy_owner
    document = legacy_owner(monkeypatch)
    monkeypatch.setattr(document_models.TextChunk, "objects", Query())
    monkeypatch.setattr(pipeline, "DOCUMENT_EXTRACTION_V1_MAX_CHUNKS", 2)

    with pytest.raises(pipeline.ExtractionCapacityError, match="chunk cap") as error:
        pipeline._ordered_chunks(document.id)
    assert error.value.code is pipeline.ExtractionCapacityCode.CHUNK_LIMIT
    assert consumed == [0, 1, 2]
