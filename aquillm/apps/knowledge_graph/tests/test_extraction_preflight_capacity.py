"""Capacity categories must survive the coordinator's normal source preflight."""

from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.documents.models import TextChunk
from apps.knowledge_graph.extraction import pipeline
from apps.knowledge_graph.services import builds


@pytest.mark.parametrize("case", ["count", "characters", "growth_after_count"])
@pytest.mark.parametrize("coordinator_preflight", [False, True])
def test_preflight_capacity_overflow_preserves_exact_category(
    monkeypatch, case, coordinator_preflight
):
    document_id = uuid4()
    content = "abc"
    source_hash = sha256(content.encode()).hexdigest()
    document = SimpleNamespace(
        id=document_id,
        collection_id=1,
        ingestion_complete=True,
        full_text=content,
        full_text_hash=source_hash,
        hash_fn=lambda text: sha256(text.encode()).hexdigest(),
    )
    monkeypatch.setattr(pipeline, "_get_concrete_document", lambda *a, **kw: document)
    monkeypatch.setattr(pipeline, "DOCUMENT_EXTRACTION_V1_MAX_CHUNKS", 2)
    monkeypatch.setattr(pipeline, "DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS", 3)
    calls = []

    class Chunks:
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
