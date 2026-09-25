from __future__ import annotations

import uuid

import pytest

from apps.knowledge_graph.extraction.windows import (
    ExtractionWindow,
    SpanMappingError,
    batch_extraction_windows,
    deduplicate_mapped_entities,
    map_entity_candidate,
)
from lib.knowledge_graph.types import EntityCandidate

DOCUMENT_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
FIGURE_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")


def _window(**overrides) -> ExtractionWindow:
    values = {
        "chunk_id": 11,
        "document_id": DOCUMENT_ID,
        "content": "Orion evaluates MMLU.",
        "start_position": 100,
        "modality": "text",
    }
    values.update(overrides)
    return ExtractionWindow(**values)


def _candidate(**overrides) -> EntityCandidate:
    values = {
        "entity_type": "model",
        "text": "Orion",
        "start": 0,
        "end": 5,
        "confidence": 0.8,
    }
    values.update(overrides)
    return EntityCandidate(**values)


def test_text_span_maps_to_document_global_offsets_and_validates_source_slice():
    full_text = "x" * 100 + "Orion evaluates MMLU."

    mapped = map_entity_candidate(_window(), _candidate(), full_text=full_text)

    assert (mapped.start, mapped.end) == (100, 105)
    assert mapped.position_basis == "document_global"
    assert mapped.raw_text == "Orion"
    assert mapped.observations[0].local_start == 0
    assert mapped.observations[0].local_end == 5


def test_source_slice_comparison_allows_only_documented_nfc_normalization():
    canonical_equivalent = "\u212b"
    normalized_source = "\u00c5"
    window = _window(content=canonical_equivalent, start_position=0)
    candidate = _candidate(text=canonical_equivalent, start=0, end=1)

    mapped = map_entity_candidate(window, candidate, full_text=normalized_source)

    assert mapped.normalized_text == normalized_source
    assert mapped.raw_text == canonical_equivalent


def test_unicode_normalization_cannot_create_an_out_of_bounds_global_end():
    decomposed = "Cafe\u0301"
    composed = "Caf\u00e9"
    window = _window(content=decomposed, start_position=0)
    candidate = _candidate(text=decomposed, start=0, end=len(decomposed))

    with pytest.raises(SpanMappingError, match="document bounds"):
        map_entity_candidate(window, candidate, full_text=composed)


def test_text_span_rejects_a_source_slice_mismatch():
    with pytest.raises(SpanMappingError, match="source slice"):
        map_entity_candidate(
            _window(),
            _candidate(),
            full_text="x" * 100 + "Altair evaluates MMLU.",
        )


def test_image_span_stays_chunk_local_and_keeps_exact_figure_provenance():
    window = _window(
        document_id=FIGURE_ID,
        content="Orion uses MMLU",
        start_position=999_999,
        modality="image",
        content_object_type_id=7,
        content_object_app_label="apps_documents",
        content_object_model="documentfigure",
        content_object_id=FIGURE_ID,
    )

    mapped = map_entity_candidate(
        window,
        _candidate(),
        full_text="This source intentionally does not contain the caption.",
    )

    assert (mapped.start, mapped.end) == (0, 5)
    assert mapped.position_basis == "chunk_content"
    assert mapped.content_object_type_id == 7
    assert mapped.content_object_id == FIGURE_ID
    assert mapped.observations[0].modality == "image"


def test_overlap_dedupe_keeps_one_global_mention_and_all_observations():
    full_text = "x" * 100 + "Orion evaluates MMLU."
    low = map_entity_candidate(
        _window(chunk_id=11, start_position=100),
        _candidate(confidence=0.71),
        full_text=full_text,
    )
    high = map_entity_candidate(
        _window(
            chunk_id=12,
            content="prefix Orion evaluates MMLU.",
            start_position=93,
        ),
        _candidate(start=7, end=12, confidence=0.94),
        full_text=full_text,
    )

    deduped = deduplicate_mapped_entities((low, high))

    assert len(deduped) == 1
    mention = deduped[0]
    assert mention.chunk_id == 12
    assert mention.confidence == 0.94
    assert [observation.chunk_id for observation in mention.observations] == [11, 12]
    assert [observation.confidence for observation in mention.observations] == [
        0.71,
        0.94,
    ]


def test_same_surface_at_distinct_document_positions_remains_distinct():
    full_text = "Orion and Orion"
    first = map_entity_candidate(
        _window(content=full_text, start_position=0),
        _candidate(),
        full_text=full_text,
    )
    second = map_entity_candidate(
        _window(chunk_id=12, content=full_text, start_position=0),
        _candidate(start=10, end=15),
        full_text=full_text,
    )

    assert len(deduplicate_mapped_entities((first, second))) == 2


def test_window_batches_are_bounded_by_count_and_total_characters():
    windows = tuple(
        _window(chunk_id=index, content="x" * size, start_position=index * 10)
        for index, size in enumerate((4, 5, 6, 2), start=1)
    )

    batches = batch_extraction_windows(
        windows,
        max_count=3,
        max_characters=10,
    )

    assert [[window.chunk_id for window in batch] for batch in batches] == [
        [1, 2],
        [3, 4],
    ]
    assert all(len(batch) <= 3 for batch in batches)
    assert all(sum(len(window.content) for window in batch) <= 10 for batch in batches)


def test_window_batch_rejects_one_chunk_larger_than_the_provider_guard():
    with pytest.raises(ValueError, match="character guard"):
        batch_extraction_windows(
            (_window(content="x" * 11),),
            max_count=2,
            max_characters=10,
        )
