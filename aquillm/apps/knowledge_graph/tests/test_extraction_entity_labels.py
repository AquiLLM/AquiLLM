"""Extractor noise must not abort graph resolution for an otherwise valid paper."""

from types import SimpleNamespace
from uuid import UUID

import pytest

from apps.knowledge_graph.extraction.pipeline import collect_document_evidence
from apps.knowledge_graph.extraction.windows import ExtractionWindow, SpanMappingError
from lib.knowledge_graph.types import EntityCandidate, ExtractionBatchResult, RelationCandidate


@pytest.mark.parametrize("noise", ["...", "—", "‘’", "_", "\u200b"])
def test_unresolvable_labels_and_their_relations_are_excluded(noise):
    source = f"{noise} uses C++ and α"
    labels = ((noise, "model"), ("C++", "dataset"), ("α", "model"))
    candidates = tuple(
        EntityCandidate(kind, text, source.index(text), source.index(text) + len(text), 0.9)
        for text, kind in labels
    )
    relation = RelationCandidate(
        "uses_dataset", noise, "C++", 0, len(noise),
        source.index("C++"), source.index("C++") + 3, 0.9,
    )
    result = ExtractionBatchResult(entities=candidates, relations=(relation,), diagnostics=())
    backend = SimpleNamespace(extract_batch=lambda *_args, **_kwargs: (result,))
    ontology = SimpleNamespace(relations={"uses_dataset": {
        "allowed_head_types": ("model",), "allowed_tail_types": ("dataset",),
        "direction": "directed",
    }})
    window = ExtractionWindow(1, UUID(int=1), source, 0, "text")

    evidence = collect_document_evidence(
        (window,), full_text=source, backend=backend, ontology=ontology,
        max_batch_count=1, max_batch_characters=100,
    )

    assert [(e.raw_text, e.start, e.end) for e in evidence.entities] == [
        ("C++", len(noise) + 6, len(noise) + 9),
        ("α", len(noise) + 14, len(noise) + 15),
    ]
    assert evidence.relations == ()
    assert evidence.diagnostic_counts == {
        "unresolvable_entity_label": 1, "unresolved_relation_endpoint": 1,
    }


def test_label_filter_does_not_hide_invalid_source_coordinates():
    result = ExtractionBatchResult(
        entities=(EntityCandidate("model", "...", 0, 3, 0.9),), relations=(), diagnostics=(),
    )
    backend = SimpleNamespace(extract_batch=lambda *_args, **_kwargs: (result,))
    window = ExtractionWindow(1, UUID(int=1), "ABC", 0, "text")

    with pytest.raises(SpanMappingError):
        collect_document_evidence(
            (window,), full_text="ABC", backend=backend,
            ontology=SimpleNamespace(relations={}),
            max_batch_count=1, max_batch_characters=100,
        )
