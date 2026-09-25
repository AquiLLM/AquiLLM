from __future__ import annotations

import uuid

from test_coreference import (
    _cluster_ids,
    _decision,
    _mention,
    _ontology,
)

from apps.knowledge_graph.resolution import DOCUMENT_RESOLVER_VERSION
from apps.knowledge_graph.resolution.coreference import (
    resolve_document_mentions,
)

DOCUMENT_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
OTHER_DOCUMENT_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")
CONTENT_OBJECT_ID = uuid.UUID("33333333-3333-4333-8333-333333333333")
RESOLVER_VERSION = DOCUMENT_RESOLVER_VERSION
MAX_DB_INTEGER = 2**63 - 1

def test_acronym_definitions_do_not_propagate_to_another_source_coordinate_space():
    text = "Retrieval-Augmented Generation (RAG) is defined."
    full = "Retrieval-Augmented Generation"
    definition = text.index("RAG")
    result = resolve_document_mentions(
        (
            _mention(
                "full",
                full,
                start=0,
                source_text=text,
                source_key="document:text",
            ),
            _mention(
                "definition",
                "RAG",
                start=definition,
                source_text=text,
                source_key="document:text",
            ),
            _mention(
                "figure",
                "RAG",
                start=0,
                source_text="RAG",
                source_key="figure:one",
                position_basis="chunk_content",
                content_object_id=CONTENT_OBJECT_ID,
            ),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {
        frozenset(("full", "definition")),
        frozenset(("figure",)),
    }
    assert _decision(result, "definition", "figure").method == "source_mismatch"
    assert _decision(result, "full", "figure").method == "source_mismatch"


def test_document_global_acronym_definition_applies_across_text_chunks():
    definition_text = "Retrieval-Augmented Generation (RAG) is defined."
    full_text = "Retrieval-Augmented Generation"
    definition_offset = 100
    acronym_start = definition_offset + definition_text.index("RAG")
    later_text = "Later, RAG is evaluated."
    later_offset = 500
    later_start = later_offset + later_text.index("RAG")
    result = resolve_document_mentions(
        (
            _mention(
                "full",
                full_text,
                start=definition_offset,
                source_text=definition_text,
                source_offset=definition_offset,
                source_key="text-chunk:definition",
                chunk_id=1,
            ),
            _mention(
                "definition",
                "RAG",
                start=acronym_start,
                source_text=definition_text,
                source_offset=definition_offset,
                source_key="text-chunk:definition",
                chunk_id=1,
            ),
            _mention(
                "later",
                "RAG",
                start=later_start,
                source_text=later_text,
                source_offset=later_offset,
                source_key="text-chunk:later",
                chunk_id=2,
            ),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {frozenset(("full", "definition", "later"))}
    assert _decision(result, "full", "later").accepted is True
    assert _decision(result, "full", "later").method == "defined_acronym"


def test_document_global_definition_can_use_overlapping_context_windows():
    document_text = "Retrieval-Augmented Generation (RAG) is defined."
    full_text = "Retrieval-Augmented Generation"
    document_offset = 100
    acronym_start = document_offset + document_text.index("RAG")
    acronym_window_start = document_text.index("Generation")
    result = resolve_document_mentions(
        (
            _mention(
                "full",
                full_text,
                start=document_offset,
                source_text=document_text,
                source_offset=document_offset,
                source_key="window:full",
                chunk_id=1,
            ),
            _mention(
                "definition",
                "RAG",
                start=acronym_start,
                source_text=document_text[acronym_window_start:],
                source_offset=document_offset + acronym_window_start,
                source_key="window:acronym",
                chunk_id=2,
            ),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {frozenset(("full", "definition"))}
    assert _decision(result, "full", "definition").method == "defined_acronym"
