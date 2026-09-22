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

def test_chunk_content_acronym_definition_stays_with_its_content_object():
    definition_text = "Retrieval-Augmented Generation (RAG) is defined."
    full_text = "Retrieval-Augmented Generation"
    acronym_start = definition_text.index("RAG")
    result = resolve_document_mentions(
        (
            _mention(
                "full",
                full_text,
                start=0,
                source_text=definition_text,
                source_key="figure:first",
                position_basis="chunk_content",
                content_object_id=CONTENT_OBJECT_ID,
            ),
            _mention(
                "definition",
                "RAG",
                start=acronym_start,
                source_text=definition_text,
                source_key="figure:first",
                position_basis="chunk_content",
                content_object_id=CONTENT_OBJECT_ID,
            ),
            _mention(
                "other-figure",
                "RAG",
                start=0,
                source_text="RAG",
                source_key="figure:second",
                position_basis="chunk_content",
                content_object_id=OTHER_DOCUMENT_ID,
            ),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {
        frozenset(("full", "definition")),
        frozenset(("other-figure",)),
    }
    assert _decision(result, "definition", "other-figure").method == "source_mismatch"


def test_chunk_content_acronym_ambiguity_is_scoped_to_each_content_object():
    first_text = "Retrieval-Augmented Generation (RAG). Later RAG."
    second_text = "Red Amber Green (RAG). Later RAG."
    first_full = "Retrieval-Augmented Generation"
    second_full = "Red Amber Green"
    first_positions = [
        index for index in range(len(first_text)) if first_text.startswith("RAG", index)
    ]
    second_positions = [
        index
        for index in range(len(second_text))
        if second_text.startswith("RAG", index)
    ]
    common = {"position_basis": "chunk_content", "source_offset": 0}
    result = resolve_document_mentions(
        (
            _mention(
                "first-full",
                first_full,
                start=0,
                source_text=first_text,
                source_key="figure:first",
                content_object_id=CONTENT_OBJECT_ID,
                **common,
            ),
            _mention(
                "first-definition",
                "RAG",
                start=first_positions[0],
                source_text=first_text,
                source_key="figure:first",
                content_object_id=CONTENT_OBJECT_ID,
                **common,
            ),
            _mention(
                "first-later",
                "RAG",
                start=first_positions[1],
                source_text=first_text,
                source_key="figure:first",
                content_object_id=CONTENT_OBJECT_ID,
                **common,
            ),
            _mention(
                "second-full",
                second_full,
                start=0,
                source_text=second_text,
                source_key="figure:second",
                content_object_id=OTHER_DOCUMENT_ID,
                **common,
            ),
            _mention(
                "second-definition",
                "RAG",
                start=second_positions[0],
                source_text=second_text,
                source_key="figure:second",
                content_object_id=OTHER_DOCUMENT_ID,
                **common,
            ),
            _mention(
                "second-later",
                "RAG",
                start=second_positions[1],
                source_text=second_text,
                source_key="figure:second",
                content_object_id=OTHER_DOCUMENT_ID,
                **common,
            ),
        ),
        _ontology(),
    )

    assert _cluster_ids(result) == {
        frozenset(("first-full", "first-definition", "first-later")),
        frozenset(("second-full", "second-definition", "second-later")),
    }
