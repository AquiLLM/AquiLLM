from __future__ import annotations

import pytest
from test_coreference import _cluster_ids, _mapping_mention, _mention, _ontology

from apps.knowledge_graph.extraction.pipeline import (
    DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS,
    ExtractionCapacityCode,
    ExtractionCapacityError,
)
from apps.knowledge_graph.resolution.coreference import (
    resolution_input_fingerprint,
    resolve_document_mentions,
)


@pytest.mark.parametrize("context_count", [1, 10])
def test_resolver_preserves_tail_evidence_at_extraction_character_capacity(
    context_count,
):
    context_length = DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS // context_count
    definition = "Retrieval Augmented Generation (RAG). RAG"
    mentions = []
    for index in range(context_count):
        text = " " * (context_length - len(definition)) + definition
        offset = index * context_length
        for label, local_start, suffix in (
            ("Retrieval Augmented Generation", text.index("Retrieval"), "full"),
            ("RAG", text.index("RAG"), "definition"),
            ("RAG", text.rindex("RAG"), "later"),
        ):
            mentions.append(
                _mapping_mention(
                    mention_id=f"{index}-{suffix}",
                    raw_text=label,
                    entity_type="method",
                    start=offset + local_start,
                    end=offset + local_start + len(label),
                    source_text=text,
                    source_key=f"chunk:{index}",
                    source_offset=offset,
                    chunk_id=index + 1,
                )
            )

    result = resolve_document_mentions(mentions, _ontology())

    assert _cluster_ids(result) == {frozenset(item["mention_id"] for item in mentions)}
    assert result.input_fingerprint == resolution_input_fingerprint(mentions)
    # Changing evidence at the very end must remain visible to snapshot validation.
    changed = [
        {**item, "source_text": item["source_text"][:-1] + "X"}
        if item["source_key"] == f"chunk:{context_count - 1}"
        else item
        for item in mentions
    ]
    assert result.input_fingerprint != resolution_input_fingerprint(changed)


def test_document_mention_accepts_source_context_above_old_individual_limit():
    text = " " * 1_000_001 + "Orion"
    mention = _mention("tail", "Orion", "model", start=1_000_001, source_text=text)

    assert _cluster_ids(resolve_document_mentions((mention,), _ontology())) == {
        frozenset(("tail",))
    }


@pytest.mark.parametrize(
    "operation",
    [
        resolution_input_fingerprint,
        lambda items: resolve_document_mentions(items, _ontology()),
    ],
)
@pytest.mark.parametrize("context_count", [1, 11])
def test_source_capacity_excess_has_typed_terminal_character_diagnostic(
    operation, context_count
):
    context_length = DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS // context_count + 1
    mentions = tuple(
        _mapping_mention(
            mention_id=f"mention-{index}",
            source_text="x" * context_length,
            source_key=f"chunk:{index}",
            chunk_id=index + 1,
        )
        for index in range(context_count)
    )

    with pytest.raises(ExtractionCapacityError) as error:
        operation(mentions)

    assert error.value.code is ExtractionCapacityCode.CHARACTER_LIMIT


def test_resolution_input_fingerprint_rejects_excess_unique_source_context():
    mentions = tuple(
        _mention(
            f"mention-{index}",
            f"Entity {index}",
            "model",
            start=index * 20,
            source_text=(chr(ord("a") + index) * 700_001),
            source_key=f"unique-context:{index}",
        )
        for index in range(DOCUMENT_EXTRACTION_V1_MAX_CHARACTERS // 700_001 + 1)
    )

    with pytest.raises(
        ExtractionCapacityError, match="aggregate.*source context"
    ) as error:
        resolution_input_fingerprint(mentions)
    assert error.value.code is ExtractionCapacityCode.CHARACTER_LIMIT
