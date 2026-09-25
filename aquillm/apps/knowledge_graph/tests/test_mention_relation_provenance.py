from __future__ import annotations

import uuid

import pytest
from test_mention_extraction import _Backend, _ontology, _result, _window

from apps.knowledge_graph.extraction.pipeline import collect_document_evidence

ONTOLOGY_PATH = (
    __import__("pathlib").Path(__file__).parents[1] / "ontologies" / "research-v1.yaml"
)
DOCUMENT_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")


def test_relation_model_accepts_an_endpoint_represented_by_an_overlapping_observation():
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import (
        EntityMention,
        GraphArtifact,
        RelationMention,
    )

    artifact = GraphArtifact(
        pk=1,
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=DOCUMENT_ID,
        status=GraphArtifact.Status.BUILDING,
    )

    representative_chunk = TextChunk(pk=10, doc_id=DOCUMENT_ID, modality="text")
    relation_chunk = TextChunk(
        pk=11,
        doc_id=DOCUMENT_ID,
        modality="text",
        content="Orion uses MMLU",
        start_position=0,
    )
    observation = {
        "chunk_id": 11,
        "position_basis": "document_global",
        "start": 0,
        "end": 5,
        "local_start": 0,
        "local_end": 5,
        "modality": "text",
    }
    head = EntityMention(
        pk=20,
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=representative_chunk,
        start=0,
        end=5,
        position_basis="document_global",
        raw_text="Orion",
        metadata={"observations": [observation]},
    )
    tail = EntityMention(
        pk=21,
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=representative_chunk,
        start=11,
        end=15,
        position_basis="document_global",
        raw_text="MMLU",
        metadata={
            "observations": [
                {
                    **observation,
                    "start": 11,
                    "end": 15,
                    "local_start": 11,
                    "local_end": 15,
                }
            ]
        },
    )
    relation = RelationMention(
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=relation_chunk,
        head=head,
        tail=tail,
        relation_type="uses_dataset",
        extraction_confidence=0.9,
    )

    relation.clean()


def test_relation_model_rejects_unrelated_or_image_endpoint_provenance():
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import (
        EntityMention,
        GraphArtifact,
        RelationMention,
    )

    artifact = GraphArtifact(
        pk=1,
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=DOCUMENT_ID,
        status=GraphArtifact.Status.BUILDING,
    )

    representative_chunk = TextChunk(pk=10, doc_id=DOCUMENT_ID, modality="text")
    relation_chunk = TextChunk(pk=99, doc_id=DOCUMENT_ID, modality="text")
    endpoint = EntityMention(
        pk=20,
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=representative_chunk,
        start=0,
        end=5,
        position_basis="document_global",
        metadata={
            "observations": [
                {
                    "chunk_id": 99,
                    "position_basis": "chunk_content",
                    "start": 0,
                    "end": 5,
                    "modality": "image",
                }
            ]
        },
    )
    relation = RelationMention(
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=relation_chunk,
        head=endpoint,
        tail=EntityMention(
            pk=21,
            artifact=artifact,
            document_id=DOCUMENT_ID,
            chunk=representative_chunk,
            start=11,
            end=15,
            position_basis="document_global",
            metadata={},
        ),
        relation_type="uses_dataset",
        extraction_confidence=0.9,
    )

    with pytest.raises(Exception, match="chunk|observation|provenance"):
        relation.clean()


def test_relation_model_rejects_spoofed_overlap_offsets_for_an_unrelated_chunk():
    from django.core.exceptions import ValidationError

    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import (
        EntityMention,
        GraphArtifact,
        RelationMention,
    )

    artifact = GraphArtifact(
        pk=1,
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=DOCUMENT_ID,
        status=GraphArtifact.Status.BUILDING,
    )

    primary = TextChunk(pk=10, doc_id=DOCUMENT_ID, modality="text")
    unrelated = TextChunk(
        pk=11,
        doc_id=DOCUMENT_ID,
        modality="text",
        content="Elsewhere in this document",
        start_position=100,
    )
    metadata = {
        "observations": [
            {
                "chunk_id": 11,
                "position_basis": "document_global",
                "start": 0,
                "end": 5,
                "local_start": 0,
                "local_end": 5,
                "modality": "text",
            }
        ]
    }
    head = EntityMention(
        pk=20,
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=primary,
        start=0,
        end=5,
        position_basis="document_global",
        raw_text="Orion",
        metadata=metadata,
    )
    tail = EntityMention(
        pk=21,
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=primary,
        start=6,
        end=10,
        position_basis="document_global",
        raw_text="MMLU",
        metadata={
            "observations": [
                {
                    **metadata["observations"][0],
                    "start": 6,
                    "end": 10,
                    "local_start": 6,
                    "local_end": 10,
                }
            ]
        },
    )
    relation = RelationMention(
        artifact=artifact,
        document_id=DOCUMENT_ID,
        chunk=unrelated,
        head=head,
        tail=tail,
        relation_type="uses_dataset",
        extraction_confidence=0.9,
    )

    with pytest.raises(ValidationError, match="observation|evidence"):
        relation.clean()


def test_raw_overlap_can_exceed_unique_entity_cap_after_bounded_deduplication(
    monkeypatch,
):
    from apps.knowledge_graph.extraction import pipeline

    monkeypatch.setattr(pipeline, "DOCUMENT_EXTRACTION_V1_MAX_ENTITIES", 2)
    full_text = "prefix Orion uses MMLU in evaluations."

    evidence = collect_document_evidence(
        (
            _window(10, "Orion uses MMLU", 7),
            _window(11, "prefix Orion uses MMLU", 0),
        ),
        full_text=full_text,
        backend=_Backend(
            (
                _result(model_start=0, dataset_start=11, confidence=0.82),
                _result(model_start=7, dataset_start=18, confidence=0.96),
            )
        ),
        ontology=_ontology(),
        max_batch_count=1,
        max_batch_characters=100,
    )

    assert len(evidence.entities) == 2
