"""Offline regressions for remediated extraction/resolution audit findings.

Run from the repository root with:
    rtk proxy python docs/audits/2026-09-21-knowledge-graph/extraction_probes.py

These probes import pure production functions. They do not initialize Django,
access a database, load a model, or contact an embedding provider.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "aquillm"))

from apps.knowledge_graph.extraction.pipeline import (  # noqa: E402
    collect_document_evidence,
)
from apps.knowledge_graph.extraction.windows import ExtractionWindow  # noqa: E402
from apps.knowledge_graph.resolution import collection as resolution  # noqa: E402
from apps.knowledge_graph.services.ontology import (  # noqa: E402
    load_ontology,
    load_ontology_yaml,
)
from lib.knowledge_graph.extractors.gliner2_local import (  # noqa: E402
    _normalize_relations,
)
from lib.knowledge_graph.types import (  # noqa: E402
    EntityCandidate,
    ExtractionBatchResult,
    RelationCandidate,
)

ONTOLOGY_PATH = ROOT / "aquillm/apps/knowledge_graph/ontologies/research-v1.yaml"


def probe_undirected_reverse_orientation() -> None:
    raw = ONTOLOGY_PATH.read_text(encoding="utf-8")
    before = (
        "description: Connects a paper to one of its authors.\n    direction: directed"
    )
    assert before in raw
    ontology = load_ontology_yaml(
        raw.replace(before, before.replace("directed", "undirected"))
    )
    text = "Alice Paper"
    entities = (
        EntityCandidate("author", "Alice", 0, 5, 0.9),
        EntityCandidate("paper", "Paper", 6, 11, 0.9),
    )
    author = {"text": "Alice", "start": 0, "end": 5, "confidence": 0.9}
    paper = {"text": "Paper", "start": 6, "end": 11, "confidence": 0.9}

    def normalized(head, tail):
        return _normalize_relations(
            {"authored_by": [{"head": head, "tail": tail}]},
            text=text,
            input_index=0,
            ontology_relations={"authored_by": ontology.relations["authored_by"]},
            entities=entities,
        )

    forward, forward_diagnostics = normalized(paper, author)
    reverse, reverse_diagnostics = normalized(author, paper)
    assert ontology.relations["authored_by"].direction == "undirected"
    assert len(forward) == 1 and not forward_diagnostics
    assert len(reverse) == 1 and not reverse_diagnostics

    candidate = RelationCandidate("authored_by", "Alice", "Paper", 0, 5, 6, 11, 0.9)
    backend = SimpleNamespace(
        extract_batch=lambda *args, **kwargs: (
            ExtractionBatchResult(entities, (candidate,), ()),
        )
    )
    evidence = collect_document_evidence(
        (ExtractionWindow(1, uuid.UUID(int=1), text, 0, "text"),),
        full_text=text,
        backend=backend,
        ontology=ontology,
        max_batch_count=1,
        max_batch_characters=100,
    )
    assert len(evidence.relations) == 1
    assert evidence.diagnostic_counts == {}
    print(
        json.dumps(
            {
                "probe": "undirected_reverse_orientation",
                "forward_provider_relations": len(forward),
                "reverse_provider_relations": len(reverse),
                "reverse_provider_diagnostics": [
                    item.code for item in reverse_diagnostics
                ],
                "reverse_pipeline_relations": len(evidence.relations),
                "reverse_pipeline_diagnostics": evidence.diagnostic_counts,
            },
            sort_keys=True,
        )
    )


def probe_cartesian_representative_scan() -> None:
    ontology = load_ontology(ONTOLOGY_PATH)
    signature = (
        "test:model@revision:endpoint="
        + "e" * 64
        + ":dims=1024:prep=kg-entity-v1:max_chars=8192:batch=64"
    )
    for count in (100, 300, 1000):
        config = resolution.CollectionResolutionConfig()
        documents = tuple(
            resolution.CollectionSnapshotInput(
                index,
                index,
                uuid.UUID(int=index),
                "a" * 64,
                "b" * 64,
                "c" * 64,
            )
            for index in range(1, count + 1)
        )
        snapshot = resolution.CollectionBuildSnapshot(
            50000,
            7,
            documents,
            "a" * 64,
            ontology.version,
            ontology.checksum,
            "c" * 64,
            resolution.resolution_config_checksum(config),
        )
        # Exactly two distinct document entities per document: ordinary upstream
        # coreference cannot merge the different labels in this fixture.
        entities = tuple(
            resolution.DocumentEntityInput(
                index + 1,
                f"{index + 1:064x}",
                index % count + 1,
                uuid.UUID(int=index % count + 1),
                "Aquila" if index < count else "Falcon",
                "aquila" if index < count else "falcon",
                "model",
            )
            for index in range(2 * count)
        )

        def backend(texts):
            return resolution.SignedEmbeddingBatch(
                vectors=tuple((1.0,) + (0.0,) * 1023 for _ in texts),
                text_hashes=tuple(
                    resolution.embedding_text_hash(text) for text in texts
                ),
                indices=tuple(range(len(texts))),
                model_signature=signature,
            )

        session = resolution.CollectionEmbeddingSession(
            expected_model_signature=signature,
            backend=backend,
        )
        original_pair = resolution._pair
        pair_calls = 0

        def counted_pair(left, right):
            nonlocal pair_calls
            pair_calls += 1
            return original_pair(left, right)

        resolution._pair = counted_pair
        started = time.perf_counter()
        try:
            result = resolution.resolve_collection_entities(
                snapshot,
                entities,
                ontology,
                embedding_session=session,
            )
        finally:
            resolution._pair = original_pair
        assert result.audit.embedding_candidate_pair_count == 1
        assert pair_calls < count * 10
        print(
            json.dumps(
                {
                    "probe": "cartesian_representative_scan",
                    "documents": count,
                    "members_per_group": count,
                    "semantic_candidate_pairs": 1,
                    "pair_calls": pair_calls,
                    "linear_bound": count * 10,
                    "instrumented_seconds": round(time.perf_counter() - started, 4),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    probe_undirected_reverse_orientation()
    probe_cartesian_representative_scan()
