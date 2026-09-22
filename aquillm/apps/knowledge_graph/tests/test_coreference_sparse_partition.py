from __future__ import annotations

import uuid

from test_coreference import _mention, _ontology

from apps.knowledge_graph.resolution import DOCUMENT_RESOLVER_VERSION
from apps.knowledge_graph.resolution.coreference import (
    resolve_document_mentions,
)

DOCUMENT_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
OTHER_DOCUMENT_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")
CONTENT_OBJECT_ID = uuid.UUID("33333333-3333-4333-8333-333333333333")
RESOLVER_VERSION = DOCUMENT_RESOLVER_VERSION
MAX_DB_INTEGER = 2**63 - 1

def test_sparse_resolver_matches_exhaustive_adversarial_cluster_partition(monkeypatch):
    import apps.knowledge_graph.resolution.coreference as coreference

    acronym_text = (
        "Retrieval-Augmented Generation (RAG) is introduced. RAG is used later."
    )
    acronym_positions = [
        index
        for index in range(len(acronym_text))
        if acronym_text.startswith("RAG", index)
    ]
    mentions = (
        _mention("name-a", "Orion", "model", start=0, source_key="name-a"),
        _mention("name-b", "Orion", "model", start=20, source_key="name-b"),
        _mention(
            "alias",
            "Orion",
            "architecture",
            start=40,
            source_key="alias",
        ),
        _mention(
            "id-a",
            "Transformer paper",
            "paper",
            start=60,
            source_key="id-a",
            identifier="doi:10.5555/12345678",
        ),
        _mention(
            "id-b",
            "Attention Is All You Need",
            "publication",
            start=100,
            source_key="id-b",
            identifier="https://doi.org/10.5555/12345678",
        ),
        _mention(
            "conflict-a",
            "Atlas",
            "model",
            start=140,
            source_key="conflict-a",
            identifier="https://github.com/example/atlas-a",
        ),
        _mention(
            "conflict-bridge",
            "Atlas",
            "model",
            start=160,
            source_key="conflict-bridge",
        ),
        _mention(
            "conflict-b",
            "Atlas",
            "model",
            start=180,
            source_key="conflict-b",
            identifier="https://github.com/example/atlas-b",
        ),
        _mention("v1-a", "Nova v1", "model", start=200, source_key="v1-a"),
        _mention("v1-b", "Nova/v1", "model", start=220, source_key="v1-b"),
        _mention("v2", "Nova v2", "model", start=240, source_key="v2"),
        _mention(
            "pronoun-id",
            "it",
            "model",
            start=260,
            source_key="pronoun-id",
            identifier="https://github.com/example/comet",
        ),
        _mention(
            "pronoun-named",
            "Comet",
            "model",
            start=280,
            source_key="pronoun-named",
            identifier="https://github.com/example/comet",
        ),
        _mention(
            "pronoun-blocked-bridge",
            "Comet",
            "model",
            start=300,
            source_key="pronoun-blocked-bridge",
        ),
        _mention(
            "full",
            "Retrieval-Augmented Generation",
            start=0,
            source_text=acronym_text,
            source_key="acronym-source",
        ),
        _mention(
            "definition",
            "RAG",
            start=acronym_positions[0],
            source_text=acronym_text,
            source_key="acronym-source",
        ),
        _mention(
            "later",
            "RAG",
            start=acronym_positions[1],
            source_text=acronym_text,
            source_key="acronym-source",
        ),
        *tuple(
            _mention(
                f"unique-{index}",
                f"Unique {index}",
                start=340 + index * 20,
                source_key=f"unique-{index}",
            )
            for index in range(20)
        ),
    )

    monkeypatch.setattr(coreference, "_EXHAUSTIVE_PAIR_LIMIT", len(mentions))
    exhaustive = resolve_document_mentions(mentions, _ontology())
    monkeypatch.setattr(coreference, "_EXHAUSTIVE_PAIR_LIMIT", 0)
    sparse = resolve_document_mentions(mentions, _ontology())

    def signature(result):
        return {
            (
                frozenset(cluster.mention_ids),
                cluster.label,
                cluster.entity_type,
                cluster.identifier,
                cluster.version_signature,
            )
            for cluster in result.clusters
        }

    assert signature(sparse) == signature(exhaustive)
    assert len(sparse.decisions) < len(exhaustive.decisions)
