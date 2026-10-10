from __future__ import annotations

import uuid

import pytest
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


@pytest.mark.parametrize("definition", ["undefined", "defined", "ambiguous"])
def test_collision_enumeration_matches_complete_memberships_and_cannot_links(
    monkeypatch, definition
):
    import apps.knowledge_graph.resolution.coreference as coreference

    labels = (
        ("pre", "RAG", "method", ""),
        ("full", "Rapid alpha generation", "method", ""),
        ("alias", "Rapid alpha generation", "approach", ""),
        ("other", "Rapid alternate generation", "method", ""),
        ("id-a", "Rapid associated generation", "method", "doi:10.5555/12345678"),
        ("id-b", "Rapid affiliated generation", "method", "doi:10.5555/12345678"),
        ("conflict-a", "Rapid argued generation", "method", "doi:10.5555/12345678"),
        ("bridge", "Rapid argued generation", "method", ""),
        ("conflict-b", "Rapid argued generation", "method", "doi:10.5555/87654321"),
    )
    mentions = [
        _mention(
            key,
            label,
            entity_type,
            start=1000 + index * 100,
            identifier=identifier,
            source_key=key,
        )
        for index, (key, label, entity_type, identifier) in enumerate(labels)
    ]
    mentions.append(
        _mention(
            "foreign-acronym",
            "RAG",
            start=10_000,
            chunk_id=2,
            position_basis="chunk_content",
            content_object_id=CONTENT_OBJECT_ID,
        )
    )
    text = "RAG. Rapid alpha generation (RAG). RAG. rag."
    if definition == "ambiguous":
        text += " Rapid alternate generation (RAG)."
    if definition != "undefined":
        for key, label, start in (
            ("source-pre", "RAG", 0),
            ("source-full", "Rapid alpha generation", text.index("Rapid")),
            ("source-definition", "RAG", text.index("(RAG") + 1),
            ("source-later", "RAG", text.index("RAG.", 4)),
            ("source-lower", "rag", text.index("rag")),
        ):
            mentions.append(
                _mention(
                    key,
                    label,
                    start=start,
                    source_text=text,
                    source_key="definition-text",
                )
            )
        if definition == "ambiguous":
            mentions.extend(
                (
                    _mention(
                        "ambiguous-full",
                        "Rapid alternate generation",
                        start=text.rindex("Rapid"),
                        source_text=text,
                        source_key="definition-text",
                    ),
                    _mention(
                        "ambiguous-definition",
                        "RAG",
                        start=text.rindex("RAG"),
                        source_text=text,
                        source_key="definition-text",
                    ),
                )
            )

    monkeypatch.setattr(coreference, "_EXHAUSTIVE_PAIR_LIMIT", len(mentions))
    exhaustive = resolve_document_mentions(mentions, _ontology())
    monkeypatch.setattr(coreference, "_EXHAUSTIVE_PAIR_LIMIT", 0)
    sparse = resolve_document_mentions(mentions, _ontology())

    # Full immutable clusters include representative, canonical key, confidence,
    # provenance tree, membership reasons, and their deterministic ordering.
    assert sparse.clusters == exhaustive.clusters
    assert sparse.mention_ids == exhaustive.mention_ids
    assert sparse.input_fingerprint == exhaustive.input_fingerprint

    def meaningful(result):
        return tuple(
            decision
            for decision in result.decisions
            if decision.method != "normalized_name_mismatch"
        )

    assert meaningful(sparse) == meaningful(exhaustive)
    assert resolve_document_mentions(reversed(mentions), _ontology()) == sparse


def test_collision_enumeration_preserves_full_form_version_cannot_links(monkeypatch):
    import apps.knowledge_graph.resolution.coreference as coreference

    mentions = (
        _mention("v1", "Rapid alpha generation v1", start=0),
        _mention("v2", "Rapid alpha generation v2", start=100),
        _mention("other", "Rapid alternate generation v1", start=200),
        _mention("acronym", "RAGV", start=300),
    )
    monkeypatch.setattr(coreference, "_EXHAUSTIVE_PAIR_LIMIT", len(mentions))
    exhaustive = resolve_document_mentions(mentions, _ontology())
    monkeypatch.setattr(coreference, "_EXHAUSTIVE_PAIR_LIMIT", 0)
    sparse = resolve_document_mentions(mentions, _ontology())
    assert sparse.clusters == exhaustive.clusters
    assert tuple(
        d for d in sparse.decisions if d.method != "normalized_name_mismatch"
    ) == tuple(
        d for d in exhaustive.decisions if d.method != "normalized_name_mismatch"
    )
    assert any(d.method == "version_mismatch" for d in sparse.decisions)
