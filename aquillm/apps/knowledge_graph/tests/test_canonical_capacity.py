import json
from hashlib import sha256
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.resolution import canonical
from apps.knowledge_graph.tests.test_canonical_resolution import _entity


def test_registry_resolution_preserves_every_entity_beyond_read_envelope():
    entities = tuple(_entity(i, i % 104 + 1, f"entity {i}") for i in range(1, 10_002))

    result = canonical.resolve_canonical_entities(entities)

    assert len(result.components) == 10_001
    assert tuple(item.entity_ids[0] for item in result.components) == tuple(
        range(1, 10_002)
    )
    assert result.decisions == ()


def test_registry_input_adapter_accepts_corpus_larger_than_read_envelope():
    rows = tuple(
        SimpleNamespace(
            pk=i,
            collection_id=i % 104 + 1,
            artifact_id=i % 104 + 1,
            cluster_key=f"{i:064x}",
            label=f"entity {i}",
            normalized_label=f"entity {i}",
            entity_type="model",
            identifier="",
            version_signature="",
        )
        for i in range(1, 10_002)
    )

    inputs = canonical.build_canonical_inputs_from_provenance(rows, (), ())

    assert len(inputs) == 10_001
    assert inputs[-1].entity_id == 10_001


def test_dense_exact_block_keeps_all_audits_and_same_collection_conflicts():
    entities = tuple(_entity(i, i % 104 + 1, "common name") for i in range(1, 680))

    result = canonical.resolve_canonical_entities(entities)

    assert len(result.components) == 679
    assert len(result.decisions) == 230_181
    assert all(
        item.outcome is canonical.CanonicalOutcome.REJECTED for item in result.decisions
    )


def test_registry_still_rejects_entity_overflow_without_truncation(monkeypatch):
    monkeypatch.setattr(canonical, "MAX_CANONICAL_REBUILD_ENTITIES", 2)
    entities = tuple(_entity(i, i, f"entity {i}") for i in range(1, 4))

    with pytest.raises(ValueError, match="entity cap"):
        canonical.resolve_canonical_entities(entities)


def test_corpus_capacity_does_not_expand_authorized_lookup_seed_envelope():
    with pytest.raises(ValueError):
        canonical.project_authorized_canonical_lookup(
            seed_collection_entity_ids=tuple(range(1, 10_002)),
            permission_endpoints=(),
            canonical_memberships=(),
            allowed_collection_ids=(),
            allowed_document_ids=(),
            active_artifact_ids=(),
        )


def test_streamed_resolution_checksum_preserves_existing_canonical_json_bytes():
    from apps.knowledge_graph.resolution import canonical_validation

    components = [{"label": "caf\u00e9", "nested": {"z": 2, "a": 1}}, {"id": 2}]
    decisions = [{"score": 0.99, "metadata": [["key", "value"]]}]
    expected = sha256(
        json.dumps(
            {
                "resolver_version": "v1",
                "components": components,
                "decisions": decisions,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    assert (
        canonical_validation._hash_resolution_records(
            resolver_version="v1",
            components=iter(components),
            decisions=iter(decisions),
        )
        == expected
    )


def test_rejected_embedding_audits_share_the_exact_decision_budget(monkeypatch):
    monkeypatch.setattr(canonical, "MAX_CANONICAL_DECISIONS", 1)
    entities = (
        _entity(1, 1, "first", identifier="repository:github.com/acme/first"),
        _entity(2, 2, "first", identifier="repository:github.com/acme/first"),
        _entity(3, 3, "second", identifier="repository:github.com/acme/second"),
    )
    candidate = canonical.CanonicalEmbeddingCandidate(
        1,
        3,
        0.99,
        "model-v1",
        "a" * 64,
        "b" * 64,
    )
    with pytest.raises(ValueError, match="decision cap"):
        canonical.resolve_canonical_entities(
            entities, embedding_candidates=(candidate,)
        )
