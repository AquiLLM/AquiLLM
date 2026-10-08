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


def _active_artifacts(count):
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import GraphArtifact
    from apps.knowledge_graph.tests.test_canonical_resolution import (
        _embedding_signature,
    )

    artifacts = tuple(
        GraphArtifact.objects.create(
            scope_type="collection",
            scope_id=Collection.objects.create(name=f"capacity fixture {index}").pk,
            status="building",
            source_hash=f"{index:064x}",
            ontology_version="ontology-v1",
            extractor_version="extractor-v1",
            resolver_version="collection-resolution-v1",
            filter_policy_version="filter-v1",
            embedding_model_signature=_embedding_signature(),
        )
        for index in range(1, count + 1)
    )
    GraphArtifact.objects.filter(pk__in=[row.pk for row in artifacts]).update(
        status="active"
    )
    return artifacts


@pytest.mark.django_db
def test_global_registry_rebuild_preserves_all_196_active_collection_artifacts():
    artifacts = _active_artifacts(196)

    result = canonical.rebuild_canonical_registry()

    assert result.active_artifact_ids == tuple(sorted(row.pk for row in artifacts))
    assert result.canonical_entity_ids == ()
    assert result.active_link_count == 0


def test_rebuild_summary_accepts_global_artifact_scope_beyond_read_envelope():
    artifacts = tuple(range(1, 197))
    result = canonical.CanonicalRebuildResult(
        resolver_version="canonical-resolution-v1",
        resolution_checksum="a" * 64,
        active_artifact_ids=artifacts,
        canonical_entity_ids=(),
        active_link_count=0,
        created_entity_count=0,
        created_link_count=0,
        superseded_entity_count=0,
        superseded_link_count=0,
    )

    assert result.active_artifact_ids == artifacts


@pytest.mark.django_db
def test_global_artifact_overflow_refuses_rebuild_without_truncation(monkeypatch):
    _active_artifacts(3)
    monkeypatch.setattr(canonical, "MAX_CANONICAL_REBUILD_ARTIFACTS", 2, raising=False)

    with pytest.raises(ValueError, match="active collection artifacts exceed"):
        canonical.rebuild_canonical_registry()


@pytest.mark.parametrize("field", ["allowed_collection_ids", "active_artifact_ids"])
def test_global_artifact_capacity_does_not_expand_authorized_read_scopes(field):
    scopes = {"allowed_collection_ids": (), "active_artifact_ids": ()}
    scopes[field] = tuple(range(1, 130))

    with pytest.raises(ValueError, match="positive integer envelope"):
        canonical.project_authorized_canonical_lookup(
            seed_collection_entity_ids=(),
            permission_endpoints=(),
            canonical_memberships=(),
            allowed_document_ids=(),
            **scopes,
        )
