from __future__ import annotations

import ast
import math
import os
import socket
import struct
import uuid
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
from django.apps import apps
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db.models import CheckConstraint, UniqueConstraint

from apps.knowledge_graph.resolution.collection import (
    AliasEvidence,
    CollectionBuildSnapshot,
    CollectionEmbeddingSession,
    CollectionResolutionConfig,
    CollectionSnapshotInput,
    DocumentEntityInput,
    SignedEmbeddingBatch,
    SupportedRelation,
    _expected_persisted_link_count,
    _semantic_acronym_automatic_allowed,
    embedding_text_hash,
    resolution_config_checksum,
    resolve_collection_entities,
)
from apps.knowledge_graph.resolution.scoring import (
    EMBEDDING_DIMENSIONS,
    ResolutionOutcome,
    ResolutionThresholds,
    ResolutionTier,
    validate_embedding,
)

_EMBEDDING_SIGNATURE = (
    f"test-local:model@revision:endpoint={'e' * 64}:dims=1024:"
    "prep=kg-entity-v1:max_chars=8192:batch=64"
)

_ENTITY_PK_BY_NAME = {
    "a": 1,
    "b": 2,
    "c": 3,
    "model": 4,
    "method": 5,
    "dataset-a": 6,
    "dataset-b": 7,
}


def _entity_pk(value: str | int) -> int:
    return value if type(value) is int else _ENTITY_PK_BY_NAME[value]


def _database_is_reachable() -> bool:
    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)), timeout=0.2
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _database_is_reachable() and os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
    reason="configured PostgreSQL database is not reachable",
)


@lru_cache(maxsize=1)
def _ontology():
    from apps.knowledge_graph.services.ontology import load_ontology

    path = Path(__file__).resolve().parents[1] / "ontologies" / "research-v1.yaml"
    return load_ontology(path)


def _snapshot(
    *artifact_ids: int,
    config: CollectionResolutionConfig | None = None,
    filter_checksum: str = "c" * 64,
) -> CollectionBuildSnapshot:
    artifact_ids = artifact_ids or (201,)
    return CollectionBuildSnapshot(
        destination_artifact_id=101,
        collection_id=7,
        inputs=tuple(
            CollectionSnapshotInput(
                manifest_input_id=1000 + artifact_id,
                document_artifact_id=artifact_id,
                document_id=uuid.UUID(int=artifact_id),
                membership_signature=f"{artifact_id + 2:064x}"[-64:],
                source_signature=f"{artifact_id:064x}"[-64:],
                build_signature=f"{artifact_id + 1:064x}"[-64:],
            )
            for artifact_id in artifact_ids
        ),
        source_hash="a" * 64,
        ontology_version=_ontology().version,
        ontology_checksum=_ontology().checksum,
        filter_policy_checksum=filter_checksum,
        resolution_config_checksum=resolution_config_checksum(
            CollectionResolutionConfig() if config is None else config
        ),
    )


def _document_entity(
    entity_id: str | int, label: str, **overrides
) -> DocumentEntityInput:
    entity_pk = _entity_pk(entity_id)
    values = {
        "entity_id": entity_pk,
        "document_cluster_key": f"{entity_pk:064x}",
        "document_artifact_id": 201,
        "document_id": uuid.UUID(int=201),
        "label": label,
        "normalized_label": label.casefold(),
        "entity_type": "model",
        "identifier": "",
        "version_signature": "",
        "alias_evidence": (),
        "description": "",
        "extraction_confidence": 0.9,
    }
    values.update(overrides)
    return DocumentEntityInput(**values)


def _unit_vector(x: float, y: float = 0.0) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[0] = x
    vector[1] = y
    return vector


class _RecordingBackend:
    def __init__(self, vectors_by_text, signature=_EMBEDDING_SIGNATURE):
        self.vectors_by_text = vectors_by_text
        self.signature = signature
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, texts):
        self.calls.append(tuple(texts))
        return SignedEmbeddingBatch(
            vectors=tuple(tuple(self.vectors_by_text[text]) for text in texts),
            text_hashes=tuple(embedding_text_hash(text) for text in texts),
            indices=tuple(range(len(texts))),
            model_signature=self.signature,
        )


def _session(vectors_by_text, *, signature=_EMBEDDING_SIGNATURE):
    backend = _RecordingBackend(vectors_by_text, signature=signature)
    return (
        CollectionEmbeddingSession(
            expected_model_signature=signature,
            backend=backend,
        ),
        backend,
    )


def _cluster_memberships(result):
    return {
        frozenset(
            next(name for name, pk in _ENTITY_PK_BY_NAME.items() if pk == entity_id)
            for entity_id in cluster.document_entity_ids
        )
        for cluster in result.clusters
    }


def _decision(result, left: str | int, right: str | int):
    left_pk = _entity_pk(left)
    right_pk = _entity_pk(right)
    return next(
        decision
        for decision in result.decisions
        if {decision.left_entity_id, decision.right_entity_id} == {left_pk, right_pk}
    )


def test_stable_identifier_equality_is_first_tier_and_never_embeds():
    session, backend = _session({})
    result = resolve_collection_entities(
        _snapshot(),
        (
            _document_entity(
                "a", "Aquila", identifier="repository:github.com/acme/aquila"
            ),
            _document_entity(
                "b", "Project Night Sky", identifier="repository:github.com/acme/aquila"
            ),
        ),
        _ontology(),
        embedding_session=session,
    )

    assert _cluster_memberships(result) == {frozenset(("a", "b"))}
    decision = _decision(result, "a", "b")
    assert decision.outcome is ResolutionOutcome.AUTOMATIC
    assert decision.tier is ResolutionTier.STABLE_IDENTIFIER
    assert decision.resolution_confidence == 1.0
    assert decision.embedding_similarity is None
    assert backend.calls == []


def test_exact_normalized_label_or_known_alias_is_second_tier():
    session, _backend = _session({})
    result = resolve_collection_entities(
        _snapshot(),
        (
            _document_entity("a", "Retrieval Augmented Generation"),
            _document_entity(
                "b",
                "RAG",
                alias_evidence=(
                    AliasEvidence(
                        alias="Retrieval-Augmented Generation",
                        method="defined_acronym",
                        mention_id=22,
                    ),
                ),
            ),
        ),
        _ontology(),
        embedding_session=session,
    )

    assert _cluster_memberships(result) == {frozenset(("a", "b"))}
    decision = _decision(result, "a", "b")
    assert decision.outcome is ResolutionOutcome.AUTOMATIC
    assert decision.tier is ResolutionTier.EXACT_LABEL_OR_ALIAS


def test_same_defined_acronym_cannot_merge_different_full_forms_across_documents():
    session, backend = _session(
        {
            "Retrieval Augmented Generation": _unit_vector(1.0, 0.0),
            "Really Awesome Graph": _unit_vector(1.0, 0.0),
        }
    )
    result = resolve_collection_entities(
        _snapshot(201, 202),
        (
            _document_entity(
                "a",
                "Retrieval Augmented Generation",
                alias_evidence=(
                    AliasEvidence(
                        alias="RAG",
                        method="defined_acronym",
                        mention_id=21,
                    ),
                ),
            ),
            _document_entity(
                "b",
                "Really Awesome Graph",
                document_artifact_id=202,
                document_id=uuid.UUID(int=202),
                alias_evidence=(
                    AliasEvidence(
                        alias="RAG",
                        method="defined_acronym",
                        mention_id=31,
                    ),
                ),
            ),
        ),
        _ontology(),
        embedding_session=session,
    )

    assert _cluster_memberships(result) == {
        frozenset(("a",)),
        frozenset(("b",)),
    }
    decision = _decision(result, "a", "b")
    assert decision.outcome is ResolutionOutcome.CANDIDATE
    assert decision.tier is ResolutionTier.EMBEDDING
    assert "acronym_requires_shared_expansion" in decision.reason_codes
    assert backend.calls


def test_one_shared_acronym_binding_cannot_mask_another_shared_conflict():
    entities = {
        1: _document_entity(
            1,
            "Expansion X",
            alias_evidence=(
                AliasEvidence(alias="RAG", method="defined_acronym", mention_id=11),
            ),
        ),
        2: _document_entity(
            2,
            "Expansion Y",
            alias_evidence=(
                AliasEvidence(alias="ABC", method="defined_acronym", mention_id=12),
            ),
        ),
        3: _document_entity(
            3,
            "Expansion Z",
            alias_evidence=(
                AliasEvidence(alias="RAG", method="defined_acronym", mention_id=13),
            ),
        ),
        4: _document_entity(
            4,
            "Expansion Y",
            alias_evidence=(
                AliasEvidence(alias="ABC", method="defined_acronym", mention_id=14),
            ),
        ),
    }

    assert not _semantic_acronym_automatic_allowed(
        1,
        3,
        deterministic_groups={1: (1, 2), 3: (3, 4)},
        entities=entities,
    )


def test_distinct_stable_identifiers_block_label_or_alias_merges():
    left = _document_entity("a", "Atlas", identifier="arxiv:2401.00001")
    right = _document_entity("b", "Atlas", identifier="arxiv:2401.00002")
    session, backend = _session(
        {
            "Atlas": _unit_vector(1.0),
        }
    )

    result = resolve_collection_entities(
        _snapshot(),
        (left, right),
        _ontology(),
        embedding_session=session,
    )

    assert _cluster_memberships(result) == {
        frozenset(("a",)),
        frozenset(("b",)),
    }
    decision = _decision(result, "a", "b")
    assert decision.outcome is ResolutionOutcome.REJECTED
    assert "conflicting_stable_identifiers" in decision.reason_codes
    assert backend.calls == []


def test_ontology_type_compatibility_precedes_similarity():
    session, backend = _session({})
    result = resolve_collection_entities(
        _snapshot(),
        (
            _document_entity("model", "Atlas", entity_type="model"),
            _document_entity("method", "Atlas", entity_type="method"),
        ),
        _ontology(),
        embedding_session=session,
    )

    assert len(result.clusters) == 2
    assert backend.calls == []
    assert result.audit.type_incompatible_pair_count == 1


def test_ontology_alias_types_are_compatible_but_persist_canonical_type():
    session, _backend = _session({})
    result = resolve_collection_entities(
        _snapshot(),
        (
            _document_entity("a", "Aquila", entity_type="architecture"),
            _document_entity("b", "aquila", entity_type="model"),
        ),
        _ontology(),
        embedding_session=session,
    )

    assert len(result.clusters) == 1
    assert result.clusters[0].entity_type == "model"


def test_embedding_candidates_are_type_constrained_capped_and_audited():
    session, backend = _session(
        {
            "Aquila encoder": _unit_vector(1.0, 0.0),
            "Aquila representation model": _unit_vector(0.999, 0.02),
            "Completely different system": _unit_vector(0.0, 1.0),
        }
    )
    config = CollectionResolutionConfig(
        thresholds=ResolutionThresholds(
            automatic=0.95,
            candidate=0.75,
            retrieval_similarity=0.70,
        ),
        max_candidates_per_entity=1,
        embedding_weight=1.0,
        neighborhood_weight=0.0,
    )

    result = resolve_collection_entities(
        _snapshot(config=config),
        (
            _document_entity("a", "Aquila encoder"),
            _document_entity("b", "Aquila representation model"),
            _document_entity("c", "Completely different system"),
        ),
        _ontology(),
        config=config,
        embedding_session=session,
    )

    assert _cluster_memberships(result) == {
        frozenset(("a", "b")),
        frozenset(("c",)),
    }
    decision = _decision(result, "a", "b")
    assert decision.outcome is ResolutionOutcome.AUTOMATIC
    assert decision.tier is ResolutionTier.EMBEDDING
    assert decision.embedding_model_signature == session.expected_model_signature
    assert decision.embedding_similarity is not None
    assert decision.candidate_rank == 1
    assert result.audit.embedding_model_signature == session.expected_model_signature
    assert result.audit.max_candidates_per_entity == 1
    assert result.audit.max_observed_candidate_fanout <= 1
    assert len(backend.calls) == 1


def test_intermediate_similarity_is_candidate_not_automatic():
    session, _backend = _session(
        {
            "Atlas": _unit_vector(1.0, 0.0),
            "Atlas family": _unit_vector(0.82, 0.57),
        }
    )
    config = CollectionResolutionConfig(
        thresholds=ResolutionThresholds(
            automatic=0.95,
            candidate=0.75,
            retrieval_similarity=0.70,
        ),
        embedding_weight=1.0,
        neighborhood_weight=0.0,
    )
    result = resolve_collection_entities(
        _snapshot(config=config),
        (
            _document_entity("a", "Atlas"),
            _document_entity("b", "Atlas family"),
        ),
        _ontology(),
        config=config,
        embedding_session=session,
    )

    assert len(result.clusters) == 2
    assert _decision(result, "a", "b").outcome is ResolutionOutcome.CANDIDATE
    assert _expected_persisted_link_count(result) == 4


def test_supported_neighborhood_agreement_is_last_tier_and_can_raise_confidence():
    session, _backend = _session(
        {
            "Atlas encoder": _unit_vector(1.0, 0.0),
            "Atlas representation model": _unit_vector(0.88, 0.475),
        }
    )
    entities = (
        _document_entity("a", "Atlas encoder"),
        _document_entity("b", "Atlas representation model"),
        _document_entity("dataset-a", "MMLU", entity_type="dataset"),
        _document_entity("dataset-b", "mmlu", entity_type="benchmark"),
    )
    relations = (
        SupportedRelation(301, 1, "uses_dataset", 6, 0.9),
        SupportedRelation(302, 2, "uses_dataset", 7, 0.9),
    )

    config = CollectionResolutionConfig(
        thresholds=ResolutionThresholds(
            automatic=0.90,
            candidate=0.75,
            retrieval_similarity=0.70,
        ),
        embedding_weight=0.8,
        neighborhood_weight=0.2,
    )
    result = resolve_collection_entities(
        _snapshot(config=config),
        entities,
        _ontology(),
        relations=relations,
        config=config,
        embedding_session=session,
    )

    decision = _decision(result, "a", "b")
    assert decision.outcome is ResolutionOutcome.AUTOMATIC
    assert decision.tier is ResolutionTier.NEIGHBORHOOD_AGREEMENT
    assert decision.neighborhood_agreement == 1.0


def test_unsupported_or_unknown_relations_never_supply_neighborhood_agreement():
    session, _backend = _session(
        {
            "Atlas encoder": _unit_vector(1.0, 0.0),
            "Atlas representation model": _unit_vector(0.88, 0.475),
        }
    )
    entities = (
        _document_entity("a", "Atlas encoder"),
        _document_entity("b", "Atlas representation model"),
        _document_entity("dataset-a", "MMLU", entity_type="dataset"),
        _document_entity("dataset-b", "mmlu", entity_type="benchmark"),
    )
    relations = (
        SupportedRelation(301, 1, "uses_dataset", 6, 0.9, supported=False),
        SupportedRelation(302, 2, "made_up_relation", 7, 0.9, supported=True),
    )

    config = CollectionResolutionConfig(
        thresholds=ResolutionThresholds(
            automatic=0.90,
            candidate=0.75,
            retrieval_similarity=0.70,
        ),
        embedding_weight=0.8,
        neighborhood_weight=0.2,
    )
    result = resolve_collection_entities(
        _snapshot(config=config),
        entities,
        _ontology(),
        relations=relations,
        config=config,
        embedding_session=session,
    )

    decision = _decision(result, "a", "b")
    assert decision.neighborhood_agreement == 0.0
    assert decision.outcome is ResolutionOutcome.CANDIDATE


def test_entities_outside_exact_document_artifact_snapshot_are_rejected_before_embed():
    session, backend = _session({})

    with pytest.raises(ValueError, match="snapshot"):
        resolve_collection_entities(
            _snapshot(201),
            (
                _document_entity("a", "Atlas"),
                _document_entity(
                    "b",
                    "Atlas family",
                    document_artifact_id=999,
                ),
            ),
            _ontology(),
            embedding_session=session,
        )

    assert backend.calls == []


def test_result_is_insertion_order_invariant_including_checksums_and_audit():
    entities = (
        _document_entity("a", "Aquila"),
        _document_entity("b", "aquila"),
        _document_entity("c", "Atlas"),
    )
    vectors = {
        "Aquila": _unit_vector(1.0, 0.0),
        "Atlas": _unit_vector(0.0, 1.0),
    }
    forward_session, _forward_backend = _session(vectors)
    reverse_session, _reverse_backend = _session(vectors)

    forward = resolve_collection_entities(
        _snapshot(),
        entities,
        _ontology(),
        embedding_session=forward_session,
    )
    reverse = resolve_collection_entities(
        _snapshot(),
        tuple(reversed(entities)),
        _ontology(),
        embedding_session=reverse_session,
    )

    assert forward == reverse
    assert forward.checksum == reverse.checksum


def test_bounded_embedding_ties_ignore_database_entity_id_assignment():
    semantic_inputs = (
        ("Atlas Alpha", f"{11:064x}"),
        ("Atlas Beta", f"{22:064x}"),
        ("Atlas Gamma", f"{33:064x}"),
    )
    vectors = {label: _unit_vector(1.0) for label, _cluster_key in semantic_inputs}
    config = CollectionResolutionConfig(
        max_candidates_per_entity=1,
        embedding_weight=1.0,
        neighborhood_weight=0.0,
    )

    def resolve_with_ids(entity_ids):
        session, _backend = _session(vectors)
        return resolve_collection_entities(
            _snapshot(config=config),
            tuple(
                _document_entity(
                    entity_id,
                    label,
                    document_cluster_key=cluster_key,
                )
                for entity_id, (label, cluster_key) in zip(
                    entity_ids, semantic_inputs, strict=True
                )
            ),
            _ontology(),
            config=config,
            embedding_session=session,
        )

    first = resolve_with_ids((1, 2, 3))
    renumbered = resolve_with_ids((3, 2, 1))

    assert {frozenset(cluster.document_cluster_keys) for cluster in first.clusters} == {
        frozenset(cluster.document_cluster_keys) for cluster in renumbered.clusters
    }


def test_automatic_threshold_is_stricter_than_retrieval_similarity():
    with pytest.raises(ValueError, match="automatic.*retrieval"):
        ResolutionThresholds(
            automatic=0.70,
            candidate=0.65,
            retrieval_similarity=0.70,
        )


def test_automatic_threshold_is_stricter_than_candidate_review_threshold():
    with pytest.raises(ValueError, match="automatic.*candidate"):
        ResolutionThresholds(
            automatic=0.80,
            candidate=0.80,
            retrieval_similarity=0.70,
        )


def test_embedding_output_must_be_exactly_1024_finite_dimensions():
    session, _backend = _session(
        {
            "A": [1.0, 0.0],
            "B": [1.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="1024"):
        resolve_collection_entities(
            _snapshot(),
            (_document_entity("a", "A"), _document_entity("b", "B")),
            _ontology(),
            embedding_session=session,
        )


def test_resolution_preserves_four_distinct_confidence_namespaces():
    session, _backend = _session({})
    result = resolve_collection_entities(
        _snapshot(),
        (_document_entity("a", "Aquila", extraction_confidence=0.83),),
        _ontology(),
        embedding_session=session,
    )

    cluster = result.clusters[0]
    assert cluster.extraction_confidence == 0.83
    assert cluster.resolution_confidence == 1.0
    assert cluster.retrieval_utility is None
    assert cluster.promotion_confidence is None


def test_filter_result_is_resolution_bound_checksums_status_and_four_scores():
    from apps.knowledge_graph.graph.filtering import (
        EntityFilterInput,
        FilterPolicy,
        FilterStatus,
        PositionKind,
        filter_collection_resolution,
    )

    session, _backend = _session({})
    policy = FilterPolicy(utility_activation_threshold=0.0)
    from apps.knowledge_graph.graph.filtering import filter_policy_checksum

    resolution = resolve_collection_entities(
        _snapshot(filter_checksum=filter_policy_checksum(policy)),
        (_document_entity("a", "Aquila", extraction_confidence=0.83),),
        _ontology(),
        embedding_session=session,
    )
    cluster = resolution.clusters[0]
    evidence = EntityFilterInput(
        entity_id=cluster.cluster_key,
        entity_type=cluster.entity_type,
        mention_ids=("mention-1",),
        document_ids=(str(cluster.document_ids[0]),),
        extraction_confidence=cluster.extraction_confidence,
        resolution_confidence=cluster.resolution_confidence,
        promotion_confidence=0.61,
        relation_participation=1,
        positions=(PositionKind.TITLE,),
    )
    filtered = filter_collection_resolution(
        resolution, (evidence,), _ontology(), policy
    )

    assert filtered.resolution_checksum == resolution.checksum
    assert filtered.policy_checksum == filtered.decisions[0].policy_checksum
    assert filtered.decisions[0].status is FilterStatus.ACTIVE
    assert filtered.decisions[0].extraction_confidence == 0.83
    assert filtered.decisions[0].resolution_confidence == 1.0
    assert filtered.decisions[0].promotion_confidence == 0.61
    assert filtered.decisions[0].retrieval_utility > 0.0
    assert len(filtered.checksum) == 64

    with pytest.raises(ValueError, match="decisions|typed tuple"):
        replace(filtered, decisions=list(filtered.decisions))


def test_resolution_configuration_and_result_are_immutable():
    config = CollectionResolutionConfig()
    session, _backend = _session({})
    result = resolve_collection_entities(
        _snapshot(),
        (_document_entity("a", "Aquila"),),
        _ontology(),
        config=config,
        embedding_session=session,
    )

    with pytest.raises((AttributeError, TypeError)):
        result.clusters[0].label = "mutated"
    with pytest.raises((AttributeError, TypeError)):
        result.audit.max_candidates_per_entity = 999


def test_changing_retrieval_threshold_does_not_change_automatic_threshold():
    original = ResolutionThresholds()
    changed = replace(original, retrieval_similarity=0.50)

    assert changed.automatic == original.automatic
    assert changed.candidate == original.candidate


def test_collection_resolution_has_no_eager_provider_or_chat_llm_imports():
    import apps.knowledge_graph.resolution.collection as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden = ("gliner2", "openai", "anthropic", "lib.llm")

    assert not any(
        name == prefix or name.startswith(f"{prefix}.")
        for name in imported
        for prefix in forbidden
    )


def test_different_version_signatures_are_hard_blocked_before_embeddings():
    session, backend = _session({})
    result = resolve_collection_entities(
        _snapshot(),
        (
            _document_entity(
                "a", "Atlas v1", normalized_label="atlas", version_signature="v1"
            ),
            _document_entity(
                "b", "Atlas v2", normalized_label="atlas", version_signature="v2"
            ),
        ),
        _ontology(),
        embedding_session=session,
    )

    assert len(result.clusters) == 2
    decision = _decision(result, "a", "b")
    assert decision.outcome is ResolutionOutcome.REJECTED
    assert "version_signature_conflict" in decision.reason_codes
    assert backend.calls == []


def test_aliases_require_task8_membership_provenance():
    with pytest.raises(ValueError, match="alias.*method"):
        AliasEvidence(
            alias="Retrieval Augmented Generation",
            method="caller_claimed_alias",
            mention_id=22,
        )


def test_undefined_acronym_only_labels_are_not_automatically_collapsed():
    session, backend = _session({})
    result = resolve_collection_entities(
        _snapshot(),
        (
            _document_entity("a", "RAG"),
            _document_entity("b", "RAG"),
        ),
        _ontology(),
        embedding_session=session,
    )

    assert len(result.clusters) == 2
    decision = _decision(result, "a", "b")
    assert decision.outcome is not ResolutionOutcome.AUTOMATIC
    assert "undefined_acronym" in decision.reason_codes
    assert backend.calls == []


def test_undefined_acronym_descriptions_cannot_promote_similarity_to_automatic():
    session, backend = _session(
        {
            "RAG first context": _unit_vector(1.0, 0.0),
            "RAG second context": _unit_vector(1.0, 0.0),
        }
    )
    result = resolve_collection_entities(
        _snapshot(),
        (
            _document_entity("a", "RAG", description="first context"),
            _document_entity("b", "RAG", description="second context"),
        ),
        _ontology(),
        embedding_session=session,
    )

    assert _cluster_memberships(result) == {
        frozenset(("a",)),
        frozenset(("b",)),
    }
    decision = _decision(result, "a", "b")
    assert decision.outcome is ResolutionOutcome.CANDIDATE
    assert "acronym_requires_shared_expansion" in decision.reason_codes
    assert backend.calls


def test_transitive_candidates_cannot_bridge_conflicting_identifiers():
    session, _backend = _session(
        {
            "Atlas alpha": _unit_vector(1.0, 0.0),
            "Atlas bridge": _unit_vector(0.999, 0.01),
            "Atlas gamma": _unit_vector(0.998, 0.02),
        }
    )
    entities = (
        _document_entity("a", "Atlas alpha", identifier="arxiv:2401.00001"),
        _document_entity("b", "Atlas bridge"),
        _document_entity("c", "Atlas gamma", identifier="arxiv:2401.00002"),
    )
    config = CollectionResolutionConfig(
        thresholds=ResolutionThresholds(
            automatic=0.95,
            candidate=0.75,
            retrieval_similarity=0.70,
        ),
        max_candidates_per_entity=2,
        embedding_weight=1.0,
        neighborhood_weight=0.0,
    )

    result = resolve_collection_entities(
        _snapshot(config=config),
        entities,
        _ontology(),
        config=config,
        embedding_session=session,
    )

    assert all(
        not {1, 3}.issubset(cluster.document_entity_ids) for cluster in result.clusters
    )
    bridge_decisions = (_decision(result, "a", "b"), _decision(result, "b", "c"))
    assert any(
        decision.outcome is not ResolutionOutcome.AUTOMATIC
        for decision in bridge_decisions
    )
    assert any(
        "transitive_identity_conflict" in decision.reason_codes
        for decision in bridge_decisions
    )


def test_exact_alias_bridge_cannot_merge_components_with_conflicting_identifiers():
    """Cannot-link constraints are checked against the evolving component."""

    entities = (
        _document_entity(
            "a",
            "Alpha",
            identifier="arxiv:2401.00001",
            alias_evidence=(
                AliasEvidence(alias="Bridge", method="ontology_alias", mention_id=1),
            ),
        ),
        _document_entity(
            "b",
            "Bridge",
            alias_evidence=(
                AliasEvidence(alias="Omega", method="ontology_alias", mention_id=2),
            ),
        ),
        _document_entity(
            "c",
            "Omega",
            identifier="arxiv:2401.00002",
        ),
    )
    session, backend = _session({})

    result = resolve_collection_entities(
        _snapshot(), entities, _ontology(), embedding_session=session
    )

    assert all(
        not {1, 3}.issubset(cluster.document_entity_ids) for cluster in result.clusters
    )
    suppressed = tuple(
        decision
        for decision in result.decisions
        if decision.outcome is not ResolutionOutcome.AUTOMATIC
    )
    assert any(
        "component" in reason and "conflict" in reason
        for decision in suppressed
        for reason in decision.reason_codes
    )
    assert backend.calls == []


@pytest.mark.parametrize(
    "left_overrides,bridge_overrides,right_overrides,reason_fragment",
    [
        (
            {"version_signature": "v1"},
            {"version_signature": "v1"},
            {"version_signature": "v2"},
            "version_signature_conflict",
        ),
        (
            {"entity_type": "model"},
            {"entity_type": "architecture"},
            {"entity_type": "method"},
            None,
        ),
    ],
)
def test_exact_alias_bridge_preserves_version_and_canonical_type_cannot_links(
    left_overrides, bridge_overrides, right_overrides, reason_fragment
):
    entities = (
        _document_entity(
            "a",
            "Alpha",
            alias_evidence=(
                AliasEvidence(alias="Bridge", method="ontology_alias", mention_id=1),
            ),
            **left_overrides,
        ),
        _document_entity(
            "b",
            "Bridge",
            alias_evidence=(
                AliasEvidence(alias="Omega", method="ontology_alias", mention_id=2),
            ),
            **bridge_overrides,
        ),
        _document_entity("c", "Omega", **right_overrides),
    )
    session, backend = _session({})

    result = resolve_collection_entities(
        _snapshot(), entities, _ontology(), embedding_session=session
    )

    assert all(
        not {1, 3}.issubset(cluster.document_entity_ids) for cluster in result.clusters
    )
    if reason_fragment is None:
        assert result.audit.type_incompatible_pair_count > 0
    else:
        assert any(
            reason_fragment in decision.reason_codes
            for decision in result.decisions
            if decision.outcome is not ResolutionOutcome.AUTOMATIC
        )
    assert backend.calls == []


@pytest.mark.parametrize(
    "anchor_field,left_anchor,right_anchor,expected_reason",
    [
        (
            "versions",
            "v1",
            "v2",
            "component_version_signature_conflict",
        ),
        (
            "entity_types",
            "model",
            "method",
            "component_ontology_type_conflict",
        ),
    ],
)
def test_disjoint_set_audits_evolving_version_and_type_component_conflicts(
    anchor_field, left_anchor, right_anchor, expected_reason
):
    from apps.knowledge_graph.resolution.collection import _DisjointSet

    empty = {value: frozenset() for value in (1, 2, 3)}
    anchors = dict(empty)
    anchors[1] = frozenset({left_anchor})
    anchors[3] = frozenset({right_anchor})
    kwargs = {
        "identifiers": dict(empty),
        "versions": dict(empty),
        "entity_types": dict(empty),
    }
    kwargs[anchor_field] = anchors
    dsu = _DisjointSet((1, 2, 3), **kwargs)
    dsu.union(1, 2)

    root, reason = dsu.try_union(2, 3)

    assert root is None
    assert reason == expected_reason
    assert dsu.find(1) == dsu.find(2)
    assert dsu.find(3) != dsu.find(1)


def test_embeddings_are_quantized_to_ieee_float32_before_audit_and_storage():
    from apps.knowledge_graph.resolution.collection import (
        _collection_entity_row_audit,
    )

    raw = [1.0 / 3.0] * EMBEDDING_DIMENSIONS
    expected = struct.unpack("!f", struct.pack("!f", 1.0 / 3.0))[0]

    first = validate_embedding(raw)
    second = validate_embedding(first)

    assert first == second
    assert first[0] == expected
    assert first[0] != 1.0 / 3.0

    audit_fields = {
        "artifact_id": 1,
        "collection_id": 2,
        "cluster_key": "cluster",
        "label": "Atlas",
        "normalized_label": "atlas",
        "version_signature": "",
        "entity_type": "model",
        "identifier": "",
        "status": "active",
        "extraction_confidence": 0.9,
        "resolution_confidence": 0.8,
        "retrieval_utility": 0.7,
        "promotion_confidence": 0.6,
        "filter_reason": "accepted",
        "embedding_model_signature": _EMBEDDING_SIGNATURE,
        "embedding_input_hash": "a" * 64,
        "metadata": {},
    }
    assert _collection_entity_row_audit(
        SimpleNamespace(embedding=raw, **audit_fields)
    ) == _collection_entity_row_audit(
        SimpleNamespace(embedding=list(first), **audit_fields)
    )


def test_initial_persistence_projection_rejects_recomputed_row_audit_corruption():
    from apps.knowledge_graph.resolution.collection import (
        _collection_entity_row_audit,
        _collection_link_row_audit,
        _collection_resolution_entity_matches,
        _collection_resolution_link_matches,
    )

    entity_fields = {
        "artifact_id": 1,
        "collection_id": 2,
        "cluster_key": "cluster-a",
        "label": "Atlas",
        "normalized_label": "atlas",
        "version_signature": "",
        "entity_type": "model",
        "identifier": "",
        "status": "active",
        "extraction_confidence": 0.9,
        "resolution_confidence": 0.8,
        "retrieval_utility": 0.7,
        "promotion_confidence": 0.6,
        "filter_reason": "accepted",
        "embedding_model_signature": _EMBEDDING_SIGNATURE,
        "embedding_input_hash": "a" * 64,
        "embedding": list(_unit_vector(1.0)),
    }
    expected_entity = SimpleNamespace(**entity_fields, metadata={"aliases": ["Atlas"]})
    expected_entity.metadata["row_audit_checksum"] = _collection_entity_row_audit(
        expected_entity
    )
    forged_entity = SimpleNamespace(
        **{**entity_fields, "label": "Forged"}, metadata={"aliases": ["Atlas"]}
    )
    forged_entity.metadata["row_audit_checksum"] = _collection_entity_row_audit(
        forged_entity
    )

    assert not _collection_resolution_entity_matches(forged_entity, expected_entity)

    expected_target = SimpleNamespace(pk=10, cluster_key="cluster-a")
    forged_target = SimpleNamespace(pk=11, cluster_key="cluster-b")
    link_fields = {
        "artifact_id": 1,
        "manifest_input_id": 20,
        "document_entity_id": 30,
        "collection_entity_id": 10,
        "collection_entity": expected_target,
        "score": 0.8,
        "identifier_score": None,
        "alias_score": 0.8,
        "embedding_similarity": None,
        "neighborhood_agreement": None,
        "method": "exact_alias",
        "resolver_version": "collection-resolution-v1",
        "outcome": "candidate",
        "candidate_rank": 1,
        "decision_checksum": "b" * 64,
        "status": "suppressed",
        "reason": "candidate_threshold",
    }
    expected_link = SimpleNamespace(**link_fields, metadata={"kind": "candidate"})
    expected_link.metadata["row_audit_checksum"] = _collection_link_row_audit(
        expected_link
    )
    forged_link = SimpleNamespace(
        **{
            **link_fields,
            "collection_entity_id": 11,
            "collection_entity": forged_target,
            "method": "forged",
        },
        metadata={"kind": "candidate"},
    )
    forged_link.metadata["row_audit_checksum"] = _collection_link_row_audit(forged_link)

    assert not _collection_resolution_link_matches(forged_link, expected_link)


def test_embedding_session_rejects_provider_or_model_signature_drift():
    expected = (
        f"local:model-a@rev:endpoint={'e' * 64}:dims=1024:"
        "prep=kg-entity-v1:max_chars=8192:batch=64"
    )
    backend = _RecordingBackend(
        {"Atlas": _unit_vector(1.0)},
        signature=(
            f"cohere:model-b@rev:endpoint={'f' * 64}:dims=1024:"
            "prep=kg-entity-v1:max_chars=8192:batch=64"
        ),
    )
    session = CollectionEmbeddingSession(
        expected_model_signature=expected,
        backend=backend,
    )

    with pytest.raises(ValueError, match="signature.*drift"):
        session.embed(("Atlas",))


@pytest.mark.parametrize(
    "batch_factory, message",
    [
        (
            lambda texts, signature: SignedEmbeddingBatch(
                vectors=(),
                text_hashes=(),
                indices=(),
                model_signature=signature,
            ),
            "one vector",
        ),
        (
            lambda texts, signature: SignedEmbeddingBatch(
                vectors=(tuple([math.nan] + [0.0] * 1023),),
                text_hashes=(embedding_text_hash(texts[0]),),
                indices=(0,),
                model_signature=signature,
            ),
            "finite",
        ),
        (
            lambda texts, signature: SignedEmbeddingBatch(
                vectors=(tuple(_unit_vector(1.0)),),
                text_hashes=("f" * 64,),
                indices=(0,),
                model_signature=signature,
            ),
            "order|hash",
        ),
    ],
)
def test_embedding_session_validates_count_finiteness_and_output_order(
    batch_factory, message
):
    signature = (
        f"local:model@rev:endpoint={'e' * 64}:dims=1024:"
        "prep=kg-entity-v1:max_chars=8192:batch=64"
    )

    def backend(texts):
        return batch_factory(texts, signature)

    session = CollectionEmbeddingSession(
        expected_model_signature=signature,
        backend=backend,
    )

    with pytest.raises(ValueError, match=message):
        session.embed(("Atlas",))


def test_embedding_session_rejects_missing_or_duplicate_provider_indices():
    def backend(texts):
        return SignedEmbeddingBatch(
            vectors=tuple(tuple(_unit_vector(1.0)) for _ in texts),
            text_hashes=tuple(embedding_text_hash(texts[0]) for _ in texts),
            indices=tuple(0 for _ in texts),
            model_signature=_EMBEDDING_SIGNATURE,
        )

    session = CollectionEmbeddingSession(
        expected_model_signature=_EMBEDDING_SIGNATURE,
        backend=backend,
    )

    with pytest.raises(ValueError, match="indices|binding|order"):
        session.embed(("Atlas", "Zephyr"))


def test_embedding_session_binds_reversed_provider_indices_to_exact_texts():
    atlas_vector = tuple(_unit_vector(1.0, 0.0))
    zephyr_vector = tuple(_unit_vector(0.0, 1.0))

    def backend(texts):
        assert texts == ("Atlas", "Zephyr")
        return SignedEmbeddingBatch(
            vectors=(zephyr_vector, atlas_vector),
            text_hashes=(
                embedding_text_hash("Zephyr"),
                embedding_text_hash("Atlas"),
            ),
            indices=(1, 0),
            model_signature=_EMBEDDING_SIGNATURE,
        )

    session = CollectionEmbeddingSession(
        expected_model_signature=_EMBEDDING_SIGNATURE,
        backend=backend,
    )

    result = session.embed(("Zephyr", "Atlas"))

    assert tuple(item.text for item in result) == ("Zephyr", "Atlas")
    assert result[0].vector == zephyr_vector
    assert result[1].vector == atlas_vector


def test_embedding_session_rejects_overlong_text_without_truncation_or_provider_call():
    calls = []

    def backend(texts):
        calls.append(texts)
        return SignedEmbeddingBatch(
            vectors=(tuple(_unit_vector(1.0)),),
            text_hashes=(embedding_text_hash(texts[0]),),
            indices=(0,),
            model_signature=_EMBEDDING_SIGNATURE,
        )

    session = CollectionEmbeddingSession(
        expected_model_signature=_EMBEDDING_SIGNATURE,
        backend=backend,
    )
    boundary = "x" * 8_192
    accepted = session.embed((boundary,))
    assert calls == [(boundary,)]
    assert accepted[0].text == boundary
    assert accepted[0].input_hash == embedding_text_hash(boundary)

    with pytest.raises(ValueError, match="maximum|8192|long"):
        session.embed(("x" * 8_193,))
    assert calls == [(boundary,)]


def test_embedding_session_batches_deterministically_and_fails_atomically():
    calls = []

    def backend(texts):
        calls.append(texts)
        if len(calls) == 2:
            raise RuntimeError("provider failed")
        return SignedEmbeddingBatch(
            vectors=tuple(tuple(_unit_vector(1.0)) for _ in texts),
            text_hashes=tuple(embedding_text_hash(text) for text in texts),
            indices=tuple(range(len(texts))),
            model_signature=_EMBEDDING_SIGNATURE.replace("batch=64", "batch=2"),
        )

    signature = _EMBEDDING_SIGNATURE.replace("batch=64", "batch=2")
    session = CollectionEmbeddingSession(
        expected_model_signature=signature,
        backend=backend,
        batch_size=2,
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        session.embed(("Delta", "Alpha", "Charlie"))
    assert calls == [("Alpha", "Charlie"), ("Delta",)]

    calls.clear()
    with pytest.raises(RuntimeError, match="provider failed"):
        session.embed(("Delta", "Alpha", "Charlie"))
    assert calls[0] == ("Alpha", "Charlie")


def test_embedding_session_deduplicates_stably_and_records_input_hashes():
    session, backend = _session(
        {
            "Atlas": _unit_vector(1.0),
            "Zephyr": _unit_vector(0.0, 1.0),
        }
    )

    embedded = session.embed(("Zephyr", "Atlas", "Atlas"))

    assert backend.calls == [("Atlas", "Zephyr")]
    assert tuple(item.text for item in embedded) == ("Zephyr", "Atlas", "Atlas")
    assert embedded[1].input_hash == embedded[2].input_hash
    assert embedded[0].input_hash == embedding_text_hash("Zephyr")


def test_resolver_rejects_a_prewarmed_embedding_session():
    session, backend = _session({"Atlas": _unit_vector(1.0)})
    session.embed(("Atlas",))

    with pytest.raises(ValueError, match="fresh|prewarmed|cache"):
        resolve_collection_entities(
            _snapshot(),
            (_document_entity("a", "Atlas"),),
            _ontology(),
            embedding_session=session,
        )
    assert backend.calls == [("Atlas",)]


def test_embedding_candidate_decisions_remain_fanout_bounded():
    entities = tuple(
        _document_entity(index, f"Atlas variant {index}") for index in range(10, 110)
    )
    vectors = {
        entity.label: _unit_vector(1.0, (entity.entity_id % 10) / 1000)
        for entity in entities
    }
    session, _backend = _session(vectors)
    config = CollectionResolutionConfig(max_candidates_per_entity=3)
    result = resolve_collection_entities(
        _snapshot(config=config),
        entities,
        _ontology(),
        config=config,
        embedding_session=session,
    )

    embedding_decisions = tuple(
        decision
        for decision in result.decisions
        if decision.embedding_similarity is not None
        and "candidate_fanout_capped" not in decision.reason_codes
    )
    observed: dict[int, int] = {}
    for decision in embedding_decisions:
        observed[decision.left_entity_id] = observed.get(decision.left_entity_id, 0) + 1
        observed[decision.right_entity_id] = (
            observed.get(decision.right_entity_id, 0) + 1
        )
    assert max(observed.values(), default=0) <= 3
    assert len(result.decisions) <= len(entities) * 3


def test_resolution_outputs_recursively_reject_forged_exact_types():
    session, _backend = _session({})
    result = resolve_collection_entities(
        _snapshot(),
        (_document_entity("a", "Atlas"), _document_entity("b", "Atlas")),
        _ontology(),
        embedding_session=session,
    )

    with pytest.raises(ValueError, match="outcome"):
        replace(result.decisions[0], outcome="automatic")
    with pytest.raises(ValueError, match="clusters|typed tuple"):
        replace(result, clusters=list(result.clusters))


def test_resolution_result_rejects_audit_caps_that_differ_from_its_config():
    session, _backend = _session({})
    result = resolve_collection_entities(
        _snapshot(),
        (_document_entity("a", "Atlas"),),
        _ontology(),
        embedding_session=session,
    )

    forged_audit = replace(
        result.audit,
        max_candidates_per_entity=result.audit.max_candidates_per_entity + 1,
    )
    with pytest.raises(ValueError, match="audit.*config|candidate caps"):
        replace(result, audit=forged_audit, checksum="")


def test_locked_source_replay_rejects_forged_result_with_recomputed_checksum():
    from apps.knowledge_graph.resolution.collection import (
        _validate_result_against_source,
        resolution_result_checksum,
    )

    entities = (
        _document_entity("a", "Atlas"),
        _document_entity("b", "Atlas"),
    )
    session, _backend = _session({})
    result = resolve_collection_entities(
        _snapshot(), entities, _ontology(), embedding_session=session
    )
    object.__setattr__(result.clusters[0], "label", "Forged Atlas")
    object.__setattr__(result, "checksum", resolution_result_checksum(result))
    artifact = SimpleNamespace(
        resolver_version=result.resolver_version,
        embedding_model_signature=result.audit.embedding_model_signature,
        ontology_version=result.snapshot.ontology_version,
        ontology_checksum=result.snapshot.ontology_checksum,
        filter_policy_checksum=result.snapshot.filter_policy_checksum,
        resolution_config_checksum=result.snapshot.resolution_config_checksum,
    )

    with pytest.raises(RuntimeError, match="deterministic locked-source replay"):
        _validate_result_against_source(
            result,
            result.snapshot,
            entities,
            (),
            artifact,
            ontology=_ontology(),
            replay=True,
        )


def test_kg_schema_uses_real_collection_fk_typed_scores_and_manifest():
    from apps.collections.models import Collection
    from apps.knowledge_graph.models import (
        CollectionEntity,
        CollectionEntityDocumentLink,
        DocumentEntity,
        GraphArtifact,
        GraphBuildRun,
    )

    collection_input = apps.get_model("apps_knowledge_graph", "CollectionArtifactInput")
    assert GraphArtifact._meta.get_field("scope_id").get_internal_type() == "CharField"
    assert GraphBuildRun._meta.get_field("scope_id").get_internal_type() == "CharField"
    for model in (GraphArtifact, GraphBuildRun):
        signature = model._meta.get_field("embedding_model_signature")
        assert signature.blank is True
        for checksum_field in (
            "ontology_checksum",
            "filter_policy_checksum",
            "resolution_config_checksum",
        ):
            field = model._meta.get_field(checksum_field)
            assert field.max_length == 64
            assert field.editable is False
    assert (
        CollectionEntity._meta.get_field("collection").remote_field.model is Collection
    )
    assert (
        collection_input._meta.get_field("collection").remote_field.model is Collection
    )
    assert (
        collection_input._meta.get_field("document_artifact").remote_field.model
        is GraphArtifact
    )
    assert collection_input._meta.get_field("membership_signature").max_length == 64
    assert (
        CollectionEntityDocumentLink._meta.get_field("artifact").remote_field.model
        is GraphArtifact
    )
    assert (
        CollectionEntityDocumentLink._meta.get_field(
            "manifest_input"
        ).remote_field.model
        is collection_input
    )
    assert DocumentEntity._meta.get_field("resolution_confidence").null is False
    for name in (
        "cluster_key",
        "version_signature",
        "extraction_confidence",
        "resolution_confidence",
        "retrieval_utility",
        "promotion_confidence",
        "filter_reason",
        "embedding_model_signature",
        "embedding_input_hash",
    ):
        assert CollectionEntity._meta.get_field(name) is not None


def test_scope_identity_and_embedding_signature_have_conditional_db_checks():
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    artifact_constraint_names = {
        constraint.name
        for constraint in GraphArtifact._meta.constraints
        if isinstance(constraint, CheckConstraint)
    }
    run_constraint_names = {
        constraint.name
        for constraint in GraphBuildRun._meta.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert "kg_artifact_typed_scope_id" in artifact_constraint_names
    assert "kg_artifact_embedding_signature_scope" in artifact_constraint_names
    assert "kg_run_typed_scope_id" in run_constraint_names
    assert "kg_run_embedding_signature_scope" in run_constraint_names


def test_polymorphic_scope_ids_canonicalize_document_uuid_and_collection_pk():
    from apps.knowledge_graph.models import GraphArtifact

    common = {
        "status": GraphArtifact.Status.BUILDING,
        "source_hash": "a" * 64,
        "ontology_version": "ontology-v1",
        "extractor_version": "extractor-v1",
        "resolver_version": "resolver-v1",
        "filter_policy_version": "filter-v1",
    }
    document_id = uuid.UUID("11111111-1111-4111-8111-111111111111")
    document = GraphArtifact(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=document_id,
        embedding_model_signature="",
        **common,
    )
    collection = GraphArtifact(
        scope_type=GraphArtifact.ScopeType.COLLECTION,
        scope_id=7,
        embedding_model_signature=_EMBEDDING_SIGNATURE,
        **common,
    )

    document.prepare_for_persistence()
    collection.prepare_for_persistence()

    assert document.scope_id == str(document_id)
    assert collection.scope_id == "7"
    document.clean()
    collection.clean()


@pytest.mark.parametrize(
    "scope_type, scope_id, signature",
    [
        ("document", "not-a-uuid", ""),
        (
            "collection",
            "007",
            "local:model@rev:dims=1024:prep=kg-entity-v1:max_chars=8192:batch=64",
        ),
        (
            "collection",
            "0",
            "local:model@rev:dims=1024:prep=kg-entity-v1:max_chars=8192:batch=64",
        ),
        ("document", str(uuid.uuid4()), "must-be-empty"),
        (
            "collection",
            "7",
            "local:model@rev:dims=1024:prep=kg-entity-v1:max_chars=8192:batch=64",
        ),
        ("collection", "7", ""),
    ],
)
def test_invalid_typed_scope_or_embedding_signature_is_rejected(
    scope_type, scope_id, signature
):
    from apps.knowledge_graph.models import GraphArtifact

    artifact = GraphArtifact(
        scope_type=scope_type,
        scope_id=scope_id,
        status=GraphArtifact.Status.BUILDING,
        source_hash="a" * 64,
        ontology_version=_ontology().version,
        extractor_version="extractor-v1",
        resolver_version="resolver-v1",
        filter_policy_version="filter-v1",
        embedding_model_signature=signature,
    )

    with pytest.raises(ValidationError, match="scope|embedding"):
        artifact.clean()


def test_document_link_has_explicit_outcomes_component_scores_and_auto_uniqueness():
    from apps.knowledge_graph.models import CollectionEntityDocumentLink

    fields = {field.name for field in CollectionEntityDocumentLink._meta.fields}
    assert {
        "artifact",
        "manifest_input",
        "outcome",
        "identifier_score",
        "alias_score",
        "embedding_similarity",
        "neighborhood_agreement",
        "candidate_rank",
        "decision_checksum",
    } <= fields
    assert any(
        isinstance(constraint, UniqueConstraint)
        and constraint.name == "kg_one_auto_collection_assignment"
        and constraint.condition is not None
        for constraint in CollectionEntityDocumentLink._meta.constraints
    )


def test_collection_cluster_key_is_stable_across_database_ids_and_rebuild_artifacts():
    first_entity = _document_entity(
        10,
        "Atlas",
        document_cluster_key="c" * 64,
    )
    second_entity = _document_entity(
        20,
        "Atlas",
        document_cluster_key="c" * 64,
    )
    first_session, _ = _session({})
    second_session, _ = _session({})

    first = resolve_collection_entities(
        _snapshot(),
        (first_entity,),
        _ontology(),
        embedding_session=first_session,
    )
    second = resolve_collection_entities(
        replace(_snapshot(), destination_artifact_id=102),
        (second_entity,),
        _ontology(),
        embedding_session=second_session,
    )

    assert first.clusters[0].cluster_key == second.clusters[0].cluster_key


def test_large_exact_label_block_records_linear_deterministic_edges():
    entities = tuple(
        _document_entity(
            entity_id,
            "Atlas",
            document_cluster_key=f"{entity_id + 1000:064x}",
        )
        for entity_id in range(1, 101)
    )
    session, backend = _session({})

    result = resolve_collection_entities(
        _snapshot(),
        entities,
        _ontology(),
        embedding_session=session,
    )

    assert len(result.clusters) == 1
    assert len(result.decisions) <= len(entities) * 2
    assert backend.calls == []


def test_result_audits_the_exact_supported_relation_snapshot():
    entities = (
        _document_entity("a", "Atlas", description="research model"),
        _document_entity("b", "Beta", description="research model"),
        _document_entity(
            "dataset-a",
            "Data One",
            entity_type="dataset",
        ),
    )
    relation = SupportedRelation(1, 1, "uses_dataset", 6, 0.9)
    session, _ = _session(
        {
            "Atlas research model": _unit_vector(1.0),
            "Beta research model": _unit_vector(0.9, 0.1),
        }
    )

    result = resolve_collection_entities(
        _snapshot(),
        entities,
        _ontology(),
        relations=(relation,),
        embedding_session=session,
    )

    assert len(result.source_relation_fingerprint) == 64


def test_strict_embedding_adapter_rejects_raw_dimension_mismatch(monkeypatch):
    from aquillm import utils

    monkeypatch.setenv("APP_EMBED_MODEL_REVISION", "rev")
    monkeypatch.setattr(
        utils,
        "get_local_embed_config",
        lambda: ("http://local", "key", "embed-model"),
    )
    monkeypatch.setattr(utils, "get_target_dims", lambda: 1024)
    monkeypatch.setattr(
        utils,
        "get_strict_indexed_embeddings_via_local_openai",
        lambda _queries: [(0, [1.0, 2.0, 3.0])],
    )

    signature = utils.strict_index_embedding_signature()
    with pytest.raises(RuntimeError, match="invalid vector|1024"):
        utils.get_strict_index_embeddings(["Atlas"], expected_model_signature=signature)


def test_strict_local_embedding_adapter_rejects_served_model_drift(monkeypatch):
    from lib.embeddings import local

    requests = []

    class Embeddings:
        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                model="different-model",
                data=[SimpleNamespace(index=0, embedding=[0.0] * 1024)],
            )

    monkeypatch.setattr(
        local,
        "get_local_embed_config",
        lambda: ("https://embeddings.example.test/v1", "secret", "embed-model"),
    )
    monkeypatch.setattr(
        local,
        "_get_local_openai_client",
        lambda _base_url, _api_key: SimpleNamespace(embeddings=Embeddings()),
    )
    monkeypatch.setattr(local, "_dims_kwargs", lambda: {})

    with pytest.raises(RuntimeError, match="model|identity"):
        local.get_strict_indexed_embeddings_via_local_openai(["Atlas"])
    assert requests == [
        {"model": "embed-model", "input": ["Atlas"], "dimensions": 1024}
    ]


def test_strict_embedding_signature_requires_an_immutable_model_revision(
    monkeypatch,
):
    from aquillm import utils

    monkeypatch.delenv("APP_EMBED_MODEL_REVISION", raising=False)
    monkeypatch.setattr(
        utils,
        "get_local_embed_config",
        lambda: ("http://local", "key", "embed-model"),
    )
    monkeypatch.setattr(utils, "get_target_dims", lambda: 1024)

    with pytest.raises(RuntimeError, match="revision|digest|immutable"):
        utils.strict_index_embedding_signature()


def test_strict_embedding_signature_binds_normalized_provider_endpoint(monkeypatch):
    from aquillm import utils

    monkeypatch.setenv("APP_EMBED_MODEL_REVISION", "rev")
    monkeypatch.setattr(utils, "get_target_dims", lambda: 1024)
    monkeypatch.setattr(
        utils,
        "get_local_embed_config",
        lambda: ("HTTPS://Embeddings.Example.test/v1/", "secret", "embed-model"),
    )
    normalized = utils.strict_index_embedding_signature()
    monkeypatch.setattr(
        utils,
        "get_local_embed_config",
        lambda: ("https://embeddings.example.test/v1", "other-secret", "embed-model"),
    )
    assert utils.strict_index_embedding_signature() == normalized

    monkeypatch.setattr(
        utils,
        "get_local_embed_config",
        lambda: ("https://other.example.test/v1", "secret", "embed-model"),
    )
    drifted = utils.strict_index_embedding_signature()
    assert drifted != normalized
    with pytest.raises(RuntimeError, match="signature drift"):
        utils.get_strict_index_embeddings(
            ["Atlas"], expected_model_signature=normalized
        )


def test_embedding_revision_is_documented_and_passed_fail_closed_to_compose():
    repository = Path(__file__).resolve().parents[4]
    env_example = (repository / ".env.example").read_text(encoding="utf-8")
    runbook = (
        repository / "docs/documents/operations/knowledge-graph-overlay-runbook.md"
    ).read_text(encoding="utf-8")

    assert "APP_EMBED_MODEL_REVISION=" in env_example
    assert "immutable" in env_example.lower()
    assert "APP_EMBED_ALLOW_DIMENSIONS_OVERRIDE=0" in env_example
    assert "provider-side dimensions=1024" in env_example.lower()
    assert "APP_EMBED_MODEL_REVISION" in runbook
    assert "fail" in runbook.lower()
    for compose_path in (
        "deploy/compose/base.yml",
        "deploy/compose/development.yml",
        "deploy/compose/production.yml",
        "deploy/compose/no_gpu_dev.yml",
    ):
        compose = (repository / compose_path).read_text(encoding="utf-8")
        assert "APP_EMBED_MODEL_REVISION" in compose
        assert "${APP_EMBED_MODEL_REVISION:-}" in compose
    no_gpu = (repository / "deploy/compose/no_gpu_dev.yml").read_text(encoding="utf-8")
    assert "APP_EMBED_ALLOW_DIMENSIONS_OVERRIDE: 1" in no_gpu
    for compose_path in (
        "deploy/compose/base.yml",
        "deploy/compose/development.yml",
        "deploy/compose/production.yml",
    ):
        compose = (repository / compose_path).read_text(encoding="utf-8")
        assert "VLLM_REVISION=${APP_EMBED_MODEL_REVISION:-}" in compose
        assert "VLLM_MODEL=${APP_EMBED_MODEL:-" in compose
        assert "VLLM_SERVED_MODEL_NAME=${APP_EMBED_MODEL:-" in compose
        assert "VLLM_TOKENIZER=${APP_EMBED_MODEL:-" in compose


@pytest.mark.django_db(transaction=True)
@database_required
def test_collection_resolution_persistence_is_idempotent_and_marker_bound():
    """PostgreSQL CI exercises the complete manifest/marker write boundary."""
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.graph.filtering import (
        FilterPolicy,
        filter_collection_resolution,
    )
    from apps.knowledge_graph.models import (
        CollectionEntityDocumentLink,
        DocumentEntity,
        DocumentEntityMention,
        EntityMention,
        GraphArtifact,
        GraphBuildRun,
    )
    from apps.knowledge_graph.resolution.collection import (
        build_collection_snapshot,
        load_collection_filter_inputs,
        load_collection_resolution_inputs,
        persist_collection_resolution,
    )

    user = User.objects.create_user(username="kg-task9", password="unused")
    collection = Collection.objects.create(name="KG Task 9")
    document = RawTextDocument(
        title="Atlas",
        full_text="Atlas is a model.",
        collection=collection,
        ingested_by=user,
        full_text_hash=RawTextDocument.hash_fn("Atlas is a model."),
    )
    document.save(dont_rechunk=True)
    source = GraphArtifact.objects.create(
        scope_type=GraphArtifact.ScopeType.DOCUMENT,
        scope_id=document.id,
        status=GraphArtifact.Status.ACTIVE,
        source_hash="d" * 64,
        ontology_version=_ontology().version,
        extractor_version="extractor-v1",
        resolver_version="document-coreference-v1",
        filter_policy_version="document-filter-v1",
        ontology_checksum=_ontology().checksum,
    )
    chunk = TextChunk.objects.create(
        content=document.full_text,
        start_position=0,
        end_position=len(document.full_text),
        chunk_number=0,
        modality=TextChunk.Modality.TEXT,
        doc_id=document.id,
        embedding=[0.0] * EMBEDDING_DIMENSIONS,
    )
    mention = EntityMention.objects.create(
        artifact=source,
        document_id=document.id,
        chunk=chunk,
        start=0,
        end=5,
        position_basis=EntityMention.PositionBasis.DOCUMENT_GLOBAL,
        raw_text="Atlas",
        normalized_text="Atlas",
        entity_type="model",
        extraction_confidence=0.9,
    )
    document_entity = DocumentEntity.objects.create(
        artifact=source,
        document_id=document.id,
        cluster_key="e" * 64,
        label="Atlas",
        normalized_label="atlas",
        entity_type="model",
        resolution_confidence=0.95,
    )
    DocumentEntityMention.objects.create(
        document_entity=document_entity,
        mention=mention,
        method=DocumentEntityMention.Method.ROOT,
        resolver_version=source.resolver_version,
    )
    policy = FilterPolicy(
        version="collection-filter-v1", utility_activation_threshold=1.0
    )
    config = CollectionResolutionConfig()
    destination, manifest = build_collection_snapshot(
        collection=collection,
        document_artifacts=(source,),
        ontology=_ontology(),
        extractor_version="extractor-v1",
        resolver_version="collection-resolution-v1",
        filter_policy_version="collection-filter-v1",
        filter_policy=policy,
        resolution_config=config,
        embedding_model_signature=_EMBEDDING_SIGNATURE,
    )
    run = GraphBuildRun.objects.create(
        artifact=destination,
        stage=GraphBuildRun.Stage.RESOLUTION,
        status=GraphBuildRun.Status.RUNNING,
        attempt=1,
    )
    snapshot, entities, relations = load_collection_resolution_inputs(
        destination.pk, run.pk
    )
    session, _backend = _session({})
    result = resolve_collection_entities(
        snapshot,
        entities,
        _ontology(),
        relations=relations,
        config=config,
        embedding_session=session,
    )
    filter_inputs = load_collection_filter_inputs(destination.pk, run.pk, result)
    filter_result = filter_collection_resolution(
        result, filter_inputs, _ontology(), policy
    )

    first = persist_collection_resolution(
        destination.pk,
        run.pk,
        result,
        filter_result,
        filter_policy=policy,
        ontology=_ontology(),
    )
    second = persist_collection_resolution(
        destination.pk,
        run.pk,
        result,
        filter_result,
        filter_policy=policy,
        ontology=_ontology(),
    )

    assert first == second
    assert len(first) == 1
    assert first[0].status == first[0].Status.SUPPRESSED
    assert first[0].retrieval_utility == filter_result.decisions[0].retrieval_utility
    assert first[0].filter_reason == filter_result.decisions[0].reason_codes[0]
    assert (
        CollectionEntityDocumentLink.objects.filter(
            artifact=destination,
            outcome=CollectionEntityDocumentLink.Outcome.AUTOMATIC,
        ).count()
        == 1
    )
    run.refresh_from_db()
    assert (
        run.stats["collection_resolution_commit"]["result_checksum"] == result.checksum
    )
    assert len(manifest) == 1

    moved_collection = Collection.objects.create(name="KG Task 9 moved")
    RawTextDocument.objects.filter(pk=document.pk).update(collection=moved_collection)
    with pytest.raises(RuntimeError, match="manifest|membership|collection"):
        persist_collection_resolution(
            destination.pk,
            run.pk,
            result,
            filter_result,
            filter_policy=policy,
            ontology=_ontology(),
        )
    RawTextDocument.objects.filter(pk=document.pk).update(collection=collection)

    run.stats["collection_resolution_commit"]["link_count"] = 0
    run.save(update_fields=["stats"])
    with pytest.raises(RuntimeError, match="marker"):
        persist_collection_resolution(
            destination.pk,
            run.pk,
            result,
            filter_result,
            filter_policy=policy,
            ontology=_ontology(),
        )
