from __future__ import annotations

import ast
import math
import socket
import uuid
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

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
    embedding_text_hash,
    resolve_collection_entities,
)
from apps.knowledge_graph.resolution.scoring import (
    EMBEDDING_DIMENSIONS,
    ResolutionOutcome,
    ResolutionThresholds,
    ResolutionTier,
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
    not _database_is_reachable(),
    reason="configured PostgreSQL database is not reachable",
)


def _entity_type(name: str, *aliases: str):
    return SimpleNamespace(name=name, aliases=aliases)


def _ontology():
    return SimpleNamespace(
        checksum="b" * 64,
        entity_types=MappingProxyType(
            {
                "model": _entity_type("model", "architecture"),
                "method": _entity_type("method", "approach"),
                "dataset": _entity_type("dataset", "benchmark"),
            }
        ),
        relations=MappingProxyType(
            {
                "uses_dataset": SimpleNamespace(name="uses_dataset"),
            }
        ),
    )


def _snapshot(*artifact_ids: int) -> CollectionBuildSnapshot:
    artifact_ids = artifact_ids or (201,)
    return CollectionBuildSnapshot(
        destination_artifact_id=101,
        collection_id=7,
        inputs=tuple(
            CollectionSnapshotInput(
                manifest_input_id=1000 + artifact_id,
                document_artifact_id=artifact_id,
                document_id=uuid.UUID(int=artifact_id),
                source_signature=f"{artifact_id:064x}"[-64:],
                build_signature=f"{artifact_id + 1:064x}"[-64:],
            )
            for artifact_id in artifact_ids
        ),
        source_hash="a" * 64,
        ontology_checksum="b" * 64,
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
    def __init__(
        self, vectors_by_text, signature="test-local:model@revision:dims=1024:prep=v1"
    ):
        self.vectors_by_text = vectors_by_text
        self.signature = signature
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, texts):
        self.calls.append(tuple(texts))
        return SignedEmbeddingBatch(
            vectors=tuple(tuple(self.vectors_by_text[text]) for text in texts),
            text_hashes=tuple(embedding_text_hash(text) for text in texts),
            model_signature=self.signature,
        )


def _session(
    vectors_by_text, *, signature="test-local:model@revision:dims=1024:prep=v1"
):
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
        _snapshot(),
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
    result = resolve_collection_entities(
        _snapshot(),
        (
            _document_entity("a", "Atlas"),
            _document_entity("b", "Atlas family"),
        ),
        _ontology(),
        config=CollectionResolutionConfig(
            thresholds=ResolutionThresholds(
                automatic=0.95,
                candidate=0.75,
                retrieval_similarity=0.70,
            ),
            embedding_weight=1.0,
            neighborhood_weight=0.0,
        ),
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

    result = resolve_collection_entities(
        _snapshot(),
        entities,
        _ontology(),
        relations=relations,
        config=CollectionResolutionConfig(
            thresholds=ResolutionThresholds(
                automatic=0.90,
                candidate=0.75,
                retrieval_similarity=0.70,
            ),
            embedding_weight=0.8,
            neighborhood_weight=0.2,
        ),
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

    result = resolve_collection_entities(
        _snapshot(),
        entities,
        _ontology(),
        relations=relations,
        config=CollectionResolutionConfig(
            thresholds=ResolutionThresholds(
                automatic=0.90,
                candidate=0.75,
                retrieval_similarity=0.70,
            ),
            embedding_weight=0.8,
            neighborhood_weight=0.2,
        ),
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
            _snapshot(),
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

    result = resolve_collection_entities(
        _snapshot(),
        entities,
        _ontology(),
        config=CollectionResolutionConfig(
            thresholds=ResolutionThresholds(
                automatic=0.95,
                candidate=0.75,
                retrieval_similarity=0.70,
            ),
            max_candidates_per_entity=2,
            embedding_weight=1.0,
            neighborhood_weight=0.0,
        ),
        embedding_session=session,
    )

    assert _cluster_memberships(result) == {
        frozenset(("a",)),
        frozenset(("b",)),
        frozenset(("c",)),
    }
    bridge_decisions = (_decision(result, "a", "b"), _decision(result, "b", "c"))
    assert all(
        decision.outcome is not ResolutionOutcome.AUTOMATIC
        for decision in bridge_decisions
    )
    assert any(
        "transitive_identity_conflict" in decision.reason_codes
        for decision in bridge_decisions
    )


def test_embedding_session_rejects_provider_or_model_signature_drift():
    expected = "local:model-a@rev:dims=1024:prep=v1"
    backend = _RecordingBackend(
        {"Atlas": _unit_vector(1.0)},
        signature="cohere:model-b@rev:dims=1024:prep=v1",
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
                model_signature=signature,
            ),
            "one vector",
        ),
        (
            lambda texts, signature: SignedEmbeddingBatch(
                vectors=(tuple([math.nan] + [0.0] * 1023),),
                text_hashes=(embedding_text_hash(texts[0]),),
                model_signature=signature,
            ),
            "finite",
        ),
        (
            lambda texts, signature: SignedEmbeddingBatch(
                vectors=(tuple(_unit_vector(1.0)),),
                text_hashes=("f" * 64,),
                model_signature=signature,
            ),
            "order|hash",
        ),
    ],
)
def test_embedding_session_validates_count_finiteness_and_output_order(
    batch_factory, message
):
    signature = "local:model@rev:dims=1024:prep=v1"

    def backend(texts):
        return batch_factory(texts, signature)

    session = CollectionEmbeddingSession(
        expected_model_signature=signature,
        backend=backend,
    )

    with pytest.raises(ValueError, match=message):
        session.embed(("Atlas",))


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


def test_embedding_candidate_decisions_remain_fanout_bounded():
    entities = tuple(
        _document_entity(index, f"Atlas variant {index}") for index in range(10, 110)
    )
    vectors = {
        entity.label: _unit_vector(1.0, (entity.entity_id % 10) / 1000)
        for entity in entities
    }
    session, _backend = _session(vectors)
    result = resolve_collection_entities(
        _snapshot(),
        entities,
        _ontology(),
        config=CollectionResolutionConfig(max_candidates_per_entity=3),
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
        embedding_model_signature="local:model@rev:dims=1024:prep=v1",
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
        ("collection", "007", "local:model@rev:dims=1024:prep=v1"),
        ("collection", "0", "local:model@rev:dims=1024:prep=v1"),
        ("document", str(uuid.uuid4()), "must-be-empty"),
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
        ontology_version="ontology-v1",
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
        "get_embeddings_via_local_openai",
        lambda _queries: [[1.0, 2.0, 3.0]],
    )

    signature = utils.strict_index_embedding_signature()
    with pytest.raises(RuntimeError, match="invalid vector|1024"):
        utils.get_strict_index_embeddings(["Atlas"], expected_model_signature=signature)


@pytest.mark.django_db(transaction=True)
@database_required
def test_collection_resolution_persistence_is_idempotent_and_marker_bound():
    """PostgreSQL CI exercises the complete manifest/marker write boundary."""
    from django.contrib.auth.models import User

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk
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
        ontology_version="ontology-v1",
        extractor_version="extractor-v1",
        resolver_version="document-coreference-v1",
        filter_policy_version="document-filter-v1",
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
    destination, manifest = build_collection_snapshot(
        collection=collection,
        document_artifacts=(source,),
        ontology_version="ontology-v1",
        ontology_checksum="b" * 64,
        extractor_version="extractor-v1",
        resolver_version="collection-resolution-v1",
        filter_policy_version="collection-filter-v1",
        embedding_model_signature="test-local:model@rev:dims=1024:prep=v1",
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
        embedding_session=session,
    )

    first = persist_collection_resolution(destination.pk, run.pk, result)
    second = persist_collection_resolution(destination.pk, run.pk, result)

    assert first == second
    assert len(first) == 1
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

    run.stats["collection_resolution_commit"]["link_count"] = 0
    run.save(update_fields=["stats"])
    with pytest.raises(RuntimeError, match="marker"):
        persist_collection_resolution(destination.pk, run.pk, result)
