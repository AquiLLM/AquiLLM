# ruff: noqa: F401
# ruff: noqa: E501
from __future__ import annotations

import inspect
from dataclasses import replace
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.retrieval import direct_seed_repository
from apps.knowledge_graph.retrieval.direct_seed_contracts import (
    DirectResolutionSpanInputV1,
    DirectResolutionTier,
)
from apps.knowledge_graph.retrieval.direct_seed_repository import (
    DirectSeedCandidateRow,
    DirectSeedRepository,
    DirectSeedScopeV1,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    AuthorizedProjectedDocumentV1,
    ReadyGenerationBundleV1,
    SelectedCollectionGenerationV1,
    ready_generation_bundle_checksum,
)
from lib.knowledge_graph.query_extractor.contracts import QueryEntitySpanV1

K = tuple(character * 64 for character in "123456789abcdef")
DOCUMENT_ID = UUID("11111111-1111-4111-8111-111111111111")


def test_canonical_seed_rejects_identity_string_in_place_of_database_pk():
    with pytest.raises(ValueError, match="canonical"):
        DirectSeedCandidateRow(9, 11, "model", "a" * 64, 1.0)


def test_direct_seeds_match_projection_entity_and_canonical_encoding():
    from apps.knowledge_graph.projection.memberships import (
        MEMBERSHIP_REGISTRY_GENERATION,
        membership_decision_checksum,
    )
    from apps.knowledge_graph.projection.projection_encoding import (
        _entities,
        _memberships,
    )
    from apps.knowledge_graph.projection.records import AutomaticCanonicalMembershipV1

    ready = _ready()
    codec = HmacSha256ProjectionIdentifierCodec(b"key", key_version="key-v1")
    span = QueryEntitySpanV1("model", 0, 5, 1.0)
    repository = DirectSeedRepository(
        scope=_scope(ready),
        codec=codec,
        span_inputs=(DirectResolutionSpanInputV1(span, "model"),),
        row_loader=lambda **_: (
            DirectSeedCandidateRow(7, 11, "model", None, 1.0),
            DirectSeedCandidateRow(9, 11, "model", 42, 1.0),
        ),
        membership_state_loader=lambda **_: _membership_state(ready),
    )
    marker = type(
        "Marker", (), dict(generation_key=K[1], artifact_key=K[2], collection_key=K[0])
    )()
    entities, _ = _entities(
        {
            "entities": (
                {
                    "id": 7,
                    "entity_type": "model",
                    "cluster_key": "cluster",
                    "retrieval_utility": 1.0,
                },
            )
        },
        lambda domain, source: (
            codec.encode(domain, generation=DOCUMENT_ID, source=source).value
        ),
        marker,
    )
    matches = repository.canonical_name_matches(span=span, ready=ready, limit=4)
    singleton = next(row for row in matches if row.entity_key == row.component_key)
    canonical = next(row for row in matches if row.entity_key != row.component_key)
    assert singleton.entity_key == entities[0].entity_key
    assert (
        canonical.component_key
        == codec.encode(
            ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY, source=42
        ).value
    )
    projected_canonical_key = codec.encode(
        ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY, source=42
    ).value
    audit = AutomaticCanonicalMembershipV1(
        codec.encode(
            ProjectionIdentifierDomain.ENTITY,
            generation=MEMBERSHIP_REGISTRY_GENERATION,
            source=7,
        ).value,
        projected_canonical_key,
        K[3],
        "resolver-v1",
        K[6],
    )
    projected_memberships = _memberships(
        {
            "memberships": (
                {
                    "entity_id": 7,
                    "canonical_entity_id": 42,
                    "outcome": "automatic",
                    "status": "active",
                    "canonical_status": "active",
                    "decision_checksum": K[3],
                },
            ),
            "artifacts": (
                {
                    "id": 11,
                    "resolver_version": "resolver-v1",
                    "resolution_config_checksum": K[6],
                },
            ),
        },
        codec,
        DOCUMENT_ID,
        {
            "artifact_id": 11,
            "membership_checksum": membership_decision_checksum((audit,)),
        },
        entities,
        {7: entities[0].entity_key},
    )
    assert canonical.component_key == projected_memberships[0].automatic_membership_key


def _ready() -> ReadyGenerationBundleV1:
    generation = SelectedCollectionGenerationV1(
        K[0],
        K[1],
        K[2],
        K[3],
        K[4],
        "schema-v1",
        "projection-v1",
        "key-v1",
        1,
        K[5],
        "resolver-v1",
        K[6],
        K[7],
        "embed-v1",
    )
    documents = (AuthorizedProjectedDocumentV1(K[8], K[0], K[1]),)
    checksum = ready_generation_bundle_checksum((generation,), documents, K[9])
    return ReadyGenerationBundleV1((generation,), documents, K[9], checksum)


# fmt: off
def _scope(ready: ReadyGenerationBundleV1) -> DirectSeedScopeV1:
    return DirectSeedScopeV1(ready_bundle_checksum=ready.bundle_checksum, selected_collection_ids=(3,), selected_artifact_ids=(11,), selected_document_ids=(DOCUMENT_ID,), selected_document_artifact_ids=(22,), generation_keys_by_artifact=((11, K[1]),), generation_ids_by_artifact=((11, DOCUMENT_ID),), ontology_checksum=K[7], resolver_version="resolver-v1")
# fmt: on


def _membership_state(
    ready: ReadyGenerationBundleV1, **changes: object
) -> tuple[dict[str, object], ...]:
    generation = ready.selected_generations[0]
    state = {
        "collection_id": 3,
        "active_artifact_id": 11,
        "registry_epoch": generation.membership_epoch,
        "membership_checksum": generation.membership_checksum,
        "resolver_version": generation.resolver_version,
        "resolution_config_checksum": generation.resolution_config_checksum,
    }
    state.update(changes)
    return (state,)




# fmt: off
# fmt: on












# fmt: off
