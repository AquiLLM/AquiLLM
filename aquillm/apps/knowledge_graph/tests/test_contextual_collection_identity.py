"""Contextual graph identity regression controls extracted from legacy suites."""
import uuid

import pytest

from apps.knowledge_graph.resolution.collection import (
    CollectionResolutionConfig, resolve_collection_entities,
)
from apps.knowledge_graph.resolution.scoring import ResolutionOutcome
from test_collection_resolution import (
    _document_entity, _session, _snapshot, _ontology, _unit_vector,
)


@pytest.mark.parametrize("other_artifact", [201, 202])
def test_contextual_collection_names_do_not_promote_from_similarity(other_artifact):
    config = CollectionResolutionConfig(identity_policy="contextual_v2")
    entities = (
        _document_entity("a", "Mercury"),
        _document_entity(
            "b",
            "Mercury",
            document_artifact_id=other_artifact,
            document_id=uuid.UUID(int=other_artifact),
        ),
    )
    session, _backend = _session({"Mercury": _unit_vector(1.0)})
    result = resolve_collection_entities(
        _snapshot(*sorted({201, other_artifact}), config=config),
        entities,
        _ontology(),
        config=config,
        embedding_session=session,
    )
    assert {cluster.document_entity_ids for cluster in result.clusters} == {(1,), (2,)}
    assert result.decisions[0].outcome is ResolutionOutcome.CANDIDATE
    assert result.resolver_version == "collection-resolution-v2-contextual"


@pytest.mark.parametrize(
    "right_identifier,outcome,cluster_count",
    [
        ("doi:10.1000/a", ResolutionOutcome.AUTOMATIC, 1),
        ("doi:10.1000/b", ResolutionOutcome.REJECTED, 2),
    ],
)
def test_contextual_collection_anchors_control_identity(
    right_identifier, outcome, cluster_count
):
    config = CollectionResolutionConfig(identity_policy="contextual_v2")
    entities = (
        _document_entity("a", "Mercury", identifier="doi:10.1000/a"),
        _document_entity("b", "Mercury", identifier=right_identifier),
    )
    session, _backend = _session({"Mercury": _unit_vector(1.0)})
    result = resolve_collection_entities(
        _snapshot(config=config),
        entities,
        _ontology(),
        config=config,
        embedding_session=session,
    )
    assert len(result.clusters) == cluster_count
    assert result.decisions[0].outcome is outcome
