"""Observers delegate every production seed/rank operation."""
from contextlib import contextmanager
from dataclasses import dataclass, field

from apps.knowledge_graph.models import CanonicalEntityLink, CollectionEntity, CollectionGraphProjection
from apps.knowledge_graph.projection.identifiers import ProjectionIdentifierDomain as Domain
from apps.knowledge_graph.projection.runtime import (
    load_projection_runtime_settings, projection_identifier_codec,
)
from apps.knowledge_graph.retrieval import graph_runtime_execution_v2 as execution
from apps.knowledge_graph.retrieval.ready_scope_v2 import _digest


@dataclass
class GraphObservation:
    allocations: list = field(default_factory=list)
    ranks: list = field(default_factory=list)

    @property
    def identities(self):
        return {key for arguments, _ in self.ranks for key in arguments["snapshot"].identity_keys}

    @property
    def seeds(self):
        return {seed.identity_key for arguments, _ in self.ranks for seed in arguments["seeds"]}


@contextmanager
def graph_observation(monkeypatch):
    observed = GraphObservation()
    allocate, rank = execution.allocate_group_seeds_v2, execution.rank_relations_v2

    def allocate_original(**kwargs):
        result = allocate(**kwargs)
        observed.allocations.append((kwargs, result))
        return result

    def rank_original(**kwargs):
        result = rank(**kwargs)
        observed.ranks.append((kwargs, result))
        return result

    with monkeypatch.context() as probe:
        probe.setattr(execution, "allocate_group_seeds_v2", allocate_original)
        probe.setattr(execution, "rank_relations_v2", rank_original)
        yield observed


def identity_labels(corpus):
    """Read actual active memberships; never manufacture topology/authority."""
    codec = projection_identifier_codec(load_projection_runtime_settings())
    labels = {}
    for projection in CollectionGraphProjection.objects.filter(
        collection_id__in=[c.pk for c in corpus.collections.values()], state="ready",
    ):
        entities = CollectionEntity.objects.filter(artifact_id=projection.artifact_id, status="active")
        for entity in entities:
            links = tuple(CanonicalEntityLink.objects.filter(collection_entity=entity,
                outcome="automatic", status="active", canonical_entity__status="active"))
            assert len(links) <= 1
            key = codec.encode(Domain.AUTOMATIC_CANONICAL_IDENTITY,
                source=links[0].canonical_entity_id).value if links else codec.encode(
                    Domain.ENTITY, generation=projection.generation_key, source=entity.pk).value
            labels.setdefault(key, set()).add(entity.label)
    return labels


def keys_for(labels, label):
    return {key for key, values in labels.items() if label in values}


def normalized_rank(observed, labels):
    weights, scores = {}, {}

    def label(key):
        assert key in labels and len(labels[key]) == 1
        return next(iter(labels[key]))

    for arguments, result in observed.ranks:
        for group in arguments["snapshot"].relation_groups:
            key = (label(group.source_identity_key), group.relation_type,
                   label(group.target_identity_key), group.direction)
            weights[key] = max(weights.get(key, 0.0), group.raw_weight)
        for key, score in result.scores:
            name = label(key)
            assert name not in scores, "comparison requires unambiguous semantic identities"
            scores[name] = score
    assert weights and scores
    return weights, scores


def assert_runtime_bounds(result, observed, *, query, documents, allow_frontier=False):
    observation = result.frontier_budget
    assert observation.accounting_complete and not observation.uncertain
    for used, maximum, ceiling in (
        (observation.queries_used, observation.max_queries, 64),
        (observation.rows_used, observation.max_rows, 5000),
        (observation.bytes_used, observation.max_bytes, 1048576),
        (observation.identities_used, observation.max_identities, 200),
        (observation.edges_used, observation.max_edges, 1000),
    ):
        assert 0 <= used <= maximum <= ceiling
    invocation = result.frontier_invocation
    assert invocation.query_sha256 == _digest(query)
    assert invocation.requested_subset_signature == _digest(sorted(str(d.id) for d in documents))
    assert invocation.request_algorithm_signature == result.handoff.algorithm_signature
    assert result.selection.representation_snapshot is observed.representation
    assert result.handoff.coverage.exhaustive is False
    assert "deadline" not in result.handoff.coverage.reasons
    if not allow_frontier:
        assert "frontier_budget" not in result.handoff.coverage.reasons
