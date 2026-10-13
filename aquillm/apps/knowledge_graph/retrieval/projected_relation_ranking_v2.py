"""Relation-sensitive transitions with capped credit per directed endpoint pair."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from hashlib import sha256

from .graph_query_contracts import GraphRelationRequestV2
from .ppr import PPRAlgorithmConfig, canonical_algorithm_json
from .ppr_kernel import WeightedEdge, run_ppr_kernel
from .projected_ppr import prepare_projected_ppr_inputs
from .projected_ppr_result import ProjectedPPRResultV1, _trace_bytes
from .projected_types import ProjectedAuthorizedGraphSnapshotV1
from .topology.contracts import ProjectedSeedV1

ALGORITHM_VERSION = "ppr_projected_relation_v2"
_POLICY = (
    b"relation-request-v2\0directed-pair-maximum-v2\0exact-provenance-v2\0"
    b"weighted-endpoint-before-destination-fanout-v2\0"
)


def relation_algorithm_signature_v2(config: PPRAlgorithmConfig) -> str:
    if type(config) is not PPRAlgorithmConfig:
        raise TypeError("config must be an exact PPRAlgorithmConfig")
    return sha256(
        ALGORITHM_VERSION.encode() + b"\0" + _POLICY + canonical_algorithm_json(config)
    ).hexdigest()


def _weighted_pairs(groups, predicate_weights, deadline_check):
    pairs = {}
    observations = {}
    for group in groups:
        deadline_check()
        # Provenance is scoped to its predicate/direction evidence class. Separate
        # support and contradiction labels survive in the immutable snapshot.
        semantic = (
            group.source_identity_key,
            group.relation_type,
            group.target_identity_key,
            group.direction,
        )
        for evidence in group.evidence:
            provenance = (*semantic, evidence.provenance_key)
            previous = observations.setdefault(provenance, evidence)
            if previous != evidence:
                raise ValueError("conflicting exact evidence provenance")
        pair = (group.source_identity_key, group.target_identity_key)
        # raw_weight already contains the frozen confidence/support/utility and
        # direction treatment. Never introduce a second reverse multiplier.
        contribution = group.raw_weight * predicate_weights[group.relation_type]
        pairs[pair] = max(pairs.get(pair, 0.0), contribution)
    return tuple(
        WeightedEdge(source, target, weight)
        for (source, target), weight in sorted(pairs.items())
    )


def _admit_weighted_pairs(snapshot, seeds, config, predicate_weights, deadline_check):
    """Cap weighted destinations, so predicate labels cannot spend fanout slots."""
    eligible = tuple(
        g for g in snapshot.relation_groups if g.admission_hop <= config.max_hops
    )
    pairs = _weighted_pairs(eligible, predicate_weights, deadline_check)
    # Closed snapshot membership fixes admission hop by source identity. All
    # observations of a directed endpoint pair therefore have the same hop.
    hops = {
        (g.source_identity_key, g.target_identity_key): g.admission_hop
        for g in eligible
    }
    by_source = defaultdict(list)
    for edge in pairs:
        deadline_check()
        by_source[edge.source].append(edge)

    def selection_key(edge):
        return -edge.weight, edge.source, edge.target

    fanout = tuple(
        edge
        for source in sorted(by_source)
        for edge in sorted(by_source[source], key=selection_key)[: config.max_fanout]
    )
    deadline_check()
    if len(fanout) <= config.max_edges:
        return fanout
    admitted = {seed.identity_key for seed in seeds}
    retained = []
    for hop in range(1, config.max_hops + 1):
        frontier = sorted(
            (edge for edge in fanout if hops[(edge.source, edge.target)] == hop),
            key=selection_key,
        )
        for edge in frontier:
            deadline_check()
            if edge.source not in admitted:
                continue
            if len(retained) >= config.max_edges or (
                edge.target not in admitted and len(admitted) >= config.max_nodes
            ):
                return tuple(retained)
            retained.append(edge)
            admitted.add(edge.target)
    return tuple(retained)


def rank_relations_v2(
    *,
    snapshot: ProjectedAuthorizedGraphSnapshotV1,
    seeds: tuple[ProjectedSeedV1, ...],
    request: GraphRelationRequestV2,
    config: PPRAlgorithmConfig,
    deadline_check: Callable[[], None],
    frontier_context=None,
) -> ProjectedPPRResultV1:
    """Use the closed V1 mathematical input without claiming its V1 signature."""
    if not callable(deadline_check):
        raise TypeError("deadline_check must be callable")
    deadline_check()
    if type(request) is not GraphRelationRequestV2:
        raise TypeError("request must be an exact GraphRelationRequestV2")
    # Reuse the exact original caps, algorithm and admitted-seed validation.
    if (
        type(snapshot) is ProjectedAuthorizedGraphSnapshotV1
        and snapshot.algorithm.algorithm_version == "canonical_frontier_v2"
    ):
        from .frontier_ranking_validation_v2 import validate_frontier_ranking

        validate_frontier_ranking(
            snapshot=snapshot,
            seeds=seeds,
            request=request,
            config=config,
            context=frontier_context,
        )
    else:
        if frontier_context is not None:
            raise ValueError("V1 snapshot cannot carry frontier context")
        prepare_projected_ppr_inputs(snapshot=snapshot, seeds=seeds, config=config)
    if {row.ontology_checksum for row in snapshot.artifact_provenance} != {
        request.ontology_checksum
    }:
        raise ValueError("relation request ontology does not match the entire snapshot")
    predicate_weights = dict(request.predicate_weights)
    if any(g.relation_type not in predicate_weights for g in snapshot.relation_groups):
        raise ValueError("snapshot predicate is not approved by relation request")
    edges = _admit_weighted_pairs(
        snapshot, seeds, config, predicate_weights, deadline_check
    )
    kernel = run_ppr_kernel(
        nodes=snapshot.identity_keys,
        edges=edges,
        seeds={seed.identity_key: seed.mass for seed in seeds},
        config=config,
        order_key=lambda key: key,
        deadline_check=deadline_check,
    )
    deadline_check()
    scores = tuple(kernel.scores)
    ranked = tuple(key for key, _ in sorted(scores, key=lambda row: (-row[1], row[0])))
    return ProjectedPPRResultV1(scores, ranked, _trace_bytes(scores, ranked))


__all__ = ["rank_relations_v2", "relation_algorithm_signature_v2"]
