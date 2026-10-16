"""Captured private authority and deterministic whole-path algorithm identity."""

from dataclasses import dataclass

from .frontier_ranking_validation_v2 import POLICY, frontier_tags
from .graph_frontier_observation_v2 import GraphFrontierInvocationV2
from .graph_handoff import GraphHandoffV2, validate_graph_handoff_v2
from .ready_scope_v2 import SelectedGraphScopeV2, _digest, graph_selection_digest_v2
from .relation_request import relation_request_digest


@dataclass(frozen=True, slots=True)
class GraphRuntimeEnvelopeV2:
    selection: SelectedGraphScopeV2
    handoff: GraphHandoffV2
    frontier_invocation: GraphFrontierInvocationV2

    @property
    def frontier_budget(self):
        return self.frontier_invocation.frontier_budget

    def __post_init__(self):
        if type(self.selection) is not SelectedGraphScopeV2:
            raise TypeError("runtime must retain the exact original selection")
        self.selection.__post_init__()
        if type(self.frontier_invocation) is not GraphFrontierInvocationV2:
            raise TypeError("runtime requires its final frontier observation")
        validate_graph_handoff_v2(self.handoff)
        if (
            self.frontier_invocation.request_algorithm_signature
            != self.handoff.algorithm_signature
        ):
            raise ValueError("observation replaced the executed request algorithm")
        groups = {g.group_checksum: g for g in self.selection.groups}
        if (
            self.handoff.coverage.selected_documents
            != self.selection.coverage.selected_documents
        ):
            raise ValueError("handoff replaced selected scope")
        for ref in self.handoff.candidates:
            group = groups.get(ref.authority_group_checksum)
            if (
                group is None
                or ref.ready_bundle_checksum != group.scope.ready.bundle_checksum
            ):
                raise ValueError("handoff replaced captured group authority")
            group.scope.projection_for_generation(ref.graph_generation_key)
            if ref.representation_generation_key not in {
                s.generation_key
                for s in self.selection.representation_snapshot.selections
                if s.document.id in group.scope.selected_document_ids
            }:
                raise ValueError("handoff representation is outside captured group")


def graph_algorithm_signature_v2(*, selected, requests, caps, budget, configs,
                                 acquisition_policy=POLICY):
    from .projected_relation_ranking_v2 import relation_algorithm_signature_v2

    tags = frontier_tags(acquisition_policy)
    groups = {g.group_checksum for g in selected.groups}
    if (
        len(requests) != len(groups)
        or {r.authority_group_checksum for r in requests} != groups
        or set(configs) != groups
    ):
        raise ValueError("algorithm requires every original group request")
    return _digest(
        {
            "version": "graph-runtime-v2",
            "resolver_policies": sorted(
                {
                    (p.resolver_version, p.resolution_config_checksum)
                    for g in selected.groups
                    for p in g.scope.projections
                }
            ),
            "selection_digest": graph_selection_digest_v2(selected),
            "relation_kernels": sorted(
                (group, relation_algorithm_signature_v2(config))
                for group, config in configs.items()
            ),
            "group_requests": sorted(
                (r.authority_group_checksum, relation_request_digest(r.request))
                for r in requests
            ),
            "frontier_policy": tags[0].replace("_", "-") + "/" + acquisition_policy,
            "seed_policy": "global-prefix/group-equal-mass/global-round-robin-cap-v2",
            "selection_policy": (
                "query-lexical-predicate-reliability/provenance-associations-v2/"
                "scientific-endpoints/both-mention-transition-debits/"
                "global-whole-reference-association-prefix/requested-same-claim-conflicts"
            ),
            "evidence_caps": {
                "mentions_per_identity": 2,
                "evidence_per_transition": 3,
                "per_document": 3,
                "hydration_rows": 45,
            },
            "source_caps": {
                name: getattr(budget.turn_budget.limits, name)
                for name in ("unique_sources", "materialized_codepoints")
            },
            "caps": {
                name: getattr(caps, name)
                for name in (
                    "max_seeds",
                    "max_depth",
                    "max_nodes",
                    "max_edges",
                    "max_results",
                )
            },
            "cumulative_caps": {
                name: getattr(budget, name)
                for name in (
                    "max_identities",
                    "max_edges",
                    "max_source_rows",
                    "max_queries",
                    "max_wire_bytes",
                )
            },
        }
    )
