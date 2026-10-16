"""Exact V2 signature context; never grants projection authority."""

from dataclasses import asdict, dataclass, fields

from apps.knowledge_graph.projection.frontier_index_contracts_v2 import digest

from .graph_query_contracts import GraphRelationRequestV2
from .ppr import PPRAlgorithmConfig, canonical_algorithm_json
from .projected_types import (
    ProjectedAlgorithmSignatureV1,
    ProjectedAuthorizedGraphSnapshotV1,
)
from .topology.contracts import (
    projected_seed_checksum,
    validate_projected_seed_sequence,
)

TAGS = (
    "canonical_frontier_v2",
    "frontier_transition_v2",
    "frontier_evidence_v2",
    "frontier_seed_v2",
)
POLICY = "global-priority-quality-keyset-v2/nonexhaustive/authorized-transit"
CONSUMED_POLICY = (
    "global-priority-quality-keyset-v3/nonexhaustive/authorized-transit/"
    "consumed-source-closure"
)
CONSUMED_TAGS = (
    "canonical_frontier_v3", "frontier_transition_v2",
    "frontier_evidence_v3", "frontier_seed_v2",
)


def frontier_tags(policy):
    if type(policy) is not str or policy not in (POLICY, CONSUMED_POLICY):
        raise ValueError("frontier acquisition policy differs")
    return CONSUMED_TAGS if policy == CONSUMED_POLICY else TAGS


@dataclass(frozen=True, slots=True)
class FrontierRankingContextV2:
    group_checksum: str
    max_identities: int
    max_edges: int
    max_source_rows: int
    max_queries: int
    max_wire_bytes: int
    acquisition_policy: str = POLICY

    def __post_init__(self):
        from ._graph_v2_validation import checksum

        checksum(self.group_checksum, "group_checksum")
        for name, maximum in (
            ("max_identities", 200),
            ("max_edges", 1000),
            ("max_source_rows", 5000),
            ("max_queries", 64),
            ("max_wire_bytes", 1048576),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError("frontier context cap is invalid")
        frontier_tags(self.acquisition_policy)


def frontier_algorithm(config, request, context):
    if (
        type(config) is not PPRAlgorithmConfig
        or type(request) is not GraphRelationRequestV2
    ):
        raise TypeError("frontier signature requires exact config and request")
    if type(context) is not FrontierRankingContextV2:
        raise TypeError("frontier signature requires its captured context")
    context.__post_init__()
    tags = frontier_tags(context.acquisition_policy)
    signature = digest(
        {
            "tags": tags,
            "config": canonical_algorithm_json(config).decode(),
            "request": asdict(request),
            "context": asdict(context),
        }
    )
    return ProjectedAlgorithmSignatureV1(*tags, signature)


def validate_frontier_ranking(*, snapshot, seeds, request, config, context):
    if type(snapshot) is not ProjectedAuthorizedGraphSnapshotV1:
        raise TypeError("snapshot must be exact")
    expected = frontier_algorithm(config, request, context)
    if (
        snapshot.algorithm != expected
        or snapshot.allowed_scope.scope_version_signature != context.group_checksum
    ):
        raise ValueError("frontier signature or captured group differs")
    if any(
        getattr(snapshot.caps, f.name) != getattr(config, f.name)
        for f in fields(snapshot.caps)
    ):
        raise ValueError("frontier snapshot/config caps differ")
    if (
        len(snapshot.identity_keys) > context.max_identities
        or len(snapshot.relation_groups) > context.max_edges
    ):
        raise ValueError("frontier snapshot exceeds cumulative limits")
    validate_projected_seed_sequence(
        seeds,
        maximum=config.max_seeds,
        expected_checksum=projected_seed_checksum(seeds),
    )
    if any(s.identity_key not in snapshot.identity_keys for s in seeds):
        raise ValueError("frontier seeds are outside snapshot")
    snapshot.__post_init__()
