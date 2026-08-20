"""Deterministic PageRank over closed opaque-key projected snapshots."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256

from .ppr import (
    EVIDENCE_VERSION,
    SEED_VERSION,
    TRANSITION_VERSION,
    PPRAlgorithmConfig,
    canonical_algorithm_json,
)
from .ppr_kernel import WeightedEdge, run_ppr_kernel
from .projected_types import (
    ProjectedAuthorizedGraphSnapshotV1,
)
from .topology.contracts import (
    ProjectedSeedV1,
    projected_seed_checksum,
    validate_projected_seed_sequence,
)


@dataclass(frozen=True, slots=True)
class ProjectedPPRResultV1:
    """Opaque score vector, rank order, and provider-neutral trace bytes."""

    scores: tuple[tuple[str, float], ...]
    ranked_identity_keys: tuple[str, ...]
    trace_bytes: bytes

    def __post_init__(self) -> None:
        if type(self.scores) is not tuple or any(
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not float
            for row in self.scores
        ):
            raise TypeError("scores must be exact opaque-key float pairs")
        keys = tuple(key for key, _ in self.scores)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("scores must be unique and opaque-key sorted")
        if (
            type(self.ranked_identity_keys) is not tuple
            or set(self.ranked_identity_keys) != set(keys)
            or len(self.ranked_identity_keys) != len(keys)
        ):
            raise ValueError("ranked identities must cover scores exactly")
        if type(self.trace_bytes) is not bytes:
            raise TypeError("trace_bytes must be exact bytes")


def _trace_bytes(
    scores: tuple[tuple[str, float], ...], ranked: tuple[str, ...]
) -> bytes:
    return json.dumps(
        {
            "ranked_identity_keys": ranked,
            "scores": [[key, score.hex()] for key, score in scores],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _validate_config(
    snapshot: ProjectedAuthorizedGraphSnapshotV1,
    config: PPRAlgorithmConfig,
) -> None:
    if type(config) is not PPRAlgorithmConfig:
        raise TypeError("config must be an exact PPRAlgorithmConfig")
    caps = snapshot.caps
    expected = (
        config.max_seeds,
        config.max_scope_documents,
        config.max_scope_collections,
        config.max_hops,
        config.max_nodes,
        config.max_edges,
        config.max_evidence_rows,
        config.max_evidence_per_edge,
        config.max_mentions_per_entity,
    )
    observed = tuple(
        getattr(caps, field)
        for field in (
            "max_seeds",
            "max_scope_documents",
            "max_scope_collections",
            "max_hops",
            "max_nodes",
            "max_edges",
            "max_evidence_rows",
            "max_evidence_per_edge",
            "max_mentions_per_entity",
        )
    )
    if observed != expected:
        raise ValueError("config does not match the projected snapshot caps")
    algorithm = snapshot.algorithm
    expected_signature = sha256(
        b"ppr_projected_v1\0" + canonical_algorithm_json(config)
    ).hexdigest()
    observed_algorithm = (
        algorithm.algorithm_version,
        algorithm.transition_version,
        algorithm.evidence_version,
        algorithm.seed_version,
        algorithm.algorithm_signature,
    )
    if observed_algorithm != (
        "ppr_projected_v1",
        TRANSITION_VERSION,
        EVIDENCE_VERSION,
        SEED_VERSION,
        expected_signature,
    ):
        raise ValueError("projected algorithm signature does not match config")


def _selection_key(group) -> tuple[object, ...]:
    return (
        -group.raw_weight,
        group.source_identity_key,
        group.relation_type,
        group.target_identity_key,
        group.direction.value,
    )


def _replay_groups(snapshot, seeds, config):
    eligible = tuple(
        sorted(
            (
                group
                for group in snapshot.relation_groups
                if group.admission_hop <= config.max_hops
            ),
            key=_selection_key,
        )
    )
    by_source = defaultdict(list)
    for group in eligible:
        by_source[group.source_identity_key].append(group)
    fanout = tuple(
        group
        for source in sorted(by_source)
        for group in sorted(by_source[source], key=_selection_key)[: config.max_fanout]
    )
    if len(fanout) <= config.max_edges:
        return fanout
    admitted = {seed.identity_key for seed in seeds}
    retained = []
    cap_reached = False
    for hop in range(1, config.max_hops + 1):
        frontier = sorted(
            (group for group in fanout if group.admission_hop == hop),
            key=_selection_key,
        )
        for group in frontier:
            if group.source_identity_key not in admitted:
                continue
            target_is_new = group.target_identity_key not in admitted
            if len(retained) >= config.max_edges or (
                target_is_new and len(admitted) >= config.max_nodes
            ):
                cap_reached = True
                break
            retained.append(group)
            admitted.add(group.target_identity_key)
        if cap_reached:
            break
    return tuple(retained)


def ppr_projected_v1(
    *,
    snapshot: ProjectedAuthorizedGraphSnapshotV1,
    seeds: tuple[ProjectedSeedV1, ...],
    config: PPRAlgorithmConfig,
) -> ProjectedPPRResultV1:
    """Rank a projected snapshot using opaque lexical order, never DB order."""

    if type(snapshot) is not ProjectedAuthorizedGraphSnapshotV1:
        raise TypeError("snapshot must be an exact projected snapshot")
    _validate_config(snapshot, config)
    validate_projected_seed_sequence(
        seeds,
        maximum=config.max_seeds,
        expected_checksum=projected_seed_checksum(seeds),
    )
    identities = set(snapshot.identity_keys)
    if any(seed.identity_key not in identities for seed in seeds):
        raise ValueError("projected seeds must reference snapshot identities")
    edges = tuple(
        WeightedEdge(
            group.source_identity_key, group.target_identity_key, group.raw_weight
        )
        for group in _replay_groups(snapshot, seeds, config)
    )
    kernel = run_ppr_kernel(
        nodes=snapshot.identity_keys,
        edges=edges,
        seeds={seed.identity_key: seed.mass for seed in seeds},
        config=config,
        order_key=lambda key: key,
    )
    scores = tuple(kernel.scores)
    score_map = dict(scores)
    ranked = tuple(sorted(score_map, key=lambda key: (-score_map[key], key)))
    return ProjectedPPRResultV1(scores, ranked, _trace_bytes(scores, ranked))


__all__ = ["ProjectedPPRResultV1", "ppr_projected_v1"]
