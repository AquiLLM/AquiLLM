from __future__ import annotations

import json
from functools import partial
from math import isfinite

from .. import projected_types as t
from . import contracts as c
from .failures import TopologyLoadError
from .projected_codec import compose_projected_snapshot_families


def _parameters(ready, seeds) -> dict[str, str]:
    compact = partial(json.dumps, separators=(",", ":"))
    return {
        "bundle_checksum": ready.bundle_checksum,
        "generation_keys_json": compact(
            [row.generation_key for row in ready.selected_generations]
        ),
        "document_keys_json": compact(
            [row.document_key for row in ready.authorized_documents]
        ),
        "membership_checksums_json": compact(
            [row.membership_checksum for row in ready.selected_generations]
        ),
        "seed_keys_json": compact([row.identity_key for row in seeds]),
    }


def _manifest_matches(rows, ready) -> bool:
    if type(rows) is not tuple or len(rows) != len(ready.selected_generations):
        return False
    expected = tuple(
        (
            row.collection_key,
            row.generation_key,
            row.projection_key,
            row.active_artifact_key,
            row.graph_checksum,
            row.membership_checksum,
        )
        for row in ready.selected_generations
    )
    try:
        observed = tuple(
            (
                row["collection_key"],
                row["generation_key"],
                row["projection_key"],
                row["active_artifact_key"],
                row["graph_checksum"],
                row["membership_checksum"],
            )
            for row in rows
        )
    except (KeyError, TypeError):
        return False
    return observed == expected


class MemgraphProjectedTopologyLoader:
    def __init__(self, driver: c.ProjectedTopologyQueryDriver):
        if not isinstance(driver, c.ProjectedTopologyQueryDriver):
            raise TypeError("driver must implement ProjectedTopologyQueryDriver")
        self.driver = driver

    def load(
        self,
        *,
        ready: c.ReadyGenerationBundleV1,
        seeds: tuple[c.ProjectedSeedV1, ...],
        caps: c.TopologyCapsV1,
        deadline: float,
    ) -> t.ProjectedAuthorizedGraphSnapshotV1:
        if (
            type(ready) is not c.ReadyGenerationBundleV1
            or type(caps) is not c.TopologyCapsV1
        ):
            raise TypeError("ready and caps must be exact topology contracts")
        if type(deadline) is not float or not isfinite(deadline) or deadline <= 0.0:
            raise ValueError("deadline must be a finite positive monotonic float")
        c.validate_projected_seed_sequence(
            seeds,
            maximum=caps.max_seeds,
            expected_checksum=c.projected_seed_checksum(seeds),
        )
        parameters = _parameters(ready, seeds)
        limits = (
            (c.TopologyQueryName.GENERATION_MANIFESTS, len(ready.selected_generations)),
            (c.TopologyQueryName.AUTOMATIC_MEMBERSHIPS, caps.max_nodes),
            (c.TopologyQueryName.RELATION_TOPOLOGY, caps.max_edges),
            (c.TopologyQueryName.EVIDENCE_MENTIONS, 3_000 + caps.max_nodes * 2),
        )
        responses = {}
        try:
            for query, maximum in limits:
                responses[query] = self.driver.execute_read(
                    query=query,
                    parameters=parameters,
                    deadline=deadline,
                    max_records=maximum,
                )
        except TopologyLoadError:
            raise
        except TimeoutError as error:
            reason = (
                c.TopologyFailureReason.DIRECT_TOPOLOGY_TIMEOUT
                if caps.branch_kind is c.HybridBranchKind.DIRECT
                else c.TopologyFailureReason.EXTENDED_TOPOLOGY_TIMEOUT
            )
            raise TopologyLoadError(reason) from error
        except Exception as error:
            raise TopologyLoadError(
                c.TopologyFailureReason.BACKEND_UNAVAILABLE
            ) from error
        if not _manifest_matches(
            responses[c.TopologyQueryName.GENERATION_MANIFESTS], ready
        ):
            raise TopologyLoadError(c.TopologyFailureReason.READINESS_MISMATCH)
        invalid = (
            c.TopologyFailureReason.DIRECT_TOPOLOGY_INVALID
            if caps.branch_kind is c.HybridBranchKind.DIRECT
            else c.TopologyFailureReason.EXTENDED_TOPOLOGY_INVALID
        )
        try:
            snapshot = compose_projected_snapshot_families(
                memberships=responses[c.TopologyQueryName.AUTOMATIC_MEMBERSHIPS],
                relations=responses[c.TopologyQueryName.RELATION_TOPOLOGY],
                evidence=responses[c.TopologyQueryName.EVIDENCE_MENTIONS],
            )
            expected_documents = tuple(
                sorted(row.document_key for row in ready.authorized_documents)
            )
            expected_collections = tuple(
                row.collection_key for row in ready.selected_generations
            )
            if (
                snapshot.allowed_scope.document_keys != expected_documents
                or snapshot.allowed_scope.collection_keys != expected_collections
                or len(snapshot.identity_keys) > caps.max_nodes
                or len(snapshot.relation_groups) > caps.max_edges
                or snapshot.load_max_hops > caps.max_depth
            ):
                raise ValueError("snapshot scope or caps disagree with request")
            return snapshot
        except (KeyError, TypeError, ValueError) as error:
            raise TopologyLoadError(invalid) from error
