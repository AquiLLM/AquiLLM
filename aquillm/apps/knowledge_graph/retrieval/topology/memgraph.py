from __future__ import annotations

import json
from dataclasses import asdict
from math import isfinite
from time import monotonic, monotonic_ns

from lib.retrieval_redaction import MAX_RETRIEVAL_LOG_ELAPSED_MS, RetrievalLogReason

from .. import projected_types as t
from ..readiness_diagnostics import record_readiness_failure
from . import contracts as c
from .diagnostics import TopologyDiagnosticPhase as Phase
from .diagnostics import record_topology_failure
from .failures import TopologyLoadError, TopologyResultCapError
from .gateway_client import TopologyGatewayRequestError
from .gateway_contracts import GatewayFailureReason
from .gateway_snapshot_contracts import (
    TopologyGatewayFamilyTransportV2,
    TopologyGatewaySnapshotV2,
)
from .projected_codec import (
    compose_projected_snapshot_families,
    decode_projected_snapshot_json,
)

_GATEWAY_LOCAL = {
    (
        c.HybridBranchKind.DIRECT,
        GatewayFailureReason.RESULT_CAP,
    ): c.TopologyFailureReason.DIRECT_TOPOLOGY_INVALID,
    (
        c.HybridBranchKind.EXTENDED,
        GatewayFailureReason.RESULT_CAP,
    ): c.TopologyFailureReason.EXTENDED_TOPOLOGY_INVALID,
    (
        c.HybridBranchKind.DIRECT,
        GatewayFailureReason.DEADLINE,
    ): c.TopologyFailureReason.DIRECT_TOPOLOGY_TIMEOUT,
    (
        c.HybridBranchKind.EXTENDED,
        GatewayFailureReason.DEADLINE,
    ): c.TopologyFailureReason.EXTENDED_TOPOLOGY_TIMEOUT,
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parameters(ready, seeds, caps) -> dict[str, str]:
    return {
        "bundle_checksum": ready.bundle_checksum,
        "generation_keys_json": _canonical(
            [row.generation_key for row in ready.selected_generations]
        ),
        "document_keys_json": _canonical(
            [row.document_key for row in ready.authorized_documents]
        ),
        "membership_checksums_json": _canonical(
            [row.membership_checksum for row in ready.selected_generations]
        ),
        "seed_keys_json": _canonical([row.identity_key for row in seeds]),
        "selected_generations_json": _canonical(
            [asdict(row) for row in ready.selected_generations]
        ),
        "authorized_documents_json": _canonical(
            [asdict(row) for row in ready.authorized_documents]
        ),
        "authorization_context_signature": ready.authorization_context_signature,
        "seeds_json": _canonical(
            [
                {"identity_key": row.identity_key, "mass": row.mass.hex()}
                for row in seeds
            ]
        ),
        "seed_checksum": c.projected_seed_checksum(seeds),
        "caps_json": _canonical(
            {**asdict(caps), "branch_kind": caps.branch_kind.value}
        ),
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
        parameters = _parameters(ready, seeds, caps)
        limits = (
            (c.TopologyQueryName.GENERATION_MANIFESTS, len(ready.selected_generations)),
            (c.TopologyQueryName.AUTOMATIC_MEMBERSHIPS, caps.max_nodes),
            (c.TopologyQueryName.RELATION_TOPOLOGY, caps.max_edges),
            (c.TopologyQueryName.EVIDENCE_MENTIONS, 3_000 + caps.max_nodes * 2),
        )
        responses = {}
        snapshot_response = None
        use_snapshot = getattr(self.driver, "snapshot_enabled", False) is True
        try:
            query, maximum = limits[0]
            manifest_started_ns = monotonic_ns()
            if use_snapshot:
                if monotonic() >= deadline:
                    raise TimeoutError
                snapshot_response = self.driver.execute_snapshot(
                    parameters=parameters,
                    deadline=deadline,
                )
                if monotonic() >= deadline:
                    raise TimeoutError
                if type(snapshot_response) not in {
                    TopologyGatewaySnapshotV2,
                    TopologyGatewayFamilyTransportV2,
                }:
                    raise TopologyLoadError(
                        c.TopologyFailureReason.BACKEND_SCHEMA_MISMATCH
                    )
                responses[query] = snapshot_response.manifests
            else:
                responses[query] = self.driver.execute_read(
                    query=query,
                    parameters=parameters,
                    deadline=deadline,
                    max_records=maximum,
                )
            if not _manifest_matches(responses[query], ready):
                record_readiness_failure(
                    reason=RetrievalLogReason.READINESS_MANIFEST,
                    elapsed_ms=min(
                        (monotonic_ns() - manifest_started_ns) / 1_000_000,
                        MAX_RETRIEVAL_LOG_ELAPSED_MS,
                    ),
                )
                raise TopologyLoadError(c.TopologyFailureReason.READINESS_MISMATCH)
            for query, maximum in (
                ()
                if type(snapshot_response) is TopologyGatewaySnapshotV2
                else limits[1:]
            ):
                if use_snapshot and monotonic() >= deadline:
                    raise TimeoutError
                responses[query] = self.driver.execute_read(
                    query=query,
                    parameters=parameters,
                    deadline=deadline,
                    max_records=maximum,
                )
        except TopologyResultCapError as error:
            reason = (
                c.TopologyFailureReason.DIRECT_TOPOLOGY_INVALID
                if caps.branch_kind is c.HybridBranchKind.DIRECT
                else c.TopologyFailureReason.EXTENDED_TOPOLOGY_INVALID
            )
            raise TopologyLoadError(reason) from error
        except TopologyLoadError:
            raise
        except TopologyGatewayRequestError as error:
            raise TopologyLoadError(
                _GATEWAY_LOCAL[(caps.branch_kind, error.reason)]
            ) from error
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
        invalid = (
            c.TopologyFailureReason.DIRECT_TOPOLOGY_INVALID
            if caps.branch_kind is c.HybridBranchKind.DIRECT
            else c.TopologyFailureReason.EXTENDED_TOPOLOGY_INVALID
        )
        started = monotonic()
        try:
            snapshot = (
                decode_projected_snapshot_json(snapshot_response.snapshot_json)
                if type(snapshot_response) is TopologyGatewaySnapshotV2
                else compose_projected_snapshot_families(
                    memberships=responses[c.TopologyQueryName.AUTOMATIC_MEMBERSHIPS],
                    relations=responses[c.TopologyQueryName.RELATION_TOPOLOGY],
                    evidence=responses[c.TopologyQueryName.EVIDENCE_MENTIONS],
                )
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
            if use_snapshot and monotonic() >= deadline:
                raise TopologyLoadError(
                    _GATEWAY_LOCAL[(caps.branch_kind, GatewayFailureReason.DEADLINE)]
                )
            return snapshot
        except (KeyError, TypeError, ValueError) as error:
            record_topology_failure(
                phase=Phase.SNAPSHOT_COMPOSE,
                started=started,
                branch=caps.branch_kind.value,
            )
            raise TopologyLoadError(invalid) from error
