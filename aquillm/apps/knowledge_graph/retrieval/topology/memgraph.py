from __future__ import annotations

import json
from functools import partial
from math import isfinite

from .. import projected_types as t
from . import contracts as c


class TopologyLoadError(RuntimeError):
    """Safe fixed-reason topology failure."""

    def __init__(self, reason: c.TopologyFailureReason):
        if type(reason) is not c.TopologyFailureReason:
            raise TypeError("reason must be an exact TopologyFailureReason")
        self.reason = reason
        super().__init__(reason.value)


def _evidence(value):
    return t.ProjectedChunkEvidenceV1(
        value["chunk_key"],
        value["document_key"],
        value["chunk_number"],
        float.fromhex(value["confidence"]),
        value["provenance_key"],
    )


def _signature(value):
    return t.ProjectedEvidenceSignatureV1(
        value["evidence_key"],
        value["relation_key"],
        value["relation_mention_key"],
        value["chunk_key"],
        value["document_key"],
        value["chunk_number"],
        float.fromhex(value["confidence"]),
        value["artifact_key"],
        value["source_document_key"],
        value["head_mention_key"],
        value["tail_mention_key"],
        value["relation_type"],
        value["head_mapping_key"],
        value["tail_mapping_key"],
        t.ProjectedEvidenceOrientationV1(value["orientation"]),
        value["ontology_checksum"],
        value["assembly_config_checksum"],
    )


def _provenance(value):
    return t.ProjectedArtifactProvenanceV1(
        value["artifact_key"],
        t.ProjectedScopeTypeV1(value["scope_type"]),
        value["scope_key"],
        value["collection_key"],
        value["rebuild_request_key"],
        value["evaluation_only"],
        value["build_key"],
        value["build_generation"],
        value["orchestration_version"],
        value["source_hash"],
        value["ontology_version"],
        value["ontology_checksum"],
        value["extractor_version"],
        value["resolver_version"],
        value["resolution_config_checksum"],
        value["filter_policy_version"],
        value["filter_policy_checksum"],
        value["embedding_model_signature"],
        value["assembly_version"],
        value["assembly_config_checksum"],
    )


def _audit(value):
    kind = value.get("kind")
    if kind == "automatic_membership":
        return t.ProjectedAutomaticMembershipAuditV1(
            value["discovery_hop"],
            value["entity_key"],
            value["automatic_membership_key"],
            value["decision_checksum"],
            value["resolver_version"],
        )
    if kind == "physical_relation":
        return t.ProjectedPhysicalRelationAuditV1(
            value["discovery_hop"],
            value["relation_key"],
            value["artifact_key"],
            value["source_entity_key"],
            value["relation_type"],
            value["target_entity_key"],
        )
    if kind == "relation_evidence":
        return t.ProjectedRelationEvidenceAuditV1(
            value["discovery_hop"], _signature(value["signature"])
        )
    if kind == "fallback_mention":
        return t.ProjectedFallbackMentionAuditV1(
            value["discovery_hop"], value["identity_key"], _evidence(value["evidence"])
        )
    raise ValueError("snapshot audit kind is not closed")


def _decode_snapshot(raw: object) -> t.ProjectedAuthorizedGraphSnapshotV1:
    if type(raw) is not str or len(raw.encode("utf-8")) > 2_000_000:
        raise ValueError("snapshot_json is not a bounded exact string")
    value = json.loads(raw)
    algorithm, caps, scope = value["algorithm"], value["caps"], value["allowed_scope"]
    snapshot = t.ProjectedAuthorizedGraphSnapshotV1(
        t.ProjectedAlgorithmSignatureV1(
            algorithm["algorithm_version"],
            algorithm["transition_version"],
            algorithm["evidence_version"],
            algorithm["seed_version"],
            algorithm["algorithm_signature"],
        ),
        t.ProjectedSnapshotCapsV1(
            caps["max_seeds"],
            caps["max_scope_documents"],
            caps["max_scope_collections"],
            caps["max_hops"],
            caps["max_nodes"],
            caps["max_edges"],
            caps["max_evidence_rows"],
            caps["max_evidence_per_edge"],
            caps["max_mentions_per_entity"],
        ),
        value["load_max_hops"],
        t.ProjectedAllowedScopeV1(
            tuple(scope["document_keys"]),
            tuple(scope["collection_keys"]),
            scope["scope_version_signature"],
        ),
        tuple(value["identity_keys"]),
        tuple(
            t.ProjectedSeedIdentityV1(row["seed_chunk_key"], row["identity_key"])
            for row in value["seed_identities"]
        ),
        tuple(
            t.ProjectedRelationGroupV1(
                row["source_identity_key"],
                row["relation_type"],
                row["target_identity_key"],
                t.ProjectedRetrievalDirectionV1(row["direction"]),
                float.fromhex(row["raw_weight"]),
                row["admission_hop"],
                tuple(_evidence(item) for item in row["evidence"]),
            )
            for row in value["relation_groups"]
        ),
        tuple(
            t.ProjectedIdentityMentionV1(
                row["identity_key"], _evidence(row["evidence"])
            )
            for row in value["mentions"]
        ),
        tuple(_provenance(row) for row in value["artifact_provenance"]),
        tuple(_audit(row) for row in value["audit_rows"]),
    )
    if t.canonical_projected_snapshot_bytes(snapshot).decode() != raw:
        raise ValueError("snapshot_json is not exact canonical projected bytes")
    return snapshot


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
            membership_rows = responses[c.TopologyQueryName.AUTOMATIC_MEMBERSHIPS]
            if type(membership_rows) is not tuple or len(membership_rows) != 1:
                raise ValueError("membership snapshot envelope is not exact")
            snapshot = _decode_snapshot(membership_rows[0]["snapshot_json"])
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
                or caps.max_depth > snapshot.load_max_hops
            ):
                raise ValueError("snapshot scope or caps disagree with request")
            return snapshot
        except (KeyError, TypeError, ValueError) as error:
            raise TopologyLoadError(invalid) from error
