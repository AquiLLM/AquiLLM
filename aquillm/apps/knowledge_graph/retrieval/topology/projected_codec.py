"""Closed JSON-family decoding for provider-neutral projected snapshots."""

from __future__ import annotations

import json
from collections.abc import Mapping

from .. import projected_types as t

_MAX_JSON_BYTES = 2_000_000


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


def decode_projected_snapshot_json(raw: object) -> t.ProjectedAuthorizedGraphSnapshotV1:
    if type(raw) is not str or len(raw.encode("utf-8")) > _MAX_JSON_BYTES:
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


def _one_json(rows, field: str, expected: frozenset[str]):
    if (
        type(rows) is not tuple
        or len(rows) != 1
        or not isinstance(rows[0], Mapping)
        or set(rows[0]) != {field}
    ):
        raise ValueError("topology response family envelope is not exact")
    raw = rows[0][field]
    if type(raw) is not str or len(raw.encode("utf-8")) > _MAX_JSON_BYTES:
        raise ValueError("topology response family is not bounded JSON")
    value = json.loads(raw)
    if type(value) is not dict or set(value) != expected:
        raise ValueError("topology response family is partial or malformed")
    if (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        != raw
    ):
        raise ValueError("topology response family is not canonical JSON")
    return value


def compose_projected_snapshot_families(
    *, memberships: object, relations: object, evidence: object
) -> t.ProjectedAuthorizedGraphSnapshotV1:
    base_keys = frozenset(
        {
            "algorithm",
            "caps",
            "load_max_hops",
            "allowed_scope",
            "identity_keys",
            "seed_identities",
            "relation_groups",
            "mentions",
            "artifact_provenance",
            "audit_rows",
        }
    )
    base = _one_json(memberships, "snapshot_json", base_keys)
    relation = _one_json(
        relations, "section_json", frozenset({"relation_groups", "audit_rows"})
    )
    evidence_rows = _one_json(
        evidence, "section_json", frozenset({"mentions", "audit_rows"})
    )
    if (
        base["relation_groups"]
        or base["mentions"]
        or any(row.get("kind") != "automatic_membership" for row in base["audit_rows"])
    ):
        raise ValueError("membership family contains another topology family")
    if any(row.get("kind") != "physical_relation" for row in relation["audit_rows"]):
        raise ValueError("relation family contains invalid audit rows")
    if any(
        row.get("kind") not in {"fallback_mention", "relation_evidence"}
        for row in evidence_rows["audit_rows"]
    ):
        raise ValueError("evidence family contains invalid audit rows")
    base["relation_groups"] = relation["relation_groups"]
    base["mentions"] = evidence_rows["mentions"]
    base["audit_rows"] = sorted(
        (*base["audit_rows"], *relation["audit_rows"], *evidence_rows["audit_rows"]),
        key=lambda row: (
            row["discovery_hop"],
            row["kind"],
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ),
    )
    raw = json.dumps(base, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(raw.encode("utf-8")) > _MAX_JSON_BYTES:
        raise ValueError("composed topology snapshot exceeds its byte cap")
    return decode_projected_snapshot_json(raw)


__all__ = ["compose_projected_snapshot_families", "decode_projected_snapshot_json"]
