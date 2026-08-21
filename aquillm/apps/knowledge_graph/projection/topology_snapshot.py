"""Assemble one bounded authorized topology snapshot from projected records."""

from __future__ import annotations

from collections import defaultdict

from apps.knowledge_graph.retrieval import projected_types as t
from apps.knowledge_graph.retrieval.projected_snapshot_codec import audit_order

from .topology_encoding import (
    algorithm,
    chunk_evidence,
    evidence_signature,
    provenance,
)
from .topology_selection import select_topology_groups


def _identity(membership) -> str:
    return membership.automatic_membership_key or membership.entity_key


def _rows(bundles, authorized_documents):
    authorized = {
        (row.generation_key, row.document_key) for row in authorized_documents
    }
    entities, memberships, relations, evidence, mentions = {}, {}, [], [], []
    semantics, provenance = {}, []
    for bundle in bundles:
        generation = bundle.generation.generation_key
        entity_map = {row.entity_key: row for row in bundle.entities}
        member_map = {row.entity_key: row for row in bundle.automatic_memberships}
        entities.update(entity_map)
        memberships.update(member_map)
        semantics.update(
            ((row.artifact_key, row.relation_type), row.direction)
            for row in bundle.relation_semantics
        )
        relations.extend(bundle.relations)
        evidence.extend(
            row
            for row in bundle.evidence
            if (generation, row.document_key) in authorized
        )
        mentions.extend(
            row
            for row in bundle.entity_mentions
            if (generation, row.document_key) in authorized
        )
        scopes = {
            ("collection", bundle.generation.collection_key),
            *(
                ("document", document)
                for selected_generation, document in authorized
                if selected_generation == generation
            ),
        }
        provenance.extend(
            row
            for row in bundle.artifact_provenance
            if (row.scope_type, row.scope_key) in scopes
        )
    return entities, memberships, relations, evidence, mentions, semantics, provenance


def build_projected_topology_snapshot(*, ready, seeds, caps, bundles):
    (
        entities,
        memberships,
        relations,
        evidence,
        mention_rows,
        semantics,
        provenance_rows,
    ) = _rows(bundles, ready.authorized_documents)
    identity_by_entity = {key: _identity(member) for key, member in memberships.items()}
    entities_by_identity = defaultdict(list)
    for entity_key, identity_key in identity_by_entity.items():
        entities_by_identity[identity_key].append(entity_key)
    evidence_by_relation = defaultdict(list)
    for row in evidence:
        evidence_by_relation[row.relation_key].append(row)
    seed_keys = {row.identity_key for row in seeds}
    identity_hops = {key: 0 for key in sorted(seed_keys) if key in entities_by_identity}
    transitions: dict[tuple[str, str, str, t.ProjectedRetrievalDirectionV1], list] = (
        defaultdict(list)
    )
    for hop in range(caps.max_depth):
        for relation in sorted(relations, key=lambda row: row.relation_key):
            if not evidence_by_relation[relation.relation_key]:
                continue
            source = identity_by_entity[relation.source_entity_key]
            target = identity_by_entity[relation.target_entity_key]
            if source == target:
                continue
            direction = semantics[(relation.artifact_key, relation.relation_type)]
            candidates = []
            if identity_hops.get(source) == hop:
                forward = (
                    t.ProjectedRetrievalDirectionV1.UNDIRECTED
                    if direction == "undirected"
                    else t.ProjectedRetrievalDirectionV1.FORWARD
                )
                candidates.append((source, target, forward))
            if identity_hops.get(target) == hop:
                reverse = (
                    t.ProjectedRetrievalDirectionV1.UNDIRECTED
                    if direction == "undirected"
                    else t.ProjectedRetrievalDirectionV1.REVERSE_DIRECTED
                )
                candidates.append((target, source, reverse))
            for origin, destination, traversal in candidates:
                if (
                    destination not in identity_hops
                    and len(identity_hops) < caps.max_nodes
                ):
                    identity_hops[destination] = hop + 1
                if destination in identity_hops:
                    transitions[
                        (origin, relation.relation_type, destination, traversal)
                    ].append(relation)
    selected_groups, selected_entities = select_topology_groups(
        transitions=transitions,
        evidence_by_relation=evidence_by_relation,
        entities=entities,
        entities_by_identity=entities_by_identity,
        identity_by_entity=identity_by_entity,
        identity_hops=identity_hops,
        max_nodes=caps.max_nodes,
        max_edges=caps.max_edges,
    )
    used_relations, groups, evidence_audits = {}, [], {}
    for key, weight, physical, selected_evidence in selected_groups:
        source, relation_type, target, direction = key
        used_relations[physical.relation_key] = physical
        for row in selected_evidence:
            evidence_audits[row.evidence_key] = row
        groups.append(
            t.ProjectedRelationGroupV1(
                source,
                relation_type,
                target,
                direction,
                weight,
                identity_hops[source] + 1,
                tuple(chunk_evidence(row) for row in selected_evidence),
            )
        )
    memberships_audit = [
        t.ProjectedAutomaticMembershipAuditV1(
            identity_hops[identity_by_entity[key]],
            key,
            identity_by_entity[key],
            memberships[key].decision_checksum,
            memberships[key].resolver_version,
        )
        for key in selected_entities
    ]
    identities = tuple(
        sorted({row.automatic_membership_key for row in memberships_audit})
    )
    mentions, mention_audits = [], []
    for identity in identities:
        candidates = [
            row
            for row in mention_rows
            if identity_by_entity[row.entity_key] == identity
        ]
        candidates.sort(
            key=lambda row: (-row.confidence, row.provenance_key, row.chunk_key)
        )
        for row in candidates[:2]:
            evidence_row = chunk_evidence(row, row.provenance_key)
            mentions.append(t.ProjectedIdentityMentionV1(identity, evidence_row))
            mention_audits.append(
                t.ProjectedFallbackMentionAuditV1(
                    identity_hops[identity], identity, evidence_row
                )
            )
    mentions.sort(key=lambda row: (row.identity_key, row.evidence.provenance_key))
    physical_audits = [
        t.ProjectedPhysicalRelationAuditV1(
            min(
                identity_hops[identity_by_entity[row.source_entity_key]],
                identity_hops[identity_by_entity[row.target_entity_key]],
            )
            + 1,
            row.relation_key,
            row.artifact_key,
            row.source_entity_key,
            row.relation_type,
            row.target_entity_key,
        )
        for row in used_relations.values()
    ]
    physical_hops = {row.relation_key: row.discovery_hop for row in physical_audits}
    relation_evidence_audits = [
        t.ProjectedRelationEvidenceAuditV1(
            physical_hops[row.relation_key], evidence_signature(row)
        )
        for row in evidence_audits.values()
    ]
    audits = sorted(
        (
            *memberships_audit,
            *physical_audits,
            *relation_evidence_audits,
            *mention_audits,
        ),
        key=audit_order,
    )
    algorithm_signature, snapshot_caps = algorithm(
        caps, ready.selected_generations[0].resolver_version
    )
    expected_scopes = len(ready.selected_generations) + len(ready.authorized_documents)
    if len(provenance_rows) != expected_scopes:
        raise ValueError("authorized artifact provenance is incomplete")
    return t.ProjectedAuthorizedGraphSnapshotV1(
        algorithm_signature,
        snapshot_caps,
        caps.max_depth,
        t.ProjectedAllowedScopeV1(
            tuple(sorted(row.document_key for row in ready.authorized_documents)),
            tuple(row.collection_key for row in ready.selected_generations),
            ready.bundle_checksum,
        ),
        identities,
        (),
        tuple(groups),
        tuple(mentions),
        tuple(
            sorted(
                (provenance(row) for row in provenance_rows),
                key=lambda row: (row.scope_type.value, row.scope_key, row.artifact_key),
            )
        ),
        tuple(audits),
    )


__all__ = ["build_projected_topology_snapshot"]
