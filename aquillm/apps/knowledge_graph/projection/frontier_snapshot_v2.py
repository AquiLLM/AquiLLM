"""Build a closed mathematical snapshot from admitted validated pages."""

from collections import defaultdict
from struct import unpack

from apps.knowledge_graph.retrieval import projected_types as t
from apps.knowledge_graph.retrieval.frontier_ranking_validation_v2 import (
    frontier_algorithm,
)
from apps.knowledge_graph.retrieval.graph_query_contracts import GraphEvidenceRefV2
from apps.knowledge_graph.retrieval.ppr import PPRAlgorithmConfig
from apps.knowledge_graph.retrieval.projected_snapshot_codec import audit_order

from .topology_encoding import algorithm, chunk_evidence, evidence_signature, provenance


def frontier_config(caps, resolver):
    return PPRAlgorithmConfig(
        canonical_resolver_version=resolver,
        max_seeds=caps.max_seeds,
        max_hops=caps.max_depth,
        max_nodes=caps.max_nodes,
        max_edges=caps.max_edges,
        max_candidates=caps.max_results,
        max_per_document=min(3, caps.max_results),
    )


class FrontierAssemblyV2:
    def __init__(self, session, binding, caps, context):
        self.session, self.binding, self.caps, self.context = (
            session,
            binding,
            caps,
            context,
        )
        self.hops, self.members, self.groups, self.mentions, self.candidates = (
            {},
            {},
            {},
            {},
            {},
        )
        self.physical, self.evidence = {}, {}

    def add_copy(self, identity, hop, rows):
        members = [
            r for (label, _), r in rows.items() if label == "AutomaticMembership"
        ]
        if (
            len(set(self.members) | {m.entity_key for m in members})
            > self.caps.max_nodes
        ):
            return False
        self.hops[identity] = min(hop, self.hops.get(identity, hop))
        self.members.update((r.entity_key, r) for r in members)
        return True

    def candidate(self, row, generation, identities, relation_types, provenance_key):
        key = row.chunk_key
        representation = self.binding.representation(generation, row.document_key)
        signature = (generation, representation)
        previous = self.candidates.get(key)
        if previous is None:
            previous = [signature, set(), set(), set(), row.confidence]
            self.candidates[key] = previous
        if previous[0] != signature:
            raise ValueError("conflicting chunk representation binding")
        previous[1].update(identities)
        previous[2].add(provenance_key)
        previous[3].update(relation_types)
        previous[4] = max(previous[4], row.confidence)

    def add_relation(self, page, entry, generation, origin, hop):
        memberships = {m.entity_key: m for m in page.memberships}
        if len(set(self.members) | set(memberships)) > self.caps.max_nodes:
            return False
        relation = page.relations[0]
        source_entity, target_entity, direction = entry["order"][2:]

        def identity(k):
            return memberships[k].automatic_membership_key or k

        if identity(source_entity) != origin:
            raise ValueError("frontier relation source differs from requested identity")
        target = identity(target_entity)
        self.hops.setdefault(target, hop + 1)
        self.members.update(memberships)
        key = (origin, relation.relation_type, target, direction)
        quality = unpack(
            ">d", bytes(255 - b for b in bytes.fromhex(entry["order"][0]))
        )[0]
        current = self.groups.get(key)
        if current is None or (-quality, relation.relation_key) < (
            -current[0],
            current[1].relation_key,
        ):
            self.groups[key] = (quality, relation, {})
        current = self.groups[key]
        if current[1] == relation:
            current[2].update((r.evidence_key, r) for r in page.relation_evidence)
        for row in page.relation_evidence:
            self.candidate(
                row,
                generation,
                (origin, target),
                (relation.relation_type,),
                row.evidence_key,
            )
        return True

    def add_mention(self, page, generation, identity):
        for row in page.mention_evidence:
            self.mentions[generation, row.mention_key] = (identity, row)
            self.candidate(row, generation, (identity,), (), row.provenance_key)

    def build(self, request):
        ready = self.session.ready
        groups, physical, evidence = [], {}, {}
        for (source, predicate, target, direction), (weight, relation, rows) in sorted(
            self.groups.items()
        ):
            selected = sorted(
                rows.values(), key=lambda r: (-r.confidence, r.evidence_key)
            )[:3]
            selected.sort(
                key=lambda r: (
                    r.evidence_key,
                    r.chunk_key,
                    r.document_key,
                    r.chunk_number,
                )
            )
            physical[relation.relation_key] = relation
            evidence.update((r.evidence_key, r) for r in selected)
            groups.append(
                t.ProjectedRelationGroupV1(
                    source,
                    predicate,
                    target,
                    t.ProjectedRetrievalDirectionV1(direction),
                    weight,
                    self.hops[source] + 1,
                    tuple(chunk_evidence(r) for r in selected),
                )
            )

        def identity(k):
            return self.members[k].automatic_membership_key or k

        member_audits = [
            t.ProjectedAutomaticMembershipAuditV1(
                self.hops[identity(k)],
                k,
                identity(k),
                r.decision_checksum,
                r.resolver_version,
            )
            for k, r in self.members.items()
        ]
        physical_audits = [
            t.ProjectedPhysicalRelationAuditV1(
                min(
                    self.hops[identity(r.source_entity_key)],
                    self.hops[identity(r.target_entity_key)],
                )
                + 1,
                r.relation_key,
                r.artifact_key,
                r.source_entity_key,
                r.relation_type,
                r.target_entity_key,
            )
            for r in physical.values()
        ]
        physical_hops = {r.relation_key: r.discovery_hop for r in physical_audits}
        evidence_audits = [
            t.ProjectedRelationEvidenceAuditV1(
                physical_hops[r.relation_key], evidence_signature(r)
            )
            for r in evidence.values()
        ]
        by_identity = defaultdict(list)
        for identity_key, row in self.mentions.values():
            by_identity[identity_key].append(row)
        mentions, mention_audits = [], []
        for identity_key, rows in sorted(by_identity.items()):
            for row in sorted(
                rows, key=lambda r: (-r.confidence, r.provenance_key, r.chunk_key)
            )[:2]:
                item = chunk_evidence(row, row.provenance_key)
                mentions.append(t.ProjectedIdentityMentionV1(identity_key, item))
                mention_audits.append(
                    t.ProjectedFallbackMentionAuditV1(
                        self.hops[identity_key], identity_key, item
                    )
                )
        config = frontier_config(
            self.caps, ready.selected_generations[0].resolver_version
        )
        _, snapshot_caps = algorithm(self.caps, config.canonical_resolver_version)
        scope_rows = {
            key: row
            for rows in self.session.scope_rows.values()
            for key, row in rows.items()
            if key[0] == "ArtifactProvenance"
        }
        snapshot = t.ProjectedAuthorizedGraphSnapshotV1(
            frontier_algorithm(config, request, self.context),
            snapshot_caps,
            self.caps.max_depth,
            t.ProjectedAllowedScopeV1(
                tuple(sorted(d.document_key for d in ready.authorized_documents)),
                tuple(g.collection_key for g in ready.selected_generations),
                self.context.group_checksum,
            ),
            tuple(sorted(self.hops)),
            (),
            tuple(groups),
            tuple(
                sorted(
                    mentions, key=lambda r: (r.identity_key, r.evidence.provenance_key)
                )
            ),
            tuple(
                sorted(
                    (provenance(r) for r in scope_rows.values()),
                    key=lambda r: (r.scope_type.value, r.scope_key, r.artifact_key),
                )
            ),
            tuple(
                sorted(
                    (
                        *member_audits,
                        *physical_audits,
                        *evidence_audits,
                        *mention_audits,
                    ),
                    key=audit_order,
                )
            ),
        )
        candidates = tuple(
            GraphEvidenceRefV2(
                chunk,
                values[0][0],
                values[0][1],
                self.context.group_checksum,
                tuple(sorted(values[1])),
                tuple(sorted(values[2])),
                tuple(sorted(values[3])),
                values[4],
            )
            for chunk, values in sorted(self.candidates.items())
        )
        return snapshot, candidates
