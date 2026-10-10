"""Derive and retain evidence bindings from admitted physical source pages."""

from apps.knowledge_graph.retrieval.graph_evidence_associations_v2 import (
    GraphEvidenceAssociationV2,
    association_unions,
)
from apps.knowledge_graph.retrieval.graph_query_contracts import GraphEvidenceRefV2
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetExceeded,
)

from .frontier_validation_v2 import validate_relation_entry, validate_source_page


def page_associations(page, *, generation, ready, origin, entry=None):
    selected = next(
        r for r in ready.selected_generations if r.generation_key == generation
    )
    if not validate_source_page(page, selected=selected, ready=ready):
        raise ValueError("association source is outside captured authority")
    members = {
        r.entity_key: r.automatic_membership_key or r.entity_key
        for r in page.memberships
    }
    if entry is None:
        rows = page.mention_evidence
        if page.relation_evidence or any(members[r.entity_key] != origin for r in rows):
            raise ValueError("mention association differs from admitted identity")
        return rows, tuple(
            GraphEvidenceAssociationV2(
                r.provenance_key, "mention", members[r.entity_key], None, None, None
            )
            for r in rows
        )
    validate_relation_entry(page, entry)
    if members[entry["order"][2]] != origin or page.mention_evidence:
        raise ValueError("relation association differs from admitted origin")
    relation, semantic = page.relations[0], page.relation_semantics[0]
    source, target = (
        members[relation.source_entity_key],
        members[relation.target_entity_key],
    )
    if semantic.direction == "undirected":
        source, target = sorted((source, target))
    rows = page.relation_evidence
    return rows, tuple(
        GraphEvidenceAssociationV2(
            r.evidence_key,
            "relation",
            source,
            target,
            relation.relation_type,
            semantic.direction,
        )
        for r in rows
    )


class FrontierEvidenceV2:
    def __init__(self, binding, group, maximum):
        self.binding, self.group, self.maximum = binding, group, maximum
        self.provenance, self.chunks = {}, {}

    def add_page(self, page, *, generation, ready, origin, entry=None):
        rows, associations = page_associations(
            page, generation=generation, ready=ready, origin=origin, entry=entry
        )
        self._retain(rows, generation, associations)

    def _retain(self, rows, generation, associations):
        """Validate a complete page before mutating any retained candidate state."""
        pending, signatures = {}, {}
        for row, association in zip(rows, associations, strict=True):
            signature = (
                generation,
                self.binding.representation(generation, row.document_key),
                row.document_key,
                row.chunk_number,
            )
            previous = self.chunks.get(row.chunk_key)
            expected = signatures.setdefault(row.chunk_key, signature)
            if expected != signature or (
                previous is not None and previous[0] != signature
            ):
                raise ValueError("conflicting chunk source/representation binding")
            value = (row.chunk_key, generation, association)
            prior = pending.setdefault(association.provenance_key, value)
            if (
                prior != value
                or self.provenance.get(association.provenance_key, value) != value
            ):
                raise ValueError("conflicting provenance association")
        if len(self.provenance.keys() | pending.keys()) > self.maximum:
            raise FrontierBudgetExceeded("retained association allowance exhausted")
        self.provenance.update(pending)
        for row, association in zip(rows, associations, strict=True):
            current = self.chunks.setdefault(
                row.chunk_key, [signatures[row.chunk_key], {}, row.confidence]
            )
            current[1][association.provenance_key] = association
            current[2] = max(current[2], row.confidence)

    def build(self):
        result = []
        for chunk, (signature, values, confidence) in sorted(self.chunks.items()):
            rows = tuple(values[k] for k in sorted(values))
            identities, predicates = association_unions(rows)
            result.append(
                GraphEvidenceRefV2(
                    chunk,
                    signature[0],
                    signature[1],
                    self.group,
                    identities,
                    tuple(r.provenance_key for r in rows),
                    predicates,
                    confidence,
                    rows,
                )
            )
        return tuple(result)
