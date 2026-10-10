"""Closed private provenance-to-mention/claim associations, never union inference."""

from dataclasses import dataclass

from ._graph_v2_validation import checksum, relation


@dataclass(frozen=True, slots=True)
class GraphEvidenceAssociationV2:
    provenance_key: str
    kind: str
    source_identity_key: str
    target_identity_key: str | None
    relation_type: str | None
    direction: str | None

    def __post_init__(self):
        checksum(self.provenance_key, "association provenance")
        checksum(self.source_identity_key, "association source")
        if type(self.kind) is not str or self.kind not in ("mention", "relation"):
            raise ValueError("association kind is invalid")
        if self.kind == "mention":
            if any(
                v is not None
                for v in (self.target_identity_key, self.relation_type, self.direction)
            ):
                raise ValueError("mention cannot declare a relation")
        else:
            checksum(self.target_identity_key, "association target")
            relation(self.relation_type)
            if type(self.direction) is not str or self.direction not in (
                "directed",
                "undirected",
            ):
                raise ValueError("association requires scientific relation direction")
            if (
                self.direction == "undirected"
                and self.source_identity_key > self.target_identity_key
            ):
                raise ValueError("undirected association endpoints must be canonical")


def association_unions(associations):
    identities, predicates = set(), set()
    for item in associations:
        identities.add(item.source_identity_key)
        if item.kind == "relation":
            identities.add(item.target_identity_key)
            predicates.add(item.relation_type)
    return tuple(sorted(identities)), tuple(sorted(predicates))


def validate_associations(ref):
    rows = ref.associations
    if type(rows) is not tuple or any(
        type(r) is not GraphEvidenceAssociationV2 for r in rows
    ):
        raise TypeError("associations must be exact private values")
    if not rows or len(rows) > 3000:
        raise ValueError("association work exceeds evidence allowance")
    keys = tuple(r.provenance_key for r in rows)
    if keys != tuple(sorted(set(keys))) or keys != ref.evidence_provenance_keys:
        raise ValueError("association provenance must cover the reference exactly")
    identities, predicates = association_unions(rows)
    if identities != ref.identity_keys or predicates != ref.relation_types:
        raise ValueError("reference unions differ from actual evidence associations")


def association_prefix(references, *, maximum):
    """Keep whole references within one cumulative bound, without partial maps."""
    if type(maximum) is not int or not 0 <= maximum <= 3000:
        raise ValueError("association allowance exceeds the existing evidence cap")
    retained, used = [], 0
    for ref in references:
        validate_associations(ref)
        if used + len(ref.associations) > maximum:
            break
        retained.append(ref)
        used += len(ref.associations)
    return tuple(retained)
