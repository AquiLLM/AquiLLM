"""Small builder-backed copies, equal-quality relations, and authenticated tampering."""

from copy import deepcopy
from dataclasses import replace

from apps.knowledge_graph.projection.frontier_adapter_v2 import FrontierSessionV2
from apps.knowledge_graph.projection.frontier_index_contracts_v2 import (
    decode_record,
    digest,
    make_record,
)
from apps.knowledge_graph.tests.frontier_fixture_support import (
    FrontierFixture,
    bridge_bundles,
    key,
)
from apps.knowledge_graph.tests.frontier_generation_batch_support import (
    GenerationDriver,
)


def multiple_bundles():
    first, second = bridge_bundles()
    source = first.relations[0].source_entity_key
    entity, relation = key("extra-copy"), key("extra-copy-relation")
    extra_entity = replace(
        next(r for r in first.entities if r.entity_key == source), entity_key=entity
    )
    extra_member = replace(
        next(r for r in first.automatic_memberships if r.entity_key == source),
        entity_key=entity,
        automatic_membership_key=source,
    )
    extra_relation = replace(
        first.relations[0], relation_key=relation, source_entity_key=entity
    )
    extra_evidence = replace(
        first.evidence[0],
        relation_key=relation,
        head_mapping_key=entity,
        evidence_key=key("extra-evidence"),
        provenance_key=key("extra-provenance"),
    )
    first = replace(
        first,
        entities=tuple(
            sorted((*first.entities, extra_entity), key=lambda r: r.entity_key)
        ),
        automatic_memberships=tuple(
            sorted(
                (*first.automatic_memberships, extra_member), key=lambda r: r.entity_key
            )
        ),
        relations=tuple(
            sorted((*first.relations, extra_relation), key=lambda r: r.relation_key)
        ),
        evidence=tuple(
            sorted((*first.evidence, extra_evidence), key=lambda r: r.evidence_key)
        ),
        counts=replace(
            first.counts,
            entity_count=3,
            automatic_membership_count=3,
            relation_count=2,
            evidence_count=2,
        ),
    )
    return first, second


class CopyFixture(FrontierFixture):
    def __init__(self, *, multiple=True, initialize=True):
        super().__init__(multiple_bundles() if multiple else None)
        self.driver = GenerationDriver(self.bundles)
        self.ledger = self.budget()
        self.session = FrontierSessionV2(
            self.driver,
            group=self.group,
            request=self.request,
            budget=self.ledger,
            deadline=self.deadline,
        )
        self.pairs = tuple(
            (g, self.seeds[0].identity_key) for g in sorted(self.session.selected)
        )
        if initialize:
            self.session.initialize()


def rewrite_index(fixture, generation, mutate):
    """Rehash a deliberately malformed tree before its fresh initial attestation."""
    original = fixture.driver.index.copy()
    rebuilt = {}

    def rebuild(key):
        if key is None:
            return None
        kind, value = decode_record(
            original[key], expected_key=key, generation=generation
        )
        value = deepcopy(value)
        if kind == "directory":
            for field in ("left", "right", "head"):
                value[field] = rebuild(value[field])
        elif kind == "entry":
            value["next"] = rebuild(value["next"])
        mutate(kind, value)
        row = make_record(generation, kind, value)
        rebuilt[row["index_key"]] = row
        return row["index_key"]

    marker = fixture.driver.markers[generation]
    root = rebuild(marker["frontier_root_key"])
    fixture.driver.index.update(rebuilt)
    checksum = digest(sorted(rebuilt))
    marker.update(
        frontier_root_key=root,
        frontier_root_checksum=root,
        frontier_checksum=checksum,
        validated_frontier_checksum=checksum,
        frontier_record_count=len(rebuilt),
    )
