"""201 requested neighbors with an answer deliberately last by quality."""

from dataclasses import replace

from apps.knowledge_graph.tests.frontier_fixture_support import bridge_bundles, key


def hub_bundles(*, preferred_last=False):
    first, second = bridge_bundles()
    entities, members = list(first.entities), list(first.automatic_memberships)
    relations, evidence = list(first.relations), list(first.evidence)
    for i in range(200):
        entity = key("hub-entity:" + str(i))
        identity = key("hub-identity:" + str(i))
        relation = key("hub-relation:" + str(i))
        entities.append(replace(first.entities[1], entity_key=entity))
        members.append(
            replace(
                first.automatic_memberships[1],
                entity_key=entity,
                automatic_membership_key=identity,
            )
        )
        relations.append(
            replace(first.relations[0], relation_key=relation, target_entity_key=entity)
        )
        evidence.append(
            replace(
                first.evidence[0],
                relation_key=relation,
                evidence_key=key("hub-evidence:" + str(i)),
                tail_mapping_key=entity,
                confidence=0.01 if i == 199 else 0.75,
            )
        )
    semantics = first.relation_semantics
    if preferred_last:
        relations[-1] = replace(relations[-1], relation_type="supports")
        evidence[-1] = replace(evidence[-1], relation_type="supports")
        semantics = (
            *semantics,
            replace(
                semantics[0], relation_type="supports", semantics_key=key("supports")
            ),
        )
    first = replace(
        first,
        entities=tuple(sorted(entities, key=lambda r: r.entity_key)),
        automatic_memberships=tuple(sorted(members, key=lambda r: r.entity_key)),
        relations=tuple(sorted(relations, key=lambda r: r.relation_key)),
        evidence=tuple(sorted(evidence, key=lambda r: r.evidence_key)),
        relation_semantics=semantics,
        counts=replace(
            first.counts,
            entity_count=202,
            automatic_membership_count=202,
            relation_count=201,
            evidence_count=201,
            relation_semantics_count=len(semantics),
        ),
    )
    return (first, second), key("hub-identity:199")
