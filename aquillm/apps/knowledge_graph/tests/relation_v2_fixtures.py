"""Synthetic relation-sensitive snapshots, with closed audit provenance."""

from dataclasses import replace
from importlib import import_module, util

from apps.knowledge_graph.retrieval import projected_types as types
from apps.knowledge_graph.tests.projected_ppr_fixtures import key, projected_snapshot


def api(name):
    module = "apps.knowledge_graph.retrieval." + name
    assert util.find_spec(module) is not None, f"missing V2 implementation: {name}"
    return import_module(module)


def relation_snapshot(extra=()):
    predicates = ("uses_dataset", "contradicts", *extra)
    # The template must itself obey V1 uniqueness before predicates are replaced.
    edges = tuple(("a", f"target-{n}") for n in range(1, len(predicates) + 1))
    template, config = projected_snapshot(edges=edges)
    snapshot, _ = projected_snapshot()
    by_relation = {key(f"relation-{n}"): p for n, p in enumerate(predicates, 1)}
    by_evidence = {key(f"evidence-{n}"): p for n, p in enumerate(predicates, 1)}
    target_by_relation = {
        key(f"relation-{n}"): key("b" if n == 1 else "c")
        for n in range(1, len(predicates) + 1)
    }
    target_by_evidence = {
        key(f"evidence-{n}"): key("b" if n == 1 else "c")
        for n in range(1, len(predicates) + 1)
    }
    groups = [
        replace(
            g,
            relation_type=by_evidence[g.evidence[0].provenance_key],
            target_identity_key=target_by_evidence[g.evidence[0].provenance_key],
        )
        for g in template.relation_groups
    ]
    audits = list(snapshot.audit_rows)
    for row in template.audit_rows:
        if type(row) is types.ProjectedAutomaticMembershipAuditV1:
            continue
        if type(row) is types.ProjectedPhysicalRelationAuditV1:
            row = replace(
                row,
                relation_type=by_relation[row.relation_key],
                target_entity_key=key(f"entity-{target_by_relation[row.relation_key]}"),
            )
        elif type(row) is types.ProjectedRelationEvidenceAuditV1:
            row = replace(
                row,
                signature=replace(
                    row.signature, relation_type=by_relation[row.signature.relation_key]
                ),
            )
        audits.append(row)
    groups.sort(key=types._group_key)
    audits.sort(key=lambda r: (r.discovery_hop, r.kind, types._canonical(r)))
    return replace(
        snapshot, relation_groups=tuple(groups), audit_rows=tuple(audits)
    ), config
