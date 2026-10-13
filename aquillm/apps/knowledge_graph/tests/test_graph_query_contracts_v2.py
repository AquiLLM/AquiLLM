"""Exact private topology inputs keep ontology and authority bindings distinct."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.tests.relation_v2_fixtures import api, key, relation_snapshot


def topology():
    contracts = api("graph_query_contracts")
    snapshot, _ = relation_snapshot()
    evidence = contracts.GraphEvidenceRefV2(
        key("chunk"),
        key("generation"),
        None,
        key("group"),
        tuple(sorted((key("a"), key("b")))),
        (key("evidence"),),
        ("supports",),
        0.9,
        (
            contracts.GraphEvidenceAssociationV2(
                key("evidence"), "relation", key("a"), key("b"), "supports", "directed"
            ),
        ),
    )
    coverage = api("graph_handoff").GraphCoverageV2(1, 1, "complete", (), False)
    return contracts.BoundedGraphTopologyV2(
        snapshot, (evidence,), coverage, key("request"), key("group")
    )


def test_topology_refuses_cross_group_or_outside_snapshot_evidence():
    value = topology()
    with pytest.raises(ValueError):
        replace(value, authority_group_checksum=key("other-group"))
    with pytest.raises(ValueError):
        replace(
            value,
            evidence_candidates=(
                replace(
                    value.evidence_candidates[0], identity_keys=(key("not-admitted"),)
                ),
            ),
        )
    with pytest.raises(ValueError):
        replace(value, evidence_candidates=value.evidence_candidates * 2)


def test_evidence_and_group_requests_refuse_coercion_or_noncanonical_references():
    value = topology()
    with pytest.raises(TypeError):
        replace(value.evidence_candidates[0], representation_generation_key="123")
    with pytest.raises(ValueError):
        replace(
            value.evidence_candidates[0], relation_types=("supports", "contradicts")
        )
    contracts = api("graph_query_contracts")
    request = api("relation_request").parse_relation_request_v2(
        "Atlas",
        ontology_checksum=key("ontology"),
        allowed_relations=frozenset({"supports"}),
    )
    assert (
        contracts.GraphGroupRelationRequestV2(key("group"), request).request is request
    )
    with pytest.raises(TypeError):
        contracts.GraphGroupRelationRequestV2(
            key("group"), {"query_digest": key("query")}
        )
