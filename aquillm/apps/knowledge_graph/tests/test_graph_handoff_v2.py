"""Private V2 handoff admission, coverage and canonical identity tests."""

from dataclasses import replace
from uuid import UUID

import pytest

from apps.knowledge_graph.tests.relation_v2_fixtures import api, key


def handoff():
    module = api("graph_handoff")
    coverage = module.GraphCoverageV2(2, 2, "complete", (), False)
    candidate = module.GraphCandidateRefV2(
        key("chunk"),
        key("generation"),
        UUID("11111111-1111-4111-8111-111111111111"),
        key("group"),
        key("bundle"),
        (key("evidence"),),
        ("supports",),
        0.5,
    )
    return module.GraphHandoffV2((candidate,), coverage, key("algorithm"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"exhaustive": True},
        {"selected_documents": True},
        {"ready_documents": 3},
        {"status": "complete", "ready_documents": 1},
        {"status": "unavailable"},
        {"reasons": ("invented",)},
        {"reasons": ("deadline", "deadline")},
    ],
)
def test_coverage_refuses_impossible_or_exhaustive_claims(overrides):
    with pytest.raises((TypeError, ValueError)):
        replace(handoff().coverage, **overrides)


def test_unavailable_cannot_carry_candidates():
    h = handoff()
    coverage = api("graph_handoff").GraphCoverageV2(
        2, 0, "unavailable", ("not_ready",), False
    )
    with pytest.raises(ValueError):
        replace(h, coverage=coverage)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, True, 1])
def test_handoff_refuses_nonfinite_negative_or_coerced_scores(score):
    with pytest.raises((TypeError, ValueError)):
        replace(handoff().candidates[0], score=score)


def test_candidate_count_order_uniqueness_and_operator_cap_are_enforced():
    h = handoff()
    with pytest.raises(ValueError):
        replace(h, candidates=h.candidates * 2)
    higher = replace(h.candidates[0], chunk_key=key("other"), score=0.8)
    with pytest.raises(ValueError):
        replace(h, candidates=(*h.candidates, higher))
    ordered = replace(h, candidates=(higher, *h.candidates))
    with pytest.raises(ValueError):
        api("graph_handoff").validate_graph_handoff_v2(ordered, maximum=1)


def test_signature_binds_refs_coverage_and_nullable_exact_uuid():
    h = handoff()
    signature = api("graph_handoff").graph_handoff_signature
    assert signature(h) != signature(
        replace(
            h,
            candidates=(replace(h.candidates[0], representation_generation_key=None),),
        )
    )
    partial = replace(h.coverage, status="partial", reasons=("frontier_budget",))
    assert signature(h) != signature(replace(h, coverage=partial))
    assert signature(h) != signature(replace(h, algorithm_signature=key("other")))
    with pytest.raises(TypeError):
        replace(
            h.candidates[0],
            representation_generation_key=str(
                h.candidates[0].representation_generation_key
            ),
        )


def test_client_provided_opaque_keys_are_not_admitted_as_private_dtos():
    with pytest.raises(TypeError):
        api("graph_handoff").validate_graph_handoff_v2({"chunk_key": key("chunk")})


def test_default_legacy_and_explicit_v2_dispatch_preserve_treatment_isolation():
    from lib.knowledge_graph.retrieval_config import load_hybrid_retrieval_settings

    assert load_hybrid_retrieval_settings({}).graph_remediation_mode == "legacy"
    assert (
        load_hybrid_retrieval_settings(
            {"KG_GRAPH_REMEDIATION_MODE": "v2"}
        ).graph_remediation_mode
        == "v2"
    )
    with pytest.raises(ValueError):
        load_hybrid_retrieval_settings({"KG_GRAPH_REMEDIATION_MODE": "invalid"})


def test_explicit_v2_dispatch_changes_identity_and_refuses_adaptive_or_shadow():
    from apps.knowledge_graph.retrieval.production_ppr_policy import (
        rank_projected_for_mode,
    )
    from apps.knowledge_graph.retrieval.topology.contracts import (
        HybridBranchKind,
        ProjectedSeedV1,
    )
    from apps.knowledge_graph.tests.relation_v2_fixtures import relation_snapshot

    snapshot, config = relation_snapshot()
    request = api("relation_request").parse_relation_request_v2(
        "Which dataset does Atlas use?",
        ontology_checksum=key("ontology"),
        allowed_relations=frozenset({"uses_dataset", "contradicts"}),
    )
    arguments = dict(
        snapshot=snapshot,
        config=config,
        seeds=(ProjectedSeedV1(key("a"), 1.0),),
        prepared=None,
        mode="fixed",
        branch=HybridBranchKind.DIRECT,
        deadline_check=lambda: None,
    )
    legacy, old_signature = rank_projected_for_mode(**arguments)
    ranking, signature = rank_projected_for_mode(
        **arguments, remediation_mode="v2", relation_request=request
    )
    assert ranking != legacy and old_signature is None
    assert signature != snapshot.algorithm.algorithm_signature
    with pytest.raises(ValueError):
        rank_projected_for_mode(
            **(arguments | {"mode": "adaptive"}),
            remediation_mode="v2",
            relation_request=request,
        )
    with pytest.raises(ValueError):
        rank_projected_for_mode(
            **arguments, remediation_mode="shadow", relation_request=request
        )
    with pytest.raises(ValueError):
        rank_projected_for_mode(**arguments, relation_request=request)


def test_cache_identity_binds_request_and_algorithm():
    request = api("relation_request").parse_relation_request_v2(
        "What supports Atlas?",
        ontology_checksum=key("ontology"),
        allowed_relations=frozenset({"supports", "contradicts"}),
    )
    cache = api("production_runtime_support").relation_cache_identity_v2
    arguments = dict(
        snapshot_checksum=key("snapshot"),
        seed_checksum=key("seeds"),
        algorithm_signature=key("algorithm"),
        request=request,
    )
    assert cache(**arguments) != cache(
        **(arguments | {"request": replace(request, query_digest=key("other-query"))})
    )
    assert cache(**arguments) != cache(
        **(arguments | {"algorithm_signature": key("other")})
    )
