"""Hand-checked relation ranking counterexamples with unchanged V1 baselines."""

from dataclasses import replace
from hashlib import sha256

import pytest

from apps.knowledge_graph.retrieval import projected_types as types
from apps.knowledge_graph.retrieval.ppr import canonical_algorithm_json
from apps.knowledge_graph.retrieval.projected_ppr import ppr_projected_v1
from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
from apps.knowledge_graph.tests.relation_v2_fixtures import (
    api,
    key,
    projected_snapshot,
    relation_snapshot,
)


def rank(query, extra=(), **kwargs):
    snapshot, config = relation_snapshot(extra)
    request = api("relation_request").parse_relation_request_v2(
        query,
        ontology_checksum=key("ontology"),
        allowed_relations=frozenset(
            {"uses_dataset", "contradicts", "supports", *extra}
        ),
    )
    return api("projected_relation_ranking_v2").rank_relations_v2(
        snapshot=snapshot,
        seeds=(ProjectedSeedV1(key("a"), 1.0),),
        request=request,
        config=config,
        deadline_check=kwargs.get("deadline_check", lambda: None),
    )


def test_same_seeds_requested_relation_changes_destination_order():
    dataset = dict(rank("Which dataset does Atlas use?").scores)
    contradiction = dict(rank("What contradicts Atlas?").scores)
    assert dataset[key("b")] == pytest.approx(0.13333333333333333)
    assert dataset[key("c")] == pytest.approx(0.02666666666666667)
    assert contradiction[key("c")] > contradiction[key("b")]


def test_irrelevant_predicate_multiplicity_cannot_inflate_endpoint_credit():
    baseline = rank("Which dataset does Atlas use?")
    crowded = rank(
        "Which dataset does Atlas use?", ("supports", "related_to", "derived_from")
    )
    assert crowded == baseline


@pytest.mark.parametrize("beyond_cap", [0, 1, 2, 6])
def test_same_pair_predicate_multiplicity_at_real_fanout_cap(beyond_cap):
    _, config = relation_snapshot()
    extra = tuple(f"aaa_{n}" for n in range(config.max_fanout - 2 + beyond_cap))
    snapshot, _ = relation_snapshot(extra)
    assert len(snapshot.relation_groups) == config.max_fanout + beyond_cap
    baseline = rank("Which dataset does Atlas use?")
    crowded = rank("Which dataset does Atlas use?", extra)
    assert crowded == baseline
    assert dict(crowded.scores)[key("b")] > dict(crowded.scores)[key("c")]


def test_v2_still_caps_distinct_destination_fanout():
    destinations = tuple(f"destination-{n}" for n in range(11))
    snapshot, config = projected_snapshot(edges=tuple(("a", d) for d in destinations))
    request = api("relation_request").parse_relation_request_v2(
        "Atlas",
        ontology_checksum=key("ontology"),
        allowed_relations=frozenset({"related_to"}),
    )
    seeds = (ProjectedSeedV1(key("a"), 1.0),)
    ranking = api("projected_relation_ranking_v2").rank_relations_v2(
        snapshot=snapshot,
        seeds=seeds,
        request=request,
        config=config,
        deadline_check=lambda: None,
    )
    scores = dict(ranking.scores)
    assert sum(scores[key(d)] > 0.0 for d in destinations) == config.max_fanout
    assert scores[max(key(d) for d in destinations)] == 0.0
    assert ranking == ppr_projected_v1(snapshot=snapshot, seeds=seeds, config=config)


def test_v2_keeps_total_edge_cap_with_bidirectional_closed_topology():
    snapshot, config = projected_snapshot(
        edges=tuple(("a", f"d-{n}") for n in range(5))
    )
    config = replace(config, max_edges=5)
    reverse = tuple(
        replace(
            g,
            source_identity_key=g.target_identity_key,
            target_identity_key=g.source_identity_key,
            direction=types.ProjectedRetrievalDirectionV1.REVERSE_DIRECTED,
            raw_weight=0.35 * g.raw_weight,
        )
        for g in snapshot.relation_groups
    )
    snapshot = replace(
        snapshot,
        caps=replace(snapshot.caps, max_edges=5),
        algorithm=replace(
            snapshot.algorithm,
            algorithm_signature=sha256(
                b"ppr_projected_v1\0" + canonical_algorithm_json(config)
            ).hexdigest(),
        ),
        relation_groups=tuple(
            sorted((*snapshot.relation_groups, *reverse), key=types._group_key)
        ),
    )
    seeds = (ProjectedSeedV1(key("a"), 1.0),)
    request = api("relation_request").parse_relation_request_v2(
        "Atlas",
        ontology_checksum=key("ontology"),
        allowed_relations=frozenset({"related_to"}),
    )
    module = api("projected_relation_ranking_v2")
    retained = module._admit_weighted_pairs(
        snapshot, seeds, config, dict(request.predicate_weights), lambda: None
    )
    assert len(snapshot.relation_groups) == 2 * config.max_edges
    assert len(retained) == config.max_edges
    ranking = module.rank_relations_v2(
        snapshot=snapshot,
        seeds=seeds,
        request=request,
        config=config,
        deadline_check=lambda: None,
    )
    assert sum(score for _, score in ranking.scores) == pytest.approx(1.0)


def test_exact_observation_duplicates_do_not_add_credit_and_labels_survive():
    snapshot, _ = relation_snapshot()
    weights = {"uses_dataset": 1.0, "contradicts": 0.2}
    prepare = api("projected_relation_ranking_v2")._weighted_pairs
    assert prepare(snapshot.relation_groups * 2, weights, lambda: None) == prepare(
        snapshot.relation_groups, weights, lambda: None
    )
    assert {g.relation_type for g in snapshot.relation_groups} == {
        "uses_dataset",
        "contradicts",
    }


def test_mixed_ontology_snapshot_is_refused():
    snapshot, config = relation_snapshot()
    request = api("relation_request").parse_relation_request_v2(
        "Atlas",
        ontology_checksum="f" * 64,
        allowed_relations=frozenset({"uses_dataset", "contradicts"}),
    )
    with pytest.raises(ValueError, match="ontology"):
        api("projected_relation_ranking_v2").rank_relations_v2(
            snapshot=snapshot,
            seeds=(ProjectedSeedV1(key("a"), 1.0),),
            request=request,
            config=config,
            deadline_check=lambda: None,
        )
    # A request matching one provenance row must not admit another ontology.
    empty, config = projected_snapshot()
    mixed = replace(
        empty,
        artifact_provenance=tuple(
            replace(row, ontology_checksum="f" * 64)
            if row.scope_type.value == "document"
            else row
            for row in empty.artifact_provenance
        ),
    )
    request = replace(request, ontology_checksum=key("ontology"))
    with pytest.raises(ValueError, match="ontology"):
        api("projected_relation_ranking_v2").rank_relations_v2(
            snapshot=mixed,
            seeds=(ProjectedSeedV1(key("a"), 1.0),),
            request=request,
            config=config,
            deadline_check=lambda: None,
        )


def test_v2_checks_deadline_and_v1_output_stays_fixed():
    snapshot, config = relation_snapshot()
    legacy = ppr_projected_v1(
        snapshot=snapshot, seeds=(ProjectedSeedV1(key("a"), 1.0),), config=config
    )
    assert dict(legacy.scores)[key("b")] == pytest.approx(0.08)
    assert dict(legacy.scores)[key("c")] == pytest.approx(0.08)

    def expired():
        raise TimeoutError("expired")

    with pytest.raises(TimeoutError):
        rank("What supports Atlas?", deadline_check=expired)
    assert (
        ppr_projected_v1(
            snapshot=snapshot, seeds=(ProjectedSeedV1(key("a"), 1.0),), config=config
        )
        == legacy
    )
