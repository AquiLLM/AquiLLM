"""Exact version vector and captured query/group/lower-cap signature checks."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.projection.frontier_adapter_v2 import (
    Neo4jCanonicalFrontierAdapterV2,
)
from apps.knowledge_graph.projection.frontier_snapshot_v2 import frontier_config
from apps.knowledge_graph.retrieval.frontier_ranking_validation_v2 import (
    validate_frontier_ranking,
)
from apps.knowledge_graph.retrieval.projected_ppr import prepare_projected_ppr_inputs
from apps.knowledge_graph.retrieval.projected_relation_ranking_v2 import (
    rank_relations_v2,
)
from apps.knowledge_graph.retrieval.topology.frontier_loader_v2 import (
    canonical_frontier_loader_v2,
)
from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture
from apps.knowledge_graph.tests.frontier_index_fixture_driver import (
    FrontierIndexFixtureDriver,
)


def loaded():
    f = FrontierFixture()
    loader = canonical_frontier_loader_v2(
        selection=f.selection,
        codec=f.codec,
        adapter=Neo4jCanonicalFrontierAdapterV2(FrontierIndexFixtureDriver(f.bundles)),
    )
    result = loader.load(
        group=f.group,
        seeds=f.seeds,
        request=f.request,
        caps=f.caps,
        budget=f.budget(),
        deadline=f.deadline,
    )
    config = frontier_config(
        f.caps, f.group.scope.ready.selected_generations[0].resolver_version
    )
    return f, result, config, loader.ranking_context(result)


def test_v2_exact_context_ranks_and_legacy_consumer_rejects():
    f, result, config, context = loaded()
    rank_relations_v2(
        snapshot=result.snapshot,
        seeds=f.seeds,
        request=f.request,
        config=config,
        frontier_context=context,
        deadline_check=lambda: None,
    )
    with pytest.raises(ValueError):
        prepare_projected_ppr_inputs(
            snapshot=result.snapshot, seeds=f.seeds, config=config
        )
    with pytest.raises(TypeError):
        rank_relations_v2(
            snapshot=result.snapshot,
            seeds=f.seeds,
            request=f.request,
            config=config,
            deadline_check=lambda: None,
        )
    with pytest.raises(ValueError):
        replace(result.snapshot.algorithm, transition_version="ppr_transition_v1")


@pytest.mark.parametrize("changed", ["query", "weights", "group", "caps", "policy"])
def test_wrong_signature_inputs_rejected(changed):
    f, result, config, context = loaded()
    request = f.request
    if changed == "query":
        request = replace(request, query_digest="a" * 64)
    elif changed == "weights":
        request = replace(request, predicate_weights=())
    elif changed == "group":
        context = replace(context, group_checksum="b" * 64)
    elif changed == "caps":
        context = replace(context, max_queries=63)
    else:
        with pytest.raises(ValueError):
            replace(context, acquisition_policy="arbitrary-window")
        return
    with pytest.raises(ValueError):
        validate_frontier_ranking(
            snapshot=result.snapshot,
            seeds=f.seeds,
            request=request,
            config=config,
            context=context,
        )


def test_empty_v2_requires_real_full_scope_provenance():
    f, result, config, context = loaded()
    empty = replace(
        result.snapshot,
        identity_keys=(),
        seed_identities=(),
        relation_groups=(),
        mentions=(),
        audit_rows=(),
    )
    assert empty.artifact_provenance == result.snapshot.artifact_provenance
    with pytest.raises(ValueError):
        replace(empty, artifact_provenance=())
    with pytest.raises(ValueError):
        replace(empty, mentions=result.snapshot.mentions)


def test_non_snapshot_keeps_original_v1_type_error():
    f, _, config, _ = loaded()
    with pytest.raises(TypeError, match="snapshot must be an exact projected snapshot"):
        rank_relations_v2(
            snapshot=None,
            seeds=f.seeds,
            request=f.request,
            config=config,
            deadline_check=lambda: None,
        )
