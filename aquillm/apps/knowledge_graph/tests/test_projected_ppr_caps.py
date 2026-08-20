from dataclasses import replace

import pytest

from apps.knowledge_graph.retrieval import projected_types as types
from apps.knowledge_graph.retrieval.projected_ppr import ppr_projected_v1
from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
from apps.knowledge_graph.tests.test_projected_ppr import _key, _projected_snapshot


def test_projected_ppr_replays_directional_edge_and_opaque_fanout_caps() -> None:
    directional, config = _projected_snapshot(edges=(("a", "b"),))
    forward = directional.relation_groups[0]
    reverse = replace(
        forward,
        source_identity_key=forward.target_identity_key,
        target_identity_key=forward.source_identity_key,
        direction=types.ProjectedRetrievalDirectionV1.REVERSE_DIRECTED,
    )
    directional = replace(
        directional,
        caps=replace(directional.caps, max_edges=1),
        relation_groups=tuple(sorted((forward, reverse), key=types._group_key)),
    )
    capped = ppr_projected_v1(
        snapshot=directional,
        seeds=(ProjectedSeedV1(_key("a"), 1.0),),
        config=replace(config, max_edges=1),
    )
    assert dict(capped.scores) == {
        _key("a"): pytest.approx(0.84),
        _key("b"): pytest.approx(0.16),
        _key("c"): 0.0,
    }

    fanout, config = _projected_snapshot(edges=(("a", "b"), ("a", "c")))
    fanout = replace(fanout, caps=replace(fanout.caps, max_edges=200))
    selected = min((_key("b"), _key("c")))
    result = ppr_projected_v1(
        snapshot=fanout,
        seeds=(ProjectedSeedV1(_key("a"), 1.0),),
        config=replace(config, max_fanout=1, max_edges=200),
    )
    scores = dict(result.scores)
    assert scores[selected] == pytest.approx(0.16)
    assert scores[({_key("b"), _key("c")} - {selected}).pop()] == 0.0
