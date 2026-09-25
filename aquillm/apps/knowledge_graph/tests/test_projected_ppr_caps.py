from dataclasses import replace
from hashlib import sha256

import pytest

from apps.knowledge_graph.retrieval import projected_types as types
from apps.knowledge_graph.retrieval.ppr import canonical_algorithm_json
from apps.knowledge_graph.retrieval.projected_ppr import ppr_projected_v1
from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
from apps.knowledge_graph.tests.projected_ppr_fixtures import (
    key,
    projected_snapshot,
)


def _signed(snapshot, config):
    signature = sha256(
        b"ppr_projected_v1\0" + canonical_algorithm_json(config)
    ).hexdigest()
    return replace(
        snapshot, algorithm=replace(snapshot.algorithm, algorithm_signature=signature)
    )


def test_projected_ppr_replays_directional_edge_and_opaque_fanout_caps() -> None:
    directional, config = projected_snapshot(edges=(("a", "b"),))
    forward = directional.relation_groups[0]
    reverse = replace(
        forward,
        source_identity_key=forward.target_identity_key,
        target_identity_key=forward.source_identity_key,
        direction=types.ProjectedRetrievalDirectionV1.REVERSE_DIRECTED,
    )
    capped_config = replace(config, max_edges=1)
    directional = _signed(
        replace(
            directional,
            caps=replace(directional.caps, max_edges=1),
            relation_groups=tuple(sorted((forward, reverse), key=types._group_key)),
        ),
        capped_config,
    )
    capped = ppr_projected_v1(
        snapshot=directional,
        seeds=(ProjectedSeedV1(key("a"), 1.0),),
        config=capped_config,
    )
    assert dict(capped.scores) == {
        key("a"): pytest.approx(0.84),
        key("b"): pytest.approx(0.16),
        key("c"): 0.0,
    }

    fanout, config = projected_snapshot(edges=(("a", "b"), ("a", "c")))
    fanout_config = replace(config, max_fanout=1, max_edges=200)
    fanout = _signed(
        replace(fanout, caps=replace(fanout.caps, max_edges=200)), fanout_config
    )
    selected = min((key("b"), key("c")))
    result = ppr_projected_v1(
        snapshot=fanout,
        seeds=(ProjectedSeedV1(key("a"), 1.0),),
        config=fanout_config,
    )
    scores = dict(result.scores)
    assert scores[selected] == pytest.approx(0.16)
    assert scores[({key("b"), key("c")} - {selected}).pop()] == 0.0


def test_projected_ppr_rejects_inexact_algorithm_signature() -> None:
    snapshot, config = projected_snapshot()
    snapshot = replace(
        snapshot,
        algorithm=replace(snapshot.algorithm, algorithm_signature="f" * 64),
    )

    with pytest.raises(ValueError, match="algorithm signature"):
        ppr_projected_v1(
            snapshot=snapshot,
            seeds=(ProjectedSeedV1(key("a"), 1.0),),
            config=config,
        )
