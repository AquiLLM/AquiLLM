"""Joint graph output has one vote and cannot mix with legacy graph votes."""

from dataclasses import replace

import pytest

from apps.documents.services.candidate_rank_fusion import (
    ChannelRankV2,
    FusedCandidateV2,
    FusionResultV2,
    checksum,
    fuse_ranked_channels_v2,
)


def test_graph_v2_has_one_original_rank_vote():
    result = fuse_ranked_channels_v2(
        {"graph_v2": ((3, 2),)}, limit=20, scope_key="0" * 64
    )
    assert result.candidates[0].channel_ranks == (ChannelRankV2("graph_v2", 2),)
    assert result.candidates[0].rrf_score == 1 / 62


@pytest.mark.parametrize("legacy", ["graph_direct", "graph_extended"])
def test_joint_channel_rejects_legacy_votes_even_on_other_rows(legacy):
    with pytest.raises(ValueError):
        fuse_ranked_channels_v2(
            {"graph_v2": ((3, 1),), legacy: ((4, 1),)}, limit=20, scope_key="0" * 64
        )
    rows = (
        FusedCandidateV2(3, (ChannelRankV2("graph_v2", 1),), 1 / 61, False),
        FusedCandidateV2(4, (ChannelRankV2(legacy, 1),), 1 / 61, False),
    )
    with pytest.raises(ValueError):
        FusionResultV2(rows, "rrf60-v2", "0" * 64, checksum(rows, "rrf60-v2", "0" * 64))


def test_absent_joint_channel_preserves_legacy_result_bytes():
    channels = {"dense": ((1, 1),), "graph_direct": ((2, 1),)}
    result = fuse_ranked_channels_v2(channels, limit=20, scope_key="0" * 64)
    empty = fuse_ranked_channels_v2(
        {**channels, "graph_v2": ()}, limit=20, scope_key="0" * 64
    )
    assert result == empty
    assert result.order_checksum == checksum(
        result.candidates, result.policy_key, result.scope_key
    )
    with pytest.raises(ValueError):
        replace(result.candidates[0], channel_ranks=(ChannelRankV2("graph_v2", 21),))


def test_rrf_off_pool_cap_preserves_entire_original_baseline_and_exact_rows(
    monkeypatch,
):
    from types import SimpleNamespace

    from apps.documents.services import chunk_search_rank_policy as policy
    from apps.documents.services.hybrid_graph_v2 import bounded_pool_v2

    monkeypatch.setattr(
        policy, "controls", lambda: {"rrf": False, "protected_exact": True}
    )
    baseline = tuple(SimpleNamespace(pk=i) for i in range(1, 627))
    graph = tuple(SimpleNamespace(pk=i) for i in range(627, 647))
    ordered, fusion = policy.ordered_candidates(
        baseline + graph,
        {"dense": tuple(r.pk for r in baseline), "exact": (626,)},
        query="exact",
        exact_rows=(baseline[-1],),
        scope=None,
        preserved_ranks={"graph_v2": tuple((r.pk, i) for i, r in enumerate(graph, 1))},
    )
    assert fusion is None
    assert bounded_pool_v2(ordered, baseline) == baseline
    assert bounded_pool_v2(ordered, baseline)[-1].pk == 626
    with pytest.raises(ValueError):
        bounded_pool_v2(ordered, baseline + graph[:1])
