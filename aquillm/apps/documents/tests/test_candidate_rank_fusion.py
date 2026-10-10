"""Independent counterexamples for bounded, canonical channel fusion."""

import math
from dataclasses import replace
from itertools import permutations

import pytest


def test_exact_answer_survives_dense_only_fallback():
    from apps.documents.services.candidate_rank_fusion import fuse_channel_ids

    channels = {"dense": tuple(range(1, 13)), "exact": (99,)}
    result = fuse_channel_ids(channels, limit=10, protected_exact_ids=(99,))
    assert result == (99, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert fuse_channel_ids(channels, limit=10) == (1, 99, 2, 3, 4, 5, 6, 7, 8, 9)


def test_completion_permutations_and_duplicates_preserve_ranks_and_checksum():
    from apps.documents.services.candidate_rank_fusion import fuse_channel_ranks_v2

    channels = [("dense", (1, 1, 2)), ("exact", (3, 2)), ("trigram", (2, 4))]
    results = [
        fuse_channel_ranks_v2(dict(p), limit=4, scope_key="a" * 64)
        for p in permutations(channels)
    ]
    assert all(result == results[0] for result in results)
    assert tuple(c.chunk_id for c in results[0].candidates) == (2, 1, 3, 4)
    first = results[0].candidates[0]
    assert [(r.channel, r.rank) for r in first.channel_ranks] == [
        ("dense", 2),
        ("exact", 2),
        ("trigram", 1),
    ]
    assert first.rrf_score == math.fsum([1 / 62, 1 / 62, 1 / 61])
    with pytest.raises(ValueError):
        replace(results[0], candidates=tuple(reversed(results[0].candidates)))


@pytest.mark.parametrize(
    "channels,limit,protected",
    [
        ({"dense": (True,)}, 1, ()),
        ({"unknown": (1,)}, 1, ()),
        ({"dense": (0,)}, 1, ()),
        ({"dense": (1,)}, True, ()),
        ({"exact": (1, 2)}, 2, (1, 2)),
        ({"dense": (1,)}, 1, (1,)),
        ({"dense": (1,)}, 627, ()),
        ({"dense": [1]}, 1, ()),
    ],
)
def test_malformed_or_unbounded_fusion_is_rejected(channels, limit, protected):
    from apps.documents.services.candidate_rank_fusion import fuse_channel_ids

    with pytest.raises((ValueError, TypeError)):
        fuse_channel_ids(channels, limit=limit, protected_exact_ids=protected)


def test_empty_channels_and_zero_limit():
    from apps.documents.services.candidate_rank_fusion import fuse_channel_ids

    assert fuse_channel_ids({}, limit=10) == ()
    assert fuse_channel_ids({"exact": (1,)}, limit=0) == ()
