"""Catch inclusive-boundary errors, unvalidated tail guesses, and overlapping partitions."""
import importlib
import importlib.util

import pytest

from aquillm_vllm_h100.contracts import SplitPlan


def policy():
    assert importlib.util.find_spec("aquillm_vllm_h100.split_policy") is not None, "split policy must exist"
    return importlib.import_module("aquillm_vllm_h100.split_policy")


@pytest.mark.parametrize("prior,want", [(0, 7), (1, 7), (2048, 7), (2049, 15), (8192, 15), (8193, 31), (131072, 31), (131073, 31), (10**9, 31)])
def test_inclusive_buckets_and_exhausted_table(prior, want):
    plan = SplitPlan(63, ((2048, 7), (8192, 15), (131072, 31)))
    assert policy().select_active_splits(prior, plan) == want


@pytest.mark.parametrize("prior", [0, 1, 2048, 10**9])
def test_empty_table_uses_fixed_maximum(prior):
    assert policy().select_active_splits(prior, SplitPlan(15, ())) == 15


@pytest.mark.parametrize("prior", [-1, True, 1.5, None, "2048"])
def test_invalid_context_rejected(prior):
    with pytest.raises(ValueError):
        policy().select_active_splits(prior, SplitPlan(15, ()))


@pytest.mark.parametrize("maximum,buckets", [(0, ()), (-1, ()), (True, ()), (7, ((2048, 15),)), (15, ((32, 0),))])
def test_maximum_split_validation(maximum, buckets):
    with pytest.raises(ValueError):
        SplitPlan(maximum, buckets)


@pytest.mark.parametrize("prior", [0, 1, 7, 15, 16, 17, 31, 32, 33, 100, 2048, 8193])
@pytest.mark.parametrize("splits", [1, 7, 15, 31, 47, 63])
def test_committed_partition_covers_exactly_once_and_raw_is_disjoint(prior, splits):
    intervals = [policy().committed_interval(prior, splits, sid) for sid in range(splits)]
    assert intervals[0][0] == 0
    assert intervals[-1][1] == prior
    assert all(0 <= start <= end <= prior for start, end in intervals)
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
    assert sum(end - start for start, end in intervals) == prior
    covered = [position for start, end in intervals for position in range(start, end)]
    assert covered == list(range(prior))
    raw = (prior, prior + 5)
    assert raw == (intervals[-1][1], prior + 5)
    assert set(covered).isdisjoint(range(*raw))


@pytest.mark.parametrize("prior,splits,sid", [(-1, 7, 0), (1, 0, 0), (1, 7, -1), (1, 7, 7), (True, 7, 0)])
def test_invalid_partition_request_rejected(prior, splits, sid):
    with pytest.raises(ValueError):
        policy().committed_interval(prior, splits, sid)
