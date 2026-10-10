"""Catch missing validation, scratch undercounting, and biased sweep ordering."""
import importlib.util
from pathlib import Path

import pytest


def benchmark():
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "splitk.py"
    assert path.exists(), "fixed-split benchmark must exist"
    spec = importlib.util.spec_from_file_location("splitk_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scratch_includes_padded_queries_heads_and_raw_slot():
    bench = benchmark()
    assert bench.scratch_shape(1, 4, 5, 6, 256, 15) == (1, 4, 16, 8, 8, 257)
    assert bench.scratch_bytes(1, 4, 5, 6, 256, 15) == 4_210_688
    assert bench.scratch_bytes(1, 4, 5, 6, 256, 31) == 8_421_376


@pytest.mark.parametrize("value", ["0", "-1", "7,7", "", "7,no", "7,0"])
def test_invalid_split_candidates_rejected(value):
    with pytest.raises(ValueError):
        benchmark().parse_split_counts(value)


def test_randomized_order_visits_every_candidate_each_round():
    bench = benchmark()
    candidates = (7, 15, 31, 47, 63)
    orders = bench.candidate_orders(candidates, rounds=8, seed=179)
    assert len(orders) == 8
    assert all(sorted(order) == list(candidates) for order in orders)
    assert len({tuple(order) for order in orders}) > 1
    assert orders == bench.candidate_orders(candidates, rounds=8, seed=179)
