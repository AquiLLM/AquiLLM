"""Bounded-oracle and paired measurement logic work without CUDA/vLLM."""
import importlib.util
from pathlib import Path

import pytest
import torch


def module():
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "deployed_prefill.py"
    spec = importlib.util.spec_from_file_location("deployed_prefill_benchmark", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_bounded_reference_preserves_causal_prefix_across_query_and_key_tiles():
    bench = module()
    q = torch.zeros(3, 2, 2)
    k = torch.zeros(5, 1, 2)
    v = torch.arange(1., 6.).reshape(5, 1, 1).expand(-1, 1, 2)
    ranges = []
    def kv(start, end):
        ranges.append((start, end))
        return k[start:end], v[start:end]
    result = bench.bounded_reference(q, kv, 5, causal_prefix=2, scale=1., query_tile=1, key_tile=2)
    torch.testing.assert_close(result, torch.tensor([2., 2.5, 3.]).reshape(3, 1, 1).expand(3, 2, 2))
    assert max(end-start for start,end in ranges) <= 2


def test_bounded_reference_is_stable_for_extreme_logits_and_matches_dense_reference():
    bench = module()
    q = torch.tensor([[[1000., 1.]], [[-1000., 1.]]])
    k = torch.tensor([[[1., 1.]], [[1.001, 1.]], [[-1., 2.]], [[1., 2.]]])
    v = torch.tensor([[[1., 2.]], [[3., 4.]], [[5., 6.]], [[7., 8.]]])
    scores = torch.einsum("qhd,khd->qhk", q, k)
    scores[0, :, 3] = -torch.inf
    expected = torch.einsum("qhk,khd->qhd", scores.softmax(-1), v)
    actual = bench.bounded_reference(q, lambda a,b: (k[a:b], v[a:b]), 4, causal_prefix=2,
                                     scale=1., query_tile=1, key_tile=2)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_current_chunk_slot_mapping_uses_logical_request_pages():
    bench = module()
    table = torch.tensor([3, 1, 4], dtype=torch.int32)
    actual = bench.current_chunk_slots(table, block_size=4, prior=3, length=5)
    torch.testing.assert_close(actual, torch.tensor([15, 4, 5, 6, 7]))


def test_paired_measurement_randomizes_round_order_and_balances_samples():
    bench = module()
    calls = []
    def baseline():
        calls.append("baseline")
        return 17.
    def candidate():
        calls.append("candidate")
        return 23.
    result = bench.paired_measurements({"baseline": baseline, "candidate": candidate},
                                      lambda operation: {"wall_ms": operation()}, repetitions=20, seed=5)
    assert result["baseline"]["samples"] == [{"wall_ms": 17.}] * 20
    assert result["candidate"]["samples"] == [{"wall_ms": 23.}] * 20
    assert len(calls) == 40
    assert set(calls[::2]) == {"baseline", "candidate"}
    assert all(set(calls[i:i+2]) == {"baseline", "candidate"} for i in range(0, 40, 2))


def test_numerical_gate_rejects_nonfinite_and_excessive_error():
    bench = module()
    expected = torch.ones(1, 1, 2)
    assert bench.numerical_gate(expected, expected, torch.float16)["max_normalized_error"] == 0.
    with pytest.raises(AssertionError):
        bench.numerical_gate(torch.full_like(expected, torch.nan), expected, torch.float16)
    with pytest.raises(AssertionError):
        bench.numerical_gate(expected + 0.1, expected, torch.float16)
