"""Narrow installed-dispatch characterization, support and distribution tests.

Opt in only in a serialized qualification process:
  AQUILLM_RUN_FLASHINFER_AUX=1 python -m pytest -q tests/gpu/test_sampler_distribution.py
No identical samples across different RNG backends are required. Input logits
are the sampler boundary; processor/penalty integration needs serving tests.
"""
import importlib
import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu]


@pytest.fixture(scope="module")
def runtime():
    if os.environ.get("AQUILLM_RUN_FLASHINFER_AUX") != "1":
        pytest.skip("serialized pinned-image qualification opt-in required")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module = importlib.import_module("vllm.v1.sample.ops.topk_topp_sampler")
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "sampler.py"
    spec = importlib.util.spec_from_file_location("sampler_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    sampler = module.TopKTopPSampler()
    assert sampler.forward.__name__ == "forward_cuda", "requires actual FlashInfer-enabled dispatcher"
    return torch, module, sampler, benchmark.DispatchTrace


@pytest.mark.parametrize("mode", ["both", "k_only", "p_only", "neither", "seeded", "fp64"])
def test_actual_dispatch_branches(runtime, mode):
    torch, module, sampler, trace_type = runtime
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], device="cuda")
    k = torch.tensor([2], device="cuda", dtype=torch.int32) if mode not in ("p_only", "neither") else None
    p = torch.tensor([0.9], device="cuda") if mode not in ("k_only", "neither") else None
    generators = {0: torch.Generator(device="cuda").manual_seed(7)} if mode == "seeded" else {}
    local = module.TopKTopPSampler(use_fp64_gumbel=mode == "fp64")
    with trace_type(module, local) as trace:
        sampled, _ = local.forward(logits, generators, k, p)
        torch.cuda.synchronize()
    assert sampled.shape == (1,)
    expected = "native" if mode in ("neither", "seeded", "fp64") else "flashinfer"
    assert trace.counts == {"flashinfer": int(expected == "flashinfer"), "native": int(expected == "native")}


@pytest.mark.parametrize("mode", ["processed_logits", "processed_logprobs"])
def test_processed_modes_bind_native_and_return_requested_values(runtime, mode):
    torch, module, _, _ = runtime
    sampler = module.TopKTopPSampler(logprobs_mode=mode)
    assert sampler.forward.__name__ == "forward_native"
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], device="cuda")
    _, processed = sampler.forward(logits, {}, torch.tensor([1], device="cuda"), None)
    assert torch.isneginf(processed[0, :3]).all()
    assert processed[0, 3].item() == (3.0 if mode == "processed_logits" else 0.0)


def test_seeded_fallback_matches_native_with_same_generator_state(runtime):
    torch, module, sampler, trace_type = runtime
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]] * 16, device="cuda")
    k = torch.full((16,), 3, device="cuda", dtype=torch.int32)
    p = torch.full((16,), 0.9, device="cuda")
    generators1 = {i: torch.Generator(device="cuda").manual_seed(100 + i) for i in range(16)}
    generators2 = {i: torch.Generator(device="cuda").manual_seed(100 + i) for i in range(16)}
    with trace_type(module, sampler) as trace:
        actual, _ = sampler.forward(logits.clone(), generators1, k, p)
        torch.cuda.synchronize()
    expected, _ = sampler.forward_native(logits.clone(), generators2, k, p)
    assert trace.counts == {"flashinfer": 0, "native": 1}
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("k_value,p_value", [(1, 1.0), (3, 0.0), (3, 0.8), (3, 1.0)])
def test_fixed_logits_support_and_distribution(runtime, k_value, p_value):
    torch, module, sampler, trace_type = runtime
    rows = 4096
    probabilities = torch.tensor([0.7, 0.2, 0.1], device="cuda")
    logits = probabilities.log().expand(rows, -1).contiguous()
    k = torch.full((rows,), k_value, device="cuda", dtype=torch.int32)
    p = torch.full((rows,), p_value, device="cuda")
    with trace_type(module, sampler) as trace:
        actual, _ = sampler.forward(logits.clone(), {}, k, p)
        torch.cuda.synchronize()
    assert trace.counts == {"flashinfer": 1, "native": 0}
    if k_value == 1 or p_value == 0.0:
        assert (actual == 0).all()
    else:
        want = torch.tensor([7 / 9, 2 / 9, 0.0] if p_value == 0.8 else [0.7, 0.2, 0.1], device="cuda")
        observed = torch.bincount(actual.long(), minlength=3) / rows
        torch.testing.assert_close(observed, want, atol=0.04, rtol=0)


def test_tied_logits_and_mixed_parameters_keep_each_rows_support(runtime):
    torch, module, sampler, trace_type = runtime
    # Row 0 has a tie with k=2. Row 1 simulates a processor banning token 2;
    # row 2 has a penalized token 2 below the top-k threshold.
    logits = torch.tensor([[0.0, 2.0, 2.0], [0.0, 1.0, -float("inf")], [0.0, 3.0, -2.0]], device="cuda")
    logits = logits.repeat(1024, 1)
    k = torch.tensor([2, 1, 1], device="cuda", dtype=torch.int32).repeat(1024)
    p = torch.tensor([1.0, 0.8, 0.0], device="cuda").repeat(1024)
    with trace_type(module, sampler) as trace:
        actual, _ = sampler.forward(logits, {}, k, p)
        torch.cuda.synchronize()
    assert trace.counts == {"flashinfer": 1, "native": 0}
    assert ((actual[::3] == 1) | (actual[::3] == 2)).all()
    assert (actual[1::3] == 1).all() and (actual[2::3] == 1).all()
