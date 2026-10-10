"""Catch compact-row aliasing, speculative cache reads and stale graph scratch."""
import importlib

import pytest


def test_fused_launcher_available():
    # This assertion intentionally runs on CPU too: no CUDA import is required
    # to expose the caller-buffer entry point.
    try:
        module = importlib.import_module("aquillm_vllm_h100.kernels.mtp_fused")
    except ModuleNotFoundError:
        module = None
    assert module is not None, "fused stage-one implementation is missing"
    assert callable(module.launch_fused_stage1)


@pytest.fixture
def runtime():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required; skipped GPU tests are not validation")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("H100 SM90 is required; skipped GPU tests are not validation")
    from aquillm_vllm_h100.contracts import SplitPlan
    from aquillm_vllm_h100.kernels.mtp_fused import launch_fused_stage1
    from aquillm_vllm_h100.kernels.reduce import reduce_verify_partials
    from reference import assert_close, make_verify_batch, reference_verify
    return (torch, SplitPlan, launch_fused_stage1, reduce_verify_partials,
            assert_close, make_verify_batch, reference_verify)


def _buffers(torch, batch, plan):
    b, length, hq, dim = batch.q.shape
    g = hq // batch.spec.num_kv_heads
    qpad, gpad = 1 << (length - 1).bit_length(), 1 << (g - 1).bit_length()
    mid = torch.full((b, batch.spec.num_kv_heads, plan.max_splits + 1,
                      qpad, gpad, dim + 1), float("nan"),
                     device=batch.q.device, dtype=torch.float32)
    return mid, torch.empty_like(batch.q)


@pytest.mark.gpu
@pytest.mark.parametrize("length,prior,strided", [
    (2, [0, 1, 37], False), (4, [2, 65], True),
    (5, [0, 3, 97], False), (5, [1, 131], True),
    (6, [0, 67, 259], True), (5, [2049], False),
])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_fused_matches_oracle_and_initializes_empty_lanes(runtime, length, prior,
                                                         strided, dtype_name):
    torch, Plan, launch, reduce, close, make, oracle = runtime
    batch = make(prior, length=length, dtype=getattr(torch, dtype_name),
                 strided=strided)
    plan = Plan(7, ((32, 1), (128, 3), (4096, 7)))
    mid, output = _buffers(torch, batch, plan)
    launch(batch, plan, mid)
    reduce(batch, mid, output)
    close(output, oracle(batch), batch.q.dtype)
    # Every scratch lane has a deterministic neutral value when not consumed.
    assert not torch.isnan(mid).any()
    g = batch.spec.num_q_heads // batch.spec.num_kv_heads
    for padded in (mid[:, :, :, length:, :, :], mid[:, :, :, :, g:, :]):
        assert torch.count_nonzero(padded[..., :-1]) == 0
        assert torch.isneginf(padded[..., -1]).all()
    for b, committed in enumerate(prior):
        active = next((splits for upper, splits in plan.buckets
                       if committed <= upper), plan.max_splits)
        empty = mid[b, :, active:plan.max_splits]
        assert torch.count_nonzero(empty[..., :-1]) == 0
        assert torch.isneginf(empty[..., -1]).all()


@pytest.mark.gpu
def test_future_tail_and_cached_speculation_cannot_change_earlier_queries(runtime):
    torch, Plan, launch, reduce, close, make, oracle = runtime
    batch = make([37, 129], length=5, strided=True)
    plan = Plan(7, ())
    mid, output = _buffers(torch, batch, plan)
    launch(batch, plan, mid)
    reduce(batch, mid, output)
    original = output.clone()
    for b, committed in enumerate([37, 129]):
        for pos in range(committed, committed + 5):
            page = batch.block_table[b, pos // batch.spec.block_size].item()
            batch.kv_cache[page, pos % batch.spec.block_size].fill_(255)
    launch(batch, plan, mid)
    reduce(batch, mid, output)
    torch.testing.assert_close(output, original, rtol=0, atol=0)
    batch.raw_k[:, 3:].fill_(20)
    batch.raw_v[:, 3:].fill_(-20)
    launch(batch, plan, mid)
    reduce(batch, mid, output)
    torch.testing.assert_close(output[:, :3], original[:, :3], rtol=0, atol=0)
    close(output, oracle(batch), batch.q.dtype)


@pytest.mark.gpu
def test_captured_replay_overwrites_long_to_short_split_state(runtime):
    torch, Plan, launch, reduce, close, make, oracle = runtime
    batch = make([2049, 97], length=5)
    plan = Plan(7, ((32, 1), (128, 3), (4096, 7)))
    mid, output = _buffers(torch, batch, plan)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch(batch, plan, mid)
            reduce(batch, mid, output)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(batch, plan, mid)
        reduce(batch, mid, output)
    for prior in ([2049, 97], [0, 3], [31, 65], [2049, 97], [1, 0]):
        batch.seq_lens.copy_(torch.tensor([p + 5 for p in prior],
                                         device="cuda", dtype=batch.seq_lens.dtype))
        mid.fill_(float("nan"))
        for _ in range(3):
            graph.replay()
        close(output, oracle(batch), batch.q.dtype)
        assert not torch.isnan(mid).any()
