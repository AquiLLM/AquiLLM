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


def _stride_last(torch, tensor):
    shape = list(tensor.shape)
    shape[-1] *= 2
    backing = torch.empty(shape, device=tensor.device, dtype=tensor.dtype)
    view = backing[..., ::2]
    view.copy_(tensor)
    return view


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
        fallback = plan.buckets[-1][1] if plan.buckets else plan.max_splits
        active = next((splits for upper, splits in plan.buckets
                       if committed <= upper), fallback)
        empty = mid[b, :, active:plan.max_splits]
        assert torch.count_nonzero(empty[..., :-1]) == 0
        assert torch.isneginf(empty[..., -1]).all()


@pytest.mark.gpu
@pytest.mark.parametrize("buckets,active", [(((2048, 7), (8192, 15)), 15), ((), 31)])
def test_exhausted_bucket_uses_last_count_and_fixed_plan_uses_max(runtime, buckets, active):
    torch, Plan, launch, reduce, close, make, oracle = runtime
    batch = make([8193], length=5)
    plan = Plan(31, buckets)
    mid, output = _buffers(torch, batch, plan)
    launch(batch, plan, mid)
    reduce(batch, mid, output)
    close(output, oracle(batch), batch.q.dtype)
    # All expected committed splits are nonempty; anything above the selected
    # count must be neutral even though enough tokens exist to populate it.
    g = batch.spec.num_q_heads // batch.spec.num_kv_heads
    consumed = mid[:, :, :active, :5, :g, -1]
    assert torch.isfinite(consumed).all()
    inactive = mid[:, :, active:31]
    assert torch.count_nonzero(inactive[..., :-1]) == 0
    assert torch.isneginf(inactive[..., -1]).all()
    assert torch.isfinite(mid[:, :, 31, :5, :g, -1]).all()


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


@pytest.mark.gpu
def test_all_authoritative_strides_are_honored(runtime):
    torch, Plan, launch, reduce, close, make, oracle = runtime
    batch = make([1, 67], length=5, strided=True)
    for name in ("q", "kv_cache", "block_table", "seq_lens", "raw_k", "raw_v"):
        setattr(batch, name, _stride_last(torch, getattr(batch, name)))
    plan = Plan(7, ())
    mid, output = _buffers(torch, batch, plan)
    mid, output = _stride_last(torch, mid), _stride_last(torch, output)
    launch(batch, plan, mid)
    reduce(batch, mid, output)
    close(output, oracle(batch), batch.q.dtype)
    assert not torch.isnan(mid).any()


@pytest.mark.gpu
@pytest.mark.parametrize("length", [2, 4, 5, 6])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_fused_matches_pinned_single_and_split_baselines(runtime, monkeypatch,
                                                        length, dtype_name):
    torch, Plan, launch, reduce, close, make, oracle = runtime
    from sndr.engines.vllm.kernels_legacy import p67_multi_query_kernel as p67
    # Match the measured 32-token policy and use actual baseline arithmetic.
    # No GENESIS_P67_USE_FUSED shortcut is eligible for the raw-tail baseline.
    for name, value in (("GENESIS_P67_USE_FUSED", "0"),
                        ("GENESIS_P67_DOT_PRECISION", "tf32x3"),
                        ("GENESIS_P67_BLOCK_KV", "32"),
                        ("GENESIS_P67_NUM_WARPS", "8"),
                        ("GENESIS_P67_NUM_STAGES", "2")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(p67, "_CACHED_KERNEL", None)
    monkeypatch.setattr(p67, "_CACHED_STAGE1_SPLITK", None)
    batch = make([3, 137], length=length, dtype=getattr(torch, dtype_name))
    plan = Plan(7, ())
    mid, output = _buffers(torch, batch, plan)
    launch(batch, plan, mid)
    reduce(batch, mid, output)
    args = (batch.q, batch.kv_cache, batch.block_table, batch.seq_lens,
            batch.raw_k, batch.raw_v, batch.scale, batch.spec.block_size,
            batch.spec.key_packed_size, batch.spec.value_data_bytes)
    single = p67.call_p67_attention(*args, use_raw_tail=1)
    split = p67.call_p67_splitk(*args, num_splits=7)
    close(output, single, batch.q.dtype)
    close(output, split, batch.q.dtype)
    close(output, oracle(batch), batch.q.dtype)


@pytest.mark.gpu
def test_bfloat16_reduction_keeps_fp16_intermediate_rounding(runtime):
    torch, Plan, launch, reduce, close, make, oracle = runtime
    batch = make([0], length=2, dtype=torch.bfloat16)
    # Logits [0, 1/512] weight the larger value just above a BF16 midpoint.
    # FP16 first rounds back to that midpoint, whose BF16 tie rounds to 1.
    batch.q.zero_()
    batch.q[..., 0] = 1.0
    batch.raw_k.zero_()
    batch.raw_k[:, 1, :, 0] = 0.03125
    batch.raw_v[:, 0].fill_(1.0)
    batch.raw_v[:, 1].fill_(1.0078125)
    plan = Plan(1, ())
    mid, output = _buffers(torch, batch, plan)
    launch(batch, plan, mid)
    reduce(batch, mid, output)
    expected = torch.tensor(1.00390625, device="cuda").half().bfloat16()
    torch.testing.assert_close(output[:, 1], expected.expand_as(output[:, 1]),
                               rtol=0, atol=0)
    assert (oracle(batch)[:, 1].bfloat16() != output[:, 1]).all()
