"""Capture once and mutate device lengths; stale partials must never survive replay."""
import pytest

torch = pytest.importorskip("torch")
H100 = torch.cuda.is_available() and "H100" in torch.cuda.get_device_name()
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not H100, reason="requires reserved pinned H100 runtime")]


@pytest.mark.parametrize("length", [2, 5, 6])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_capture_reuses_pointers_and_overwrites_every_inactive_lane(length, dtype, batch_size, monkeypatch):
    from aquillm_vllm_h100.contracts import SplitPlan
    from aquillm_vllm_h100.kernels.reduce import reduce_verify_partials
    from aquillm_vllm_h100.kernels.splitk_reference import launch_reference_stage1
    from aquillm_vllm_h100.split_policy import committed_interval, select_active_splits
    from reference import assert_close, make_verify_batch, reference_verify
    from sndr.engines.vllm.kernels_legacy import p67_multi_query_kernel as baseline

    monkeypatch.setenv("GENESIS_P67_BLOCK_KV", "32")
    monkeypatch.setenv("GENESIS_P67_DOT_PRECISION", "tf32x3")
    # Synthetic boundaries exercise policy mechanics; these are not measured profiles.
    plan = SplitPlan(31, ((16, 7), (32, 15), (2048, 31)))
    batch = make_verify_batch([2049] * batch_size, length=length, dtype=dtype, strided=True)
    qpad = 1 << (length - 1).bit_length()
    mid = torch.empty((batch_size, 4, 32, qpad, 8, 257), device="cuda", dtype=torch.float32)
    output = torch.empty_like(batch.q)
    eager = torch.empty_like(batch.q)
    baseline_output = torch.empty_like(batch.q)
    baseline_mid = torch.empty_like(mid)
    pointers = tuple(t.data_ptr() for t in (batch.q, batch.kv_cache, batch.block_table, batch.seq_lens,
                                           batch.raw_k, batch.raw_v, mid, output))

    def launch(destination):
        launch_reference_stage1(batch, plan, mid)
        reduce_verify_partials(batch, mid, destination)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            launch(output)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(output)
    sequence = [0, 1, 7, 15, 16, 17, 31, 32, 33, 2048, 2049, 0, 2049, 1, 2048, 7]
    lengths = [torch.tensor([sequence[(position + i) % len(sequence)] + length for i in range(batch_size)],
                            device="cuda", dtype=torch.int32) for position in range(len(sequence))]
    for position, seq in enumerate(lengths):
        priors = [sequence[(position + i) % len(sequence)] for i in range(batch_size)]
        batch.seq_lens.copy_(seq)
        launch(eager)
        mid.fill_(float("nan"))
        before = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == before
        assert pointers == tuple(t.data_ptr() for t in (batch.q, batch.kv_cache, batch.block_table,
                                                       batch.seq_lens, batch.raw_k, batch.raw_v, mid, output))
        assert_close(output, reference_verify(batch), dtype)
        torch.testing.assert_close(output, eager, atol=0, rtol=0)
        baseline.call_p67_splitk(batch.q, batch.kv_cache, batch.block_table, batch.seq_lens,
                                batch.raw_k, batch.raw_v, batch.scale, batch.spec.block_size,
                                batch.spec.key_packed_size, batch.spec.value_data_bytes,
                                output=baseline_output, num_splits=31, mid_o=baseline_mid)
        assert_close(output, baseline_output, dtype)
        for index, prior in enumerate(priors):
            active = select_active_splits(prior, plan)
            for sid in range(plan.max_splits):
                empty = sid >= active or committed_interval(prior, active, sid)[0] == committed_interval(prior, active, sid)[1]
                if empty:
                    assert (mid[index, :, sid, ..., :256] == 0).all()
                    assert torch.isneginf(mid[index, :, sid, ..., 256]).all()
        # Padding is neutral too, so poison cannot leak into any later consumer.
        assert (mid[..., length:, :, :256] == 0).all()
        assert torch.isneginf(mid[..., length:, :, 256]).all()
        assert (mid[..., 6:, :256] == 0).all()
        assert torch.isneginf(mid[..., 6:, 256]).all()
