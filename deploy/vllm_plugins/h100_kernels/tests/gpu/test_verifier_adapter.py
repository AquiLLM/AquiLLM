import pytest
import torch

from aquillm_vllm_h100.adapters import make_verifier_adapter
from reference import assert_close, make_verify_batch, reference_verify

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available() or
              torch.cuda.get_device_capability() != (9, 0), reason="H100 required")]


def test_actual_adapter_capture_replay_does_not_fall_back():
    batch = make_verify_batch([2048])
    mid = torch.empty((1, 4, 16, 8, 8, 257), device="cuda", dtype=torch.float32)
    out = torch.empty_like(batch.q)
    def unexpected(**kwargs):
        pytest.fail("eligible adapter fell back to baseline")
    wrapper = make_verifier_adapter(unexpected, {"mtp": "fused", "split": "baseline"})
    def call():
        wrapper(batch.q, batch.kv_cache, batch.block_table, batch.seq_lens,
                batch.raw_k, batch.raw_v, batch.scale, 32, 256, 128,
                output=out, num_splits=15, mid_o=mid)
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    pointers = (mid.data_ptr(), out.data_ptr())
    for prior in (0, 1, 2048, 7, 2048, 0):
        batch.seq_lens.fill_(prior + 5)
        mid.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert_close(out, reference_verify(batch), batch.q.dtype)
        assert (mid.data_ptr(), out.data_ptr()) == pointers
