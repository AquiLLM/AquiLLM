"""Reject invalid caller buffers before any kernel construction or CUDA launch."""
import pytest

torch = pytest.importorskip("torch")

from aquillm_vllm_h100.contracts import KVSpec, SplitPlan, VerifyBatch
from aquillm_vllm_h100.kernels.splitk_reference import launch_reference_stage1


def cpu_batch():
    spec = KVSpec("turboquant_k8v4", 256, 24, 4, 32, 256, 128)
    return VerifyBatch(torch.empty((1, 5, 24, 256), dtype=torch.float16),
                       torch.empty((1, 32, 4, 388), dtype=torch.uint8),
                       torch.zeros((1, 1), dtype=torch.int32),
                       torch.tensor([5], dtype=torch.int32),
                       torch.empty((1, 5, 4, 256), dtype=torch.float16),
                       torch.empty((1, 5, 4, 256), dtype=torch.float16),
                       1 / 16, spec)


@pytest.mark.parametrize("bad", ["scratch_shape", "scratch_dtype", "query_dtype", "raw_shape", "raw_dtype", "seq_shape", "seq_dtype", "cache_dtype", "cache_slot", "block_table"])
def test_bad_caller_geometry_is_rejected_before_launch(bad):
    batch = cpu_batch()
    mid = torch.empty((1, 4, 16, 8, 8, 257), dtype=torch.float32)
    if bad == "scratch_shape":
        mid = mid[:, :, :-1]
    elif bad == "scratch_dtype":
        mid = mid.half()
    elif bad == "query_dtype":
        batch.q = batch.q.float()
    elif bad == "raw_shape":
        batch.raw_k = batch.raw_k[:, :4]
    elif bad == "raw_dtype":
        batch.raw_v = batch.raw_v.float()
    elif bad == "seq_shape":
        batch.seq_lens = batch.seq_lens[:, None]
    elif bad == "seq_dtype":
        batch.seq_lens = batch.seq_lens.float()
    elif bad == "cache_dtype":
        batch.kv_cache = batch.kv_cache.float()
    elif bad == "cache_slot":
        batch.kv_cache = batch.kv_cache[..., :384]
    elif bad == "block_table":
        batch.block_table = batch.block_table.float()
    with pytest.raises(ValueError) as rejected:
        launch_reference_stage1(batch, SplitPlan(15, ()), mid)
    assert "CUDA device" not in str(rejected.value), "geometry must be validated before launch eligibility"


def test_valid_cpu_buffers_cannot_enter_cuda_kernel_builder():
    with pytest.raises(ValueError, match="CUDA device"):
        launch_reference_stage1(cpu_batch(), SplitPlan(15, ()),
                                torch.empty((1, 4, 16, 8, 8, 257), dtype=torch.float32))
