import pytest

from aquillm_vllm_h100.adapters import make_verifier_adapter


class Shape:
    def __init__(self, shape):
        self.shape = shape


def test_unsupported_verifier_falls_back_before_kernel():
    calls = []
    original = lambda **kw: calls.append(kw) or "baseline"
    wrapped = make_verifier_adapter(original, {"mtp": "fused", "split": "baseline"})
    result = wrapped(q=Shape((1, 5, 32, 128)), kv_cache=None, block_table=None,
                     seq_lens=None, k_chunk=None, v_chunk=None, scale=1,
                     block_size=32, kps=128, val_data_bytes=64)
    assert result == "baseline" and len(calls) == 1


def test_missing_caller_buffers_fall_back_before_cuda():
    wrapped = make_verifier_adapter(lambda **kw: "baseline", {"mtp": "fused", "split": "baseline"})
    assert wrapped(Shape((1, 5, 24, 256)), None, None, None, None, None, 1,
                   32, 256, 128) == "baseline"


def test_other_gqa_geometry_preserves_baseline(monkeypatch):
    import torch
    import aquillm_vllm_h100.adapters as adapters
    q = torch.empty((1, 5, 24, 256), dtype=torch.float16)
    raw = torch.empty((1, 5, 2, 256), dtype=q.dtype)
    mid = torch.empty((1, 2, 16, 8, 16, 257))
    cache = torch.empty((1, 32, 2, 400), dtype=torch.uint8)
    monkeypatch.setattr(adapters, "_h100", lambda device: pytest.fail("invalid geometry reached device probe"))
    wrapped = make_verifier_adapter(lambda **kw: "baseline", {"mtp": "fused", "split": "baseline"})
    assert wrapped(q, cache, torch.zeros((1, 1), dtype=torch.int32),
                   torch.tensor([5], dtype=torch.int32), raw, raw, 1, 32, 256, 128,
                   output=torch.empty_like(q), mid_o=mid) == "baseline"
