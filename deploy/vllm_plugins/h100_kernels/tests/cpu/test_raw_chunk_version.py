"""Forward executes the captured FA choice after model construction has ended."""
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("version", [2, 3])
def test_explicit_version_never_redetects_from_missing_current_config(monkeypatch, version):
    import aquillm_vllm_h100.prefill as prefill
    device = SimpleNamespace(type="cuda", index=0)
    q = SimpleNamespace(ndim=3, shape=(129, 24, 256), dtype="fp16", device=device, is_cuda=True)
    k = SimpleNamespace(ndim=3, shape=(129, 4, 256), dtype="fp16", device=device)
    v = SimpleNamespace(shape=k.shape, dtype=k.dtype, device=device)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16="fp16", bfloat16="bf16", int32="int32",
        tensor=lambda *args, **kwargs: "cumulative_lengths"))
    def launch(**kwargs):
        assert kwargs["fa_version"] == version
        assert kwargs["return_softmax_lse"] is True
        return "output", "lse"
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.fa_utils", SimpleNamespace(
        flash_attn_varlen_func=launch,
        get_flash_attn_version=lambda **kwargs: pytest.fail("forward redetected FA outside construction context")))
    monkeypatch.setattr(prefill, "normalize_flash_attention_state", lambda *args, **kwargs: args)
    assert prefill.raw_chunk_attention(q, k, v, .0625, fa_version=version) == ("output", "lse")


@pytest.mark.parametrize("version", [1, True, "2"])
def test_invalid_explicit_version_rejected_before_tensor_or_launch_access(version):
    from aquillm_vllm_h100.prefill import raw_chunk_attention
    with pytest.raises(ValueError, match="FlashAttention version"):
        raw_chunk_attention(None, None, None, .0625, fa_version=version)
