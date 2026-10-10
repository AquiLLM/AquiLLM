import pytest
import torch

from reference import assert_close, make_verify_batch, reference_attention

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_raw_chunk_is_causal_and_exposes_natural_lse(dtype):
    from aquillm_vllm_h100.prefill import raw_chunk_attention
    batch = make_verify_batch([37], length=129, dtype=dtype)
    q, k, v = batch.q[0], batch.raw_k[0], batch.raw_v[0]
    actual = raw_chunk_attention(q, k, v, batch.scale)
    expected = reference_attention(q, k, v)
    assert_close(actual.output, expected.output, dtype)
    torch.testing.assert_close(actual.lse, expected.lse, atol=0.015, rtol=0.002)
    before = actual.output.clone()
    k[65:] = 40
    v[65:] = -70
    after = raw_chunk_attention(q, k, v, batch.scale)
    torch.testing.assert_close(after.output[:65], before[:65], atol=0, rtol=0)


def test_complete_continuation_merges_only_committed_prefix_plus_raw_causal_chunk():
    from aquillm_vllm_h100.prefill import raw_chunk_attention
    from aquillm_vllm_h100.kernels.merge import merge_attention_states
    from aquillm_vllm_h100.kernels.prefix import prefix_attention
    from aquillm_vllm_h100.contracts import AttentionState
    from reference import unpack_prefix
    batch = make_verify_batch([257], length=129)
    q = batch.q[0]
    prefix = AttentionState(torch.empty_like(q, dtype=torch.float32), torch.empty(q.shape[:2], device="cuda"))
    prefix_attention(q, batch.kv_cache, batch.block_table[0], 257, batch.scale, batch.spec, prefix)
    chunk = raw_chunk_attention(q, batch.raw_k[0], batch.raw_v[0], batch.scale)
    out = torch.empty_like(q)
    merge_attention_states(prefix, chunk, out)
    pk, pv = unpack_prefix(batch, 0)
    expected = reference_attention(q, torch.cat((pk, batch.raw_k[0].float())),
                                   torch.cat((pv, batch.raw_v[0].float())), causal_prefix=257)
    assert_close(out, expected.output, q.dtype)
