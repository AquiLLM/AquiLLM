import pytest
import torch

from aquillm_vllm_h100.contracts import AttentionState
from reference import assert_close, make_verify_batch, reference_attention, unpack_prefix

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
@pytest.mark.parametrize("prior", [0, 1, 31, 65, 257])
def test_prefix_matches_independent_oracle_with_reordered_pages_and_strides(dtype, block_size, prior):
    from aquillm_vllm_h100.kernels.prefix import prefix_attention
    batch = make_verify_batch([prior], length=129, dtype=dtype, block_size=block_size, strided=True)
    q = batch.q[0]
    state = AttentionState(torch.empty_like(q, dtype=torch.float32),
                           torch.empty(q.shape[:2], device=q.device, dtype=torch.float32))
    prefix_attention(q, batch.kv_cache, batch.block_table[0], prior, batch.scale, batch.spec, state)
    k, v = unpack_prefix(batch, 0)
    expected = reference_attention(q, k, v, causal_prefix=None)
    assert_close(state.output, expected.output, dtype)
    torch.testing.assert_close(state.lse, expected.lse, atol=0.015, rtol=0.002)


@pytest.mark.parametrize("value_scale", [0., 0.125])
def test_prefix_excludes_poisoned_future_slots_and_handles_constant_values(value_scale):
    from aquillm_vllm_h100.kernels.prefix import prefix_attention
    batch = make_verify_batch([37], length=129)
    offset = 384
    batch.kv_cache[..., offset:offset + 2] = torch.tensor([value_scale], device="cuda", dtype=torch.float16).view(torch.uint8)
    batch.kv_cache[..., offset + 2:offset + 4] = torch.tensor([2.], device="cuda", dtype=torch.float16).view(torch.uint8)
    q = batch.q[0]
    state = AttentionState(torch.empty_like(q, dtype=torch.float32), torch.empty(q.shape[:2], device="cuda"))
    prefix_attention(q, batch.kv_cache, batch.block_table[0], 37, batch.scale, batch.spec, state)
    k, v = unpack_prefix(batch, 0)
    expected = reference_attention(q, k, v, causal_prefix=None)
    assert_close(state.output, expected.output, torch.float16)
    before = state.output.clone()
    # Poison the raw chunk and its compressed slots. Only committed positions are visible.
    for position in range(37, 166):
        page = batch.block_table[0, position // batch.spec.block_size]
        batch.kv_cache[page, position % batch.spec.block_size].fill_(0x7f)
    prefix_attention(q, batch.kv_cache, batch.block_table[0], 37, batch.scale, batch.spec, state)
    torch.testing.assert_close(state.output, before, atol=0, rtol=0)


def test_prefix_honors_query_dimension_cache_byte_table_and_state_strides():
    from aquillm_vllm_h100.kernels.prefix import prefix_attention
    batch = make_verify_batch([65], length=129)
    q = torch.empty(129, 24, 512, device="cuda", dtype=torch.float16)[..., ::2]
    q.copy_(batch.q[0])
    packed = torch.empty(*batch.kv_cache.shape[:-1], 800, device="cuda", dtype=torch.uint8)[..., ::2]
    packed.copy_(batch.kv_cache)
    table = torch.empty(batch.block_table.shape[1] * 2, device="cuda", dtype=torch.int32)[::2]
    table.copy_(batch.block_table[0])
    state = AttentionState(torch.empty(129, 24, 512, device="cuda")[..., ::2],
                           torch.empty(24, 129, device="cuda").t())
    prefix_attention(q, packed, table, 65, batch.scale, batch.spec, state)
    pk, pv = unpack_prefix(batch, 0)
    expected = reference_attention(q, pk, pv, causal_prefix=None)
    assert_close(state.output, expected.output, q.dtype)
    torch.testing.assert_close(state.lse, expected.lse, atol=0.015, rtol=0.002)
