import pytest
import torch

from reference import make_verify_batch, reference_attention, reference_verify, unpack_prefix


def test_uniform_causal_attention_analytic():
    q = torch.zeros(3, 2, 2)
    k = torch.ones(3, 1, 2)
    v = torch.tensor([[[2., 0.]], [[0., 4.]], [[4., 2.]]])
    state = reference_attention(q, k, v)
    expected = torch.tensor([[[2., 0.]], [[1., 2.]], [[2., 2.]]]).expand(3, 2, 2)
    torch.testing.assert_close(state.output, expected)
    torch.testing.assert_close(state.lse, torch.arange(1, 4).float().log()[:, None].expand(3, 2))


def test_one_key_analytic():
    state = reference_attention(torch.ones(1, 2, 4), torch.ones(1, 1, 4),
                                torch.tensor([[[1., 2., 3., 4.]]]))
    torch.testing.assert_close(state.output, torch.tensor([[[1., 2., 3., 4.]]]).expand(1, 2, 4))
    torch.testing.assert_close(state.lse, torch.full((1, 2), 2.))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")
def test_packed_reference_and_poisoned_tail():
    batch = make_verify_batch([0, 1, 33], strided=True)
    expected = reference_verify(batch)
    # Packed speculative slots must never affect the semantic reference.
    for b in range(3):
        prior = int(batch.seq_lens[b]) - batch.q.shape[1]
        for token in range(prior, prior + batch.q.shape[1]):
            page = batch.block_table[b, token // batch.spec.block_size]
            batch.kv_cache[page, token % batch.spec.block_size].fill_(0xFF)
    torch.testing.assert_close(reference_verify(batch), expected)
    k, v = unpack_prefix(batch, 2)
    assert k.shape == v.shape == (33, 4, 256)

