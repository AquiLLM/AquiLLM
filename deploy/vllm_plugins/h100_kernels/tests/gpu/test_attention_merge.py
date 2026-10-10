import math

import pytest
import torch

from aquillm_vllm_h100.contracts import AttentionState

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")]


def test_gpu_merge_strides_empty_states_and_extreme_natural_lse():
    from aquillm_vllm_h100.kernels.merge import merge_attention_states
    op = torch.tensor([2., 6.], device="cuda").expand(4, 2, 2)
    oc = torch.tensor([4., 2.], device="cuda").expand(4, 2, 2)
    lp = torch.tensor([0., -math.inf, -math.inf, 10000.], device="cuda").expand(2, 4).t()
    lc = torch.tensor([0., 0., -math.inf, 9999.], device="cuda").expand(2, 4).t()
    output = torch.empty(4, 2, 4, device="cuda")[..., ::2]
    merge_attention_states(AttentionState(op, lp), AttentionState(oc, lc), output)
    expected = torch.tensor([[3., 4.], [4., 2.], [0., 0.], [2.53788284, 4.92423431]], device="cuda")
    torch.testing.assert_close(output, expected[:, None, :].expand_as(output), atol=1e-6, rtol=1e-6)
