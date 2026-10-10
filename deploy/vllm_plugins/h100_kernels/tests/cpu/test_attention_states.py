"""Hand-derived partial-state cases catch weight/base/empty-state mistakes."""
import math

import pytest
import torch

from aquillm_vllm_h100.contracts import AttentionState


def state(values, lse):
    return AttentionState(torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1),
                          torch.tensor([[lse]], dtype=torch.float32))


@pytest.mark.parametrize("prefix,chunk,expected", [
    (state([2, 6], 0), state([4, 2], 0), [3, 4]),
    (state([float('nan'), float('nan')], -math.inf), state([4, 2], 0), [4, 2]),
    (state([2, 6], 0), state([float('nan'), float('nan')], -math.inf), [2, 6]),
    (state([float('nan'), float('nan')], -math.inf), state([float('nan'), float('nan')], -math.inf), [0, 0]),
    (state([2, 6], 10000), state([4, 2], 9999), [2.5378828427, 4.9242343145]),
    (state([2, 6], -10000), state([4, 2], -9999), [3.4621171573, 3.0757656855]),
])
def test_merge_uses_stable_natural_log_weights(prefix, chunk, expected):
    from aquillm_vllm_h100.kernels.merge import merge_attention_states
    output = torch.empty_like(prefix.output)
    merge_attention_states(prefix, chunk, output)
    torch.testing.assert_close(output.flatten(), torch.tensor(expected, dtype=torch.float32), atol=1e-6, rtol=1e-6)


def test_flash_lse_heads_by_token_and_log2_are_normalized_explicitly():
    from aquillm_vllm_h100.prefill import normalize_flash_attention_state
    output = torch.zeros(3, 2, 4)
    lse = torch.tensor([[0., 1., 2.], [3., 4., 5.]])
    result = normalize_flash_attention_state(output, lse, lse_layout="heads_tokens", log_base="log2")
    torch.testing.assert_close(result.lse, torch.tensor([[0., 3.], [1., 4.], [2., 5.]]) * math.log(2))


def test_flash_lse_square_shape_does_not_guess_orientation():
    from aquillm_vllm_h100.prefill import normalize_flash_attention_state
    output = torch.zeros(2, 2, 4)
    lse = torch.tensor([[0., 1.], [3., 4.]])
    result = normalize_flash_attention_state(output, lse, lse_layout="heads_tokens", log_base="natural")
    torch.testing.assert_close(result.lse, torch.tensor([[0., 3.], [1., 4.]]))


def test_flash_state_rejects_unknown_log_base_instead_of_silent_weight_mismatch():
    from aquillm_vllm_h100.prefill import normalize_flash_attention_state
    with pytest.raises(ValueError, match="log base"):
        normalize_flash_attention_state(torch.zeros(3, 2, 4), torch.zeros(2, 3),
                                        lse_layout="heads_tokens", log_base="unknown")


def test_merge_honors_noncontiguous_output_and_input_strides():
    from aquillm_vllm_h100.kernels.merge import merge_attention_states
    op = torch.arange(24.).reshape(2, 3, 4).transpose(0, 1)
    oc = op + 2
    lp = torch.zeros(2, 3).t()
    output = torch.empty(2, 3, 8)[..., ::2].transpose(0, 1)
    merge_attention_states(AttentionState(op, lp), AttentionState(oc, lp), output)
    torch.testing.assert_close(output, op + 1)


def test_raw_chunk_rejects_history_in_raw_kv_before_any_cuda_launch():
    from aquillm_vllm_h100.prefill import raw_chunk_attention
    with pytest.raises(ValueError, match="current chunk"):
        raw_chunk_attention(torch.empty(129, 24, 256), torch.empty(257, 4, 256),
                            torch.empty(257, 4, 256), 0.0625)
