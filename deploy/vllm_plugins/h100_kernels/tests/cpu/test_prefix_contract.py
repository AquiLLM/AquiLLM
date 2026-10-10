import pytest
import torch

from aquillm_vllm_h100.contracts import AttentionState, KVSpec


def test_prefix_rejects_non_fp32_state_before_launch():
    from aquillm_vllm_h100.kernels.prefix import validate_prefix_metadata
    spec = KVSpec("turboquant_k8v4", 256, 12, 2, 32, 256, 128)
    q = torch.empty(129, 12, 256, dtype=torch.float16)
    cache = torch.empty(3, 32, 2, 388, dtype=torch.uint8)
    state = AttentionState(torch.empty_like(q), torch.empty(129, 12))
    with pytest.raises(ValueError, match="FP32"):
        validate_prefix_metadata(q, cache, torch.zeros(3, dtype=torch.int32), 65, spec, state)


def test_prefix_rejects_insufficient_block_table_before_launch():
    from aquillm_vllm_h100.kernels.prefix import validate_prefix_metadata
    spec = KVSpec("turboquant_k8v4", 256, 12, 2, 32, 256, 128)
    q = torch.empty(129, 12, 256, dtype=torch.float16)
    state = AttentionState(torch.empty(q.shape), torch.empty(129, 12))
    with pytest.raises(ValueError, match="block table"):
        validate_prefix_metadata(q, torch.empty(3, 32, 2, 388, dtype=torch.uint8),
                                 torch.zeros(2, dtype=torch.int32), 65, spec, state)
