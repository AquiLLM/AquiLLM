"""Continuation attention state boundaries; runtime routing belongs to adapters."""
import math

from aquillm_vllm_h100.contracts import AttentionState


def normalize_flash_attention_state(output, lse, *, lse_layout: str, log_base: str) -> AttentionState:
    """Normalize an explicitly identified FA return layout to the shared ABI."""
    import torch
    if output.ndim != 3:
        raise ValueError("FlashAttention output must be [Q,Hq,D]")
    if lse_layout == "heads_tokens":
        lse = lse.transpose(0, 1)
    elif lse_layout != "tokens_heads":
        raise ValueError("unknown FlashAttention LSE layout")
    if lse.shape != output.shape[:2]:
        raise ValueError("FlashAttention LSE does not match output [Q,Hq]")
    if log_base not in ("natural", "log2"):
        raise ValueError("unknown FlashAttention log base")
    lse = lse.to(dtype=torch.float32)
    if log_base == "log2":
        lse = lse * math.log(2.)
    return AttentionState(output, lse)


def raw_chunk_attention(q, k, v, scale: float) -> AttentionState:
    """Installed FA causal attention over current raw K/V, including LSE.

    The inspected CUDA vLLM interface returns (output, natural LSE[Hq,Q])
    when return_softmax_lse=True. Equal Q/K lengths make the bottom-right FA
    mask the ordinary causal current-chunk mask. Historical cache is excluded.
    """
    import torch
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or k.shape[0] != q.shape[0] or k.shape[2] != q.shape[2]:
        raise ValueError("raw K/V must contain exactly the current chunk [Q,Hkv,D]")
    if q.shape[0] < 1 or q.shape[1] != 6 * k.shape[1] or q.shape[2] != 256:
        raise ValueError("raw chunk requires nonempty D256/GQA6")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("raw chunk inputs must have matching FP16/BF16 dtype")
    if not q.is_cuda or k.device != q.device or v.device != q.device:
        raise ValueError("raw chunk inputs must share a CUDA device")
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version
    cu = torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32)
    version = get_flash_attn_version(head_size=q.shape[2])
    options = {} if version is None else {"fa_version": version}
    result = flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                   max_seqlen_q=q.shape[0], max_seqlen_k=q.shape[0],
                                   softmax_scale=float(scale), causal=True, return_softmax_lse=True,
                                   **options)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("installed FlashAttention LSE return ABI differs from validated runtime")
    return normalize_flash_attention_state(*result, lse_layout="heads_tokens", log_base="natural")
