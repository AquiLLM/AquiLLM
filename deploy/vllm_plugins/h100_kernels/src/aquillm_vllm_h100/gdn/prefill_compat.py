"""Explicit FI 0.6.18 compatibility for the pinned vLLM GDN prefill caller."""
import functools
import importlib
import inspect


def prepare_install(module=None):
    """Widen sequence offsets on-device; keep the post-Genesis caller intact.

    FI 0.6.18's SM90 CP prefill kernel requires int64 offsets. The pinned
    vLLM caller supplies int32. Integer widening is exact and preserves graph
    replay through a captured device conversion; no CPU tensor reads occur.
    This is installed only for the explicitly verified candidate runtime.
    """
    import torch
    if module is None:
        module = importlib.import_module('vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn')
    original = module.fi_chunk_gated_delta_rule
    if getattr(original, '_aquillm_fi_int64_offsets', False):
        return module, original
    signature = inspect.signature(original)
    expected = ('q','k','v','g','beta','initial_state','output_final_state',
                'cu_seqlens','use_qk_l2norm_in_kernel')
    if tuple(signature.parameters) != expected:
        raise RuntimeError('GDN prefill caller signature changed')

    @functools.wraps(original)
    def call(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        offsets = bound.arguments.get('cu_seqlens')
        if offsets is not None and offsets.dtype == torch.int32:
            bound.arguments['cu_seqlens'] = offsets.to(dtype=torch.int64)
        return original(*bound.args, **bound.kwargs)

    call._aquillm_fi_int64_offsets = True
    return module, call
