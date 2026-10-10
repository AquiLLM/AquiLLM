"""Hash-bound post-Genesis continuation dispatch, installed once at startup."""
from __future__ import annotations

import functools
import hashlib
import inspect
import textwrap

from aquillm_vllm_h100.contracts import AttentionState, KVSpec
from aquillm_vllm_h100.prefill import (
    PrefillProfile, PrefillRequest, select_prefill_route, validated_request_lengths,
)

PREFILL_FINGERPRINT = "fd7a69b113eecb1e8337426fe1c8b8a1f49d95bcbf2b913c46cb074a2494d1d3"
_CONSTRUCTOR_PARAMETERS = (
    "self", "num_heads", "head_size", "scale", "num_kv_heads", "alibi_slopes",
    "sliding_window", "kv_cache_dtype", "logits_soft_cap", "attn_type",
    "kv_sharing_target_layer_name", "kwargs",
)


def rewrite_prefill_method(original, source: str, route):
    """Compile the inspected method with one per-request dispatch insertion.

    Every existing fresh/PN401/fallback statement remains in its original order.
    The complete source fingerprint is checked before the unique P101 anchor.
    """
    if hashlib.sha256(source.lstrip().rstrip().encode()).hexdigest() != PREFILL_FINGERPRINT:
        raise ValueError("post-Genesis prefill method fingerprint mismatch")
    source = textwrap.dedent(source)
    anchor = "cached_len = seq_len - q_len"
    lines = source.splitlines()
    matches = [i for i, line in enumerate(lines) if line.strip() == anchor]
    if len(matches) != 1:
        raise ValueError("post-Genesis P101 anchor is not unique")
    position = matches[0]
    indent = lines[position][:len(lines[position]) - len(lines[position].lstrip())]
    insertion = [
        indent + "_h100_out = _aquillm_h100_continuation(self, layer, q_seq, k_seq, v_seq,",
        indent + "    kv_cache, attn_metadata, i, N, output[q_start:q_end])",
        indent + "if _h100_out is not None:",
        indent + "    output[q_start:q_end] = _h100_out",
        indent + "    continue",
    ]
    lines[position + 1:position + 1] = insertion
    namespace = dict(original.__globals__)
    namespace["_aquillm_h100_continuation"] = route
    exec(compile("\n".join(lines), "<aquillm-h100-prefill>", "exec"), namespace)
    rewritten = namespace[original.__name__]
    return functools.update_wrapper(rewritten, original)


def _capture_constructor(original):
    signature = inspect.signature(original)
    if tuple(signature.parameters) != _CONSTRUCTOR_PARAMETERS:
        raise ValueError("post-Genesis constructor ABI mismatch")

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        original(self, *args, **kwargs)
        # Some installed backends discard these options. Preserve the passed
        # semantic values for the eligibility guard without altering init.
        values = bound.arguments
        attn_type = getattr(values["attn_type"], "value", values["attn_type"])
        self._aquillm_h100_semantics = {
            "alibi": values["alibi_slopes"] is not None,
            "sliding_window": values["sliding_window"],
            "soft_cap": values["logits_soft_cap"] or 0.,
            "causal": str(attn_type).lower() in ("decoder", "attentiontype.decoder"),
            "kv_sharing": values["kv_sharing_target_layer_name"] is not None,
            "unknown_overlays": bool(values["kwargs"]),
        }
    return initialize


def _make_route(profile, runtime_key):
    """Unsupported requests return None before allocating scratch or launching."""
    def route(impl, layer, q, k, v, cache, metadata, index, total_tokens, destination):
        semantics = getattr(impl, "_aquillm_h100_semantics", None)
        if semantics is None or not getattr(metadata, "is_prefill", False):
            return None
        lengths = validated_request_lengths(getattr(metadata, "query_start_loc_cpu", None),
                                            getattr(metadata, "seq_lens_cpu", None), index, total_tokens)
        if lengths is None or lengths[0] != q.shape[0]:
            return None
        import torch
        if not q.is_cuda or torch.cuda.is_current_stream_capturing():
            return None
        if cache is None or cache.ndim != 4 or cache.dtype != torch.uint8:
            return None
        cfg = impl.tq_config
        if not cfg.key_fp8 or cfg.effective_value_quant_bits != 4:
            return None
        # Normalize only metadata. Launch errors after eligibility propagate;
        # falling back after a potentially failed CUDA launch is unsafe.
        try:
            spec = KVSpec(impl.kv_cache_dtype, q.shape[2], q.shape[1], k.shape[1],
                          cache.shape[1], cfg.key_packed_size, impl._val_data_bytes)
        except (ValueError, TypeError, AttributeError, IndexError):
            return None
        raw_valid = (k.ndim == 3 and v.shape == k.shape and k.shape == (q.shape[0], spec.num_kv_heads, spec.head_dim)
                     and q.dtype in (torch.float16, torch.bfloat16) and k.dtype == q.dtype and v.dtype == q.dtype
                     and k.device == q.device and v.device == q.device)
        is_verification = any(bool(getattr(metadata, name, False)) for name in
                              ("is_verification", "is_speculative", "is_spec_decode"))
        masked = any(getattr(metadata, name, None) is not None for name in
                     ("mm_prefix_ranges", "multimodal_prefix_mask", "prefix_mask"))
        request = PrefillRequest(lengths[0], lengths[1], spec,
                                sm=torch.cuda.get_device_capability(q.device), raw_kv_valid=raw_valid,
                                causal=semantics["causal"], is_verification=is_verification,
                                execution_mode="eager", sliding_window=semantics["sliding_window"],
                                alibi=semantics["alibi"], soft_cap=semantics["soft_cap"],
                                kv_sharing=semantics["kv_sharing"], unknown_overlays=semantics["unknown_overlays"],
                                sinks=getattr(impl, "sinks", None) is not None or getattr(metadata, "sinks", None) is not None,
                                multimodal_prefix_mask=masked)
        if not select_prefill_route(request, enabled=True, profile=profile, runtime_key=runtime_key).eligible:
            return None
        if cache.device != q.device or destination.device != q.device:
            return None
        table = metadata.block_table
        if table.ndim != 2 or table.device != q.device or not 0 <= index < table.shape[0]:
            return None
        if table.dtype not in (torch.int32, torch.int64) or table.shape[1] < (lengths[1] + spec.block_size - 1) // spec.block_size:
            return None
        if cache.shape[2] != spec.num_kv_heads or cache.shape[3] < spec.key_packed_size + spec.value_data_bytes + 4:
            return None
        from aquillm_vllm_h100.kernels.prefix import prefix_attention
        from aquillm_vllm_h100.kernels.merge import merge_attention_states
        from aquillm_vllm_h100.prefill import raw_chunk_attention
        # Transient same-stream buffers scale with current queries. They are
        # never attached during init/capture or sized by historical prefix.
        prefix = AttentionState(torch.empty_like(q, dtype=torch.float32),
                                torch.empty(q.shape[:2], dtype=torch.float32, device=q.device))
        prefix_attention(q, cache, table[index], lengths[1], impl.scale, spec, prefix)
        chunk = raw_chunk_attention(q, k, v, impl.scale)
        merge_attention_states(prefix, chunk, destination)
        return destination
    return route


def install_prefill_adapter(profile: PrefillProfile, runtime_key: str) -> dict:
    """Startup-only install; missing/unmeasured profiles remain feature off."""
    if not isinstance(profile, PrefillProfile) or not profile.regions or profile.runtime_key != runtime_key:
        return {"installed": False, "reason": "missing_or_mismatched_measured_profile"}
    from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionImpl
    cls = TurboQuantAttentionImpl
    existing = getattr(cls, "_aquillm_h100_prefill_install", None)
    if existing is not None:
        return {"installed": existing == (runtime_key, profile), "reason": "already_installed"}
    try:
        original = cls._prefill_attention
        rewritten = rewrite_prefill_method(original, inspect.getsource(original), _make_route(profile, runtime_key))
        initializer = _capture_constructor(cls.__init__)
    except (ValueError, TypeError, OSError) as exc:
        return {"installed": False, "reason": str(exc)}
    cls.__init__ = initializer
    cls._prefill_attention = rewritten
    cls._aquillm_h100_prefill_install = (runtime_key, profile)
    return {"installed": True, "reason": "measured_profile_and_runtime_abi_validated"}
