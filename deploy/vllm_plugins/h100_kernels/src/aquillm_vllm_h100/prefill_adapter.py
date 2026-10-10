"""Hash-bound post-Genesis continuation dispatch, installed once at startup."""
from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import textwrap

from aquillm_vllm_h100.contracts import AttentionState, KVSpec
from aquillm_vllm_h100.prefill import (
    PrefillProfile, PrefillRequest, select_prefill_route, validated_request_lengths,
)
from aquillm_vllm_h100.prefill_profiles import MODEL_REVISION, matches_runtime

log = logging.getLogger("aquillm.h100")
_constructor_diagnostics_logged = False


@functools.lru_cache(maxsize=8)
def _worker_properties(device):
    import torch
    return torch.cuda.get_device_properties(device)

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
        global _constructor_diagnostics_logged
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        original(self, *args, **kwargs)
        # Some installed backends discard these options. Preserve the passed
        # semantic values for the eligibility guard without altering init.
        values = bound.arguments
        try:
            from vllm.config import get_current_vllm_config
            current_config = get_current_vllm_config()
        except (ImportError, AttributeError, RuntimeError, AssertionError, ValueError):
            current_config = None
        resolved_revision = getattr(getattr(getattr(current_config, "model_config", None), "hf_config", None), "_commit_hash", None)
        flash_attn_version = getattr(getattr(current_config, "attention_config", None), "flash_attn_version", None)
        attn_type = getattr(values["attn_type"], "value", values["attn_type"])
        self._aquillm_h100_semantics = {
            "alibi": values["alibi_slopes"] is not None,
            "sliding_window": values["sliding_window"],
            "soft_cap": values["logits_soft_cap"] or 0.,
            "causal": str(attn_type).lower() in ("decoder", "attentiontype.decoder"),
            "kv_sharing": values["kv_sharing_target_layer_name"] is not None,
            "unknown_overlays": bool(values["kwargs"]),
            "model_revision": resolved_revision,
            "flash_attn_version": flash_attn_version,
        }
        if not _constructor_diagnostics_logged:
            _constructor_diagnostics_logged = True
            backend_version = getattr(self, "fa_version", None)
            # Identity comparisons, backend version numbers, semantic booleans,
            # and argument names only. No model hashes or argument values.
            log.warning("AQUILLM_H100 prefill_constructor revision_present=%s revision_match=%s configured_fa=%s "
                        "backend_fa=%s kwargs_keys=%s causal=%s sliding_window=%s soft_cap=%s",
                        resolved_revision is not None, resolved_revision == MODEL_REVISION,
                        flash_attn_version if type(flash_attn_version) is int else None,
                        backend_version if type(backend_version) is int else None,
                        tuple(sorted(values["kwargs"])), self._aquillm_h100_semantics["causal"],
                        values["sliding_window"] is not None, bool(values["logits_soft_cap"]))
    return initialize


def _make_route(profile, runtime_key):
    """Unsupported requests return None before allocating scratch or launching."""
    exercised = False
    diagnosed = set()
    def route(impl, layer, q, k, v, cache, metadata, index, total_tokens, destination):
        nonlocal exercised
        if not getattr(metadata, "is_prefill", False):
            return None
        lengths = validated_request_lengths(getattr(metadata, "query_start_loc_cpu", None),
                                            getattr(metadata, "seq_lens_cpu", None), index, total_tokens)
        diagnostic_region = (lengths is not None and isinstance(profile, PrefillProfile) and
                             any(region.prefix_min <= lengths[1] <= region.prefix_max and
                                 region.query_min <= lengths[0] <= region.query_max for region in profile.regions))

        def fallback(reason, detail=""):
            # CPU mirrors only: no additional CUDA properties, copies, or sync.
            # One record per reason per worker route, rather than per layer.
            if diagnostic_region and reason not in diagnosed:
                diagnosed.add(reason)
                log.warning("AQUILLM_H100 prefill_fallback reason=%s cached_len=%s query_len=%s%s",
                            reason, lengths[1], lengths[0], detail)
            return None

        semantics = getattr(impl, "_aquillm_h100_semantics", None)
        if semantics is None:
            return fallback("missing_semantics")
        if semantics.get("model_revision") != MODEL_REVISION:
            return fallback("model_revision_missing" if semantics.get("model_revision") is None else "model_revision_mismatch")
        if semantics.get("flash_attn_version") != 2:
            return fallback("flash_attn_version_missing" if semantics.get("flash_attn_version") is None else "flash_attn_version_mismatch")
        if getattr(q, "ndim", None) != 3:
            return fallback("query_rank")
        if lengths is None or lengths[0] != q.shape[0]:
            return fallback("invalid_lengths")
        import torch
        tensors = (q, k, v, cache, destination)
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            return fallback("non_tensor_input")
        if not q.is_cuda:
            return fallback("non_cuda_input")
        if torch.cuda.is_current_stream_capturing():
            return fallback("cuda_graph_capture")
        if cache is None or cache.ndim != 4 or cache.dtype != torch.uint8:
            return fallback("cache_layout")
        cfg = getattr(impl, "tq_config", None)
        if not getattr(cfg, "key_fp8", False) or getattr(cfg, "effective_value_quant_bits", None) != 4:
            return fallback("unsupported_quantization")
        # Normalize only metadata. Launch errors after eligibility propagate;
        # falling back after a potentially failed CUDA launch is unsafe.
        try:
            spec = KVSpec(impl.kv_cache_dtype, q.shape[2], q.shape[1], k.shape[1],
                          cache.shape[1], cfg.key_packed_size, impl._val_data_bytes)
        except (ValueError, TypeError, AttributeError, IndexError):
            return fallback("invalid_cache_spec")
        properties = _worker_properties(q.device)
        if not matches_runtime(spec, q.dtype, properties):
            return fallback("runtime_geometry",
                " actual_spec=%r actual_dtype=%s device=(%r,%s,%s,%s) "
                "expected=(H10080GB,SM90,132SM,float16,k8v4,Q24,Hkv4,D256,page2128,key256,value128)" %
                (spec, q.dtype, properties.name, properties.major, properties.minor, properties.multi_processor_count))
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
        decision = select_prefill_route(request, enabled=True, profile=profile, runtime_key=runtime_key)
        if not decision.eligible:
            return fallback(decision.reason)
        if (cache.device != q.device or destination.device != q.device
                or destination.shape != q.shape or destination.dtype != q.dtype
                or not all(all(stride > 0 for stride in tensor.stride()) for tensor in tensors)):
            return fallback("destination_layout")
        table = getattr(metadata, "block_table", None)
        if (not isinstance(table, torch.Tensor) or table.ndim != 2 or table.device != q.device
                or not 0 <= index < table.shape[0] or any(stride <= 0 for stride in table.stride())):
            return fallback("block_table_layout")
        if table.dtype not in (torch.int32, torch.int64) or table.shape[1] < (lengths[1] + spec.block_size - 1) // spec.block_size:
            return fallback("block_table_capacity")
        if cache.shape[2] != spec.num_kv_heads or cache.shape[3] < spec.key_packed_size + spec.value_data_bytes + 4:
            return fallback("cache_storage_geometry")
        from aquillm_vllm_h100.kernels.prefix import prefix_attention
        from aquillm_vllm_h100.kernels.merge import merge_attention_states
        from aquillm_vllm_h100.prefill import raw_chunk_attention
        # Transient same-stream buffers scale with current queries. They are
        # never attached during init/capture or sized by historical prefix.
        prefix = AttentionState(torch.empty_like(q, dtype=torch.float32),
                                torch.empty(q.shape[:2], dtype=torch.float32, device=q.device))
        prefix_attention(q, cache, table[index], lengths[1], impl.scale, spec, prefix)
        chunk = raw_chunk_attention(q, k, v, impl.scale, fa_version=semantics["flash_attn_version"])
        merge_attention_states(prefix, chunk, destination)
        if not exercised:
            exercised = True
            log.warning("AQUILLM_H100 route_exercised prefill runtime=%s cached_len=%s query_len=%s",
                        runtime_key, lengths[1], lengths[0])
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
