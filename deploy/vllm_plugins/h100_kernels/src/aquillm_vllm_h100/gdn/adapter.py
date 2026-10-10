"""Experimental, opt-in N=1 FlashInfer MTP bridge after Genesis patching.

Only static metadata is inspected on the host. Device offsets, accepted
columns and state slots are resolved on the GPU. Candidate errors terminate
the worker: falling back after a launch could reuse partially mutated state.
"""
from __future__ import annotations

import functools
import inspect
import logging
import math

from .capability import validate_api

log = logging.getLogger("aquillm.h100")


@functools.lru_cache(maxsize=8)
def _h100(device):
    import torch
    properties = torch.cuda.get_device_properties(device)
    return properties.major == 9 and properties.minor == 0 and "H100" in properties.name


def supported(A_log, a, b, dt_bias, q, k, v, beta=1.0, threshold=20.0,
              scale=None, initial_state=None, inplace_final_state=True,
              cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
              use_qk_l2norm_in_kernel=False, is_kda=False):
    """Fail closed using tensor metadata only; never read a device scalar."""
    import torch
    tensors = (A_log, a, b, dt_bias, q, k, v, initial_state,
               cu_seqlens, ssm_state_indices, num_accepted_tokens)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return False
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[2:] != (16,128):
        return False
    t = q.shape[1]
    if not 2 <= t <= 5 or k.shape != q.shape or v.shape != (1,t,48,128):
        return False
    if a.shape not in ((t,48), (1,t,48)) or b.shape not in ((t,48), (1,t,48)):
        return False
    if A_log.shape != (48,) or dt_bias.shape != (48,):
        return False
    if any(x.dtype != torch.float16 for x in (q,k,v,a,b)):
        return False
    if any(x.dtype not in (torch.float16, torch.float32) for x in (A_log,dt_bias)):
        return False
    for tensor,heads in ((q,16),(k,16),(v,48)):
        if (tensor.stride(-1) != 1 or tensor.stride(-2) < 128
                or tensor.stride(-2) % 8 or tensor.stride(-3) % 8
                or tensor.stride(-3) < (heads-1)*tensor.stride(-2)+128
                or tensor.storage_offset() % 8):
            return False
    for tensor in (a,b):
        if tensor.stride(-1) != 1 or tensor.stride(-2) < 48:
            return False
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        return False
    state = initial_state
    if (state.dtype != torch.float32 or state.ndim != 4
            or state.shape[0] < 2 or state.shape[1:] != (48,128,128)
            or state.stride()[1:] != (16384,128,1)
            or state.stride(0) < 48*128*128 or state.stride(0) % 4
            or state.storage_offset() % 4):
        return False
    if (cu_seqlens.shape != (2,) or num_accepted_tokens.shape != (1,)
            or ssm_state_indices.ndim != 2 or ssm_state_indices.shape[0] != 1
            or ssm_state_indices.shape[1] < t or ssm_state_indices.stride(1) != 1
            or cu_seqlens.stride(0) != 1):
        return False
    if any(x.dtype not in (torch.int32,torch.int64)
           for x in (cu_seqlens,ssm_state_indices,num_accepted_tokens)):
        return False
    if not all(x.device == q.device and all(s > 0 for s in x.stride()) for x in tensors):
        return False
    if (q.device.type != "cuda" or not isinstance(beta, (int,float))
            or not isinstance(threshold, (int,float)) or beta != 1.0 or threshold != 20.0
            or inplace_final_state is not True or is_kda is not False
            or not isinstance(use_qk_l2norm_in_kernel, bool)):
        return False
    if scale is not None and (not isinstance(scale, (int,float))
                              or not math.isfinite(scale) or scale <= 0):
        return False
    return _h100(q.device)


def make_adapter(original, candidate=None, *, route="flashinfer"):
    """Capture the post-Genesis callable and retain positional/default semantics."""
    exercised = False

    @functools.wraps(original)
    def call(A_log, a, b, dt_bias, q, k, v, beta=1.0, threshold=20.0,
             scale=None, initial_state=None, inplace_final_state=True,
             cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
             use_qk_l2norm_in_kernel=False, is_kda=False):
        nonlocal exercised
        args = dict(A_log=A_log,a=a,b=b,dt_bias=dt_bias,q=q,k=k,v=v,beta=beta,
                    threshold=threshold,scale=scale,initial_state=initial_state,
                    inplace_final_state=inplace_final_state,cu_seqlens=cu_seqlens,
                    ssm_state_indices=ssm_state_indices,num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,is_kda=is_kda)
        if not supported(**args):
            return original(**args)
        try:
            if candidate is None:
                from .native import launch
            else:
                launch = candidate
            output = launch(**args)
        except Exception as error:
            # Genesis catches Exception to retry; BaseException ends this worker.
            raise SystemExit(f"AquiLLM FlashInfer GDN launch failed: {error}") from error
        if not exercised:
            exercised = True
            log.warning("AQUILLM_H100 route_exercised gdn=%s shape=%s "
                        "precision=%s gate_strides=%s state_strides=%s", route, tuple(q.shape),
                        getattr(launch,"_operand_precision","fp16"),
                        (tuple(a.stride()),tuple(b.stride())),tuple(initial_state.stride()))
        return output, initial_state

    call._aquillm_gdn_adapter = True
    call._aquillm_gdn_route = route
    return call


def capture_original(module=None):
    """Capture the actual serving alias for qualification, unwrapping our bridge."""
    if module is None:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as module
    original = module.fused_sigmoid_gating_delta_rule_update
    if getattr(original, "_aquillm_gdn_adapter", False):
        original = original.__wrapped__
    return original


def _prepare_install(module=None, candidate=None, *, route="flashinfer"):
    """Validate every import/signature before changing any runtime alias."""
    if module is None:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as module
    original = module.fused_sigmoid_gating_delta_rule_update
    if getattr(original, "_aquillm_gdn_adapter", False):
        if getattr(original,"_aquillm_gdn_route","flashinfer") == route:
            return module, original
        original = original.__wrapped__
    if route == "flashinfer":
        if candidate is None:
            from flashinfer import gdn_decode as candidate
        validate_api(candidate)
    expected = inspect.signature(supported)
    actual = inspect.signature(original)
    if tuple(actual.parameters) != tuple(expected.parameters):
        raise ValueError(f"unsupported post-Genesis fused GDN signature: {actual}")
    for name, parameter in expected.parameters.items():
        if actual.parameters[name].default != parameter.default:
            raise ValueError(f"unsupported post-Genesis GDN default: {name}")
    # Import our launch dependencies during setup, before committing the alias.
    from . import native  # noqa: F401
    return module, make_adapter(original, route=route)


def prepare_install(module=None, candidate=None):
    """Keep the explicit FlashInfer profile's public API compatibility gate."""
    return _prepare_install(module,candidate)


def prepare_native_install(module=None):
    """Validate the serving alias/native imports without unused FI public APIs."""
    return _prepare_install(module,route="native-fp16")


def install_adapter(module=None, candidate=None):
    module, call = prepare_install(module, candidate)
    module.fused_sigmoid_gating_delta_rule_update = call
    return {"installed": True, "qualification": "experimental_pending_gpu_serving"}


def install_native_adapter(module=None):
    module, call = prepare_native_install(module)
    module.fused_sigmoid_gating_delta_rule_update = call
    return {"installed": True, "qualification": "experimental_pending_serving"}
