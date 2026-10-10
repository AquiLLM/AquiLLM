"""Continuation attention state boundaries; runtime routing belongs to adapters."""
import math
import operator
from dataclasses import dataclass

from aquillm_vllm_h100.contracts import AttentionState, KVSpec, RouteDecision


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


def raw_chunk_attention(q, k, v, scale: float, *, fa_version: int | None = None) -> AttentionState:
    """Installed FA causal attention over current raw K/V, including LSE.

    The inspected CUDA vLLM interface returns (output, natural LSE[Hq,Q])
    when return_softmax_lse=True. Equal Q/K lengths make the bottom-right FA
    mask the ordinary causal current-chunk mask. Historical cache is excluded.
    Runtime adapters pass their validated construction-time version explicitly;
    forward does not require a current vLLM configuration context in that case.
    """
    if fa_version is not None and (type(fa_version) is not int or fa_version not in (2, 3)):
        raise ValueError("unsupported explicit FlashAttention version")
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
    version = fa_version if fa_version is not None else get_flash_attn_version(head_size=q.shape[2])
    options = {} if version is None else {"fa_version": version}
    result = flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                   max_seqlen_q=q.shape[0], max_seqlen_k=q.shape[0],
                                   softmax_scale=float(scale), causal=True, return_softmax_lse=True,
                                   **options)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("installed FlashAttention LSE return ABI differs from validated runtime")
    return normalize_flash_attention_state(*result, lse_layout="heads_tokens", log_base="natural")


@dataclass(frozen=True)
class PrefillRequest:
    """Per-request, already validated metadata from the runtime adapter.

    is_verification comes from execution semantics, never a length heuristic.
    CPU mirrors supply cached_len. No field requires a new device read.
    """
    q_len: int
    cached_len: int
    spec: KVSpec
    sm: tuple[int, int]
    raw_kv_valid: bool
    causal: bool
    is_verification: bool
    execution_mode: str
    sliding_window: int | None = None
    sinks: bool = False
    soft_cap: float = 0.
    alibi: bool = False
    multimodal_prefix_mask: bool = False
    kv_sharing: bool = False
    unknown_overlays: bool = False
    metadata_consistent: bool = True


def continuation_eligibility(request: PrefillRequest) -> RouteDecision:
    """Pure support gate; passing it alone never activates a tuned route."""
    conditions = (
        (request.metadata_consistent, "inconsistent_metadata"),
        (request.sm == (9, 0), "unsupported_sm"),
        (request.spec.dtype == "turboquant_k8v4" and request.spec.head_dim == 256 and
         request.spec.num_q_heads == request.spec.num_kv_heads * 6, "unsupported_geometry"),
        (type(request.q_len) is int and request.q_len >= 129, "short_query"),
        (type(request.cached_len) is int and request.cached_len > 0, "not_continuation"),
        (request.raw_kv_valid, "invalid_raw_kv"),
        (request.causal, "noncausal_attention"),
        (not request.is_verification, "verification_request"),
        (request.execution_mode in ("eager", "piecewise"), "unsupported_execution"),
        (request.sliding_window is None, "sliding_window"),
        (not request.sinks, "attention_sinks"),
        (request.soft_cap == 0., "soft_cap"),
        (not request.alibi, "alibi"),
        (not request.multimodal_prefix_mask, "multimodal_prefix_mask"),
        (not request.kv_sharing, "kv_sharing"),
        (not request.unknown_overlays, "unknown_overlays"),
    )
    for supported, reason in conditions:
        if not supported:
            return RouteDecision(False, reason)
    return RouteDecision(True, "supported_continuation")


@dataclass(frozen=True)
class PrefillRegion:
    """Inclusive measured region; units are tokens."""
    prefix_min: int
    prefix_max: int
    query_min: int
    query_max: int

    def __post_init__(self):
        if any(type(n) is not int for n in (self.prefix_min, self.prefix_max, self.query_min, self.query_max)):
            raise ValueError("profile bounds must be integers")
        if not (0 < self.prefix_min <= self.prefix_max and 129 <= self.query_min <= self.query_max):
            raise ValueError("profile bounds must describe eligible continuation lengths")

    def contains(self, request: PrefillRequest) -> bool:
        return (self.prefix_min <= request.cached_len <= self.prefix_max and
                self.query_min <= request.q_len <= self.query_max)


@dataclass(frozen=True)
class PrefillProfile:
    """Measured crossover supplied by runtime-keyed profile validation.

    runtime_key is the coordinator's identity including image, dependencies,
    device, model/dtype and geometry. evidence identifies the complete operation
    and TTFT run used to establish these winning regions. No default profile.
    """
    runtime_key: str
    regions: tuple[PrefillRegion, ...]
    evidence: str

    def __post_init__(self):
        if not isinstance(self.runtime_key, str) or not self.runtime_key:
            raise ValueError("profile requires a runtime key")
        if not isinstance(self.evidence, str) or not self.evidence:
            raise ValueError("profile requires measurement evidence")
        if not isinstance(self.regions, tuple) or any(not isinstance(r, PrefillRegion) for r in self.regions):
            raise ValueError("profile requires a tuple of measured regions")


def select_prefill_route(request: PrefillRequest, *, enabled: bool = False,
                        profile: PrefillProfile | None = None, runtime_key: str | None = None) -> RouteDecision:
    if not enabled:
        return RouteDecision(False, "disabled")
    support = continuation_eligibility(request)
    if not support.eligible:
        return support
    if profile is None:
        return RouteDecision(False, "missing_measured_profile")
    if profile.runtime_key != runtime_key:
        return RouteDecision(False, "profile_runtime_mismatch")
    if not any(region.contains(request) for region in profile.regions):
        return RouteDecision(False, "outside_measured_regions")
    return RouteDecision(True, "measured_winning_region")


def validated_request_lengths(query_start_loc_cpu, seq_lens_cpu, request_index: int,
                              total_tokens: int) -> tuple[int, int] | None:
    """Conservative PN401-safe lengths from complete CPU mirrors only.

    A fresh large request says nothing about a different shorter request. Absent,
    device-resident, malformed or mutually inconsistent mirrors return None so
    the adapter uses its captured conservative fallback without synchronization.
    """
    if query_start_loc_cpu is None or seq_lens_cpu is None:
        return None
    for mirror in (query_start_loc_cpu, seq_lens_cpu):
        device = getattr(mirror, "device", None)
        if device is not None and getattr(device, "type", "cpu") != "cpu":
            return None
    if type(request_index) is not int or type(total_tokens) is not int or total_tokens < 0:
        return None
    try:
        offsets = [operator.index(n) for n in query_start_loc_cpu]
        lengths = [operator.index(n) for n in seq_lens_cpu]
    except (TypeError, ValueError, RuntimeError):
        return None
    if not offsets or len(offsets) != len(lengths) + 1 or not 0 <= request_index < len(lengths):
        return None
    if offsets[0] != 0 or offsets[-1] != total_tokens:
        return None
    queries = [b - a for a, b in zip(offsets, offsets[1:])]
    if any(q <= 0 or n < q for q, n in zip(queries, lengths)):
        return None
    q_len = queries[request_index]
    return q_len, lengths[request_index] - q_len
