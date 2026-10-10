"""The per-request policy protects PN401 and refuses unmeasured dispatch."""
from dataclasses import replace

import pytest

from aquillm_vllm_h100.contracts import KVSpec


def request(**changes):
    from aquillm_vllm_h100.prefill import PrefillRequest
    base = PrefillRequest(129, 4096, KVSpec("turboquant_k8v4", 256, 24, 4, 32, 256, 128),
                         sm=(9, 0), raw_kv_valid=True, causal=True, is_verification=False,
                         execution_mode="eager")
    return replace(base, **changes)


@pytest.mark.parametrize("length", [129, 256, 512, 2048, 8192])
def test_ordinary_continuations_are_eligible_even_beyond_p101_32768(length):
    from aquillm_vllm_h100.prefill import continuation_eligibility
    assert continuation_eligibility(request(q_len=length, cached_len=65536)).eligible


@pytest.mark.parametrize("changes", [
    {"q_len": 128}, {"q_len": 5}, {"cached_len": 0}, {"raw_kv_valid": False},
    {"sm": (8, 9)}, {"causal": False}, {"is_verification": True},
    {"execution_mode": "full_graph"}, {"sliding_window": 4096}, {"sinks": True},
    {"soft_cap": 50.}, {"alibi": True}, {"multimodal_prefix_mask": True},
    {"kv_sharing": True}, {"unknown_overlays": True}, {"metadata_consistent": False},
    {"spec": KVSpec("turboquant_k8v4", 128, 24, 4, 32, 128, 64)},
    {"spec": KVSpec("turboquant_k8v4", 256, 16, 4, 32, 256, 128)},
])
def test_unsupported_attention_semantics_are_refused(changes):
    from aquillm_vllm_h100.prefill import continuation_eligibility
    assert not continuation_eligibility(request(**changes)).eligible


def test_large_verification_is_not_classified_by_length():
    from aquillm_vllm_h100.prefill import continuation_eligibility
    assert not continuation_eligibility(request(q_len=512, is_verification=True)).eligible


def test_no_profile_never_invents_a_winning_crossover():
    from aquillm_vllm_h100.prefill import select_prefill_route
    assert not select_prefill_route(request()).eligible
    assert not select_prefill_route(request(), enabled=True).eligible


def test_profile_only_routes_exact_measured_regions_and_runtime():
    from aquillm_vllm_h100.prefill import PrefillProfile, PrefillRegion, select_prefill_route
    profile = PrefillProfile("runtime-a", (PrefillRegion(4096, 16384, 129, 512),), "run-verified-a")
    assert select_prefill_route(request(), enabled=True, profile=profile, runtime_key="runtime-a").eligible
    assert not select_prefill_route(request(cached_len=32768), enabled=True, profile=profile, runtime_key="runtime-a").eligible
    assert not select_prefill_route(request(), enabled=True, profile=profile, runtime_key="runtime-b").eligible
    assert not select_prefill_route(request(), enabled=False, profile=profile, runtime_key="runtime-a").eligible


def test_profile_without_measurement_evidence_is_rejected():
    from aquillm_vllm_h100.prefill import PrefillProfile, PrefillRegion
    with pytest.raises(ValueError, match="evidence"):
        PrefillProfile("runtime-a", (PrefillRegion(4096, 16384, 129, 512),), "")


def test_pn401_mixed_batch_mirrors_preserve_shorter_continuation_prefix():
    from aquillm_vllm_h100.prefill import validated_request_lengths
    offsets = [0, 1, 513, 642]
    seq_lens = [1025, 512, 4096 + 129]
    assert validated_request_lengths(offsets, seq_lens, 0, 642) == (1, 1024)
    assert validated_request_lengths(offsets, seq_lens, 1, 642) == (512, 0)
    assert validated_request_lengths(offsets, seq_lens, 2, 642) == (129, 4096)


@pytest.mark.parametrize("offsets,lens,index,total", [
    (None, [512], 0, 512), ([0, 512], None, 0, 512),
    ([0, 512], [511], 0, 512), ([0, 512, 511], [512, 512], 0, 511),
    ([1, 513], [512], 0, 513), ([0, 512], [512], 1, 512),
    ([0, 512], [512], 0, 513), ([0, 512], [512, 4096], 0, 512),
])
def test_absent_or_inconsistent_mirrors_keep_conservative_fallback(offsets, lens, index, total):
    from aquillm_vllm_h100.prefill import validated_request_lengths
    assert validated_request_lengths(offsets, lens, index, total) is None
