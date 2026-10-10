"""Upgraded runtimes must not reuse baseline prefill qualification identity."""
import pytest

from aquillm_vllm_h100 import prefill_profiles
from aquillm_vllm_h100.contracts import KVSpec
from aquillm_vllm_h100.prefill import PrefillRequest, select_prefill_route


def request(prefix=32768, query=1024):
    return PrefillRequest(query, prefix,
                         KVSpec("turboquant_k8v4", 256, 24, 4, 16, 256, 128),
                         (9, 0), True, True, False, "eager")


@pytest.mark.parametrize("name", [None, "h100-long-prefill-dev-v1"])
def test_candidate_cannot_reuse_baseline_prefill_identity(monkeypatch, name):
    monkeypatch.delenv("AQUILLM_H100_RUNTIME_PROFILE", raising=False)
    baseline = prefill_profiles.development_profile(name)
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", "flashinfer-0.6.18")
    candidate = prefill_profiles.development_profile(name)

    decision = select_prefill_route(request(), enabled=True, profile=candidate,
                                   runtime_key=baseline.runtime_key)
    assert not decision.eligible
    assert decision.reason == "profile_runtime_mismatch"
    decision = select_prefill_route(request(), enabled=True, profile=baseline,
                                   runtime_key=candidate.runtime_key)
    assert not decision.eligible
    assert decision.reason == "profile_runtime_mismatch"


def test_candidate_evidence_is_explicitly_pending_serving(monkeypatch):
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", "flashinfer-0.6.18")
    profile = prefill_profiles.development_profile()
    assert profile.evidence.startswith("experimental_pending_serving:flashinfer-0.6.18")


@pytest.mark.parametrize("selector", ["typo", "", "flashinfer-0.6.19"])
def test_unknown_runtime_cannot_select_baseline_prefill(monkeypatch, selector):
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", selector)
    with pytest.raises(ValueError, match="runtime profile"):
        prefill_profiles.development_profile()


@pytest.mark.parametrize("selector", [None, "baseline"])
def test_baseline_identity_restored_after_candidate_selection(monkeypatch, selector):
    monkeypatch.delenv("AQUILLM_H100_RUNTIME_PROFILE", raising=False)
    original = prefill_profiles.development_profile()
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", "flashinfer-0.6.18")
    prefill_profiles.development_profile()
    if selector is None:
        monkeypatch.delenv("AQUILLM_H100_RUNTIME_PROFILE")
    else:
        monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", selector)
    restored = prefill_profiles.development_profile()
    assert select_prefill_route(request(), enabled=True, profile=restored,
                               runtime_key=original.runtime_key).eligible
    assert restored.evidence == original.evidence


@pytest.mark.parametrize("prefix,query,eligible", [
    (32768, 1024, True), (65536, 4096, True),
    (32767, 1024, False), (65537, 1024, False),
    (32768, 1023, False), (32768, 4097, False),
])
def test_candidate_retains_bounded_experimental_prefill_region(monkeypatch, prefix, query, eligible):
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", "flashinfer-0.6.18")
    profile = prefill_profiles.development_profile()
    current = request(prefix, query)
    assert select_prefill_route(current, enabled=True, profile=profile,
                               runtime_key=profile.runtime_key).eligible is eligible
    assert not select_prefill_route(current, profile=profile,
                                   runtime_key=profile.runtime_key).eligible


def test_candidate_still_rejects_unknown_prefill_profile_name(monkeypatch):
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", "flashinfer-0.6.18")
    with pytest.raises(ValueError, match="prefill profile"):
        prefill_profiles.development_profile("unmeasured")
