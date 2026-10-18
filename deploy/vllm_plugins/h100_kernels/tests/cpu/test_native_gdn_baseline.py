"""Explicit native route retains the exact baseline dependency/prefill stack."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from aquillm_vllm_h100 import bootstrap, prefill_profiles


def test_native_route_requires_the_explicit_profile_pair():
    config = bootstrap.settings({"AQUILLM_H100_GDN":"native-fp16",
                                 "AQUILLM_H100_RUNTIME_PROFILE":"native-gdn-baseline"})
    assert config["gdn"] == "native-fp16"
    assert bootstrap.settings({})["gdn"] == "baseline"


@pytest.mark.parametrize("gdn,profile", [("native-fp16","baseline"),
    ("native-fp16","flashinfer-0.6.18"),("baseline","native-gdn-baseline"),
    ("flashinfer","native-gdn-baseline")])
def test_native_mixed_profile_selector_pairs_reject(gdn, profile):
    with pytest.raises(ValueError, match="runtime|profile"):
        bootstrap.settings({"AQUILLM_H100_GDN":gdn,"AQUILLM_H100_RUNTIME_PROFILE":profile})


def test_native_profile_reuses_exact_baseline_prefill_identity(monkeypatch):
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", "baseline")
    baseline = prefill_profiles.development_profile()
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", "native-gdn-baseline")
    assert prefill_profiles.development_profile() is baseline
    assert baseline.runtime_key == prefill_profiles.RUNTIME_KEY


def test_native_prepare_validates_saved_alias_without_flashinfer_api(monkeypatch):
    from aquillm_vllm_h100.gdn import adapter
    from test_gdn_contract import _original
    qwen = SimpleNamespace(fused_sigmoid_gating_delta_rule_update=_original)
    monkeypatch.setitem(sys.modules, "aquillm_vllm_h100.gdn.native", SimpleNamespace())
    monkeypatch.setattr(adapter,"validate_api",lambda *a: pytest.fail("native route inspected FI public API"))
    module, call = adapter.prepare_native_install(qwen)
    assert module is qwen and call.__wrapped__ is _original
    assert call._aquillm_gdn_route == "native-fp16"
    qwen.fused_sigmoid_gating_delta_rule_update = call
    assert adapter.prepare_native_install(qwen)[1] is call
    qwen.fused_sigmoid_gating_delta_rule_update = lambda *args, **kw: None
    with pytest.raises(ValueError, match="signature"):
        adapter.prepare_native_install(qwen)


def test_native_setup_failure_precedes_any_runtime_mutation(monkeypatch):
    from aquillm_vllm_h100 import adapters, prefill_adapter
    from aquillm_vllm_h100.gdn import adapter
    parent = ModuleType("sndr.engines.vllm.kernels_legacy")
    parent.p67_multi_query_kernel = SimpleNamespace()
    monkeypatch.setitem(sys.modules,parent.__name__,parent)
    monkeypatch.setattr(prefill_profiles,"development_profile",lambda *a: SimpleNamespace(runtime_key="baseline"))
    monkeypatch.setattr(prefill_adapter,"install_prefill_adapter",lambda *a: pytest.fail("prefill changed before setup"))
    def fail():
        raise ValueError("native setup rejected")
    monkeypatch.setattr(adapter,"prepare_native_install",fail)
    with pytest.raises(ValueError,match="native setup"):
        adapters.install_adapters(dict(mtp="baseline",split="baseline",prefill="1",gdn="native-fp16",
                                       runtime_profile="native-gdn-baseline"))


def test_native_install_preserves_prefill_and_skips_fi_offset_wrapper(monkeypatch):
    from aquillm_vllm_h100 import adapters, prefill_adapter
    from aquillm_vllm_h100.gdn import adapter, prefill_compat
    parent = ModuleType("sndr.engines.vllm.kernels_legacy")
    parent.p67_multi_query_kernel = SimpleNamespace()
    monkeypatch.setitem(sys.modules,parent.__name__,parent)
    qwen, call = SimpleNamespace(fused_sigmoid_gating_delta_rule_update="old"), object()
    monkeypatch.setattr(adapter,"prepare_native_install",lambda: (qwen,call))
    monkeypatch.setattr(prefill_compat,"prepare_install",lambda: pytest.fail("baseline prefill wrapper requested"))
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE","native-gdn-baseline")
    def install_prefill(profile, key):
        assert key == prefill_profiles.RUNTIME_KEY and profile is prefill_profiles._PROFILE
        return dict(installed=True,reason="eligible")
    monkeypatch.setattr(prefill_adapter,"install_prefill_adapter",install_prefill)
    result = adapters.install_adapters(dict(mtp="baseline",split="baseline",prefill="1",gdn="native-fp16",
                                           runtime_profile="native-gdn-baseline"))
    assert qwen.fused_sigmoid_gating_delta_rule_update is call
    assert result["gdn"] == "native-fp16" and result["prefill"]
    assert not result["gdn_prefill_int64_offsets"]


def test_native_first_route_log_identifies_selector_precision_and_real_layout(monkeypatch,caplog):
    from aquillm_vllm_h100.gdn import adapter
    from test_gdn_contract import _original, _call_metadata
    args = _call_metadata()
    monkeypatch.setattr(adapter,"_h100",lambda *a:True)
    monkeypatch.setitem(sys.modules,"aquillm_vllm_h100.gdn.native",
                        SimpleNamespace(launch=lambda **kw: args["v"]))
    module = SimpleNamespace(fused_sigmoid_gating_delta_rule_update=_original)
    _,call = adapter.prepare_native_install(module)
    call(**args)
    message = caplog.records[-1].getMessage()
    assert "route_exercised gdn=native-fp16" in message and "precision=fp16" in message
    assert "gate_strides=" in message and "state_strides=" in message
