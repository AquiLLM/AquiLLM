"""Breaks caught: enabling an unqualified API, or missing rollback controls."""
from types import SimpleNamespace

import pytest


def inspect(module):
    try:
        from aquillm_vllm_h100.gdn.capability import inspect as inspect_api
    except ModuleNotFoundError:
        pytest.fail("missing GDN capability gate")
    return inspect_api(module)


def decode(q, k, v, state, A_log, a, dt_bias, b, scale=None,
           output=None, use_qk_l2norm=True):
    raise AssertionError("inspection must never execute a kernel")


def mtp(q, k, v, initial_state, initial_state_indices, A_log, a, dt_bias, b,
        scale=None, output=None, intermediate_states_buffer=None,
        disable_state_update=None, use_qk_l2norm=True):
    raise AssertionError("inspection must never execute a kernel")


def test_installed_shape_of_api_does_not_authorize_unqualified_state_mutation():
    result = inspect(SimpleNamespace(gated_delta_rule_decode=decode,
                                     gated_delta_rule_mtp=mtp))
    assert not result.eligible
    assert "T5" in result.reason and "checkpoint" in result.reason


def test_missing_mtp_api_retains_baseline():
    result = inspect(SimpleNamespace(gated_delta_rule_decode=decode))
    assert not result.eligible
    assert "gated_delta_rule_mtp" in result.reason


def test_mtp_without_explicit_state_update_control_retains_baseline():
    def unsafe(q, k, v, initial_state, initial_state_indices, A_log, a, dt_bias, b,
               intermediate_states_buffer=None):
        raise AssertionError("must not run")

    result = inspect(SimpleNamespace(gated_delta_rule_decode=decode,
                                     gated_delta_rule_mtp=unsafe))
    assert not result.eligible
    assert "disable_state_update" in result.reason


def test_mtp_without_checkpoint_buffer_retains_baseline():
    def unsafe(q, k, v, initial_state, initial_state_indices, A_log, a, dt_bias, b,
               disable_state_update=None):
        raise AssertionError("must not run")

    result = inspect(SimpleNamespace(gated_delta_rule_decode=decode,
                                     gated_delta_rule_mtp=unsafe))
    assert not result.eligible
    assert "intermediate_states_buffer" in result.reason


def _original(A_log, a, b, dt_bias, q, k, v, beta=1.0, threshold=20.0,
              scale=None, initial_state=None, inplace_final_state=True,
              cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
              use_qk_l2norm_in_kernel=False, is_kda=False):
    return "original", initial_state


def _call_metadata():
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        def tensor(shape, dtype=torch.float16):
            return torch.empty(shape, dtype=dtype, device="cuda")
        return dict(A_log=tensor((48,), torch.float32), a=tensor((1, 5, 48)),
                    b=tensor((1, 5, 48)), dt_bias=tensor((48,)),
                    q=tensor((1, 5, 16, 128)), k=tensor((1, 5, 16, 128)),
                    v=tensor((1, 5, 48, 128)),
                    initial_state=tensor((9, 48, 128, 128), torch.float32),
                    cu_seqlens=tensor((2,), torch.int32),
                    ssm_state_indices=tensor((1, 7), torch.int32),
                    num_accepted_tokens=tensor((1,), torch.int32))


def test_gdn_requires_explicit_candidate_runtime():
    from aquillm_vllm_h100.bootstrap import settings
    with pytest.raises(ValueError, match="runtime"):
        settings({"AQUILLM_H100_GDN": "flashinfer"})
    config = settings({"AQUILLM_H100_GDN": "flashinfer",
                       "AQUILLM_H100_RUNTIME_PROFILE": "flashinfer-0.6.18"})
    assert config["gdn"] == "flashinfer"


def test_candidate_contract_is_shape_only_and_accepts_padded_slot_stride(monkeypatch):
    import torch
    from aquillm_vllm_h100.gdn import adapter
    monkeypatch.setattr(adapter, "_h100", lambda device: True)
    args = _call_metadata()
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        args["initial_state"] = torch.empty_strided(
            (9, 48, 128, 128), (48*128*128+128, 128*128, 128, 1),
            device="cuda", dtype=torch.float32)
    assert adapter.supported(**args)


@pytest.mark.parametrize("change", ["T1", "N2", "bf16", "bad_state_stride",
                                   "index_stride", "no_acceptance", "beta", "scale"])
def test_unqualified_calls_use_saved_original_without_candidate_launch(monkeypatch, change):
    import torch
    from aquillm_vllm_h100.gdn import adapter
    monkeypatch.setattr(adapter, "_h100", lambda device: True)
    args = _call_metadata()
    if change == "T1":
        args["q"] = args["q"][:, :1]
    elif change == "N2":
        args["cu_seqlens"] = args["cu_seqlens"].new_empty(3)
    elif change == "bf16":
        args["v"] = args["v"].to(torch.bfloat16)
    elif change == "bad_state_stride":
        args["initial_state"] = args["initial_state"].transpose(2, 3)
    elif change == "index_stride":
        args["ssm_state_indices"] = args["ssm_state_indices"][:, ::2]
    elif change == "no_acceptance":
        args["num_accepted_tokens"] = None
    elif change == "beta":
        args["beta"] = 2
    elif change == "scale":
        args["scale"] = float("nan")
    result = adapter.make_adapter(_original, lambda **kw: pytest.fail("candidate launched"))(**args)
    assert result[0] == "original"
    assert result[1] is args["initial_state"]


def test_alias_registration_preserves_signature_and_is_idempotent(monkeypatch):
    import sys
    import inspect as pyinspect
    from aquillm_vllm_h100.gdn import adapter
    module = SimpleNamespace(fused_sigmoid_gating_delta_rule_update=_original)
    candidate = SimpleNamespace(gated_delta_rule_mtp=lambda **kw: None)
    monkeypatch.setattr(adapter, "validate_api", lambda module: None)
    # CuTe is an external CUDA-only dependency unavailable on CPU CI.
    monkeypatch.setitem(sys.modules, "aquillm_vllm_h100.gdn.native", SimpleNamespace())
    adapter.install_adapter(module, candidate)
    installed = module.fused_sigmoid_gating_delta_rule_update
    assert pyinspect.signature(installed) == pyinspect.signature(_original)
    assert installed.__wrapped__ is _original
    adapter.install_adapter(module, candidate)
    assert module.fused_sigmoid_gating_delta_rule_update is installed


def test_failed_api_setup_preserves_original_alias(monkeypatch):
    from aquillm_vllm_h100.gdn import adapter
    module = SimpleNamespace(fused_sigmoid_gating_delta_rule_update=_original)
    def reject(module):
        raise ValueError("missing dense checkpoints")
    monkeypatch.setattr(adapter, "validate_api", reject)
    with pytest.raises(ValueError, match="checkpoints"):
        adapter.install_adapter(module, SimpleNamespace())
    assert module.fused_sigmoid_gating_delta_rule_update is _original


def test_gdn_only_bootstrap_runs_and_failed_setup_never_reports_installed(monkeypatch):
    from aquillm_vllm_h100 import bootstrap, adapters, compatibility
    monkeypatch.setattr(bootstrap, "_installed", False)
    monkeypatch.setattr(compatibility, "verify_runtime", lambda: None)
    def reject(config):
        raise ValueError("GDN setup rejected")
    monkeypatch.setattr(adapters, "install_adapters", reject)
    result = bootstrap.install({"AQUILLM_H100_GDN": "flashinfer",
                                "AQUILLM_H100_RUNTIME_PROFILE": "flashinfer-0.6.18"})
    assert result["status"] == "inactive"
    assert not bootstrap._installed


def test_gdn_setup_failure_occurs_before_prefill_installation(monkeypatch):
    import sys
    from types import ModuleType
    from aquillm_vllm_h100 import adapters, prefill_profiles, prefill_adapter
    from aquillm_vllm_h100.gdn import adapter
    parent = ModuleType("sndr.engines.vllm.kernels_legacy")
    parent.p67_multi_query_kernel = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "sndr.engines.vllm.kernels_legacy", parent)
    monkeypatch.setattr(prefill_profiles, "development_profile", lambda name: SimpleNamespace(runtime_key="candidate"))
    monkeypatch.setattr(prefill_adapter, "install_prefill_adapter", lambda *a: pytest.fail("prefill mutated before GDN validation"))
    def reject():
        raise ValueError("GDN setup rejected")
    monkeypatch.setattr(adapter, "prepare_install", reject)
    with pytest.raises(ValueError, match="GDN"):
        adapters.install_adapters(dict(mtp="baseline", split="baseline", prefill="1", gdn="flashinfer",
                                       runtime_profile="flashinfer-0.6.18"))


def test_candidate_failure_terminates_worker_without_retrying_original(monkeypatch):
    from aquillm_vllm_h100.gdn import adapter
    args = _call_metadata()
    monkeypatch.setattr(adapter, "_h100", lambda device: True)
    def fallback(**kw):
        pytest.fail("baseline retried after candidate failure")
    def fail_launch(**kw):
        assert kw["initial_state"] is args["initial_state"]
        assert kw["ssm_state_indices"] is args["ssm_state_indices"]
        assert kw["dt_bias"] is args["dt_bias"]
        raise RuntimeError("injected launch error")
    with pytest.raises(SystemExit, match="injected launch error"):
        adapter.make_adapter(fallback, fail_launch)(**args)


def test_registration_loads_actual_nested_qwen_module_and_retains_its_alias(monkeypatch):
    import sys
    from types import ModuleType
    from aquillm_vllm_h100.gdn import adapter
    names = ("vllm", "vllm.model_executor", "vllm.model_executor.layers",
             "vllm.model_executor.layers.mamba", "vllm.model_executor.layers.mamba.gdn")
    for name in names:
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    qwen = ModuleType(names[-1] + ".qwen_gdn_linear_attn")
    qwen.fused_sigmoid_gating_delta_rule_update = _original
    monkeypatch.setitem(sys.modules, qwen.__name__, qwen)
    monkeypatch.setitem(sys.modules, "aquillm_vllm_h100.gdn.native", SimpleNamespace())
    monkeypatch.setattr(adapter, "validate_api", lambda module: None)
    module, installed = adapter.prepare_install(candidate=SimpleNamespace(gated_delta_rule_mtp=lambda **kw: None))
    assert module is qwen
    assert installed.__wrapped__ is _original


def test_qualification_captures_saved_post_genesis_alias_even_when_installed():
    from aquillm_vllm_h100.gdn import adapter
    qwen = SimpleNamespace(fused_sigmoid_gating_delta_rule_update=_original)
    assert adapter.capture_original(qwen) is _original
    qwen.fused_sigmoid_gating_delta_rule_update = adapter.make_adapter(_original)
    assert adapter.capture_original(qwen) is _original


def test_eligible_native_route_has_no_packing_or_parameter_conversion(monkeypatch, caplog):
    import sys
    from aquillm_vllm_h100.gdn import adapter
    args = _call_metadata()
    monkeypatch.setattr(adapter, "_h100", lambda device: True)
    def native_launch(**kw):
        assert kw["q"] is args["q"]
        assert kw["a"] is args["a"]
        assert kw["dt_bias"] is args["dt_bias"]
        assert kw["initial_state"] is args["initial_state"]
        return args["v"]
    monkeypatch.setitem(sys.modules, "aquillm_vllm_h100.gdn.native", SimpleNamespace(launch=native_launch))
    monkeypatch.setitem(sys.modules, "aquillm_vllm_h100.gdn.kernels", SimpleNamespace(
        prepare=lambda **kw: pytest.fail("native route must not pack operands"),
        scatter_and_unpack=lambda *a: None))
    output,state = adapter.make_adapter(_original)(**args)
    assert output is args["v"]
    assert state is args["initial_state"]
    message = caplog.records[-1].getMessage()
    assert "precision=fp16" in message
    assert "gate_strides=" in message and "state_strides=" in message


def test_native_precision_variants_compile_separately_without_gpu_dependencies():
    import ast
    import pathlib
    import threading
    from aquillm_vllm_h100.gdn import adapter
    # Execute the real host launcher with every allocation, descriptor, stream
    # and compiler dependency stubbed. No CuTe import or CUDA pointer is needed.
    source = pathlib.Path(adapter.__file__).with_name("native.py").read_text()
    launch_ast = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.FunctionDef) and node.name == "launch")
    compilations, launches = [], []
    def compile_kernel(*args, **kwargs):
        compilations.append(kwargs)
        return lambda *values: launches.append(values)
    class Tensor:
        ndim = 3
        shape = (1,5,48,128)
        device = "cuda:0"
        dtype = "fp16"
        def stride(self):
            return (5*48*128,48*128,128,1)
    state = Tensor()
    state.shape, state.dtype = (9,48,128,128), "fp32"
    indices = Tensor()
    indices.shape = (1,7)
    namespace = dict(_compiled={}, _compile_lock=threading.Lock(), _entry=object(),
                     torch=SimpleNamespace(float16="fp16", empty=lambda *a, **kw: Tensor(),
                         cuda=SimpleNamespace(current_stream=lambda device: SimpleNamespace(cuda_stream=123))),
                     cuda=SimpleNamespace(CUstream=lambda value: value),
                     from_dlpack=lambda *a, **kw: SimpleNamespace(mark_layout_dynamic=lambda: object()),
                     cute=SimpleNamespace(compile=compile_kernel))
    exec(compile(ast.Module(body=[launch_ast], type_ignores=[]), "native.py", "exec"), namespace)
    kwargs = {name:Tensor() for name in ("A_log","a","b","dt_bias","q","k","v",
                                       "cu_seqlens","num_accepted_tokens")}
    kwargs.update(initial_state=state,ssm_state_indices=indices)
    launch = namespace["launch"]
    launch(**kwargs)
    launch(**kwargs, _operand_precision="bf16")
    launch(**kwargs)
    assert [entry["ROUND_BF16"] for entry in compilations] == [False, True]
    assert len(namespace["_compiled"]) == 2 and len(launches) == 3
    with pytest.raises(ValueError, match="precision"):
        launch(**kwargs, _operand_precision="fp32")


@pytest.mark.parametrize("layout", ["inner_stride", "misaligned_offset", "misaligned_row"])
def test_native_unaligned_or_unpacked_inputs_fall_back_before_launch(monkeypatch, layout):
    import sys
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from aquillm_vllm_h100.gdn import adapter
    args = _call_metadata()
    monkeypatch.setitem(sys.modules, "aquillm_vllm_h100.gdn.kernels", SimpleNamespace(
        prepare=lambda **kw: pytest.fail("unsupported layout entered GPU preparation"),
        scatter_and_unpack=lambda *a: None))
    monkeypatch.setattr(adapter, "_h100", lambda device: True)
    with FakeTensorMode():
        if layout == "inner_stride":
            args["q"] = torch.empty((1,5,16,256),device="cuda",dtype=torch.float16)[...,::2]
        elif layout == "misaligned_offset":
            args["q"] = torch.empty(1+5*16*128,device="cuda",dtype=torch.float16)[1:].view(1,5,16,128)
        else:
            args["q"] = torch.empty_strided((1,5,16,128),(10300,2060,128,1),device="cuda",dtype=torch.float16)
    assert adapter.make_adapter(_original, lambda **kw: pytest.fail("native launched"))(**args)[0] == "original"
