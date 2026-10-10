"""Breaks caught: opt-in leaks, wrong runtime geometry, and stale model identity."""
import importlib
import sys
from types import SimpleNamespace

import pytest


def profiles():
    try:
        return importlib.import_module("aquillm_vllm_h100.prefill_profiles")
    except ModuleNotFoundError:
        pytest.fail("missing bounded development prefill profile")


def test_bundled_profile_only_covers_measured_long_prefix_region():
    from aquillm_vllm_h100.contracts import KVSpec
    from aquillm_vllm_h100.prefill import PrefillRequest, select_prefill_route
    profile = profiles().development_profile(None)
    spec = KVSpec("turboquant_k8v4", 256, 24, 4, 2128, 256, 128)
    for prefix, query, want in ((32768, 1024, True), (65536, 4096, True),
                                (8192, 4096, False), (32767, 4096, False),
                                (65537, 1024, False), (65536, 1023, False),
                                (32768, 4097, False)):
        request = PrefillRequest(query, prefix, spec, (9, 0), True, True, False, "eager")
        assert select_prefill_route(request, enabled=True, profile=profile, runtime_key=profile.runtime_key).eligible is want
        assert not select_prefill_route(request, profile=profile, runtime_key=profile.runtime_key).eligible


def test_unknown_profile_cannot_silently_select_default():
    with pytest.raises(ValueError, match="profile"):
        profiles().development_profile("unmeasured")


def baseline_module(monkeypatch):
    original = lambda **kwargs: "baseline"
    module = SimpleNamespace(call_p67_splitk=original)
    monkeypatch.setitem(sys.modules, "sndr.engines.vllm.kernels_legacy", SimpleNamespace(p67_multi_query_kernel=module))
    return module, original


@pytest.mark.parametrize("mtp", ["baseline", "fused"])
def test_optin_installs_prefill_independently_of_mtp_without_cuda(monkeypatch, mtp):
    import aquillm_vllm_h100.prefill_adapter as prefill_adapter
    import aquillm_vllm_h100.adapters as adapters
    import torch
    module, original = baseline_module(monkeypatch)
    captured = []
    monkeypatch.setattr(prefill_adapter, "install_prefill_adapter", lambda profile, key:
                        captured.append((profile, key)) or {"installed": True, "reason": "test_boundary"})
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args: pytest.fail("registration initialized CUDA"))
    result = adapters.install_adapters(dict(mtp=mtp, split="baseline", prefill="1", profile=None))
    assert result["prefill"] is True
    assert captured[0][0].regions[0].prefix_min == 32768
    assert captured[0][0].runtime_key == captured[0][1]
    assert (module.call_p67_splitk is original) is (mtp == "baseline")


def test_prefill_off_does_not_select_or_install_default_profile(monkeypatch):
    import aquillm_vllm_h100.adapters as adapters
    import aquillm_vllm_h100.prefill_adapter as prefill_adapter
    import aquillm_vllm_h100.prefill_profiles as profile_module
    module, original = baseline_module(monkeypatch)
    monkeypatch.setattr(prefill_adapter, "install_prefill_adapter", lambda *args: pytest.fail("prefill disabled"))
    monkeypatch.setattr(profile_module, "development_profile", lambda *args: pytest.fail("disabled prefill selected default profile"))
    result = adapters.install_adapters(dict(mtp="baseline", split="baseline", prefill="0", profile=None))
    assert result["prefill"] is False and module.call_p67_splitk is original


def test_unknown_profile_rejected_before_any_adapter_mutation(monkeypatch):
    import aquillm_vllm_h100.adapters as adapters
    module, original = baseline_module(monkeypatch)
    with pytest.raises(ValueError, match="profile"):
        adapters.install_adapters(dict(mtp="fused", split="baseline", prefill="1", profile="typo"))
    assert module.call_p67_splitk is original


def constructor():
    class Impl:
        def __init__(self, num_heads, head_size, scale, num_kv_heads=None, alibi_slopes=None,
                     sliding_window=None, kv_cache_dtype="auto", logits_soft_cap=None,
                     attn_type="decoder", kv_sharing_target_layer_name=None, **kwargs):
            self.initialized = (num_heads, head_size, scale)
    return Impl


@pytest.mark.parametrize("revision", ["2d783431e303148fc6e16622fac5edac83a6b5c4", None, "wrong"])
def test_constructor_uses_resolved_commit_hash_even_when_requested_revision_is_none(monkeypatch, revision):
    from aquillm_vllm_h100.prefill_adapter import _capture_constructor
    config = SimpleNamespace(model_config=SimpleNamespace(revision=None, hf_config=SimpleNamespace(_commit_hash=revision)),
                             attention_config=SimpleNamespace(flash_attn_version=2))
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(get_current_vllm_config=lambda: config))
    Impl = constructor()
    Impl.__init__ = _capture_constructor(Impl.__init__)
    impl = Impl(24, 256, 0.0625, 4)
    assert impl.initialized == (24, 256, 0.0625)
    assert "model_revision" in impl._aquillm_h100_semantics
    assert impl._aquillm_h100_semantics.get("model_revision") == revision


@pytest.mark.parametrize("version", [2, 3, None])
def test_constructor_snapshots_explicit_flash_attention_version(monkeypatch, version):
    from aquillm_vllm_h100.prefill_adapter import _capture_constructor
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(_commit_hash=profiles().MODEL_REVISION)),
                             attention_config=SimpleNamespace(flash_attn_version=version))
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(get_current_vllm_config=lambda: config))
    Impl = constructor()
    Impl.__init__ = _capture_constructor(Impl.__init__)
    impl = Impl(24, 256, 0.0625, 4)
    assert "flash_attn_version" in impl._aquillm_h100_semantics
    assert impl._aquillm_h100_semantics["flash_attn_version"] == version


@pytest.mark.parametrize("version", [None, 3])
def test_unqualified_fa_version_falls_back_before_tensor_access(version):
    from aquillm_vllm_h100.prefill_adapter import _make_route
    impl = SimpleNamespace(_aquillm_h100_semantics={"model_revision": profiles().MODEL_REVISION,
                                                  "flash_attn_version": version})
    class Query:
        @property
        def ndim(self):
            pytest.fail("unqualified FA version reached tensor boundary")
    metadata = SimpleNamespace(is_prefill=True)
    assert _make_route(None, "unit")(impl, None, Query(), None, None, None, metadata, 0, 1024, None) is None


@pytest.mark.parametrize("revision", [None, "wrong"])
def test_unqualified_model_revision_falls_back_before_tensor_or_cuda_access(revision):
    from aquillm_vllm_h100.prefill_adapter import _make_route
    impl = SimpleNamespace(_aquillm_h100_semantics={"model_revision": revision})
    route = _make_route(None, "unit")
    class Query:
        shape = (1024, 24, 256)
        @property
        def is_cuda(self):
            pytest.fail("unqualified revision reached CUDA boundary")
    metadata = SimpleNamespace(is_prefill=True, query_start_loc_cpu=[0, 1024], seq_lens_cpu=[33792])
    assert route(impl, None, Query(), None, None, None, metadata, 0, 1024, None) is None


def test_adaptive_remains_blocked_when_prefill_is_selected(monkeypatch):
    import aquillm_vllm_h100.adapters as adapters
    module, original = baseline_module(monkeypatch)
    with pytest.raises(ValueError, match="adaptive"):
        adapters.install_adapters(dict(mtp="fused", split="adaptive", prefill="1", profile=None))
    assert module.call_p67_splitk is original


@pytest.mark.parametrize("change", [None, "dtype", "sm", "sm_count", "gpu_name", "page", "heads", "dimension"])
def test_profile_runtime_geometry_rejects_unmeasured_device_dtype_and_page(change):
    from aquillm_vllm_h100.contracts import KVSpec
    fields = dict(dtype="turboquant_k8v4", head_dim=256, num_q_heads=24, num_kv_heads=4,
                  block_size=2128, key_packed_size=256, value_data_bytes=128)
    props = SimpleNamespace(major=9, minor=0, multi_processor_count=132, name="NVIDIA H100 80GB HBM3")
    dtype = "torch.float16"
    if change == "dtype":
        dtype = "torch.bfloat16"
    elif change == "sm":
        props.major = 10
    elif change == "sm_count":
        props.multi_processor_count = 114
    elif change == "gpu_name":
        props.name = "NVIDIA A100 80GB"
    elif change == "page":
        fields["block_size"] = 32
    elif change == "heads":
        fields.update(num_q_heads=48, num_kv_heads=8)
    elif change == "dimension":
        fields.update(head_dim=128, key_packed_size=128, value_data_bytes=64)
    assert profiles().matches_runtime(KVSpec(**fields), dtype, props) is (change is None)


@pytest.mark.parametrize("change", [None, "dtype", "gpu_name", "sm_count", "page", "heads", "destination_shape", "destination_dtype", "device", "zero_stride", "outside_region", "capture"])
def test_actual_worker_route_preserves_fallback_and_logs_success_once(monkeypatch, caplog, change):
    import aquillm_vllm_h100.prefill_adapter as adapter
    device = SimpleNamespace(type="cuda", index=0)
    class Tensor:
        def __init__(self, shape, dtype, *, strides=None):
            self.shape, self.dtype, self.device = shape, dtype, device
            self.ndim, self.is_cuda, self.value = len(shape), True, "untouched"
            self.strides = strides or tuple(1 for _ in shape)
        def stride(self):
            return self.strides
        def __getitem__(self, index):
            return Tensor(self.shape[1:], self.dtype)
    def no_allocation(*args, **kwargs):
        pytest.fail("unsupported route allocated GPU scratch")
    properties = SimpleNamespace(major=9, minor=0, multi_processor_count=132, name="NVIDIA H100 80GB HBM3")
    fake_torch = SimpleNamespace(Tensor=Tensor, uint8="torch.uint8", float16="torch.float16",
        bfloat16="torch.bfloat16", float32="torch.float32", int32="torch.int32", int64="torch.int64",
        cuda=SimpleNamespace(is_current_stream_capturing=lambda: change == "capture",
                             get_device_capability=lambda _: (9, 0)),
        empty_like=no_allocation, empty=no_allocation)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)  # GPU allocation boundary is unavailable on CPU.
    monkeypatch.setattr(adapter, "_worker_properties", lambda _: properties)
    profile = profiles().development_profile()
    impl = SimpleNamespace(kv_cache_dtype="turboquant_k8v4", scale=0.0625, _val_data_bytes=128,
        tq_config=SimpleNamespace(key_fp8=True, effective_value_quant_bits=4, key_packed_size=256),
        _aquillm_h100_semantics=dict(model_revision=profiles().MODEL_REVISION, flash_attn_version=2, causal=True,
            sliding_window=None, soft_cap=0., alibi=False, kv_sharing=False, unknown_overlays=False))
    q, k, v = Tensor((1024, 24, 256), "torch.float16"), Tensor((1024, 4, 256), "torch.float16"), Tensor((1024, 4, 256), "torch.float16")
    destination = Tensor(q.shape, q.dtype)
    cache = Tensor((17, 2128, 4, 388), "torch.uint8")
    metadata = SimpleNamespace(is_prefill=True, query_start_loc_cpu=[0, 1024], seq_lens_cpu=[33792],
                               block_table=Tensor((1, 17), "torch.int32"))
    if change == "dtype":
        q.dtype = k.dtype = v.dtype = destination.dtype = "torch.bfloat16"
    elif change == "gpu_name":
        properties.name = "NVIDIA A100 80GB"
    elif change == "sm_count":
        properties.multi_processor_count = 114
    elif change == "page":
        cache.shape = (17, 32, 4, 388)
    elif change == "heads":
        q.shape, k.shape, v.shape = (1024, 48, 256), (1024, 8, 256), (1024, 8, 256)
    elif change == "destination_shape":
        destination.shape = (512, 24, 256)
    elif change == "destination_dtype":
        destination.dtype = "torch.bfloat16"
    elif change == "device":
        cache.device = SimpleNamespace(type="cuda", index=1)
    elif change == "zero_stride":
        destination.strides = (0, 1, 1)
    elif change == "outside_region":
        metadata.seq_lens_cpu = [9216]
    route = adapter._make_route(profile, profile.runtime_key)
    if change is None:
        active_config = [SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(_commit_hash=profiles().MODEL_REVISION)),
            attention_config=SimpleNamespace(flash_attn_version=2))]
        def current_config():
            assert active_config[0] is not None, "forward queried an ended construction context"
            return active_config[0]
        monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(get_current_vllm_config=current_config))
        Constructed = constructor()
        Constructed.__init__ = adapter._capture_constructor(Constructed.__init__)
        initialized = Constructed(24, 256, .0625, 4)
        impl._aquillm_h100_semantics = initialized._aquillm_h100_semantics
        active_config[0] = None  # Normal serving forward runs after construction context ends.
        fake_torch.empty_like = lambda tensor, dtype: Tensor(tensor.shape, dtype)
        fake_torch.empty = lambda shape, dtype, device: Tensor(shape, dtype)
        def prefix(q, cache, table, prior, scale, spec, state):
            assert prior == 32768 and spec.block_size == 2128
            state.output.value = 12
        def chunk(q, k, v, scale, *, fa_version=None):
            assert fa_version == 2, "worker forward must use captured FA2 without a current config"
            return SimpleNamespace(output=SimpleNamespace(value=23))
        def merge(prefix, chunk, output):
            output.value = prefix.output.value + chunk.output.value
        monkeypatch.setitem(sys.modules, "aquillm_vllm_h100.kernels.prefix", SimpleNamespace(prefix_attention=prefix))
        monkeypatch.setitem(sys.modules, "aquillm_vllm_h100.kernels.merge", SimpleNamespace(merge_attention_states=merge))
        import aquillm_vllm_h100.prefill as prefill
        monkeypatch.setattr(prefill, "raw_chunk_attention", chunk)
        for _ in range(2):
            assert route(impl, None, q, k, v, cache, metadata, 0, 1024, destination) is destination
            assert destination.value == 35
        assert sum("route_exercised prefill" in record.message for record in caplog.records) == 1
    else:
        assert route(impl, None, q, k, v, cache, metadata, 0, 1024, destination) is None
        assert destination.value == "untouched"
        assert not caplog.records
