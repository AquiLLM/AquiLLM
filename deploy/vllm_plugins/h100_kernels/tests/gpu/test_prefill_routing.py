"""Actual inspected caller plus real prefix/FA/merge; old routes must not run."""
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from reference import assert_close, make_verify_batch, reference_attention, unpack_prefix

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")]


@pytest.fixture
def fa2_runtime():
    """Configure the actual helper lookup, as in the deployed FA2 runtime."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    config = VllmConfig()
    config.attention_config.flash_attn_version = 2
    with set_current_vllm_config(config):
        assert get_flash_attn_version(head_size=256) == 2
        yield


def caller(monkeypatch, profile):
    from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionImpl
    from aquillm_vllm_h100.prefill_adapter import _make_route, rewrite_prefill_method
    original = TurboQuantAttentionImpl._prefill_attention
    # If startup already installed the extension, unwrap to its captured baseline.
    original = inspect.unwrap(original)
    def forbidden(*args, **kwargs):
        raise AssertionError("old continuation/full-cache path executed for eligible route")
    buffer_module = SimpleNamespace(TurboQuantBufferManager=SimpleNamespace(
        acquire_prefill_output=lambda **kw: torch.zeros(kw["num_tokens"], kw["num_q_heads"], kw["head_size"],
                                                      device=kw["device"], dtype=kw["dtype"])))
    monkeypatch.setitem(sys.modules, "sndr.engines.vllm.kernels_legacy.dequant_buffer", buffer_module)
    patched = rewrite_prefill_method(original, inspect.getsource(original), _make_route(profile, "gpu-unit-runtime"))
    patched.__globals__["triton_turboquant_decode_attention"] = forbidden
    impl = object.__new__(TurboQuantAttentionImpl)
    impl.scale = 0.0625
    impl.fa_version = 2
    impl.kv_cache_dtype = "turboquant_k8v4"
    impl._val_data_bytes = 128
    impl.tq_config = SimpleNamespace(key_fp8=True, effective_value_quant_bits=4, key_packed_size=256)
    impl._aquillm_h100_semantics = dict(alibi=False, sliding_window=None, soft_cap=0., causal=True,
                                       kv_sharing=False, unknown_overlays=False,
                                       model_revision="2d783431e303148fc6e16622fac5edac83a6b5c4",
                                       flash_attn_version=2)
    impl._continuation_prefill = forbidden
    return impl, patched


def meta(offsets, lengths, table, *, mirrors=True):
    return SimpleNamespace(query_start_loc=torch.tensor(offsets, device="cuda", dtype=torch.int32),
                           seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
                           query_start_loc_cpu=torch.tensor(offsets) if mirrors else None,
                           seq_lens_cpu=torch.tensor(lengths) if mirrors else None,
                           block_table=table, max_query_len=max(b-a for a,b in zip(offsets, offsets[1:])),
                           max_seq_len=max(lengths), is_prefill=True)


@pytest.mark.parametrize("prior", [4096, 32769, 65536])
def test_eligible_real_caller_bypasses_old_routes_with_long_and_reused_pages(monkeypatch, prior, fa2_runtime):
    from aquillm_vllm_h100.prefill import PrefillProfile, PrefillRegion
    profile = PrefillProfile("gpu-unit-runtime", (PrefillRegion(prior, prior, 129, 129),), "synthetic_correctness_case_only")
    impl, patched = caller(monkeypatch, profile)
    batch = make_verify_batch([prior], length=129, strided=True, block_size=16)
    q, k, v = batch.q[0], batch.raw_k[0], batch.raw_v[0]
    # Simulate a reused physical page: the current logical block table is authoritative.
    batch.block_table[0, 0] = batch.block_table[0, -1]
    # Forward must use captured FA2 even when runtime discovery is unavailable.
    from vllm.v1.attention.backends import fa_utils
    monkeypatch.setattr(fa_utils, "get_flash_attn_version", lambda **kwargs:
                        pytest.fail("worker forward redetected FlashAttention version"))
    actual = patched(impl, q, k, v, batch.kv_cache, meta([0, 129], [prior + 129], batch.block_table), None, None)
    pk, pv = unpack_prefix(batch, 0)
    expected = reference_attention(q, torch.cat((pk, k.float())), torch.cat((pv, v.float())), prior)
    assert_close(actual, expected.output, q.dtype)


def test_pn401_fresh512_shorter_continuation_actual_caller(monkeypatch, fa2_runtime):
    from aquillm_vllm_h100.prefill import PrefillProfile, PrefillRegion
    profile = PrefillProfile("gpu-unit-runtime", (PrefillRegion(4096, 4096, 129, 129),), "synthetic_correctness_case_only")
    impl, patched = caller(monkeypatch, profile)
    batch = make_verify_batch([0, 4096], length=512, block_size=16)
    q = torch.cat((batch.q[0], batch.q[1, :129]))
    k = torch.cat((batch.raw_k[0], batch.raw_k[1, :129]))
    v = torch.cat((batch.raw_v[0], batch.raw_v[1, :129]))
    actual = patched(impl, q, k, v, batch.kv_cache, meta([0, 512, 641], [512, 4225], batch.block_table), None, None)
    fresh = reference_attention(q[:512], k[:512], v[:512])
    pk, pv = unpack_prefix(batch, 1)
    continuation = reference_attention(q[512:], torch.cat((pk, k[512:].float())), torch.cat((pv, v[512:].float())), 4096)
    assert_close(actual[:512], fresh.output, q.dtype)
    assert_close(actual[512:], continuation.output, q.dtype)


@pytest.mark.parametrize("reason", ["missing_mirrors", "verification", "sliding_window", "outside_profile"])
def test_unsupported_real_caller_uses_original_continuation(monkeypatch, reason, fa2_runtime):
    from aquillm_vllm_h100.prefill import PrefillProfile, PrefillRegion
    profile = PrefillProfile("gpu-unit-runtime", (PrefillRegion(4096, 4096, 129, 129),), "synthetic_correctness_case_only")
    impl, patched = caller(monkeypatch, profile)
    batch = make_verify_batch([4096], length=129, block_size=16)
    metadata = meta([0, 129], [4225], batch.block_table, mirrors=reason != "missing_mirrors")
    if reason == "verification":
        metadata.is_verification = True
    elif reason == "sliding_window":
        impl._aquillm_h100_semantics["sliding_window"] = 4096
    elif reason == "outside_profile":
        metadata.seq_lens_cpu[0] += 1
    impl._continuation_prefill = lambda *args: torch.full_like(args[1], 23.)
    actual = patched(impl, batch.q[0], batch.raw_k[0], batch.raw_v[0], batch.kv_cache, metadata, None, None)
    torch.testing.assert_close(actual, torch.full_like(batch.q[0], 23.))


def test_bundled_profile_installed_constructor_routes_fa2_after_config_context_exits(monkeypatch, caplog):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends import fa_utils, turboquant_attn as backend
    from aquillm_vllm_h100.prefill_adapter import install_prefill_adapter
    from aquillm_vllm_h100.prefill_profiles import development_profile, MODEL_REVISION
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from deployed_prefill import bounded_reference, reference_loader
    cls = backend.TurboQuantAttentionImpl
    assert getattr(cls, "_aquillm_h100_prefill_install", None) is None, "GPU qualification requires a fresh baseline class"
    # Register each class mutation with monkeypatch so this qualification is isolated.
    monkeypatch.setattr(cls, "__init__", cls.__init__)
    monkeypatch.setattr(cls, "_prefill_attention", cls._prefill_attention)
    monkeypatch.setattr(cls, "_aquillm_h100_prefill_install", None, raising=False)
    profile = development_profile()
    assert install_prefill_adapter(profile, profile.runtime_key)["installed"]
    config = VllmConfig()
    config.attention_config.flash_attn_version = 2
    config.model_config = SimpleNamespace(hf_config=SimpleNamespace(_commit_hash=MODEL_REVISION))
    with set_current_vllm_config(config):
        assert fa_utils.get_flash_attn_version(head_size=256) == 2
        impl = cls(num_heads=24, head_size=256, scale=.0625, num_kv_heads=4, kv_cache_dtype="turboquant_k8v4")
    assert impl._aquillm_h100_semantics["model_revision"] == MODEL_REVISION
    assert impl._aquillm_h100_semantics["flash_attn_version"] == 2

    def forbidden(*args, **kwargs):
        pytest.fail("bundled eligible forward reached old path or runtime FA discovery")
    monkeypatch.setattr(fa_utils, "get_flash_attn_version", forbidden)
    monkeypatch.setattr(backend, "get_current_vllm_config", forbidden)
    monkeypatch.setattr(backend, "triton_turboquant_decode_attention", forbidden)
    # The rewritten method owns a copied globals dictionary from installation.
    impl._prefill_attention.__func__.__globals__["triton_turboquant_decode_attention"] = forbidden
    impl._continuation_prefill = forbidden
    pool = SimpleNamespace(TurboQuantBufferManager=SimpleNamespace(
        acquire_prefill_output=lambda **kw: torch.zeros(kw["num_tokens"], kw["num_q_heads"], kw["head_size"],
                                                      device=kw["device"], dtype=kw["dtype"])))
    monkeypatch.setitem(sys.modules, "sndr.engines.vllm.kernels_legacy.dequant_buffer", pool)
    prior, length = 32768, 1024
    batch = make_verify_batch([prior], length=length, block_size=16)
    q, k, v = batch.q[0], batch.raw_k[0], batch.raw_v[0]
    actual = impl._prefill_attention(q, k, v, batch.kv_cache,
                                    meta([0, length], [prior + length], batch.block_table), None, None)
    expected = bounded_reference(q, reference_loader(batch, prior, compressed_tail=False), prior + length,
                                 causal_prefix=prior, scale=impl.scale, query_tile=32, key_tile=1024)
    assert_close(actual, expected, q.dtype)
    assert sum("route_exercised prefill" in record.message for record in caplog.records) == 1
