"""Actual inspected caller plus real prefix/FA/merge; old routes must not run."""
import inspect
import sys
from types import SimpleNamespace

import pytest
import torch

from reference import assert_close, make_verify_batch, reference_attention, unpack_prefix

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")]


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
def test_eligible_real_caller_bypasses_old_routes_with_long_and_reused_pages(monkeypatch, prior):
    from aquillm_vllm_h100.prefill import PrefillProfile, PrefillRegion
    profile = PrefillProfile("gpu-unit-runtime", (PrefillRegion(prior, prior, 129, 129),), "synthetic_correctness_case_only")
    impl, patched = caller(monkeypatch, profile)
    batch = make_verify_batch([prior], length=129, strided=True, block_size=2128)
    q, k, v = batch.q[0], batch.raw_k[0], batch.raw_v[0]
    # Simulate a reused physical page: the current logical block table is authoritative.
    batch.block_table[0, 0] = batch.block_table[0, -1]
    actual = patched(impl, q, k, v, batch.kv_cache, meta([0, 129], [prior + 129], batch.block_table), None, None)
    pk, pv = unpack_prefix(batch, 0)
    expected = reference_attention(q, torch.cat((pk, k.float())), torch.cat((pv, v.float())), prior)
    assert_close(actual, expected.output, q.dtype)


def test_pn401_fresh512_shorter_continuation_actual_caller(monkeypatch):
    from aquillm_vllm_h100.prefill import PrefillProfile, PrefillRegion
    profile = PrefillProfile("gpu-unit-runtime", (PrefillRegion(4096, 4096, 129, 129),), "synthetic_correctness_case_only")
    impl, patched = caller(monkeypatch, profile)
    batch = make_verify_batch([0, 4096], length=512, block_size=2128)
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
def test_unsupported_real_caller_uses_original_continuation(monkeypatch, reason):
    from aquillm_vllm_h100.prefill import PrefillProfile, PrefillRegion
    profile = PrefillProfile("gpu-unit-runtime", (PrefillRegion(4096, 4096, 129, 129),), "synthetic_correctness_case_only")
    impl, patched = caller(monkeypatch, profile)
    batch = make_verify_batch([4096], length=129, block_size=2128)
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
