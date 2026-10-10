"""Execute the inspected post-Genesis method; GPU primitives are boundaries."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


def installed_method(monkeypatch, route):
    from aquillm_vllm_h100.prefill_adapter import rewrite_prefill_method
    source = (Path(__file__).resolve().parents[1] / "fixtures" / "post_genesis_prefill.py").read_text(encoding="utf-8")
    def forbidden(*args, **kwargs):
        raise AssertionError("old continuation/dequant path executed")
    # Source includes the real buffer-manager import. Replace only that external resource.
    buffer_module = SimpleNamespace(TurboQuantBufferManager=SimpleNamespace(
        acquire_prefill_output=lambda **kw: torch.zeros(kw["num_tokens"], kw["num_q_heads"], kw["head_size"])))
    monkeypatch.setitem(sys.modules, "sndr.engines.vllm.kernels_legacy.dequant_buffer", buffer_module)
    globals_ = dict(torch=torch, _HAS_FLASH_ATTN=True, TurboQuantMetadata=object, Any=object,
                    _CONTINUATION_DECODE_THRESHOLD=64, _CONTINUATION_DECODE_MAX_CACHED_LEN=32768,
                    triton_turboquant_decode_attention=forbidden)
    exec(compile("from __future__ import annotations\n" + source, "fixture", "exec"), globals_)
    original = globals_["_prefill_attention"]
    patched = rewrite_prefill_method(original, source, route)
    class Impl:
        scale = 0.0625
        _continuation_prefill = forbidden
        _prefill_attention = patched
        def _flash_attn_varlen(self, **kw):
            # Fresh raw output is independent of previously cached continuation state.
            return torch.full_like(kw["q"], 7.)
    return Impl(), source, original


def metadata(offsets, lengths, *, cpu=True):
    return SimpleNamespace(query_start_loc=torch.tensor(offsets, dtype=torch.int32),
                           seq_lens=torch.tensor(lengths, dtype=torch.int32),
                           query_start_loc_cpu=torch.tensor(offsets) if cpu else None,
                           seq_lens_cpu=torch.tensor(lengths) if cpu else None,
                           block_table=torch.zeros(len(lengths), 4096, dtype=torch.int32),
                           max_query_len=max(b-a for a,b in zip(offsets, offsets[1:])),
                           max_seq_len=max(lengths), is_prefill=True)


@pytest.mark.parametrize("prior", [4096, 32768, 65536])
def test_route_precedes_both_old_continuation_branches(monkeypatch, prior):
    def route(impl, layer, q, k, v, cache, meta, index, total_tokens, destination):
        assert index == 0 and total_tokens == 129
        assert meta.seq_lens_cpu[index] - q.shape[0] == prior
        return torch.full_like(q, 11.)
    impl, _, _ = installed_method(monkeypatch, route)
    q = torch.zeros(129, 24, 256)
    kv = torch.zeros(129, 4, 256)
    out = impl._prefill_attention(q, kv, kv, None, metadata([0, 129], [prior + 129]), None, None)
    torch.testing.assert_close(out, torch.full_like(q, 11.))


def test_pn401_fresh512_and_shorter_continuation_preserve_both_outputs(monkeypatch):
    def route(impl, layer, q, k, v, cache, meta, index, total_tokens, destination):
        assert index == 1 and q.shape[0] == 129 and total_tokens == 641
        return torch.full_like(q, 13.)
    impl, _, _ = installed_method(monkeypatch, route)
    q = torch.zeros(641, 24, 256)
    kv = torch.zeros(641, 4, 256)
    out = impl._prefill_attention(q, kv, kv, None, metadata([0, 512, 641], [512, 4225]), None, None)
    torch.testing.assert_close(out[:512], torch.full_like(q[:512], 7.))
    torch.testing.assert_close(out[512:], torch.full_like(q[512:], 13.))


def test_fresh_batch_retains_original_fast_helper(monkeypatch):
    def forbidden_route(*args):
        raise AssertionError("fresh request attempted continuation route")
    impl, _, _ = installed_method(monkeypatch, forbidden_route)
    q = torch.zeros(512, 24, 256)
    kv = torch.zeros(512, 4, 256)
    out = impl._prefill_attention(q, kv, kv, None, metadata([0, 512], [512]), None, None)
    torch.testing.assert_close(out, torch.full_like(q, 7.))


def test_unsupported_request_dispatches_captured_baseline(monkeypatch):
    impl, _, _ = installed_method(monkeypatch, lambda *args: None)
    impl._continuation_prefill = lambda *args: torch.full_like(args[1], 17.)
    q = torch.zeros(129, 24, 256)
    kv = torch.zeros(129, 4, 256)
    out = impl._prefill_attention(q, kv, kv, None, metadata([0, 129], [4225]), None, None)
    torch.testing.assert_close(out, torch.full_like(q, 17.))


def test_changed_runtime_method_fails_closed(monkeypatch):
    from aquillm_vllm_h100.prefill_adapter import rewrite_prefill_method
    _, source, original = installed_method(monkeypatch, lambda *args: None)
    with pytest.raises(ValueError, match="fingerprint"):
        rewrite_prefill_method(original, source.replace("cached_len = seq_len - q_len", "cached_len = seq_len"), lambda *args: None)


def test_constructor_capture_keeps_original_initialization_and_records_unsupported_options():
    from aquillm_vllm_h100.prefill_adapter import _capture_constructor
    class Impl:
        def __init__(self, num_heads, head_size, scale, num_kv_heads=None, alibi_slopes=None,
                     sliding_window=None, kv_cache_dtype="auto", logits_soft_cap=None,
                     attn_type="decoder", kv_sharing_target_layer_name=None, **kwargs):
            self.initialized = (num_heads, head_size, scale, kwargs)
    Impl.__init__ = _capture_constructor(Impl.__init__)
    ordinary = Impl(24, 256, 0.0625, 4)
    assert ordinary.initialized == (24, 256, 0.0625, {})
    assert ordinary._aquillm_h100_semantics["causal"]
    unusual = Impl(24, 256, 0.0625, 4, sliding_window=4096, logits_soft_cap=50., sinks=[0])
    assert unusual.initialized == (24, 256, 0.0625, {"sinks": [0]})
    assert unusual._aquillm_h100_semantics["sliding_window"] == 4096
    assert unusual._aquillm_h100_semantics["soft_cap"] == 50.
    assert unusual._aquillm_h100_semantics["unknown_overlays"]


def test_existing_instances_without_constructor_snapshot_keep_baseline():
    from aquillm_vllm_h100.prefill_adapter import _make_route
    route = _make_route(None, "unit")
    assert route(SimpleNamespace(), None, None, None, None, None, None, 0, 0, None) is None
