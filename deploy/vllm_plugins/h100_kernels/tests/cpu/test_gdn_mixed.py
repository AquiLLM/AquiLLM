"""Captured serving method: gate/QKV alignment, unchanged routes, closed install.

GPU kernels are replaced at their boundary; all tensor operations run on CPU.
"""
import ast
import importlib
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "post_genesis_gdn_core.py"
MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"


def implementation():
    try:
        return importlib.import_module("aquillm_vllm_h100.gdn_mixed")
    except ModuleNotFoundError:
        pytest.fail("missing mixed GDN gate-alignment backport")


def fixture_method(metadata):
    calls = []

    def recurrent(**kw):
        calls.append(("recurrent", kw))
        return kw["q"].clone(), kw["initial_state"]

    def prep(**kw):
        calls.append(("prep", kw))
        x = kw["conv_output"][:, None, :]
        return x, x, x, kw["a"][:, None, :], kw["b"][:, None, :]

    namespace = dict(torch=torch, GDNAttentionMetadata=SimpleNamespace,
        get_forward_context=lambda: SimpleNamespace(attn_metadata={"gdn": metadata}),
        is_conv_state_dim_first=lambda: True,
        causal_conv1d_update=lambda x, *args, **kwargs: x,
        causal_conv1d_fn=lambda x, *args, **kwargs: x,
        fused_post_conv_prep=prep, fused_sigmoid_gating_delta_rule_update=recurrent)
    source = FIXTURE.read_text(encoding="utf-8")
    exec(compile(source, str(FIXTURE), "exec"), namespace)
    return namespace["_forward_core"], calls


def inputs(lengths, speculative, strided=False):
    total = sum(lengths)
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    indices = [torch.arange(int(offsets[i]), int(offsets[i + 1])) for i in range(len(lengths))]
    spec_idx = torch.cat([indices[i] for i, flag in enumerate(speculative) if flag]) if any(speculative) else None
    non_idx = torch.cat([indices[i] for i, flag in enumerate(speculative) if not flag]) if not all(speculative) else torch.empty(0, dtype=torch.int64)
    slots = torch.arange(1, len(lengths) * 5 + 1, dtype=torch.int32).view(-1, 5)
    mask = torch.tensor(speculative, dtype=torch.bool)
    spec_lens = [n for n, flag in zip(lengths, speculative) if flag]
    non_lens = [n for n, flag in zip(lengths, speculative) if not flag]
    cumulative = lambda ns: torch.tensor([0, *torch.tensor(ns).cumsum(0).tolist()], dtype=torch.int32)
    num_decodes = 0 if any(speculative) else sum(n == 1 for n in lengths)
    num_prefills = len(non_lens) if any(speculative) else len(lengths) - num_decodes
    prefill_slots = slots[~mask, 0] if any(speculative) else slots[num_decodes:, 0]
    md = SimpleNamespace(has_initial_state=torch.ones(len(lengths), dtype=torch.bool),
        spec_query_start_loc=cumulative(spec_lens) if any(speculative) else None,
        non_spec_query_start_loc=cumulative(non_lens),
        spec_sequence_masks=mask if any(speculative) else None,
        spec_token_indx=spec_idx, non_spec_token_indx=non_idx,
        spec_state_indices_tensor=slots[mask] if any(speculative) else None,
        non_spec_state_indices_tensor=slots[~mask, 0], num_actual_tokens=total,
        num_accepted_tokens=torch.tensor([1 + i % 5 for i in range(len(spec_lens))], dtype=torch.int32),
        num_prefills=num_prefills, num_decodes=num_decodes,
        num_spec_decodes=len(spec_lens), num_decode_tokens=num_decodes,
        prefill_state_indices=prefill_slots,
        prefill_has_initial_state=torch.ones(len(prefill_slots), dtype=torch.bool),
        prefill_query_start_loc=cumulative(non_lens if any(speculative) else lengths[num_decodes:]),
        chunk_indices=None, chunk_offsets=None)
    ids = torch.cat([torch.arange(100 * (i + 1), 100 * (i + 1) + n)
                     for i, n in enumerate(lengths)]).float()[:, None]
    model = SimpleNamespace(prefix="gdn", enable_packed_recurrent_decode=False,
        kv_cache=(torch.zeros(32, 1, 1), torch.zeros(32, 1, 1, 1)),
        conv1d=SimpleNamespace(weight=torch.zeros(1, 1, 1), bias=None), activation="silu",
        num_k_heads=1, tp_size=1, head_k_dim=1, head_v_dim=1,
        A_log=torch.zeros(1), dt_bias=torch.zeros(1),
        rearrange_mixed_qkv=lambda x: (None, None, None) if x is None else (x[None, :, None, :],) * 3,
        chunk_gated_delta_rule=lambda **kw: (kw["q"].clone(), kw["initial_state"].clone()))
    b, a = ids + 2000, ids + 1000
    if strided:
        ba = torch.cat([b, a], dim=1)
        b, a = ba[:, :1], ba[:, 1:]
    return md, model, ids, b, a


def call(method, model, ids, b, a):
    output = torch.empty(ids.shape[0], 1, 1)
    method(model, ids, b, a, output)
    return output


@pytest.mark.parametrize("lengths,mask", [([1, 5], [False, True]), ([10, 5], [False, True]),
    ([5, 1, 3, 5], [True, False, False, True]), ([1, 5, 2, 5], [False, True, False, True]),
    ([5, 3], [True, False])])
@pytest.mark.parametrize("strided", [False, True])
def test_mixed_gates_follow_gathered_qkv(lengths, mask, strided):
    md, model, ids, b, a = inputs(lengths, mask, strided)
    original, calls = fixture_method(md)
    patched = implementation().rewrite_gdn_method(original, inspect.getsource(original))
    output = call(patched, model, ids, b, a)
    rec = next(kw for kind, kw in calls if kind == "recurrent")
    expected = ids.index_select(0, md.spec_token_indx)
    assert torch.equal(rec["q"].flatten(), expected.flatten())
    assert torch.equal(rec["a"], expected + 1000)
    assert torch.equal(rec["b"], expected + 2000)
    assert torch.equal(rec["ssm_state_indices"], md.spec_state_indices_tensor)
    assert rec["num_accepted_tokens"] is md.num_accepted_tokens
    prep = next(kw for kind, kw in calls if kind == "prep")
    assert torch.equal(prep["a"], a.index_select(0, md.non_spec_token_indx))
    assert torch.equal(prep["b"], b.index_select(0, md.non_spec_token_indx))
    assert torch.equal(output.flatten(), ids.flatten())


def test_original_fixture_reproduces_wrong_gate_order():
    md, model, ids, b, a = inputs([1, 5], [False, True])
    original, calls = fixture_method(md)
    call(original, model, ids, b, a)
    rec = next(kw for kind, kw in calls if kind == "recurrent")
    assert rec["q"].flatten().tolist() == [200, 201, 202, 203, 204]
    assert rec["a"][:5].flatten().tolist() == [1100, 1200, 1201, 1202, 1203]


@pytest.mark.parametrize("lengths", [[5], [5, 5], [1, 5, 3, 5]])
@pytest.mark.parametrize("strided", [False, True])
def test_pure_spec_preserves_gate_views_and_outputs(lengths, strided):
    md, model, ids, b, a = inputs(lengths, [True] * len(lengths), strided)
    original, calls = fixture_method(md)
    before = call(original, model, ids, b, a)
    calls.clear()
    after = call(implementation().rewrite_gdn_method(original, inspect.getsource(original)), model, ids, b, a)
    rec = calls[0][1]
    assert rec["a"].data_ptr() == a.data_ptr() and rec["b"].data_ptr() == b.data_ptr()
    assert rec["a"].stride() == a.stride() and rec["b"].stride() == b.stride()
    assert torch.equal(before, after)


@pytest.mark.parametrize("lengths", [[1], [1, 1], [3], [3, 4], [1, 3]])
def test_pure_nonspec_retains_original_execution(lengths):
    md, model, ids, b, a = inputs(lengths, [False] * len(lengths))
    original, calls = fixture_method(md)
    before = call(original, model, ids, b, a)
    recorded = [(kind, {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kw.items()})
                for kind, kw in calls]
    calls.clear()
    after = call(implementation().rewrite_gdn_method(original, inspect.getsource(original)), model, ids, b, a)
    assert torch.equal(before, after)
    assert [kind for kind, _ in recorded] == [kind for kind, _ in calls]
    for (_, old), (_, new) in zip(recorded, calls):
        for key in old:
            assert torch.equal(old[key], new[key]) if isinstance(old[key], torch.Tensor) else old[key] == new[key]


@pytest.mark.parametrize("old,new", [("a=a,", "a=b,"),
    ("mixed_qkv.index_select(0, spec_token_indx)", "mixed_qkv.index_select(0, non_spec_token_indx)"),
    ("num_accepted_tokens=num_accepted_tokens,", "num_accepted_tokens=None,"),
    ("inplace_final_state=True,", "inplace_final_state=False,")])
def test_source_drift_rejected(old, new):
    md, *_ = inputs([1, 5], [False, True])
    original, _ = fixture_method(md)
    source = inspect.getsource(original)
    assert old in source
    with pytest.raises(ValueError, match="fingerprint|contract"):
        implementation().rewrite_gdn_method(original, source.replace(old, new))


def test_source_contract_ignores_comments_and_whitespace():
    md, *_ = inputs([1, 5], [False, True])
    original, _ = fixture_method(md)
    source = inspect.getsource(original)
    assert implementation().rewrite_gdn_method(original, "# harmless comment\n" + source)


def test_source_fingerprint_is_python_version_independent():
    md, *_ = inputs([1, 5], [False, True])
    original, _ = fixture_method(md)
    # Canonical all-fields JSON, including empty args/lists; Python 3.13's
    # ast.dump now omits those by default, unlike the serving Python 3.12.
    assert implementation()._source_fingerprint(ast.parse(inspect.getsource(original))) == (
        "8af7d81452b2f73fac77821e865714baa5c3e9a704d2d42cae562b7304642da1")


def test_install_idempotent_and_preserves_original(monkeypatch):
    md, *_ = inputs([1, 5], [False, True])
    original, _ = fixture_method(md)
    cls = type("QwenGatedDeltaNetAttention", (), {"_forward_core": original})
    monkeypatch.setitem(sys.modules, MODULE, SimpleNamespace(QwenGatedDeltaNetAttention=cls))
    module = implementation()
    first = module.install_gdn_mixed()
    installed = cls._forward_core
    second = module.install_gdn_mixed()
    assert first["installed"] and second["installed"]
    assert cls._forward_core is installed and installed.__wrapped__ is original


def test_unknown_source_terminates_startup_without_mutation(monkeypatch):
    def unknown(self, mixed_qkv, b, a, core_attn_out):
        return None
    cls = type("QwenGatedDeltaNetAttention", (), {"_forward_core": unknown})
    monkeypatch.setitem(sys.modules, MODULE, SimpleNamespace(QwenGatedDeltaNetAttention=cls))
    with pytest.raises(SystemExit, match="GDN.*(contract|fingerprint)"):
        implementation().install_gdn_mixed()
    assert cls._forward_core is unknown


def test_install_adapters_hooks_correction_before_other_mutations(monkeypatch):
    import aquillm_vllm_h100.adapters as adapters
    import aquillm_vllm_h100.prefill_adapter as prefill
    module = implementation()
    events = []
    baseline = SimpleNamespace(call_p67_splitk=lambda **kw: None)
    monkeypatch.setitem(sys.modules, "sndr.engines.vllm.kernels_legacy", SimpleNamespace(p67_multi_query_kernel=baseline))
    monkeypatch.setattr(module, "install_gdn_mixed", lambda: events.append("gdn") or {"installed": True})
    monkeypatch.setattr(prefill, "install_prefill_adapter", lambda *args: events.append("prefill") or {"installed": True, "reason": "test"})
    monkeypatch.setattr(adapters, "make_verifier_adapter", lambda *args: events.append("mtp") or "wrapper")
    result = adapters.install_adapters(dict(mtp="fused", split="baseline", prefill="1", profile=None))
    assert events == ["gdn", "prefill", "mtp"]
    assert result["gdn_mixed"] is True
