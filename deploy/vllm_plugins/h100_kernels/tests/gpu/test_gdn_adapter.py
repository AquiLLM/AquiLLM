"""Real H100 qualification; a missing adapter fails collection on the candidate.

Limits are declared before running the candidate. BF16-rounded vLLM is the
primary recurrence comparator. FP16 vLLM errors are reported separately.
Large-gate stress deliberately fails if upstream arithmetic produces NaNs.
"""
import json

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires the candidate H100 runtime")]

if torch.cuda.is_available():
    from aquillm_vllm_h100.gdn.adapter import make_adapter, supported

# Different reduction orders and BF16 boundaries are bounded explicitly.
PRIMARY_OUTPUT = dict(atol=0.005, rtol=0.025)
PRIMARY_STATE = dict(atol=0.0003, rtol=0.02)
FP16_OUTPUT = dict(atol=0.01, rtol=0.05)
FP16_STATE = dict(atol=0.003, rtol=0.05)
STATE_SIZE = 48 * 128 * 128


@pytest.fixture(scope="module", autouse=True)
def genesis_before_vllm_imports():
    """Patch sources before loading vLLM or FlashInfer's transitive imports."""
    import os
    previous = os.environ.get("AQUILLM_H100_GDN")
    os.environ["AQUILLM_H100_GDN"] = "baseline"
    try:
        import sndr.plugin
        sndr.plugin.register()
        yield
    finally:
        if previous is None:
            os.environ.pop("AQUILLM_H100_GDN", None)
        else:
            os.environ["AQUILLM_H100_GDN"] = previous


def original():
    from aquillm_vllm_h100.gdn.adapter import capture_original
    return capture_original()


def pool(slots=9, padded=True):
    stride = STATE_SIZE + (128 if padded else 0)
    # Offset and every padding element have a distinct sentinel value.
    backing = torch.full((128 + slots * stride + 128,), 987.25, device="cuda")
    state = backing.as_strided((slots, 48, 128, 128),
                               (stride, 128*128, 128, 1), storage_offset=128)
    state.copy_(torch.randn_like(state) * 0.03)
    return state, backing


def clone_pool(state, backing):
    copied = backing.clone()
    view = copied.as_strided(state.shape, state.stride(), state.storage_offset())
    return view, copied


def inputs(t=5, accepted=1, indices=None, offsets=None, padded=True):
    torch.manual_seed(431 + t)
    state, backing = pool(padded=padded)
    def rand(shape):
        return (torch.randn(shape, device="cuda") * 0.2).half()
    indices = indices or list(range(1, t+1)) + [7, 8]
    args = dict(A_log=torch.full((48,), -1.0, device="cuda"),
                a=rand((1,t,48)), b=rand((1,t,48)),
                dt_bias=torch.full((48,), -0.2, device="cuda", dtype=torch.float16),
                q=rand((1,t,16,128)), k=rand((1,t,16,128)), v=rand((1,t,48,128)),
                initial_state=state,
                cu_seqlens=torch.tensor(offsets or [0,t], device="cuda", dtype=torch.int32),
                ssm_state_indices=torch.tensor([indices], device="cuda", dtype=torch.int32),
                num_accepted_tokens=torch.tensor([accepted], device="cuda", dtype=torch.int32))
    return args, backing


def reference(args, backing, rounded):
    state, copied = clone_pool(args["initial_state"], backing)
    args = dict(args, initial_state=state)
    if rounded:
        # Preserve the original FP16 output boundary while rounding operands.
        for name in ("q", "k", "v", "a", "b"):
            args[name] = args[name].bfloat16().half()
    return original()(**args)[0], state, copied


def errors(actual, expected):
    error = actual.float() - expected.float()
    return {"max": error.abs().max().item(), "rms": error.square().mean().sqrt().item()}


def assert_untouched(args, before, after):
    state = args["initial_state"]
    mask = torch.ones(before.numel(), dtype=torch.bool, device="cuda")
    begin, end = args["cu_seqlens"].cpu().tolist()
    accepted = args["num_accepted_tokens"].item()
    indices = args["ssm_state_indices"][0].cpu().tolist()
    if end > begin and 1 <= accepted <= len(indices) and 0 < indices[accepted-1] < state.shape[0]:
        for slot in indices[:end-begin]:
            if 0 < slot < state.shape[0]:
                start = state.storage_offset() + slot * state.stride(0)
                mask[start:start+STATE_SIZE] = False
    assert torch.equal(before[mask], after[mask]), "untouched pool or backing padding changed"


def compare(args, backing):
    before = backing.clone()
    primary_o, primary_s, _ = reference(args, before, True)
    fp16_o, fp16_s, _ = reference(args, before, False)
    route_calls = []
    def fallback(*a, **kw):
        route_calls.append(True)
        return original()(*a, **kw)
    assert supported(**args), "nominal qualification input must exercise candidate"
    output, returned = make_adapter(fallback)(**args)
    assert not route_calls, "qualification silently used baseline"
    assert returned is args["initial_state"]
    begin, end = args["cu_seqlens"].cpu().tolist()
    indices = args["ssm_state_indices"][0].cpu().tolist()
    accepted = args["num_accepted_tokens"].item()
    valid = end > begin and 0 < indices[accepted-1] < returned.shape[0]
    assert_untouched(args, before, backing)
    if valid:
        active = output[:,begin:end]
        po, fo = primary_o[:,begin:end], fp16_o[:,begin:end]
        print(json.dumps({"T": args["q"].shape[1], "accepted": accepted,
                          "rounded_output": errors(active, po),
                          "rounded_state": errors(returned, primary_s),
                          "fp16_output": errors(active, fo),
                          "fp16_state": errors(returned, fp16_s)}))
        torch.testing.assert_close(active, po, **PRIMARY_OUTPUT)
        torch.testing.assert_close(returned, primary_s, **PRIMARY_STATE)
        torch.testing.assert_close(active, fo, **FP16_OUTPUT)
        torch.testing.assert_close(returned, fp16_s, **FP16_STATE)
    return output


@pytest.mark.parametrize("t,accepted", [(t,a) for t in (2,3,4,5) for a in range(1,t+1)])
@pytest.mark.parametrize("padded", [False, True])
def test_all_prefixes_exact_geometry(t, accepted, padded):
    args, backing = inputs(t, accepted, padded=padded)
    compare(args, backing)


@pytest.mark.parametrize("offsets", [[1,4], [3,3], [0,0]])
@pytest.mark.parametrize("indices,accepted", [([2,0,-2,2,4,7,8], 1),
                                              ([2,3,2,4,5,7,8], 4),
                                              ([0,2,3,4,5,7,8], 1),
                                              ([-1,2,3,4,5,7,8], 1)])
def test_masks_accepted_source_repeated_destinations_and_padding(offsets, indices, accepted):
    args, backing = inputs(5, accepted, indices, offsets)
    compare(args, backing)


@pytest.mark.parametrize("accepted", [0, -1, 8])
def test_impossible_acceptance_fails_closed_without_changing_pool(accepted):
    args, backing = inputs(5, accepted)
    before = backing.clone()
    make_adapter(original())(**args)
    assert torch.equal(before, backing)


@pytest.mark.parametrize("indices,offsets", [([99,2,3,4,5,7,8], [0,5]),
                                            ([1,99,3,4,5,7,8], [0,5]),
                                            ([1,2,3,4,5,7,8], [-1,4]),
                                            ([1,2,3,4,5,7,8], [3,2]),
                                            ([1,2,3,4,5,7,8], [0,6])])
def test_gpu_bounds_guards(indices, offsets):
    args, backing = inputs(5, 1, indices, offsets)
    before = backing.clone()
    make_adapter(original())(**args)
    if indices[0] == 99 or offsets != [0,5]:
        assert torch.equal(before, backing)
    else:
        # Impossible destination is skipped; every unrelated slot and padding
        # still requires exact preservation, without running unsafe baseline.
        assert_untouched(args, before, backing)


@pytest.mark.parametrize("case", ["T1", "N2", "bf16", "wrong_heads"])
def test_gpu_unsupported_contract_uses_original(case):
    args, _ = inputs(1 if case == "T1" else 5)
    if case == "N2":
        args["cu_seqlens"] = torch.tensor([0,2,5], device="cuda", dtype=torch.int32)
    elif case == "bf16":
        args["q"] = args["q"].bfloat16()
    elif case == "wrong_heads":
        args["q"] = args["q"][:,:,:8]
    def saved_original(**kw):
        assert kw["initial_state"] is args["initial_state"]
        return "fallback", kw["initial_state"]
    assert make_adapter(saved_original)(**args)[0] == "fallback"


def test_rollback_acceptance_changes_and_slot_reuse():
    args, backing = inputs()
    for accepted in (5,1,3,2):
        args["num_accepted_tokens"].fill_(accepted)
        compare(args, backing)
    args["ssm_state_indices"].copy_(torch.tensor([[5,4,3,2,1,7,8]], device="cuda", dtype=torch.int32))
    compare(args, backing)


@pytest.mark.parametrize("norm,scale", [(False,None), (True,None), (False,0.17), (True,0.17)])
def test_scale_and_normalization(norm, scale):
    args, backing = inputs()
    # Qwen passes flattened 2D a/b gates; also retain 3D coverage elsewhere.
    args["a"] = args["a"].squeeze(0)
    args["b"] = args["b"].squeeze(0)
    args.update(use_qk_l2norm_in_kernel=norm, scale=scale)
    compare(args, backing)


def test_output_survives_next_call_and_pool_identity():
    args, backing = inputs()
    call = make_adapter(original())
    first, state = call(**args)
    saved = first.clone()
    second, again = call(**args)
    assert first.data_ptr() != second.data_ptr()
    assert state is again is args["initial_state"]
    assert torch.equal(first, saved)


def test_real_alias_installation_wraps_actual_post_genesis_and_exercises_route(monkeypatch, caplog):
    from aquillm_vllm_h100.gdn.adapter import install_adapter
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qwen
    saved = original()
    # Restore the alias after this integration test without changing other
    # Qwen methods, prefill dispatch or the packed-decode callable.
    monkeypatch.setattr(qwen, "fused_sigmoid_gating_delta_rule_update", saved)
    install_adapter()
    installed = qwen.fused_sigmoid_gating_delta_rule_update
    assert installed.__wrapped__ is saved
    args, backing = inputs()
    before = backing.clone()
    expected_o, expected_s, _ = reference(args, before, True)
    output, state = installed(**args)
    assert any("route_exercised gdn=flashinfer" in record.getMessage() for record in caplog.records)
    assert state is args["initial_state"]
    torch.testing.assert_close(output, expected_o, **PRIMARY_OUTPUT)
    torch.testing.assert_close(state, expected_s, **PRIMARY_STATE)


def test_graph_replay_changed_metadata_and_two_independent_workspaces():
    fixtures = [inputs(), inputs(5, 3)]
    call = make_adapter(original())
    graphs, outputs, seeds = [], [], []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for args, backing in fixtures:
            seed = backing.clone()
            for _ in range(3):
                call(**args)
            backing.copy_(seed)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output, _ = call(**args)
            graphs.append(graph)
            outputs.append(output)
            seeds.append(seed)
    torch.cuda.current_stream().wait_stream(stream)
    for replay in range(3):
        for i, ((args, backing), graph, seed) in enumerate(zip(fixtures, graphs, seeds)):
            backing.copy_(seed)
            args["num_accepted_tokens"].fill_((replay+i)%5+1)
            args["ssm_state_indices"].copy_(torch.tensor(
                [[2,3,4,5,1,7,8] if replay%2 else [1,2,3,4,5,7,8]], device="cuda", dtype=torch.int32))
            args["cu_seqlens"].copy_(torch.tensor([replay%2,5], device="cuda", dtype=torch.int32))
            expected_o, expected_s, _ = reference(args, seed, True)
            graph.replay()
            begin = replay%2
            torch.testing.assert_close(outputs[i][:,begin:5], expected_o[:,begin:5], **PRIMARY_OUTPUT)
            torch.testing.assert_close(args["initial_state"], expected_s, **PRIMARY_STATE)
            assert_untouched(args, seed, backing)


def test_large_gate_stress_reports_upstream_overflow_as_failure():
    args, backing = inputs()
    args["a"].fill_(100.0)
    output = compare(args, backing)
    assert torch.isfinite(output).all(), "candidate softplus overflow for finite a+dt_bias >88"
