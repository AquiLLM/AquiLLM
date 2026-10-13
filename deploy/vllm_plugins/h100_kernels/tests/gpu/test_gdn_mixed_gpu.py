"""Corrected deployed GDN forward versus independent request execution.

Real convolution, Genesis QKV split, recurrent update and selected FlashInfer
chunk kernels execute here. Only forward-context lookup is injected. CPU tests
prove the source transformation; these tests qualify numerical/state isolation.
Mixed metadata reclassifies ordinary one-token rows as prefills, so isolated
references keep that classification too. This is an eager mixed-batch gate,
not a claim that vLLM captures mixed prefill batches in a CUDA graph.
"""
import inspect
from types import FunctionType, MethodType, SimpleNamespace

import pytest
import torch


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA runtime required")]

OUTPUT_ATOL, OUTPUT_RTOL = 1e-4, 0.002
STATE_ATOL, STATE_RTOL = 1e-5, 0.002
Q_HEADS, V_HEADS, DIM, CONV_WIDTH, DRAFTS = 16, 48, 128, 4, 4
QKV_DIM = (2 * Q_HEADS + V_HEADS) * DIM
PREFIX = "gdn_mixed_gpu_gate"

# (speculative, length); include ordinary rows before, between and after MTP.
CASES = [
    ((False, 1), (True, 5)),
    ((True, 5), (False, 1)),
    ((False, 10), (True, 5)),
    ((True, 5), (False, 10)),
    ((True, 5), (False, 1), (False, 3), (True, 5)),
    ((False, 1), (True, 5), (True, 5), (False, 3)),
    ((True, 5), (True, 5)),
]


@pytest.fixture(scope="module")
def runtime():
    # The root harness must register Genesis before importing this module.
    # Unknown/unpatched source is rejected by the production installer itself.
    from aquillm_vllm_h100.gdn_mixed import install_gdn_mixed
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as module
    assert torch.cuda.get_device_capability() == (9, 0), "H100 qualification required"
    install_gdn_mixed()
    candidate = module.QwenGatedDeltaNetAttention._forward_core
    marker = getattr(candidate, "_aquillm_h100_gdn_mixed", None)
    assert marker is not None and candidate.__wrapped__ is marker[1]
    original = inspect.unwrap(candidate)
    # No fixture recurrence or reconstructed math enters either side.
    assert candidate.__globals__["fused_sigmoid_gating_delta_rule_update"] is (
        original.__globals__["fused_sigmoid_gating_delta_rule_update"])
    assert candidate.__globals__["causal_conv1d_update"] is (
        original.__globals__["causal_conv1d_update"])
    assert candidate.__globals__["causal_conv1d_fn"] is (
        original.__globals__["causal_conv1d_fn"])
    return module, original, candidate


def cumulative(lengths, device="cuda"):
    result, total = [0], 0
    for length in lengths:
        total += length
        result.append(total)
    return torch.tensor(result, dtype=torch.int32, device=device)


def metadata(module, rows, row_ids, accepted):
    """Pinned builder contract, including CPU-built convolution/chunk metadata."""
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
    from vllm.model_executor.layers.fla.ops.index import (
        prepare_chunk_indices, prepare_chunk_offsets)

    spec_lengths = [length for spec, length in rows if spec]
    non_lengths = [length for spec, length in rows if not spec]
    spec_slots = [[1 + row_id * 6 + i for i in range(5)]
                  for (spec, _), row_id in zip(rows, row_ids) if spec]
    non_slots = [1 + row_id * 6 for (spec, _), row_id in zip(rows, row_ids) if not spec]
    spec_tokens, non_tokens, start = [], [], 0
    for spec, length in rows:
        (spec_tokens if spec else non_tokens).extend(range(start, start + length))
        start += length
    tensor = lambda values, dtype=torch.int32: torch.tensor(values, dtype=dtype, device="cuda")
    non_offsets = cumulative(non_lengths) if non_lengths else None
    initial = tensor([True] * len(non_lengths), torch.bool) if non_lengths else None
    md = module.GDNAttentionMetadata(
        num_prefills=len(non_lengths), num_prefill_tokens=sum(non_lengths),
        num_decodes=0, num_decode_tokens=0, num_spec_decodes=len(spec_lengths),
        num_spec_decode_tokens=sum(spec_lengths), num_actual_tokens=start,
        has_initial_state=initial,
        spec_query_start_loc=cumulative(spec_lengths) if spec_lengths else None,
        non_spec_query_start_loc=non_offsets,
        spec_state_indices_tensor=tensor(spec_slots) if spec_lengths else None,
        non_spec_state_indices_tensor=tensor(non_slots) if non_lengths else None,
        spec_sequence_masks=tensor([spec for spec, _ in rows], torch.bool) if spec_lengths else None,
        spec_token_indx=tensor(spec_tokens, torch.int64) if spec_lengths else None,
        non_spec_token_indx=tensor(non_tokens, torch.int64) if spec_lengths else None,
        num_accepted_tokens=tensor([accepted] * len(spec_lengths)) if spec_lengths else None,
        prefill_query_start_loc=non_offsets,
        prefill_state_indices=tensor(non_slots) if non_lengths else None,
        prefill_has_initial_state=initial,
    )
    if non_lengths:
        cpu_offsets = cumulative(non_lengths, "cpu")
        md.chunk_indices = prepare_chunk_indices(cpu_offsets, FLA_CHUNK_SIZE).to("cuda")
        md.chunk_offsets = prepare_chunk_offsets(cpu_offsets, FLA_CHUNK_SIZE).to("cuda")
        md.nums_dict, md.batch_ptr, md.token_chunk_offset_ptr = (
            compute_causal_conv1d_metadata(cpu_offsets, device=torch.device("cuda")))
    return md


class Probe:
    """Small layer shell: every invoked tensor operation remains installed code."""


def model(module, pool, weights):
    instance = Probe()
    instance.prefix = PREFIX
    instance.tp_size = 1
    instance.num_k_heads = Q_HEADS
    instance.num_v_heads = V_HEADS
    instance.key_dim = Q_HEADS * DIM
    instance.value_dim = V_HEADS * DIM
    instance.head_k_dim = instance.head_v_dim = DIM
    instance.enable_packed_recurrent_decode = False
    instance.activation = "silu"
    instance.kv_cache = pool
    instance.conv1d = SimpleNamespace(weight=weights[0], bias=None)
    instance.A_log, instance.dt_bias = weights[1:]
    instance.rearrange_mixed_qkv = MethodType(
        module.QwenGatedDeltaNetAttention.rearrange_mixed_qkv, instance)
    # This is the pinned Hopper auto selection, without loading model weights or
    # constructing unrelated projections/distributed process groups.
    config = SimpleNamespace(additional_config={}, model_config=SimpleNamespace(
        dtype=torch.float16, hf_text_config=SimpleNamespace(
            linear_key_head_dim=DIM, linear_value_head_dim=DIM)))
    assert module._resolve_gdn_prefill_backend(config)[1] == "flashinfer"
    instance.chunk_gated_delta_rule = MethodType(
        module.ChunkGatedDeltaRule.forward_cuda, instance)
    return instance


def invoke(function, instance, md, qkv, b, a):
    # Never replace a real module/global kernel alias. Context is per invocation.
    namespace = dict(function.__globals__)
    namespace["get_forward_context"] = lambda: SimpleNamespace(attn_metadata={PREFIX: md})
    call = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    call.__kwdefaults__ = function.__kwdefaults__
    output = torch.full((md.num_actual_tokens + 2, V_HEADS, DIM), 7.,
                        dtype=torch.float16, device="cuda")
    call(instance, qkv.clone(), b, a, output)
    assert torch.equal(output[md.num_actual_tokens:], torch.full_like(output[-2:], 7.))
    return output[:md.num_actual_tokens]


def inputs(rows, stride, step):
    generator = torch.Generator(device="cuda").manual_seed(617 + step)
    total = sum(length for _, length in rows)
    qkv = torch.randn((total, QKV_DIM), device="cuda", dtype=torch.float16,
                      generator=generator) * 0.3
    backing_a = torch.full((total, stride), float("nan"), device="cuda", dtype=torch.float16)
    backing_b = torch.full_like(backing_a, float("nan"))
    a, b = backing_a[:, :V_HEADS], backing_b[:, :V_HEADS]
    start = 0
    for row_id, (_, length) in enumerate(rows):
        a[start:start + length] = torch.randn((length, V_HEADS), device="cuda",
            dtype=torch.float16, generator=generator) * 0.25 + (row_id - 1) * 0.8
        b[start:start + length] = torch.randn((length, V_HEADS), device="cuda",
            dtype=torch.float16, generator=generator) * 0.25 + (row_id - 1) * 0.5
        start += length
    assert a.stride() == b.stride() == (stride, 1)
    return qkv, b, a, backing_a, backing_b


def initialize(module, batch_size):
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
    conv_shape, state_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, Q_HEADS, V_HEADS, DIM, DIM, CONV_WIDTH, DRAFTS)
    count = batch_size * 6 + 2
    generator = torch.Generator(device="cuda").manual_seed(1901)
    conv = torch.randn((count, *conv_shape), device="cuda", dtype=torch.float16,
                       generator=generator) * 0.1
    state = torch.randn((count, *state_shape), device="cuda", dtype=torch.float32,
                        generator=generator) * 0.03
    weights = (
        torch.randn((QKV_DIM, 1, CONV_WIDTH), device="cuda", dtype=torch.float16,
                    generator=generator) * 0.2,
        torch.nn.Parameter(torch.linspace(-1., 0.5, V_HEADS, device="cuda", dtype=torch.float32)),
        torch.nn.Parameter(torch.linspace(-0.5, 0.5, V_HEADS, device="cuda", dtype=torch.float16)),
    )
    # Three pools plus comparison temporaries stay comfortably under 2GiB.
    assert 8 * sum(value.numel() * value.element_size() for value in (conv, state)) < 2 * 1024**3
    return (conv, state), weights


def close(actual, expected, *, state=False):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected,
        atol=STATE_ATOL if state else OUTPUT_ATOL,
        rtol=STATE_RTOL if state else OUTPUT_RTOL)


@pytest.mark.parametrize("rows", CASES)
@pytest.mark.parametrize("stride", [48, 96])
@pytest.mark.parametrize("accepted", [1, 3, 5])
def test_corrected_actual_forward_matches_isolated_requests(runtime, rows, stride, accepted):
    module, original, candidate = runtime
    with torch.inference_mode():
        initial, weights = initialize(module, len(rows))
        mixed_pool = tuple(value.clone() for value in initial)
        reference_pool = tuple(value.clone() for value in initial)
        mixed_model = model(module, mixed_pool, weights)
        reference_model = model(module, reference_pool, weights)
        active_conv = [1 + row_id * 6 for row_id in range(len(rows))]
        active_state = [slot for row_id, (spec, _) in enumerate(rows)
                        for slot in range(1 + row_id * 6, 1 + row_id * 6 + (5 if spec else 1))]
        untouched_conv = [i for i in range(initial[0].shape[0]) if i not in active_conv]
        untouched_state = [i for i in range(initial[1].shape[0]) if i not in active_state]
        for step in range(2):
            # Change accepted source on the second call, retaining each pool.
            current_accepted = accepted if step == 0 else {1: 5, 3: 1, 5: 3}[accepted]
            qkv, b, a, backing_a, backing_b = inputs(rows, stride, step)
            gates_before = (backing_a.clone(), backing_b.clone())
            actual = invoke(candidate, mixed_model,
                metadata(module, rows, range(len(rows)), current_accepted), qkv, b, a)
            expected, start = [], 0
            for row_id, row in enumerate(rows):
                stop = start + row[1]
                expected.append(invoke(original, reference_model,
                    metadata(module, [row], [row_id], current_accepted),
                    qkv[start:stop], b[start:stop], a[start:stop]))
                start = stop
            close(actual, torch.cat(expected))
            close(mixed_pool[0], reference_pool[0])
            close(mixed_pool[1], reference_pool[1], state=True)
            assert torch.equal(mixed_pool[0][untouched_conv], initial[0][untouched_conv])
            assert torch.equal(mixed_pool[1][untouched_state], initial[1][untouched_state])
            torch.testing.assert_close(backing_a, gates_before[0], rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(backing_b, gates_before[1], rtol=0, atol=0, equal_nan=True)


def test_original_mixed_forward_is_a_failing_numerical_control(runtime):
    """Ensure the oracle detects the original reachable gate/QKV mismatch."""
    module, original, _ = runtime
    rows = CASES[0]
    with torch.inference_mode():
        initial, weights = initialize(module, len(rows))
        wrong_model = model(module, tuple(value.clone() for value in initial), weights)
        reference_model = model(module, tuple(value.clone() for value in initial), weights)
        qkv, b, a, *_ = inputs(rows, 96, 0)
        wrong = invoke(original, wrong_model, metadata(module, rows, [0, 1], 3), qkv, b, a)
        expected = torch.cat([
            invoke(original, reference_model, metadata(module, [rows[0]], [0], 3), qkv[:1], b[:1], a[:1]),
            invoke(original, reference_model, metadata(module, [rows[1]], [1], 3), qkv[1:], b[1:], a[1:]),
        ])
        assert torch.isfinite(wrong).all() and torch.isfinite(expected).all()
        assert not torch.allclose(wrong, expected, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL), (
            "negative control failed to expose original mixed gate mismatch")
