"""CPU-only contract reproducer using extracted, deployed source functions.

No vLLM imports, GPU allocation, kernel compilation, or recurrence execution.
The real reorder function and GDN metadata branch run on CPU. The real Qwen
_forward_core runs up to a stubbed recurrence call, which records its inputs.
--verify-proposed-fix modifies only the in-memory AST, never deployed source.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"


def find_function(filename, name):
    tree = ast.parse((SOURCES / filename).read_text(encoding="utf-8"))
    matches = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(matches) == 1, (filename, name, len(matches))
    node = copy.deepcopy(matches[0])
    node.decorator_list = []
    return node


def compile_function(node, namespace):
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<captured-source>", "exec"), namespace)
    return namespace[node.name]


class InputBatch:
    def __init__(self, rows):
        self.req_ids = [r["id"] for r in rows]
        self.num_computed_tokens_cpu = np.array([r["computed"] for r in rows])
        self.num_prompt_tokens = np.array([r["prompt"] for r in rows])

    def swap_states(self, left, right):
        self.req_ids[left], self.req_ids[right] = self.req_ids[right], self.req_ids[left]
        for value in (self.num_computed_tokens_cpu, self.num_prompt_tokens):
            value[left], value[right] = value[right], value[left]


class Captured(Exception):
    def __init__(self, kwargs):
        self.kwargs = kwargs


def recurrence(**kwargs):
    raise Captured(kwargs)


def post_conv(**kwargs):
    x = kwargs["conv_output"][:, None, :]
    return x, x, x, kwargs["a"][:, None, :], kwargs["b"][:, None, :]


def proposed_fix(node):
    """Narrow mixed-branch gate gather, pure-spec original views retained."""
    changed = 0
    for child in ast.walk(node):
        if (isinstance(child, ast.If)
                and ast.unparse(child.test) == "spec_sequence_masks is not None"
                and child.body and isinstance(child.body[0], ast.If)
                and "attn_metadata.num_prefills == 0" in ast.unparse(child.body[0].test)):
            pure_or_mixed = child.body[0]
            pure_or_mixed.body.extend(ast.parse("a_spec = a\nb_spec = b").body)
            pure_or_mixed.orelse.extend(ast.parse(
                "a_spec = a.index_select(0, spec_token_indx)\n"
                "b_spec = b.index_select(0, spec_token_indx)").body)
            changed += 1
        if (isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "fused_sigmoid_gating_delta_rule_update"
                and any(k.arg == "q" and isinstance(k.value, ast.Name)
                        and k.value.id == "query_spec" for k in child.keywords)):
            for kw in child.keywords:
                if kw.arg in ("a", "b"):
                    kw.value = ast.Name(id=kw.arg + "_spec", ctx=ast.Load())
            changed += 1
    assert changed == 2, changed
    return node


def metadata_branch():
    tree = ast.parse((SOURCES / "gdn_attn.py").read_text(encoding="utf-8"))
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.If)
               and ast.unparse(n.test) == "spec_sequence_masks is None"
               and n.lineno == 228]
    assert len(matches) == 1
    return ast.fix_missing_locations(ast.Module(
        body=copy.deepcopy(matches[0].orelse), type_ignores=[]))


def run_case(name, rows, fixed):
    by_id = {r["id"]: r for r in rows}
    batch = InputBatch(rows)
    scheduler = SimpleNamespace(num_scheduled_tokens={r["id"]: r["tokens"] for r in rows})
    reorder = compile_function(find_function("utils.py", "reorder_batch_to_split_decodes_and_prefills"),
                               {"np": np})
    swapped = reorder(batch, scheduler, decode_threshold=1)
    ordered = [by_id[rid] for rid in batch.req_ids]
    lens = torch.tensor([r["tokens"] for r in ordered], dtype=torch.int32)
    qsl = torch.cat([torch.zeros(1, dtype=torch.int32), lens.cumsum(0).to(torch.int32)])
    drafts = torch.tensor([r["drafts"] for r in ordered], dtype=torch.int32)
    spec = drafts >= 0
    assert drafts[spec].sum() > 0
    ns = dict(torch=torch, self=SimpleNamespace(num_spec=4,
              spec_token_arange=torch.arange(int(qsl[-1])), non_spec_token_indx=torch.empty(0, dtype=torch.int64)),
              spec_sequence_masks=spec, spec_sequence_masks_cpu=spec,
              query_start_loc=qsl, query_start_loc_cpu=qsl,
              num_spec_decodes=int(spec.sum()), block_table_tensor=torch.tensor(
                  [r["slots"] for r in ordered], dtype=torch.int32),
              num_accepted_tokens=torch.tensor([r["accepted"] for r in ordered], dtype=torch.int32))
    exec(compile(metadata_branch(), "<captured-gdn-metadata>", "exec"), ns)
    md = SimpleNamespace(**{k: ns[k] for k in (
        "spec_query_start_loc", "non_spec_query_start_loc", "spec_sequence_masks",
        "spec_token_indx", "non_spec_token_indx", "spec_state_indices_tensor",
        "non_spec_state_indices_tensor", "num_prefills", "num_decodes",
        "num_spec_decodes", "num_decode_tokens", "num_accepted_tokens")})
    md.has_initial_state = torch.ones(len(ordered), dtype=torch.bool)
    md.num_actual_tokens = int(qsl[-1])
    token_ids = torch.cat([torch.arange(r["base"], r["base"] + r["tokens"])
                           for r in ordered]).to(torch.float32)[:, None]
    a, b = token_ids + 1000, token_ids + 2000
    model = SimpleNamespace(prefix="gdn", enable_packed_recurrent_decode=False,
        kv_cache=(torch.zeros(16, 1, 1), torch.zeros(16, 1, 1, 1)),
        conv1d=SimpleNamespace(weight=torch.zeros(1, 1, 1), bias=None),
        activation="silu", num_k_heads=1, tp_size=1, head_k_dim=1, head_v_dim=1,
        A_log=torch.zeros(1), dt_bias=torch.zeros(1),
        rearrange_mixed_qkv=lambda x: (None, None, None) if x is None else
            (x[None, :, None, :],) * 3)
    node = find_function("qwen_gdn_linear_attn.py", "_forward_core")
    if fixed:
        node = proposed_fix(node)
    forward = compile_function(node, dict(torch=torch, GDNAttentionMetadata=SimpleNamespace,
        get_forward_context=lambda: SimpleNamespace(attn_metadata={"gdn": md}),
        is_conv_state_dim_first=lambda: True,
        causal_conv1d_update=lambda x, *args, **kwargs: x,
        causal_conv1d_fn=lambda x, *args, **kwargs: x,
        fused_post_conv_prep=post_conv,
        fused_sigmoid_gating_delta_rule_update=recurrence))
    try:
        forward(model, token_ids, b, a, torch.empty_like(token_ids))
    except Captured as captured:
        call = captured.kwargs
    else:
        raise AssertionError("Spec recurrence was not reached")
    n = call["q"].shape[1]
    ids = call["q"].flatten()
    actual_a, actual_b = call["a"][:n].flatten(), call["b"][:n].flatten()
    matched = bool(torch.equal(actual_a, ids + 1000) and torch.equal(actual_b, ids + 2000))
    pure = md.num_prefills == 0 and md.num_decodes == 0
    if fixed or name in ("pure_spec", "spec_then_new_prefill"):
        assert matched, name
    else:
        assert not matched, name
    if pure:
        assert call["a"].data_ptr() == a.data_ptr()
        assert call["b"].data_ptr() == b.data_ptr()
    return dict(case=name, in_memory_fix=fixed, source_order=[r["id"] for r in rows],
                reordered=batch.req_ids, reorder_swapped=swapped,
                query_start_loc=qsl.tolist(), draft_counts=drafts.tolist(),
                spec_sequence_mask=spec.tolist(), spec_token_indices=md.spec_token_indx.tolist(),
                recurrence_qkv_token_ids=ids.tolist(), recurrence_a=actual_a.tolist(),
                expected_a=(ids + 1000).tolist(), recurrence_b=actual_b.tolist(),
                expected_b=(ids + 2000).tolist(), gates_match_qkv=matched,
                compacted_query_start_loc=call["cu_seqlens"].tolist(),
                state_indices=call["ssm_state_indices"].tolist(),
                accepted=call["num_accepted_tokens"].tolist(), pure_spec_zero_copy=pure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-proposed-fix", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = dict(id="mtp", computed=100, prompt=10, tokens=5, drafts=4,
                slots=[6, 7, 8, 9, 10], accepted=2, base=200)
    prefill = dict(id="chunk_tail", computed=100, prompt=101, tokens=1, drafts=-1,
                   slots=[1, 0, 0, 0, 0], accepted=1, base=100)
    fresh = dict(prefill, id="new_prefill", computed=0, prompt=3, tokens=3)
    long = dict(prefill, id="long_chunk", prompt=110, tokens=10)
    ordinary = dict(prefill, id="ordinary_no_draft_entry", prompt=10)
    other = dict(spec, id="other_mtp", slots=[11, 12, 13, 14, 15], base=300)
    cases = [
        ("pure_spec", [spec, other]),
        ("spec_then_new_prefill", [spec, fresh]),
        ("spec_then_chunk_tail", [spec, prefill]),
        ("chunk_tail_then_spec", [prefill, spec]),
        ("long_chunk_then_spec", [long, spec]),
        ("ordinary_no_draft_entry_then_spec", [ordinary, spec]),
    ]
    results = [run_case(name, rows, args.verify_proposed_fix) for name, rows in cases]
    evidence = dict(cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        scope="CPU source-boundary proof; no GPU kernels, recurrence, or serving run",
        source_sha256={name: hashlib.sha256((SOURCES / name).read_bytes()).hexdigest()
                       for name in ("utils.py", "gdn_attn.py", "qwen_gdn_linear_attn.py")},
        cases=results)
    if args.output:
        args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(dict(output=str(args.output),
            cases=[dict(case=c["case"], gates_match_qkv=c["gates_match_qkv"]) for c in results])))
    else:
        print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
