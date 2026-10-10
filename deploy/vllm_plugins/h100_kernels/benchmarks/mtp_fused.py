"""Synthetic total-verifier comparison on the coordinator's reserved H100.

Run in the pinned serving image. Timing includes both stage one and stage two.
Serving acceptance/TPS requires the coordinator's fixed quality/replay cases;
this microbenchmark never equates synthetic query rows to accepted tokens.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import statistics
import sys


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", default="2048,8192,32768")
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--splits", type=int, default=15)
    parser.add_argument("--buckets", default="", help="inclusive upper:splits pairs")
    parser.add_argument("--reference-stage1", help="module:function for adaptive baseline")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _measure(torch, callback, warmup, samples, graph):
    for _ in range(warmup):
        callback()
    torch.cuda.synchronize()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            callback()
        callback = capture.replay
    # Preallocate events. GPU elapsed time excludes host launch gaps in eager
    # execution only insofar as they overlap existing device work.
    events = [(torch.cuda.Event(enable_timing=True),
               torch.cuda.Event(enable_timing=True)) for _ in range(samples)]
    for start, end in events:
        start.record()
        callback()
        end.record()
    torch.cuda.synchronize()
    us = sorted(start.elapsed_time(end) * 1000 for start, end in events)
    return {"median_us": statistics.median(us),
            "p95_us": us[min(len(us) - 1, int(len(us) * 0.95))],
            "samples": samples}


def _resources(jit, torch):
    # Best-effort compiler diagnostics, separate from numerical/timing gates.
    result = []
    caches = getattr(jit, "device_caches", {})
    device_cache = caches.get(torch.cuda.current_device())
    if device_cache:
        for compiled in device_cache[0].values():
            result.append({"registers": getattr(compiled, "n_regs", None),
                           "spills": getattr(compiled, "n_spills", None),
                           "shared_bytes": getattr(compiled.metadata, "shared", None)})
    return result


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.samples < 1 or args.warmup < 1 or args.batch_size < 1:
        raise ValueError("samples, warmup and batch size must be positive")
    import torch
    import triton

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Run this benchmark on the reserved H100 SM90")
    if "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("This benchmark is keyed to the H100 runtime")
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root / "tests" / "gpu")]
    from aquillm_vllm_h100.contracts import SplitPlan
    from aquillm_vllm_h100.kernels.mtp_fused import _kernel, launch_fused_stage1
    from aquillm_vllm_h100.kernels.reduce import reduce_verify_partials
    from reference import assert_close, make_verify_batch, reference_verify
    from sndr.engines.vllm.kernels_legacy import p67_multi_query_kernel as p67

    # This isolated benchmark fixes the same precision/tile policy for both
    # candidates. Environment mutation occurs once before any JIT or capture.
    os.environ.update(GENESIS_P67_USE_FUSED="0", GENESIS_P67_DOT_PRECISION="tf32x3",
                      GENESIS_P67_BLOCK_KV="32", GENESIS_P67_NUM_WARPS="8",
                      GENESIS_P67_NUM_STAGES="2")
    p67._CACHED_KERNEL = None
    p67._CACHED_STAGE1_SPLITK = None
    buckets = tuple(tuple(map(int, pair.split(":")))
                    for pair in args.buckets.split(",") if pair)
    plan = SplitPlan(args.splits, buckets)
    reference_stage1 = None
    if args.reference_stage1:
        module, function = args.reference_stage1.rsplit(":", 1)
        reference_stage1 = getattr(importlib.import_module(module), function)
    if buckets and reference_stage1 is None:
        raise ValueError("Adaptive comparison requires --reference-stage1 module:function")
    rows = []
    for prior in map(int, args.prior.split(",")):
        batch = make_verify_batch([prior] * args.batch_size, length=args.length,
                                  dtype=getattr(torch, args.dtype))
        b, length, hq, dim = batch.q.shape
        hk, group = batch.spec.num_kv_heads, hq // batch.spec.num_kv_heads
        shape = (b, hk, plan.max_splits + 1, triton.next_power_of_2(length),
                 triton.next_power_of_2(group), dim + 1)
        mid = torch.empty(shape, dtype=torch.float32, device="cuda")
        baseline_mid = torch.empty_like(mid)
        out, baseline_out, single_out = (torch.empty_like(batch.q) for _ in range(3))
        upstream_args = (batch.q, batch.kv_cache, batch.block_table, batch.seq_lens,
                         batch.raw_k, batch.raw_v, batch.scale, batch.spec.block_size,
                         batch.spec.key_packed_size, batch.spec.value_data_bytes)

        def candidate():
            launch_fused_stage1(batch, plan, mid)
            reduce_verify_partials(batch, mid, out)

        def baseline():
            if reference_stage1:
                reference_stage1(batch, plan, baseline_mid)
                reduce_verify_partials(batch, baseline_mid, baseline_out)
            else:
                p67.call_p67_splitk(*upstream_args, output=baseline_out,
                                   num_splits=plan.max_splits, mid_o=baseline_mid)

        def single():
            p67.call_p67_attention(*upstream_args, output=single_out, use_raw_tail=1)

        candidate()
        baseline()
        single()
        expected = reference_verify(batch)
        for actual in (out, baseline_out, single_out):
            assert_close(actual, expected, batch.q.dtype)
        assert_close(out, baseline_out, batch.q.dtype)
        baseline_time = _measure(torch, baseline, args.warmup, args.samples, args.graph)
        candidate_time = _measure(torch, candidate, args.warmup, args.samples, args.graph)
        single_time = _measure(torch, single, args.warmup, args.samples, args.graph)
        rows.append({"prior": prior, "baseline_split": baseline_time,
                     "fused": candidate_time, "baseline_single": single_time,
                     "speedup_vs_split": baseline_time["median_us"] / candidate_time["median_us"],
                     "max_absolute_difference_vs_split":
                         (out.float() - baseline_out.float()).abs().max().item()})
    report = {"kind": "synthetic_total_verifier", "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "triton": triton.__version__,
              "dtype": args.dtype, "length": args.length, "batch_size": args.batch_size,
              "max_splits": args.splits, "buckets": buckets,
              "tile": 32, "qk": "tf32", "pv": "tf32x3", "graph": args.graph,
              "compiled_resources": _resources(_kernel(), torch), "rows": rows,
              "accepted_output_tokens_per_second": None,
              "acceptance_status": "requires separate fixed serving quality/replay measurement"}
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return report


if __name__ == "__main__":
    main()
