"""Serial fixed/adaptive split-K sweep; all buffers and JIT work precede timing.

Run inside the pinned H100 overlay. Measurements are evidence only: this command
never creates an enabled profile or changes serving configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys


def parse_split_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(piece.strip()) for piece in value.split(","))
    if not counts or any(count <= 0 for count in counts) or len(set(counts)) != len(counts):
        raise ValueError("split counts must be unique positive integers")
    return counts


def scratch_shape(batch: int, hkv: int, length: int, gqa: int, dim: int, splits: int) -> tuple[int, ...]:
    pad = lambda size: 1 << (size - 1).bit_length()
    return batch, hkv, splits + 1, pad(length), pad(gqa), dim + 1


def scratch_bytes(batch: int, hkv: int, length: int, gqa: int, dim: int, splits: int) -> int:
    return math.prod(scratch_shape(batch, hkv, length, gqa, dim, splits)) * 4


def candidate_orders(candidates, *, rounds: int, seed: int):
    generator = random.Random(seed)
    orders = []
    for _ in range(rounds):
        order = list(candidates)
        generator.shuffle(order)
        orders.append(order)
    return orders


def fixed_stage_launches(batch, splits, mid, output):
    """Expose the exact baseline stage boundaries without allocator timing."""
    from sndr.engines.vllm.kernels_legacy import p67_multi_query_kernel as baseline

    b, length, hq, dim = batch.q.shape
    hkv = batch.spec.num_kv_heads
    gqa = hq // hkv
    stage1 = baseline._get_stage1_splitk_kernel()
    stage2 = baseline._get_stage2_kernel()
    if stage1 is None or stage2 is None:
        raise RuntimeError("baseline split-K kernels unavailable")
    config = baseline._autoconfig(9, 0, dim)
    if config["BLOCK_KV"] != 32 or baseline._detect_fp8_mode() != 0:
        raise RuntimeError("benchmark requires BLOCK_KV=32 and H100 e4m3fn keys")
    strides1 = (*batch.q.stride(), *batch.kv_cache.stride()[:3], batch.block_table.stride(0),
                *batch.raw_k.stride(), *batch.raw_v.stride(), *mid.stride())
    geometry = dict(K_PLUS_1=length, BLOCK_D=dim, HEAD_DIM=dim,
                    HEADS_PER_KV=gqa, BLOCK_QH=mid.shape[4], Hq_TOTAL=hq, KP1_PAD=mid.shape[3])

    def launch1():
        stage1[(b, hkv, splits + 1)](
            batch.q, batch.kv_cache, batch.block_table, batch.seq_lens,
            batch.raw_k, batch.raw_v, mid, *strides1,
            SCALE=batch.scale, BLOCK_SIZE=batch.spec.block_size, BLOCK_KV=32,
            KPS=batch.spec.key_packed_size, VAL_DATA_BYTES=batch.spec.value_data_bytes,
            NUM_SPLITS=splits, RAW_SID=splits, FP8_E4B15=0, DOT_FP16=0,
            **geometry, num_warps=config["num_warps"], num_stages=config["num_stages"],
        )

    def launch2():
        stage2[(b, hkv)](mid, output, *mid.stride(), *output.stride(),
                          NUM_SPLITS_P1=splits + 1, **geometry)

    return launch1, launch2


def timed_runner(function, *, mode, iterations, warmup):
    import torch

    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            function()
        run = graph.replay
    else:
        run = function
    # Precreate events; only replay/launch is between the timed events.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def measure():
        start.record()
        for _ in range(iterations):
            run()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000 / iterations

    return measure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=parse_split_counts, default=(7, 15, 31, 47, 63))
    parser.add_argument("--contexts", type=parse_split_counts, default=(1, 32, 2048, 8192, 32768, 131072))
    parser.add_argument("--batches", type=parse_split_counts, default=(1, 2, 4))
    parser.add_argument("--mode", choices=("graph", "eager"), default="graph")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=179)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.rounds, args.iterations, args.warmup) <= 0:
        parser.error("rounds, iterations, and warmup must be positive")
    # Set once before the first baseline builder call, never in capture/replay.
    os.environ["GENESIS_P67_BLOCK_KV"] = "32"
    if os.environ.get("GENESIS_P67_DOT_PRECISION", "tf32x3").strip().lower() == "fp16":
        parser.error("fixed precision requires the existing tf32/tf32x3 path")
    import torch
    import triton
    from sndr.engines.vllm.kernels_legacy import p67_multi_query_kernel as baseline

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("requires the reserved H100 GPU")
    properties = torch.cuda.get_device_properties(0)
    if "H100" not in properties.name or properties.multi_processor_count != 132:
        raise RuntimeError("requires H100 80GB / 132 SM runtime")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "gpu"))
    from reference import assert_close, make_verify_batch, reference_verify

    identity = dict(gpu=properties.name, sm_count=properties.multi_processor_count,
                    torch=torch.__version__, triton=triton.__version__, dtype="float16",
                    length=5, gqa=6, head_dim=256, block_kv=32,
                    dot_precision="committed QK=tf32, PV=tf32x3; raw=FP32 scalar; output=FP16",
                    genesis_source_sha256=hashlib.sha256(Path(inspect.getfile(baseline)).read_bytes()).hexdigest())
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    rows = []
    orders = candidate_orders(args.splits, rounds=args.rounds, seed=args.seed)
    for batch_size in args.batches:
        for prior in args.contexts:
            batch = make_verify_batch([prior] * batch_size, length=5, dtype=torch.float16, seed=args.seed)
            expected = reference_verify(batch)
            runners = {}
            buffers = {}
            for splits in args.splits:
                mid = torch.empty(scratch_shape(batch_size, batch.spec.num_kv_heads, 5, 6, 256, splits),
                                  device="cuda", dtype=torch.float32)
                output = torch.empty_like(batch.q)
                launch1, launch2 = fixed_stage_launches(batch, splits, mid, output)
                total = lambda one=launch1, two=launch2: (one(), two())
                # Also qualify the public unchanged launcher with these buffers.
                baseline.call_p67_splitk(batch.q, batch.kv_cache, batch.block_table, batch.seq_lens,
                                        batch.raw_k, batch.raw_v, batch.scale, batch.spec.block_size,
                                        batch.spec.key_packed_size, batch.spec.value_data_bytes,
                                        output=output, num_splits=splits, mid_o=mid)
                assert_close(output, expected, batch.q.dtype)
                buffers[splits] = (mid, output)
                runners[splits] = {name: timed_runner(fn, mode=args.mode, iterations=args.iterations, warmup=args.warmup)
                                   for name, fn in (("stage1", launch1), ("reduction", launch2), ("total", total))}
            samples = {splits: {name: [] for name in runners[splits]} for splits in args.splits}
            for order in orders:
                for splits in order:
                    for name, runner in runners[splits].items():
                        samples[splits][name].append(runner())
            for splits in args.splits:
                mid, output = buffers[splits]
                row = dict(batch=batch_size, prior_len=prior, committed_splits=splits,
                           raw_slot=splits, workspace_bytes=mid.numel() * mid.element_size(),
                           correct=True, samples_us=samples[splits],
                           median_us={name: statistics.median(values) for name, values in samples[splits].items()})
                rows.append(row)
                print(json.dumps(row), flush=True)
    result = dict(schema=1, identity=identity, baseline_manifest=manifest,
                  qualification="microbenchmark only; serving acceptance/TPS and runtime matching required",
                  baseline_splits=15, mode=args.mode, seed=args.seed, candidate_order=orders,
                  iterations=args.iterations, warmup=args.warmup, measurements=rows,
                  selected_profile=None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
