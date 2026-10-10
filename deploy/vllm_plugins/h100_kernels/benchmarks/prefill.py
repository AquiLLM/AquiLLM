"""Serialized complete continuation timing; measurements do not enable routes.

Run from plugin root: python benchmarks/prefill.py --max-prefix 131072.
An installed baseline adapter may be supplied as module:callable accepting
(VerifyBatch, output). It must measure the deployed continuation path; absent
that callable, timings are candidate-only and cannot establish winning regions.
"""
import argparse
import importlib
import json
import statistics
import sys
from pathlib import Path


def benchmark_cases(max_prefix):
    prefixes = sorted(set(p for p in (4096, 16384, 32768, 65536, max_prefix) if 0 < p <= max_prefix))
    return [(p, q) for p in prefixes for q in (129, 256, 512, 2048, 8192)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-prefix", type=int, required=True, help="largest prefix allowed by tested runtime")
    parser.add_argument("--prefix", type=int, help="run one prefix length instead of full grid")
    parser.add_argument("--chunk", type=int, help="run one query length instead of full grid")
    parser.add_argument("--block-q", type=int, choices=(32, 64), default=32)
    parser.add_argument("--warps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--baseline", help="module:callable for captured deployed continuation")
    args = parser.parse_args()
    if args.max_prefix < 1 or args.repetitions < 1 or args.warmup < 0:
        parser.error("positive max prefix/repetitions and nonnegative warmup are required")
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root / "tests" / "gpu")]
    import torch
    from reference import make_verify_batch
    from aquillm_vllm_h100.contracts import AttentionState
    from aquillm_vllm_h100.kernels.prefix import prefix_attention
    from aquillm_vllm_h100.kernels.merge import merge_attention_states
    from aquillm_vllm_h100.prefill import raw_chunk_attention
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("benchmark requires the coordinator's serialized SM90 runtime")
    baseline = None
    if args.baseline:
        mod, name = args.baseline.split(":", 1)
        baseline = getattr(importlib.import_module(mod), name)
    cases = benchmark_cases(args.max_prefix)
    if args.prefix is not None or args.chunk is not None:
        if args.prefix is None or args.chunk is None or not 0 < args.prefix <= args.max_prefix or args.chunk < 1:
            parser.error("--prefix and --chunk must be provided together within allowed lengths")
        cases = [(args.prefix, args.chunk)]
    for prior, length in cases:
        batch = make_verify_batch([prior], length=length)
        q = batch.q[0]
        prefix = AttentionState(torch.empty_like(q, dtype=torch.float32), torch.empty(q.shape[:2], device="cuda"))
        output = torch.empty_like(q)

        def candidate():
            prefix_attention(q, batch.kv_cache, batch.block_table[0], prior, batch.scale, batch.spec,
                             prefix, block_q=args.block_q, num_warps=args.warps)
            chunk = raw_chunk_attention(q, batch.raw_k[0], batch.raw_v[0], batch.scale)
            merge_attention_states(prefix, chunk, output)

        def measure(operation):
            for _ in range(args.warmup):
                operation()
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.repetitions):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                operation()
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end))
            return {"median_ms": statistics.median(samples), "min_ms": min(samples), "samples_ms": samples}

        record = {"prefix": prior, "chunk": length, "block_q": args.block_q, "warps": args.warps,
                  "candidate": measure(candidate), "profile_enabled": False,
                  "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                  "prefix_state_bytes": prefix.output.numel() * 4 + prefix.lse.numel() * 4,
                  "classification": "candidate_only"}
        if baseline is not None:
            record["baseline"] = measure(lambda: baseline(batch, output))
            record["classification"] = "microbenchmark_pair_requires_ttft_validation"
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
