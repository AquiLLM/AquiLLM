"""Trace actual installed vLLM sampler dispatch; optional microbenchmark only.

Run only in an isolated qualification process, never in the serving process.
These timings cannot establish complete-request throughput or an A/B speedup.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json


class DispatchTrace:
    """Count successful real backend calls and restore temporary wrappers."""

    def __init__(self, module, sampler):
        self.module, self.sampler = module, sampler
        self.counts = {"flashinfer": 0, "native": 0}

    def __enter__(self):
        self.original_fi = self.module.flashinfer_sample
        self.original_native = self.sampler.forward_native

        def fi(*args, **kwargs):
            result = self.original_fi(*args, **kwargs)
            self.counts["flashinfer"] += 1
            return result

        def native(*args, **kwargs):
            result = self.original_native(*args, **kwargs)
            self.counts["native"] += 1
            return result

        self.module.flashinfer_sample = fi
        self.sampler.forward_native = native
        return self

    def __exit__(self, *exc):
        self.module.flashinfer_sample = self.original_fi
        self.sampler.forward_native = self.original_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbenchmark", action="store_true", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--vocab-size", type=int, default=248320)
    args = parser.parse_args()
    if args.iterations <= 0 or args.vocab_size < 50:
        parser.error("iterations must be positive; vocab-size must be >= 50")
    import torch
    module = importlib.import_module("vllm.v1.sample.ops.topk_topp_sampler")
    sampler = module.TopKTopPSampler()
    if getattr(sampler.forward, "__name__", None) != "forward_cuda":
        raise RuntimeError("installed sampler is not bound to the FlashInfer CUDA dispatcher")
    logits = torch.randn((1, args.vocab_size), device="cuda", dtype=torch.float32)
    k = torch.tensor([50], device="cuda", dtype=torch.int32)
    p = torch.tensor([0.9], device="cuda", dtype=torch.float32)
    result = {"kind": "sampler_microbenchmark_only", "complete_request_speedup": None,
              "vllm": importlib.metadata.version("vllm"),
              "flashinfer": importlib.metadata.version("flashinfer-python"), "cases": []}
    for name, generators, case_k, case_p in (
        ("unseeded_k50_p09", {}, k, p),
        ("unfiltered_native", {}, None, None),
        ("seeded_native", {0: torch.Generator(device="cuda").manual_seed(123)}, k, p),
    ):
        for _ in range(10):
            sampler.forward(logits.clone(), generators, case_k, case_p)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with DispatchTrace(module, sampler) as trace:
            start.record()
            for _ in range(args.iterations):
                sampler.forward(logits.clone(), generators, case_k, case_p)
            end.record()
            end.synchronize()
        result["cases"].append({"name": name, "counts": trace.counts,
                                "microseconds_including_logits_copy": start.elapsed_time(end) * 1000 / args.iterations})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
