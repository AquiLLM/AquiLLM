"""Read-only installed GDN source probe. Never launches a CUDA kernel.

Run in the pinned image with the overlay source on PYTHONPATH:
    python benchmarks/gdn.py --inspect
Source/symbol evidence is emitted as JSON for the serialized GPU coordinator.
This is a capability experiment, not a throughput benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--benchmark", action="store_true",
                      help="time full T5 adapter and original, eager and CUDA graph")
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.benchmark:
        benchmark(args.repeats)
        return
    from aquillm_vllm_h100.gdn.capability import inspect as inspect_capability

    result = {"kind": "source_only_no_gpu_launch", "packages": {}, "modules": {}}
    for package in ("vllm", "torch", "triton", "flashinfer-python"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    decision = inspect_capability()
    result["decision"] = {"eligible": decision.eligible, "reason": decision.reason}
    for name in (
        "flashinfer.gdn_decode", "flashinfer.gdn_kernels.gdn_decode_mtp",
        "flashinfer.gdn_kernels.gdn_decode_pretranspose",
        "flashinfer.gdn_kernels.gdn_decode_nontranspose",
        "flashinfer.gdn_kernels.gdn_decode_bf16_state",
        "vllm.model_executor.layers.fla.ops.fused_recurrent",
        "vllm.model_executor.layers.fla.ops.fused_sigmoid_gating",
    ):
        try:
            module = importlib.import_module(name)
            source = inspect.getsource(module)
            symbols = {}
            for symbol, value in vars(module).items():
                if callable(value) and ("gated_delta" in symbol or "mtp" in symbol or "decode" in symbol):
                    try:
                        symbols[symbol] = str(inspect.signature(value))
                    except (ValueError, TypeError):
                        pass
            result["modules"][name] = {
                "path": module.__file__, "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "symbols": symbols, "source": source,
            }
        except (ImportError, RuntimeError, OSError) as error:
            result["modules"][name] = {"error": str(error)}
    print(json.dumps(result, indent=2))


def benchmark(repeats):
    # Genesis rewrites source before any backend/FlashInfer imports. Capture
    # the actual Qwen baseline alias, even if the opt-in bridge is installed.
    import sndr.plugin
    sndr.plugin.register()
    import torch
    from aquillm_vllm_h100.gdn.adapter import make_adapter, supported, capture_original
    original = capture_original()
    if repeats < 1:
        raise ValueError("repeats must be positive")
    torch.manual_seed(2026)
    size = 48*128*128
    stride = size+128
    backing = torch.zeros(128+9*stride, device="cuda")
    state = backing.as_strided((9,48,128,128), (stride,16384,128,1), storage_offset=128)
    state.copy_(torch.randn_like(state)*0.03)
    seed = backing.clone()
    def activation(shape):
        return (torch.randn(shape,device="cuda")*0.2).half()
    kwargs = dict(A_log=torch.full((48,),-1.0,device="cuda"),
                  dt_bias=torch.full((48,),-0.2,device="cuda",dtype=torch.float16),
                  q=activation((1,5,16,128)), k=activation((1,5,16,128)),
                  v=activation((1,5,48,128)), a=activation((5,48)), b=activation((5,48)),
                  initial_state=state, use_qk_l2norm_in_kernel=True,
                  cu_seqlens=torch.tensor([0,5],device="cuda",dtype=torch.int32),
                  ssm_state_indices=torch.tensor([[1,2,3,4,5,7,8]],device="cuda",dtype=torch.int32),
                  num_accepted_tokens=torch.tensor([3],device="cuda",dtype=torch.int32))
    if not supported(**kwargs):
        raise ValueError("full adapter benchmark requires eligible H100 contract")
    result = {"kind":"full_adapter_including_pack_convert_checkpoints_scatter",
              "T":5,"shape":[1,5,16,48,128],"state_stride":list(state.stride()),
              "repeats":repeats,"packages":{},"timings_us":{}}
    for name in ("flashinfer-python","nvidia-cutlass-dsl","vllm","torch","triton"):
        result["packages"][name] = importlib.metadata.version(name)
    for name, function in (("original",original),("full_adapter",make_adapter(original))):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(20):
                function(**kwargs)
            backing.copy_(seed)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                output = function(**kwargs)
            end.record()
            end.synchronize()
            eager = start.elapsed_time(end)*1000/repeats
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = function(**kwargs)
            backing.copy_(seed)
            start.record()
            for _ in range(repeats):
                graph.replay()
            end.record()
            end.synchronize()
            graphed = start.elapsed_time(end)*1000/repeats
        torch.cuda.current_stream().wait_stream(stream)
        result["timings_us"][name] = {"eager":eager,"cuda_graph":graphed}
    result["speedup"] = {mode:result["timings_us"]["original"][mode]/
                         result["timings_us"]["full_adapter"][mode]
                         for mode in ("eager","cuda_graph")}
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
