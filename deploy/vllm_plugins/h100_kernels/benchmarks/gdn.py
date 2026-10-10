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
    parser.add_argument("--inspect", action="store_true", required=True)
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
