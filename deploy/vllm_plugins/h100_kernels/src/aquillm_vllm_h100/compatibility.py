"""Exact first-wave runtime gate; unsupported versions never install adapters."""
import importlib.metadata
import os
import subprocess

GENESIS_COMMIT = "34e269301cc3df71ae4b0da00a0a159b16b4e5d8"
PACKAGES = {"vllm": "0.23.1rc1.dev748+g2dfaae752", "torch": "2.11.0+cu130",
            "triton": "3.6.0", "flashinfer-python": "0.6.13"}
MODEL = "hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16"


def verify_runtime():
    for name, expected in PACKAGES.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"unsupported {name}: {actual}, expected {expected}")
    actual = subprocess.check_output(["git", "-C", "/opt/genesis", "rev-parse", "HEAD"], text=True).strip()
    if actual != GENESIS_COMMIT:
        raise ValueError("unsupported Genesis commit")
    if os.environ.get("VLLM_MODEL") != MODEL:
        raise ValueError("model outside the qualified H100 geometry")
    for name in ("GENESIS_ENABLE_PN521_TQ_RAW_TAIL_VERIFY", "GENESIS_ENABLE_PN521_SPLIT_K",
                 "GENESIS_ENABLE_PN401_TQ_PREFILL_CONTINUATION_GUARD"):
        if os.environ.get(name) != "1":
            raise ValueError(f"required baseline guard {name} is absent")
    if os.environ.get("GENESIS_P67_BLOCK_KV") != "32":
        raise ValueError("first-wave kernels require the validated 32-token tile")
    # CUDA properties are checked lazily in worker dispatch. Registration also
    # happens in the API process before spawned workers; never initialize CUDA here.

