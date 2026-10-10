"""Exact first-wave runtime gate; unsupported versions never install adapters."""
import importlib.metadata
import os
import subprocess
import sys

GENESIS_COMMIT = "34e269301cc3df71ae4b0da00a0a159b16b4e5d8"
PACKAGES = {"vllm": "0.23.1rc1.dev748+g2dfaae752", "torch": "2.11.0+cu130",
            "triton": "3.6.0", "flashinfer-python": "0.6.13"}
# Captured baseline-packages.json: native kernel dependencies are already
# present in the baseline image; this profile installs/upgrades none of them.
NATIVE_BASELINE_PACKAGES = {**PACKAGES, "flashinfer-cubin":"0.6.13",
    "flashinfer-jit-cache":"0.6.13+cu130", "apache-tvm-ffi":"0.1.9",
    "nvidia-cutlass-dsl":"4.5.2", "nvidia-cutlass-dsl-libs-base":"4.5.2",
    "nvidia-cutlass-dsl-libs-cu13":"4.5.2"}
CANDIDATE_PACKAGES = {**PACKAGES, "flashinfer-python": "0.6.18",
                      "apache-tvm-ffi": "0.1.10",
                      "flashinfer-cubin": "0.6.18",
                      "flashinfer-jit-cache": "0.6.18+cu130",
                      "nvidia-cutlass-dsl": "4.6.2",
                      "nvidia-cutlass-dsl-libs-cu13": "4.6.2"}
MODEL = "hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16"


def model_identity(argv, env):
    """CLI is authoritative; spawned workers inherit the verified API identity."""
    model = env.get("AQUILLM_H100_MODEL_ID", env.get("VLLM_MODEL"))
    for index, value in enumerate(argv):
        if value == "--model" and index + 1 < len(argv):
            model = argv[index + 1]
        elif value.startswith("--model="):
            model = value.split("=", 1)[1]
    return model


def verify_runtime():
    profile = os.environ.get("AQUILLM_H100_RUNTIME_PROFILE", "baseline")
    if profile == "baseline":
        packages = PACKAGES
    elif profile == "native-gdn-baseline":
        packages = NATIVE_BASELINE_PACKAGES
    elif profile == "flashinfer-0.6.18":
        packages = CANDIDATE_PACKAGES
    else:
        raise ValueError(f"unknown H100 runtime profile: {profile}")
    for name, expected in packages.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"unsupported {name}: {actual}, expected {expected}")
    actual = subprocess.check_output(["git", "-C", "/opt/genesis", "rev-parse", "HEAD"], text=True).strip()
    if actual != GENESIS_COMMIT:
        raise ValueError("unsupported Genesis commit")
    if model_identity(sys.argv, os.environ) != MODEL:
        raise ValueError("model outside the qualified H100 geometry")
    os.environ["AQUILLM_H100_MODEL_ID"] = MODEL
    for name in ("GENESIS_ENABLE_PN521_TQ_RAW_TAIL_VERIFY", "GENESIS_ENABLE_PN521_SPLIT_K",
                 "GENESIS_ENABLE_PN401_TQ_PREFILL_CONTINUATION_GUARD"):
        if os.environ.get(name) != "1":
            raise ValueError(f"required baseline guard {name} is absent")
    if os.environ.get("GENESIS_P67_BLOCK_KV") != "32":
        raise ValueError("first-wave kernels require the validated 32-token tile")
    # CUDA properties are checked lazily in worker dispatch. Registration also
    # happens in the API process before spawned workers; never initialize CUDA here.
