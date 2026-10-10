#!/usr/bin/env python3
"""Native build diagnostics and fail-closed experimental storage preflight.

--base-identity checks the inherited serving tuple, --native also imports compiled
extensions. --plan performs resource/endpoint checks then always refuses readiness.
No checkpoint is loaded. A TCP connection is not an LMCache protocol handshake.
"""
import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys

from kv_cache_config import ConfigurationError
from kv_storage_layout import validate_layout
from kv_storage_plan import PINS, plan_storage, require_storage_ready


def validate_identity(actual):
    for key in ("python", "torch", "vllm", "transformers", "huggingface-hub", "cxx11_abi", "cuda_image", "genesis"):
        if actual.get(key) != PINS[key]:
            raise ConfigurationError(f"runtime {key} mismatch: expected {PINS[key]!r}, got {actual.get(key)!r}; rebuild from pinned Genesis image")


def base_identity():
    try:
        import torch
        result = {key: metadata.version(key) for key in ("torch", "vllm", "transformers", "huggingface-hub")}
        result.update(python=platform.python_version(), cxx11_abi=torch.compiled_with_cxx11_abi(),
                      cuda_image=os.environ.get("CUDA_VERSION"),
                      genesis=subprocess.check_output(["git", "-C", "/opt/genesis", "rev-parse", "HEAD"], text=True).strip())
        validate_identity(result)
        importlib.import_module("aquillm_vllm_h100")
        if not any(ep.value == "sndr.plugin:register" for ep in metadata.entry_points(group="vllm.general_plugins")):
            raise ConfigurationError("Genesis plugin entry point absent")
        return result
    except (OSError, ImportError, KeyError, metadata.PackageNotFoundError, subprocess.CalledProcessError) as exc:
        raise ConfigurationError(f"cannot inspect pinned serving identity: {exc}") from exc


def import_required_native(name):
    try:
        return importlib.import_module(name)
    except (ImportError, OSError, RuntimeError) as exc:
        raise ConfigurationError(f"native module {name} unavailable or ABI-incompatible; build Dockerfile.kv-storage against the pinned image: {exc}") from exc


def native_identity():
    actual = base_identity()
    try:
        lmcache_version = metadata.version("lmcache")
    except metadata.PackageNotFoundError as exc:
        raise ConfigurationError("LMCache 0.5.5 candidate is not installed; build Dockerfile.kv-storage against the pinned Genesis image") from exc
    if lmcache_version != "0.5.5":
        raise ConfigurationError("LMCache 0.5.5 candidate required; bundled 0.5.0 misclassifies packed K8V4")
    for source, expected in (("/opt/LMCache", PINS["lmcache"]), ("/opt/Mooncake", PINS["mooncake"])):
        observed = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
        if observed != expected:
            raise ConfigurationError(f"native source pin mismatch at {source}")
    import_required_native("lmcache.cuda_ops")
    mooncake = import_required_native("lmcache.lmcache_mooncake")
    for symbol in ("L1RegistrationConfig", "LMCacheMooncakeClient"):
        if not hasattr(mooncake, symbol):
            raise ConfigurationError(f"native Mooncake C++ symbol missing: {symbol}")
    actual.update(lmcache=lmcache_version, native_import="passed", gpu_roundtrip="not-validated")
    return actual


def check_endpoint(host, port):
    try:
        with socket.create_connection((host, port), timeout=2):
            return "tcp-only-not-handshake"
    except OSError as exc:
        raise ConfigurationError(f"required service {host}:{port} unreachable; verify Compose DNS and IPC peer configuration: {exc}") from exc


def check_disk(path, cap):
    path = Path(path)
    if not path.is_dir() or not os.access(path, os.W_OK):
        raise ConfigurationError("SSD path must be an existing writable directory owned by the LMCache process")
    if shutil.disk_usage(path).free < cap:
        raise ConfigurationError("disk free space is below configured physical SSD cap plus reserve")


def inspect_allocations(layout, tensors):
    """Inspect actual tensors by group; only geometry, never a readiness permit.

    Attention lists contain one 4D tensor per layer. Recurrent lists contain
    (conv, ssm) tensors per layer, both striding over the same byte backing page.
    This helper is not wired to the upstream registration callback yet.
    """
    validate_layout(layout)
    for index, group in enumerate(layout["groups"]):
        registered = tensors.get(index, [])
        for entry in registered:
            if group["kind"] == "attention":
                if list(entry.shape) != group["shape"] or list(entry.stride()) != group["stride"]:
                    raise ConfigurationError("registered attention shape/stride differs from layout record")
                if str(entry.dtype) != "torch.uint8":
                    raise ConfigurationError("registered attention dtype must be uint8")
                if entry.device.type != "cuda":
                    raise ConfigurationError("registered attention must be CUDA resident")
            else:
                conv, ssm = entry
                if conv.untyped_storage().data_ptr() != ssm.untyped_storage().data_ptr():
                    raise ConfigurationError("recurrent conv/SSM must share backing storage")
                for tensor, state in zip(entry, group["states"]):
                    if tensor.device.type != "cuda" or str(tensor.dtype) != "torch.bfloat16":
                        raise ConfigurationError("recurrent tensor must be CUDA bfloat16")
                    if list(tensor.shape[1:]) != state["shape"] or tensor.stride(0) * tensor.element_size() != group["page_bytes"]:
                        raise ConfigurationError("recurrent shape/page stride mismatch")
                base = conv.storage_offset() * conv.element_size()
                for tensor, state in zip(entry, group["states"]):
                    if tensor.storage_offset() * tensor.element_size() - base != state["offset_bytes"]:
                        raise ConfigurationError("recurrent relative storage offset mismatch")
        if len(registered) != group["layers"]:
            raise ConfigurationError("registered group layer count differs from layout record")
    return {"geometry": "checked", "gpu_roundtrip": "not-validated", "ready": False}


def check_resources(plan):
    limits = plan["limits"]
    if not hasattr(os, "sysconf"):
        raise ConfigurationError("Linux host/cgroup resource inspection required for storage preflight")
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    if plan["memory"]["total_bytes"] + limits["reserve_bytes"] > available:
        raise ConfigurationError("current host RAM headroom insufficient")
    max_file, current_file = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
    if not max_file.exists() or max_file.read_text().strip() == "max":
        raise ConfigurationError("bounded cgroup v2 memory limit required")
    if plan["memory"]["total_bytes"] > int(max_file.read_text()) - int(current_file.read_text()):
        raise ConfigurationError("current container memory headroom insufficient")
    if plan["profile"]["storage_mode"] == "mooncake":
        check_disk(limits["ssd_path"], plan["profile"]["tier_budgets"]["ssd"] + limits["reserve_bytes"])


def reconstruct_plan(raw):
    """Never execute caller-provided argv or environment from a JSON document."""
    profile = raw["profile"]
    env = {"KV_CACHE_STORAGE_MODE": profile["storage_mode"], "KV_CACHE_EXECUTION_MODE": profile["execution_mode"],
           "KV_CACHE_TARGET_ACTIVE_SEQUENCES": str(profile["active_sequences"]),
           "KV_CACHE_RETAINED_CONTEXTS": str(profile["retained_contexts"]),
           "VLLM_MAX_MODEL_LEN": str(profile["context_tokens"])}
    env.update({f"KV_CACHE_{key.upper()}_GIB": value for key, value in profile["tier_budget_requests"].items() if value is not None})
    plan = plan_storage(env, raw["layout"], raw["limits"])
    if plan is None:
        raise ConfigurationError("storage preflight requires local or mooncake plan")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base-identity", action="store_true")
    mode.add_argument("--native", action="store_true")
    mode.add_argument("--plan", type=Path)
    mode.add_argument("--server-plan", type=Path, help="start experimental infrastructure only; never a model")
    mode.add_argument("--master-plan", type=Path, help="start bounded metadata master only")
    args = parser.parse_args()
    try:
        path = args.plan or args.server_plan or args.master_plan
        if path:
            plan = reconstruct_plan(json.loads(path.read_text()))
            profile = plan["profile"]
            native_identity()
            if args.master_plan:
                if plan["master_argv"] is None:
                    raise ConfigurationError("backend mismatch: local mode has no Mooncake master")
                os.execv("/opt/mooncake-sdk/bin/mooncake_master", plan["master_argv"])
            check_resources(plan)
            if args.server_plan:
                if profile["storage_mode"] == "mooncake":
                    check_endpoint("mooncake", 50051)
                os.execvpe("lmcache", plan["server_argv"], {**os.environ, **plan["server_environment"]})
            check_endpoint("lmcache", 5555)
            if profile["storage_mode"] == "mooncake":
                check_endpoint("mooncake", 50051)
            require_storage_ready(plan)
        else:
            print(json.dumps(native_identity() if args.native else base_identity(), sort_keys=True))
    except (ConfigurationError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: KV storage preflight: {exc}", file=sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
