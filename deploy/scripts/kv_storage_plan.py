#!/usr/bin/env python3
"""Emit experimental LMCache MP configuration. This CLI never launches a model.

--layout is schema 1 with pinned identity and effective mixed KVCacheConfig groups.
--limits supplies explicit physical allocations and current resource headroom.
Both are planning inputs, not trusted readiness evidence. See kv_storage_layout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

from kv_cache_config import ConfigurationError, _resolve
from kv_storage_layout import integer, validate_layout

PINS = {
    "base_image": "vllm/vllm-openai@sha256:6a93ae4316826f3dd8a92bee5442cbed50184a9cbd688d310f9e56ecad1eabeb",
    "genesis": "34e269301cc3df71ae4b0da00a0a159b16b4e5d8",
    "lmcache": "05a013b29da78cf2321b9b46ec5039dde2fb0bb0",
    "mooncake": "719735896c86b56fabec6cf3e825fb2ea640597a",
    "python": "3.12.13", "torch": "2.11.0+cu130", "cuda_image": "13.0.2",
    "vllm": "0.23.1rc1.dev748+g2dfaae752", "transformers": "5.12.1",
    "huggingface-hub": "1.21.0", "cxx11_abi": True,
}


def memory_plan(profile, limits):
    if not isinstance(limits, dict):
        raise ConfigurationError("explicit limits record required; automatic memory sizing is unsupported")
    mode = profile["storage_mode"]
    sizes = {key: integer(limits.get(key), key, zero=key.endswith("bytes")) for key in (
        "server_processes", "engine_workers", "adapter_threads", "l1_bytes", "segment_bytes",
        "client_buffer_bytes", "ssd_staging_bytes", "worker_staging_bytes", "process_overhead_bytes",
        "master_metadata_bytes",
        "host_available_bytes", "container_limit_bytes", "disk_available_bytes", "reserve_bytes",
    )}
    if sizes["server_processes"] != 1:
        raise ConfigurationError("one embedded RealClient per IPC namespace: offload port 50052 is fixed; multiple processes unsupported")
    for key in ("l1_bytes", "worker_staging_bytes", "process_overhead_bytes", "reserve_bytes"):
        integer(sizes[key], key)
    if sizes["l1_bytes"] % 4096:
        raise ConfigurationError("l1_bytes must be 4096-byte aligned for exact bounded GiB serialization")
    ram = profile["tier_budgets"]["ram"]
    ssd = profile["tier_budgets"]["ssd"]
    if ram is None or (mode == "mooncake" and ssd is None):
        raise ConfigurationError("explicit RAM/SSD caps required; auto needs a matching runtime measurement")
    total = sizes["l1_bytes"] + sizes["process_overhead_bytes"] + sizes["engine_workers"] * sizes["worker_staging_bytes"]
    if mode == "mooncake":
        total += sum(integer(sizes[k], k) for k in ("segment_bytes", "client_buffer_bytes", "ssd_staging_bytes"))
        if sizes["master_metadata_bytes"] < 512 * (1 << 20):
            raise ConfigurationError("master_metadata_bytes must cover the Compose master's 512 MiB memory cap")
        total += sizes["master_metadata_bytes"]
        integer(ssd, "SSD budget")
        if ssd + sizes["reserve_bytes"] > sizes["disk_available_bytes"]:
            raise ConfigurationError("SSD cap plus reserve exceeds disk headroom")
        path = PurePosixPath(limits.get("ssd_path", ""))
        if not path.is_absolute() or ".." in path.parts or str(path) == "/":
            raise ConfigurationError("SSD path must be a unique existing absolute server directory")
    elif ssd not in (None, 0):
        raise ConfigurationError("local mode is RAM only; select mooncake for bounded SSD")
    for cap, name in ((ram, "RAM budget"), (sizes["container_limit_bytes"], "container limit"),
                      (sizes["host_available_bytes"] - sizes["reserve_bytes"], "host headroom")):
        if total > cap:
            raise ConfigurationError(f"all replicated/staging allocations exceed {name}")
    return {"total_bytes": total, "allocations": sizes,
            "owner": "LMCache MP server (embedded RealClient); engine staging separately"}


def plan_storage(env, layout, limits, extra_args=()):
    profile = _resolve(env, extra_args, planning=True)
    if profile is None or profile["storage_mode"] == "off":
        return None
    chunk = validate_layout(layout)
    memory = memory_plan(profile, limits)
    for flag in extra_args:
        if flag.split("=", 1)[0].replace("_", "-") in {"--no-enable-prefix-caching", "--disable-prefix-caching", "--mamba-cache-mode"}:
            raise ConfigurationError("remove conflicting legacy prefix/mamba flags; this experimental plan requires prefix caching and align")
    digest = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()
    server = ["lmcache", "server", "--host", "0.0.0.0", "--port", "5555", "--chunk-size", str(chunk),
              "--separate-object-groups", "--l1-size-gb", str(limits["l1_bytes"] / (1 << 30)), "--eviction-policy", "LRU"]
    master, environment = None, {}
    if profile["storage_mode"] == "mooncake":
        adapter = {"type": "mooncake_store", "num_workers": limits["adapter_threads"],
                   "local_hostname": "lmcache", "metadata_server": "P2PHANDSHAKE",
                   "master_server_addr": "mooncake:50051", "protocol": "tcp",
                   "global_segment_size": str(limits["segment_bytes"]),
                   "local_buffer_size": str(limits["client_buffer_bytes"]),
                   "enable_ssd_offload": "true", "ssd_offload_path": limits["ssd_path"],
                   "tenant_id": "aquillm-" + digest}
        server += ["--l2-adapter", json.dumps(adapter, sort_keys=True)]
        master = ["mooncake_master", "--rpc_port=50051", "--enable_offload=true"]
        environment = {"MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR": "bucket_storage_backend",
                       "MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE": str(profile["tier_budgets"]["ssd"]),
                       "MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES": str(profile["tier_budgets"]["ssd"]),
                       "MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY": "lru",
                       "MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES": str(limits["ssd_staging_bytes"])}
    return {"schema_version": 1, "status": "experimental-not-launchable", "profile": profile,
            "active_authority": "gpu-resident", "storage_authority": "reusable-prefix-only",
            "chunk_tokens": chunk, "layout_sha256": digest, "layout": layout, "limits": limits,
            "connector": {"kv_connector": "LMCacheMPConnector",
                          "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
                          "kv_role": "kv_both", "kv_connector_extra_config": {
                              "lmcache.mp.host": "tcp://lmcache", "lmcache.mp.port": 5555,
                              "lmcache.mp.mq_timeout": 30}},
            "vllm_required_args": ["--enable-prefix-caching", "--mamba-cache-mode", "align"],
            "server_argv": server, "master_argv": master, "server_environment": environment,
            "memory": memory, "manifest": {"pins": PINS, "gpu_roundtrip": "not-validated",
                                            "active_pager": "not-implemented"}}


def require_storage_ready(plan):
    raise ConfigurationError("storage readiness blocked: live mixed-group allocation registration and GPU/SSD roundtrip validation are not implemented; no boolean bypass is accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--limits", type=Path)
    args, extra = parser.parse_known_args()
    try:
        layout = json.loads(args.layout.read_text(encoding="utf-8-sig")) if args.layout else None
        limits = json.loads(args.limits.read_text(encoding="utf-8-sig")) if args.limits else None
        print(json.dumps(plan_storage(os.environ, layout, limits, extra), sort_keys=True, indent=2))
    except (ConfigurationError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: KV storage planning: {exc}", file=sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
