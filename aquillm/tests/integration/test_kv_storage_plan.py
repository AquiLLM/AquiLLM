"""Storage plans remain experimental and account for all physical allocations."""
import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "deploy/scripts"))


@pytest.fixture
def storage():
    class LazyStorage:
        def __getattr__(self, name):
            assert (ROOT / "deploy/scripts/kv_storage_plan.py").exists(), "storage planner missing"
            return getattr(importlib.import_module("kv_storage_plan"), name)
    return LazyStorage()


@pytest.fixture
def layout():
    # Synthetic geometry, never the measured model's registration evidence.
    return {"schema_version": 1, "identity": {
        "model": "hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16",
        "revision": "2d783431e303148fc6e16622fac5edac83a6b5c4",
        "quantization": "awq", "kv_dtype": "turboquant_k8v4",
        "tensor_parallel_size": 1, "pipeline_parallel_size": 1, "mtp_depth": 4,
    }, "groups": [
        {"kind": "attention", "layers": 16, "dtype": "uint8",
         "shape": [8, 16, 4, 388], "stride": [24832, 1552, 388, 1],
         "logical_block_tokens": 16, "page_bytes": 24832},
        {"kind": "recurrent", "layers": 48, "dtype": "bfloat16",
         "logical_block_tokens": 32, "page_bytes": 1024,
         "states": [{"shape": [8, 16], "offset_bytes": 0},
                    {"shape": [8, 32], "offset_bytes": 256}]},
    ]}


@pytest.fixture
def limits():
    return {"server_processes": 1, "engine_workers": 1, "adapter_threads": 4,
            "l1_bytes": 1 << 30, "segment_bytes": 1 << 30,
            "client_buffer_bytes": 1 << 27, "ssd_staging_bytes": 1 << 26,
            "worker_staging_bytes": 1 << 27, "process_overhead_bytes": 1 << 28,
            "master_metadata_bytes": 1 << 29,
            "host_available_bytes": 16 << 30, "container_limit_bytes": 8 << 30,
            "disk_available_bytes": 20 << 30, "reserve_bytes": 1 << 30,
            "ssd_path": "/var/lib/mooncake/test/server-0"}


def env(mode="mooncake"):
    return {"KV_CACHE_STORAGE_MODE": mode, "KV_CACHE_RAM_GIB": "8",
            "KV_CACHE_SSD_GIB": "8" if mode == "mooncake" else "0"}


def test_generated_mooncake_contract(storage, layout, limits):
    result = json.loads(json.dumps(storage.plan_storage(env(), layout, limits)))
    assert result["status"] == "experimental-not-launchable"
    assert result["active_authority"] == "gpu-resident"
    assert result["chunk_tokens"] == 32
    connector = result["connector"]
    assert connector["kv_connector"] == "LMCacheMPConnector"
    assert connector["kv_connector_module_path"] == "lmcache.integration.vllm.lmcache_mp_connector"
    assert result["vllm_required_args"] == ["--enable-prefix-caching", "--mamba-cache-mode", "align"]
    command = result["server_argv"]
    assert "--separate-object-groups" in command
    adapter = json.loads(command[command.index("--l2-adapter") + 1])
    assert adapter["type"] == "mooncake_store"
    assert adapter["enable_ssd_offload"] == "true"
    assert adapter["global_segment_size"] == "1073741824"
    assert adapter["num_workers"] == 4
    assert adapter["ssd_offload_path"] == "/var/lib/mooncake/test/server-0"
    assert result["memory"]["total_bytes"] == 3288334336
    assert result["memory"]["owner"] == "LMCache MP server (embedded RealClient); engine staging separately"
    assert result["server_environment"]["MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES"] == "8589934592"
    assert result["manifest"]["gpu_roundtrip"] == "not-validated"


def test_metadata_master_is_part_of_same_host_budget(storage, layout, limits):
    limits["master_metadata_bytes"] = 536870912
    result = storage.plan_storage(env(), layout, limits)
    assert result["memory"]["total_bytes"] == 3288334336
    limits["master_metadata_bytes"] = 8 << 30
    with pytest.raises(ValueError, match="RAM"):
        storage.plan_storage(env(), layout, limits)


def test_local_has_no_mooncake_or_ssd_owner(storage, layout, limits):
    result = storage.plan_storage(env("local"), layout, limits)
    assert "--l2-adapter" not in result["server_argv"]
    assert result["master_argv"] is None
    assert result["server_environment"] == {}
    assert result["memory"]["total_bytes"] == 1476395008


def test_off_does_not_require_measurements_or_dependencies(storage):
    assert storage.plan_storage({}, None, None) is None


@pytest.mark.parametrize("key,value", [("model", "other"), ("revision", "main"),
                                      ("quantization", "fp8"), ("tensor_parallel_size", 2)])
def test_layout_bound_to_identity(storage, layout, limits, key, value):
    layout["identity"][key] = value
    with pytest.raises(ValueError, match="identity"):
        storage.plan_storage(env(), layout, limits)


@pytest.mark.parametrize("mutate,match", [
    (lambda x: x["groups"][0].update(logical_block_tokens=2128), "logical"),
    (lambda x: x["groups"][0].update(dtype="float16"), "uint8"),
    (lambda x: x["groups"][0].update(stride=[24832, 1552, 400, 1]), "stride"),
    (lambda x: x["groups"][1]["states"][1].update(offset_bytes=128), "overlap"),
    (lambda x: x.update(groups=x["groups"][:1]), "recurrent"),
])
def test_rejects_unsupported_geometry(storage, layout, limits, mutate, match):
    mutate(layout)
    with pytest.raises(ValueError, match=match):
        storage.plan_storage(env(), layout, limits)


@pytest.mark.parametrize("change,match", [
    ({"container_limit_bytes": 1 << 30}, "container"),
    ({"host_available_bytes": 1 << 30}, "host"),
    ({"disk_available_bytes": 1 << 30}, "disk"),
    ({"engine_workers": 100}, "RAM"),
    ({"server_processes": 2}, "50052"),
    ({"ssd_path": "../relative"}, "absolute"),
    ({"l1_bytes": 1073741825}, "4096"),
])
def test_budget_and_namespace_errors(storage, layout, limits, change, match):
    limits.update(change)
    with pytest.raises(ValueError, match=match):
        storage.plan_storage(env(), layout, limits)


def test_auto_without_matching_measurement_is_rejected(storage, layout, limits):
    request = env()
    request["KV_CACHE_RAM_GIB"] = "auto"
    with pytest.raises(ValueError, match="explicit"):
        storage.plan_storage(request, layout, limits)


def test_no_readiness_boolean_bypass(storage, layout, limits):
    layout["supported"] = True
    layout["gpu_validated"] = True
    plan = storage.plan_storage(env(), layout, limits)
    assert plan["status"] == "experimental-not-launchable"
    with pytest.raises(ValueError, match="registration"):
        storage.require_storage_ready(plan)
