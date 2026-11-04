"""Capacity math and opt-in validation without a serving runtime."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "deploy/scripts/kv_cache_config.py"


@pytest.fixture
def config():
    class LazyConfig:
        module = None

        def __getattr__(self, name):
            assert SCRIPT.is_file(), "capacity configuration helper is not implemented"
            if self.module is None:
                spec = importlib.util.spec_from_file_location("kv_cache_config", SCRIPT)
                self.module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self.module)
            return getattr(self.module, name)
    return LazyConfig()


def test_eight_context_profile(config):
    profile = config.resolve_profile({
        "KV_CACHE_TARGET_ACTIVE_SEQUENCES": "8",
        "VLLM_MAX_MODEL_LEN": "262144",
    }, ["--kv-cache-dtype", "turboquant_k8v4"])
    assert profile["active_sequences"] == 8
    assert profile["retained_contexts"] == 8
    assert profile["full_attention_payload_bytes"] == 52_076_478_464
    assert "8" in config.profile_arguments(profile)


def test_no_profile_preserves_defaults(config):
    assert config.resolve_profile({}, []) is None
    assert config.resolve_profile({"VLLM_MAX_MODEL_LEN": "999999"}, [
        "--kv-cache-dtype", "fp8", "--cpu-offload-gb", "4",
    ]) is None
    assert config.resolve_profile({"KV_CACHE_RAM_GIB": ""}, []) is None


@pytest.mark.parametrize(("n", "t", "expected"), [
    ("4", "262144", 26_038_239_232),
    ("4", "131072", 13_019_119_616),
    ("3", "262144", 19_528_679_424),
])
def test_capacity_scales_with_active_count_and_context(config, n, t, expected):
    profile = config.resolve_profile({
        "KV_CACHE_TARGET_ACTIVE_SEQUENCES": n, "VLLM_MAX_MODEL_LEN": t,
    }, [])
    assert profile["full_attention_payload_bytes"] == expected
    assert profile["validation_status"] == "unvalidated"
    assert profile["kv_dtype"] == "turboquant_k8v4"


def test_retained_capacity_does_not_change_scheduler(config):
    profile = config.resolve_profile({
        "KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4", "KV_CACHE_RETAINED_CONTEXTS": "12",
    }, [])
    assert profile["retained_full_attention_payload_bytes"] == 78_114_717_696
    assert profile["retained_contexts"] == 12
    assert config.profile_arguments(profile) == [
        "--max-num-seqs", "4", "--max-model-len", "262144",
        "--kv-cache-dtype", "turboquant_k8v4",
    ]


@pytest.mark.parametrize("setting", [
    "KV_CACHE_TARGET_ACTIVE_SEQUENCES", "KV_CACHE_RETAINED_CONTEXTS", "VLLM_MAX_MODEL_LEN",
])
@pytest.mark.parametrize("bad", ["0", "-1", "1.5", "nan", "inf", "1e6", "9" * 200])
def test_invalid_and_overflow_counts_fail(config, setting, bad):
    with pytest.raises(config.ConfigurationError, match=setting):
        config.resolve_profile({"KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4", setting: bad}, [])


def test_context_limit_and_retained_lower_bound(config):
    for env in ({"VLLM_MAX_MODEL_LEN": "262145"}, {"KV_CACHE_RETAINED_CONTEXTS": "3"}):
        with pytest.raises(config.ConfigurationError):
            config.resolve_profile({"KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4", **env}, [])


@pytest.mark.parametrize("setting", ["KV_CACHE_VRAM_GIB", "KV_CACHE_RAM_GIB", "KV_CACHE_SSD_GIB"])
@pytest.mark.parametrize("bad", ["-1", "nan", "NaN", "inf", "Infinity", "1e9999", "abc"])
def test_invalid_tier_caps_fail(config, setting, bad):
    with pytest.raises(config.ConfigurationError, match=setting):
        config.resolve_profile({setting: bad}, [])


def test_exact_byte_budgets_and_unresolved_auto(config):
    profile = config.resolve_profile({
        "KV_CACHE_VRAM_GIB": "0.5", "KV_CACHE_RAM_GIB": "auto", "KV_CACHE_SSD_GIB": "0",
    }, [])
    assert profile["tier_budgets"] == {"vram": 536_870_912, "ram": None, "ssd": 0}
    assert profile["tier_budget_requests"] == {"vram": "0.5", "ram": "auto", "ssd": "0"}
    assert profile["validation_status"] == "unvalidated"
    assert config.profile_arguments(profile)[-2:] == ["--kv-cache-memory-bytes", "536870912"]


def test_sub_byte_budget_is_rejected(config):
    with pytest.raises(config.ConfigurationError, match="whole bytes"):
        config.resolve_profile({"KV_CACHE_RAM_GIB": "0.0000000001"}, [])


def test_zero_vram_cap_does_not_silently_select_automatic_allocation(config):
    with pytest.raises(config.ConfigurationError, match="KV_CACHE_VRAM_GIB"):
        config.resolve_profile({"KV_CACHE_VRAM_GIB": "0"}, [])


def test_full_payload_is_bounded_even_when_count_itself_is_valid(config):
    with pytest.raises(config.ConfigurationError, match="overflows"):
        config.resolve_profile({"KV_CACHE_TARGET_ACTIVE_SEQUENCES": "2147483647"}, [])


@pytest.mark.parametrize("equals", [False, True])
def test_matching_legacy_flags_are_accepted(config, equals):
    pairs = [("--max-num-seqs", "4"), ("--max-model-len", "131072"),
             ("--kv-cache-dtype", "turboquant_k8v4")]
    args = [f"{k}={v}" for k, v in pairs] if equals else [x for pair in pairs for x in pair]
    profile = config.resolve_profile({"KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4"}, args)
    assert profile["context_tokens"] == 131072
    assert profile["full_attention_payload_bytes"] == 13_019_119_616


@pytest.mark.parametrize(("flag", "values"), [
    ("--max-num-seqs", ["8"]), ("--max-model-len", ["131072"]),
    ("--kv-cache-dtype", ["turboquant_k4v4"]),
    ("--max-num-seqs", ["4", "4"]), ("--max-model-len", ["262144", "262144"]),
    ("--kv-cache-dtype", ["turboquant_k8v4", "turboquant_k8v4"]),
])
@pytest.mark.parametrize("equals", [False, True])
def test_conflicting_or_duplicate_flags_fail(config, flag, values, equals):
    args = [f"{flag}={v}" for v in values] if equals else [x for v in values for x in (flag, v)]
    with pytest.raises(config.ConfigurationError, match=flag):
        config.resolve_profile({"KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4", "VLLM_MAX_MODEL_LEN": "262144"}, args)


@pytest.mark.parametrize("flag", ["--max-num-seqs", "--max-model-len", "--kv-cache-dtype"])
def test_missing_flag_values_fail(config, flag):
    with pytest.raises(config.ConfigurationError, match=flag):
        config.resolve_profile({"KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4"}, [flag])


@pytest.mark.parametrize("args", [
    ["--cpu-offload-gb", "1"], ["--cpu-offload-gb=-1"], ["--cpu-offload-gb=nan"],
    ["--offload-group-size", "1"], ["--offload-num-in-group=1"],
    ["--offload-prefetch-step", "2"], ["--offload-params", "layer.weight"],
    ["--kv-offloading-size", "8"], ["--disable-hybrid-kv-cache-manager"],
    ["--kv-transfer-config", '{"kv_connector":"LMCacheConnectorV1"}'],
    ["--hf-overrides", '{"num_hidden_layers":40}'],
])
def test_incompatible_offload_and_hybrid_flags_fail(config, args):
    with pytest.raises(config.ConfigurationError):
        config.resolve_profile({"KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4"}, args)


def test_zero_weight_offload_is_allowed(config):
    assert config.resolve_profile({"KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4"}, ["--cpu-offload-gb=0"])


@pytest.mark.parametrize("env", [
    {"KV_CACHE_EXECUTION_MODE": "paged"}, {"KV_CACHE_STORAGE_MODE": "local"},
    {"KV_CACHE_STORAGE_MODE": "mooncake"}, {"KV_CACHE_STORAGE_MODE": "typo"},
])
def test_unvalidated_modes_fail_closed(config, env):
    with pytest.raises(config.ConfigurationError):
        config.resolve_profile(env, [])


def test_mode_only_request_enables_validation(config):
    profile = config.resolve_profile({"KV_CACHE_STORAGE_MODE": "off"}, [])
    assert profile["active_sequences"] == 4
    assert profile["execution_mode"] == "resident"


def test_legacy_vram_bytes_larger_than_32_bits_are_preserved(config):
    profile = config.resolve_profile({"KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4"},
                                     ["--kv-cache-memory-bytes=52076478464"])
    assert profile["tier_budgets"]["vram"] == 52_076_478_464
    assert config.profile_arguments(profile)[-2:] == ["--kv-cache-memory-bytes", "52076478464"]


@pytest.mark.parametrize("env,args", [
    ({"VLLM_MODEL": "google/gemma-3-12b-it"}, []),
    ({}, ["--model=example/other"]),
])
def test_capacity_rejects_other_model_geometry(config, env, args):
    with pytest.raises(config.ConfigurationError, match="layout"):
        config.resolve_profile({"KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4", **env}, args)


def test_planning_cli_can_report_unsupported_modes_without_launch_arguments(config):
    result = subprocess.run([sys.executable, str(SCRIPT), "--plan"], env={
        "KV_CACHE_TARGET_ACTIVE_SEQUENCES": "8", "KV_CACHE_STORAGE_MODE": "mooncake",
        "KV_CACHE_EXECUTION_MODE": "paged",
    }, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    profile = json.loads(result.stdout)
    assert profile["full_attention_payload_bytes"] == 52_076_478_464
    assert profile["validation_status"] == "unvalidated"
    with pytest.raises(config.ConfigurationError):
        config.profile_arguments(profile)


@pytest.mark.parametrize("args", [
    ["--cpu_offload_gb=1"], ["--offload_group_size", "1"],
    ["--offload_num_in_group=1"], ["--offload_prefetch_step", "1"],
    ["--offload_params", "layer.weight"], ["--kv_offloading_size=8"],
    ["--disable_hybrid_kv_cache_manager"], ["--kv_transfer_config", "{}"],
    ["--hf_overrides.num_hidden_layers=40"], ["--hf-overrides.num_hidden_layers", "40"],
    ["--kv-transfer-config.kv_connector=LMCacheConnectorV1"],
    ["--kv_transfer_config.kv_connector", "LMCacheConnectorV1"],
    ["--max_num_seqs=8"], ["--max_model_len=131072"],
    ["--kv_cache_dtype=turboquant_k4v4"],
    ["--kv_cache_memory_bytes=12"], ["--max-num-seqs.foo=4"],
])
def test_downstream_option_spellings_cannot_bypass_guards(config, args):
    with pytest.raises(config.ConfigurationError):
        config.resolve_profile({"KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4",
                                "VLLM_MAX_MODEL_LEN": "262144", "KV_CACHE_VRAM_GIB": "1"}, args)


@pytest.mark.parametrize("equals", [False, True])
def test_matching_underscore_profile_options_consolidate_in_launch_cli(config, equals):
    pairs = [("--max_num_seqs", "4"), ("--max_model_len", "131072"),
             ("--kv_cache_dtype", "turboquant_k8v4"), ("--kv_cache_memory_bytes", "536870912")]
    args = [f"{k}={v}" for k, v in pairs] if equals else [x for pair in pairs for x in pair]
    result = subprocess.run([sys.executable, str(SCRIPT), "--launch-argv", "--", *args], env={
        "KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4",
    }, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split(b"\0")[:-1] == [
        b"--max-num-seqs", b"4", b"--max-model-len", b"131072",
        b"--kv-cache-dtype", b"turboquant_k8v4", b"--kv-cache-memory-bytes", b"536870912",
    ]


@pytest.mark.parametrize("args", [
    ["--max-num-seqs=4", "--max_num_seqs=4"],
    ["--max-model-len=262144", "--max_model_len=262144"],
    ["--kv-cache-dtype=turboquant_k8v4", "--kv_cache_dtype=turboquant_k8v4"],
    ["--kv-cache-memory-bytes=12", "--kv_cache_memory_bytes=12"],
])
def test_mixed_spelling_duplicates_reject(config, args):
    with pytest.raises(config.ConfigurationError, match="duplicate"):
        config.resolve_profile({"KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4"}, args)


@pytest.mark.parametrize("args", [
    ["--config", "/tmp/weight-offload.yaml"], ["--config=/tmp/weight-offload.yaml"],
    ["--conf", "/tmp/weight-offload.yaml"], ["--con=/tmp/weight-offload.yaml"],
    ["--c", "/tmp/weight-offload.yaml"], ["-c", "/tmp/weight-offload.yaml"],
    ["-c/tmp/weight-offload.yaml"], ["--config.cpu_offload_gb=1"],
    ["--cpu_offload_g=1"], ["--offload-group=1"], ["--offload_p=1"],
    ["--kv-offl=8"], ["--disable_hybrid_kv=1"],
    ["--hf-over.num_hidden_layers=40"], ["--kv-transfer.kv_connector=LMCacheConnectorV1"],
    ["--max-num-se=8"], ["--max_model_l=131072"], ["--kv-cache-d=fp8"],
    ["--mo=example/other"],
])
def test_external_configs_and_guarded_abbreviations_reject(config, args):
    with pytest.raises(config.ConfigurationError):
        config.resolve_profile({"KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4"}, args)


def test_no_profile_leaves_extended_parser_syntax_unchanged(config):
    assert config.resolve_profile({}, ["--config", "/tmp/config.yaml",
                                       "--cpu_offload_gb=1", "--hf-overrides.foo=40"]) is None


def test_unrelated_dotted_json_options_remain_usable(config):
    assert config.resolve_profile({"KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4"}, [
        "--speculative-config.method=mtp", "--compilation_config.cudagraph_mode=PIECEWISE",
    ])["active_sequences"] == 4


@pytest.mark.parametrize("args", [["-cc", "{}"], ["-cc={}"],
                                  ["-cc.mode", "0"], ["-cc.mode=0"]])
def test_compilation_alias_preserves_argv_in_requested_profile(config, args):
    assert config.resolve_profile({"KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4"}, args)
    result = subprocess.run([sys.executable, str(SCRIPT), "--launch-argv", "--", *args],
                            env={"KV_CACHE_TARGET_ACTIVE_SEQUENCES": "4"}, capture_output=True)
    assert result.returncode == 0, result.stderr
    tokens = [token.decode() for token in result.stdout.split(b"\0")[:-1]]
    assert tokens == args + ["--max-num-seqs", "4", "--max-model-len", "262144",
                             "--kv-cache-dtype", "turboquant_k8v4"]
