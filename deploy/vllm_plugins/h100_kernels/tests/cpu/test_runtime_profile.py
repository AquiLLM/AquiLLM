"""Runtime profiles must opt in explicitly and retain the baseline invariants."""
from importlib.metadata import PackageNotFoundError
import sys

import pytest

from aquillm_vllm_h100 import compatibility


@pytest.fixture
def runtime(monkeypatch):
    # Fake the installed distribution metadata and Genesis checkout boundaries.
    # Keep the version gate, profile selection, model and guard checks real.
    versions = {
        "vllm": "0.23.1rc1.dev748+g2dfaae752",
        "torch": "2.11.0+cu130",
        "triton": "3.6.0",
        "flashinfer-python": "0.6.13",
    }
    def package_version(name):
        try:
            return versions[name]
        except KeyError:
            raise PackageNotFoundError(name) from None

    monkeypatch.setattr(compatibility.importlib.metadata, "version", package_version)
    monkeypatch.setattr(compatibility.subprocess, "check_output", lambda *args, **kwargs:
                        "34e269301cc3df71ae4b0da00a0a159b16b4e5d8\n")
    monkeypatch.setattr(sys, "argv", ["spawn_main"])
    monkeypatch.setenv("AQUILLM_H100_MODEL_ID", "hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16")
    monkeypatch.setenv("GENESIS_ENABLE_PN521_TQ_RAW_TAIL_VERIFY", "1")
    monkeypatch.setenv("GENESIS_ENABLE_PN521_SPLIT_K", "1")
    monkeypatch.setenv("GENESIS_ENABLE_PN401_TQ_PREFILL_CONTINUATION_GUARD", "1")
    monkeypatch.setenv("GENESIS_P67_BLOCK_KV", "32")
    monkeypatch.delenv("AQUILLM_H100_RUNTIME_PROFILE", raising=False)
    return versions


@pytest.fixture
def candidate(runtime):
    runtime.update({
        "flashinfer-python": "0.6.18",
        "flashinfer-cubin": "0.6.18",
        "flashinfer-jit-cache": "0.6.18+cu130",
        "nvidia-cutlass-dsl": "4.6.2",
        "nvidia-cutlass-dsl-libs-cu13": "4.6.2",
        "apache-tvm-ffi": "0.1.10",
    })
    return runtime


@pytest.mark.parametrize("profile", [None, "baseline"])
def test_baseline_runtime_still_accepts_original_packages(monkeypatch, runtime, profile):
    if profile is not None:
        monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", profile)
    assert compatibility.verify_runtime() is None


@pytest.mark.parametrize("profile", [None, "baseline"])
def test_candidate_packages_need_explicit_opt_in(monkeypatch, candidate, profile):
    if profile is not None:
        monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", profile)
    with pytest.raises(ValueError, match="unsupported flashinfer-python"):
        compatibility.verify_runtime()


def test_exact_candidate_is_accepted_from_process_environment(monkeypatch, candidate):
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", "flashinfer-0.6.18")
    assert compatibility.verify_runtime() is None


@pytest.mark.parametrize("name,wrong", [
    ("flashinfer-python", "0.6.13"),
    ("flashinfer-cubin", "0.6.13"),
    ("flashinfer-jit-cache", "0.6.18+cu129"),
    ("nvidia-cutlass-dsl", "4.5.2"),
    ("nvidia-cutlass-dsl-libs-cu13", "4.5.2"),
    ("apache-tvm-ffi", "0.1.9"),
])
def test_candidate_rejects_mixed_companion_versions(monkeypatch, candidate, name, wrong):
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", "flashinfer-0.6.18")
    candidate[name] = wrong
    with pytest.raises(ValueError, match=f"unsupported {name}"):
        compatibility.verify_runtime()


@pytest.mark.parametrize("name", ["flashinfer-cubin", "flashinfer-jit-cache",
                                 "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-cu13", "apache-tvm-ffi"])
def test_candidate_requires_installed_companion_wheels(monkeypatch, candidate, name):
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", "flashinfer-0.6.18")
    del candidate[name]
    with pytest.raises(PackageNotFoundError, match=name):
        compatibility.verify_runtime()


@pytest.mark.parametrize("profile", ["typo", "", "flashinfer-0.6.19"])
def test_unknown_runtime_profile_rejects_valid_baseline(monkeypatch, runtime, profile):
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", profile)
    with pytest.raises(ValueError, match="runtime profile"):
        compatibility.verify_runtime()


@pytest.mark.parametrize("name,wrong", [
    ("vllm", "0.23.1rc1.dev749+g2dfaae752"),
    ("torch", "2.11.0+cu129"),
    ("triton", "3.7.0"),
])
def test_candidate_preserves_core_package_pins(monkeypatch, candidate, name, wrong):
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", "flashinfer-0.6.18")
    candidate[name] = wrong
    with pytest.raises(ValueError, match=f"unsupported {name}"):
        compatibility.verify_runtime()


def test_candidate_preserves_genesis_commit(monkeypatch, candidate):
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", "flashinfer-0.6.18")
    monkeypatch.setattr(compatibility.subprocess, "check_output", lambda *args, **kwargs: "wrong\n")
    with pytest.raises(ValueError, match="Genesis commit"):
        compatibility.verify_runtime()


def test_candidate_preserves_authoritative_cli_model_check(monkeypatch, candidate):
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", "flashinfer-0.6.18")
    monkeypatch.setattr(sys, "argv", ["api_server", "--model=other"])
    with pytest.raises(ValueError, match="model outside"):
        compatibility.verify_runtime()


@pytest.mark.parametrize("guard", ["GENESIS_ENABLE_PN521_TQ_RAW_TAIL_VERIFY",
                                  "GENESIS_ENABLE_PN521_SPLIT_K",
                                  "GENESIS_ENABLE_PN401_TQ_PREFILL_CONTINUATION_GUARD"])
def test_candidate_preserves_required_baseline_guards(monkeypatch, candidate, guard):
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", "flashinfer-0.6.18")
    monkeypatch.delenv(guard)
    with pytest.raises(ValueError, match=f"required baseline guard {guard}"):
        compatibility.verify_runtime()


def test_candidate_preserves_validated_tile(monkeypatch, candidate):
    monkeypatch.setenv("AQUILLM_H100_RUNTIME_PROFILE", "flashinfer-0.6.18")
    monkeypatch.setenv("GENESIS_P67_BLOCK_KV", "16")
    with pytest.raises(ValueError, match="32-token tile"):
        compatibility.verify_runtime()
