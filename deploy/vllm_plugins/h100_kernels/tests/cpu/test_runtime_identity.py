import pytest

from aquillm_vllm_h100.compatibility import MODEL, model_identity


def test_api_server_identity_survives_wrapper_env_cleanup():
    assert model_identity(["api_server", "--model", MODEL], {}) == MODEL


def test_worker_inherits_verified_identity():
    assert model_identity(["spawn_main"], {"AQUILLM_H100_MODEL_ID": MODEL}) == MODEL


def test_effective_cli_override_wins_over_environment():
    assert model_identity(["api_server", "--model", MODEL, "--model=other"],
                          {"VLLM_MODEL": MODEL, "AQUILLM_H100_MODEL_ID": MODEL}) == "other"


def test_wrapper_help_probe_can_use_original_env():
    assert model_identity(["api_server", "--help"], {"VLLM_MODEL": MODEL}) == MODEL
