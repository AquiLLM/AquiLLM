from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_exposes_one_vllm_general_plugin_entry_point() -> None:
    with (PLUGIN_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    entry_points = pyproject["project"]["entry-points"]["vllm.general_plugins"]

    assert entry_points == {
        "aquillm_nemotron_asr": "aquillm_vllm_nemotron_asr:register"
    }


def test_importing_plugin_does_not_import_model(fresh_plugin_module: None) -> None:
    importlib.import_module("aquillm_vllm_nemotron_asr")

    assert "aquillm_vllm_nemotron_asr.model" not in sys.modules


def test_register_adds_the_lazy_model_target(
    fake_model_registry, fresh_plugin_module: None
) -> None:
    plugin = importlib.import_module("aquillm_vllm_nemotron_asr")

    plugin.register()

    assert fake_model_registry.registry.register_calls == [
        (
            "Nemotron3_5AsrForRNNT",
            "aquillm_vllm_nemotron_asr.model:Nemotron3_5AsrForRNNT",
        )
    ]
    assert "aquillm_vllm_nemotron_asr.model" not in sys.modules


def test_register_is_a_no_op_when_the_same_model_is_already_registered(
    fake_model_registry, fresh_plugin_module: None
) -> None:
    plugin = importlib.import_module("aquillm_vllm_nemotron_asr")

    plugin.register()
    plugin.register()

    assert fake_model_registry.registry.register_calls == [
        (
            "Nemotron3_5AsrForRNNT",
            "aquillm_vllm_nemotron_asr.model:Nemotron3_5AsrForRNNT",
        )
    ]


def test_register_rejects_a_conflicting_existing_architecture(
    fake_model_registry, fresh_plugin_module: None
) -> None:
    plugin = importlib.import_module("aquillm_vllm_nemotron_asr")
    fake_model_registry.registry.models["Nemotron3_5AsrForRNNT"] = (
        fake_model_registry.model_info("another_module", "OtherModel")
    )

    with pytest.raises(RuntimeError, match="Nemotron3_5AsrForRNNT"):
        plugin.register()

    assert fake_model_registry.registry.models["Nemotron3_5AsrForRNNT"] == (
        fake_model_registry.model_info("another_module", "OtherModel")
    )
    assert fake_model_registry.registry.register_calls == []


def test_register_rejects_an_eager_conflicting_architecture(
    fake_model_registry, fresh_plugin_module: None
) -> None:
    plugin = importlib.import_module("aquillm_vllm_nemotron_asr")
    existing_model = fake_model_registry.registered_model(object)
    fake_model_registry.registry.models["Nemotron3_5AsrForRNNT"] = existing_model

    with pytest.raises(
        RuntimeError,
        match=(
            r"builtins\.object.*aquillm_vllm_nemotron_asr\.model:"
            r"Nemotron3_5AsrForRNNT"
        ),
    ):
        plugin.register()

    assert (
        fake_model_registry.registry.models["Nemotron3_5AsrForRNNT"] is existing_model
    )
    assert fake_model_registry.registry.register_calls == []


def test_register_rejects_an_unsupported_vllm_before_registry_mutation(
    fake_model_registry, fresh_plugin_module: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = importlib.import_module("aquillm_vllm_nemotron_asr")
    monkeypatch.setattr(sys.modules["vllm"], "__version__", "0.21.1")

    with pytest.raises(RuntimeError, match="0.21.0"):
        plugin.register()

    assert fake_model_registry.registry.register_calls == []


def test_register_rejects_v2_before_registry_mutation(
    fake_model_registry, fresh_plugin_module: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = importlib.import_module("aquillm_vllm_nemotron_asr")
    monkeypatch.setattr(sys.modules["vllm.envs"], "VLLM_USE_V2_MODEL_RUNNER", True)

    with pytest.raises(RuntimeError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        plugin.register()

    assert fake_model_registry.registry.register_calls == []
