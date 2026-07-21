from __future__ import annotations

import sys
from collections import namedtuple
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PLUGIN_ROOT / "src"


@pytest.fixture(autouse=True)
def plugin_source_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(SRC_ROOT))


@pytest.fixture
def fake_model_registry(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Install the public registry surface used by vLLM 0.21 plugins."""
    model_info = namedtuple("ModelInfo", "module_name class_name")
    registered_model = namedtuple("RegisteredModel", "model_cls")

    class FakeModelRegistry:
        models: dict[str, object] = {}
        register_calls: list[tuple[str, str]] = []

        @classmethod
        def register_model(cls, architecture: str, target: str) -> None:
            cls.register_calls.append((architecture, target))
            module_name, class_name = target.split(":", maxsplit=1)
            cls.models[architecture] = model_info(module_name, class_name)

    registry_module = ModuleType("vllm.model_executor.models.registry")
    registry_module.ModelRegistry = FakeModelRegistry
    models_module = ModuleType("vllm.model_executor.models")
    models_module.registry = registry_module
    model_executor_module = ModuleType("vllm.model_executor")
    model_executor_module.models = models_module
    vllm_module = ModuleType("vllm")
    vllm_module.model_executor = model_executor_module
    vllm_module.__version__ = "0.21.0"
    envs_module = ModuleType("vllm.envs")
    envs_module.VLLM_USE_V2_MODEL_RUNNER = False
    vllm_module.envs = envs_module

    class FakeOpenAISpeechToText:
        async def _create_speech_to_text(self, *args, **kwargs):
            raise AssertionError("fake compatibility wrapper was not replaced")

        async def _preprocess_speech_to_text(self, *args, **kwargs):
            raise AssertionError("fake compatibility wrapper was not replaced")

    entrypoints_module = ModuleType("vllm.entrypoints")
    openai_module = ModuleType("vllm.entrypoints.openai")
    speech_to_text_package = ModuleType("vllm.entrypoints.openai.speech_to_text")
    speech_to_text_module = ModuleType(
        "vllm.entrypoints.openai.speech_to_text.speech_to_text"
    )
    speech_to_text_module.OpenAISpeechToText = FakeOpenAISpeechToText
    entrypoints_module.openai = openai_module
    openai_module.speech_to_text = speech_to_text_package
    speech_to_text_package.speech_to_text = speech_to_text_module

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", models_module)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.registry", registry_module
    )
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", entrypoints_module)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai", openai_module)
    monkeypatch.setitem(
        sys.modules, "vllm.entrypoints.openai.speech_to_text", speech_to_text_package
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.openai.speech_to_text.speech_to_text",
        speech_to_text_module,
    )

    return SimpleNamespace(
        registry=FakeModelRegistry,
        model_info=model_info,
        registered_model=registered_model,
    )


@pytest.fixture
def fresh_plugin_module() -> None:
    for module_name in tuple(sys.modules):
        if module_name == "aquillm_vllm_nemotron_asr" or module_name.startswith(
            "aquillm_vllm_nemotron_asr."
        ):
            del sys.modules[module_name]
