from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

DEFAULT_MODEL = "fastino/gliner2-base-v1"
DEFAULT_REVISION = "8437ba583a733d87f56ae902f3b197934eedd58e"


def test_backend_protocol_exposes_provider_neutral_batch_contract() -> None:
    from lib.knowledge_graph.extractors.base import ExtractionBackend

    signature = inspect.signature(ExtractionBackend.extract_batch)

    assert tuple(signature.parameters) == ("self", "texts", "ontology")
    assert signature.parameters["texts"].annotation == "tuple[str, ...]"
    assert signature.parameters["ontology"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["ontology"].annotation == "OntologyDefinition"
    assert signature.return_annotation == "tuple[ExtractionBatchResult, ...]"


def test_extractor_package_does_not_reexport_a_provider_class() -> None:
    extractors = importlib.import_module("lib.knowledge_graph.extractors")

    assert extractors.__all__ == ["ExtractionBackend", "get_extraction_backend"]
    assert not hasattr(extractors, "GLiNER2LocalBackend")


def test_factory_imports_provider_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = importlib.import_module("lib.knowledge_graph.extractors.factory")
    config = importlib.import_module("lib.knowledge_graph.config")
    imports: list[str] = []
    received_settings = []

    class FakeBackend:
        def __init__(self, *, settings: object) -> None:
            received_settings.append(settings)

        def extract_batch(self, texts, *, ontology):  # pragma: no cover - structural
            return ()

    fake_module = SimpleNamespace(GLiNER2LocalBackend=FakeBackend)

    def fake_import(name: str):
        imports.append(name)
        return fake_module

    monkeypatch.setattr(factory.importlib, "import_module", fake_import)
    settings = config.load_extraction_settings({})

    assert imports == []
    backend = factory.get_extraction_backend(settings=settings)

    assert isinstance(backend, FakeBackend)
    assert imports == ["lib.knowledge_graph.extractors.gliner2_local"]
    assert received_settings == [settings]


def test_factory_rejects_unknown_provider_without_importing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = importlib.import_module("lib.knowledge_graph.extractors.factory")
    config = importlib.import_module("lib.knowledge_graph.config")
    settings = config.load_extraction_settings(
        {"KG_EXTRACTOR_PROVIDER": "remote_magic"}
    )

    def unexpected_import(name: str):  # pragma: no cover - must not be reached
        pytest.fail(f"unexpected provider import: {name}")

    monkeypatch.setattr(factory.importlib, "import_module", unexpected_import)

    with pytest.raises(
        factory.UnsupportedExtractionProviderError, match="remote_magic"
    ):
        factory.get_extraction_backend(settings=settings)


def test_config_defaults_are_disabled_and_checkpoint_pinned() -> None:
    from lib.knowledge_graph.config import load_extraction_settings

    settings = load_extraction_settings({})

    assert settings.build_enabled is False
    assert settings.provider == "gliner2_local"
    assert settings.model_id == DEFAULT_MODEL
    assert settings.model_revision == DEFAULT_REVISION
    assert settings.device == "cpu"
    assert settings.batch_size == 8
    assert settings.cache_dir == Path("/root/.cache/huggingface")
    assert settings.local_files_only is False
    assert settings.fail_open is True


def test_provider_neutral_getters_expose_each_setting() -> None:
    from lib.knowledge_graph import config

    source = {
        "KG_BUILD_ENABLED": "1",
        "KG_GLINER2_LOCAL_FILES_ONLY": "1",
        "KG_EXTRACTOR_FAIL_OPEN": "0",
    }

    assert config.get_build_enabled(source) is True
    assert config.get_extractor_provider(source) == "gliner2_local"
    assert config.get_extractor_model(source) == DEFAULT_MODEL
    assert config.get_extractor_revision(source) == DEFAULT_REVISION
    assert config.get_extractor_device(source) == "cpu"
    assert config.get_extractor_batch_size(source) == 8
    assert config.get_extractor_cache_dir(source) == Path("/root/.cache/huggingface")
    assert config.get_extractor_local_files_only(source) is True
    assert config.get_extractor_fail_open(source) is False


def test_config_parses_supported_overrides() -> None:
    from lib.knowledge_graph.config import load_extraction_settings

    settings = load_extraction_settings(
        {
            "KG_BUILD_ENABLED": "yes",
            "KG_EXTRACTOR_PROVIDER": " gliner2_local ",
            "KG_GLINER2_MODEL": "example/research-model",
            "KG_GLINER2_REVISION": "0123456789abcdef0123456789abcdef01234567",
            "KG_GLINER2_DEVICE": " cuda:1 ",
            "KG_GLINER2_BATCH_SIZE": "16",
            "KG_GLINER2_CACHE_DIR": " C:/models/cache ",
            "KG_GLINER2_LOCAL_FILES_ONLY": "on",
            "KG_EXTRACTOR_FAIL_OPEN": "false",
        }
    )

    assert settings.build_enabled is True
    assert settings.provider == "gliner2_local"
    assert settings.model_id == "example/research-model"
    assert settings.model_revision == "0123456789abcdef0123456789abcdef01234567"
    assert settings.device == "cuda:1"
    assert settings.batch_size == 16
    assert settings.cache_dir == Path("C:/models/cache")
    assert settings.local_files_only is True
    assert settings.fail_open is False


@pytest.mark.parametrize("value", ["", "maybe", "2", "enabled"])
def test_invalid_build_boolean_fails_closed(value: str) -> None:
    from lib.knowledge_graph.config import load_extraction_settings

    settings = load_extraction_settings({"KG_BUILD_ENABLED": value})

    assert settings.build_enabled is False


@pytest.mark.parametrize("value", ["", "0", "-1", "1.5", "many"])
def test_invalid_batch_size_uses_safe_default(value: str) -> None:
    from lib.knowledge_graph.config import load_extraction_settings

    settings = load_extraction_settings({"KG_GLINER2_BATCH_SIZE": value})

    assert settings.batch_size == 8


@pytest.mark.parametrize(
    "overrides",
    [
        {"KG_BUILD_ENABLED": "1", "KG_GLINER2_REVISION": ""},
        {
            "KG_BUILD_ENABLED": "1",
            "KG_GLINER2_MODEL": "example/custom-model",
        },
        {"KG_BUILD_ENABLED": "1", "KG_GLINER2_REVISION": "main"},
    ],
)
def test_enabled_build_rejects_an_unpinned_model(overrides: dict[str, str]) -> None:
    from lib.knowledge_graph.config import (
        KnowledgeGraphConfigError,
        load_extraction_settings,
    )

    with pytest.raises(KnowledgeGraphConfigError, match="revision"):
        load_extraction_settings(overrides)


def test_settings_value_cannot_bypass_enabled_revision_validation() -> None:
    from dataclasses import replace

    from lib.knowledge_graph.config import (
        KnowledgeGraphConfigError,
        load_extraction_settings,
    )

    safe_settings = load_extraction_settings({})

    with pytest.raises(KnowledgeGraphConfigError, match="revision"):
        replace(safe_settings, build_enabled=True, model_revision="")


def test_disabled_build_tolerates_missing_revision_without_startup_failure() -> None:
    from lib.knowledge_graph.config import load_extraction_settings

    settings = load_extraction_settings(
        {
            "KG_BUILD_ENABLED": "0",
            "KG_GLINER2_MODEL": "example/custom-model",
            "KG_GLINER2_REVISION": "",
        }
    )

    assert settings.model_revision == ""


def test_optional_dependency_is_exact_and_not_in_default_dependencies() -> None:
    import tomllib

    repository_root = Path(__file__).resolve().parents[4]
    project = tomllib.loads(
        (repository_root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert project["optional-dependencies"]["knowledge-graph-local"] == [
        "gliner2[local]==1.3.2",
        "torch==2.11.0",
    ]
    assert all(
        "gliner2" not in dependency.lower() for dependency in project["dependencies"]
    )
    assert (
        "gliner2"
        not in (repository_root / "requirements.txt")
        .read_text(encoding="utf-8")
        .lower()
    )


def test_pytest_default_discovery_includes_both_knowledge_graph_suites() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    pytest_ini = (repository_root / "pytest.ini").read_text(encoding="utf-8")

    assert "aquillm/apps/knowledge_graph/tests" in pytest_ini
    assert "aquillm/lib/knowledge_graph/tests" in pytest_ini
