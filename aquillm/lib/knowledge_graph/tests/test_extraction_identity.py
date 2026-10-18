from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from lib.knowledge_graph.config import load_extraction_settings


def test_identity_is_bounded_stable_and_ignores_operational_settings():
    from lib.knowledge_graph.extraction_identity import (
        extraction_contract,
        extraction_identity,
    )

    settings = load_extraction_settings({})
    identity = extraction_identity(settings)
    assert identity.startswith("extract-v2:")
    assert len(identity) == 75
    assert identity == extraction_identity(replace(settings))
    assert identity == extraction_identity(
        replace(
            settings,
            cache_dir=Path("elsewhere"),
            local_files_only=True,
            fail_open=not settings.fail_open,
            build_enabled=not settings.build_enabled,
        )
    )
    contract = extraction_contract(settings)
    with pytest.raises(FrozenInstanceError):
        contract.provider = "changed"


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "other/checkpoint"},
        {"model_revision": "a" * 40},
        {"batch_size": 2},
        {"device": "cuda"},
        {"max_batch_characters": 100},
        {"provider": "fixture"},
    ],
)
def test_identity_changes_with_effective_settings(change):
    from lib.knowledge_graph.extraction_identity import extraction_identity

    settings = load_extraction_settings({})
    assert extraction_identity(settings) != extraction_identity(
        replace(settings, **change)
    )


@pytest.mark.parametrize(
    "name",
    [
        "GLINER2_VERSION",
        "ADAPTER_CONTRACT_VERSION",
        "NORMALIZER_CONTRACT_VERSION",
    ],
)
def test_identity_binds_runtime_adapter_and_normalizer(monkeypatch, name):
    import lib.knowledge_graph.extraction_identity as identity_module

    settings = load_extraction_settings({})
    before = identity_module.extraction_identity(settings)
    monkeypatch.setattr(identity_module, name, "changed-contract")
    assert before != identity_module.extraction_identity(settings)


def test_candidate_compiler_has_distinct_manifest_and_identity():
    from lib.knowledge_graph.extraction_identity import (
        extraction_contract,
        extraction_identity,
    )

    settings = load_extraction_settings({})
    candidate = extraction_contract(
        settings, schema_compiler_version="gliner2-structures-v1"
    )
    assert candidate.schema_compiler_version == "gliner2-structures-v1"
    assert dict(candidate.inference_options) == {
        "batch_size": 8,
        "format_results": False,
        "include_confidence": True,
        "include_spans": True,
        "threshold": 0.5,
        "num_workers": 0,
        "max_len": None,
    }
    assert extraction_identity(settings) != extraction_identity(
        settings, schema_compiler_version="gliner2-structures-v1"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"model_revision": "main"},
        {"batch_size": 0},
        {"batch_size": True},
        {"max_batch_characters": 0},
        {"device": ""},
    ],
)
def test_invalid_contract_fails_closed(change):
    from lib.knowledge_graph.extraction_identity import extraction_identity

    with pytest.raises(ValueError):
        extraction_identity(replace(load_extraction_settings({}), **change))


def test_fixture_provider_contract_does_not_require_a_gliner_checkpoint():
    from lib.knowledge_graph.extraction_identity import extraction_identity

    settings = replace(
        load_extraction_settings({}), provider="fixture", model_revision="fixture-v1"
    )
    assert extraction_identity(settings) == extraction_identity(settings)


def test_effective_options_are_shared_with_adapter(monkeypatch):
    from lib.knowledge_graph.extraction_identity import extraction_contract
    from lib.knowledge_graph.extractors import gliner2_local

    calls = []
    schema = type(
        "Schema",
        (),
        {"entities": lambda self, _: self, "relations": lambda self, _: self},
    )()
    model = type(
        "Model",
        (),
        {
            "create_schema": lambda self: schema,
            "batch_extract": lambda self, texts, schema, **options: (
                calls.append(options) or [{"entities": [{}]}]
            ),
        },
    )()
    ontology = type("Ontology", (), {"entity_types": {}, "relations": {}})()
    monkeypatch.setattr(gliner2_local, "_load_model", lambda _: model)
    settings = load_extraction_settings({})
    gliner2_local.GLiNER2LocalBackend(settings=settings).extract_batch(
        ("text",), ontology=ontology
    )
    assert calls == [dict(extraction_contract(settings).inference_options)]
