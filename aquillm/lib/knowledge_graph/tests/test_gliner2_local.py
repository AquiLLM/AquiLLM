from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any

import pytest

from lib.knowledge_graph.config import load_extraction_settings
from lib.knowledge_graph.extractors import gliner2_local
from lib.knowledge_graph.extractors.gliner2_local import (
    ExtractionBackendError,
    GLiNER2LocalBackend,
)
from lib.knowledge_graph.types import EntityCandidate, RelationCandidate

MODEL_ID = "fastino/gliner2-base-v1"
MODEL_REVISION = "8437ba583a733d87f56ae902f3b197934eedd58e"
GLINER2_VERSION = "1.3.2"


def _ontology() -> SimpleNamespace:
    entity_types = {
        name: SimpleNamespace(name=name, description=f"{name} description")
        for name in ("paper", "model", "dataset", "metric")
    }
    relations = {
        "uses_dataset": SimpleNamespace(
            name="uses_dataset",
            description="Connects research work to a dataset.",
            allowed_head_types=("paper", "model"),
            allowed_tail_types=("dataset",),
        ),
        "reports_metric": SimpleNamespace(
            name="reports_metric",
            description="Connects a paper to a metric.",
            allowed_head_types=("paper",),
            allowed_tail_types=("metric",),
        ),
    }
    return SimpleNamespace(
        version="1.0.0",
        checksum="ontology-checksum",
        entity_types=MappingProxyType(entity_types),
        relations=MappingProxyType(relations),
    )


def _entity_result(*entities: tuple[str, str, float, int, int]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entity_type, text, confidence, start, end in entities:
        grouped.setdefault(entity_type, []).append(
            {
                "text": text,
                "confidence": confidence,
                "start": start,
                "end": end,
            }
        )
    return {"entities": grouped}


def _spanned_relation(
    relation_type: str,
    head: tuple[str, float, int, int],
    tail: tuple[str, float, int, int],
) -> dict[str, Any]:
    head_text, head_confidence, head_start, head_end = head
    tail_text, tail_confidence, tail_start, tail_end = tail
    return {
        "relation_extraction": {
            relation_type: [
                {
                    "head": {
                        "text": head_text,
                        "confidence": head_confidence,
                        "start": head_start,
                        "end": head_end,
                    },
                    "tail": {
                        "text": tail_text,
                        "confidence": tail_confidence,
                        "start": tail_start,
                        "end": tail_end,
                    },
                }
            ]
        }
    }


def _install_fake_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    entity_results: list[dict[str, Any]] | None = None,
    relation_results: list[dict[str, Any]] | None = None,
    load_error: Exception | None = None,
    inference_error: Exception | None = None,
) -> SimpleNamespace:
    calls = SimpleNamespace(
        snapshot=[],
        pretrained=[],
        schema_entities=[],
        schema_relations=[],
        batch=[],
    )
    resolved_path = "C:/model-cache/snapshots/pinned"

    class FakeSchema:
        def entities(self, definitions):
            calls.schema_entities.append(definitions)
            return self

        def relations(self, definitions):
            calls.schema_relations.append(definitions)
            return self

    class FakeModel:
        def create_schema(self):
            return FakeSchema()

        def batch_extract(
            self,
            texts,
            schema,
            *,
            batch_size,
            include_confidence,
            include_spans,
        ):
            kwargs = {
                "batch_size": batch_size,
                "include_confidence": include_confidence,
                "include_spans": include_spans,
            }
            calls.batch.append((texts, schema, kwargs))
            if inference_error is not None:
                raise inference_error
            resolved_entities = entity_results or [{"entities": {}} for _ in texts]
            resolved_relations = relation_results or [
                {"relation_extraction": {}} for _ in texts
            ]
            return [
                {**entity_result, **relation_result}
                for entity_result, relation_result in zip(
                    resolved_entities, resolved_relations, strict=True
                )
            ]

    class FakeGLiNER2:
        @classmethod
        def from_pretrained(cls, repo_or_dir, *, map_location):
            calls.pretrained.append(((repo_or_dir,), {"map_location": map_location}))
            if load_error is not None:
                raise load_error
            return FakeModel()

    def snapshot_download(*, repo_id, revision, cache_dir, local_files_only):
        kwargs = {
            "repo_id": repo_id,
            "revision": revision,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        calls.snapshot.append(kwargs)
        if load_error is not None:
            raise load_error
        return resolved_path

    hub_module = ModuleType("huggingface_hub")
    hub_module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    provider_module = ModuleType("gliner2")
    provider_module.__version__ = GLINER2_VERSION  # type: ignore[attr-defined]
    provider_module.GLiNER2 = FakeGLiNER2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setitem(sys.modules, "gliner2", provider_module)
    return calls


@pytest.fixture(autouse=True)
def _clear_process_model_cache() -> None:
    gliner2_local._MODEL_CACHE.clear()


def _backend(**overrides: str) -> GLiNER2LocalBackend:
    source = {
        "KG_GLINER2_MODEL": MODEL_ID,
        "KG_GLINER2_REVISION": MODEL_REVISION,
        "KG_GLINER2_CACHE_DIR": "C:/model-cache",
        "KG_GLINER2_DEVICE": "cpu",
        **overrides,
    }
    return GLiNER2LocalBackend(settings=load_extraction_settings(source))


def test_loads_exact_pinned_snapshot_and_supported_gliner2_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_provider(monkeypatch)

    _backend(KG_GLINER2_LOCAL_FILES_ONLY="1").extract_batch(
        ("A paper.",), ontology=_ontology()
    )

    assert calls.snapshot == [
        {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "cache_dir": Path("C:/model-cache"),
            "local_files_only": True,
        }
    ]
    assert calls.pretrained == [
        (("C:/model-cache/snapshots/pinned",), {"map_location": "cpu"})
    ]
    expected_kwargs = {
        "batch_size": 8,
        "include_confidence": True,
        "include_spans": True,
    }
    assert calls.schema_entities == [
        {
            "paper": "paper description",
            "model": "model description",
            "dataset": "dataset description",
            "metric": "metric description",
        }
    ]
    assert calls.schema_relations == [
        {
            "uses_dataset": "Connects research work to a dataset.",
            "reports_metric": "Connects a paper to a metric.",
        }
    ]
    assert len(calls.batch) == 1
    assert calls.batch[0][0] == ["A paper."]
    assert calls.batch[0][2] == expected_kwargs


def test_empty_input_returns_without_importing_or_loading_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "gliner2", raising=False)
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)

    assert _backend().extract_batch((), ontology=_ontology()) == ()
    assert gliner2_local._MODEL_CACHE == {}


def test_nonempty_extraction_rejects_unpinned_revision_before_provider_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_provider(monkeypatch)

    with pytest.raises(ExtractionBackendError, match="revision"):
        _backend(KG_GLINER2_REVISION="main").extract_batch(
            ("A paper.",), ontology=_ontology()
        )

    assert calls.snapshot == []
    assert calls.pretrained == []


def test_model_load_is_shared_once_across_instances_and_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_provider(monkeypatch)

    def extract(_: int):
        return _backend().extract_batch(("A paper.",), ontology=_ontology())

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(extract, range(16)))

    assert all(len(result) == 1 for result in results)
    assert len(calls.snapshot) == 1
    assert len(calls.pretrained) == 1


def test_normalizes_entities_and_spanned_relations_to_neutral_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Qwen3 uses MMLU."
    calls = _install_fake_provider(
        monkeypatch,
        entity_results=[
            _entity_result(
                ("model", "Qwen3", 0.95, 0, 5),
                ("dataset", "MMLU", 0.91, 11, 15),
            )
        ],
        relation_results=[
            _spanned_relation(
                "uses_dataset", ("Qwen3", 0.89, 0, 5), ("MMLU", 0.87, 11, 15)
            )
        ],
    )

    result = _backend().extract_batch((text,), ontology=_ontology())[0]

    assert calls.batch
    assert result.entities == (
        EntityCandidate("model", "Qwen3", 0, 5, 0.95),
        EntityCandidate("dataset", "MMLU", 11, 15, 0.91),
    )
    assert result.relations == (
        RelationCandidate(
            "uses_dataset", "Qwen3", "MMLU", 0, 5, 11, 15, 0.87
        ),
    )
    assert result.diagnostics == ()


@pytest.mark.parametrize(
    ("raw_entity", "expected_code"),
    [
        (("model", "Qwen3", 0.9, -1, 5), "malformed_entity_span"),
        (("model", "Qwen3", 0.9, 0, 99), "malformed_entity_span"),
        (("model", "Qwen3", float("nan"), 0, 5), "invalid_entity_confidence"),
        (("unknown", "Qwen3", 0.9, 0, 5), "unknown_entity_type"),
    ],
)
def test_rejects_invalid_entities_to_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    raw_entity: tuple[str, str, float, int, int],
    expected_code: str,
) -> None:
    _install_fake_provider(
        monkeypatch, entity_results=[_entity_result(raw_entity)]
    )

    result = _backend().extract_batch(("Qwen3",), ontology=_ontology())[0]

    assert result.entities == ()
    assert [diagnostic.code for diagnostic in result.diagnostics] == [expected_code]
    assert result.diagnostics[0].candidate_kind == "entity"
    assert result.diagnostics[0].input_index == 0


def test_rejects_unknown_relation_and_nan_endpoint_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = _entity_result(
        ("model", "Qwen3", 0.95, 0, 5),
        ("dataset", "MMLU", 0.91, 11, 15),
    )
    relation_results = [
        {
            "relation_extraction": {
                "invented_relation": [
                    {
                        "head": {
                            "text": "Qwen3",
                            "confidence": 0.8,
                            "start": 0,
                            "end": 5,
                        },
                        "tail": {
                            "text": "MMLU",
                            "confidence": 0.8,
                            "start": 11,
                            "end": 15,
                        },
                    }
                ],
                "uses_dataset": [
                    {
                        "head": {
                            "text": "Qwen3",
                            "confidence": float("nan"),
                            "start": 0,
                            "end": 5,
                        },
                        "tail": {
                            "text": "MMLU",
                            "confidence": 0.8,
                            "start": 11,
                            "end": 15,
                        },
                    }
                ],
            }
        }
    ]
    _install_fake_provider(
        monkeypatch, entity_results=[entities], relation_results=relation_results
    )

    result = _backend().extract_batch(
        ("Qwen3 uses MMLU.",), ontology=_ontology()
    )[0]

    assert result.relations == ()
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "unknown_relation_type",
        "invalid_relation_confidence",
    ]


def test_rejects_malformed_relation_span_instead_of_matching_by_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(
        monkeypatch,
        entity_results=[
            _entity_result(
                ("model", "Qwen3", 0.95, 0, 5),
                ("dataset", "MMLU", 0.91, 11, 15),
            )
        ],
        relation_results=[
            _spanned_relation(
                "uses_dataset", ("Qwen3", 0.8, -1, 5), ("MMLU", 0.8, 11, 15)
            )
        ],
    )

    result = _backend().extract_batch(
        ("Qwen3 uses MMLU.",), ontology=_ontology()
    )[0]

    assert result.relations == ()
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "malformed_relation_span"
    ]


def test_text_only_relation_endpoints_resolve_to_unique_compatible_mentions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(
        monkeypatch,
        entity_results=[
            _entity_result(
                ("model", "Qwen3", 0.95, 0, 5),
                ("dataset", "MMLU", 0.91, 11, 15),
            )
        ],
        relation_results=[
            {
                "relation_extraction": {
                    "uses_dataset": [
                        {
                            "head": {"text": "Qwen3", "confidence": 0.8},
                            "tail": {"text": "MMLU", "confidence": 0.7},
                        }
                    ]
                }
            }
        ],
    )

    result = _backend().extract_batch(
        ("Qwen3 uses MMLU.",), ontology=_ontology()
    )[0]

    assert result.relations == (
        RelationCandidate(
            "uses_dataset", "Qwen3", "MMLU", 0, 5, 11, 15, 0.7
        ),
    )
    assert result.diagnostics == ()


def test_ambiguous_text_only_endpoint_is_retained_only_as_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Qwen3 compares to Qwen3 on MMLU."
    _install_fake_provider(
        monkeypatch,
        entity_results=[
            _entity_result(
                ("model", "Qwen3", 0.95, 0, 5),
                ("model", "Qwen3", 0.94, 18, 23),
                ("dataset", "MMLU", 0.91, 27, 31),
            )
        ],
        relation_results=[
            {
                "relation_extraction": {
                    "uses_dataset": [
                        {
                            "head": {"text": "Qwen3", "confidence": 0.8},
                            "tail": {"text": "MMLU", "confidence": 0.7},
                        }
                    ]
                }
            }
        ],
    )

    result = _backend().extract_batch((text,), ontology=_ontology())[0]

    assert result.relations == ()
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "ambiguous_relation_endpoint"
    assert ("endpoint", "head") in diagnostic.details
    assert ("endpoint_text", "Qwen3") in diagnostic.details


def test_disallowed_endpoint_type_is_diagnostic_not_promoted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(
        monkeypatch,
        entity_results=[
            _entity_result(
                ("metric", "Qwen3", 0.95, 0, 5),
                ("dataset", "MMLU", 0.91, 11, 15),
            )
        ],
        relation_results=[
            _spanned_relation(
                "uses_dataset", ("Qwen3", 0.8, 0, 5), ("MMLU", 0.7, 11, 15)
            )
        ],
    )

    result = _backend().extract_batch(
        ("Qwen3 uses MMLU.",), ontology=_ontology()
    )[0]

    assert result.relations == ()
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "disallowed_relation_endpoint"
    ]
    assert ("endpoint", "head") in result.diagnostics[0].details


@pytest.mark.parametrize("stage", ["load", "inference"])
def test_provider_failures_are_wrapped_without_django(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    provider_error = RuntimeError(f"{stage} failed")
    _install_fake_provider(
        monkeypatch,
        load_error=provider_error if stage == "load" else None,
        inference_error=provider_error if stage == "inference" else None,
    )

    with pytest.raises(ExtractionBackendError, match=stage) as captured:
        _backend().extract_batch(("A paper.",), ontology=_ontology())

    assert captured.value.__cause__ is provider_error


def test_rejects_a_different_gliner2_runtime_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    sys.modules["gliner2"].__version__ = "1.3.1"  # type: ignore[attr-defined]

    with pytest.raises(ExtractionBackendError, match="1.3.2"):
        _backend().extract_batch(("A paper.",), ontology=_ontology())


def test_backend_module_has_no_top_level_provider_or_framework_imports() -> None:
    source = Path(gliner2_local.__file__).read_text(encoding="utf-8")

    assert "from gliner2 import" not in source.partition("def _load_model")[0]
    assert "from huggingface_hub import" not in source.partition("def _load_model")[0]
    assert "django" not in source.lower()
