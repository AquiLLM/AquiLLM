"""Provider doubles for graph adapter contract tests; no model execution."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

GLINER2_VERSION = "1.3.2"


def _install_fake_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    entity_results: list[dict[str, Any]] | None = None,
    relation_results: list[dict[str, Any]] | None = None,
    raw_results: list[dict[str, Any]] | None = None,
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
            format_results,
            include_confidence,
            include_spans,
            threshold,
            num_workers,
            max_len,
        ):
            kwargs = {
                "batch_size": batch_size,
                "format_results": format_results,
                "include_confidence": include_confidence,
                "include_spans": include_spans,
                "threshold": threshold,
                "num_workers": num_workers,
                "max_len": max_len,
            }
            calls.batch.append((texts, schema, kwargs))
            if inference_error is not None:
                raise inference_error
            if raw_results is not None:
                return raw_results
            resolved_entities = entity_results or [{"entities": [{}]} for _ in texts]
            default_relations = {name: [] for name in calls.schema_relations[-1]}
            resolved_relations = relation_results or [default_relations for _ in texts]
            return [
                {**default_relations, **entity_result, **relation_result}
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
