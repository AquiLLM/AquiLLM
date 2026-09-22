from __future__ import annotations

import json

import pytest


def test_direct_vllm_adapter_never_debug_logs_sentinel_prompt_text(monkeypatch, caplog):
    """Using an SDK logger that emits request messages would leak collection text."""

    from apps.collections.services import schema_generation_support
    from apps.collections.services.schema_generation import generate_schema_candidate

    sentinel = "SENTINEL COLLECTION TEXT MUST NEVER BE LOGGED"
    captured = {}

    def post(url, payload, headers, timeout):
        captured.update(url=url, payload=payload, headers=headers, timeout=timeout)
        return {
            "choices": [{"message": {"content": json.dumps({
                "entities": [
                    {"name": "researcher", "description": "A person.", "aliases": []},
                    {"name": "organization", "description": "A company.", "aliases": []},
                ],
                "relations": [{"name": "works_for", "description": "Employment.", "direction": "directed", "allowed_head_types": ["researcher"], "allowed_tail_types": ["organization"]}],
            })}}]
        }

    monkeypatch.setattr(schema_generation_support, "_post_local_vllm_json", post)
    monkeypatch.setenv("VLLM_API_KEY", "sentinel-api-key")
    caplog.set_level("DEBUG")

    generate_schema_candidate([{"document_id": "doc", "chunk_id": 1, "text": sentinel}])

    assert captured["url"] == "http://vllm:8000/v1/chat/completions"
    assert sentinel in captured["payload"]["messages"][0]["content"]
    assert sentinel not in caplog.text
    assert "sentinel-api-key" not in caplog.text


def test_evidence_removes_zero_evidence_types_and_never_returns_raw_text():
    """Removing evidence filtering would allow an unsupported generated definition."""

    from apps.collections.services.schema_generation import collect_candidate_evidence
    from apps.collections.services.schema_generation import normalize_schema_candidate
    from lib.knowledge_graph.types import EntityCandidate, ExtractionBatchResult, RelationCandidate

    candidate = normalize_schema_candidate(
        {
            "entities": [
                {"name": "researcher", "description": "A person.", "aliases": []},
                {"name": "organization", "description": "A company.", "aliases": []},
                {"name": "paper", "description": "A paper.", "aliases": []},
            ],
            "relations": [
                {"name": "works_for", "description": "Employment.", "direction": "directed", "allowed_head_types": ["researcher"], "allowed_tail_types": ["organization"]},
                {"name": "authors", "description": "Authorship.", "direction": "directed", "allowed_head_types": ["researcher"], "allowed_tail_types": ["paper"]},
            ],
        }
    )

    class Backend:
        def extract_batch(self, texts, *, ontology):
            return (
                ExtractionBatchResult(
                    entities=(
                        EntityCandidate("researcher", "Alice", 0, 5, 0.9),
                        EntityCandidate("organization", "Acme", 15, 19, 0.8),
                    ),
                    relations=(RelationCandidate("works_for", "Alice", "Acme", 0, 5, 15, 19, 0.7),),
                    diagnostics=(),
                ),
            )

    definitions, statistics = collect_candidate_evidence(
        candidate,
        [{"document_id": "document-1", "chunk_id": 4, "text": "Alice works for Acme."}],
        backend=Backend(),
    )

    assert [item["key"] for item in definitions["entities"]] == ["organization", "researcher"]
    assert [item["key"] for item in definitions["relations"]] == ["works_for"]
    assert statistics == {
        "entities": {
            "organization": {"count": 1, "mean_confidence": 0.8, "sources": [{"document_id": "document-1", "chunk_id": 4}]},
            "researcher": {"count": 1, "mean_confidence": 0.9, "sources": [{"document_id": "document-1", "chunk_id": 4}]},
        },
        "relations": {
            "works_for": {"count": 1, "mean_confidence": 0.7, "sources": [{"document_id": "document-1", "chunk_id": 4}]},
        },
    }
    assert "Alice works for Acme." not in repr(statistics)
