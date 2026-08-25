from __future__ import annotations

import json

import pytest


def test_local_vllm_configuration_uses_bounded_defaults_and_local_service(monkeypatch):
    """A remote host or unbounded caps must not become an inference request."""

    from apps.collections.services.schema_generation import (
        SchemaGenerationConfigurationError,
        load_schema_generation_config,
    )

    monkeypatch.delenv("KG_SCHEMA_GENERATION_MAX_CHUNKS", raising=False)
    monkeypatch.delenv("KG_SCHEMA_GENERATION_MAX_CHARACTERS", raising=False)
    monkeypatch.delenv("KG_SCHEMA_GENERATION_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    config = load_schema_generation_config()

    assert config.max_chunks == 32
    assert config.max_characters == 48_000
    assert config.timeout_seconds == 180
    assert config.base_url == "http://vllm:8000/v1"

    monkeypatch.setenv("VLLM_BASE_URL", "https://api.openai.com/v1")
    with pytest.raises(SchemaGenerationConfigurationError, match="Docker service"):
        load_schema_generation_config()


def test_generate_schema_candidate_calls_only_supplied_local_client(monkeypatch):
    """Replacing the local client with a remote fallback must make this fail."""

    from apps.collections.services.schema_generation import generate_schema_candidate

    captured = {}

    def client(url, payload, headers, timeout):
        captured.update(url=url, payload=payload, headers=headers, timeout=timeout)
        return {"choices": [{"message": {"content": json.dumps({
            "entities": [
                {"name": "Researcher", "description": "A researcher.", "aliases": ["scientist"]},
                {"name": "Organization", "description": "An organization.", "aliases": []},
            ],
            "relations": [{"name": "Works For", "description": "Employment.", "direction": "directed", "allowed_head_types": ["Researcher"], "allowed_tail_types": ["Organization"]}],
        })}}]}
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm:8000")
    monkeypatch.setenv("VLLM_SERVED_MODEL_NAME", "local-test-model")

    candidate = generate_schema_candidate(
        [{"document_id": "doc-1", "chunk_id": 1, "text": "Alice works at Acme."}],
        client=client,
    )

    assert candidate["entities"][0]["key"] == "organization"
    assert candidate["entities"][1]["key"] == "researcher"
    assert candidate["relations"][0]["key"] == "works_for"
    assert captured["url"] == "http://vllm:8000/v1/chat/completions"
    assert captured["payload"]["model"] == "local-test-model"
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_candidate_rejects_duplicate_names_invalid_endpoints_and_invalid_ontology():
    """Dropping candidate validation would allow malformed schema drafts."""

    from apps.collections.services.schema_generation import InvalidSchemaCandidate
    from apps.collections.services.schema_generation import normalize_schema_candidate

    with pytest.raises(InvalidSchemaCandidate, match="duplicate"):
        normalize_schema_candidate(
            {
                "entities": [
                    {"name": "Researcher", "description": "A person.", "aliases": []},
                    {"name": "researcher", "description": "Duplicate.", "aliases": []},
                ],
                "relations": [
                    {
                        "name": "works_for",
                        "description": "Employment.",
                        "direction": "directed",
                        "allowed_head_types": ["researcher"],
                        "allowed_tail_types": ["organization"],
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    ("entity_count", "relation_count", "message"),
    ((1, 1, "2-24 entities"), (25, 1, "2-24 entities"), (2, 0, "1-32 relations"), (2, 33, "1-32 relations")),
)
def test_candidate_enforces_exact_entity_and_relation_bounds(entity_count, relation_count, message):
    """Weakening strict bounds permits an unbounded local inference candidate."""

    from apps.collections.services.schema_generation import InvalidSchemaCandidate, normalize_schema_candidate

    candidate = {
        "entities": [{"name": f"type_{index}", "description": "A type.", "aliases": []} for index in range(entity_count)],
        "relations": [{"name": f"relation_{index}", "description": "A relation.", "direction": "directed", "allowed_head_types": ["type_0"], "allowed_tail_types": ["type_1"]} for index in range(relation_count)],
    }

    with pytest.raises(InvalidSchemaCandidate, match=message):
        normalize_schema_candidate(candidate)

    with pytest.raises(InvalidSchemaCandidate, match="unknown endpoint"):
        normalize_schema_candidate(
            {
                "entities": [
                    {"name": "researcher", "description": "A person.", "aliases": []},
                    {"name": "organization", "description": "A company.", "aliases": []},
                ],
                "relations": [
                    {
                        "name": "works_for",
                        "description": "Employment.",
                        "direction": "directed",
                        "allowed_head_types": ["researcher"],
                        "allowed_tail_types": ["missing"],
                    }
                ],
            }
        )


def test_balanced_sampler_round_robins_documents_and_enforces_both_caps():
    """Changing the sampler to take one document first would starve other documents."""

    from apps.collections.services.schema_generation import SchemaSample
    from apps.collections.services.schema_generation import balanced_samples

    samples = balanced_samples(
        {
            "document-a": [
                SchemaSample("document-a", 1, 0, "alpha"),
                SchemaSample("document-a", 2, 1, "bravo"),
            ],
            "document-b": [SchemaSample("document-b", 3, 0, "charlie")],
        },
        max_chunks=3,
        max_characters=13,
    )

    assert [sample.document_id for sample in samples] == ["document-a", "document-b", "document-a"]
    assert [sample.text for sample in samples] == ["alpha", "charlie", "b"]
    assert len(samples) == 3
    assert sum(len(sample.text) for sample in samples) == 13


def test_public_sampler_reads_at_most_one_bounded_chunk_per_selected_slot(monkeypatch):
    """Replacing bounded per-slot reads with a full content queryset must fail."""

    from apps.collections.services import schema_generation

    queried = []
    rows = {
        "document-a": [(1, 0, "alpha"), (2, 1, "bravo")],
        "document-b": [(3, 0, "charlie")],
    }

    monkeypatch.setattr(
        schema_generation,
        "_eligible_collection_documents",
        lambda collection_id: iter(
            {"id": document_id}
            for document_id in sorted(rows)
        ),
    )
    monkeypatch.setattr(
        schema_generation,
        "_next_sample_chunk",
        lambda document_id, after_chunk_number, character_limit: queried.append(
            (document_id, after_chunk_number, character_limit)
        ) or next(
            (
                schema_generation.SchemaSample(document_id, chunk_id, number, text[:character_limit])
                for chunk_id, number, text in rows[document_id]
                if number > after_chunk_number
            ),
            None,
        ),
    )

    samples = schema_generation.sample_collection_chunks(1, max_chunks=3, max_characters=13)

    assert [sample.document_id for sample in samples] == ["document-a", "document-b", "document-a"]
    assert [sample.text for sample in samples] == ["alpha", "charlie", "b"]
    assert queried == [("document-a", -1, 13), ("document-b", -1, 8), ("document-a", 0, 1)]


def test_source_signature_streams_only_document_identity_and_hash(monkeypatch):
    """Adding collection text to a durable signature input would leak source content."""

    from apps.collections.services import schema_generation

    monkeypatch.setattr(
        schema_generation,
        "_completed_collection_documents",
        lambda collection_id, **kwargs: iter((
            {"id": "document-b", "full_text_hash": "b" * 64},
            {"id": "document-a", "full_text_hash": "a" * 64},
        )),
    )

    signature = schema_generation.collection_source_signature(1)

    assert signature == "b7767870591b408c300da43aa8ea4fd50ac5897c6c8a30d5361e1a79bf0cfa52"


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
