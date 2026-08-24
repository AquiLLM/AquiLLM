"""Bounded, local-only collection schema proposal helpers.

This module intentionally keeps collection text and inference output in local
variables.  Callers receive only canonical definitions and aggregate evidence.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

_DEFAULT_MAX_CHUNKS = 32
_DEFAULT_MAX_CHARACTERS = 48_000
_DEFAULT_TIMEOUT_SECONDS = 180
_MAX_ENTITY_TYPES = 24
_MIN_ENTITY_TYPES = 2
_MAX_RELATION_TYPES = 32
_MIN_RELATION_TYPES = 1
_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class SchemaGenerationConfigurationError(ValueError):
    """The local-only generation configuration is unsafe or malformed."""


class InvalidSchemaCandidate(ValueError):
    """A model proposal cannot become a bounded collection draft."""


@dataclass(frozen=True, slots=True)
class SchemaGenerationConfig:
    base_url: str
    api_key: str
    model: str
    max_chunks: int
    max_characters: int
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class SchemaSample:
    document_id: str
    chunk_id: int
    chunk_number: int
    text: str


def _positive_env_int(name: str, default: int) -> int:
    value = (os.environ.get(name) or "").strip()
    if not value:
        return default
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise SchemaGenerationConfigurationError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise SchemaGenerationConfigurationError(f"{name} must be a positive integer")
    return parsed


def _enabled_from_environment() -> bool:
    return (os.environ.get("KG_SCHEMA_GENERATION_ENABLED") or "0").strip() == "1"


def load_schema_generation_config() -> SchemaGenerationConfig:
    """Load bounded settings and reject every endpoint outside the vLLM service."""

    raw_url = (os.environ.get("VLLM_BASE_URL") or "http://vllm:8000/v1").strip()
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SchemaGenerationConfigurationError("VLLM_BASE_URL must be an HTTP(S) URL")
    if parsed.hostname.lower() != "vllm":
        raise SchemaGenerationConfigurationError(
            "VLLM_BASE_URL host must equal the configured Docker service host"
        )
    normalized_path = parsed.path.rstrip("/")
    if not normalized_path:
        normalized_path = "/v1"
    elif normalized_path != "/v1":
        raise SchemaGenerationConfigurationError("VLLM_BASE_URL must use the /v1 API path")
    base_url = f"{parsed.scheme}://{parsed.netloc}{normalized_path}"
    model = (os.environ.get("VLLM_SERVED_MODEL_NAME") or "qwen3.5:27b").strip()
    return SchemaGenerationConfig(
        base_url=base_url,
        api_key=(os.environ.get("VLLM_API_KEY") or "EMPTY").strip() or "EMPTY",
        model=model,
        max_chunks=min(_positive_env_int("KG_SCHEMA_GENERATION_MAX_CHUNKS", _DEFAULT_MAX_CHUNKS), _DEFAULT_MAX_CHUNKS),
        max_characters=min(_positive_env_int("KG_SCHEMA_GENERATION_MAX_CHARACTERS", _DEFAULT_MAX_CHARACTERS), _DEFAULT_MAX_CHARACTERS),
        timeout_seconds=_positive_env_int("KG_SCHEMA_GENERATION_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS),
    )


def _completed_collection_documents(collection_id: int) -> list[dict[str, object]]:
    from django.apps import apps

    records: list[dict[str, object]] = []
    for name in (
        "PDFDocument", "TeXDocument", "RawTextDocument", "VTTDocument",
        "HandwrittenNotesDocument", "ImageUploadDocument", "MediaUploadDocument",
        "DocumentFigure",
    ):
        model = apps.get_model("apps_documents", name)
        records.extend(
            model.objects.filter(collection_id=collection_id, ingestion_complete=True)
            .values("id", "full_text_hash")
        )
    return sorted(records, key=lambda record: str(record["id"]))


def collection_source_signature(collection_id) -> str:
    """Return a text-free signature for completed collection document content."""

    if type(collection_id) is not int or collection_id <= 0:
        raise ValueError("collection_id must be a positive database integer")
    digest = hashlib.sha256()
    for document in _completed_collection_documents(collection_id):
        digest.update(str(document["id"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(document["full_text_hash"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def sample_collection_chunks(collection_id, max_chunks, max_characters):
    """Select deterministic, round-robin text chunks without exceeding either cap."""

    if type(max_chunks) is not int or max_chunks <= 0:
        raise ValueError("max_chunks must be a positive integer")
    if type(max_characters) is not int or max_characters <= 0:
        raise ValueError("max_characters must be a positive integer")
    from apps.documents.models.chunks import TextChunk

    document_ids = [row["id"] for row in _completed_collection_documents(collection_id)]
    if not document_ids:
        return []
    groups: dict[str, list[SchemaSample]] = defaultdict(list)
    rows = (
        TextChunk.objects.filter(doc_id__in=document_ids, modality=TextChunk.Modality.TEXT)
        .exclude(content="")
        .order_by("doc_id", "chunk_number", "pk")
        .values("pk", "doc_id", "chunk_number", "content")
    )
    for row in rows:
        text = row["content"].strip()
        if text:
            groups[str(row["doc_id"])].append(
                SchemaSample(str(row["doc_id"]), int(row["pk"]), int(row["chunk_number"]), text)
            )
    return balanced_samples(groups, max_chunks=max_chunks, max_characters=max_characters)


def balanced_samples(
    groups: dict[str, list[SchemaSample]], *, max_chunks: int, max_characters: int
) -> list[SchemaSample]:
    """Round-robin ordered document groups and truncate only at the hard cap."""

    selected: list[SchemaSample] = []
    used_characters = 0
    offsets = {document_id: 0 for document_id in sorted(groups)}
    while len(selected) < max_chunks and used_characters < max_characters:
        progressed = False
        for document_id in sorted(groups):
            offset = offsets[document_id]
            if offset >= len(groups[document_id]) or len(selected) >= max_chunks:
                continue
            remaining = max_characters - used_characters
            if remaining <= 0:
                break
            sample = groups[document_id][offset]
            offsets[document_id] = offset + 1
            selected.append(
                SchemaSample(sample.document_id, sample.chunk_id, sample.chunk_number, sample.text[:remaining])
            )
            used_characters += len(selected[-1].text)
            progressed = True
            if used_characters >= max_characters:
                break
        if not progressed:
            break
    return selected


from .schema_generation_support import (
    collect_candidate_evidence,
    generate_schema_candidate,
    normalize_schema_candidate,
)


__all__ = [
    "InvalidSchemaCandidate", "SchemaGenerationConfig", "SchemaGenerationConfigurationError",
    "SchemaSample", "collect_candidate_evidence", "collection_source_signature",
    "generate_schema_candidate", "load_schema_generation_config", "normalize_schema_candidate",
    "balanced_samples", "sample_collection_chunks",
]
