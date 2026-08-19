"""Pure-Python, deterministic evaluation support for the knowledge graph.

The loaders resolve their default fixture paths beside this module, so they are
safe to use from any working directory.  They make no Django, ORM, LLM,
GLiNER2, database, or network calls.

Metrics use structural set equality: each expected/predicted record is
canonicalized as sorted JSON before set intersection.  Precision and recall use
``1.0`` when their denominator is zero (no predicted records for precision, or
no gold records for recall); this makes empty-vs-empty a perfect match.

Usage from ``aquillm/``::

    python -m apps.knowledge_graph.evals.run_kg_eval --baseline-only

``--baseline-only`` prints compact, key-sorted JSON.  It records vector result
IDs only when injected by ``--retrieval-results`` or supplied by a fixture; a
case with neither emits ``SKIP`` and never fabricates IDs or scores.

Retrieval-result JSON maps each case ID either to a simple result-ID list or to
``{"result_ids": [...], "id_collections": {"<id>": "<collection_id>"}}``.
Native integer IDs need this optional collection evidence to receive a resolved
security status; JSON object keys use the string form of each result ID.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from contextlib import nullcontext
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from hashlib import sha256
from math import isfinite, log2
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

import yaml

from .fixture_manifest import (
    FixtureEmbeddingBinding,
    FixtureValidationError,
    ResolvedFixtureManifest,
    assemble_fixture_document,
    canonical_embedding_sha256,
    embedding_endpoint_signature,
    fixture_checksum,
    is_safe_huggingface_repo_id,
    load_fixture_manifest,
)
from .fixture_manifest import (
    validate_fixture_manifest as _validate_fixture_manifest,
)

_HERE = Path(__file__).resolve().parent
_DEFAULT_EXTRACTION_CASES_PATH = _HERE / "extraction_cases.yaml"
_DEFAULT_RETRIEVAL_CASES_PATH = _HERE / "retrieval_cases.yaml"
_DEFAULT_RUNBOOK_PATH = (
    _HERE.parents[3]
    / "docs"
    / "documents"
    / "operations"
    / "knowledge-graph-overlay-runbook.md"
)
_RERANK_TEMPLATE_PATH = (
    _HERE.parents[3]
    / "deploy"
    / "docker"
    / "vllm"
    / "chat_templates"
    / "qwen3_vl_reranker.jinja"
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_STRICT_EMBED_EXTRA_ARGS = (
    "--quantization bitsandbytes --load-format bitsandbytes "
    "--model-loader-extra-config "
    '\'{"load_in_4bit":true,"bnb_4bit_compute_dtype":"float16",'
    '"bnb_4bit_quant_type":"nf4","bnb_4bit_use_double_quant":true}\''
)
_STRICT_RERANK_EXTRA_ARGS = (
    "--chat-template /templates/qwen3_vl_reranker.jinja "
    "--hf-overrides "
    '\'{"architectures":["Qwen3VLForSequenceClassification"],'
    '"classifier_from_token":["no","yes"],'
    '"is_original_qwen3_reranker":true}\''
)
_COMPARISON_ARMS = ("vector_only", "one_hop", "ppr_v1")
_FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "candidate_contributions",
        "node_scores",
        "ppr_scores",
        "private_trace",
        "restart_vector",
        "seed_weights",
    }
)
_GATE_REQUIREMENTS = (
    (
        "Permission isolation",
        "zero inaccessible chunks",
    ),
    (
        "Fail-open parity",
        "exact baseline on graph miss/error",
    ),
    (
        "Identity precision",
        "automatic links stricter than candidates",
    ),
    (
        "Retrieval quality",
        "positive Recall@10 and nDCG movement on "
        "relationship/alias/cross-document/cross-collection cases",
    ),
    (
        "Multi-hop value",
        "PPR no worse than one-hop and better on at least one distance-two case",
    ),
    (
        "Latency",
        "graph p95 within the configured local-DB budget",
    ),
    (
        "Determinism",
        "repeated PPR ranking is identical",
    ),
    (
        "Citations",
        "100% curated real-chunk evidence coverage",
    ),
)
PENDING_GATE_TABLE = "\n".join(
    (
        "| Gate | Required outcome | Current value | Status |",
        "| --- | --- | --- | --- |",
        *(
            f"| {name} | {requirement} | `PENDING_MEASUREMENT` | "
            "`PENDING_MEASUREMENT` |"
            for name, requirement in _GATE_REQUIREMENTS
        ),
    )
)


class ComparisonValidationError(ValueError):
    """Raised when a comparison request or bundle is not exact and reproducible."""


class ComparisonAborted(RuntimeError):
    """Raised when one arm cannot be measured from the shared snapshot."""


def _rerank_template_checksum() -> str:
    try:
        return sha256(_RERANK_TEMPLATE_PATH.read_bytes()).hexdigest()
    except OSError as error:
        raise ComparisonAborted(
            "strict local reranker chat template is unavailable"
        ) from error


def _validate_live_embedding_contract(
    binding: FixtureEmbeddingBinding,
    *,
    base_url: str,
    api_key: str,
    configured_sidecar_api_key: str | None,
    configured_model: str,
    configured_checkpoint: str | None,
    configured_tokenizer_checkpoint: str | None,
    configured_code_checkpoint: str | None,
    configured_extra_args: str | None,
    configured_trust_remote_code: str | None,
    configured_runner: str | None,
    configured_dtype: str | None,
    configured_tensor_parallel_size: str | None,
    configured_gpu_memory_utilization: str | None,
    configured_max_model_len: str | None,
    configured_strict_protected_args: str | None,
    configured_download_dir: str | None,
    configured_python_bin: str | None,
    configured_dimensions: int,
) -> Mapping[str, Any]:
    """Bind live query embeddings to the exact synthetic-fixture endpoint."""

    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except (TypeError, ValueError) as error:
        raise ComparisonAborted(
            "strict local embedding endpoint configuration is invalid"
        ) from error
    strict_endpoint = (
        type(base_url) is str
        and parsed.scheme == "http"
        and parsed.hostname == "vllm_embed"
        and port == 8000
        and parsed.path.rstrip("/") == "/v1"
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )
    if not strict_endpoint:
        raise ComparisonAborted(
            "strict local embedding endpoint configuration is invalid"
        )
    if api_key != "EMPTY" or configured_sidecar_api_key != "EMPTY":
        raise ComparisonAborted("strict local embedding API key must be EMPTY")
    if not is_safe_huggingface_repo_id(configured_model):
        raise ComparisonAborted(
            "strict local embedding model must be a Hugging Face repository ID"
        )
    if (
        type(configured_checkpoint) is not str
        or re.fullmatch(r"[0-9a-f]{40}", configured_checkpoint) is None
    ):
        raise ComparisonAborted(
            "strict local embedding revision must be an immutable commit"
        )
    if (
        type(binding) is not FixtureEmbeddingBinding
        or type(configured_dimensions) is not int
        or configured_dimensions != 1024
        or binding.model != configured_model
        or binding.checkpoint != configured_checkpoint
        or binding.dimensions != configured_dimensions
        or binding.input_type != "search_document"
    ):
        raise ComparisonAborted(
            "fixture embedding contract differs from strict local endpoint"
        )
    expected_signature = embedding_endpoint_signature(
        model=configured_model,
        checkpoint=configured_checkpoint,
        dimensions=configured_dimensions,
        input_type="search_document",
    )
    if binding.endpoint_signature != expected_signature:
        raise ComparisonAborted(
            "fixture embedding signature differs from strict local endpoint"
        )
    if (
        type(configured_tokenizer_checkpoint) is not str
        or configured_tokenizer_checkpoint != configured_checkpoint
    ):
        raise ComparisonAborted(
            "strict local embedding tokenizer revision differs from model revision"
        )
    if (
        type(configured_code_checkpoint) is not str
        or configured_code_checkpoint != configured_checkpoint
    ):
        raise ComparisonAborted(
            "strict local embedding code revision differs from model revision"
        )
    if configured_extra_args != _STRICT_EMBED_EXTRA_ARGS:
        raise ComparisonAborted(
            "strict local embedding extra arguments are not canonical"
        )
    if configured_trust_remote_code != "1":
        raise ComparisonAborted(
            "strict local embedding remote code configuration is invalid"
        )
    if configured_runner != "pooling":
        raise ComparisonAborted("strict local embedding runner is invalid")
    if configured_dtype != "float16":
        raise ComparisonAborted("strict local embedding dtype is invalid")
    if configured_tensor_parallel_size != "1":
        raise ComparisonAborted(
            "strict local embedding tensor parallel size is invalid"
        )
    if configured_gpu_memory_utilization != "0.12":
        raise ComparisonAborted(
            "strict local embedding GPU memory utilization is invalid"
        )
    if configured_max_model_len != "2048":
        raise ComparisonAborted(
            "strict local embedding maximum model length is invalid"
        )
    if configured_strict_protected_args != "1":
        raise ComparisonAborted(
            "strict local embedding protected-argument fence is invalid"
        )
    if configured_download_dir != "/root/.cache/huggingface/hub":
        raise ComparisonAborted("strict local embedding download directory is invalid")
    if configured_python_bin != "python3":
        raise ComparisonAborted("strict local embedding python binary is invalid")
    payload: dict[str, Any] = {
        "model": binding.model,
        "checkpoint": binding.checkpoint,
        "tokenizer_checkpoint": configured_tokenizer_checkpoint,
        "code_checkpoint": configured_code_checkpoint,
        "dimensions": binding.dimensions,
        "input_type": binding.input_type,
        "endpoint_signature": binding.endpoint_signature,
        "extra_args_signature": sha256(
            _STRICT_EMBED_EXTRA_ARGS.encode("utf-8")
        ).hexdigest(),
        "trust_remote_code": True,
        "runner": configured_runner,
        "dtype": configured_dtype,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.12,
        "max_model_len": 2048,
        "strict_protected_args": True,
        "api_key_signature": sha256(b"EMPTY").hexdigest(),
        "download_dir": configured_download_dir,
        "python_bin": configured_python_bin,
    }
    payload["config_signature"] = comparison_snapshot_signature(payload)
    return MappingProxyType(payload)


def _validate_live_reranker_contract(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    configured_model: str,
    loaded_model: str,
    configured_tokenizer: str,
    configured_checkpoint: str,
    configured_tokenizer_checkpoint: str,
    configured_code_checkpoint: str,
    configured_extra_args: str,
    configured_trust_remote_code: str,
    configured_runner: str,
    configured_task: str,
    configured_dtype: str,
    configured_tensor_parallel_size: str,
    configured_gpu_memory_utilization: str,
    configured_max_model_len: str,
    configured_strict_protected_args: str,
    configured_download_dir: str,
    configured_python_bin: str,
    configured_cache_enabled: bool,
    timeout_seconds: int,
    document_char_limit: int,
    multimodal: bool,
) -> Mapping[str, Any]:
    """Attest the only fail-closed reranker configuration allowed for evals."""

    if provider != "local":
        raise ComparisonAborted("strict eval reranker provider must be local")
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except (TypeError, ValueError) as error:
        raise ComparisonAborted("strict local reranker endpoint is invalid") from error
    if not (
        type(base_url) is str
        and base_url == "http://vllm_rerank:8000/v1"
        and parsed.scheme == "http"
        and parsed.hostname == "vllm_rerank"
        and port == 8000
        and parsed.path.rstrip("/") == "/v1"
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    ):
        raise ComparisonAborted("strict local reranker endpoint is invalid")
    if api_key != "EMPTY":
        raise ComparisonAborted("strict local reranker API key must be EMPTY")
    if not is_safe_huggingface_repo_id(configured_model):
        raise ComparisonAborted("strict local reranker model is invalid")
    if loaded_model != configured_model:
        raise ComparisonAborted(
            "strict local reranker loaded model differs from its served model"
        )
    if configured_tokenizer != configured_model:
        raise ComparisonAborted(
            "strict local reranker tokenizer differs from its served model"
        )
    if (
        type(configured_checkpoint) is not str
        or re.fullmatch(r"[0-9a-f]{40}", configured_checkpoint) is None
    ):
        raise ComparisonAborted(
            "strict local reranker revision must be an immutable commit"
        )
    if configured_tokenizer_checkpoint != configured_checkpoint:
        raise ComparisonAborted(
            "strict local reranker tokenizer revision differs from model revision"
        )
    if configured_code_checkpoint != configured_checkpoint:
        raise ComparisonAborted(
            "strict local reranker code revision differs from model revision"
        )
    if configured_extra_args != _STRICT_RERANK_EXTRA_ARGS:
        raise ComparisonAborted(
            "strict local reranker extra arguments are not canonical"
        )
    if configured_trust_remote_code != "1":
        raise ComparisonAborted(
            "strict local reranker remote code configuration is invalid"
        )
    if configured_runner != "pooling":
        raise ComparisonAborted("strict local reranker runner is invalid")
    if configured_task != "score":
        raise ComparisonAborted("strict local reranker task is invalid")
    if configured_dtype != "float16":
        raise ComparisonAborted("strict local reranker dtype is invalid")
    if configured_tensor_parallel_size != "1":
        raise ComparisonAborted("strict local reranker tensor parallel size is invalid")
    if configured_gpu_memory_utilization != "0.25":
        raise ComparisonAborted(
            "strict local reranker GPU memory utilization is invalid"
        )
    if configured_max_model_len != "1024":
        raise ComparisonAborted("strict local reranker maximum model length is invalid")
    if configured_strict_protected_args != "1":
        raise ComparisonAborted(
            "strict local reranker protected-argument fence is invalid"
        )
    if configured_download_dir != "/root/.cache/huggingface/hub":
        raise ComparisonAborted("strict local reranker download directory is invalid")
    if configured_python_bin != "python3":
        raise ComparisonAborted("strict local reranker python binary is invalid")
    if configured_cache_enabled is not False:
        raise ComparisonAborted("strict local reranker cache must be disabled")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ComparisonAborted("strict local reranker timeout must be positive")
    if type(document_char_limit) is not int or document_char_limit <= 0:
        raise ComparisonAborted("strict local reranker document limit must be positive")
    if type(multimodal) is not bool or multimodal != (
        "qwen3-vl-reranker" in configured_model.lower()
    ):
        raise ComparisonAborted(
            "strict local reranker multimodal configuration is invalid"
        )
    payload: dict[str, Any] = {
        "provider": provider,
        "model": configured_model,
        "checkpoint": configured_checkpoint,
        "tokenizer_checkpoint": configured_tokenizer_checkpoint,
        "code_checkpoint": configured_code_checkpoint,
        "endpoint_signature": sha256(base_url.encode("utf-8")).hexdigest(),
        "timeout_seconds": timeout_seconds,
        "document_char_limit": document_char_limit,
        "multimodal": multimodal,
        "extra_args_signature": sha256(
            _STRICT_RERANK_EXTRA_ARGS.encode("utf-8")
        ).hexdigest(),
        "trust_remote_code": True,
        "runner": configured_runner,
        "task": configured_task,
        "dtype": configured_dtype,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.25,
        "max_model_len": 1024,
        "strict_protected_args": True,
        "api_key_signature": sha256(b"EMPTY").hexdigest(),
        "download_dir": configured_download_dir,
        "python_bin": configured_python_bin,
        "chat_template_sha256": _rerank_template_checksum(),
        "cache_enabled": False,
    }
    payload["config_signature"] = comparison_snapshot_signature(payload)
    return MappingProxyType(payload)


def _validate_live_extraction_settings(value: object) -> object:
    """Require the pinned, offline, fail-closed extractor before live eval work."""

    from lib.knowledge_graph.config import ExtractionSettings

    if type(value) is not ExtractionSettings:
        raise ComparisonAborted("strict extraction settings type is invalid")
    if value.build_enabled is not False:
        raise ComparisonAborted("strict extraction evaluation requires builds disabled")
    if value.provider != "gliner2_local":
        raise ComparisonAborted("strict extraction provider must be gliner2_local")
    if not is_safe_huggingface_repo_id(value.model_id):
        raise ComparisonAborted("strict extraction model must be a repository ID")
    if re.fullmatch(r"[0-9a-f]{40}", value.model_revision) is None:
        raise ComparisonAborted("strict extraction revision must be immutable")
    if value.local_files_only is not True or value.fail_open is not False:
        raise ComparisonAborted("strict extraction must be offline and fail closed")
    if (
        type(value.device) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}", value.device) is None
    ):
        raise ComparisonAborted("strict extraction device is invalid")
    if type(value.batch_size) is not int or not 1 <= value.batch_size <= 64:
        raise ComparisonAborted("strict extraction batch size is invalid")
    if (
        type(value.max_batch_characters) is not int
        or not 1 <= value.max_batch_characters <= 1_000_000
    ):
        raise ComparisonAborted("strict extraction character cap is invalid")
    return value


def _extraction_config_provenance(value: object) -> Mapping[str, Any]:
    value = _validate_live_extraction_settings(value)
    payload: dict[str, Any] = {
        "provider": value.provider,
        "model": value.model_id,
        "checkpoint": value.model_revision,
        "build_enabled": value.build_enabled,
        "device": value.device,
        "batch_size": value.batch_size,
        "max_batch_characters": value.max_batch_characters,
        "local_files_only": value.local_files_only,
        "fail_open": value.fail_open,
    }
    payload["config_signature"] = comparison_snapshot_signature(payload)
    return MappingProxyType(payload)


class GateVerificationError(RuntimeError):
    """Raised when measured rollout gates remain pending or fail verification."""


def _freeze(value: Any, active_ids: set[int] | None = None) -> Any:
    """Deep-freeze JSON-like fixture data without coercing YAML key types."""
    active_ids = active_ids if active_ids is not None else set()
    if isinstance(value, (Mapping, list, tuple)):
        identity = id(value)
        if identity in active_ids:
            raise FixtureValidationError("recursive fixture aliases are not supported")
        active_ids.add(identity)
        try:
            if isinstance(value, Mapping):
                if not all(isinstance(key, str) for key in value):
                    raise FixtureValidationError("mapping keys must be strings")
                return MappingProxyType(
                    {key: _freeze(value[key], active_ids) for key in sorted(value)}
                )
            return tuple(_freeze(item, active_ids) for item in value)
        finally:
            active_ids.remove(identity)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise FixtureValidationError(
        f"unsupported fixture value type {type(value).__name__}"
    )


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixtureValidationError(f"{context} must be a mapping")
    return value


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FixtureValidationError(f"{context} must be a non-empty string")
    return value


def _require_sequence(
    value: Any, context: str, *, nonempty: bool = False
) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FixtureValidationError(f"{context} must be a list")
    if nonempty and not value:
        raise FixtureValidationError(f"{context} must not be empty")
    return value


def _require_fields(
    record: Mapping[str, Any], fields: Sequence[str], context: str
) -> None:
    for field in fields:
        if field not in record:
            raise FixtureValidationError(f"{context} missing required field {field!r}")


def _validate_documents(value: Any, context: str) -> dict[str, tuple[str, str]]:
    """Return validated ``chunk_id -> (text, collection_id)`` evidence."""
    document_ids: set[str] = set()
    chunks: dict[str, tuple[str, str]] = {}
    for document_index, document in enumerate(
        _require_sequence(value, f"{context}.documents", nonempty=True)
    ):
        document_context = f"{context}.documents[{document_index}]"
        document = _require_mapping(document, document_context)
        _require_fields(
            document, ("doc_id", "collection_id", "chunks"), document_context
        )
        document_id = _require_nonempty_string(
            document["doc_id"], f"{document_context}.doc_id"
        )
        if document_id in document_ids:
            raise FixtureValidationError(
                f"{document_context} has duplicate document id {document_id!r}"
            )
        document_ids.add(document_id)
        collection_id = _require_nonempty_string(
            document["collection_id"], f"{document_context}.collection_id"
        )
        for chunk_index, chunk in enumerate(
            _require_sequence(
                document["chunks"], f"{document_context}.chunks", nonempty=True
            )
        ):
            chunk_context = f"{document_context}.chunks[{chunk_index}]"
            chunk = _require_mapping(chunk, chunk_context)
            _require_fields(chunk, ("chunk_id", "text"), chunk_context)
            chunk_id = _require_nonempty_string(
                chunk["chunk_id"], f"{chunk_context}.chunk_id"
            )
            text = _require_nonempty_string(chunk["text"], f"{chunk_context}.text")
            if chunk_id in chunks:
                raise FixtureValidationError(
                    f"{chunk_context} has duplicate chunk id {chunk_id!r}"
                )
            chunks[chunk_id] = (text, collection_id)
    return chunks


def _validate_expected(
    value: Any, context: str, chunks: Mapping[str, tuple[str, str]]
) -> None:
    expected = _require_mapping(value, f"{context}.expected")
    for field in ("entities", "relations", "auto_links", "suppressed_evidence"):
        if field not in expected:
            raise FixtureValidationError(
                f"{context}.expected missing required field {field!r}"
            )
        _require_sequence(expected[field], f"{context}.expected.{field}")

    entity_ids: set[str] = set()
    for index, raw_entity in enumerate(expected["entities"]):
        entity_context = f"{context}.expected.entities[{index}]"
        entity = _require_mapping(raw_entity, entity_context)
        _require_fields(
            entity,
            ("id", "text", "type", "chunk_id", "start", "end"),
            entity_context,
        )
        entity_id = _require_nonempty_string(entity["id"], f"{entity_context}.id")
        text = _require_nonempty_string(entity["text"], f"{entity_context}.text")
        _require_nonempty_string(entity["type"], f"{entity_context}.type")
        chunk_id = _require_nonempty_string(
            entity["chunk_id"], f"{entity_context}.chunk_id"
        )
        start = entity["start"]
        end = entity["end"]
        if type(start) is not int:
            raise FixtureValidationError(f"{entity_context}.start must be an integer")
        if type(end) is not int:
            raise FixtureValidationError(f"{entity_context}.end must be an integer")
        if start < 0:
            raise FixtureValidationError(f"{entity_context}.start must be non-negative")
        if end <= start:
            raise FixtureValidationError(
                f"{entity_context}.end must be greater than start"
            )
        if entity_id in entity_ids:
            raise FixtureValidationError(
                f"{entity_context} has duplicate entity id {entity_id!r}"
            )
        if chunk_id not in chunks:
            raise FixtureValidationError(
                f"{entity_context} references unknown chunk {chunk_id!r}"
            )
        chunk_text = chunks[chunk_id][0]
        if start >= len(chunk_text):
            raise FixtureValidationError(
                f"{entity_context}.start is outside referenced chunk {chunk_id!r}"
            )
        if end > len(chunk_text):
            raise FixtureValidationError(
                f"{entity_context}.end is outside referenced chunk {chunk_id!r}"
            )
        if chunk_text[start:end] != text:
            raise FixtureValidationError(
                f"{entity_context} span does not exactly match text"
            )
        entity_ids.add(entity_id)

    for field in ("relations", "auto_links"):
        seen: set[tuple[str, str, str]] = set()
        for index, raw_record in enumerate(expected[field]):
            record_context = f"{context}.expected.{field}[{index}]"
            record = _require_mapping(raw_record, record_context)
            _require_fields(record, ("source", "target", "type"), record_context)
            source = _require_nonempty_string(
                record["source"], f"{record_context}.source"
            )
            target = _require_nonempty_string(
                record["target"], f"{record_context}.target"
            )
            relation_type = _require_nonempty_string(
                record["type"], f"{record_context}.type"
            )
            if source not in entity_ids or target not in entity_ids:
                raise FixtureValidationError(
                    f"{record_context} references unknown entity"
                )
            identity = (source, target, relation_type)
            if identity in seen:
                raise FixtureValidationError(
                    f"{record_context} has duplicate {field[:-1]} record"
                )
            seen.add(identity)

    seen_suppression: set[tuple[str, str, str, str]] = set()
    for index, raw_evidence in enumerate(expected["suppressed_evidence"]):
        evidence_context = f"{context}.expected.suppressed_evidence[{index}]"
        evidence = _require_mapping(raw_evidence, evidence_context)
        _require_fields(
            evidence, ("entity", "type", "chunk_id", "reason"), evidence_context
        )
        entity = _require_nonempty_string(
            evidence["entity"], f"{evidence_context}.entity"
        )
        evidence_type = _require_nonempty_string(
            evidence["type"], f"{evidence_context}.type"
        )
        chunk_id = _require_nonempty_string(
            evidence["chunk_id"], f"{evidence_context}.chunk_id"
        )
        reason = _require_nonempty_string(
            evidence["reason"], f"{evidence_context}.reason"
        )
        if chunk_id not in chunks:
            raise FixtureValidationError(
                f"{evidence_context} references unknown chunk {chunk_id!r}"
            )
        if entity.lower() not in chunks[chunk_id][0].lower():
            raise FixtureValidationError(
                f"{evidence_context}.entity is not anchored in chunk {chunk_id!r}"
            )
        identity = (entity, evidence_type, chunk_id, reason)
        if identity in seen_suppression:
            raise FixtureValidationError(
                f"{evidence_context} has duplicate suppression record"
            )
        seen_suppression.add(identity)


def _validate_extraction_case(case: Mapping[str, Any], index: int) -> None:
    context = f"cases[{index}]"
    for field in ("id", "description", "privacy_intent", "documents", "expected"):
        if field not in case:
            raise FixtureValidationError(f"{context} missing required field {field!r}")
    _require_nonempty_string(case["id"], f"{context}.id")
    _require_nonempty_string(case["description"], f"{context}.description")
    _require_nonempty_string(case["privacy_intent"], f"{context}.privacy_intent")
    chunks = _validate_documents(case["documents"], context)
    _validate_expected(case["expected"], context, chunks)


def _validate_retrieval_case(case: Mapping[str, Any], index: int) -> None:
    context = f"cases[{index}]"
    _require_fields(
        case,
        (
            "id",
            "description",
            "privacy_intent",
            "query",
            "accessible_collection_ids",
            "documents",
            "expected_retrieval_chunk_ids",
        ),
        context,
    )
    for field in ("id", "description", "privacy_intent", "query"):
        _require_nonempty_string(case[field], f"{context}.{field}")
    accessible_collections: set[str] = set()
    for item_index, item in enumerate(
        _require_sequence(
            case["accessible_collection_ids"],
            f"{context}.accessible_collection_ids",
            nonempty=True,
        )
    ):
        collection_id = _require_nonempty_string(
            item, f"{context}.accessible_collection_ids[{item_index}]"
        )
        if collection_id in accessible_collections:
            raise FixtureValidationError(
                f"{context} has duplicate accessible collection id {collection_id!r}"
            )
        accessible_collections.add(collection_id)
    chunks = _validate_documents(case["documents"], context)
    expected_ids: set[str] = set()
    for item_index, item in enumerate(
        _require_sequence(
            case["expected_retrieval_chunk_ids"],
            f"{context}.expected_retrieval_chunk_ids",
            nonempty=False,
        )
    ):
        chunk_id = _require_nonempty_string(
            item, f"{context}.expected_retrieval_chunk_ids[{item_index}]"
        )
        if chunk_id not in chunks:
            raise FixtureValidationError(
                f"{context} references unknown chunk {chunk_id!r}"
            )
        if chunks[chunk_id][1] not in accessible_collections:
            raise FixtureValidationError(
                f"{context} retrieval evidence is not in an accessible collection"
            )
        if chunk_id in expected_ids:
            raise FixtureValidationError(
                f"{context} has duplicate retrieval chunk id {chunk_id!r}"
            )
        expected_ids.add(chunk_id)
    if "baseline_vector_result_ids" in case:
        baseline_ids: set[str | int] = set()
        for item_index, item in enumerate(
            _require_sequence(
                case["baseline_vector_result_ids"],
                f"{context}.baseline_vector_result_ids",
            )
        ):
            item_context = f"{context}.baseline_vector_result_ids[{item_index}]"
            if isinstance(item, str):
                result_id: str | int = _require_nonempty_string(item, item_context)
            elif type(item) is int and item > 0:
                result_id = item
            else:
                raise FixtureValidationError(
                    f"{item_context} must be a positive integer or non-empty string"
                )
            if result_id in baseline_ids:
                raise FixtureValidationError(
                    f"{context} has duplicate baseline result id {result_id!r}"
                )
            baseline_ids.add(result_id)
    if "canonical_identity_links" in case:
        seen_links: set[tuple[str, str, str]] = set()
        for link_index, raw_link in enumerate(
            _require_sequence(
                case["canonical_identity_links"],
                f"{context}.canonical_identity_links",
            )
        ):
            link_context = f"{context}.canonical_identity_links[{link_index}]"
            link = _require_mapping(raw_link, link_context)
            _require_fields(
                link, ("source_chunk_id", "target_chunk_id", "type"), link_context
            )
            source = _require_nonempty_string(
                link["source_chunk_id"], f"{link_context}.source_chunk_id"
            )
            target = _require_nonempty_string(
                link["target_chunk_id"], f"{link_context}.target_chunk_id"
            )
            link_type = _require_nonempty_string(link["type"], f"{link_context}.type")
            if source not in chunks or target not in chunks:
                raise FixtureValidationError(f"{link_context} references unknown chunk")
            identity = (source, target, link_type)
            if identity in seen_links:
                raise FixtureValidationError(
                    f"{link_context} has duplicate canonical identity link"
                )
            seen_links.add(identity)
    if "quality_tags" in case:
        seen_tags: set[str] = set()
        for tag_index, raw_tag in enumerate(
            _require_sequence(case["quality_tags"], f"{context}.quality_tags")
        ):
            tag = _require_nonempty_string(
                raw_tag,
                f"{context}.quality_tags[{tag_index}]",
            )
            if tag in seen_tags:
                raise FixtureValidationError(
                    f"{context} has duplicate quality tag {tag!r}"
                )
            seen_tags.add(tag)
    if "expected_min_semantic_distance" in case:
        raw_distances = _require_mapping(
            case["expected_min_semantic_distance"],
            f"{context}.expected_min_semantic_distance",
        )
        for chunk_id, distance in raw_distances.items():
            if chunk_id not in expected_ids:
                raise FixtureValidationError(
                    f"{context}.expected_min_semantic_distance references "
                    f"non-gold chunk {chunk_id!r}"
                )
            if type(distance) is not int or not 0 <= distance <= 2:
                raise FixtureValidationError(
                    f"{context}.expected_min_semantic_distance values must be "
                    "exact integers in [0, 2]"
                )


def _load_cases(path: Path, validator: Any) -> tuple[Mapping[str, Any], ...]:
    try:
        with path.open(encoding="utf-8") as fixture_file:
            payload = yaml.safe_load(fixture_file)
    except (OSError, yaml.YAMLError) as exc:
        raise FixtureValidationError(f"could not read fixture {path}: {exc}") from exc
    payload = _require_mapping(_freeze(payload), "top level")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise FixtureValidationError("top level schema_version must be integer 1")
    if "cases" not in payload:
        raise FixtureValidationError("top level missing required field 'cases'")
    cases = _require_sequence(payload["cases"], "top level cases", nonempty=True)
    normalized: list[Mapping[str, Any]] = []
    case_ids: set[str] = set()
    for index, raw_case in enumerate(cases):
        case = _require_mapping(raw_case, f"cases[{index}]")
        case_id = case.get("id")
        _require_nonempty_string(case_id, f"cases[{index}].id")
        if case_id in case_ids:
            raise FixtureValidationError(f"duplicate case id {case_id!r}")
        case_ids.add(case_id)
        validator(case, index)
        normalized.append(case)
    return tuple(normalized)


def load_extraction_cases(path: Path | None = None) -> tuple[Mapping[str, Any], ...]:
    """Load validated, deep-immutable extraction fixtures without external services."""
    return _load_cases(
        path or _DEFAULT_EXTRACTION_CASES_PATH, _validate_extraction_case
    )


def load_retrieval_cases(path: Path | None = None) -> tuple[Mapping[str, Any], ...]:
    """Load validated, deep-immutable retrieval fixtures without external services."""
    return _load_cases(path or _DEFAULT_RETRIEVAL_CASES_PATH, _validate_retrieval_case)


def _structural_set(records: Any) -> set[str]:
    if records is None:
        return set()
    return {
        json.dumps(_thaw(record), sort_keys=True, separators=(",", ":"))
        for record in records
    }


def _entity_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Prediction entity contract: type, text, chunk_id, start, and end.

    IDs and confidence are intentionally ignored; ``id`` is only a legacy fallback
    for old hand-written unit predictions that omit all semantic fields.
    """
    if all(key in record for key in ("type", "text", "chunk_id")):
        return (
            record["type"],
            str(record["text"]).casefold(),
            record["chunk_id"],
            record.get("start"),
            record.get("end"),
        )
    return ("legacy-id", record.get("id"))


def _entity_set(records: Any) -> set[tuple[Any, ...]]:
    return {
        _entity_key(_require_mapping(record, "prediction entity"))
        for record in records or ()
    }


def _relation_set(records: Any, entities: Any) -> set[tuple[Any, ...]]:
    by_id = {
        record.get("id"): _entity_key(_require_mapping(record, "prediction entity"))
        for record in entities or ()
    }
    result = set()
    for record in records or ():
        record = _require_mapping(record, "prediction relation")
        result.add(
            (
                record.get("type"),
                by_id.get(record.get("source"), record.get("source")),
                by_id.get(record.get("target"), record.get("target")),
            )
        )
    return result


def _relation_endpoint_set(records: Any, entities: Any) -> set[tuple[Any, ...]]:
    """Return typed, direction-neutral endpoint pairs for endpoint scoring."""
    by_id = {
        record.get("id"): _entity_key(_require_mapping(record, "prediction entity"))
        for record in entities or ()
    }
    result: set[tuple[Any, ...]] = set()
    for raw_record in records or ():
        record = _require_mapping(raw_record, "prediction relation")
        source = by_id.get(record.get("source"), record.get("source"))
        target = by_id.get(record.get("target"), record.get("target"))
        result.add(
            (
                record.get("type"),
                tuple(sorted((source, target), key=repr)),
            )
        )
    return result


def _precision(gold: AbstractSet[Hashable], predicted: AbstractSet[Hashable]) -> float:
    return 1.0 if not predicted else len(gold & predicted) / len(predicted)


def _recall(gold: AbstractSet[Hashable], predicted: AbstractSet[Hashable]) -> float:
    return 1.0 if not gold else len(gold & predicted) / len(gold)


def score_extraction(
    case: Mapping[str, Any], predictions: Mapping[str, Any]
) -> Mapping[str, float]:
    """Return set-based offline extraction metrics for injected prediction records."""
    expected = _require_mapping(case.get("expected"), "case.expected")
    report: dict[str, float] = {}
    gold_entities = _entity_set(expected.get("entities"))
    predicted_entities = _entity_set(predictions.get("entities"))
    mention_precision = _precision(gold_entities, predicted_entities)
    mention_recall = _recall(gold_entities, predicted_entities)
    report["entity_precision"] = mention_precision
    report["entity_recall"] = mention_recall
    report["mention_span_precision"] = mention_precision
    report["mention_span_recall"] = mention_recall
    gold_relations = _relation_set(expected.get("relations"), expected.get("entities"))
    predicted_relations = _relation_set(
        predictions.get("relations"), predictions.get("entities")
    )
    direction_precision = _precision(gold_relations, predicted_relations)
    direction_recall = _recall(gold_relations, predicted_relations)
    report["relation_precision"] = direction_precision
    report["relation_recall"] = direction_recall
    report["relation_direction_precision"] = direction_precision
    report["relation_direction_recall"] = direction_recall
    gold_endpoints = _relation_endpoint_set(
        expected.get("relations"), expected.get("entities")
    )
    predicted_endpoints = _relation_endpoint_set(
        predictions.get("relations"), predictions.get("entities")
    )
    report["relation_endpoint_precision"] = _precision(
        gold_endpoints, predicted_endpoints
    )
    report["relation_endpoint_recall"] = _recall(gold_endpoints, predicted_endpoints)
    gold_links = _relation_set(expected.get("auto_links"), expected.get("entities"))
    predicted_links = _relation_set(
        predictions.get("auto_links"), predictions.get("entities")
    )
    automatic_precision = _precision(gold_links, predicted_links)
    automatic_recall = _recall(gold_links, predicted_links)
    report["auto_link_precision"] = automatic_precision
    report["automatic_link_precision"] = automatic_precision
    report["automatic_link_recall"] = automatic_recall
    predicted_candidates = _relation_set(
        predictions.get("candidate_links", predictions.get("auto_links")),
        predictions.get("entities"),
    )
    report["candidate_link_precision"] = _precision(gold_links, predicted_candidates)
    report["candidate_link_recall"] = _recall(gold_links, predicted_candidates)
    gold_suppression = _structural_set(expected.get("suppressed_evidence"))
    predicted_suppression = _structural_set(predictions.get("suppressed_evidence"))
    report["suppression_precision"] = _precision(
        gold_suppression,
        predicted_suppression,
    )
    report["suppression_recall"] = _recall(
        gold_suppression,
        predicted_suppression,
    )
    return MappingProxyType(report)


def evaluate_live_extraction_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    predict_case: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Score only predictions returned by one live, injectable projection."""

    if not callable(predict_case):
        raise ComparisonValidationError("live extraction predictor must be callable")
    reports: dict[str, Mapping[str, float]] = {}
    for case in sorted(cases, key=lambda value: str(value.get("id", ""))):
        case_id = case.get("id")
        if type(case_id) is not str or not case_id:
            raise ComparisonValidationError("live extraction case requires an ID")
        prediction = predict_case(case)
        if not isinstance(prediction, Mapping):
            raise ComparisonAborted("live extraction predictor returned no projection")
        reports[case_id] = score_extraction(case, prediction)
    metric_names = tuple(
        sorted({name for report in reports.values() for name in report})
    )
    metrics = {
        name: _mean([float(report[name]) for report in reports.values()])
        for name in metric_names
    }
    return MappingProxyType(
        {
            "cases": MappingProxyType(dict(sorted(reports.items()))),
            "metrics": MappingProxyType(metrics),
        }
    )


def _production_extraction_prediction(
    case: Mapping[str, Any],
    *,
    projection: Callable[..., Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Lazily expose the exact pure production inference/decision kernels."""

    from apps.knowledge_graph.extraction.pipeline import collect_document_evidence
    from apps.knowledge_graph.graph.filtering import decide_entity_filter
    from apps.knowledge_graph.resolution.collection import resolve_collection_entities
    from apps.knowledge_graph.resolution.coreference import resolve_document_mentions

    if not callable(projection):
        raise ComparisonAborted("live extraction projection must be callable")
    return projection(
        case,
        collect_document_evidence=collect_document_evidence,
        resolve_document_mentions=resolve_document_mentions,
        resolve_collection_entities=resolve_collection_entities,
        decide_entity_filter=decide_entity_filter,
    )


def evaluate_production_extraction_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    projection_context: object,
) -> Mapping[str, Any]:
    """Run every curated case through the lazy, read-only production kernels."""

    def predict_case(case: Mapping[str, Any]) -> Mapping[str, Any]:
        return _production_extraction_prediction(
            case,
            projection=lambda value, **kernels: _project_production_extraction_case(
                value,
                context=projection_context,
                **kernels,
            ),
        )

    return evaluate_live_extraction_cases(cases, predict_case=predict_case)


def _deduplicate_prediction_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        thawed = dict(record)
        key = json.dumps(
            thawed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        by_key[key] = thawed
    return [by_key[key] for key in sorted(by_key)]


def _project_production_extraction_case(
    case: Mapping[str, Any],
    *,
    context: object,
    collect_document_evidence: Callable[..., object],
    resolve_document_mentions: Callable[..., object],
    resolve_collection_entities: Callable[..., object],
    decide_entity_filter: Callable[..., object],
) -> Mapping[str, Any]:
    """Project one model result through the exact shipping resolution policies."""

    from apps.knowledge_graph.extraction.pipeline import _build_backend
    from apps.knowledge_graph.extraction.windows import ExtractionWindow
    from apps.knowledge_graph.graph.filtering import (
        EntityFilterInput,
        FilterStatus,
        PositionKind,
    )
    from apps.knowledge_graph.resolution.collection import (
        AliasEvidence,
        DocumentEntityInput,
        SupportedRelation,
        default_collection_embedding_session,
    )
    from apps.knowledge_graph.resolution.coreference import DocumentMention
    from apps.knowledge_graph.resolution.scoring import ResolutionOutcome

    if not isinstance(context, Mapping):
        raise ComparisonAborted("live extraction projection context is malformed")
    required_context = {
        "canonical_assertions",
        "collection_snapshots",
        "document_artifact_ids",
        "embedding_signatures",
        "extraction_settings",
        "filter_policy",
        "manifest",
        "ontology",
        "resolution_config",
    }
    if set(context) != required_context:
        raise ComparisonAborted("live extraction projection context is not exact")
    manifest = context["manifest"]
    if type(manifest) is not ResolvedFixtureManifest:
        raise ComparisonAborted("live extraction fixture manifest changed")
    ontology = context["ontology"]
    extraction_settings = context["extraction_settings"]
    collection_snapshots = context["collection_snapshots"]
    document_artifact_ids = context["document_artifact_ids"]
    embedding_signatures = context["embedding_signatures"]
    filter_policy = context["filter_policy"]
    resolution_config = context["resolution_config"]
    canonical_assertions = context["canonical_assertions"]
    if (
        not all(
            isinstance(value, Mapping)
            for value in (
                collection_snapshots,
                document_artifact_ids,
                embedding_signatures,
            )
        )
        or type(canonical_assertions) is not tuple
    ):
        raise ComparisonAborted("live extraction projection bindings changed")

    case_id = case.get("id")
    documents = case.get("documents")
    if type(case_id) is not str or not case_id or not isinstance(documents, Sequence):
        raise ComparisonAborted("live extraction case topology is malformed")
    backend = _build_backend(extraction_settings)
    chunk_symbol_by_id = {
        binding.chunk_id: symbol for symbol, binding in manifest.chunks.items()
    }
    predicted_entities: list[dict[str, Any]] = []
    predicted_relations: list[dict[str, Any]] = []
    automatic_links: list[dict[str, Any]] = []
    candidate_links: list[dict[str, Any]] = []
    entity_record_by_id: dict[str, dict[str, Any]] = {}
    occurrence_ids_by_identity: dict[tuple[object, ...], tuple[str, ...]] = {}
    occurrence_by_local_span: dict[tuple[tuple[object, ...], int, int, int], str] = {}
    mention_identity_by_id: dict[str, tuple[object, ...]] = {}
    mention_number_by_id: dict[str, int] = {}
    document_results: list[tuple[object, object]] = []
    relation_inputs: list[tuple[object, str, object, float]] = []
    next_occurrence = 1
    next_mention = 1

    for raw_document in documents:
        document = _require_mapping(raw_document, "live extraction document")
        document_symbol = _require_nonempty_string(
            document.get("doc_id"), "live extraction document ID"
        )
        try:
            document_binding = manifest.documents[document_symbol]
        except KeyError as error:
            raise ComparisonAborted(
                "live extraction document escaped fixture manifest"
            ) from error
        raw_chunks = _require_sequence(
            document.get("chunks"),
            f"live extraction document {document_symbol!r} chunks",
            nonempty=True,
        )
        chunk_texts = tuple(
            _require_nonempty_string(
                _require_mapping(raw_chunk, "live extraction chunk").get("text"),
                "live extraction chunk text",
            )
            for raw_chunk in raw_chunks
        )
        full_text, spans = assemble_fixture_document(chunk_texts)
        windows: list[object] = []
        chunk_text_by_id: dict[int, str] = {}
        for raw_chunk, text, span in zip(
            raw_chunks,
            chunk_texts,
            spans,
            strict=True,
        ):
            chunk = _require_mapping(raw_chunk, "live extraction chunk")
            chunk_symbol = _require_nonempty_string(
                chunk.get("chunk_id"), "live extraction chunk ID"
            )
            try:
                binding = manifest.chunks[chunk_symbol]
            except KeyError as error:
                raise ComparisonAborted(
                    "live extraction chunk escaped fixture manifest"
                ) from error
            if (
                binding.document_symbol != document_symbol
                or (binding.start, binding.end) != span
            ):
                raise ComparisonAborted("live extraction chunk span changed")
            chunk_text_by_id[binding.chunk_id] = text
            windows.append(
                ExtractionWindow(
                    chunk_id=binding.chunk_id,
                    document_id=document_binding.document_id,
                    content=text,
                    start_position=binding.start,
                    modality="text",
                )
            )
        evidence = collect_document_evidence(
            tuple(windows),
            full_text=full_text,
            backend=backend,
            ontology=ontology,
            max_batch_count=extraction_settings.batch_size,
            max_batch_characters=extraction_settings.max_batch_characters,
        )
        document_mentions: list[object] = []
        for entity in evidence.entities:
            mention_id = f"m{next_mention}"
            mention_number_by_id[mention_id] = next_mention
            next_mention += 1
            mention_identity_by_id[mention_id] = entity.identity_key
            occurrence_ids: list[str] = []
            for observation in entity.observations:
                try:
                    chunk_symbol = chunk_symbol_by_id[observation.chunk_id]
                    chunk_text = chunk_text_by_id[observation.chunk_id]
                except KeyError as error:
                    raise ComparisonAborted(
                        "extractor returned evidence outside the fixture case"
                    ) from error
                occurrence_id = f"o{next_occurrence}"
                next_occurrence += 1
                occurrence = {
                    "id": occurrence_id,
                    "text": chunk_text[observation.local_start : observation.local_end],
                    "type": entity.entity_type,
                    "chunk_id": chunk_symbol,
                    "start": observation.local_start,
                    "end": observation.local_end,
                }
                predicted_entities.append(occurrence)
                entity_record_by_id[occurrence_id] = occurrence
                occurrence_ids.append(occurrence_id)
                occurrence_by_local_span[
                    (
                        entity.identity_key,
                        observation.chunk_id,
                        observation.local_start,
                        observation.local_end,
                    )
                ] = occurrence_id
            occurrence_ids_by_identity[entity.identity_key] = tuple(occurrence_ids)
            for left, right in zip(occurrence_ids, occurrence_ids[1:]):
                automatic_links.append(
                    {
                        "source": left,
                        "target": right,
                        "type": "canonical_identity",
                    }
                )
            document_mentions.append(
                DocumentMention(
                    mention_id=mention_id,
                    raw_text=entity.raw_text,
                    entity_type=entity.entity_type,
                    start=entity.start,
                    end=entity.end,
                    source_text=full_text,
                    source_offset=0,
                    confidence=entity.confidence,
                    document_id=document_binding.document_id,
                    source_key=f"{case_id}:{document_symbol}",
                    chunk_id=entity.chunk_id,
                    position_basis=entity.position_basis,
                    content_object_id=entity.content_object_id,
                )
            )
        for relation in evidence.relations:
            relation_inputs.append(
                (
                    relation.head_identity,
                    relation.relation_type,
                    relation.tail_identity,
                    relation.confidence,
                )
            )
            for observation in relation.observations:
                head = occurrence_by_local_span.get(
                    (
                        relation.head_identity,
                        observation.chunk_id,
                        observation.head_local_start,
                        observation.head_local_end,
                    )
                )
                tail = occurrence_by_local_span.get(
                    (
                        relation.tail_identity,
                        observation.chunk_id,
                        observation.tail_local_start,
                        observation.tail_local_end,
                    )
                )
                if head is None or tail is None:
                    raise ComparisonAborted("extractor relation lost a mapped endpoint")
                predicted_relations.append(
                    {
                        "source": head,
                        "target": tail,
                        "type": relation.relation_type,
                    }
                )
        document_result = resolve_document_mentions(tuple(document_mentions), ontology)
        document_results.append((document_binding, document_result))
        for cluster in document_result.clusters:
            representative_by_mention = {
                mention_id: occurrence_ids_by_identity[
                    mention_identity_by_id[mention_id]
                ][0]
                for mention_id in cluster.mention_ids
            }
            for membership in cluster.memberships:
                if membership.parent_mention_id is None:
                    continue
                automatic_links.append(
                    {
                        "source": representative_by_mention[
                            membership.parent_mention_id
                        ],
                        "target": representative_by_mention[membership.mention_id],
                        "type": "canonical_identity",
                    }
                )

    document_entities: list[object] = []
    document_entity_occurrences: dict[int, tuple[str, ...]] = {}
    document_entity_by_mention: dict[str, int] = {}
    document_entity_collection: dict[int, int] = {}
    next_entity_id = 1
    alias_methods = {
        "stable_identifier",
        "defined_acronym",
        "ontology_alias",
        "normalized_name",
    }
    for document_binding, result in document_results:
        try:
            document_artifact_id = document_artifact_ids[document_binding.document_id]
        except KeyError as error:
            raise ComparisonAborted(
                "live extraction document artifact binding is incomplete"
            ) from error
        for cluster in result.clusters:
            entity_id = next_entity_id
            next_entity_id += 1
            occurrence_ids = tuple(
                occurrence_id
                for mention_id in cluster.mention_ids
                for occurrence_id in occurrence_ids_by_identity[
                    mention_identity_by_id[mention_id]
                ]
            )
            aliases = tuple(
                AliasEvidence(
                    alias=entity_record_by_id[
                        occurrence_ids_by_identity[
                            mention_identity_by_id[membership.mention_id]
                        ][0]
                    ]["text"],
                    method=membership.method,
                    mention_id=mention_number_by_id[membership.mention_id],
                )
                for membership in cluster.memberships
                if membership.method in alias_methods
            )
            document_entities.append(
                DocumentEntityInput(
                    entity_id=entity_id,
                    document_cluster_key=cluster.cluster_key,
                    document_artifact_id=document_artifact_id,
                    document_id=document_binding.document_id,
                    label=cluster.label,
                    normalized_label=cluster.normalized_label,
                    entity_type=cluster.entity_type,
                    identifier=cluster.identifier,
                    version_signature=cluster.version_signature,
                    alias_evidence=aliases,
                    extraction_confidence=cluster.confidence,
                    document_resolution_confidence=cluster.confidence,
                )
            )
            document_entity_occurrences[entity_id] = occurrence_ids
            document_entity_collection[entity_id] = document_binding.collection_id
            for mention_id in cluster.mention_ids:
                document_entity_by_mention[mention_id] = entity_id

    supported_relations: list[object] = []
    seen_supported: set[tuple[int, str, int]] = set()
    for head_identity, relation_type, tail_identity, confidence in relation_inputs:
        head_mentions = tuple(
            mention_id
            for mention_id, identity in mention_identity_by_id.items()
            if identity == head_identity
        )
        tail_mentions = tuple(
            mention_id
            for mention_id, identity in mention_identity_by_id.items()
            if identity == tail_identity
        )
        if len(head_mentions) != 1 or len(tail_mentions) != 1:
            raise ComparisonAborted("collection relation endpoint is ambiguous")
        head_entity = document_entity_by_mention[head_mentions[0]]
        tail_entity = document_entity_by_mention[tail_mentions[0]]
        key = (head_entity, relation_type, tail_entity)
        if head_entity == tail_entity or key in seen_supported:
            continue
        seen_supported.add(key)
        supported_relations.append(
            SupportedRelation(
                relation_id=len(supported_relations) + 1,
                source_entity_id=head_entity,
                relation_type=relation_type,
                target_entity_id=tail_entity,
                confidence=confidence,
            )
        )

    active_occurrence_ids: set[str] = set()
    suppressed_evidence: list[dict[str, Any]] = []
    for collection_id in sorted(set(document_entity_collection.values())):
        collection_entities = tuple(
            entity
            for entity in document_entities
            if document_entity_collection[entity.entity_id] == collection_id
        )
        collection_relations = tuple(
            relation
            for relation in supported_relations
            if relation.source_entity_id in document_entity_collection
            and relation.target_entity_id in document_entity_collection
            and document_entity_collection[relation.source_entity_id] == collection_id
            and document_entity_collection[relation.target_entity_id] == collection_id
        )
        try:
            snapshot = collection_snapshots[collection_id]
            signature = embedding_signatures[collection_id]
        except KeyError as error:
            raise ComparisonAborted(
                "live extraction collection snapshot binding is incomplete"
            ) from error
        resolution = resolve_collection_entities(
            snapshot,
            collection_entities,
            ontology,
            relations=collection_relations,
            config=resolution_config,
            embedding_session=default_collection_embedding_session(signature),
        )
        for decision in resolution.decisions:
            link = {
                "source": document_entity_occurrences[decision.left_entity_id][0],
                "target": document_entity_occurrences[decision.right_entity_id][0],
                "type": "canonical_identity",
            }
            if decision.outcome is ResolutionOutcome.AUTOMATIC:
                automatic_links.append(link)
            elif decision.outcome is ResolutionOutcome.CANDIDATE:
                candidate_links.append(link)
        for cluster in resolution.clusters:
            occurrence_ids = tuple(
                occurrence_id
                for entity_id in cluster.document_entity_ids
                for occurrence_id in document_entity_occurrences[entity_id]
            )
            relation_participation = sum(
                relation.source_entity_id in cluster.document_entity_ids
                or relation.target_entity_id in cluster.document_entity_ids
                for relation in collection_relations
            )
            decision = decide_entity_filter(
                EntityFilterInput(
                    entity_id=cluster.cluster_key,
                    entity_type=cluster.entity_type,
                    mention_ids=occurrence_ids,
                    document_ids=tuple(
                        str(
                            manifest.documents[
                                manifest.chunks[
                                    entity_record_by_id[occurrence_id]["chunk_id"]
                                ].document_symbol
                            ].document_id
                        )
                        for occurrence_id in occurrence_ids
                    ),
                    extraction_confidence=cluster.extraction_confidence,
                    resolution_confidence=cluster.resolution_confidence,
                    promotion_confidence=cluster.promotion_confidence,
                    relation_participation=relation_participation,
                    positions=tuple(PositionKind.BODY for _value in occurrence_ids),
                ),
                ontology,
                filter_policy,
            )
            if decision.status is FilterStatus.ACTIVE:
                active_occurrence_ids.update(occurrence_ids)
            else:
                reason = decision.reason_codes[0]
                for occurrence_id in occurrence_ids:
                    entity = entity_record_by_id[occurrence_id]
                    suppressed_evidence.append(
                        {
                            "entity": entity["text"],
                            "type": entity["type"],
                            "chunk_id": entity["chunk_id"],
                            "reason": reason,
                        }
                    )

    for source_symbol, target_symbol in canonical_assertions:
        source_entities = tuple(
            row
            for row in predicted_entities
            if row["id"] in active_occurrence_ids and row["chunk_id"] == source_symbol
        )
        target_entities = tuple(
            row
            for row in predicted_entities
            if row["id"] in active_occurrence_ids and row["chunk_id"] == target_symbol
        )
        matching = tuple(
            (source, target)
            for source in source_entities
            for target in target_entities
            if source["text"].casefold() == target["text"].casefold()
            and source["type"] == target["type"]
        )
        if len(matching) == 1:
            source, target = matching[0]
            automatic_links.append(
                {
                    "source": source["id"],
                    "target": target["id"],
                    "type": "canonical_identity",
                }
            )

    def active_link(record: Mapping[str, Any]) -> bool:
        return (
            record["source"] in active_occurrence_ids
            and record["target"] in active_occurrence_ids
        )

    return MappingProxyType(
        {
            "entities": _deduplicate_prediction_records(
                row for row in predicted_entities if row["id"] in active_occurrence_ids
            ),
            "relations": _deduplicate_prediction_records(
                row for row in predicted_relations if active_link(row)
            ),
            "auto_links": _deduplicate_prediction_records(
                row for row in automatic_links if active_link(row)
            ),
            "candidate_links": _deduplicate_prediction_records(
                row for row in candidate_links if active_link(row)
            ),
            "suppressed_evidence": _deduplicate_prediction_records(suppressed_evidence),
        }
    )


def score_retrieval(
    case: Mapping[str, Any], retrieved_chunk_ids: Sequence[str]
) -> Mapping[str, float]:
    """Return recall@10 after stable de-duplication of injected retrieval output IDs."""
    gold = set(case.get("expected_retrieval_chunk_ids", ()))
    seen: set[str] = set()
    top_ten: list[str] = []
    for chunk_id in retrieved_chunk_ids:
        if chunk_id not in seen:
            seen.add(chunk_id)
            top_ten.append(chunk_id)
        if len(top_ten) == 10:
            break
    return MappingProxyType({"retrieval_recall_at_10": _recall(gold, set(top_ten))})


def _stable_unique(values: Iterable[Hashable], context: str) -> tuple[Hashable, ...]:
    result: list[Hashable] = []
    seen: set[Hashable] = set()
    try:
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonValidationError(
            f"{context} must contain hashable deterministic identifiers"
        ) from exc
    return tuple(result)


def _unit_metric(value: Any, context: str) -> float:
    if type(value) not in (int, float):
        raise ComparisonValidationError(f"{context} must be a finite number in [0, 1]")
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ComparisonValidationError(f"{context} must be a finite number in [0, 1]")
    return number


def _nonnegative_metric(value: Any, context: str) -> float:
    if type(value) not in (int, float):
        raise ComparisonValidationError(f"{context} must be finite and nonnegative")
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ComparisonValidationError(f"{context} must be finite and nonnegative")
    return number


def score_ranked_retrieval(
    *,
    expected_chunk_ids: Iterable[Hashable],
    ranked_chunk_ids: Iterable[Hashable],
    k: int = 10,
    accessible_chunk_ids: AbstractSet[Hashable],
    graph_chunk_ids: Iterable[Hashable] = (),
    citation_evidence_chunk_ids: AbstractSet[Hashable] = frozenset(),
    seed_chunk_ids: Iterable[Hashable] = (),
    mapped_seed_chunk_ids: AbstractSet[Hashable] = frozenset(),
    semantic_distances: Mapping[Hashable, int] | None = None,
    latency_ms: float = 0.0,
    node_count: int = 0,
    edge_count: int = 0,
) -> Mapping[str, float | int]:
    """Score one ordered arm without exposing its private graph trace.

    Relevance is binary.  Ranks are de-duplicated by first occurrence so a
    provider cannot inflate Recall or DCG by repeating a chunk.
    """

    if type(k) is not int or not 1 <= k <= 1_000:
        raise ComparisonValidationError("k must be an exact integer in [1, 1000]")
    if not isinstance(accessible_chunk_ids, AbstractSet):
        raise ComparisonValidationError("accessible_chunk_ids must be a set")
    if not isinstance(citation_evidence_chunk_ids, AbstractSet):
        raise ComparisonValidationError("citation_evidence_chunk_ids must be a set")
    expected = set(_stable_unique(expected_chunk_ids, "expected_chunk_ids"))
    ranked = _stable_unique(ranked_chunk_ids, "ranked_chunk_ids")
    graph = _stable_unique(graph_chunk_ids, "graph_chunk_ids")
    seeds = _stable_unique(seed_chunk_ids, "seed_chunk_ids")
    top_k = ranked[:k]

    recall_at_k = (
        1.0 if not expected else len(expected.intersection(top_k)) / len(expected)
    )
    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in expected:
            reciprocal_rank = 1.0 / rank
            break
    if not expected:
        reciprocal_rank = 1.0
    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, chunk_id in enumerate(top_k, start=1)
        if chunk_id in expected
    )
    ideal_count = min(k, len(expected))
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_count + 1))
    ndcg = 1.0 if ideal_count == 0 else dcg / ideal_dcg

    inaccessible_count = sum(
        1 for chunk_id in ranked if chunk_id not in accessible_chunk_ids
    )
    citation_coverage = (
        1.0
        if not ranked
        else sum(1 for chunk_id in ranked if chunk_id in citation_evidence_chunk_ids)
        / len(ranked)
    )
    seed_coverage = (
        1.0
        if not seeds
        else len(set(seeds).intersection(mapped_seed_chunk_ids)) / len(seeds)
    )
    distances = semantic_distances or {}
    graph_novel = tuple(chunk_id for chunk_id in graph if chunk_id not in set(seeds))
    distance_two_fraction = (
        0.0
        if not graph_novel
        else sum(1 for chunk_id in graph_novel if distances.get(chunk_id) == 2)
        / len(graph_novel)
    )
    if type(node_count) is not int or node_count < 0:
        raise ComparisonValidationError("node_count must be nonnegative")
    if type(edge_count) is not int or edge_count < 0:
        raise ComparisonValidationError("edge_count must be nonnegative")

    return MappingProxyType(
        {
            f"recall_at_{k}": recall_at_k,
            "mrr": reciprocal_rank,
            f"ndcg_at_{k}": ndcg,
            "graph_hit_rate": 1.0 if graph else 0.0,
            "inaccessible_result_count": inaccessible_count,
            "latency_ms": _nonnegative_metric(latency_ms, "latency_ms"),
            "citation_evidence_coverage": citation_coverage,
            "seed_coverage": seed_coverage,
            "node_count": node_count,
            "edge_count": edge_count,
            "distance_2_novel_fraction": distance_two_fraction,
        }
    )


def _canonical_hash_value(value: Any, active: set[int] | None = None) -> Any:
    """Convert a private snapshot value to exact, stable JSON material."""

    active = active if active is not None else set()
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ComparisonValidationError("snapshot values must be finite")
        return {"__float_hex__": value.hex()}
    if type(value) is UUID:
        return str(value)
    if isinstance(value, Enum):
        return _canonical_hash_value(value.value, active)
    if is_dataclass(value):
        identity = id(value)
        if identity in active:
            raise ComparisonValidationError("snapshot values must not be recursive")
        active.add(identity)
        try:
            return {
                field.name: _canonical_hash_value(getattr(value, field.name), active)
                for field in fields(value)
                if field.init
            }
        finally:
            active.remove(identity)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ComparisonValidationError("snapshot values must not be recursive")
        if not all(type(key) is str for key in value):
            raise ComparisonValidationError("snapshot mapping keys must be strings")
        active.add(identity)
        try:
            return {
                key: _canonical_hash_value(value[key], active) for key in sorted(value)
            }
        finally:
            active.remove(identity)
    if isinstance(value, (tuple, list)):
        identity = id(value)
        if identity in active:
            raise ComparisonValidationError("snapshot values must not be recursive")
        active.add(identity)
        try:
            return [_canonical_hash_value(item, active) for item in value]
        finally:
            active.remove(identity)
    raise ComparisonValidationError(
        f"snapshot contains unsupported {type(value).__name__} value"
    )


def comparison_snapshot_signature(payload: Mapping[str, Any]) -> str:
    """Hash the complete private one-snapshot comparison material."""

    if not isinstance(payload, Mapping):
        raise ComparisonValidationError("comparison snapshot payload must be a mapping")
    encoded = json.dumps(
        _canonical_hash_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def validate_fixture_manifest(
    manifest: Any,
    *,
    extraction_cases: Sequence[Mapping[str, Any]],
    retrieval_cases: Sequence[Mapping[str, Any]],
    collection_requests: tuple[tuple[int, UUID], ...],
    expected_fixture_checksum: str | None = None,
) -> ResolvedFixtureManifest:
    """Validate the shared provider-neutral resolved fixture contract."""

    return _validate_fixture_manifest(
        manifest,
        extraction_cases=extraction_cases,
        retrieval_cases=retrieval_cases,
        collection_requests=collection_requests,
        expected_fixture_checksum=(
            fixture_checksum()
            if expected_fixture_checksum is None
            else expected_fixture_checksum
        ),
    )


def revalidate_fixture_database_rows(
    manifest: ResolvedFixtureManifest,
    *,
    document_rows: tuple[object, ...],
    chunk_rows: tuple[object, ...],
) -> Mapping[str, Any]:
    """Revalidate exact authorized synthetic rows inside the live RR snapshot."""

    if type(manifest) is not ResolvedFixtureManifest:
        raise ComparisonAborted("resolved fixture manifest type changed")
    if type(document_rows) is not tuple or type(chunk_rows) is not tuple:
        raise ComparisonAborted("fixture database rows must be immutable tuples")
    expected_documents = {
        binding.document_id: binding
        for binding in manifest.documents.values()
        if manifest.collections[binding.collection_symbol].authorized
    }
    observed_documents: dict[UUID, object] = {}
    for row in document_rows:
        document_id = getattr(row, "id", None)
        collection_id = getattr(row, "collection_id", None)
        full_text = getattr(row, "full_text", None)
        full_text_hash = getattr(row, "full_text_hash", None)
        if (
            type(document_id) is not UUID
            or type(collection_id) is not int
            or type(full_text) is not str
            or type(full_text_hash) is not str
            or document_id in observed_documents
            or document_id not in expected_documents
        ):
            raise ComparisonAborted("fixture document materialization is ambiguous")
        expected = expected_documents[document_id]
        actual_hash = sha256(full_text.encode("utf-8")).hexdigest()
        if (
            collection_id != expected.collection_id
            or full_text_hash != expected.full_text_sha256
            or actual_hash != expected.full_text_sha256
        ):
            raise ComparisonAborted("fixture document hash/topology mismatch")
        observed_documents[document_id] = row
    if set(observed_documents) != set(expected_documents):
        raise ComparisonAborted("fixture document scope is incomplete")

    expected_chunks = {
        binding.chunk_id: binding
        for binding in manifest.chunks.values()
        if manifest.collections[binding.collection_symbol].authorized
    }
    observed_chunks: dict[int, object] = {}
    for row in chunk_rows:
        chunk_id = getattr(row, "pk", None)
        document_id = getattr(row, "doc_id", None)
        chunk_number = getattr(row, "chunk_number", None)
        start = getattr(row, "start_position", None)
        end = getattr(row, "end_position", None)
        content = getattr(row, "content", None)
        embedding = getattr(row, "embedding", None)
        if (
            type(chunk_id) is not int
            or chunk_id <= 0
            or type(document_id) is not UUID
            or type(chunk_number) is not int
            or type(start) is not int
            or type(end) is not int
            or type(content) is not str
            or chunk_id in observed_chunks
            or chunk_id not in expected_chunks
        ):
            raise ComparisonAborted("fixture chunk materialization is ambiguous")
        expected = expected_chunks[chunk_id]
        expected_document = manifest.documents[expected.document_symbol]
        document = observed_documents.get(expected_document.document_id)
        full_text = getattr(document, "full_text", None)
        if (
            document_id != expected_document.document_id
            or chunk_number != expected.chunk_number
            or start != expected.start
            or end != expected.end
            or sha256(content.encode("utf-8")).hexdigest() != expected.content_sha256
            or type(full_text) is not str
            or full_text[start:end] != content
        ):
            raise ComparisonAborted("fixture chunk span/text topology mismatch")
        try:
            vector = tuple(embedding)
        except TypeError as error:
            raise ComparisonAborted("fixture chunk embedding is missing") from error
        if canonical_embedding_sha256(vector) != expected.embedding_sha256:
            raise ComparisonAborted("fixture chunk embedding digest mismatch")
        observed_chunks[chunk_id] = row
    if set(observed_chunks) != set(expected_chunks):
        raise ComparisonAborted("fixture chunk scope is incomplete")
    return MappingProxyType(
        {
            "document_ids": tuple(
                sorted(observed_documents, key=lambda value: value.int)
            ),
            "chunk_ids": tuple(sorted(observed_chunks)),
            "manifest_checksum": manifest.manifest_checksum,
        }
    )


def revalidate_fixture_graph_assertions(
    manifest: ResolvedFixtureManifest,
    *,
    graph_snapshot: object,
    request: object,
) -> tuple[tuple[str, str], ...]:
    """Prove canonical links and hidden-neighbor exclusion from one live snapshot."""

    identities_by_chunk: dict[int, set[tuple[object, ...]]] = {}
    for row in getattr(graph_snapshot, "seed_identities", ()):
        chunk_id = getattr(row, "seed_chunk_id", None)
        identity = getattr(row, "identity_key", None)
        if type(chunk_id) is not int or type(identity) is not tuple:
            raise ComparisonAborted("fixture seed identity projection is malformed")
        identities_by_chunk.setdefault(chunk_id, set()).add(identity)
    evidence_chunk_ids: set[int] = set()
    for mention in getattr(graph_snapshot, "mentions", ()):
        evidence = getattr(mention, "evidence", None)
        chunk_id = getattr(evidence, "chunk_id", None)
        identity = getattr(mention, "identity_key", None)
        if type(chunk_id) is not int or type(identity) is not tuple:
            raise ComparisonAborted("fixture mention identity projection is malformed")
        evidence_chunk_ids.add(chunk_id)
        identities_by_chunk.setdefault(chunk_id, set()).add(identity)
    for group in getattr(graph_snapshot, "relation_groups", ()):
        for evidence in getattr(group, "evidence", ()):
            chunk_id = getattr(evidence, "chunk_id", None)
            if type(chunk_id) is not int:
                raise ComparisonAborted("fixture relation evidence is malformed")
            evidence_chunk_ids.add(chunk_id)

    hidden_chunk_ids = {
        binding.chunk_id
        for binding in manifest.chunks.values()
        if not manifest.collections[binding.collection_symbol].authorized
    }
    seed_ids = {
        getattr(seed, "chunk_id", None) for seed in getattr(request, "seeds", ())
    }
    if hidden_chunk_ids.intersection((*evidence_chunk_ids, *seed_ids)):
        raise ComparisonAborted("fixture inaccessible neighbor entered graph snapshot")

    observed: list[tuple[str, str]] = []
    for assertion in manifest.canonical_identity_assertions:
        source = manifest.chunk(assertion.source_chunk_symbol).chunk_id
        target = manifest.chunk(assertion.target_chunk_symbol).chunk_id
        if identities_by_chunk.get(source, set()).intersection(
            identities_by_chunk.get(target, set())
        ):
            observed.append(
                (assertion.source_chunk_symbol, assertion.target_chunk_symbol)
            )
    return tuple(observed)


def canonicalize_collection_scope(values: Iterable[Any]) -> tuple[int, ...]:
    """Validate one-to-four positive collection IDs and sort them once."""

    try:
        supplied = tuple(values)
    except TypeError as exc:
        raise ComparisonValidationError("collection scope must be iterable") from exc
    if not 1 <= len(supplied) <= 4:
        raise ComparisonValidationError(
            "comparison requires one through four collections"
        )
    if any(type(value) is not int or value <= 0 for value in supplied):
        raise ComparisonValidationError(
            "collection values must be positive integer primary keys"
        )
    if len(set(supplied)) != len(supplied):
        raise ComparisonValidationError("collection scope contains a duplicate")
    return tuple(sorted(supplied))


def canonicalize_collection_requests(
    collections: Iterable[Any],
    rebuild_requests: Iterable[Any],
) -> tuple[tuple[int, UUID], ...]:
    """Pair CLI occurrences positionally, then canonicalize by collection PK."""

    try:
        supplied_collections = tuple(collections)
        supplied_requests = tuple(rebuild_requests)
    except TypeError as error:
        raise ComparisonValidationError(
            "collection/rebuild request mapping must be iterable"
        ) from error
    scope = canonicalize_collection_scope(supplied_collections)
    if len(supplied_requests) != len(supplied_collections):
        raise ComparisonValidationError(
            "each collection requires exactly one paired --rebuild-request"
        )
    parsed: list[UUID] = []
    for value in supplied_requests:
        if type(value) is not str:
            raise ComparisonValidationError(
                "rebuild request values must be canonical UUID strings"
            )
        try:
            request_id = UUID(value)
        except ValueError as error:
            raise ComparisonValidationError(
                "rebuild request values must be canonical UUID strings"
            ) from error
        if str(request_id) != value:
            raise ComparisonValidationError(
                "rebuild request values must be canonical UUID strings"
            )
        parsed.append(request_id)
    if len(set(parsed)) != len(parsed):
        raise ComparisonValidationError("rebuild request mapping contains a duplicate")
    paired = tuple(zip(supplied_collections, parsed, strict=True))
    canonical = tuple(sorted(paired, key=lambda row: row[0]))
    if tuple(row[0] for row in canonical) != scope:
        raise ComparisonValidationError(
            "collection/rebuild request mapping changed scope"
        )
    return canonical


def validate_eval_bypass(
    *,
    eval_only: bool,
    debug: bool,
    environment: str,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Permit the disabled-overlay bypass only in an explicit debug/test process."""

    if eval_only is not True:
        raise ComparisonValidationError("comparison mode requires --eval-only")
    if type(debug) is not bool or type(environment) is not str:
        raise ComparisonValidationError("evaluation runtime context is invalid")
    environment_name = environment.strip().casefold()
    if environment_name in {"prod", "production"}:
        raise ComparisonValidationError("evaluation bypass is forbidden in production")
    if not debug and environment_name not in {"test", "testing", "pytest"}:
        raise ComparisonValidationError(
            "evaluation bypass requires explicit debug/test settings"
        )
    effective = os.environ if environ is None else environ
    if effective.get("KG_EVAL_BYPASS_ALLOWED") != "1":
        raise ComparisonValidationError("KG_EVAL_BYPASS_ALLOWED must equal 1")
    if effective.get("KG_BUILD_ENABLED", "0") != "0":
        raise ComparisonValidationError("KG_BUILD_ENABLED must remain 0")
    if effective.get("KG_OVERLAY_ENABLED", "0") != "0":
        raise ComparisonValidationError("KG_OVERLAY_ENABLED must remain 0")


def _scope_key(
    scope: object, requested_collections: tuple[int, ...]
) -> tuple[Any, ...]:
    document_ids = getattr(scope, "allowed_doc_ids", None)
    collection_ids = getattr(scope, "allowed_collection_ids", None)
    documents = getattr(scope, "documents", None)
    if type(document_ids) is not tuple or not document_ids:
        raise ComparisonAborted("authorized scope has no exact document tuple")
    if any(type(document_id) is not UUID for document_id in document_ids):
        raise ComparisonAborted("authorized scope contains an invalid document UUID")
    if document_ids != tuple(sorted(document_ids, key=lambda value: value.int)):
        raise ComparisonAborted("authorized scope document order is not canonical")
    if len(set(document_ids)) != len(document_ids):
        raise ComparisonAborted("authorized scope contains duplicate documents")
    if type(collection_ids) is not tuple:
        raise ComparisonAborted("authorized scope has no exact collection tuple")
    if collection_ids != requested_collections:
        raise ComparisonAborted("authorized scope collection set changed")
    if type(documents) is not tuple or len(documents) != len(document_ids):
        raise ComparisonAborted("authorized scope document materialization mismatch")
    projected = tuple(getattr(document, "id", None) for document in documents)
    if projected != document_ids:
        raise ComparisonAborted("authorized scope document order mismatch")
    projected_collections = tuple(
        sorted({getattr(document, "collection_id", None) for document in documents})
    )
    if projected_collections != requested_collections:
        raise ComparisonAborted("authorized scope document membership mismatch")
    return document_ids, collection_ids


def _positive_chunk_tuple(value: object, context: str) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise ComparisonAborted(f"{context} must be an exact tuple")
    if any(type(item) is not int or item <= 0 for item in value):
        raise ComparisonAborted(f"{context} contains an invalid chunk ID")
    if len(set(value)) != len(value):
        raise ComparisonAborted(f"{context} contains duplicate chunk IDs")
    return value


def _snapshot_evidence_payload(evidence: object) -> list[Any]:
    return [
        getattr(evidence, "chunk_id", None),
        getattr(evidence, "document_id", None),
        getattr(evidence, "chunk_number", None),
        getattr(evidence, "confidence", None),
        getattr(evidence, "provenance_key", None),
    ]


def _comparison_signature_payload(
    *,
    scope: object,
    candidate_snapshot: object,
    request: object,
    graph_snapshot: object,
) -> Mapping[str, Any]:
    """Project every comparison-relevant private snapshot component."""

    baseline_candidates = getattr(candidate_snapshot, "baseline_candidates", None)
    if type(baseline_candidates) is not tuple:
        raise ComparisonAborted("baseline candidate snapshot must be an exact tuple")
    baseline_ids = tuple(getattr(row, "pk", None) for row in baseline_candidates)
    _positive_chunk_tuple(baseline_ids, "baseline candidate order")
    seeds = getattr(request, "seeds", None)
    if type(seeds) is not tuple or not seeds:
        raise ComparisonAborted("comparison requires a nonempty exact seed snapshot")
    groups = getattr(graph_snapshot, "relation_groups", None)
    mentions = getattr(graph_snapshot, "mentions", None)
    identities = getattr(graph_snapshot, "identity_keys", None)
    seed_identities = getattr(graph_snapshot, "seed_identities", None)
    raw_audit = getattr(graph_snapshot, "raw_audit_rows", None)
    if any(
        type(item) is not tuple
        for item in (groups, mentions, identities, seed_identities, raw_audit)
    ):
        raise ComparisonAborted("authorized graph snapshot is not immutable")
    pre_normalized_groups = [
        [
            list(getattr(group, "source_key")),
            getattr(group, "relation_type"),
            list(getattr(group, "target_key")),
            getattr(getattr(group, "direction"), "value", None),
            getattr(group, "raw_weight"),
            getattr(group, "admission_hop"),
            [
                _snapshot_evidence_payload(evidence)
                for evidence in getattr(group, "evidence", ())
            ],
        ]
        for group in groups
    ]
    evidence = [
        _snapshot_evidence_payload(row)
        for group in groups
        for row in getattr(group, "evidence", ())
    ] + [
        _snapshot_evidence_payload(getattr(mention, "evidence", None))
        for mention in mentions
    ]
    return MappingProxyType(
        {
            "scope": {
                "documents": list(getattr(scope, "allowed_doc_ids")),
                "collections": list(getattr(scope, "allowed_collection_ids")),
            },
            "candidates": {
                "vector": list(getattr(candidate_snapshot, "vector_chunk_ids")),
                "trigram": list(getattr(candidate_snapshot, "trigram_chunk_ids")),
                "exact": list(getattr(candidate_snapshot, "exact_chunk_ids")),
                "baseline": list(baseline_ids),
                "exact_terms": list(getattr(candidate_snapshot, "exact_terms", ())),
            },
            "seeds": [
                [seed.chunk_id, seed.rank, seed.restart_weight] for seed in seeds
            ],
            "artifact_versions": [
                getattr(graph_snapshot, "scope_version_signature"),
                list(raw_audit),
                list(getattr(graph_snapshot, "artifact_provenance", ())),
            ],
            "canonical_memberships": [
                [row.seed_chunk_id, list(row.identity_key)] for row in seed_identities
            ],
            "mention_identity_mappings": [
                [
                    _snapshot_evidence_payload(getattr(mention, "evidence", None)),
                    list(getattr(mention, "identity_key")),
                ]
                for mention in mentions
            ],
            "nodes": [list(identity) for identity in identities],
            "pre_normalized_groups": pre_normalized_groups,
            "relations": [
                [
                    list(getattr(group, "source_key")),
                    getattr(group, "relation_type"),
                    list(getattr(group, "target_key")),
                    getattr(getattr(group, "direction"), "value", None),
                    getattr(group, "admission_hop"),
                ]
                for group in groups
            ],
            "evidence": evidence,
        }
    )


def _seed_snapshot_signature(candidate_snapshot: object) -> str:
    baseline = getattr(candidate_snapshot, "baseline_candidates", ())
    payload = {
        "vector": list(getattr(candidate_snapshot, "vector_chunk_ids", ())),
        "trigram": list(getattr(candidate_snapshot, "trigram_chunk_ids", ())),
        "exact": list(getattr(candidate_snapshot, "exact_chunk_ids", ())),
        "baseline": [getattr(row, "pk", None) for row in baseline],
        "seeds": [
            [seed.chunk_id, seed.rank, seed.restart_weight]
            for seed in getattr(candidate_snapshot, "graph_seeds", ())
        ],
    }
    return comparison_snapshot_signature(payload)


def _parse_private_trace(
    raw_trace: object,
    *,
    expected_hops: int,
    expected_algorithm_signature: str,
    expected_graph_version_signature: str,
    expected_chunk_ids: tuple[int, ...],
) -> Mapping[str, Any]:
    """Validate and reduce a private trace; raw trace material never escapes."""

    if type(raw_trace) is not bytes:
        raise ComparisonAborted("graph arm did not emit one private byte trace")
    try:
        payload = json.loads(raw_trace.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonAborted("graph arm emitted an invalid private trace") from exc
    if not isinstance(payload, Mapping):
        raise ComparisonAborted("graph arm emitted a malformed private trace")
    if payload.get("effective_max_hops") != expected_hops:
        raise ComparisonAborted("private trace hop configuration mismatch")
    if payload.get("algorithm_signature") != expected_algorithm_signature:
        raise ComparisonAborted("private trace algorithm signature mismatch")
    if payload.get("graph_version_signature") != expected_graph_version_signature:
        raise ComparisonAborted("private trace graph version signature mismatch")
    rows = payload.get("candidate_contributions")
    if not isinstance(rows, list):
        raise ComparisonAborted("private trace has no candidate contribution list")
    reduced: dict[int, tuple[int, int, float]] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != 6:
            raise ComparisonAborted("private trace contains a malformed candidate")
        (
            chunk_id,
            contribution_hex,
            distance,
            seed_rank,
            _document_id,
            _chunk_number,
        ) = row
        if (
            type(chunk_id) is not int
            or chunk_id <= 0
            or type(contribution_hex) is not str
            or type(distance) is not int
            or not 0 <= distance <= 2
            or type(seed_rank) is not int
            or seed_rank <= 0
            or chunk_id in reduced
        ):
            raise ComparisonAborted("private trace candidate values are invalid")
        try:
            contribution = float.fromhex(contribution_hex)
        except ValueError as exc:
            raise ComparisonAborted(
                "private trace contribution is not hexadecimal"
            ) from exc
        if not isfinite(contribution) or contribution <= 0.0:
            raise ComparisonAborted("private trace contribution must be positive")
        reduced[chunk_id] = (distance, seed_rank, contribution)
    if tuple(reduced) != expected_chunk_ids:
        raise ComparisonAborted("private trace candidate order mismatch")
    ppr_scores = payload.get("ppr_scores")
    retained_groups = payload.get("retained_groups")
    if not isinstance(ppr_scores, list) or not isinstance(retained_groups, list):
        raise ComparisonAborted("private trace has no retained graph metrics")
    node_keys: list[object] = []
    for row in ppr_scores:
        if not isinstance(row, list) or len(row) != 2:
            raise ComparisonAborted("private trace contains a malformed node score")
        node_key, score_hex = row
        if not isinstance(node_key, list) or type(score_hex) is not str:
            raise ComparisonAborted("private trace contains an invalid node score")
        try:
            score = float.fromhex(score_hex)
        except ValueError as exc:
            raise ComparisonAborted(
                "private trace node score is not hexadecimal"
            ) from exc
        if not isfinite(score) or score < 0.0:
            raise ComparisonAborted("private trace node score must be nonnegative")
        node_keys.append(node_key)
    if len({json.dumps(row, sort_keys=True) for row in node_keys}) != len(node_keys):
        raise ComparisonAborted("private trace contains duplicate nodes")
    for row in retained_groups:
        if not isinstance(row, list) or len(row) != 7:
            raise ComparisonAborted("private trace contains a malformed retained edge")
    return MappingProxyType(
        {
            "candidates": MappingProxyType(reduced),
            "node_count": len(ppr_scores),
            "edge_count": len(retained_groups),
        }
    )


def _ranking_row_ids(rows: object, context: str) -> tuple[int, ...]:
    if type(rows) is not tuple:
        raise ComparisonAborted(f"{context} must be an exact tuple")
    identifiers = tuple(getattr(row, "pk", None) for row in rows)
    return _positive_chunk_tuple(identifiers, context)


def _ranking_fingerprint(ranking: object, context: str) -> Mapping[str, Any]:
    combined = _ranking_row_ids(
        getattr(ranking, "combined_candidates", None),
        f"{context} combined candidates",
    )
    graph = _ranking_row_ids(
        getattr(ranking, "graph_candidates", None),
        f"{context} graph candidates",
    )
    ranked = _ranking_row_ids(
        getattr(ranking, "ranked_results", None),
        f"{context} ranked results",
    )
    if any(identifier not in set(combined) for identifier in (*graph, *ranked)):
        raise ComparisonAborted(f"{context} contains a row outside its candidate pool")
    inaccessible = getattr(ranking, "inaccessible_candidate_count", None)
    materialization_ms = getattr(ranking, "materialization_ms", None)
    rerank_ms = getattr(ranking, "rerank_ms", None)
    if type(inaccessible) is not int or inaccessible < 0:
        raise ComparisonAborted(f"{context} has an invalid inaccessible count")
    if any(
        type(value) not in (int, float)
        or not isfinite(float(value))
        or float(value) < 0.0
        for value in (materialization_ms, rerank_ms)
    ):
        raise ComparisonAborted(f"{context} has invalid timing metrics")
    return MappingProxyType(
        {
            "combined": combined,
            "graph": graph,
            "ranked": ranked,
            "inaccessible": inaccessible,
            "materialization_ms": float(materialization_ms),
            "rerank_ms": float(rerank_ms),
        }
    )


def run_one_snapshot_comparison(
    *,
    query: str,
    collection_ids: Iterable[int],
    top_k: int,
    model_cls: Any,
    graph_config: object,
    timeout_ms: int,
    prepare_embedding: Callable[[str], object],
    resolve_scope: Callable[[tuple[int, ...], object], object],
    authorized_snapshot: Callable[..., Any],
    collect_candidates: Callable[..., object],
    load_graph: Callable[..., object],
    rank_graph: Callable[..., object],
    materialize_and_rerank: Callable[..., object],
    app_config_getter: Callable[[str], object] | None = None,
    relevant_chunk_ids: tuple[int, ...] = (),
    validate_graph_snapshot: Callable[[object, object], None] | None = None,
    clock: Callable[[], float] = perf_counter,
) -> Mapping[str, Any]:
    """Compare vector, one-hop, and shipping PPR from one immutable snapshot.

    Dependencies are explicit so unit tests use deterministic fakes.  The live
    wrapper below injects the exact Task 15/16 functions without reimplementing
    search, RRF, ORM loading, caps, fan-out, or PPR.
    """

    if not isinstance(query, str) or not query.strip():
        raise ComparisonValidationError("comparison query must be nonempty")
    if type(top_k) is not int or top_k <= 0:
        raise ComparisonValidationError("top_k must be a positive integer")
    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 150:
        raise ComparisonValidationError("timeout_ms must be in [1, 150]")
    if not callable(clock):
        raise ComparisonValidationError("clock must be callable")
    relevant_ids = frozenset(
        _positive_chunk_tuple(relevant_chunk_ids, "relevant chunk IDs")
    )

    # Embedding work is intentionally outside the database transaction.
    query_embedding = prepare_embedding(query)
    canonical_scope = canonicalize_collection_scope(collection_ids)
    initial_scope = resolve_scope(canonical_scope, graph_config)
    initial_scope_key = _scope_key(initial_scope, canonical_scope)

    try:
        # The live caller owns the one outer repeatable-read transaction.  Only
        # database/graph phases below borrow fresh bounded query budgets; local
        # reranking deliberately runs outside those monotonic deadlines.
        with nullcontext():
            with authorized_snapshot(timeout_ms=timeout_ms):
                revalidated_scope = resolve_scope(canonical_scope, graph_config)
            if _scope_key(revalidated_scope, canonical_scope) != initial_scope_key:
                raise ComparisonAborted(
                    "authorized scope changed inside the repeatable-read snapshot"
                )
            candidate_started = clock()
            with authorized_snapshot(timeout_ms=timeout_ms):
                candidate_snapshot = collect_candidates(
                    model_cls,
                    query,
                    top_k,
                    revalidated_scope.documents,
                    query_embedding=query_embedding,
                    graph_config=graph_config,
                    initial_vector_error=None,
                    app_config_getter=app_config_getter,
                )
            candidate_ms = (clock() - candidate_started) * 1_000
            if not isfinite(candidate_ms) or candidate_ms < 0.0:
                raise ComparisonAborted("candidate acquisition timing is invalid")
            if (
                getattr(candidate_snapshot, "documents", None)
                != revalidated_scope.documents
            ):
                raise ComparisonAborted("candidate snapshot scope mismatch")
            if getattr(candidate_snapshot, "vector_error", None) is not None:
                raise ComparisonAborted("candidate vector acquisition error")
            if getattr(candidate_snapshot, "graph_seed_error", None) is not False:
                raise ComparisonAborted("candidate seed acquisition error")
            _positive_chunk_tuple(
                getattr(candidate_snapshot, "vector_chunk_ids", None),
                "vector candidate order",
            )
            _positive_chunk_tuple(
                getattr(candidate_snapshot, "trigram_chunk_ids", None),
                "trigram candidate order",
            )
            _positive_chunk_tuple(
                getattr(candidate_snapshot, "exact_chunk_ids", None),
                "exact candidate order",
            )
            graph_seeds = getattr(candidate_snapshot, "graph_seeds", None)
            if type(graph_seeds) is not tuple or not graph_seeds:
                raise ComparisonAborted("candidate snapshot has no graph seeds")
            baseline_candidates = getattr(
                candidate_snapshot, "baseline_candidates", None
            )
            if type(baseline_candidates) is not tuple:
                raise ComparisonAborted("candidate snapshot has no baseline tuple")
            baseline_candidate_ids = _ranking_row_ids(
                baseline_candidates,
                "baseline candidate order",
            )

            baseline_ranking = materialize_and_rerank(
                model_cls,
                query,
                top_k,
                baseline_candidates,
                authorized_scope=revalidated_scope,
                graph_chunk_ids=(),
                max_graph_candidates=0,
            )
            baseline_fingerprint = _ranking_fingerprint(
                baseline_ranking,
                "vector-only ranking",
            )
            from apps.documents.services import chunk_search as production_search

            ranking_identity_fields = (
                "combined",
                "graph",
                "ranked",
                "inaccessible",
            )
            for fail_open_status, capability in (
                ("miss", production_search._EVALUATION_GRAPH_MISS),
                ("error", production_search._EVALUATION_GRAPH_FAILURE),
            ):
                fail_open_graph_ids, fail_open_diagnostics = (
                    production_search._apply_graph_overlay(
                        model_cls,
                        candidate_snapshot,
                        revalidated_scope,
                        graph_config,
                        preflight_status=(
                            "miss" if fail_open_status == "miss" else None
                        ),
                        _eval_failure_capability=capability,
                    )
                )
                if (
                    fail_open_graph_ids != ()
                    or fail_open_diagnostics.get("graph_status") != fail_open_status
                    or fail_open_diagnostics.get("graph_candidate_count") != 0
                ):
                    raise ComparisonAborted(
                        "measured fail-open overlay did not return an exact "
                        f"{fail_open_status}"
                    )
                fail_open_ranking = materialize_and_rerank(
                    model_cls,
                    query,
                    top_k,
                    baseline_candidates,
                    authorized_scope=revalidated_scope,
                    graph_chunk_ids=fail_open_graph_ids,
                    max_graph_candidates=0,
                )
                fail_open_fingerprint = _ranking_fingerprint(
                    fail_open_ranking,
                    f"measured fail-open {fail_open_status} ranking",
                )
                if {
                    key: fail_open_fingerprint[key] for key in ranking_identity_fields
                } != {
                    key: baseline_fingerprint[key] for key in ranking_identity_fields
                }:
                    raise ComparisonAborted(
                        "measured fail-open ranking differs from exact baseline"
                    )

            from apps.knowledge_graph.retrieval.types import GraphExpansionRequest

            request = GraphExpansionRequest(
                seeds=graph_seeds,
                allowed_doc_ids=revalidated_scope.allowed_doc_ids,
                allowed_collection_ids=revalidated_scope.allowed_collection_ids,
            )
            graph_load_started = clock()
            with authorized_snapshot(timeout_ms=timeout_ms):
                graph_snapshot = load_graph(request, load_max_hops=2)
            graph_load_ms = (clock() - graph_load_started) * 1_000
            if not isfinite(graph_load_ms) or graph_load_ms < 0.0:
                raise ComparisonAborted("graph snapshot load timing is invalid")
            if getattr(graph_snapshot, "load_max_hops", None) != 2:
                raise ComparisonAborted("graph snapshot was not loaded to two hops")
            if (
                getattr(graph_snapshot, "allowed_doc_ids", None)
                != request.allowed_doc_ids
                or getattr(graph_snapshot, "allowed_collection_ids", None)
                != request.allowed_collection_ids
            ):
                raise ComparisonAborted("graph snapshot scope mismatch")
            shipping_config = getattr(graph_snapshot, "config", None)
            if (
                getattr(shipping_config, "max_hops", None) != 2
                or getattr(shipping_config, "ppr_iterations", None) != 8
            ):
                raise ComparisonAborted(
                    "comparison requires shipping max_hops=2 and eight iterations"
                )
            if validate_graph_snapshot is not None:
                if not callable(validate_graph_snapshot):
                    raise ComparisonAborted("graph snapshot validator is not callable")
                validate_graph_snapshot(graph_snapshot, request)

            snapshot_payload = _comparison_signature_payload(
                scope=revalidated_scope,
                candidate_snapshot=candidate_snapshot,
                request=request,
                graph_snapshot=graph_snapshot,
            )
            common_signature = comparison_snapshot_signature(snapshot_payload)
            seed_signature = _seed_snapshot_signature(candidate_snapshot)

            from apps.knowledge_graph.retrieval.expansion import (
                _EvaluationTraceCapability,
            )
            from apps.knowledge_graph.retrieval.ppr import graph_algorithm_signature

            def rank_and_materialize(
                arm_name: str,
                max_hops: int,
            ) -> tuple[
                object,
                Mapping[str, Any] | None,
                float,
                object,
                Mapping[str, Any],
            ]:
                traces: list[bytes] = []
                started = clock()
                with authorized_snapshot(timeout_ms=timeout_ms) as deadline:
                    result = rank_graph(
                        graph_snapshot,
                        request,
                        effective_max_hops=max_hops,
                        _eval_trace=_EvaluationTraceCapability(traces.append),
                        _deadline=deadline,
                    )
                elapsed_ms = (clock() - started) * 1_000
                if not isfinite(elapsed_ms) or elapsed_ms < 0.0:
                    raise ComparisonAborted(f"{arm_name} ranking timing is invalid")
                diagnostics = getattr(result, "diagnostics", None)
                status = getattr(diagnostics, "status", None)
                if status not in {"hit", "miss", "error"}:
                    raise ComparisonAborted(
                        f"{arm_name} graph arm returned {status or 'invalid'}"
                    )
                chunk_ids = _positive_chunk_tuple(
                    getattr(result, "chunk_ids", None),
                    f"{arm_name} graph candidate order",
                )
                expected_algorithm = graph_algorithm_signature(
                    replace(shipping_config, max_hops=max_hops)
                )
                algorithm_signature = getattr(diagnostics, "algorithm_signature", None)
                graph_version_signature = getattr(
                    diagnostics, "graph_version_signature", None
                )
                if algorithm_signature != expected_algorithm:
                    raise ComparisonAborted(f"{arm_name} algorithm signature mismatch")
                if status == "hit" and (
                    type(graph_version_signature) is not str
                    or _SHA256_PATTERN.fullmatch(graph_version_signature) is None
                ):
                    raise ComparisonAborted(
                        f"{arm_name} graph version signature mismatch"
                    )
                if status == "hit":
                    if not chunk_ids or len(traces) != 1:
                        raise ComparisonAborted(
                            f"{arm_name} hit must emit candidates and one private trace"
                        )
                    trace = _parse_private_trace(
                        traces[0],
                        expected_hops=max_hops,
                        expected_algorithm_signature=algorithm_signature,
                        expected_graph_version_signature=graph_version_signature,
                        expected_chunk_ids=chunk_ids,
                    )
                else:
                    if chunk_ids or traces:
                        raise ComparisonAborted(
                            f"{arm_name} fail-open result emitted graph material"
                        )
                    trace = None
                ranking = materialize_and_rerank(
                    model_cls,
                    query,
                    top_k,
                    baseline_candidates,
                    authorized_scope=revalidated_scope,
                    graph_chunk_ids=chunk_ids,
                    max_graph_candidates=(
                        shipping_config.max_candidates if chunk_ids else 0
                    ),
                )
                fingerprint = _ranking_fingerprint(ranking, f"{arm_name} ranking")
                if any(
                    identifier not in set(chunk_ids)
                    or identifier in set(baseline_candidate_ids)
                    for identifier in fingerprint["graph"]
                ):
                    raise ComparisonAborted(
                        f"{arm_name} materialized an unauthorized graph row"
                    )
                if status != "hit" and {
                    key: fingerprint[key] for key in ranking_identity_fields
                } != {
                    key: baseline_fingerprint[key] for key in ranking_identity_fields
                }:
                    raise ComparisonAborted(
                        f"{arm_name} fail-open ranking differs from exact baseline"
                    )
                return result, trace, elapsed_ms, ranking, fingerprint

            graph_arm_material = {
                arm_name: rank_and_materialize(arm_name, max_hops)
                for arm_name, max_hops in (("one_hop", 1), ("ppr_v1", 2))
            }

            if (
                graph_arm_material["one_hop"][0].diagnostics.status == "hit"
                and graph_arm_material["ppr_v1"][0].diagnostics.status == "hit"
                and graph_arm_material["one_hop"][0].diagnostics.graph_version_signature
                == graph_arm_material["ppr_v1"][0].diagnostics.graph_version_signature
            ):
                raise ComparisonAborted(
                    "graph arms must retain distinct request-induced versions"
                )

            (
                first_ppr_result,
                first_ppr_trace,
                _first_elapsed,
                _first_ppr_ranking,
                first_ppr_fingerprint,
            ) = graph_arm_material["ppr_v1"]
            (
                repeated_ppr_result,
                repeated_trace,
                _repeated_elapsed,
                _repeated_ranking,
                repeated_fingerprint,
            ) = rank_and_materialize(
                "repeated PPR",
                2,
            )
            repeated_diagnostics = getattr(repeated_ppr_result, "diagnostics", None)
            if (
                repeated_ppr_result.chunk_ids != first_ppr_result.chunk_ids
                or repeated_diagnostics.status != first_ppr_result.diagnostics.status
                or repeated_diagnostics.algorithm_signature
                != first_ppr_result.diagnostics.algorithm_signature
                or repeated_diagnostics.graph_version_signature
                != first_ppr_result.diagnostics.graph_version_signature
                or repeated_trace != first_ppr_trace
                or {
                    key: repeated_fingerprint[key]
                    for key in ("combined", "graph", "ranked", "inaccessible")
                }
                != {
                    key: first_ppr_fingerprint[key]
                    for key in ("combined", "graph", "ranked", "inaccessible")
                }
            ):
                raise ComparisonAborted(
                    "deterministic repeated PPR ranking or trace metrics changed"
                )
    except ComparisonAborted:
        raise
    except TimeoutError as exc:
        raise ComparisonAborted("comparison snapshot timed out") from exc
    except Exception as exc:
        raise ComparisonAborted(
            f"comparison snapshot failed atomically: {type(exc).__name__}"
        ) from exc

    vector_algorithm = comparison_snapshot_signature({"algorithm": "vector_only_v1"})
    common_arm = {
        "collection_scope": canonical_scope,
        "comparison_snapshot_signature": common_signature,
        "seed_snapshot_signature": seed_signature,
    }
    arms: dict[str, Mapping[str, Any]] = {
        "vector_only": MappingProxyType(
            {
                **common_arm,
                "name": "vector_only",
                "max_hops": 0,
                "ppr_iterations": 0,
                "algorithm_signature": vector_algorithm,
                "graph_version_signature": None,
                "ranked_chunk_ids": baseline_fingerprint["ranked"],
                "graph_chunk_ids": (),
                "graph_hit_rate": 0.0,
                "inaccessible_result_count": baseline_fingerprint["inaccessible"],
                "latency_ms": round(
                    candidate_ms
                    + baseline_fingerprint["materialization_ms"]
                    + baseline_fingerprint["rerank_ms"],
                    6,
                ),
                "graph_added_latency_ms": 0.0,
                "citation_evidence_coverage": 1.0,
                "seed_coverage": 1.0,
                "node_count": 0,
                "edge_count": 0,
                "distance_2_novel_fraction": 0.0,
            }
        )
    }
    mapped_seed_ids = {
        getattr(row, "seed_chunk_id", None) for row in graph_snapshot.seed_identities
    }
    seed_ids = tuple(seed.chunk_id for seed in request.seeds)
    evidence_ids = {
        getattr(evidence, "chunk_id", None)
        for group in graph_snapshot.relation_groups
        for evidence in getattr(group, "evidence", ())
    } | {
        getattr(getattr(mention, "evidence", None), "chunk_id", None)
        for mention in graph_snapshot.mentions
    }
    fail_open_miss_observations = 1
    fail_open_error_observations = 1
    for arm_name, max_hops in (("one_hop", 1), ("ppr_v1", 2)):
        result, trace, elapsed_ms, _ranking, fingerprint = graph_arm_material[arm_name]
        status = result.diagnostics.status
        if status == "miss":
            fail_open_miss_observations += 1
        elif status == "error":
            fail_open_error_observations += 1
        ranked_ids = fingerprint["ranked"]
        materialized_graph_ids = set(fingerprint["graph"])
        returned_graph_ids = tuple(
            chunk_id for chunk_id in ranked_ids if chunk_id in materialized_graph_ids
        )
        trace_candidates = {} if trace is None else trace["candidates"]
        distance_two_ids = tuple(
            chunk_id
            for chunk_id in returned_graph_ids
            if trace_candidates[chunk_id][0] == 2
        )
        distance_two = (
            len(distance_two_ids) / len(returned_graph_ids)
            if returned_graph_ids
            else 0.0
        )
        citation_coverage = (
            1.0
            if not returned_graph_ids
            else len(set(returned_graph_ids).intersection(evidence_ids))
            / len(returned_graph_ids)
        )
        seed_coverage = len(set(seed_ids).intersection(mapped_seed_ids)) / len(seed_ids)
        arms[arm_name] = MappingProxyType(
            {
                **common_arm,
                "name": arm_name,
                "max_hops": max_hops,
                "ppr_iterations": shipping_config.ppr_iterations,
                "algorithm_signature": result.diagnostics.algorithm_signature,
                "graph_version_signature": result.diagnostics.graph_version_signature,
                "ranked_chunk_ids": ranked_ids,
                "graph_chunk_ids": returned_graph_ids,
                "graph_hit_rate": 1.0 if status == "hit" else 0.0,
                "inaccessible_result_count": fingerprint["inaccessible"],
                "latency_ms": round(
                    candidate_ms
                    + graph_load_ms
                    + elapsed_ms
                    + fingerprint["materialization_ms"]
                    + fingerprint["rerank_ms"],
                    6,
                ),
                "graph_added_latency_ms": round(
                    graph_load_ms + elapsed_ms + fingerprint["materialization_ms"],
                    6,
                ),
                "citation_evidence_coverage": citation_coverage,
                "seed_coverage": seed_coverage,
                "node_count": 0 if trace is None else trace["node_count"],
                "edge_count": 0 if trace is None else trace["edge_count"],
                "distance_2_novel_fraction": distance_two,
                "distance_2_relevant_hit": bool(
                    set(distance_two_ids).intersection(relevant_ids)
                ),
            }
        )
    return MappingProxyType(
        {
            "collection_scope": canonical_scope,
            "comparison_snapshot_signature": common_signature,
            "seed_snapshot_signature": seed_signature,
            "deterministic_repeated_ppr": True,
            "fail_open_miss_observation_count": fail_open_miss_observations,
            "fail_open_error_observation_count": fail_open_error_observations,
            "exact_fail_open_parity": True,
            "arms": MappingProxyType(arms),
        }
    )


def _comparison_sha(value: Any, context: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ComparisonValidationError(
            f"{context} must be an exact lowercase SHA-256 digest"
        )
    return value


def _scan_forbidden_report_keys(value: Any, context: str = "bundle") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_REPORT_KEYS:
                raise ComparisonValidationError(
                    f"{context} contains forbidden private field {key!r}"
                )
            _scan_forbidden_report_keys(item, f"{context}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _scan_forbidden_report_keys(item, f"{context}[{index}]")


def _exact_string_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise ComparisonValidationError(f"{context} must be a string-keyed mapping")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: AbstractSet[str],
    context: str,
) -> None:
    if set(value) != set(expected):
        raise ComparisonValidationError(f"{context} fields are not exact")


def _validate_arm_metrics(value: Any, context: str) -> Mapping[str, Any]:
    metrics = _exact_string_mapping(value, context)
    _require_exact_fields(
        metrics,
        {
            "recall_at_10",
            "mrr",
            "ndcg_at_10",
            "graph_hit_rate",
            "inaccessible_result_count",
            "latency_p95_ms",
            "graph_added_latency_p95_ms",
            "citation_evidence_coverage",
            "seed_coverage",
            "node_count",
            "edge_count",
            "distance_2_novel_fraction",
        },
        context,
    )
    unit_fields = (
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "graph_hit_rate",
        "citation_evidence_coverage",
        "seed_coverage",
        "distance_2_novel_fraction",
    )
    for field in unit_fields:
        if field not in metrics:
            raise ComparisonValidationError(f"{context} missing metric {field!r}")
        _unit_metric(metrics[field], f"{context}.{field}")
    for field in ("latency_p95_ms", "graph_added_latency_p95_ms"):
        if field not in metrics:
            raise ComparisonValidationError(f"{context} missing metric {field!r}")
        _nonnegative_metric(metrics[field], f"{context}.{field}")
    for field in ("inaccessible_result_count", "node_count", "edge_count"):
        if type(metrics.get(field)) is not int or metrics[field] < 0:
            raise ComparisonValidationError(
                f"{context}.{field} must be a nonnegative exact integer"
            )
    return metrics


def current_comparison_algorithm_signatures() -> Mapping[str, str]:
    """Return exact signatures for the three currently shipping comparison arms."""

    from apps.knowledge_graph.retrieval.expansion import _load_algorithm_config
    from apps.knowledge_graph.retrieval.ppr import graph_algorithm_signature

    shipping = _load_algorithm_config()
    if shipping.max_hops != 2 or shipping.ppr_iterations != 8:
        raise ComparisonValidationError(
            "shipping comparison requires max_hops=2 and eight iterations"
        )
    return MappingProxyType(
        {
            "vector_only": comparison_snapshot_signature(
                {"algorithm": "vector_only_v1"}
            ),
            "one_hop": graph_algorithm_signature(replace(shipping, max_hops=1)),
            "ppr_v1": graph_algorithm_signature(shipping),
        }
    )


def _validate_version_bindings(value: Any) -> Mapping[str, Any]:
    versions = _exact_string_mapping(value, "artifact versions")
    if set(versions) != {"ontology", "resolver", "filter"}:
        raise ComparisonValidationError(
            "artifact versions require exact ontology/resolver/filter bindings"
        )
    for kind in ("ontology", "resolver", "filter"):
        rows = versions[kind]
        if type(rows) is not list or not rows:
            raise ComparisonValidationError(f"artifact versions missing {kind!r}")
        canonical: list[tuple[str, str]] = []
        for index, raw in enumerate(rows):
            row = _exact_string_mapping(raw, f"artifact versions.{kind}[{index}]")
            if set(row) != {"version", "checksum"}:
                raise ComparisonValidationError(
                    f"artifact versions.{kind}[{index}] fields are not exact"
                )
            version = row.get("version")
            if type(version) is not str or not version.strip():
                raise ComparisonValidationError(
                    f"artifact versions.{kind}[{index}] version is invalid"
                )
            checksum = _comparison_sha(
                row.get("checksum"),
                f"artifact versions.{kind}[{index}] checksum",
            )
            canonical.append((version, checksum))
        if canonical != sorted(set(canonical)):
            raise ComparisonValidationError(
                f"artifact versions.{kind} must be sorted and unique"
            )
    return versions


def validate_comparison_bundle(bundle: Any) -> Mapping[str, Any]:
    """Fail closed unless a report is an exact three-arm one-snapshot bundle."""

    bundle = _exact_string_mapping(bundle, "comparison bundle")
    _scan_forbidden_report_keys(bundle)
    _require_exact_fields(
        bundle,
        {
            "schema_version",
            "mode",
            "eval_only",
            "comparison_snapshot_signature",
            "collection_scope",
            "seed_snapshot_signature",
            "fixture_checksum",
            "fixture_manifest_checksum",
            "versions",
            "model",
            "extraction",
            "embedding",
            "reranker",
            "artifacts",
            "latency_budget_ms",
            "extraction_metrics",
            "invariants",
            "arms",
        },
        "comparison bundle",
    )
    if type(bundle.get("schema_version")) is not int or bundle["schema_version"] != 1:
        raise ComparisonValidationError("comparison schema_version must be integer 1")
    if bundle.get("mode") != "comparison" or bundle.get("eval_only") is not True:
        raise ComparisonValidationError(
            "comparison bundle must be evaluation-only comparison mode"
        )
    scope = canonicalize_collection_scope(bundle.get("collection_scope", ()))
    if list(scope) != bundle.get("collection_scope"):
        raise ComparisonValidationError("comparison collection scope is not canonical")
    common_signature = _comparison_sha(
        bundle.get("comparison_snapshot_signature"),
        "comparison snapshot signature",
    )
    seed_signature = _comparison_sha(
        bundle.get("seed_snapshot_signature"),
        "seed snapshot signature",
    )
    fixture_signature = _comparison_sha(
        bundle.get("fixture_checksum"),
        "fixture checksum",
    )
    if fixture_signature != fixture_checksum():
        raise ComparisonValidationError(
            "comparison fixture checksum differs from current fixtures"
        )
    manifest_signature = _comparison_sha(
        bundle.get("fixture_manifest_checksum"),
        "fixture manifest checksum",
    )
    versions = _validate_version_bindings(bundle.get("versions"))
    model = _exact_string_mapping(bundle.get("model"), "model")
    if set(model) != {"provider", "name", "checkpoint"}:
        raise ComparisonValidationError(
            "model provenance requires exact provider/name/checkpoint fields"
        )
    if model.get("provider") != "gliner2_local":
        raise ComparisonValidationError("model provider must be gliner2_local")
    if not is_safe_huggingface_repo_id(model.get("name")):
        raise ComparisonValidationError(
            "model name must be a Hugging Face repository ID"
        )
    if (
        type(model.get("checkpoint")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", model["checkpoint"]) is None
    ):
        raise ComparisonValidationError("model checkpoint must be an immutable commit")
    extraction = _exact_string_mapping(
        bundle.get("extraction"), "extraction provenance"
    )
    extraction_fields = {
        "provider",
        "model",
        "checkpoint",
        "build_enabled",
        "device",
        "batch_size",
        "max_batch_characters",
        "local_files_only",
        "fail_open",
        "config_signature",
    }
    if set(extraction) != extraction_fields:
        raise ComparisonValidationError("extraction provenance fields are not exact")
    if (
        extraction.get("provider") != "gliner2_local"
        or extraction.get("provider") != model["provider"]
    ):
        raise ComparisonValidationError("extraction provider is invalid")
    if (
        not is_safe_huggingface_repo_id(extraction.get("model"))
        or extraction.get("model") != model["name"]
    ):
        raise ComparisonValidationError("extraction model is invalid")
    if (
        type(extraction.get("checkpoint")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", extraction["checkpoint"]) is None
        or extraction["checkpoint"] != model["checkpoint"]
    ):
        raise ComparisonValidationError("extraction checkpoint is invalid")
    if extraction.get("build_enabled") is not False:
        raise ComparisonValidationError("extraction build flag is invalid")
    if (
        type(extraction.get("device")) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}", extraction["device"])
        is None
    ):
        raise ComparisonValidationError("extraction device is invalid")
    if (
        type(extraction.get("batch_size")) is not int
        or not 1 <= extraction["batch_size"] <= 64
    ):
        raise ComparisonValidationError("extraction batch size is invalid")
    if (
        type(extraction.get("max_batch_characters")) is not int
        or not 1 <= extraction["max_batch_characters"] <= 1_000_000
    ):
        raise ComparisonValidationError("extraction character cap is invalid")
    if extraction.get("local_files_only") is not True:
        raise ComparisonValidationError("extraction offline flag is invalid")
    if extraction.get("fail_open") is not False:
        raise ComparisonValidationError("extraction fail-open flag is invalid")
    extraction_signature = _comparison_sha(
        extraction.get("config_signature"), "extraction config signature"
    )
    if extraction_signature != comparison_snapshot_signature(
        {
            key: extraction[key]
            for key in sorted(extraction_fields - {"config_signature"})
        }
    ):
        raise ComparisonValidationError("extraction config signature mismatch")
    embedding = _exact_string_mapping(bundle.get("embedding"), "embedding provenance")
    if set(embedding) != {
        "model",
        "checkpoint",
        "tokenizer_checkpoint",
        "code_checkpoint",
        "dimensions",
        "input_type",
        "endpoint_signature",
        "extra_args_signature",
        "trust_remote_code",
        "runner",
        "dtype",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "max_model_len",
        "strict_protected_args",
        "api_key_signature",
        "download_dir",
        "python_bin",
        "config_signature",
    }:
        raise ComparisonValidationError("embedding provenance fields are not exact")
    for field in ("model", "checkpoint"):
        if type(embedding.get(field)) is not str or not embedding[field].strip():
            raise ComparisonValidationError(
                f"embedding provenance {field} must be nonempty"
            )
    if not is_safe_huggingface_repo_id(embedding.get("model")):
        raise ComparisonValidationError(
            "embedding model must be a Hugging Face repository ID"
        )
    embedding_checkpoint = embedding.get("checkpoint")
    if (
        type(embedding_checkpoint) is not str
        or re.fullmatch(r"[0-9a-f]{40}", embedding_checkpoint) is None
    ):
        raise ComparisonValidationError(
            "embedding checkpoint must be an immutable commit"
        )
    if embedding.get("tokenizer_checkpoint") != embedding.get("checkpoint"):
        raise ComparisonValidationError(
            "embedding tokenizer checkpoint differs from model checkpoint"
        )
    if embedding.get("code_checkpoint") != embedding.get("checkpoint"):
        raise ComparisonValidationError(
            "embedding code checkpoint differs from model checkpoint"
        )
    if type(embedding.get("dimensions")) is not int or embedding["dimensions"] != 1024:
        raise ComparisonValidationError("embedding dimensions must be exact 1024")
    if embedding.get("input_type") != "search_document":
        raise ComparisonValidationError("embedding provenance must use search_document")
    embedding_endpoint = _comparison_sha(
        embedding.get("endpoint_signature"),
        "embedding endpoint signature",
    )
    expected_embedding_endpoint = embedding_endpoint_signature(
        model=embedding["model"],
        checkpoint=embedding_checkpoint,
        dimensions=embedding["dimensions"],
        input_type=embedding["input_type"],
    )
    if embedding_endpoint != expected_embedding_endpoint:
        raise ComparisonValidationError("embedding endpoint signature mismatch")
    embedding_extra_signature = _comparison_sha(
        embedding.get("extra_args_signature"),
        "embedding extra arguments signature",
    )
    if (
        embedding_extra_signature
        != sha256(_STRICT_EMBED_EXTRA_ARGS.encode("utf-8")).hexdigest()
    ):
        raise ComparisonValidationError(
            "embedding extra arguments signature is not canonical"
        )
    if embedding.get("trust_remote_code") is not True:
        raise ComparisonValidationError(
            "embedding remote code configuration is invalid"
        )
    if embedding.get("runner") != "pooling":
        raise ComparisonValidationError("embedding runner is invalid")
    if embedding.get("dtype") != "float16":
        raise ComparisonValidationError("embedding dtype is invalid")
    if (
        type(embedding.get("tensor_parallel_size")) is not int
        or embedding["tensor_parallel_size"] != 1
    ):
        raise ComparisonValidationError("embedding tensor parallel size is invalid")
    if (
        type(embedding.get("gpu_memory_utilization")) is not float
        or embedding["gpu_memory_utilization"] != 0.12
    ):
        raise ComparisonValidationError("embedding GPU memory utilization is invalid")
    if (
        type(embedding.get("max_model_len")) is not int
        or embedding["max_model_len"] != 2048
    ):
        raise ComparisonValidationError("embedding maximum model length is invalid")
    if embedding.get("strict_protected_args") is not True:
        raise ComparisonValidationError("embedding protected-argument fence is invalid")
    if (
        _comparison_sha(
            embedding.get("api_key_signature"), "embedding API key signature"
        )
        != sha256(b"EMPTY").hexdigest()
    ):
        raise ComparisonValidationError("embedding API key signature is invalid")
    if embedding.get("download_dir") != "/root/.cache/huggingface/hub":
        raise ComparisonValidationError("embedding download directory is invalid")
    if embedding.get("python_bin") != "python3":
        raise ComparisonValidationError("embedding python binary is invalid")
    embedding_config_signature = _comparison_sha(
        embedding.get("config_signature"),
        "embedding config signature",
    )
    if embedding_config_signature != comparison_snapshot_signature(
        {key: embedding[key] for key in sorted(set(embedding) - {"config_signature"})}
    ):
        raise ComparisonValidationError("embedding config signature mismatch")
    reranker = _exact_string_mapping(bundle.get("reranker"), "reranker provenance")
    reranker_fields = {
        "provider",
        "model",
        "checkpoint",
        "tokenizer_checkpoint",
        "code_checkpoint",
        "endpoint_signature",
        "timeout_seconds",
        "document_char_limit",
        "multimodal",
        "extra_args_signature",
        "trust_remote_code",
        "runner",
        "task",
        "dtype",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "max_model_len",
        "strict_protected_args",
        "api_key_signature",
        "download_dir",
        "python_bin",
        "chat_template_sha256",
        "cache_enabled",
        "config_signature",
    }
    if set(reranker) != reranker_fields:
        raise ComparisonValidationError("reranker provenance fields are not exact")
    if reranker.get("provider") != "local":
        raise ComparisonValidationError("reranker provider must be exact local")
    reranker_model = reranker.get("model")
    if not is_safe_huggingface_repo_id(reranker_model):
        raise ComparisonValidationError("reranker model is invalid")
    checkpoint = reranker.get("checkpoint")
    if type(checkpoint) is not str or re.fullmatch(r"[0-9a-f]{40}", checkpoint) is None:
        raise ComparisonValidationError(
            "reranker checkpoint must be an immutable commit"
        )
    if reranker.get("tokenizer_checkpoint") != checkpoint:
        raise ComparisonValidationError(
            "reranker tokenizer checkpoint differs from model checkpoint"
        )
    if reranker.get("code_checkpoint") != checkpoint:
        raise ComparisonValidationError(
            "reranker code checkpoint differs from model checkpoint"
        )
    endpoint_signature = _comparison_sha(
        reranker.get("endpoint_signature"),
        "reranker endpoint signature",
    )
    expected_endpoint_signature = sha256(b"http://vllm_rerank:8000/v1").hexdigest()
    if endpoint_signature != expected_endpoint_signature:
        raise ComparisonValidationError(
            "reranker endpoint signature is not the isolated local endpoint"
        )
    timeout_seconds = reranker.get("timeout_seconds")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ComparisonValidationError("reranker timeout must be positive")
    document_char_limit = reranker.get("document_char_limit")
    if type(document_char_limit) is not int or document_char_limit <= 0:
        raise ComparisonValidationError("reranker document limit must be positive")
    multimodal = reranker.get("multimodal")
    if type(multimodal) is not bool or multimodal != (
        "qwen3-vl-reranker" in reranker_model.lower()
    ):
        raise ComparisonValidationError("reranker multimodal configuration is invalid")
    reranker_extra_signature = _comparison_sha(
        reranker.get("extra_args_signature"),
        "reranker extra arguments signature",
    )
    if (
        reranker_extra_signature
        != sha256(_STRICT_RERANK_EXTRA_ARGS.encode("utf-8")).hexdigest()
    ):
        raise ComparisonValidationError(
            "reranker extra arguments signature is not canonical"
        )
    if reranker.get("trust_remote_code") is not True:
        raise ComparisonValidationError("reranker remote code configuration is invalid")
    if reranker.get("runner") != "pooling":
        raise ComparisonValidationError("reranker runner is invalid")
    if reranker.get("task") != "score":
        raise ComparisonValidationError("reranker task is invalid")
    if reranker.get("dtype") != "float16":
        raise ComparisonValidationError("reranker dtype is invalid")
    if (
        type(reranker.get("tensor_parallel_size")) is not int
        or reranker["tensor_parallel_size"] != 1
    ):
        raise ComparisonValidationError("reranker tensor parallel size is invalid")
    if (
        type(reranker.get("gpu_memory_utilization")) is not float
        or reranker["gpu_memory_utilization"] != 0.25
    ):
        raise ComparisonValidationError("reranker GPU memory utilization is invalid")
    if (
        type(reranker.get("max_model_len")) is not int
        or reranker["max_model_len"] != 1024
    ):
        raise ComparisonValidationError("reranker maximum model length is invalid")
    if reranker.get("strict_protected_args") is not True:
        raise ComparisonValidationError("reranker protected-argument fence is invalid")
    if (
        _comparison_sha(reranker.get("api_key_signature"), "reranker API key signature")
        != sha256(b"EMPTY").hexdigest()
    ):
        raise ComparisonValidationError("reranker API key signature is invalid")
    if reranker.get("download_dir") != "/root/.cache/huggingface/hub":
        raise ComparisonValidationError("reranker download directory is invalid")
    if reranker.get("python_bin") != "python3":
        raise ComparisonValidationError("reranker python binary is invalid")
    chat_template_sha256 = _comparison_sha(
        reranker.get("chat_template_sha256"),
        "reranker chat template checksum",
    )
    try:
        current_template_sha256 = _rerank_template_checksum()
    except ComparisonAborted as error:
        raise ComparisonValidationError(str(error)) from error
    if chat_template_sha256 != current_template_sha256:
        raise ComparisonValidationError(
            "reranker chat template checksum differs from checked-in bytes"
        )
    if reranker.get("cache_enabled") is not False:
        raise ComparisonValidationError("reranker cache must be disabled")
    config_signature = _comparison_sha(
        reranker.get("config_signature"),
        "reranker config signature",
    )
    if config_signature != comparison_snapshot_signature(
        {key: reranker[key] for key in sorted(reranker_fields - {"config_signature"})}
    ):
        raise ComparisonValidationError("reranker config signature mismatch")
    from apps.knowledge_graph.retrieval.expansion import _load_algorithm_config

    latency_budget = bundle.get("latency_budget_ms")
    expected_latency_budget = float(_load_algorithm_config().timeout_ms)
    if type(latency_budget) is not float or latency_budget != expected_latency_budget:
        raise ComparisonValidationError(
            "latency budget must equal the current shipping graph timeout"
        )

    artifacts = bundle.get("artifacts")
    if type(artifacts) is not list or len(artifacts) != len(scope):
        raise ComparisonValidationError(
            "artifact provenance must bind every collection exactly once"
        )
    artifact_scope: list[int] = []
    artifact_requests: set[UUID] = set()
    for index, raw in enumerate(artifacts):
        artifact = _exact_string_mapping(raw, f"artifacts[{index}]")
        if set(artifact) != {
            "collection_id",
            "build_key",
            "source_hash",
            "rebuild_request",
        }:
            raise ComparisonValidationError(
                f"artifacts[{index}] provenance fields are not exact"
            )
        collection_id = artifact.get("collection_id")
        if type(collection_id) is not int or collection_id <= 0:
            raise ComparisonValidationError(
                f"artifacts[{index}] collection ID is invalid"
            )
        artifact_scope.append(collection_id)
        _comparison_sha(artifact.get("build_key"), f"artifacts[{index}].build_key")
        _comparison_sha(artifact.get("source_hash"), f"artifacts[{index}].source_hash")
        raw_request = artifact.get("rebuild_request")
        if type(raw_request) is not str:
            raise ComparisonValidationError(
                f"artifacts[{index}] rebuild request is invalid"
            )
        try:
            request_id = UUID(raw_request)
        except ValueError as error:
            raise ComparisonValidationError(
                f"artifacts[{index}] rebuild request is invalid"
            ) from error
        if str(request_id) != raw_request or request_id in artifact_requests:
            raise ComparisonValidationError(
                f"artifacts[{index}] rebuild request is duplicated or noncanonical"
            )
        artifact_requests.add(request_id)
    if artifact_scope != list(scope):
        raise ComparisonValidationError(
            "artifact provenance collection scope is not canonical"
        )

    arms = _exact_string_mapping(bundle.get("arms"), "comparison arms")
    if set(arms) != set(_COMPARISON_ARMS) or len(arms) != 3:
        raise ComparisonValidationError(
            "comparison bundle must contain the exact three arms"
        )
    expected_hops = {"vector_only": 0, "one_hop": 1, "ppr_v1": 2}
    expected_case_ids: tuple[str, ...] | None = None
    algorithm_signatures: dict[str, str] = {}
    graph_signatures: dict[str, str | None] = {}
    for arm_name in _COMPARISON_ARMS:
        arm = _exact_string_mapping(arms[arm_name], f"arm {arm_name}")
        _require_exact_fields(
            arm,
            {
                "name",
                "collection_scope",
                "comparison_snapshot_signature",
                "seed_snapshot_signature",
                "fixture_checksum",
                "fixture_manifest_checksum",
                "versions",
                "max_hops",
                "ppr_iterations",
                "algorithm_signature",
                "graph_version_signature",
                "metrics",
                "cases",
            },
            f"arm {arm_name}",
        )
        if arm.get("name") != arm_name:
            raise ComparisonValidationError(f"arm {arm_name} name mismatch")
        if arm.get("comparison_snapshot_signature") != common_signature:
            raise ComparisonValidationError(
                f"arm {arm_name} comparison snapshot mismatch"
            )
        if arm.get("collection_scope") != list(scope):
            raise ComparisonValidationError(f"arm {arm_name} collection scope mismatch")
        if arm.get("seed_snapshot_signature") != seed_signature:
            raise ComparisonValidationError(f"arm {arm_name} seed snapshot mismatch")
        if arm.get("fixture_checksum") != fixture_signature:
            raise ComparisonValidationError(f"arm {arm_name} fixture checksum mismatch")
        if arm.get("fixture_manifest_checksum") != manifest_signature:
            raise ComparisonValidationError(
                f"arm {arm_name} fixture manifest checksum mismatch"
            )
        if arm.get("versions") != versions:
            raise ComparisonValidationError(
                f"arm {arm_name} artifact versions mismatch"
            )
        if arm.get("max_hops") != expected_hops[arm_name]:
            if arm_name == "ppr_v1":
                raise ComparisonValidationError(
                    "shipping ppr_v1 arm must use max_hops=2"
                )
            raise ComparisonValidationError(
                f"arm {arm_name} hop configuration mismatch"
            )
        iterations = arm.get("ppr_iterations")
        if arm_name == "vector_only":
            if iterations != 0:
                raise ComparisonValidationError(
                    "vector-only arm cannot run PPR iterations"
                )
        elif iterations != 8:
            raise ComparisonValidationError(
                "graph arms must use the shipping eight iterations"
            )
        algorithm_signatures[arm_name] = _comparison_sha(
            arm.get("algorithm_signature"),
            f"arm {arm_name} algorithm signature",
        )
        graph_signatures[arm_name] = _comparison_sha(
            arm.get("graph_version_signature"),
            f"arm {arm_name} graph version signature",
            optional=arm_name == "vector_only",
        )
        if arm_name == "vector_only" and graph_signatures[arm_name] is not None:
            raise ComparisonValidationError(
                "vector-only arm cannot have a graph version signature"
            )
        _validate_arm_metrics(arm.get("metrics"), f"arm {arm_name}.metrics")
        cases = _exact_string_mapping(arm.get("cases"), f"arm {arm_name}.cases")
        current_case_metadata = {
            str(fixture_case["id"]): (
                list(fixture_case.get("quality_tags", ())),
                max(
                    fixture_case.get("expected_min_semantic_distance", {}).values(),
                    default=0,
                ),
            )
            for fixture_case in load_retrieval_cases()
        }
        case_ids = tuple(sorted(cases))
        if case_ids != tuple(sorted(current_case_metadata)):
            raise ComparisonValidationError(
                f"arm {arm_name} cases differ from the current fixture"
            )
        if expected_case_ids is None:
            expected_case_ids = case_ids
        elif case_ids != expected_case_ids:
            raise ComparisonValidationError("comparison arms have a case-set mismatch")
        for case_id, raw_case in cases.items():
            case = _exact_string_mapping(raw_case, f"arm {arm_name}.cases.{case_id}")
            _require_exact_fields(
                case,
                {
                    "quality_tags",
                    "minimum_semantic_distance",
                    "distance_2_relevant_hit",
                    "recall_at_10",
                    "ndcg_at_10",
                },
                f"arm {arm_name}.cases.{case_id}",
            )
            tags = case.get("quality_tags")
            if not isinstance(tags, list) or any(
                type(tag) is not str or not tag for tag in tags
            ):
                raise ComparisonValidationError(
                    f"arm {arm_name}.cases.{case_id} has invalid quality tags"
                )
            expected_tags, expected_distance = current_case_metadata[case_id]
            if tags != expected_tags:
                raise ComparisonValidationError(
                    f"arm {arm_name}.cases.{case_id} tags differ from current fixture"
                )
            _unit_metric(
                case.get("recall_at_10"),
                f"arm {arm_name}.cases.{case_id}.recall_at_10",
            )
            _unit_metric(
                case.get("ndcg_at_10"),
                f"arm {arm_name}.cases.{case_id}.ndcg_at_10",
            )
            if (
                type(case["minimum_semantic_distance"]) is not int
                or case["minimum_semantic_distance"] != expected_distance
            ):
                raise ComparisonValidationError(
                    f"arm {arm_name}.cases.{case_id} distance differs from "
                    "current fixture"
                )
            if type(case.get("distance_2_relevant_hit")) is not bool:
                raise ComparisonValidationError(
                    f"arm {arm_name}.cases.{case_id} distance-2 hit must be boolean"
                )
    current_algorithms = current_comparison_algorithm_signatures()
    if algorithm_signatures != dict(current_algorithms):
        raise ComparisonValidationError(
            "comparison algorithm signatures differ from current algorithms"
        )
    if algorithm_signatures["one_hop"] == algorithm_signatures["ppr_v1"]:
        raise ComparisonValidationError(
            "one-hop and ppr_v1 algorithm signatures must be distinct"
        )
    if graph_signatures["one_hop"] == graph_signatures["ppr_v1"]:
        raise ComparisonValidationError(
            "one-hop and ppr_v1 graph versions must be request-distinct"
        )
    extraction = _exact_string_mapping(
        bundle.get("extraction_metrics"), "extraction metrics"
    )
    extraction_fields = (
        "mention_span_precision",
        "mention_span_recall",
        "relation_endpoint_precision",
        "relation_endpoint_recall",
        "relation_direction_precision",
        "relation_direction_recall",
        "automatic_link_precision",
        "automatic_link_recall",
        "candidate_link_precision",
        "candidate_link_recall",
        "suppression_precision",
        "suppression_recall",
    )
    _require_exact_fields(extraction, set(extraction_fields), "extraction metrics")
    for field in extraction_fields:
        _unit_metric(extraction.get(field), f"extraction_metrics.{field}")
    invariants = _exact_string_mapping(bundle.get("invariants"), "invariants")
    _require_exact_fields(
        invariants,
        {
            "exact_baseline_on_graph_failure",
            "deterministic_repeated_ppr",
            "strict_local_reranking",
            "rerank_cache_enabled",
            "graph_miss_observations",
            "graph_error_observations",
        },
        "invariants",
    )
    for field in (
        "exact_baseline_on_graph_failure",
        "deterministic_repeated_ppr",
        "strict_local_reranking",
        "rerank_cache_enabled",
    ):
        if type(invariants.get(field)) is not bool:
            raise ComparisonValidationError(f"invariant {field!r} must be boolean")
    if invariants["strict_local_reranking"] is not True:
        raise ComparisonValidationError(
            "comparison requires strict local reranking without fallback"
        )
    if invariants["rerank_cache_enabled"] is not False:
        raise ComparisonValidationError("comparison rerank cache must be disabled")
    for field in ("graph_miss_observations", "graph_error_observations"):
        if type(invariants.get(field)) is not int or invariants[field] < 0:
            raise ComparisonValidationError(
                f"invariant {field} must be a nonnegative exact integer"
            )
    return bundle


def _gate_record(
    *, passed: bool, current_value: str, required_outcome: str
) -> Mapping[str, str]:
    return MappingProxyType(
        {
            "required_outcome": required_outcome,
            "current_value": current_value,
            "status": "PASS" if passed else "FAIL",
        }
    )


def evaluate_measured_gates(bundle: Any) -> Mapping[str, Mapping[str, str]]:
    """Derive all rollout gates only after validating the comparison bundle."""

    bundle = validate_comparison_bundle(bundle)
    arms = bundle["arms"]
    baseline = arms["vector_only"]
    one_hop = arms["one_hop"]
    ppr = arms["ppr_v1"]
    requirements = dict(_GATE_REQUIREMENTS)
    gates: dict[str, Mapping[str, str]] = {}

    inaccessible = sum(
        int(arms[name]["metrics"]["inaccessible_result_count"])
        for name in _COMPARISON_ARMS
    )
    gates["Permission isolation"] = _gate_record(
        passed=inaccessible == 0,
        current_value=f"{inaccessible} inaccessible chunks",
        required_outcome=requirements["Permission isolation"],
    )
    fail_open = bundle["invariants"]["exact_baseline_on_graph_failure"]
    miss_observations = bundle["invariants"]["graph_miss_observations"]
    error_observations = bundle["invariants"]["graph_error_observations"]
    gates["Fail-open parity"] = _gate_record(
        passed=(fail_open is True and miss_observations > 0 and error_observations > 0),
        current_value=(
            f"exact baseline parity={str(fail_open).lower()}; "
            f"miss={miss_observations}; error={error_observations}"
        ),
        required_outcome=requirements["Fail-open parity"],
    )
    automatic_precision = float(
        bundle["extraction_metrics"]["automatic_link_precision"]
    )
    candidate_precision = float(
        bundle["extraction_metrics"]["candidate_link_precision"]
    )
    gates["Identity precision"] = _gate_record(
        passed=automatic_precision > candidate_precision,
        current_value=(
            f"automatic={automatic_precision:.6g}; candidate={candidate_precision:.6g}"
        ),
        required_outcome=requirements["Identity precision"],
    )

    required_tags = {
        "relationship",
        "alias",
        "cross_document",
        "cross_collection",
    }
    observed_tags: set[str] = set()
    improving_tags: set[str] = set()
    for case_id, ppr_case in ppr["cases"].items():
        baseline_case = baseline["cases"][case_id]
        tags = set(ppr_case["quality_tags"]).intersection(required_tags)
        observed_tags.update(tags)
        if (
            ppr_case["recall_at_10"] > baseline_case["recall_at_10"]
            and ppr_case["ndcg_at_10"] > baseline_case["ndcg_at_10"]
        ):
            improving_tags.update(tags)
    quality_passed = observed_tags == required_tags and improving_tags == required_tags
    gates["Retrieval quality"] = _gate_record(
        passed=quality_passed,
        current_value=(
            "improved=" + ",".join(sorted(improving_tags))
            if improving_tags
            else "improved=none"
        ),
        required_outcome=requirements["Retrieval quality"],
    )

    no_worse = (
        ppr["metrics"]["recall_at_10"] >= one_hop["metrics"]["recall_at_10"]
        and ppr["metrics"]["ndcg_at_10"] >= one_hop["metrics"]["ndcg_at_10"]
    )
    better_distance_two = any(
        ppr_case.get("minimum_semantic_distance") == 2
        and ppr_case.get("distance_2_relevant_hit") is True
        and (
            ppr_case["recall_at_10"] > one_hop["cases"][case_id]["recall_at_10"]
            or ppr_case["ndcg_at_10"] > one_hop["cases"][case_id]["ndcg_at_10"]
        )
        for case_id, ppr_case in ppr["cases"].items()
    )
    gates["Multi-hop value"] = _gate_record(
        passed=no_worse and better_distance_two,
        current_value=(
            f"no_worse={str(no_worse).lower()}; "
            f"distance_two_better={str(better_distance_two).lower()}"
        ),
        required_outcome=requirements["Multi-hop value"],
    )

    latency = float(ppr["metrics"]["graph_added_latency_p95_ms"])
    budget = float(bundle["latency_budget_ms"])
    gates["Latency"] = _gate_record(
        passed=latency <= budget,
        current_value=f"graph_added_p95={latency:.6g}ms; budget={budget:.6g}ms",
        required_outcome=requirements["Latency"],
    )
    deterministic = bundle["invariants"]["deterministic_repeated_ppr"]
    gates["Determinism"] = _gate_record(
        passed=deterministic is True,
        current_value=f"identical={str(deterministic).lower()}",
        required_outcome=requirements["Determinism"],
    )
    citation_coverage = float(ppr["metrics"]["citation_evidence_coverage"])
    gates["Citations"] = _gate_record(
        passed=citation_coverage == 1.0,
        current_value=f"coverage={citation_coverage:.6g}",
        required_outcome=requirements["Citations"],
    )
    return MappingProxyType(gates)


def _load_comparison_report(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonValidationError(
            f"could not read comparison report {path}: {exc}"
        ) from exc
    return validate_comparison_bundle(payload)


def _atomic_replace_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def atomic_write_json(path: Path, payload: Any) -> None:
    """Serialize fully, then atomically replace one comparison report."""

    contents = (
        json.dumps(
            _thaw(payload),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    _atomic_replace_text(path, contents)


def _gate_rows(markdown: str) -> dict[str, tuple[int, list[str]]]:
    rows: dict[str, tuple[int, list[str]]] = {}
    for line_number, line in enumerate(markdown.splitlines()):
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4 or cells[0] not in dict(_GATE_REQUIREMENTS):
            continue
        if cells[0] in rows:
            raise GateVerificationError(f"duplicate measured gate row {cells[0]!r}")
        rows[cells[0]] = (line_number, cells)
    if set(rows) != set(dict(_GATE_REQUIREMENTS)):
        raise GateVerificationError("runbook measured gate table is incomplete")
    return rows


def write_measured_gates(
    comparison_report: Path, runbook: Path = _DEFAULT_RUNBOOK_PATH
) -> Mapping[str, Mapping[str, str]]:
    """Atomically replace pending runbook values from one validated report."""

    bundle = _load_comparison_report(comparison_report)
    gates = evaluate_measured_gates(bundle)
    try:
        original = runbook.read_text(encoding="utf-8")
    except OSError as exc:
        raise GateVerificationError(f"could not read runbook {runbook}: {exc}") from exc
    rows = _gate_rows(original)
    lines = original.splitlines()
    requirements = dict(_GATE_REQUIREMENTS)
    for gate_name, gate in gates.items():
        line_number, _cells = rows[gate_name]
        lines[line_number] = (
            f"| {gate_name} | {requirements[gate_name]} | "
            f"`{gate['current_value']}` | `{gate['status']}` |"
        )
    trailing_newline = "\n" if original.endswith(("\n", "\r")) else ""
    _atomic_replace_text(runbook, "\n".join(lines) + trailing_newline)
    return gates


def verify_measured_gates(
    comparison_report: Path, runbook: Path = _DEFAULT_RUNBOOK_PATH
) -> Mapping[str, Mapping[str, str]]:
    """Require exact passing, non-pending runbook values for this report."""

    bundle = _load_comparison_report(comparison_report)
    expected = evaluate_measured_gates(bundle)
    try:
        markdown = runbook.read_text(encoding="utf-8")
    except OSError as exc:
        raise GateVerificationError(f"could not read runbook {runbook}: {exc}") from exc
    rows = _gate_rows(markdown)
    if any(
        "PENDING_MEASUREMENT" in cell
        for _line_number, cells in rows.values()
        for cell in cells
    ):
        raise GateVerificationError("measured gates are still pending")
    failures: list[str] = []
    for gate_name, gate in expected.items():
        _line_number, cells = rows[gate_name]
        if (
            gate["status"] != "PASS"
            or cells[2] != gate["current_value"]
            or cells[3] != "PASS"
        ):
            failures.append(gate_name)
    if failures:
        raise GateVerificationError(
            f"measured gates are failing or mismatched: {', '.join(failures)}"
        )
    return expected


def format_comparison_table(bundle: Any) -> str:
    """Render the operator-facing metric summary without private trace data."""

    bundle = validate_comparison_bundle(bundle)
    columns = (
        ("Recall@10", "recall_at_10"),
        ("MRR", "mrr"),
        ("nDCG@10", "ndcg_at_10"),
        ("Graph hit rate", "graph_hit_rate"),
        ("Inaccessible", "inaccessible_result_count"),
        ("p95 latency", "latency_p95_ms"),
        ("Graph-added p95", "graph_added_latency_p95_ms"),
        ("Citation coverage", "citation_evidence_coverage"),
        ("Seed coverage", "seed_coverage"),
        ("Nodes", "node_count"),
        ("Edges", "edge_count"),
        ("Distance-2 novel", "distance_2_novel_fraction"),
    )
    version_text = "; ".join(
        f"{kind}="
        + ",".join(
            f"{row['version']}@{row['checksum']}" for row in bundle["versions"][kind]
        )
        for kind in ("ontology", "resolver", "filter")
    )
    lines = [
        "Comparison scope: " + ", ".join(map(str, bundle["collection_scope"])),
        f"Extractor provider: {bundle['model']['provider']}",
        f"Model: {bundle['model']['name']}",
        f"Checkpoint: {bundle['model']['checkpoint']}",
        f"Extractor device: {bundle['extraction']['device']}",
        "Extractor batch envelope: "
        f"count={bundle['extraction']['batch_size']} "
        f"characters={bundle['extraction']['max_batch_characters']}",
        "Extractor offline/fail-open/build: "
        f"{str(bundle['extraction']['local_files_only']).lower()} / "
        f"{str(bundle['extraction']['fail_open']).lower()} / "
        f"{str(bundle['extraction']['build_enabled']).lower()}",
        f"Extractor config signature: {bundle['extraction']['config_signature']}",
        f"Embedding model: {bundle['embedding']['model']}",
        f"Embedding checkpoint: {bundle['embedding']['checkpoint']}",
        "Embedding tokenizer checkpoint: "
        f"{bundle['embedding']['tokenizer_checkpoint']}",
        f"Embedding code checkpoint: {bundle['embedding']['code_checkpoint']}",
        f"Embedding dimensions: {bundle['embedding']['dimensions']}",
        f"Embedding input type: {bundle['embedding']['input_type']}",
        f"Embedding endpoint signature: {bundle['embedding']['endpoint_signature']}",
        "Embedding extra-args signature: "
        f"{bundle['embedding']['extra_args_signature']}",
        f"Embedding config signature: {bundle['embedding']['config_signature']}",
        "Embedding runner/dtype: "
        f"{bundle['embedding']['runner']} / {bundle['embedding']['dtype']}",
        "Embedding resources: "
        f"TP={bundle['embedding']['tensor_parallel_size']} "
        f"GPU={bundle['embedding']['gpu_memory_utilization']} "
        f"max_len={bundle['embedding']['max_model_len']}",
        "Embedding protected-argument fence: "
        f"{str(bundle['embedding']['strict_protected_args']).lower()}",
        f"Embedding API-key signature: {bundle['embedding']['api_key_signature']}",
        "Embedding runtime: "
        f"python={bundle['embedding']['python_bin']} "
        f"download={bundle['embedding']['download_dir']}",
        f"Reranker provider: {bundle['reranker']['provider']}",
        f"Reranker model: {bundle['reranker']['model']}",
        f"Reranker checkpoint: {bundle['reranker']['checkpoint']}",
        f"Reranker tokenizer checkpoint: {bundle['reranker']['tokenizer_checkpoint']}",
        f"Reranker code checkpoint: {bundle['reranker']['code_checkpoint']}",
        f"Reranker endpoint signature: {bundle['reranker']['endpoint_signature']}",
        f"Reranker extra-args signature: {bundle['reranker']['extra_args_signature']}",
        f"Reranker chat-template SHA-256: {bundle['reranker']['chat_template_sha256']}",
        "Reranker runner/dtype: "
        f"{bundle['reranker']['runner']} / {bundle['reranker']['dtype']}",
        f"Reranker task: {bundle['reranker']['task']}",
        "Reranker resources: "
        f"TP={bundle['reranker']['tensor_parallel_size']} "
        f"GPU={bundle['reranker']['gpu_memory_utilization']} "
        f"max_len={bundle['reranker']['max_model_len']}",
        "Reranker protected-argument fence: "
        f"{str(bundle['reranker']['strict_protected_args']).lower()}",
        f"Reranker cache enabled: {str(bundle['reranker']['cache_enabled']).lower()}",
        f"Reranker API-key signature: {bundle['reranker']['api_key_signature']}",
        "Reranker runtime: "
        f"python={bundle['reranker']['python_bin']} "
        f"download={bundle['reranker']['download_dir']}",
        f"Reranker config signature: {bundle['reranker']['config_signature']}",
        f"Versions: {version_text}",
        f"Fixture checksum: {bundle['fixture_checksum']}",
        f"Fixture manifest checksum: {bundle['fixture_manifest_checksum']}",
        f"Comparison snapshot: {bundle['comparison_snapshot_signature']}",
        "Arm signatures: "
        + "; ".join(
            f"{name}=algorithm:{bundle['arms'][name]['algorithm_signature']},"
            f"graph:{bundle['arms'][name]['graph_version_signature']}"
            for name in _COMPARISON_ARMS
        ),
        "",
        "| Arm | " + " | ".join(label for label, _field in columns) + " |",
        "| --- | " + " | ".join("---:" for _column in columns) + " |",
    ]
    for arm_name in _COMPARISON_ARMS:
        metrics = bundle["arms"][arm_name]["metrics"]
        rendered: list[str] = []
        for _label, field in columns:
            value = metrics[field]
            rendered.append(f"{value:.6g}" if type(value) is float else str(value))
        lines.append(f"| {arm_name} | " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _runtime_eval_context() -> tuple[bool, str]:
    """Read runtime mode lazily so importing this module remains pure Python."""

    from django.conf import settings

    debug = bool(getattr(settings, "DEBUG", False))
    environment = (
        os.environ.get("AQUILLM_ENV")
        or os.environ.get("DJANGO_ENV")
        or os.environ.get("ENVIRONMENT")
        or ("test" if os.environ.get("PYTEST_CURRENT_TEST") else "development")
    )
    return debug, environment


def _percentile_95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return ordered[index]


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _build_production_extraction_context(
    *,
    manifest: ResolvedFixtureManifest,
    selected_artifacts: tuple[dict[str, object], ...],
    manifest_rows: tuple[dict[str, object], ...],
    ontology: object,
    extraction_settings: object,
    filter_policy: object,
    resolution_config: object,
    canonical_assertions: tuple[tuple[str, str], ...],
) -> Mapping[str, Any]:
    """Validate and freeze the read-only production extraction projection input."""

    from apps.knowledge_graph.graph.filtering import (
        FilterPolicy,
        filter_policy_checksum,
    )
    from apps.knowledge_graph.models.inputs import collection_manifest_source_hash
    from apps.knowledge_graph.resolution.collection import (
        COLLECTION_RESOLVER_VERSION,
        CollectionBuildSnapshot,
        CollectionResolutionConfig,
        CollectionSnapshotInput,
        resolution_config_checksum,
    )

    if type(manifest) is not ResolvedFixtureManifest:
        raise ComparisonAborted("live extraction fixture manifest type changed")
    if type(selected_artifacts) is not tuple or any(
        type(row) is not dict for row in selected_artifacts
    ):
        raise ComparisonAborted("live extraction artifact selection is malformed")
    if type(manifest_rows) is not tuple or any(
        type(row) is not dict for row in manifest_rows
    ):
        raise ComparisonAborted("live extraction artifact manifest is malformed")
    if (
        type(filter_policy) is not FilterPolicy
        or type(resolution_config) is not CollectionResolutionConfig
    ):
        raise ComparisonAborted("live extraction policy/config type changed")
    if type(canonical_assertions) is not tuple:
        raise ComparisonAborted("live extraction canonical proof is malformed")

    expected_scope = tuple(
        collection_id for collection_id, _request in manifest.authorized_scope
    )
    selected_by_collection: dict[int, dict[str, object]] = {}
    for artifact in selected_artifacts:
        collection_id = artifact.get("collection_scope_id")
        artifact_id = artifact.get("id")
        request_id = artifact.get("rebuild_request_id")
        if (
            type(collection_id) is not int
            or type(artifact_id) is not int
            or type(request_id) is not UUID
            or artifact.get("scope_type") != "collection"
            or artifact.get("scope_id") != str(collection_id)
            or artifact.get("status") != "superseded"
            or artifact.get("evaluation_only") is not True
            or collection_id in selected_by_collection
        ):
            raise ComparisonAborted("live extraction collection artifact changed")
        selected_by_collection[collection_id] = artifact
    if tuple(sorted(selected_by_collection)) != expected_scope:
        raise ComparisonAborted("live extraction collection artifact scope changed")

    configured_extractor = (
        f"{extraction_settings.provider}:{extraction_settings.model_id}@"
        f"{extraction_settings.model_revision}"
    )
    resolver_checksum = resolution_config_checksum(resolution_config)
    policy_checksum = filter_policy_checksum(filter_policy)
    for collection_id, artifact in selected_by_collection.items():
        expected_request = dict(manifest.authorized_scope)[collection_id]
        if (
            artifact.get("rebuild_request_id") != expected_request
            or artifact.get("ontology_version") != getattr(ontology, "version", None)
            or artifact.get("ontology_checksum") != getattr(ontology, "checksum", None)
            or artifact.get("extractor_version") != configured_extractor
            or artifact.get("resolver_version") != COLLECTION_RESOLVER_VERSION
            or artifact.get("resolution_config_checksum") != resolver_checksum
            or artifact.get("filter_policy_version") != filter_policy.version
            or artifact.get("filter_policy_checksum") != policy_checksum
        ):
            raise ComparisonAborted(
                "live extraction artifact differs from production provenance"
            )

    expected_documents = {
        binding.document_id: binding
        for binding in manifest.documents.values()
        if manifest.collections[binding.collection_symbol].authorized
    }
    if len(manifest_rows) != len(expected_documents):
        raise ComparisonAborted("live extraction manifest document count changed")
    rows_by_collection: dict[int, list[dict[str, object]]] = {
        collection_id: [] for collection_id in expected_scope
    }
    document_artifact_ids: dict[UUID, int] = {}
    for row in manifest_rows:
        document_id = row.get("document_id")
        collection_id = row.get("collection_id")
        artifact_id = row.get("artifact_id")
        document_artifact_id = row.get("document_artifact_id")
        if (
            type(document_id) is not UUID
            or type(collection_id) is not int
            or type(artifact_id) is not int
            or type(document_artifact_id) is not int
            or document_artifact_id <= 0
            or document_id in document_artifact_ids
            or document_id not in expected_documents
            or collection_id not in selected_by_collection
            or artifact_id != selected_by_collection[collection_id]["id"]
        ):
            raise ComparisonAborted("live extraction manifest ownership changed")
        binding = expected_documents[document_id]
        request_id = selected_by_collection[collection_id]["rebuild_request_id"]
        if (
            binding.collection_id != collection_id
            or row.get("document_artifact__scope_type") != "document"
            or row.get("document_artifact__scope_id") != str(document_id)
            or row.get("document_artifact__rebuild_request_id") != request_id
            or row.get("document_artifact__status") != "superseded"
            or row.get("document_artifact__evaluation_only") is not True
            or row.get("document_artifact__ontology_version")
            != getattr(ontology, "version", None)
            or row.get("document_artifact__ontology_checksum")
            != getattr(ontology, "checksum", None)
            or row.get("document_artifact__extractor_version") != configured_extractor
        ):
            raise ComparisonAborted(
                "live extraction document artifact provenance changed"
            )
        for field in ("source_signature", "membership_signature", "build_signature"):
            if (
                type(row.get(field)) is not str
                or _SHA256_PATTERN.fullmatch(row[field]) is None
            ):
                raise ComparisonAborted(
                    "live extraction manifest signature is malformed"
                )
        document_artifact_ids[document_id] = document_artifact_id
        rows_by_collection[collection_id].append(row)
    if set(document_artifact_ids) != set(expected_documents):
        raise ComparisonAborted("live extraction manifest document scope changed")

    collection_snapshots: dict[int, object] = {}
    embedding_signatures: dict[int, str] = {}
    for collection_id in expected_scope:
        artifact = selected_by_collection[collection_id]
        rows = tuple(
            sorted(rows_by_collection[collection_id], key=lambda row: int(row["id"]))
        )
        source_signatures = tuple(str(row["source_signature"]) for row in rows)
        if collection_manifest_source_hash(source_signatures) != artifact.get(
            "source_hash"
        ):
            raise ComparisonAborted("live extraction artifact source hash changed")
        collection_snapshots[collection_id] = CollectionBuildSnapshot(
            destination_artifact_id=int(artifact["id"]),
            collection_id=collection_id,
            inputs=tuple(
                CollectionSnapshotInput(
                    manifest_input_id=int(row["id"]),
                    document_artifact_id=int(row["document_artifact_id"]),
                    document_id=row["document_id"],
                    membership_signature=str(row["membership_signature"]),
                    source_signature=str(row["source_signature"]),
                    build_signature=str(row["build_signature"]),
                )
                for row in rows
            ),
            source_hash=str(artifact["source_hash"]),
            ontology_version=str(artifact["ontology_version"]),
            ontology_checksum=str(artifact["ontology_checksum"]),
            filter_policy_checksum=str(artifact["filter_policy_checksum"]),
            resolution_config_checksum=str(artifact["resolution_config_checksum"]),
        )
        signature = artifact.get("embedding_model_signature")
        if type(signature) is not str or not signature:
            raise ComparisonAborted("live extraction embedding provenance changed")
        embedding_signatures[collection_id] = signature

    expected_assertions = tuple(
        sorted(
            (
                row.source_chunk_symbol,
                row.target_chunk_symbol,
            )
            for row in manifest.canonical_identity_assertions
        )
    )
    if tuple(sorted(canonical_assertions)) != expected_assertions:
        raise ComparisonAborted("live extraction canonical proof changed")
    return MappingProxyType(
        {
            "canonical_assertions": expected_assertions,
            "collection_snapshots": MappingProxyType(collection_snapshots),
            "document_artifact_ids": MappingProxyType(document_artifact_ids),
            "embedding_signatures": MappingProxyType(embedding_signatures),
            "extraction_settings": extraction_settings,
            "filter_policy": filter_policy,
            "manifest": manifest,
            "ontology": ontology,
            "resolution_config": resolution_config,
        }
    )


def _live_comparison_bundle(
    *,
    collection_scope: tuple[int, ...],
    evaluation_rebuild_requests: tuple[tuple[int, UUID], ...],
    fixture_manifest_path: Path,
    extraction_cases: Sequence[Mapping[str, Any]],
    retrieval_cases: Sequence[Mapping[str, Any]],
    extraction_path: Path = _DEFAULT_EXTRACTION_CASES_PATH,
    retrieval_path: Path = _DEFAULT_RETRIEVAL_CASES_PATH,
) -> Mapping[str, Any]:
    """Run all curated queries inside one Task 15 repeatable-read snapshot.

    Query embeddings are prepared before the transaction.  Each query then
    invokes Task 16 candidate acquisition and Task 15 graph loading once.  The
    shared outer transaction, scope, and per-query immutable snapshots are
    folded into one suite signature used by all three report arms.
    """

    from contextlib import contextmanager

    from django.apps import apps
    from django.conf import settings

    from apps.documents.models import Document, TextChunk
    from apps.documents.services.chunk_rerank import _STRICT_EVALUATION_RERANK
    from apps.documents.services.chunk_rerank_config import (
        rerank_api_key,
        rerank_base_url,
        rerank_code_revision,
        rerank_doc_char_limit,
        rerank_dtype,
        rerank_extra_args,
        rerank_gpu_memory_utilization,
        rerank_max_model_len,
        rerank_model,
        rerank_model_is_qwen3_vl,
        rerank_model_revision,
        rerank_provider,
        rerank_runner,
        rerank_task,
        rerank_tensor_parallel_size,
        rerank_timeout_seconds,
        rerank_tokenizer,
        rerank_tokenizer_revision,
        rerank_trust_remote_code,
        rerank_vllm_model,
    )
    from apps.documents.services.chunk_search import materialize_and_rerank_candidates
    from apps.documents.services.chunk_search_candidates import (
        collect_hybrid_candidate_snapshot,
        freeze_authorized_document_scope,
    )
    from apps.knowledge_graph.graph.filtering import FilterPolicy
    from apps.knowledge_graph.models import CollectionArtifactInput
    from apps.knowledge_graph.resolution.collection import CollectionResolutionConfig
    from apps.knowledge_graph.retrieval import get_graph_expansion_config
    from apps.knowledge_graph.retrieval.expansion import (
        _authorize_evaluation_collection_scope,
        _EvaluationArtifactCapability,
        _require_live_snapshot_state,
        authorized_retrieval_query_snapshot,
        authorized_retrieval_snapshot,
        load_authorized_graph_snapshot,
        rank_authorized_graph_snapshot,
    )
    from apps.knowledge_graph.services.builds import _active_ontology
    from lib.embeddings.config import get_local_embed_config, get_target_dims
    from lib.embeddings.local import get_strict_indexed_embeddings_via_local_openai
    from lib.knowledge_graph.config import load_extraction_settings

    if not retrieval_cases:
        raise ComparisonAborted("comparison fixture cannot skip every retrieval case")
    extraction_settings = _validate_live_extraction_settings(load_extraction_settings())
    extraction_provenance = _extraction_config_provenance(extraction_settings)
    fixture_signature = fixture_checksum(extraction_path, retrieval_path)
    resolved_manifest = validate_fixture_manifest(
        load_fixture_manifest(fixture_manifest_path),
        extraction_cases=extraction_cases,
        retrieval_cases=retrieval_cases,
        collection_requests=evaluation_rebuild_requests,
        expected_fixture_checksum=fixture_signature,
    )
    embedding_base_url, embedding_api_key, embedding_model = get_local_embed_config()
    embedding_provenance = _validate_live_embedding_contract(
        resolved_manifest.embedding,
        base_url=embedding_base_url,
        api_key=embedding_api_key,
        configured_sidecar_api_key=os.environ.get("APP_EMBED_VLLM_API_KEY"),
        configured_model=embedding_model,
        configured_checkpoint=os.environ.get("APP_EMBED_MODEL_REVISION"),
        configured_tokenizer_checkpoint=os.environ.get("APP_EMBED_TOKENIZER_REVISION"),
        configured_code_checkpoint=os.environ.get("APP_EMBED_CODE_REVISION"),
        configured_extra_args=os.environ.get("MEM0_EMBED_VLLM_EXTRA_ARGS"),
        configured_trust_remote_code=os.environ.get(
            "MEM0_EMBED_VLLM_TRUST_REMOTE_CODE"
        ),
        configured_runner=os.environ.get("APP_EMBED_VLLM_RUNNER"),
        configured_dtype=os.environ.get("APP_EMBED_VLLM_DTYPE"),
        configured_tensor_parallel_size=os.environ.get(
            "MEM0_EMBED_TENSOR_PARALLEL_SIZE"
        ),
        configured_gpu_memory_utilization=os.environ.get(
            "MEM0_EMBED_GPU_MEMORY_UTILIZATION"
        ),
        configured_max_model_len=os.environ.get("MEM0_EMBED_MAX_MODEL_LEN"),
        configured_strict_protected_args=os.environ.get(
            "APP_EMBED_VLLM_STRICT_PROTECTED_ARGS"
        ),
        configured_download_dir=os.environ.get("APP_EMBED_VLLM_DOWNLOAD_DIR"),
        configured_python_bin=os.environ.get("APP_EMBED_VLLM_PYTHON_BIN"),
        configured_dimensions=get_target_dims(),
    )
    reranker_provenance = _validate_live_reranker_contract(
        provider=rerank_provider(),
        base_url=rerank_base_url(),
        api_key=rerank_api_key(),
        configured_model=rerank_model(),
        loaded_model=rerank_vllm_model(),
        configured_tokenizer=rerank_tokenizer(),
        configured_checkpoint=rerank_model_revision(),
        configured_tokenizer_checkpoint=rerank_tokenizer_revision(),
        configured_code_checkpoint=rerank_code_revision(),
        configured_extra_args=rerank_extra_args(),
        configured_trust_remote_code=rerank_trust_remote_code(),
        configured_runner=rerank_runner(),
        configured_task=rerank_task(),
        configured_dtype=rerank_dtype(),
        configured_tensor_parallel_size=rerank_tensor_parallel_size(),
        configured_gpu_memory_utilization=rerank_gpu_memory_utilization(),
        configured_max_model_len=rerank_max_model_len(),
        configured_strict_protected_args=os.environ.get(
            "APP_RERANK_VLLM_STRICT_PROTECTED_ARGS"
        ),
        configured_download_dir=os.environ.get("APP_RERANK_VLLM_DOWNLOAD_DIR", ""),
        configured_python_bin=os.environ.get("APP_RERANK_VLLM_PYTHON_BIN", ""),
        configured_cache_enabled=getattr(settings, "RAG_CACHE_ENABLED", None),
        timeout_seconds=rerank_timeout_seconds(),
        document_char_limit=rerank_doc_char_limit(),
        multimodal=rerank_model_is_qwen3_vl(),
    )
    sorted_retrieval_cases = tuple(
        sorted(retrieval_cases, key=lambda item: str(item["id"]))
    )
    indexed_embeddings = get_strict_indexed_embeddings_via_local_openai(
        [str(case["query"]) for case in sorted_retrieval_cases]
    )
    if tuple(index for index, _vector in indexed_embeddings) != tuple(
        range(len(sorted_retrieval_cases))
    ):
        raise ComparisonAborted("strict local query embeddings lost index order")
    prepared_embeddings: dict[str, tuple[object, ...]] = {}
    for case, (_index, vector) in zip(
        sorted_retrieval_cases,
        indexed_embeddings,
        strict=True,
    ):
        frozen_vector = tuple(vector)
        canonical_embedding_sha256(frozen_vector)
        prepared_embeddings[str(case["id"])] = frozen_vector
    graph_config = get_graph_expansion_config()
    eval_artifacts = _EvaluationArtifactCapability(evaluation_rebuild_requests)

    def materialize_scope() -> object:
        documents = Document.filter(collection_id__in=collection_scope)
        return freeze_authorized_document_scope(documents, graph_config)

    timeout_ms = int(getattr(settings, "KG_OVERLAY_TIMEOUT_MS", 150))
    per_case: dict[str, Mapping[str, Any]] = {}
    observed_canonical_assertions: set[tuple[str, str]] = set()

    def strict_eval_materialize(*args: Any, **kwargs: Any) -> object:
        return materialize_and_rerank_candidates(
            *args,
            **kwargs,
            _eval_rerank_capability=_STRICT_EVALUATION_RERANK,
        )

    with authorized_retrieval_snapshot(timeout_ms=timeout_ms):
        with authorized_retrieval_query_snapshot(timeout_ms=timeout_ms):
            selected_artifacts = _authorize_evaluation_collection_scope(
                eval_artifacts,
                collection_scope,
                _require_live_snapshot_state(),
            )
            live_state = _require_live_snapshot_state()
            expected_authorized_documents = tuple(
                sorted(
                    binding.document_id
                    for binding in resolved_manifest.documents.values()
                    if resolved_manifest.collections[
                        binding.collection_symbol
                    ].authorized
                )
            )
            live_state.deadline.check()
            extraction_manifest_rows = tuple(
                CollectionArtifactInput.objects.using(live_state.using)
                .filter(
                    artifact_id__in=tuple(int(row["id"]) for row in selected_artifacts),
                    collection_id__in=collection_scope,
                    document_id__in=expected_authorized_documents,
                )
                .values(
                    "id",
                    "artifact_id",
                    "collection_id",
                    "document_id",
                    "document_artifact_id",
                    "source_signature",
                    "membership_signature",
                    "build_signature",
                    "document_artifact__scope_type",
                    "document_artifact__scope_id",
                    "document_artifact__rebuild_request_id",
                    "document_artifact__status",
                    "document_artifact__evaluation_only",
                    "document_artifact__ontology_version",
                    "document_artifact__ontology_checksum",
                    "document_artifact__extractor_version",
                )
                .order_by("pk")[: len(expected_authorized_documents) + 1]
            )
            live_state.deadline.check()
            if len(extraction_manifest_rows) != len(expected_authorized_documents):
                raise ComparisonAborted(
                    "live extraction artifact manifest is incomplete or oversized"
                )
            live_ontology = _active_ontology()
            live_filter_policy = FilterPolicy()
            live_resolution_config = CollectionResolutionConfig()
            revalidated_scope = materialize_scope()
            expected_document_ids = tuple(
                sorted(
                    (
                        binding.document_id
                        for binding in resolved_manifest.documents.values()
                        if resolved_manifest.collections[
                            binding.collection_symbol
                        ].authorized
                    ),
                    key=lambda value: value.int,
                )
            )
            if revalidated_scope.allowed_doc_ids != expected_document_ids:
                raise ComparisonAborted(
                    "authorized collections differ from exact fixture documents"
                )
            expected_chunk_ids = tuple(
                sorted(
                    binding.chunk_id
                    for binding in resolved_manifest.chunks.values()
                    if resolved_manifest.collections[
                        binding.collection_symbol
                    ].authorized
                )
            )
            chunk_rows = tuple(
                TextChunk.objects.filter(
                    pk__in=expected_chunk_ids,
                    doc_id__in=expected_document_ids,
                )
                .only(
                    "pk",
                    "doc_id",
                    "chunk_number",
                    "start_position",
                    "end_position",
                    "content",
                    "embedding",
                )
                .order_by("pk")[: len(expected_chunk_ids) + 1]
            )
            database_attestation = revalidate_fixture_database_rows(
                resolved_manifest,
                document_rows=revalidated_scope.documents,
                chunk_rows=chunk_rows,
            )
        initial_scope = revalidated_scope
        initial_key = _scope_key(initial_scope, collection_scope)

        @contextmanager
        def borrowed_snapshot(*, timeout_ms: int):
            if timeout_ms != int(getattr(settings, "KG_OVERLAY_TIMEOUT_MS", 150)):
                raise ComparisonAborted("borrowed snapshot timeout mismatch")
            with authorized_retrieval_query_snapshot(timeout_ms=timeout_ms) as deadline:
                yield deadline

        def load_evaluation_graph(request: object, *, load_max_hops: int) -> object:
            return load_authorized_graph_snapshot(
                request,
                load_max_hops=load_max_hops,
                _eval_artifacts=eval_artifacts,
            )

        def resolved_scope(
            requested: tuple[int, ...], requested_config: object
        ) -> object:
            if requested != collection_scope or requested_config is not graph_config:
                raise ComparisonAborted("internal comparison scope/config mismatch")
            if _scope_key(revalidated_scope, collection_scope) != initial_key:
                raise ComparisonAborted("internal comparison scope changed")
            return revalidated_scope

        def validate_live_graph(graph_snapshot: object, request: object) -> None:
            observed_canonical_assertions.update(
                revalidate_fixture_graph_assertions(
                    resolved_manifest,
                    graph_snapshot=graph_snapshot,
                    request=request,
                )
            )

        for case in sorted_retrieval_cases:
            case_id = str(case["id"])
            relevant_ids = tuple(
                resolved_manifest.chunk(str(symbol)).chunk_id
                for symbol in case["expected_retrieval_chunk_ids"]
            )
            per_case[case_id] = run_one_snapshot_comparison(
                query=str(case["query"]),
                collection_ids=collection_scope,
                top_k=10,
                model_cls=TextChunk,
                graph_config=graph_config,
                timeout_ms=timeout_ms,
                prepare_embedding=lambda _query, case_id=case_id: prepared_embeddings[
                    case_id
                ],
                resolve_scope=resolved_scope,
                authorized_snapshot=borrowed_snapshot,
                collect_candidates=collect_hybrid_candidate_snapshot,
                load_graph=load_evaluation_graph,
                rank_graph=rank_authorized_graph_snapshot,
                materialize_and_rerank=strict_eval_materialize,
                app_config_getter=apps.get_app_config,
                relevant_chunk_ids=relevant_ids,
                validate_graph_snapshot=validate_live_graph,
            )
        expected_canonical_assertions = {
            (row.source_chunk_symbol, row.target_chunk_symbol)
            for row in resolved_manifest.canonical_identity_assertions
        }
        if observed_canonical_assertions != expected_canonical_assertions:
            raise ComparisonAborted(
                "live eval canonical projection did not prove fixture assertions"
            )

    extraction_projection_context = _build_production_extraction_context(
        manifest=resolved_manifest,
        selected_artifacts=selected_artifacts,
        manifest_rows=extraction_manifest_rows,
        ontology=live_ontology,
        extraction_settings=extraction_settings,
        filter_policy=live_filter_policy,
        resolution_config=live_resolution_config,
        canonical_assertions=tuple(sorted(observed_canonical_assertions)),
    )
    symbol_to_id = {
        symbol: binding.chunk_id
        for symbol, binding in resolved_manifest.chunks.items()
        if resolved_manifest.collections[binding.collection_symbol].authorized
    }

    artifact_report = [
        {
            "collection_id": int(row["collection_scope_id"]),
            "build_key": str(row["build_key"]),
            "source_hash": str(row["source_hash"]),
            "rebuild_request": str(row["rebuild_request_id"]),
        }
        for row in sorted(
            selected_artifacts,
            key=lambda item: int(item["collection_scope_id"]),
        )
    ]
    suite_signature = comparison_snapshot_signature(
        {
            "scope": {
                "collections": list(collection_scope),
                "documents": list(initial_scope.allowed_doc_ids),
            },
            "artifacts": artifact_report,
            "fixture_manifest_checksum": resolved_manifest.manifest_checksum,
            "extraction": dict(extraction_provenance),
            "embedding": dict(embedding_provenance),
            "reranker": dict(reranker_provenance),
            "database_attestation": database_attestation,
            "cases": [
                [case_id, result["comparison_snapshot_signature"]]
                for case_id, result in sorted(per_case.items())
            ],
        }
    )
    suite_seed_signature = comparison_snapshot_signature(
        {
            "case_seed_snapshots": [
                [case_id, result["seed_snapshot_signature"]]
                for case_id, result in sorted(per_case.items())
            ]
        }
    )
    versions = {
        "ontology": [
            {"version": version, "checksum": checksum}
            for version, checksum in sorted(
                {
                    (str(row["ontology_version"]), str(row["ontology_checksum"]))
                    for row in selected_artifacts
                }
            )
        ],
        "resolver": [
            {"version": version, "checksum": checksum}
            for version, checksum in sorted(
                {
                    (
                        str(row["resolver_version"]),
                        str(row["resolution_config_checksum"]),
                    )
                    for row in selected_artifacts
                }
            )
        ],
        "filter": [
            {"version": version, "checksum": checksum}
            for version, checksum in sorted(
                {
                    (
                        str(row["filter_policy_version"]),
                        str(row["filter_policy_checksum"]),
                    )
                    for row in selected_artifacts
                }
            )
        ],
    }

    arms: dict[str, dict[str, Any]] = {}
    cases_by_id = {str(case["id"]): case for case in retrieval_cases}
    for arm_name in _COMPARISON_ARMS:
        case_reports: dict[str, dict[str, Any]] = {}
        scored: list[Mapping[str, float | int]] = []
        latencies: list[float] = []
        graph_added_latencies: list[float] = []
        graph_versions: list[str] = []
        algorithm_signatures: set[str] = set()
        for case_id, result in sorted(per_case.items()):
            raw_arm = result["arms"][arm_name]
            fixture_case = cases_by_id[case_id]
            expected_ids = tuple(
                symbol_to_id[str(symbol)]
                for symbol in fixture_case["expected_retrieval_chunk_ids"]
            )
            ranked_ids = tuple(raw_arm["ranked_chunk_ids"])
            graph_ids = tuple(raw_arm["graph_chunk_ids"])
            metric = dict(
                score_ranked_retrieval(
                    expected_chunk_ids=expected_ids,
                    ranked_chunk_ids=ranked_ids,
                    k=10,
                    accessible_chunk_ids=frozenset(database_attestation["chunk_ids"]),
                    graph_chunk_ids=graph_ids,
                    citation_evidence_chunk_ids=frozenset(),
                    seed_chunk_ids=(),
                    mapped_seed_chunk_ids=frozenset(),
                    semantic_distances={},
                    latency_ms=float(raw_arm["latency_ms"]),
                    node_count=int(raw_arm["node_count"]),
                    edge_count=int(raw_arm["edge_count"]),
                )
            )
            metric["graph_hit_rate"] = float(raw_arm["graph_hit_rate"])
            metric["inaccessible_result_count"] = int(
                metric["inaccessible_result_count"]
            ) + int(raw_arm["inaccessible_result_count"])
            metric["citation_evidence_coverage"] = float(
                raw_arm["citation_evidence_coverage"]
            )
            metric["seed_coverage"] = float(raw_arm["seed_coverage"])
            metric["distance_2_novel_fraction"] = float(
                raw_arm["distance_2_novel_fraction"]
            )
            scored.append(metric)
            latencies.append(float(raw_arm["latency_ms"]))
            graph_added_latencies.append(float(raw_arm["graph_added_latency_ms"]))
            algorithm_signatures.add(str(raw_arm["algorithm_signature"]))
            if raw_arm["graph_version_signature"] is not None:
                graph_versions.append(str(raw_arm["graph_version_signature"]))
            distances = fixture_case.get("expected_min_semantic_distance", {})
            case_reports[case_id] = {
                "quality_tags": list(fixture_case.get("quality_tags", ())),
                "minimum_semantic_distance": max(distances.values(), default=0),
                "distance_2_relevant_hit": bool(
                    raw_arm.get("distance_2_relevant_hit", False)
                ),
                "recall_at_10": metric["recall_at_10"],
                "ndcg_at_10": metric["ndcg_at_10"],
            }
        if len(algorithm_signatures) != 1:
            raise ComparisonAborted(f"{arm_name} algorithm changed across cases")
        arm_algorithm = next(iter(algorithm_signatures))
        graph_version = (
            None
            if arm_name == "vector_only"
            else comparison_snapshot_signature(
                {"case_graph_versions": graph_versions, "arm": arm_name}
            )
        )
        arms[arm_name] = {
            "name": arm_name,
            "collection_scope": list(collection_scope),
            "comparison_snapshot_signature": suite_signature,
            "seed_snapshot_signature": suite_seed_signature,
            "fixture_checksum": fixture_signature,
            "fixture_manifest_checksum": resolved_manifest.manifest_checksum,
            "versions": dict(versions),
            "max_hops": {"vector_only": 0, "one_hop": 1, "ppr_v1": 2}[arm_name],
            "ppr_iterations": 0 if arm_name == "vector_only" else 8,
            "algorithm_signature": arm_algorithm,
            "graph_version_signature": graph_version,
            "metrics": {
                "recall_at_10": _mean([float(item["recall_at_10"]) for item in scored]),
                "mrr": _mean([float(item["mrr"]) for item in scored]),
                "ndcg_at_10": _mean([float(item["ndcg_at_10"]) for item in scored]),
                "graph_hit_rate": _mean(
                    [float(item["graph_hit_rate"]) for item in scored]
                ),
                "inaccessible_result_count": sum(
                    int(item["inaccessible_result_count"]) for item in scored
                ),
                "latency_p95_ms": _percentile_95(latencies),
                "graph_added_latency_p95_ms": _percentile_95(graph_added_latencies),
                "citation_evidence_coverage": _mean(
                    [float(item["citation_evidence_coverage"]) for item in scored]
                ),
                "seed_coverage": _mean(
                    [
                        float(per_case[case_id]["arms"][arm_name]["seed_coverage"])
                        for case_id in sorted(per_case)
                    ]
                ),
                "node_count": max(int(item["node_count"]) for item in scored),
                "edge_count": max(int(item["edge_count"]) for item in scored),
                "distance_2_novel_fraction": _mean(
                    [
                        float(
                            per_case[case_id]["arms"][arm_name][
                                "distance_2_novel_fraction"
                            ]
                        )
                        for case_id in sorted(per_case)
                    ]
                ),
            },
            "cases": case_reports,
        }

    production_extraction_report = evaluate_production_extraction_cases(
        extraction_cases,
        projection_context=extraction_projection_context,
    )
    extraction_metrics = dict(production_extraction_report["metrics"])
    graph_miss_observations = sum(
        int(result["fail_open_miss_observation_count"]) for result in per_case.values()
    )
    graph_error_observations = sum(
        int(result["fail_open_error_observation_count"]) for result in per_case.values()
    )
    bundle = {
        "schema_version": 1,
        "mode": "comparison",
        "eval_only": True,
        "collection_scope": list(collection_scope),
        "model": {
            "provider": extraction_settings.provider,
            "name": extraction_settings.model_id,
            "checkpoint": extraction_settings.model_revision,
        },
        "extraction": dict(extraction_provenance),
        "embedding": dict(embedding_provenance),
        "reranker": dict(reranker_provenance),
        "versions": versions,
        "artifacts": artifact_report,
        "comparison_snapshot_signature": suite_signature,
        "seed_snapshot_signature": suite_seed_signature,
        "fixture_checksum": fixture_signature,
        "fixture_manifest_checksum": resolved_manifest.manifest_checksum,
        "latency_budget_ms": float(timeout_ms),
        "extraction_metrics": extraction_metrics,
        "invariants": {
            "exact_baseline_on_graph_failure": all(
                result["exact_fail_open_parity"] is True for result in per_case.values()
            ),
            "deterministic_repeated_ppr": all(
                result["deterministic_repeated_ppr"] is True
                for result in per_case.values()
            ),
            "strict_local_reranking": True,
            "rerank_cache_enabled": False,
            "graph_miss_observations": graph_miss_observations,
            "graph_error_observations": graph_error_observations,
        },
        "arms": arms,
    }
    return validate_comparison_bundle(bundle)


def build_baseline_records(
    cases: Sequence[Mapping[str, Any]],
    injected_results: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Record vector IDs and classify security only from collection evidence."""
    injected_results = injected_results or {}
    records: list[Mapping[str, Any]] = []
    for case in sorted(cases, key=lambda item: str(item["id"])):
        case_id = str(case["id"])
        raw_results = injected_results.get(
            case_id, case.get("baseline_vector_result_ids")
        )
        if raw_results is None:
            record: dict[str, Any] = {
                "id": case_id,
                "reason": "no fixture-backed or injected vector results",
                "status": "SKIP",
            }
        else:
            id_collections: Mapping[str, Any] = MappingProxyType({})
            if isinstance(raw_results, Mapping):
                _require_fields(
                    raw_results, ("result_ids",), f"results for {case_id!r}"
                )
                result_ids = raw_results["result_ids"]
                if "id_collections" in raw_results:
                    id_collections = _require_mapping(
                        raw_results["id_collections"],
                        f"id_collections for {case_id!r}",
                    )
                    for result_key, collection_id in id_collections.items():
                        _require_nonempty_string(
                            result_key,
                            f"id_collections key for {case_id!r}",
                        )
                        _require_nonempty_string(
                            collection_id,
                            f"collection for result {result_key!r}",
                        )
            else:
                result_ids = raw_results
            if isinstance(result_ids, (str, bytes)) or not isinstance(
                result_ids, Sequence
            ):
                raise FixtureValidationError(
                    f"baseline IDs for {case_id!r} must be a list of valid IDs"
                )
            deduped: list[str | int] = []
            for item in result_ids:
                if not (
                    (isinstance(item, str) and item)
                    or (
                        isinstance(item, int)
                        and not isinstance(item, bool)
                        and item > 0
                    )
                ):
                    raise FixtureValidationError(
                        f"baseline IDs for {case_id!r} must be valid IDs"
                    )
                if item not in deduped:
                    deduped.append(item)
            chunk_collections = {
                chunk["chunk_id"]: document["collection_id"]
                for document in case.get("documents", ())
                for chunk in document.get("chunks", ())
            }
            accessible = set(case.get("accessible_collection_ids", ()))
            inaccessible: list[str | int] = []
            unresolved: list[str | int] = []
            for item in deduped:
                collection_id = (
                    chunk_collections.get(item)
                    if isinstance(item, str) and item in chunk_collections
                    else id_collections.get(str(item))
                )
                if collection_id is None:
                    unresolved.append(item)
                elif collection_id not in accessible:
                    inaccessible.append(item)
            if inaccessible:
                security_status = "LEAKAGE"
            elif unresolved:
                security_status = "UNKNOWN"
            else:
                security_status = "OK"
            record = {
                "id": case_id,
                "result_ids": tuple(deduped),
                "inaccessible_result_ids": tuple(inaccessible),
                "inaccessible_result_count": len(inaccessible),
                "unresolved_result_ids": tuple(unresolved),
                "unresolved_result_count": len(unresolved),
                "security_status": security_status,
                "status": "RECORDED",
            }
        records.append(MappingProxyType(record))
    return tuple(records)


def _load_injected_results(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return MappingProxyType({})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureValidationError(
            f"could not read injected retrieval results {path}: {exc}"
        ) from exc
    payload = _require_mapping(payload, "injected retrieval results")
    return _freeze(payload)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline knowledge-graph evaluation runner"
    )
    parser.add_argument(
        "--extraction-cases", type=Path, default=_DEFAULT_EXTRACTION_CASES_PATH
    )
    parser.add_argument(
        "--retrieval-cases", type=Path, default=_DEFAULT_RETRIEVAL_CASES_PATH
    )
    parser.add_argument(
        "--retrieval-results",
        type=Path,
        help=(
            "JSON mapping of case ID to result IDs, optionally with ID-to-collection "
            "evidence"
        ),
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Record existing vector-only result IDs without scoring",
    )
    parser.add_argument(
        "--mode",
        choices=("comparison",),
        help="Run the one-snapshot three-arm operator comparison",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Explicitly acknowledge the debug/test-only comparison bypass",
    )
    parser.add_argument(
        "--collection",
        action="append",
        type=int,
        default=[],
        help="Authorized positive collection primary key; repeat one to four times",
    )
    parser.add_argument(
        "--rebuild-request",
        action="append",
        default=[],
        help="Eval-only rebuild request UUID paired positionally with --collection",
    )
    parser.add_argument(
        "--fixture-manifest",
        type=Path,
        help="Resolved immutable synthetic-fixture manifest JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Atomic comparison JSON destination",
    )
    parser.add_argument(
        "--write-measured-gates",
        action="store_true",
        help="Write measured gate values from a validated comparison report",
    )
    parser.add_argument(
        "--verify-gates",
        action="store_true",
        help="Verify that all measured gates pass for a comparison report",
    )
    parser.add_argument(
        "--comparison-report",
        type=Path,
        help="Existing atomic comparison report used by gate workflows",
    )
    parser.add_argument(
        "--runbook",
        type=Path,
        default=_DEFAULT_RUNBOOK_PATH,
        help="Runbook whose measured-gate table is written or verified",
    )
    args = parser.parse_args(argv)
    try:
        action_count = sum(
            (
                bool(args.mode),
                bool(args.write_measured_gates),
                bool(args.verify_gates),
            )
        )
        if action_count > 1 or (args.baseline_only and action_count):
            raise ComparisonValidationError(
                "comparison, baseline, and gate actions are mutually exclusive"
            )
        if args.write_measured_gates or args.verify_gates:
            if args.comparison_report is None:
                raise ComparisonValidationError(
                    "gate workflow requires --comparison-report"
                )
            gates = (
                write_measured_gates(args.comparison_report, args.runbook)
                if args.write_measured_gates
                else verify_measured_gates(args.comparison_report, args.runbook)
            )
            print(
                json.dumps(
                    _thaw(gates),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if args.mode == "comparison":
            debug, environment = _runtime_eval_context()
            validate_eval_bypass(
                eval_only=args.eval_only,
                debug=debug,
                environment=environment,
            )
            evaluation_rebuild_requests = canonicalize_collection_requests(
                args.collection,
                args.rebuild_request,
            )
            collection_scope = tuple(
                collection_id
                for collection_id, _request_id in evaluation_rebuild_requests
            )
            if args.output is None:
                raise ComparisonValidationError(
                    "comparison mode requires an atomic --output path"
                )
            if args.fixture_manifest is None:
                raise ComparisonValidationError(
                    "comparison mode requires --fixture-manifest"
                )
            extraction_cases = load_extraction_cases(args.extraction_cases)
            retrieval_cases = load_retrieval_cases(args.retrieval_cases)
            bundle = _live_comparison_bundle(
                collection_scope=collection_scope,
                evaluation_rebuild_requests=evaluation_rebuild_requests,
                fixture_manifest_path=args.fixture_manifest,
                extraction_cases=extraction_cases,
                retrieval_cases=retrieval_cases,
                extraction_path=args.extraction_cases,
                retrieval_path=args.retrieval_cases,
            )
            bundle = validate_comparison_bundle(bundle)
            if bundle["collection_scope"] != list(collection_scope):
                raise ComparisonValidationError(
                    "live comparison returned an internal collection scope mismatch"
                )
            atomic_write_json(args.output, bundle)
            print(format_comparison_table(bundle))
            return 0
        # Validate both fixture classes for a meaningful invalid-fixture exit code.
        load_extraction_cases(args.extraction_cases)
        retrieval_cases = load_retrieval_cases(args.retrieval_cases)
        injected_results = _load_injected_results(args.retrieval_results)
        unknown_case_ids = set(injected_results) - {
            case["id"] for case in retrieval_cases
        }
        if unknown_case_ids:
            raise FixtureValidationError(
                f"unknown injected case IDs: {sorted(unknown_case_ids)!r}"
            )
        if args.baseline_only:
            payload = {
                "mode": "baseline-only",
                "records": build_baseline_records(retrieval_cases, injected_results),
            }
        else:
            payload = {
                "mode": "fixtures-validated",
                "retrieval_case_ids": [case["id"] for case in retrieval_cases],
            }
    except GateVerificationError as exc:
        print(
            json.dumps(
                {"error": str(exc), "status": "GATE_VERIFICATION_FAILED"},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 3
    except (
        FixtureValidationError,
        ComparisonValidationError,
        ComparisonAborted,
    ) as exc:
        status = (
            "INVALID_FIXTURE"
            if type(exc) is FixtureValidationError
            else "INVALID_COMPARISON"
        )
        print(
            json.dumps(
                {"error": str(exc), "status": status},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(_thaw(payload), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
