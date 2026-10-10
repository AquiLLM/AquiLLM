"""Response-bound receipts. Declarations and model echoes are not attestations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from os import getenv
import re
import struct
from typing import Any

from .utils import EmbeddingContractError, fit_embedding_dims, validate_embedding


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    provenance: dict[str, Any]


def input_digest(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def vector_digest(vector) -> str:
    """Hash pgvector's float32 representation, including after a DB round trip."""
    values = [float(value) for value in vector]
    validate_embedding(values)
    try:
        packed = struct.pack(f"!{len(values)}f", *values)
        validate_embedding(list(struct.unpack(f"!{len(values)}f", packed)))
    except (OverflowError, struct.error) as exc:
        raise EmbeddingContractError(
            "Embedding is not representable in storage"
        ) from exc
    return sha256(packed).hexdigest()


def make_result(
    vector,
    *,
    provider,
    route,
    role,
    prepared_input,
    original_input,
    model,
    response_model=None,
) -> EmbeddingResult:
    validate_embedding(vector)
    binding = vector_digest(vector)
    return EmbeddingResult(
        list(vector),
        {
            "schema_version": 1,
            "provider": provider,
            "route": route,
            "role": role,
            "prepared_input_sha256": input_digest(prepared_input),
            "input_transformations": (
                ["character-truncation"] if prepared_input != original_input else []
            ),
            "declared_identity": {
                "model": model,
                "revision": (
                    getenv("APP_EMBED_MODEL_REVISION") or None
                    if provider == "local-openai"
                    else None
                ),
                "precision": (
                    getenv("APP_EMBED_PRECISION") or None
                    if provider == "local-openai"
                    else None
                ),
            },
            "observed_identity": {
                "model": response_model if type(response_model) is str else None,
                "revision": None,
                "precision": None,
            },
            "dimensions": {
                "raw": len(vector),
                "fitted": len(vector),
                "adaptation": "none",
            },
            "raw_vector_sha256": binding,
            "vector_sha256": binding,
        },
    )


def fit_result(result: EmbeddingResult) -> EmbeddingResult:
    vector = fit_embedding_dims(result.vector)
    receipt = deepcopy(result.provenance)
    raw = receipt["dimensions"]["raw"]
    fitted = len(vector)
    receipt["dimensions"] = {
        "raw": raw,
        "fitted": fitted,
        "adaptation": (
            "none" if raw == fitted else "pad-zero" if raw < fitted else "truncate"
        ),
    }
    receipt["vector_sha256"] = vector_digest(vector)
    return EmbeddingResult(vector, receipt)


def valid_provenance(vector, provenance):
    """Keep a detached receipt only when its version and storage binding match."""
    if vector is None or type(provenance) is not dict:
        return None
    try:
        required = {
            "schema_version",
            "provider",
            "route",
            "role",
            "prepared_input_sha256",
            "input_transformations",
            "declared_identity",
            "observed_identity",
            "dimensions",
            "raw_vector_sha256",
            "vector_sha256",
        }
        if (
            set(provenance) != required
            or type(provenance["schema_version"]) is not int
            or provenance["schema_version"] != 1
        ):
            return None
        routes = {
            "local-openai": {
                "openai-embeddings",
                "vllm-multi-modal-data",
                "vllm-openai-content",
            },
            "cohere": {"cohere-embed"},
        }
        if provenance["route"] not in routes.get(provenance["provider"], set()):
            return None
        if provenance["role"] not in {
            "search_document",
            "search_query",
            "classification",
            "clustering",
        }:
            return None
        if provenance["input_transformations"] not in ([], ["character-truncation"]):
            return None
        for key in ("prepared_input_sha256", "raw_vector_sha256", "vector_sha256"):
            if (
                type(provenance[key]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", provenance[key]) is None
            ):
                return None
        for key in ("declared_identity", "observed_identity"):
            identity = provenance[key]
            if type(identity) is not dict or set(identity) != {
                "model",
                "revision",
                "precision",
            }:
                return None
            if any(
                value is not None and type(value) is not str
                for value in identity.values()
            ):
                return None
        # Schema v1 providers expose model echoes, never revision/precision proof.
        if (
            provenance["observed_identity"]["revision"] is not None
            or provenance["observed_identity"]["precision"] is not None
        ):
            return None
        dimensions = provenance["dimensions"]
        if set(dimensions) != {"raw", "fitted", "adaptation"}:
            return None
        raw, fitted = dimensions["raw"], dimensions["fitted"]
        if (
            type(raw) is not int
            or type(fitted) is not int
            or raw < 1
            or fitted != len(vector)
        ):
            return None
        adaptation = (
            "none" if raw == fitted else "pad-zero" if raw < fitted else "truncate"
        )
        if dimensions["adaptation"] != adaptation:
            return None
        if provenance["vector_sha256"] != vector_digest(vector):
            return None
        return deepcopy(provenance)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
