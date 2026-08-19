"""Bounded contract types and identities for the synthetic KG fixture."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, UUID, uuid5

from django.conf import settings

from .fixture_manifest import is_safe_huggingface_repo_id

FIXTURE_ID = "kg-task20-synthetic-v1"
FIXTURE_NAMESPACE = uuid5(NAMESPACE_URL, FIXTURE_ID)
VISIBLE_USERNAME = f"{FIXTURE_ID}-visible"
HIDDEN_USERNAME = f"{FIXTURE_ID}-hidden"
INPUT_TYPE = "search_document"
DIMENSIONS = 1024
PHYSICAL_BINDINGS = {
    "collection-policy-a": "authorized-a",
    "collection-policy-b": "authorized-b",
    "collection-public": "authorized-a",
    "collection-research-a": "authorized-c",
    "collection-research-b": "authorized-d",
    "collection-security-private": "hidden",
}
PHYSICAL_LABELS = tuple(sorted(set(PHYSICAL_BINDINGS.values())))
SHA_PATTERN = re.compile(r"[0-9a-f]{64}")
CHECKPOINT_PATTERN = re.compile(r"[0-9a-f]{40}")


class FixtureSeedError(RuntimeError):
    """A bounded, privacy-safe fixture command failure."""


@dataclass(frozen=True, slots=True)
class LogicalChunk:
    symbol: str
    text: str


@dataclass(frozen=True, slots=True)
class LogicalDocument:
    symbol: str
    collection_symbol: str
    title: str
    chunks: tuple[LogicalChunk, ...]


@dataclass(frozen=True, slots=True)
class LogicalFixture:
    extraction_cases: tuple
    retrieval_cases: tuple
    documents: dict[str, LogicalDocument]
    chunks: dict[str, tuple[str, str, int]]
    canonical: tuple[tuple[str, str], ...]
    inaccessible: tuple[tuple[str, str], ...]
    checksum: str


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    model: str
    checkpoint: str
    model_signature: str
    endpoint_signature: str


@dataclass(frozen=True, slots=True)
class FixtureSeedResult:
    fixture_checksum: str
    manifest_checksum: str
    manifest_path: Path
    authorized_scope: tuple[tuple[int, UUID], ...]
    collection_count: int
    document_count: int
    chunk_count: int


def safe_token(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and value == value.strip()
        and len(value) <= 256
        and not any(character in value for character in "\x00\r\n")
    )


def require_safe_environment() -> tuple[str, str]:
    if getattr(settings, "DEBUG", False) is not True:
        raise FixtureSeedError("fixture command requires DEBUG=True")
    required = {
        "KG_EVAL_BYPASS_ALLOWED": "1",
        "KG_BUILD_ENABLED": "0",
        "KG_OVERLAY_ENABLED": "0",
    }
    for name, expected in required.items():
        if os.environ.get(name) != expected:
            raise FixtureSeedError(f"{name} must equal {expected}")
    if os.environ.get("COHERE_KEY") != "":
        raise FixtureSeedError("COHERE_KEY must be explicitly blank")
    if os.environ.get("APP_EMBED_API_KEY") != "EMPTY":
        raise FixtureSeedError("local embedding API key must equal EMPTY")
    if os.environ.get("APP_EMBED_DIMS") != str(DIMENSIONS):
        raise FixtureSeedError("APP_EMBED_DIMS must equal 1024")
    if os.environ.get("APP_EMBED_ALLOW_DIMENSIONS_OVERRIDE") != "0":
        raise FixtureSeedError("embedding dimensions override must remain disabled")
    try:
        parsed = urlsplit(os.environ.get("APP_EMBED_BASE_URL", ""))
        port = parsed.port
    except ValueError as error:
        raise FixtureSeedError(
            "embedding endpoint must be strict local vllm_embed"
        ) from error
    if (
        parsed.scheme != "http"
        or parsed.hostname != "vllm_embed"
        or port != 8000
        or parsed.path.rstrip("/") != "/v1"
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
    ):
        raise FixtureSeedError("embedding endpoint must be strict local vllm_embed")
    model = os.environ.get("APP_EMBED_MODEL")
    checkpoint = os.environ.get("APP_EMBED_MODEL_REVISION")
    tokenizer_checkpoint = os.environ.get("APP_EMBED_TOKENIZER_REVISION")
    code_checkpoint = os.environ.get("APP_EMBED_CODE_REVISION")
    if not is_safe_huggingface_repo_id(model):
        raise FixtureSeedError("embedding model identity is invalid")
    if not safe_token(checkpoint) or CHECKPOINT_PATTERN.fullmatch(checkpoint) is None:
        raise FixtureSeedError("embedding checkpoint identity is invalid")
    if (
        not safe_token(tokenizer_checkpoint)
        or CHECKPOINT_PATTERN.fullmatch(tokenizer_checkpoint) is None
        or tokenizer_checkpoint != checkpoint
    ):
        raise FixtureSeedError("embedding tokenizer checkpoint identity is invalid")
    if (
        not safe_token(code_checkpoint)
        or CHECKPOINT_PATTERN.fullmatch(code_checkpoint) is None
        or code_checkpoint != checkpoint
    ):
        raise FixtureSeedError("embedding code checkpoint identity is invalid")
    return model, checkpoint


def manifest_path(value: object, *, must_exist: bool) -> Path:
    if (
        type(value) is not str
        or not value
        or len(value) > 4_096
        or any(character in value for character in "\x00\r\n")
    ):
        raise FixtureSeedError("fixture manifest path is invalid")
    raw = Path(value)
    if raw.is_symlink():
        raise FixtureSeedError("fixture manifest must not be a symbolic link")
    if must_exist:
        try:
            resolved = raw.resolve(strict=True)
        except OSError as error:
            raise FixtureSeedError("fixture manifest does not exist") from error
        if not resolved.is_file():
            raise FixtureSeedError("fixture manifest must be a regular file")
        return resolved
    try:
        parent = raw.parent.resolve(strict=True)
    except OSError as error:
        raise FixtureSeedError("fixture manifest parent does not exist") from error
    if not parent.is_dir() or raw.name in ("", ".", ".."):
        raise FixtureSeedError("fixture manifest destination is invalid")
    return parent / raw.name


def document_ids(logical: LogicalFixture) -> dict[str, UUID]:
    return {
        symbol: uuid5(FIXTURE_NAMESPACE, f"document:{symbol}")
        for symbol in logical.documents
    }


def request_ids() -> dict[str, UUID]:
    return {
        label: uuid5(FIXTURE_NAMESPACE, f"rebuild:{label}")
        for label in PHYSICAL_LABELS
        if label != "hidden"
    }
