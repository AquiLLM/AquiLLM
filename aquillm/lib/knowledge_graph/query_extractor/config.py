"""Strict, provider-neutral configuration for the query extractor boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from lib.knowledge_graph.retrieval_config import (
    QUERY_EXTRACTOR_MODEL,
    QUERY_EXTRACTOR_MODEL_REVISION,
    SecretSetting,
)

from .contracts import (
    MAX_QUERY_CODE_POINTS,
    MAX_QUERY_REQUEST_BODY_BYTES,
    MAX_QUERY_RESPONSE_BODY_BYTES,
    MAX_QUERY_SPANS,
    MAX_QUERY_UTF8_BYTES,
    QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM,
    QUERY_EXTRACTION_RESPONSE_SCHEMA_VERSION,
)

_DIGEST = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,255}")


class QueryExtractorConfigError(ValueError):
    """Raised when the extractor client or sidecar configuration is unsafe."""


@dataclass(frozen=True, slots=True)
class QueryExtractorSettings:
    url: str = field(repr=False)
    bearer_token: SecretSetting = field(repr=False)
    model_identifier: str
    model_revision: str
    build_hash: str
    schema_version: str
    schema_checksum: str
    ontology_path: Path
    ontology_checksum: str
    timeout_ms: int
    max_query_utf8_bytes: int
    max_query_code_points: int
    max_spans: int
    max_request_body_bytes: int = MAX_QUERY_REQUEST_BODY_BYTES
    max_response_body_bytes: int = MAX_QUERY_RESPONSE_BODY_BYTES


def _error(key: str, reason: str) -> QueryExtractorConfigError:
    return QueryExtractorConfigError(f"{key} {reason}")


def _text(env: Mapping[str, str], key: str) -> str:
    value = env.get(key)
    if type(value) is not str:
        raise _error(key, "must be an exact string")
    if (
        not value
        or len(value) > 4096
        or value != value.strip()
        or any(
            ord(character) < 32
            or ord(character) == 127
            or 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
    ):
        raise _error(key, "must be nonempty bounded canonical text")
    return value


def _integer(env: Mapping[str, str], key: str, *, minimum: int, maximum: int) -> int:
    raw = _text(env, key)
    if not raw.isascii() or not raw.isdecimal() or (len(raw) > 1 and raw[0] == "0"):
        raise _error(key, "must be a canonical decimal integer")
    value = int(raw)
    if not minimum <= value <= maximum:
        raise _error(key, "is outside the supported range")
    return value


def _validate_url(value: str) -> None:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise _error("KG_QUERY_EXTRACTOR_URL", "must be a canonical HTTP URL") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not value.startswith(f"{parsed.scheme}://")
        or not parsed.hostname
        or parsed.hostname != parsed.hostname.lower()
        or parsed.netloc != parsed.netloc.lower()
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "\\" in parsed.path
        or not value.isascii()
        or port == 0
    ):
        raise _error("KG_QUERY_EXTRACTOR_URL", "must be a canonical HTTP URL")


def _digest(env: Mapping[str, str], key: str) -> str:
    value = _text(env, key)
    if _DIGEST.fullmatch(value) is None:
        raise _error(key, "must be a lowercase SHA-256 digest")
    return value


def load_query_extractor_settings(env: Mapping[str, str]) -> QueryExtractorSettings:
    """Load exact client/sidecar identity and resource caps from ``env``."""

    if not isinstance(env, Mapping):
        raise QueryExtractorConfigError("configuration source must be a mapping")
    url = _text(env, "KG_QUERY_EXTRACTOR_URL")
    _validate_url(url)
    token = _text(env, "KG_QUERY_EXTRACTOR_BEARER_TOKEN")
    model = _text(env, "KG_QUERY_EXTRACTOR_MODEL")
    revision = _text(env, "KG_QUERY_EXTRACTOR_MODEL_REVISION")
    schema_version = _text(env, "KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_VERSION")
    schema_checksum = _digest(env, "KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_CHECKSUM")
    if model != QUERY_EXTRACTOR_MODEL or _MODEL.fullmatch(model) is None:
        raise _error("KG_QUERY_EXTRACTOR_MODEL", "does not match the frozen model")
    if (
        revision != QUERY_EXTRACTOR_MODEL_REVISION
        or _REVISION.fullmatch(revision) is None
    ):
        raise _error(
            "KG_QUERY_EXTRACTOR_MODEL_REVISION", "does not match the frozen revision"
        )
    if schema_version != QUERY_EXTRACTION_RESPONSE_SCHEMA_VERSION:
        raise _error(
            "KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_VERSION",
            "does not match the frozen schema",
        )
    if schema_checksum != QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM:
        raise _error(
            "KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_CHECKSUM",
            "does not match the frozen schema",
        )
    ontology_path = Path(_text(env, "KG_QUERY_EXTRACTOR_ONTOLOGY_PATH"))
    return QueryExtractorSettings(
        url=url,
        bearer_token=SecretSetting(token),
        model_identifier=model,
        model_revision=revision,
        build_hash=_digest(env, "KG_QUERY_EXTRACTOR_BUILD_HASH"),
        schema_version=schema_version,
        schema_checksum=schema_checksum,
        ontology_path=ontology_path,
        ontology_checksum=_digest(env, "KG_QUERY_EXTRACTOR_ONTOLOGY_CHECKSUM"),
        timeout_ms=_integer(
            env, "KG_QUERY_EXTRACTOR_TIMEOUT_MS", minimum=10, maximum=1000
        ),
        max_query_utf8_bytes=_integer(
            env, "KG_QUERY_MAX_BYTES", minimum=1, maximum=MAX_QUERY_UTF8_BYTES
        ),
        max_query_code_points=_integer(
            env,
            "KG_QUERY_MAX_CODEPOINTS",
            minimum=1,
            maximum=MAX_QUERY_CODE_POINTS,
        ),
        max_spans=_integer(
            env, "KG_QUERY_MAX_SPANS", minimum=1, maximum=MAX_QUERY_SPANS
        ),
    )


__all__ = [
    "QueryExtractorConfigError",
    "QueryExtractorSettings",
    "load_query_extractor_settings",
]
