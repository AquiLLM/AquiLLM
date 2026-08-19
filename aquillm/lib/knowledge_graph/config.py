"""Environment-backed configuration for optional graph extraction workers.

This module deliberately has no Django or provider imports so web and task
registration processes can inspect graph configuration without loading an ML
runtime.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from os import environ as process_environ
from pathlib import Path

DEFAULT_EXTRACTOR_PROVIDER = "gliner2_local"
DEFAULT_GLINER2_MODEL = "fastino/gliner2-base-v1"
DEFAULT_GLINER2_REVISION = "8437ba583a733d87f56ae902f3b197934eedd58e"
DEFAULT_GLINER2_DEVICE = "cpu"
DEFAULT_GLINER2_BATCH_SIZE = 8
DEFAULT_GLINER2_MAX_BATCH_CHARACTERS = 64_000
DEFAULT_GLINER2_CACHE_DIR = Path("/root/.cache/huggingface")
DEFAULT_ARTIFACT_RETENTION_DAYS = 30
DEFAULT_ARTIFACT_KEEP_SUPERSEDED = 2

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")


class KnowledgeGraphConfigError(ValueError):
    """Raised when enabled graph extraction would use unsafe configuration."""


@dataclass(frozen=True, slots=True)
class ExtractionSettings:
    """Provider-neutral settings needed to construct an extraction backend."""

    build_enabled: bool
    provider: str
    model_id: str
    model_revision: str
    device: str
    batch_size: int
    max_batch_characters: int
    cache_dir: Path
    local_files_only: bool
    fail_open: bool

    def __post_init__(self) -> None:
        if self.build_enabled and not _IMMUTABLE_REVISION.fullmatch(
            self.model_revision
        ):
            raise KnowledgeGraphConfigError(
                "KG_GLINER2_REVISION must be an immutable 40-character commit "
                "revision when KG_BUILD_ENABLED is true"
            )


@dataclass(frozen=True, slots=True)
class RetentionSettings:
    """Conservative graph-artifact retention settings."""

    retention_days: int = DEFAULT_ARTIFACT_RETENTION_DAYS
    keep_superseded: int = DEFAULT_ARTIFACT_KEEP_SUPERSEDED

    def __post_init__(self) -> None:
        if type(self.retention_days) is not int or self.retention_days < 1:
            raise KnowledgeGraphConfigError("retention_days must be positive")
        if type(self.keep_superseded) is not int or self.keep_superseded < 0:
            raise KnowledgeGraphConfigError("keep_superseded must be nonnegative")


def _parse_bool(source: Mapping[str, str], key: str, *, default: bool) -> bool:
    raw_value = source.get(key)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def _parse_positive_int(source: Mapping[str, str], key: str, *, default: int) -> int:
    raw_value = source.get(key)
    if raw_value is None:
        return default
    try:
        value = int(raw_value.strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _parse_nonnegative_int(source: Mapping[str, str], key: str, *, default: int) -> int:
    raw_value = source.get(key)
    if raw_value is None:
        return default
    try:
        value = int(raw_value.strip())
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _text_or_default(source: Mapping[str, str], key: str, default: str) -> str:
    raw_value = source.get(key)
    if raw_value is None:
        return default
    return raw_value.strip() or default


def load_extraction_settings(
    source: Mapping[str, str] | None = None,
) -> ExtractionSettings:
    """Read extraction settings, validating immutable revisions when enabled.

    Invalid booleans and batch sizes use their safe defaults. Explicitly empty
    revision values remain empty so enabling builds cannot accidentally turn an
    unpinned model into the default checkpoint.
    """

    values = process_environ if source is None else source
    build_enabled = _parse_bool(values, "KG_BUILD_ENABLED", default=False)
    provider = _text_or_default(
        values, "KG_EXTRACTOR_PROVIDER", DEFAULT_EXTRACTOR_PROVIDER
    )
    model_id = _text_or_default(values, "KG_GLINER2_MODEL", DEFAULT_GLINER2_MODEL)

    if "KG_GLINER2_REVISION" in values:
        model_revision = values["KG_GLINER2_REVISION"].strip()
    elif model_id == DEFAULT_GLINER2_MODEL:
        model_revision = DEFAULT_GLINER2_REVISION
    else:
        model_revision = ""

    device = _text_or_default(values, "KG_GLINER2_DEVICE", DEFAULT_GLINER2_DEVICE)
    batch_size = _parse_positive_int(
        values, "KG_GLINER2_BATCH_SIZE", default=DEFAULT_GLINER2_BATCH_SIZE
    )
    max_batch_characters = _parse_positive_int(
        values,
        "KG_GLINER2_MAX_BATCH_CHARACTERS",
        default=DEFAULT_GLINER2_MAX_BATCH_CHARACTERS,
    )
    cache_dir = Path(
        _text_or_default(
            values,
            "KG_GLINER2_CACHE_DIR",
            str(DEFAULT_GLINER2_CACHE_DIR),
        )
    )
    local_files_only = _parse_bool(values, "KG_GLINER2_LOCAL_FILES_ONLY", default=False)
    fail_open = _parse_bool(values, "KG_EXTRACTOR_FAIL_OPEN", default=True)

    return ExtractionSettings(
        build_enabled=build_enabled,
        provider=provider,
        model_id=model_id,
        model_revision=model_revision,
        device=device,
        batch_size=batch_size,
        max_batch_characters=max_batch_characters,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        fail_open=fail_open,
    )


def get_build_enabled(source: Mapping[str, str] | None = None) -> bool:
    return load_extraction_settings(source).build_enabled


def get_extractor_provider(source: Mapping[str, str] | None = None) -> str:
    return load_extraction_settings(source).provider


def get_extractor_model(source: Mapping[str, str] | None = None) -> str:
    return load_extraction_settings(source).model_id


def get_extractor_revision(source: Mapping[str, str] | None = None) -> str:
    return load_extraction_settings(source).model_revision


def get_extractor_device(source: Mapping[str, str] | None = None) -> str:
    return load_extraction_settings(source).device


def get_extractor_batch_size(source: Mapping[str, str] | None = None) -> int:
    return load_extraction_settings(source).batch_size


def get_extractor_max_batch_characters(
    source: Mapping[str, str] | None = None,
) -> int:
    return load_extraction_settings(source).max_batch_characters


def get_extractor_cache_dir(source: Mapping[str, str] | None = None) -> Path:
    return load_extraction_settings(source).cache_dir


def get_extractor_local_files_only(source: Mapping[str, str] | None = None) -> bool:
    return load_extraction_settings(source).local_files_only


def get_extractor_fail_open(source: Mapping[str, str] | None = None) -> bool:
    return load_extraction_settings(source).fail_open


def get_eval_bypass_allowed(source: Mapping[str, str] | None = None) -> bool:
    """Fail-closed parse of the explicit evaluation-only bypass switch."""

    values = process_environ if source is None else source
    return _parse_bool(values, "KG_EVAL_BYPASS_ALLOWED", default=False)


def load_retention_settings(
    source: Mapping[str, str] | None = None,
) -> RetentionSettings:
    values = process_environ if source is None else source
    return RetentionSettings(
        retention_days=_parse_positive_int(
            values,
            "KG_ARTIFACT_RETENTION_DAYS",
            default=DEFAULT_ARTIFACT_RETENTION_DAYS,
        ),
        keep_superseded=_parse_nonnegative_int(
            values,
            "KG_ARTIFACT_KEEP_SUPERSEDED",
            default=DEFAULT_ARTIFACT_KEEP_SUPERSEDED,
        ),
    )
