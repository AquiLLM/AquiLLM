"""Environment-backed configuration for optional graph extraction workers.

This module deliberately has no Django or provider imports so web and task
registration processes can inspect graph configuration without loading an ML
runtime.
"""

from __future__ import annotations

import os as os
from collections.abc import Mapping
from dataclasses import dataclass
from os import environ as process_environ
from pathlib import Path

from .config_defaults import (
    _FALSE_VALUES,
    _IMMUTABLE_REVISION,
    _TRUE_VALUES,
    DEFAULT_ARTIFACT_KEEP_SUPERSEDED,
    DEFAULT_ARTIFACT_RETENTION_DAYS,
    DEFAULT_EXTRACTOR_PROVIDER,
    DEFAULT_GLINER2_BATCH_SIZE,
    DEFAULT_GLINER2_CACHE_DIR,
    DEFAULT_GLINER2_DEVICE,
    DEFAULT_GLINER2_MAX_BATCH_CHARACTERS,
    DEFAULT_GLINER2_MODEL,
    DEFAULT_GLINER2_REVISION,
    KnowledgeGraphConfigError,
)
from .config_defaults import (
    DEFAULT_EXTRACTION_CPU_UTILIZATION_PERCENT as DEFAULT_EXTRACTION_CPU_UTILIZATION_PERCENT,  # noqa: E501
)
from .config_defaults import (
    DEFAULT_EXTRACTION_QUEUE as DEFAULT_EXTRACTION_QUEUE,
)
from .config_defaults import (
    DEFAULT_GLINER2_CPU_THREADS_PER_WORKER as DEFAULT_GLINER2_CPU_THREADS_PER_WORKER,
)
from .config_defaults import (
    INVALID_EXTRACTION_QUEUE as INVALID_EXTRACTION_QUEUE,
)
from .config_defaults import (
    MAX_EXTRACTION_QUEUE_LENGTH as MAX_EXTRACTION_QUEUE_LENGTH,
)
from .config_defaults import (
    MAX_EXTRACTION_WORKER_CONCURRENCY as MAX_EXTRACTION_WORKER_CONCURRENCY,
)
from .config_defaults import (
    RESERVED_NON_GRAPH_QUEUES as RESERVED_NON_GRAPH_QUEUES,
)
from .worker_config import (
    ExtractionWorkerCpuLayout as ExtractionWorkerCpuLayout,
)
from .worker_config import (
    ExtractionWorkerLaunch as ExtractionWorkerLaunch,
)
from .worker_config import (
    build_extraction_worker_launch as build_extraction_worker_launch,
)
from .worker_config import (
    detect_available_cpu_count as detect_available_cpu_count,
)
from .worker_config import (
    load_extraction_queue as load_extraction_queue,
)
from .worker_config import (
    resolve_extraction_worker_cpu_layout as resolve_extraction_worker_cpu_layout,
)
from .worker_config import (
    run_extraction_worker as run_extraction_worker,
)
from .worker_config import (
    validate_extraction_queue as validate_extraction_queue,
)


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
