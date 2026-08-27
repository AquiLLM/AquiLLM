"""Environment-backed configuration for optional graph extraction workers.

This module deliberately has no Django or provider imports so web and task
registration processes can inspect graph configuration without loading an ML
runtime.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
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
DEFAULT_EXTRACTION_QUEUE = "knowledge-graph-extraction"
INVALID_EXTRACTION_QUEUE = "invalid-knowledge-graph-extraction"
MAX_EXTRACTION_QUEUE_LENGTH = 64
RESERVED_NON_GRAPH_QUEUES = frozenset({"celery", "memory-promotion"})
DEFAULT_EXTRACTION_CPU_UTILIZATION_PERCENT = 80
DEFAULT_GLINER2_CPU_THREADS_PER_WORKER = 8
MAX_EXTRACTION_WORKER_CONCURRENCY = 8

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_EXTRACTION_QUEUE = re.compile(
    rf"^[A-Za-z0-9][A-Za-z0-9._-]{{0,{MAX_EXTRACTION_QUEUE_LENGTH - 1}}}$"
)
_CpuCountGetter = Callable[[], int]
_ProcessExecutor = Callable[[str, tuple[str, ...], dict[str, str]], object]


class KnowledgeGraphConfigError(ValueError):
    """Raised when enabled graph extraction would use unsafe configuration."""


@dataclass(frozen=True, slots=True)
class ExtractionWorkerCpuLayout:
    """Bounded process and inference-thread allocation for one graph worker."""

    detected_cpu_count: int
    cpu_budget: int
    worker_concurrency: int
    intraop_threads: int


@dataclass(frozen=True, slots=True)
class ExtractionWorkerLaunch:
    """Validated process arguments and CPU-thread environment for Celery."""

    layout: ExtractionWorkerCpuLayout
    argv: tuple[str, ...]
    thread_environment: dict[str, str]


def resolve_extraction_worker_cpu_layout(
    *,
    cpu_count: int,
    source: Mapping[str, str] | None = None,
) -> ExtractionWorkerCpuLayout:
    """Scale extraction across CPUs while retaining capacity for other services."""

    if type(cpu_count) is not int or cpu_count < 1:
        raise KnowledgeGraphConfigError("graph worker CPU count must be positive")
    values = process_environ if source is None else source

    def optional_positive_int(name: str) -> int | None:
        raw_value = values.get(name)
        if raw_value is None or raw_value.strip().lower() == "auto":
            return None
        try:
            value = int(raw_value.strip())
        except (TypeError, ValueError) as exc:
            raise KnowledgeGraphConfigError(f"{name} must be auto or positive") from exc
        if value < 1:
            raise KnowledgeGraphConfigError(f"{name} must be auto or positive")
        return value

    worker_override = optional_positive_int("KG_EXTRACTION_WORKER_CONCURRENCY")
    thread_override = optional_positive_int("KG_GLINER2_CPU_THREADS")
    cpu_budget = max(
        1,
        cpu_count * DEFAULT_EXTRACTION_CPU_UTILIZATION_PERCENT // 100,
    )
    if worker_override is None and thread_override is None:
        worker_concurrency = min(
            MAX_EXTRACTION_WORKER_CONCURRENCY,
            max(
                1,
                (cpu_budget + DEFAULT_GLINER2_CPU_THREADS_PER_WORKER - 1)
                // DEFAULT_GLINER2_CPU_THREADS_PER_WORKER,
            ),
        )
        intraop_threads = max(
            1,
            min(
                DEFAULT_GLINER2_CPU_THREADS_PER_WORKER,
                cpu_budget // worker_concurrency,
            ),
        )
    elif worker_override is None:
        assert thread_override is not None
        intraop_threads = thread_override
        worker_concurrency = min(
            MAX_EXTRACTION_WORKER_CONCURRENCY,
            max(1, cpu_budget // intraop_threads),
        )
    elif thread_override is None:
        worker_concurrency = worker_override
        intraop_threads = max(
            1,
            min(
                DEFAULT_GLINER2_CPU_THREADS_PER_WORKER,
                cpu_budget // worker_concurrency,
            ),
        )
    else:
        worker_concurrency = worker_override
        intraop_threads = thread_override
    if worker_concurrency > MAX_EXTRACTION_WORKER_CONCURRENCY:
        raise KnowledgeGraphConfigError(
            "KG_EXTRACTION_WORKER_CONCURRENCY exceeds the safe process cap"
        )
    if worker_concurrency * intraop_threads > cpu_count:
        raise KnowledgeGraphConfigError(
            "graph worker process and thread overrides oversubscribe available CPUs"
        )
    return ExtractionWorkerCpuLayout(
        detected_cpu_count=cpu_count,
        cpu_budget=cpu_budget,
        worker_concurrency=worker_concurrency,
        intraop_threads=intraop_threads,
    )


def build_extraction_worker_launch(
    *,
    cpu_count: int,
    source: Mapping[str, str] | None = None,
) -> ExtractionWorkerLaunch:
    """Build a shell-free, validated launch specification for the graph worker."""

    values = process_environ if source is None else source
    layout = resolve_extraction_worker_cpu_layout(
        cpu_count=cpu_count,
        source=values,
    )
    queue = load_extraction_queue(values)
    threads = str(layout.intraop_threads)
    return ExtractionWorkerLaunch(
        layout=layout,
        argv=(
            "/opt/venv/bin/celery",
            "-A",
            "aquillm",
            "worker",
            "--loglevel=info",
            f"--queues={queue}",
            f"--concurrency={layout.worker_concurrency}",
            "--prefetch-multiplier=1",
            "--hostname=worker-knowledge-graph@%h",
        ),
        thread_environment={
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
        },
    )


def detect_available_cpu_count() -> int:
    """Return the process affinity width, falling back to the host CPU count."""

    affinity_getter = getattr(os, "sched_getaffinity", None)
    if callable(affinity_getter):
        try:
            affinity_count = len(affinity_getter(0))
        except OSError:
            affinity_count = 0
        if affinity_count > 0:
            return affinity_count
    return max(1, os.cpu_count() or 1)


def run_extraction_worker(
    *,
    source: Mapping[str, str] | None = None,
    cpu_count_getter: _CpuCountGetter | None = None,
    execute: _ProcessExecutor | None = None,
) -> object:
    """Replace this process with a CPU-scaled, isolated graph worker."""

    values = process_environ if source is None else source
    detected_cpu_count = (
        detect_available_cpu_count() if cpu_count_getter is None else cpu_count_getter()
    )
    launch = build_extraction_worker_launch(
        cpu_count=detected_cpu_count,
        source=values,
    )
    environment = dict(values)
    environment.update(launch.thread_environment)
    process_exec = os.execvpe if execute is None else execute
    return process_exec(launch.argv[0], launch.argv, environment)


def validate_extraction_queue(value: object) -> str:
    """Return one bounded literal Celery queue token or fail closed."""

    if (
        type(value) is not str
        or _EXTRACTION_QUEUE.fullmatch(value) is None
        or value in RESERVED_NON_GRAPH_QUEUES
    ):
        raise KnowledgeGraphConfigError(
            "KG_EXTRACTION_QUEUE must be one bounded literal queue token"
        )
    return value


def load_extraction_queue(source: Mapping[str, str] | None = None) -> str:
    """Load the exact producer/consumer queue without normalizing bad input."""

    values = process_environ if source is None else source
    return validate_extraction_queue(
        values.get("KG_EXTRACTION_QUEUE", DEFAULT_EXTRACTION_QUEUE)
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
