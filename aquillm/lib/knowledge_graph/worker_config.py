"""CPU allocation, launch arguments, and queue validation for graph workers."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from os import environ as process_environ

from .config_defaults import (
    _EXTRACTION_QUEUE,
    DEFAULT_EXTRACTION_CPU_UTILIZATION_PERCENT,
    DEFAULT_EXTRACTION_QUEUE,
    DEFAULT_GLINER2_CPU_THREADS_PER_WORKER,
    MAX_EXTRACTION_WORKER_CONCURRENCY,
    RESERVED_NON_GRAPH_QUEUES,
    KnowledgeGraphConfigError,
)

_CpuCountGetter = Callable[[], int]
_ProcessExecutor = Callable[[str, tuple[str, ...], dict[str, str]], object]

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
