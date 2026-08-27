from __future__ import annotations

import pytest

from lib.knowledge_graph import config


def test_cpu_layout_scales_twenty_cores_without_oversubscribing() -> None:
    """Catch regressions that collapse the graph worker back to one CPU."""

    layout = config.resolve_extraction_worker_cpu_layout(cpu_count=20, source={})

    assert layout.worker_concurrency == 2
    assert layout.intraop_threads == 8
    assert layout.cpu_budget == 16
    assert layout.worker_concurrency * layout.intraop_threads <= layout.cpu_budget


@pytest.mark.parametrize(
    ("cpu_count", "expected_budget", "expected_workers", "expected_threads"),
    (
        (1, 1, 1, 1),
        (4, 3, 1, 3),
        (8, 6, 1, 6),
        (16, 12, 2, 6),
        (32, 25, 4, 6),
    ),
)
def test_cpu_layout_scales_with_the_container_cpu_allocation(
    cpu_count: int,
    expected_budget: int,
    expected_workers: int,
    expected_threads: int,
) -> None:
    layout = config.resolve_extraction_worker_cpu_layout(
        cpu_count=cpu_count,
        source={},
    )

    assert (
        layout.cpu_budget,
        layout.worker_concurrency,
        layout.intraop_threads,
    ) == (expected_budget, expected_workers, expected_threads)


def test_cpu_layout_honors_explicit_process_and_thread_overrides() -> None:
    layout = config.resolve_extraction_worker_cpu_layout(
        cpu_count=20,
        source={
            "KG_EXTRACTION_WORKER_CONCURRENCY": "4",
            "KG_GLINER2_CPU_THREADS": "4",
        },
    )

    assert layout.worker_concurrency == 4
    assert layout.intraop_threads == 4


@pytest.mark.parametrize(
    "source",
    (
        {"KG_EXTRACTION_WORKER_CONCURRENCY": "0"},
        {"KG_GLINER2_CPU_THREADS": "not-an-integer"},
        {
            "KG_EXTRACTION_WORKER_CONCURRENCY": "3",
            "KG_GLINER2_CPU_THREADS": "8",
        },
    ),
)
def test_cpu_layout_rejects_invalid_or_oversubscribed_overrides(
    source: dict[str, str],
) -> None:
    with pytest.raises(config.KnowledgeGraphConfigError):
        config.resolve_extraction_worker_cpu_layout(cpu_count=20, source=source)


def test_cpu_detection_honors_container_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config.os,
        "sched_getaffinity",
        lambda _pid: frozenset(range(6)),
        raising=False,
    )

    assert config.detect_available_cpu_count() == 6


def test_worker_launch_replaces_inherited_single_thread_limits() -> None:
    """Catch a global OMP=1 setting silently serializing GLiNER2 again."""

    launch = config.build_extraction_worker_launch(
        cpu_count=20,
        source={
            "KG_EXTRACTION_QUEUE": "graph-fast-path",
            "OMP_NUM_THREADS": "1",
        },
    )

    assert launch.argv == (
        "/opt/venv/bin/celery",
        "-A",
        "aquillm",
        "worker",
        "--loglevel=info",
        "--queues=graph-fast-path",
        "--concurrency=2",
        "--prefetch-multiplier=1",
        "--hostname=worker-knowledge-graph@%h",
    )
    assert launch.thread_environment == {
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
    }


def test_worker_runtime_executes_celery_with_the_scaled_environment() -> None:
    observed: list[tuple[str, tuple[str, ...], dict[str, str]]] = []

    config.run_extraction_worker(
        source={
            "KG_EXTRACTION_QUEUE": "graph-fast-path",
            "OMP_NUM_THREADS": "1",
            "DATABASE_SENTINEL": "preserved",
        },
        cpu_count_getter=lambda: 20,
        execute=lambda executable, argv, environment: observed.append(
            (executable, tuple(argv), dict(environment))
        ),
    )

    assert len(observed) == 1
    executable, argv, environment = observed[0]
    assert executable == "/opt/venv/bin/celery"
    assert "--concurrency=2" in argv
    assert environment["OMP_NUM_THREADS"] == "8"
    assert environment["DATABASE_SENTINEL"] == "preserved"
