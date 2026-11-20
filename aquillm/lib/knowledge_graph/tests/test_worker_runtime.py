from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lib.knowledge_graph import cgroup_cpu, config


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


@pytest.fixture
def cgroup_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "cgroup"
    root.mkdir()
    membership = tmp_path / "membership"
    membership.write_text("0::/\n")
    monkeypatch.setattr(cgroup_cpu, "_CGROUP_ROOT", root)
    monkeypatch.setattr(cgroup_cpu, "_CGROUP_MEMBERSHIP", membership)
    monkeypatch.setattr(cgroup_cpu, "_MOUNTINFO", tmp_path / "mountinfo")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        config.os, "sched_getaffinity", lambda _pid: set(range(20)), raising=False
    )
    return root, membership


def test_cpu_detection_honors_container_affinity(
    monkeypatch: pytest.MonkeyPatch,
    cgroup_files: tuple[Path, Path],
) -> None:
    root, _ = cgroup_files
    (root / "cpu.max").write_text("1200000 100000\n")
    monkeypatch.setattr(
        config.os,
        "sched_getaffinity",
        lambda _pid: frozenset(range(6)),
        raising=False,
    )

    assert config.detect_available_cpu_count() == 6


def test_worker_launch_uses_container_quota(cgroup_files: tuple[Path, Path]) -> None:
    root, _ = cgroup_files
    (root / "cpu.max").write_text("200000 100000\n")
    assert config.detect_available_cpu_count() == 2
    argv, env = config.run_extraction_worker(
        source={}, execute=lambda _exe, argv, env: (argv, env)
    )
    assert "--concurrency=1" in argv
    assert env["OMP_NUM_THREADS"] == "1"
    _, env = config.run_extraction_worker(
        source={"KG_GLINER2_CPU_THREADS": "2"},
        execute=lambda _exe, argv, env: (argv, env),
    )
    assert env["OMP_NUM_THREADS"] == "2"


@pytest.mark.parametrize("quota, expected", [("275000 100000", 2), ("50000 100000", 1)])
def test_cpu_detection_observes_constrained_v2_parent(
    cgroup_files: tuple[Path, Path], quota: str, expected: int
) -> None:
    root, membership = cgroup_files
    membership.write_text("0::/tenant/worker\n")
    leaf = root / "tenant" / "worker"
    leaf.mkdir(parents=True)
    (leaf / "cpu.max").write_text("max 100000")
    (leaf.parent / "cpu.max").write_text(quota)
    assert config.detect_available_cpu_count() == expected


@pytest.mark.parametrize("mount", ["cpu", "cpu,cpuacct", "cpuacct,cpu"])
def test_cpu_detection_uses_v1_cpu_controller_and_tightest_quota(
    cgroup_files: tuple[Path, Path], mount: str
) -> None:
    root, membership = cgroup_files
    membership.write_text("2:cpuset:/other\n5:cpu,cpuacct:/team/worker\n")
    leaf = root / mount / "team" / "worker"
    leaf.mkdir(parents=True)
    for directory, quota, period in (
        (leaf, "200000", "50000"),
        (leaf.parent, "350000", "100000"),
    ):
        (directory / "cpu.cfs_quota_us").write_text(quota)
        (directory / "cpu.cfs_period_us").write_text(period)
    assert config.detect_available_cpu_count() == 3


@pytest.mark.parametrize(
    "value", ["max 100000", "not a quota", "100000 0", "100000 -2", "0 100000"]
)
def test_unusable_cpu_quota_preserves_affinity(
    cgroup_files: tuple[Path, Path], value: str
) -> None:
    root, _ = cgroup_files
    (root / "cpu.max").write_text(value)
    assert config.detect_available_cpu_count() == 20


@pytest.mark.parametrize(
    ("filesystem", "member", "quota_file"),
    [
        ("cgroup2 none rw", "0::", "cpu.max"),
        ("cgroup none rw,cpu,cpuacct", "2:cpu,cpuacct:", "cpu.cfs_quota_us"),
    ],
)
def test_cpu_quota_inside_delegated_mount(
    cgroup_files: tuple[Path, Path], filesystem: str, member: str, quota_file: str
) -> None:
    root, membership = cgroup_files
    membership.write_text(f"{member}/delegated/team/worker\n")
    mount = root / "controller mount"
    (mount / "team" / "worker").mkdir(parents=True)
    escaped = mount.as_posix().replace(" ", r"\040")
    cgroup_cpu._MOUNTINFO.write_text(
        f"10 9 0:1 /delegated {escaped} rw - {filesystem}\n"
    )
    (mount / "team" / quota_file).write_text(
        "200000 100000" if quota_file == "cpu.max" else "200000"
    )
    (mount / "team" / "cpu.cfs_period_us").write_text("100000")
    assert config.detect_available_cpu_count() == 2


def test_cgroup_name_with_non_utf8_bytes_does_not_break_launch(
    cgroup_files: tuple[Path, Path],
) -> None:
    root, membership = cgroup_files
    membership.write_bytes(b"0::/group-\x81\n")
    (root / "cpu.max").write_text("200000 100000")
    assert config.detect_available_cpu_count() == 2


def test_non_linux_detection_preserves_host_fallback(
    cgroup_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = cgroup_files
    (root / "cpu.max").write_text("200000 100000")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(config.os, "sched_getaffinity", None)
    monkeypatch.setattr(config.os, "cpu_count", lambda: 12)
    assert config.detect_available_cpu_count() == 12


def test_mount_of_unrelated_subtree_does_not_limit_worker(
    cgroup_files: tuple[Path, Path],
) -> None:
    root, membership = cgroup_files
    membership.write_text("0::/other/service\n")
    cgroup_cpu._MOUNTINFO.write_text(
        f"10 9 0:1 /delegated {root.as_posix()} rw - cgroup2 none rw\n"
    )
    (root / "cpu.max").write_text("100000 100000")
    assert config.detect_available_cpu_count() == 20


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
