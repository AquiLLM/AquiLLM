"""Detect the effective CPU quota in the process's visible Linux cgroup hierarchy."""

import re
import sys
from pathlib import Path, PurePosixPath

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_CGROUP_MEMBERSHIP = Path("/proc/self/cgroup")
_MOUNTINFO = Path("/proc/self/mountinfo")


def _cpu_quota(directory: Path, *, unified: bool) -> int | None:
    """Read the CFS quota in whole CPUs. Unlimited or unreadable quotas add no cap."""
    try:
        if unified:
            quota, period = (directory / "cpu.max").read_text().split()
        else:
            quota = (directory / "cpu.cfs_quota_us").read_text()
            period = (directory / "cpu.cfs_period_us").read_text()
        quota_us, period_us = int(quota), int(period)
    except (OSError, ValueError):
        return None
    if quota_us > 0 and period_us > 0:
        return max(1, quota_us // period_us)
    return None


def _proc_lines(path: Path) -> list[str]:
    try:
        data = path.read_bytes().decode(sys.getfilesystemencoding(), "surrogateescape")
        return data.split("\n")
    except OSError:
        return []


def _cpu_cgroup_mounts() -> dict[tuple[Path, bool], PurePosixPath]:
    groups: dict[str, PurePosixPath] = {}
    for line in _proc_lines(_CGROUP_MEMBERSHIP):
        fields = line.split(":", 2)
        if len(fields) == 3:
            group = PurePosixPath(fields[2])
            if group.root == "/" and ".." not in group.parts:
                for controller in fields[1].split(","):
                    groups[controller] = group

    # Conventional mounts also work when procfs is hidden or the process
    # is in a private cgroup namespace that exposes just the constrained root.
    filesystem_root = PurePosixPath("/")
    mounts = {
        (_CGROUP_ROOT / name, not name): groups.get(
            "cpu" if name else "", filesystem_root
        ).relative_to(filesystem_root)
        for name in ("", "cpu", "cpu,cpuacct", "cpuacct,cpu")
    }
    for line in _proc_lines(_MOUNTINFO):
        location, _, filesystem = line.partition(" - ")
        fields, fs = location.split(), filesystem.split()
        if len(fields) < 5 or len(fs) < 3:
            continue
        unified = fs[0] == "cgroup2"
        if not unified and (fs[0] != "cgroup" or "cpu" not in fs[2].split(",")):
            continue
        group = groups.get("" if unified else "cpu")
        # Mountinfo escapes whitespace and backslashes inside each path.
        root, target = (
            re.sub(r"\\(040|011|012|134)", lambda m: chr(int(m[1], 8)), field)
            for field in fields[3:5]
        )
        mount = Path(target)
        if group is None or not mount.is_absolute():
            continue
        if group.is_relative_to(root):
            mounts[mount, unified] = group.relative_to(root)
        elif group != filesystem_root and ".." not in PurePosixPath(root).parts:
            # A bind mount of an unrelated subtree cannot constrain this process.
            mounts.pop((mount, unified), None)
    return mounts


def apply_cgroup_cpu_limits(cpu_count: int) -> int:
    """Limit worker CPU allocation to visible Linux quota boundaries."""
    if sys.platform != "linux":
        return cpu_count
    for (mount, unified), relative in _cpu_cgroup_mounts().items():
        # An unlimited leaf still inherits every visible parent cap.
        directory = mount.joinpath(*relative.parts)
        while True:
            limit = _cpu_quota(directory, unified=unified)
            if limit is not None:
                cpu_count = min(cpu_count, limit)
            if directory == mount:
                break
            directory = directory.parent
    return cpu_count
