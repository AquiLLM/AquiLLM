from __future__ import annotations

import os
import subprocess
import time
import uuid

import pytest

IMAGE = "memgraph/memgraph-mage:3.8.1"
LABEL = "aquillm.kg.projection-test"


def _docker(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("docker", *arguments),
        check=check,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _wait_for_bolt(container_name: str) -> None:
    for _attempt in range(60):
        probe = _docker(
            "exec",
            container_name,
            "bash",
            "-lc",
            "echo 'RETURN 1;' | mgconsole --host 127.0.0.1 --port 7687",
            check=False,
        )
        if probe.returncode == 0:
            return
        time.sleep(0.25)
    raise RuntimeError("isolated Memgraph Bolt endpoint did not become ready")


def _verified_cleanup(name: str, network: str, volume: str, run_label: str) -> None:
    inspected = _docker(
        "inspect",
        "--format",
        '{{ index .Config.Labels "aquillm.kg.projection-test" }}',
        name,
        check=False,
    )
    if inspected.returncode == 0 and inspected.stdout.strip() == run_label:
        _docker("rm", "--force", name, check=False)
    for kind, target in (("network", network), ("volume", volume)):
        labels = _docker(
            kind,
            "inspect",
            "--format",
            '{{ index .Labels "aquillm.kg.projection-test" }}',
            target,
            check=False,
        )
        if labels.returncode == 0 and labels.stdout.strip() == run_label:
            _docker(kind, "rm", target, check=False)
    samples = [
        _docker(*command).stdout.strip()
        for command in (
            ("ps", "-aq", "--filter", f"label={LABEL}={run_label}"),
            ("network", "ls", "-q", "--filter", f"label={LABEL}={run_label}"),
            ("volume", "ls", "-q", "--filter", f"label={LABEL}={run_label}"),
        )
    ]
    assert samples == [""] * 3


@pytest.fixture
def isolated_memgraph_container():
    if os.environ.get("KG_REQUIRE_MEMGRAPH_TESTS") != "1":
        pytest.skip("set KG_REQUIRE_MEMGRAPH_TESTS=1 for isolated Memgraph tests")
    run_label = uuid.uuid4().hex
    name = f"aquillm-kg-projection-{run_label}"
    network = f"aquillm-kg-network-{run_label}"
    volume = f"aquillm-kg-volume-{run_label}"
    try:
        _docker("network", "create", "--label", f"{LABEL}={run_label}", network)
        _docker("volume", "create", "--label", f"{LABEL}={run_label}", volume)
        _docker(
            "run",
            "--detach",
            "--name",
            name,
            "--label",
            f"{LABEL}={run_label}",
            "--network",
            network,
            "--publish",
            "127.0.0.1::7687",
            "--mount",
            f"type=volume,source={volume},target=/var/lib/memgraph",
            IMAGE,
        )
        _wait_for_bolt(name)
        binding = _docker("port", name, "7687/tcp").stdout.strip().splitlines()[0]
        yield {"uri": f"bolt://{binding}", "database": "memgraph"}
    finally:
        _verified_cleanup(name, network, volume, run_label)
