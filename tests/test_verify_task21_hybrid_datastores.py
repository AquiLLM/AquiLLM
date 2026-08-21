# ruff: noqa: E501
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_task21_hybrid_datastores.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("verify_task21_hybrid_datastores", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeRunner:
    def __init__(self, outputs: dict[tuple[str, ...], str] | None = None):
        self.outputs = outputs or {}
        self.calls: list[tuple[str, ...]] = []

    def run(self, arguments, **_kwargs):
        command = tuple(arguments)
        self.calls.append(command)
        return self.outputs.get(command, "")


def test_runtime_commands_pin_full_rollback_reapply_and_projection_lifecycle() -> None:
    module = _module()
    commands = module.runtime_commands()
    assert commands[:4] == (
        ("python", "manage.py", "migrate", "--noinput"),
        ("python", "manage.py", "migrate", "apps_knowledge_graph", "0006", "--noinput"),
        ("python", "manage.py", "migrate", "apps_knowledge_graph", "0007", "--noinput"),
        ("python", "manage.py", "migrate", "apps_knowledge_graph", "0008", "--noinput"),
    )
    assert ("python", "manage.py", "migrate", "--check") in commands
    assert ("python", "manage.py", "project_knowledge_graph", "--all") in commands
    assert (
        "python",
        "manage.py",
        "reconcile_knowledge_graph_projection",
        "--all",
    ) in commands
    assert (
        "python",
        "manage.py",
        "prune_knowledge_graph_projection",
        "--all",
    ) in commands
    test_command = next(command for command in commands if command[:3] == ("python", "-m", "pytest"))
    assert {
        "apps/knowledge_graph/tests/test_projected_snapshot_parity.py",
        "apps/knowledge_graph/tests/test_projection_end_to_end.py",
        "apps/knowledge_graph/tests/test_memgraph_topology_integration.py",
        "apps/knowledge_graph/tests/test_retrieval_overlay_permissions.py",
    }.issubset(test_command)
    assert "container" not in test_command
    assert all("--dry-run" not in command for command in commands)


def test_identity_is_unique_bounded_and_labels_every_owned_resource() -> None:
    module = _module()
    first = module.HarnessIdentity.create("a" * 32)
    second = module.HarnessIdentity.create("b" * 32)
    assert first != second
    assert first.project.startswith("aquillm-task21-")
    assert first.image.startswith("aquillm-task21-hybrid:")
    assert first.label == "com.aquillm.task21-hybrid=" + "a" * 32
    assert {first.postgres_volume, first.memgraph_volume, first.network} == {
        f"aquillm-task21-{kind}-{'a' * 32}"
        for kind in ("postgres", "memgraph", "network")
    }
    with pytest.raises(ValueError, match="run id"):
        module.HarnessIdentity.create("../ambient")


def test_compose_is_fresh_labeled_unpublished_and_uses_fixed_pg_roles(tmp_path) -> None:
    module = _module()
    identity = module.HarnessIdentity.create("e" * 32)
    values = {name: f"secret-{name}" for name in ("app", "source", "state", "graph")}
    compose = yaml.safe_load(
        module._compose_yaml(identity, values, tmp_path / "init.sql")
    )

    assert compose["services"]["postgres"]["image"].startswith("pgvector/pgvector:")
    for service in compose["services"].values():
        assert service["labels"] == [identity.label]
        assert "ports" not in service
    assert compose["volumes"] == {
        "postgres_data": {"name": identity.postgres_volume, "labels": [identity.label]},
        "memgraph_data": {"name": identity.memgraph_volume, "labels": [identity.label]},
    }
    assert compose["networks"]["runtime"] == {
        "name": identity.network,
        "labels": [identity.label],
    }
    assert compose["services"]["app"]["environment"][
        "GOOGLE_OAUTH2_CLIENT_ID"
    ] == compose["services"]["app"]["environment"]["GOOGLE_OAUTH2_CLIENT_SECRET"]
    init = module._postgres_init(values)
    assert init.count("CREATE ROLE") == 2
    assert "aquillm_projection_source" in init
    assert "aquillm_projection_state" in init
    assert "SUPERUSER" not in init and "GRANT " not in init


def test_capture_precedes_scoped_teardown_and_cleanup_has_three_zero_samples() -> None:
    module = _module()
    identity = module.HarnessIdentity.create("c" * 32)
    runner = FakeRunner()
    harness = module.HybridDatastoreHarness(
        runner=runner,
        identity=identity,
        repository=Path("C:/workspace/AquiLLM"),
    )

    proof = harness.capture_and_cleanup(require_cleanup_proof=True)

    calls = runner.calls
    capture_indexes = [
        index
        for index, call in enumerate(calls)
        if call[:-1] == harness.compose_prefix + ("ps", "--all", "--quiet")
    ]
    down_index = calls.index(harness.compose_prefix + ("down", "--volumes", "--remove-orphans"))
    assert len(capture_indexes) == 4
    assert max(capture_indexes) < down_index
    assert len(proof.zero_samples) == 3
    assert all(all(not value for value in sample.values()) for sample in proof.zero_samples)
    forbidden = {("docker", "system", "prune"), ("docker", "volume", "prune")}
    assert not forbidden.intersection(calls)


def test_cleanup_refuses_an_image_without_the_exact_run_label() -> None:
    module = _module()
    identity = module.HarnessIdentity.create("d" * 32)
    inspect = (
        "docker",
        "image",
        "inspect",
        "--format",
        '{{ index .Config.Labels "com.aquillm.task21-hybrid" }}',
        identity.image,
    )
    runner = FakeRunner({inspect: "different-run"})
    harness = module.HybridDatastoreHarness(
        runner=runner,
        identity=identity,
        repository=Path("C:/workspace/AquiLLM"),
    )
    with pytest.raises(RuntimeError, match="label"):
        harness.capture_and_cleanup(require_cleanup_proof=True)
    assert ("docker", "image", "rm", identity.image) not in runner.calls


def test_cli_requires_both_datastores_and_cleanup_proof() -> None:
    module = _module()
    parser = module.parser()
    values = parser.parse_args(["--postgres", "--memgraph", "--require-cleanup-proof"])
    module.validate_arguments(values)
    for arguments in (
        ["--postgres", "--memgraph"],
        ["--postgres", "--require-cleanup-proof"],
        ["--memgraph", "--require-cleanup-proof"],
    ):
        with pytest.raises(ValueError, match="required"):
            module.validate_arguments(parser.parse_args(arguments))
