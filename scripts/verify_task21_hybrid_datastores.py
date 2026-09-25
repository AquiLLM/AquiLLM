#!/usr/bin/env python3
# ruff: noqa: E501
"""Run the isolated Task21 PostgreSQL/Memgraph acceptance gate on a host."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import secrets
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

LABEL_KEY = "com.aquillm.task21-hybrid"
SERVICES = ("postgres", "redis", "memgraph", "app")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")


def runtime_commands() -> tuple[tuple[str, ...], ...]:
    manage = ("python", "manage.py")
    return (
        (*manage, "migrate", "--noinput"),
        (*manage, "migrate", "apps_knowledge_graph", "0006", "--noinput"),
        (*manage, "migrate", "apps_knowledge_graph", "0007", "--noinput"),
        (*manage, "migrate", "apps_knowledge_graph", "0008", "--noinput"),
        (*manage, "migrate", "--check"),
        ("python", "-m", "pytest", "apps/knowledge_graph/tests/test_projected_snapshot_parity.py", "apps/knowledge_graph/tests/test_projection_end_to_end.py", "apps/knowledge_graph/tests/test_memgraph_topology_integration.py", "apps/knowledge_graph/tests/test_retrieval_overlay_permissions.py", "-q"),
        (*manage, "project_knowledge_graph", "--all"),
        (*manage, "reconcile_knowledge_graph_projection", "--all"),
        (*manage, "prune_knowledge_graph_projection", "--all"),
    )


@dataclass(frozen=True, slots=True)
class HarnessIdentity:
    run_id: str
    project: str
    image: str
    postgres_volume: str
    memgraph_volume: str
    network: str
    label: str

    @classmethod
    def create(cls, run_id: str | None = None):
        value = uuid.uuid4().hex if run_id is None else run_id
        if type(value) is not str or _RUN_ID.fullmatch(value) is None:
            raise ValueError("run id must be exactly 32 lowercase hexadecimal characters")
        prefix = f"aquillm-task21-{value}"
        return cls(
            value,
            prefix,
            f"aquillm-task21-hybrid:{value}",
            f"aquillm-task21-postgres-{value}",
            f"aquillm-task21-memgraph-{value}",
            f"aquillm-task21-network-{value}",
            f"{LABEL_KEY}={value}",
        )


@dataclass(frozen=True, slots=True)
class CleanupProof:
    service_ids: tuple[tuple[str, str], ...]
    log_sha256: str
    zero_samples: tuple[dict[str, str], ...]


class CommandRunner:
    def run(
        self, arguments: tuple[str, ...], *, check: bool = True, timeout: int = 1800
    ) -> str:
        result = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if check and result.returncode:
            raise RuntimeError(
                f"acceptance command failed ({result.returncode}): "
                + " ".join(arguments[:4])
            )
        return result.stdout


class HybridDatastoreHarness:
    def __init__(self, *, runner, identity: HarnessIdentity, repository: Path):
        self.runner = runner
        self.identity = identity
        self.repository = repository.resolve()
        self.compose_file = self.repository / f".{identity.project}.yml"

    @property
    def compose_prefix(self) -> tuple[str, ...]:
        return (
            "docker",
            "compose",
            "--project-name",
            self.identity.project,
            "--file",
            str(self.compose_file),
        )

    def _zero_sample(self) -> dict[str, str]:
        label = self.identity.label
        commands = {
            "containers": ("docker", "ps", "-aq", "--filter", f"label={label}"),
            "networks": ("docker", "network", "ls", "-q", "--filter", f"label={label}"),
            "volumes": ("docker", "volume", "ls", "-q", "--filter", f"label={label}"),
            "tag": ("docker", "image", "ls", "-q", self.identity.image),
            "labeled_images": (
                "docker",
                "image",
                "ls",
                "-q",
                "--filter",
                f"label={label}",
            ),
        }
        return {name: self.runner.run(command, check=False).strip() for name, command in commands.items()}

    def _remove_owned_image(self) -> None:
        inspect = (
            "docker",
            "image",
            "inspect",
            "--format",
            f'{{{{ index .Config.Labels "{LABEL_KEY}" }}}}',
            self.identity.image,
        )
        observed = self.runner.run(inspect, check=False).strip()
        if observed and observed != self.identity.run_id:
            raise RuntimeError("refusing to remove image without the exact run label")
        if observed == self.identity.run_id:
            self.runner.run(("docker", "image", "rm", self.identity.image), check=False)

    def capture_and_cleanup(self, *, require_cleanup_proof: bool) -> CleanupProof:
        service_ids = tuple(
            (
                service,
                self.runner.run(
                    self.compose_prefix + ("ps", "--all", "--quiet", service),
                    check=False,
                ).strip(),
            )
            for service in SERVICES
        )
        logs = self.runner.run(
            self.compose_prefix + ("logs", "--no-color", "--tail", "200"),
            check=False,
        )
        self.runner.run(
            self.compose_prefix + ("down", "--volumes", "--remove-orphans"),
            check=False,
        )
        self._remove_owned_image()
        samples = []
        for _ in range(3):
            samples.append(self._zero_sample())
            if any(samples[-1].values()) and require_cleanup_proof:
                time.sleep(0.25)
        proof = CleanupProof(
            service_ids,
            hashlib.sha256(logs.encode()).hexdigest(),
            tuple(samples),
        )
        if require_cleanup_proof and any(
            value for sample in proof.zero_samples for value in sample.values()
        ):
            raise RuntimeError("Task21 labeled Docker resources survived cleanup")
        return proof

    def execute(self, runtime_directory: Path) -> CleanupProof:
        self.compose_file = runtime_directory / "compose.yml"
        init_file = runtime_directory / "init.sql"
        secrets_map = {name: secrets.token_hex(24) for name in ("app", "source", "state", "graph")}
        init_file.write_text(_postgres_init(secrets_map), encoding="utf-8")
        self.compose_file.write_text(
            _compose_yaml(self.identity, secrets_map, init_file),
            encoding="utf-8",
        )
        os.chmod(init_file, 0o600)
        os.chmod(self.compose_file, 0o600)
        try:
            self.runner.run(
                (
                    "docker",
                    "build",
                    "--label",
                    self.identity.label,
                    "--tag",
                    self.identity.image,
                    "--file",
                    str(self.repository / "deploy/docker/knowledge-graph/Dockerfile"),
                    str(self.repository),
                )
            )
            self.runner.run(self.compose_prefix + ("up", "--detach", "--wait", "postgres", "redis", "memgraph"))
            for command in runtime_commands():
                self.runner.run(self.compose_prefix + ("run", "--rm", "app", *command))
        finally:
            proof = self.capture_and_cleanup(require_cleanup_proof=True)
        return proof


def _postgres_init(values: dict[str, str]) -> str:
    return (
        f"CREATE ROLE aquillm_projection_source LOGIN PASSWORD '{values['source']}';\n"
        f"CREATE ROLE aquillm_projection_state LOGIN PASSWORD '{values['state']}';\n"
    )


def _compose_yaml(identity, value: dict[str, str], init_file: Path) -> str:
    init = init_file.as_posix()
    return f"""services:
  postgres:
    image: pgvector/pgvector:0.8.0-pg17
    labels: [{identity.label}]
    environment: {{POSTGRES_DB: aquillm, POSTGRES_USER: aquillm, POSTGRES_PASSWORD: {value['app']}}}
    volumes: [\"postgres_data:/var/lib/postgresql/data\", \"{init}:/docker-entrypoint-initdb.d/task21.sql:ro\"]
    networks: [runtime]
    healthcheck: {{test: [CMD-SHELL, \"pg_isready -U aquillm -d aquillm\"], interval: 2s, timeout: 2s, retries: 30}}
  redis:
    image: redis:7.4-bookworm
    labels: [{identity.label}]
    networks: [runtime]
    healthcheck: {{test: [CMD, redis-cli, ping], interval: 2s, timeout: 2s, retries: 30}}
  memgraph:
    image: memgraph/memgraph-mage:3.8.1
    labels: [{identity.label}]
    environment: {{MEMGRAPH_USER: projection, MEMGRAPH_PASSWORD: {value['graph']}}}
    volumes: [\"memgraph_data:/var/lib/memgraph\"]
    networks: [runtime]
    healthcheck: {{test: [CMD-SHELL, \"mgconsole --host 127.0.0.1 --username projection --password $$MEMGRAPH_PASSWORD --execute 'RETURN 1;'\"], interval: 2s, timeout: 2s, retries: 30}}
  app:
    image: {identity.image}
    labels: [{identity.label}]
    working_dir: /app/aquillm
    networks: [runtime]
    environment:
      DJANGO_DEBUG: '1'
      SECRET_KEY: task21-runtime
      GOOGLE_OAUTH2_CLIENT_ID: task21-runtime
      GOOGLE_OAUTH2_CLIENT_SECRET: task21-runtime
      OPENAI_API_KEY: task21-runtime
      GEMINI_API_KEY: task21-runtime
      POSTGRES_HOST: postgres
      POSTGRES_NAME: aquillm
      POSTGRES_USER: aquillm
      POSTGRES_PASSWORD: {value['app']}
      KG_MEMGRAPH_PROJECTION_ENABLED: '1'
      KG_MEMGRAPH_URI: bolt://memgraph:7687
      KG_MEMGRAPH_PROJECTION_USERNAME: projection
      KG_MEMGRAPH_PROJECTION_PASSWORD: {value['graph']}
      KG_PROJECTION_POSTGRES_SOURCE_DSN: postgresql://aquillm_projection_source:{value['source']}@postgres:5432/aquillm
      KG_PROJECTION_POSTGRES_STATE_DSN: postgresql://aquillm_projection_state:{value['state']}@postgres:5432/aquillm
      KG_PROJECTION_IDENTIFIER_HMAC_KEY: task21-runtime-hmac
      KG_PROJECTION_IDENTIFIER_KEY_VERSION: task21-key-v1
      KG_REQUIRE_CONTAINER_TESTS: '1'
    depends_on: {{postgres: {{condition: service_healthy}}, redis: {{condition: service_healthy}}, memgraph: {{condition: service_healthy}}}}
volumes:
  postgres_data: {{name: {identity.postgres_volume}, labels: [{identity.label}]}}
  memgraph_data: {{name: {identity.memgraph_volume}, labels: [{identity.label}]}}
networks:
  runtime: {{name: {identity.network}, labels: [{identity.label}]}}
"""


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--postgres", action="store_true")
    value.add_argument("--memgraph", action="store_true")
    value.add_argument("--require-cleanup-proof", action="store_true")
    return value


def validate_arguments(arguments) -> None:
    if not (arguments.postgres and arguments.memgraph and arguments.require_cleanup_proof):
        raise ValueError("--postgres, --memgraph, and --require-cleanup-proof are required")


def main() -> int:
    arguments = parser().parse_args()
    validate_arguments(arguments)
    identity = HarnessIdentity.create()
    repository = Path(__file__).resolve().parents[1]
    harness = HybridDatastoreHarness(runner=CommandRunner(), identity=identity, repository=repository)
    with tempfile.TemporaryDirectory(prefix=f"{identity.project}-") as directory:
        proof = harness.execute(Path(directory))
    print(f"task21_hybrid_datastores=passed cleanup_samples={len(proof.zero_samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
