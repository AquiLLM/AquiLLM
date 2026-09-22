import subprocess
from pathlib import Path

import pytest
import yaml

from tests.integration.compose_render_test_support import (
    render_compose_with_reviewed_env,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_render_compose_uses_disposable_env_when_repository_env_is_absent(tmp_path):
    checkout = tmp_path / "checkout"
    compose_file = checkout / "deploy/compose/development.yml"
    compose_file.parent.mkdir(parents=True)
    compose_file.write_bytes(
        (REPOSITORY_ROOT / "deploy/compose/development.yml").read_bytes()
    )
    assert not (checkout / ".env").exists()
    rendered = render_compose_with_reviewed_env((compose_file,), profile="vllm")
    assert "vllm_transcribe" in rendered["services"]


@pytest.mark.parametrize("operator_env_exists", [False, True])
def test_older_compose_env_file_validation_uses_only_disposable_environment(
    tmp_path, monkeypatch, operator_env_exists
):
    checkout = tmp_path / "checkout"
    compose_dir = checkout / "deploy/compose"
    compose_dir.mkdir(parents=True)
    operator_env = checkout / ".env"
    secret = "OPERATOR_SECRET=must-not-read"
    if operator_env_exists:
        operator_env.write_text(secret, encoding="utf-8")
    compose_file = compose_dir / "base.yml"
    compose_file.write_text(
        "services:\n  web:\n    build:\n      context: ../..\n"
        "    env_file: [../../.env]\n"
        "    volumes: [../../artifacts:/app/artifacts]\n"
        "    environment:\n      RAG_DIRECT_ENABLED: ${RAG_DIRECT_ENABLED:-1}\n",
        encoding="utf-8",
    )
    override = compose_dir / "override.yml"
    override.write_text(
        "services:\n  worker:\n    image: redis:7\n    env_file: ../../.env\n",
        encoding="utf-8",
    )
    originals = [path.read_bytes() for path in (compose_file, override)]
    real_run = subprocess.run
    validated_services = []

    def run_with_legacy_env_file_stat(command, **kwargs):
        files = [
            Path(command[index + 1])
            for index, value in enumerate(command)
            if value == "-f"
        ]
        project_dir = (
            Path(command[command.index("--project-directory") + 1])
            if "--project-directory" in command
            else files[0].parent
        )
        # Compose releases in CI validate required service env_file paths even
        # with --no-env-resolution. Reproduce that pre-render validation.
        for path in files:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            for name, service in config["services"].items():
                env_paths = service.get("env_file", [])
                if isinstance(env_paths, str):
                    env_paths = [env_paths]
                for env_path in env_paths:
                    if isinstance(env_path, dict):
                        env_path = env_path["path"]
                    resolved = (project_dir / env_path).resolve()
                    if not resolved.exists():
                        return subprocess.CompletedProcess(
                            command, 1, "", f"env file {resolved} not found"
                        )
                    assert resolved != operator_env.resolve()
                    assert "OPERATOR_SECRET" not in resolved.read_text(encoding="utf-8")
                    validated_services.append(name)
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_with_legacy_env_file_stat)
    rendered = render_compose_with_reviewed_env(
        (compose_file, override),
        profile="vllm",
        environment_overrides={"RAG_DIRECT_ENABLED": "0"},
    )
    assert validated_services == ["web", "worker"]
    web = rendered["services"]["web"]
    assert Path(web["build"]["context"]).resolve() == checkout.resolve()
    assert (
        Path(web["volumes"][0]["source"]).resolve()
        == (checkout / "artifacts").resolve()
    )
    assert web["environment"]["RAG_DIRECT_ENABLED"] == "0"
    assert [path.read_bytes() for path in (compose_file, override)] == originals
    if operator_env_exists:
        assert operator_env.read_text(encoding="utf-8") == secret
    else:
        assert not operator_env.exists()
