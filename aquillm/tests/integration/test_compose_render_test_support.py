from pathlib import Path

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
