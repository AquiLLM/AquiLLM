"""Deployment contract tests for the main Genesis-patched vLLM image."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VLLM_IMAGE = (
    "vllm/vllm-openai@sha256:"
    "6a93ae4316826f3dd8a92bee5442cbed50184a9cbd688d310f9e56ecad1eabeb"
)
GENESIS_REF = "34e269301cc3df71ae4b0da00a0a159b16b4e5d8"


def test_genesis_image_pins_the_validated_vllm_and_plugin_pair():
    dockerfile = (
        REPO_ROOT / "deploy/docker/vllm/Dockerfile.genesis"
    ).read_text(encoding="utf-8")

    assert f"ARG VLLM_GENESIS_BASE_IMAGE={VLLM_IMAGE}" in dockerfile
    assert f"ARG GENESIS_REF={GENESIS_REF}" in dockerfile
    assert "git fetch --depth 1 origin \"${GENESIS_REF}\"" in dockerfile
    assert "git checkout --detach FETCH_HEAD" in dockerfile
    assert "pip install --no-deps --no-cache-dir -e /opt/genesis" in dockerfile
    assert "import sndr" in dockerfile
    assert '"sndr.plugin:register"' in dockerfile
    assert "vllm/_genesis" not in dockerfile
    assert "tools/genesis_vllm_plugin" not in dockerfile


def test_production_compose_uses_the_same_validated_pair_by_default():
    compose = (REPO_ROOT / "deploy/compose/production.yml").read_text(
        encoding="utf-8"
    )

    expected_image_arg = (
        f"VLLM_GENESIS_BASE_IMAGE: ${{VLLM_GENESIS_BASE_IMAGE:-{VLLM_IMAGE}}}"
    )
    assert expected_image_arg in compose
    assert f"GENESIS_REF: ${{GENESIS_REF:-{GENESIS_REF}}}" in compose


def test_genesis_entrypoint_requires_the_current_plugin_contract():
    entrypoint = (
        REPO_ROOT / "deploy/scripts/genesis_vllm_entrypoint.sh"
    ).read_text(encoding="utf-8")

    assert "import sndr" in entrypoint
    assert "vllm.general_plugins" in entrypoint
    assert "sndr.plugin" in entrypoint
    assert "import vllm._genesis" not in entrypoint


def test_example_environment_enables_the_turboquant_mtp_workspace_stack():
    lines = set((REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines())

    assert f"VLLM_GENESIS_BASE_IMAGE={VLLM_IMAGE}" in lines
    assert f"GENESIS_REF={GENESIS_REF}" in lines
    for setting in (
        "GENESIS_ENFORCE_VERSION_RANGE=1",
        "GENESIS_ENABLE_P98=1",
        "GENESIS_ENABLE_PN118=1",
        "GENESIS_ENABLE_PN119=1",
        "GENESIS_ENABLE_PN399_TQ_DECODE_SCRATCH_IMA=1",
        "GENESIS_ENABLE_PN401_TQ_PREFILL_CONTINUATION_GUARD=1",
        "GENESIS_ENABLE_PN521_TQ_RAW_TAIL_VERIFY=1",
        "GENESIS_ENABLE_PN521_SPLIT_K=1",
        "GENESIS_P67_BLOCK_KV=32",
        "GENESIS_ENABLE_PN522_TQ_RAW_TAIL_WARMUP=1",
        "GENESIS_ENABLE_P82=0",
    ):
        assert setting in lines
    assert "GENESIS_ENABLE_PN34_WORKSPACE_LOCK_RELAX=1" not in lines


def test_example_environment_enables_mtp_speculation_by_default():
    environment = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    vllm_extra_args = next(
        line for line in environment.splitlines() if line.startswith("VLLM_EXTRA_ARGS=")
    )

    assert "--kv-cache-dtype turboquant_k8v4" in vllm_extra_args
    assert (
        "--speculative-config '{\\\"method\\\":\\\"mtp\\\",\\\"num_speculative_tokens\\\":4}'"
        in vllm_extra_args
    )


def test_main_vllm_context_limit_defaults_are_synced_with_the_ui_limit():
    environment = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    production = (REPO_ROOT / "deploy/compose/production.yml").read_text(
        encoding="utf-8"
    )
    development = (REPO_ROOT / "deploy/compose/development.yml").read_text(
        encoding="utf-8"
    )
    page_view = (REPO_ROOT / "aquillm/apps/chat/views/pages.py").read_text(
        encoding="utf-8"
    )

    assert "VLLM_MAX_MODEL_LEN=131072" in environment.splitlines()
    assert "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-131072}" in production
    assert "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-131072}" in development
    assert 'os.getenv("VLLM_MAX_MODEL_LEN", "")' in page_view
