"""Independent embedding/reranking revision contracts for shipping Compose files."""

import shlex
from pathlib import Path

import pytest
from dotenv import dotenv_values

from tests.integration.compose_render_test_support import (
    render_compose_with_reviewed_env,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "compose_name", ["base.yml", "development.yml", "production.yml"]
)
def test_shipping_sidecars_forward_their_own_model_tokenizer_and_code_revisions(
    compose_name,
):
    config = render_compose_with_reviewed_env(
        (ROOT / "deploy/compose" / compose_name,),
        profile="vllm",
        environment_overrides={
            "APP_EMBED_MODEL_REVISION": "a" * 40,
            "APP_EMBED_TOKENIZER_REVISION": "b" * 40,
            "APP_EMBED_CODE_REVISION": "c" * 40,
            "APP_RERANK_MODEL_REVISION": "d" * 40,
            "APP_RERANK_TOKENIZER_REVISION": "e" * 40,
            "APP_RERANK_CODE_REVISION": "f" * 40,
            "VLLM_REVISION": "global-model-must-not-win",
            "VLLM_TOKENIZER_REVISION": "global-tokenizer-must-not-win",
            "VLLM_CODE_REVISION": "global-code-must-not-win",
            "MEM0_EMBED_VLLM_EXTRA_ARGS": "--max-num-seqs 2",
            "APP_RERANK_VLLM_EXTRA_ARGS": "--max-num-seqs 3",
        },
    )
    services = config["services"]
    for name, model, tokenizer, code, extra in (
        ("vllm_embed", "a" * 40, "b" * 40, "c" * 40, "--max-num-seqs 2"),
        ("vllm_rerank", "d" * 40, "e" * 40, "f" * 40, "--max-num-seqs 3"),
    ):
        environment = services[name]["environment"]
        assert environment["VLLM_REVISION"] == model
        assert environment["VLLM_TOKENIZER_REVISION"] == tokenizer
        assert environment["VLLM_CODE_REVISION"] == code
        assert environment["VLLM_RUNNER"] == "pooling"
        assert environment["VLLM_DTYPE"] == "float16"
        assert environment["VLLM_TRUST_REMOTE_CODE"] == "1"
        assert environment["VLLM_EXTRA_ARGS"] == extra
        # Shipping sidecars reject extra arguments that could override the
        # independently pinned runner, dtype, or checkpoint identity.
        assert environment["VLLM_STRICT_PROTECTED_ARGS"] == "1"
    assert services["vllm_rerank"]["environment"]["VLLM_TASK"] == ""


def test_environment_example_exposes_separate_revision_pins_and_valid_extra_args():
    example = ROOT / ".env.example"
    values = dotenv_values(example, interpolate=False)
    lines = example.read_text(encoding="utf-8").splitlines()
    for name in (
        "APP_EMBED_MODEL_REVISION",
        "APP_EMBED_TOKENIZER_REVISION",
        "APP_EMBED_CODE_REVISION",
        "APP_RERANK_MODEL_REVISION",
        "APP_RERANK_TOKENIZER_REVISION",
        "APP_RERANK_CODE_REVISION",
    ):
        assert values[name] == ""
        assert sum(line.startswith(f"{name}=") for line in lines) == 1
    assert values["APP_EMBED_VLLM_RUNNER"] == "pooling"
    assert values["APP_RERANK_VLLM_RUNNER"] == "pooling"
    assert values["APP_EMBED_VLLM_DTYPE"] == "float16"
    assert values["APP_RERANK_VLLM_DTYPE"] == "float16"
    assert values.get("VLLM_STRICT_PROTECTED_ARGS", "0") == "0"
    for name in ("MEM0_EMBED_VLLM_EXTRA_ARGS", "APP_RERANK_VLLM_EXTRA_ARGS"):
        args = shlex.split(values[name])
        assert args
        assert "--runner" not in args
        assert "--dtype" not in args
        assert not any(arg.startswith(("--runner=", "--dtype=")) for arg in args)
