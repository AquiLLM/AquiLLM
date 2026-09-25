from __future__ import annotations

from tests.integration.task21_compose_test_support import (
    COMPOSE_DIR,
    EMBED_EXTRA_ARGS,
    EVAL_OVERRIDE,
    RERANK_EXTRA_ARGS,
    RERANK_REVISION,
    compose_service_source,
    env_file_shim,
    render,
    reviewed_env,
)


def test_shipping_compose_variants_wire_sidecar_revisions(tmp_path) -> None:
    env_file = reviewed_env(
        tmp_path,
        {
            "VLLM_API_KEY": "must-not-win",
            "VLLM_DOWNLOAD_DIR": "/must-not-win",
            "VLLM_PYTHON_BIN": "must-not-win",
        },
    )
    for name in ("base.yml", "development.yml", "production.yml"):
        compose_file = COMPOSE_DIR / name
        shim = env_file_shim(tmp_path, env_file, compose_file)
        config = render(env_file, compose_file, shim)
        embed_env = config["services"]["vllm_embed"]["environment"]
        rerank_env = config["services"]["vllm_rerank"]["environment"]
        for key in ("VLLM_REVISION", "VLLM_TOKENIZER_REVISION", "VLLM_CODE_REVISION"):
            assert embed_env[key] == embed_env["VLLM_REVISION"]
            assert rerank_env[key] == RERANK_REVISION
        assert embed_env["VLLM_RUNNER"] == "pooling"
        assert embed_env["VLLM_DTYPE"] == "float16"
        assert embed_env["VLLM_TRUST_REMOTE_CODE"] == "1"
        assert embed_env["VLLM_STRICT_PROTECTED_ARGS"] == "1"
        assert embed_env["VLLM_API_KEY"] == "EMPTY"
        assert embed_env["VLLM_DOWNLOAD_DIR"] == "/root/.cache/huggingface/hub"
        assert embed_env["VLLM_PYTHON_BIN"] == "python3"
        assert rerank_env["VLLM_RUNNER"] == "pooling"
        assert rerank_env["VLLM_DTYPE"] == "float16"
        assert rerank_env["VLLM_TRUST_REMOTE_CODE"] == "1"
        assert rerank_env["VLLM_STRICT_PROTECTED_ARGS"] == "1"
        assert rerank_env["VLLM_API_KEY"] == "EMPTY"
        assert rerank_env["VLLM_DOWNLOAD_DIR"] == "/root/.cache/huggingface/hub"
        assert rerank_env["VLLM_PYTHON_BIN"] == "python3"
        assert rerank_env["VLLM_TASK"] == ""
        embed = compose_service_source(compose_file, "vllm_embed")
        rerank = compose_service_source(compose_file, "vllm_rerank")
        assert embed.count("VLLM_REVISION=${APP_EMBED_MODEL_REVISION:-}") == 1
        assert (
            embed.count("VLLM_TOKENIZER_REVISION=${APP_EMBED_TOKENIZER_REVISION:-}")
            == 1
        )
        assert embed.count("VLLM_CODE_REVISION=${APP_EMBED_CODE_REVISION:-}") == 1
        assert embed.count("VLLM_RUNNER=${APP_EMBED_VLLM_RUNNER:-pooling}") == 1
        assert embed.count("VLLM_DTYPE=${APP_EMBED_VLLM_DTYPE:-float16}") == 1
        assert (
            embed.count(
                "VLLM_TRUST_REMOTE_CODE=${MEM0_EMBED_VLLM_TRUST_REMOTE_CODE:-1}"
            )
            == 1
        )
        assert embed.count("VLLM_STRICT_PROTECTED_ARGS=1") == 1
        assert embed.count("VLLM_API_KEY=EMPTY") == 1
        assert embed.count("VLLM_DOWNLOAD_DIR=/root/.cache/huggingface/hub") == 1
        assert embed.count("VLLM_PYTHON_BIN=python3") == 1
        assert rerank.count("VLLM_REVISION=${APP_RERANK_MODEL_REVISION:-}") == 1
        assert (
            rerank.count("VLLM_TOKENIZER_REVISION=${APP_RERANK_TOKENIZER_REVISION:-}")
            == 1
        )
        assert rerank.count("VLLM_CODE_REVISION=${APP_RERANK_CODE_REVISION:-}") == 1
        assert rerank.count("VLLM_RUNNER=${APP_RERANK_VLLM_RUNNER:-pooling}") == 1
        assert rerank.count("VLLM_DTYPE=${APP_RERANK_VLLM_DTYPE:-float16}") == 1
        assert (
            rerank.count(
                "VLLM_TRUST_REMOTE_CODE=${APP_RERANK_VLLM_TRUST_REMOTE_CODE:-1}"
            )
            == 1
        )
        assert rerank.count("VLLM_STRICT_PROTECTED_ARGS=1") == 1
        assert rerank.count("VLLM_API_KEY=EMPTY") == 1
        assert rerank.count("VLLM_DOWNLOAD_DIR=/root/.cache/huggingface/hub") == 1
        assert rerank.count("VLLM_PYTHON_BIN=python3") == 1
        assert rerank.count("VLLM_TASK=") == 1
        assert "APP_RERANK_VLLM_TASK" not in rerank
        for ordinary_service in ("vllm", "vllm_ocr", "vllm_transcribe"):
            ordinary = compose_service_source(compose_file, ordinary_service)
            assert "VLLM_STRICT_PROTECTED_ARGS" not in ordinary
            assert "VLLM_PYTHON_BIN" not in ordinary
        if name == "base.yml":
            assert (
                "VLLM_GPU_MEMORY_UTILIZATION=${MEM0_EMBED_GPU_MEMORY_UTILIZATION:-0.20}"
            ) in embed
            assert "VLLM_MAX_MODEL_LEN=${MEM0_EMBED_MAX_MODEL_LEN:-2048}" in embed
            assert (
                "VLLM_MODEL=${APP_RERANK_VLLM_MODEL:-Qwen/Qwen3-VL-Reranker-2B}"
            ) in rerank
            assert (
                "VLLM_SERVED_MODEL_NAME=${APP_RERANK_MODEL:-Qwen/Qwen3-VL-Reranker-2B}"
            ) in rerank
            assert (
                "VLLM_TOKENIZER=${APP_RERANK_TOKENIZER:-Qwen/Qwen3-VL-Reranker-2B}"
            ) in rerank
            assert (
                "VLLM_GPU_MEMORY_UTILIZATION=${APP_RERANK_GPU_MEMORY_UTILIZATION:-0.30}"
            ) in rerank
            assert "VLLM_MAX_MODEL_LEN=${APP_RERANK_MAX_MODEL_LEN:-1024}" in rerank
            assert "Qwen/Qwen3-Reranker-4B" not in rerank


def test_env_example_declares_reranker_revision() -> None:
    lines = (
        (COMPOSE_DIR.parents[1] / ".env.example")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert lines.count("APP_RERANK_MODEL_REVISION=") == 1
    assert lines.count("APP_RERANK_TOKENIZER_REVISION=") == 1
    assert lines.count("APP_RERANK_CODE_REVISION=") == 1
    assert lines.count("APP_EMBED_TOKENIZER_REVISION=") == 1
    assert lines.count("APP_EMBED_CODE_REVISION=") == 1
    assert lines.count("APP_EMBED_VLLM_RUNNER=pooling") == 1
    assert lines.count("APP_EMBED_VLLM_DTYPE=float16") == 1
    assert lines.count("APP_RERANK_VLLM_RUNNER=pooling") == 1
    assert lines.count("APP_RERANK_VLLM_DTYPE=float16") == 1
    assert lines.count("APP_RERANK_VLLM_TASK=") == 1
    assert lines.count("APP_RERANK_GPU_MEMORY_UTILIZATION=0.30") == 1
    rerank_extra_args = next(
        line for line in lines if line.startswith("APP_RERANK_VLLM_EXTRA_ARGS=")
    )
    embed_extra_args = next(
        line for line in lines if line.startswith("MEM0_EMBED_VLLM_EXTRA_ARGS=")
    )
    for extra_args in (embed_extra_args, rerank_extra_args):
        assert "--runner" not in extra_args
        assert "--dtype" not in extra_args
    escaped_embed = EMBED_EXTRA_ARGS.replace('"', '\\"')
    escaped_rerank = RERANK_EXTRA_ARGS.replace('"', '\\"')
    assert embed_extra_args == f'MEM0_EMBED_VLLM_EXTRA_ARGS="{escaped_embed}"'
    assert rerank_extra_args == f'APP_RERANK_VLLM_EXTRA_ARGS="{escaped_rerank}"'


def test_eval_override_maps_explicit_embedding_revisions() -> None:
    worker = compose_service_source(EVAL_OVERRIDE, "worker_knowledge_graph")
    embed = compose_service_source(EVAL_OVERRIDE, "vllm_embed")
    for prefix, block in (("APP_EMBED", worker), ("VLLM", embed)):
        assert (
            block.count(
                f"{prefix}_TOKENIZER_REVISION: "
                "${APP_EMBED_TOKENIZER_REVISION:?"
                "APP_EMBED_TOKENIZER_REVISION is required}"
            )
            == 1
        )
        assert (
            block.count(
                f"{prefix}_CODE_REVISION: "
                "${APP_EMBED_CODE_REVISION:?APP_EMBED_CODE_REVISION is required}"
            )
            == 1
        )
