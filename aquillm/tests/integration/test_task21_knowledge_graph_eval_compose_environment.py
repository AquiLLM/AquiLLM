from __future__ import annotations

from tests.integration.task21_compose_test_support import (
    COMPOSE_DIR,
    EMBED_MODEL,
    EMBED_REVISION,
    EVAL_OVERRIDE,
    GLINER_MODEL,
    GLINER_REVISION,
    RERANK_MODEL,
    RERANK_REVISION,
    render,
    reviewed_env,
)


def test_eval_render_ignores_hostile_ambient_interpolation(
    tmp_path, monkeypatch
) -> None:
    env_file = reviewed_env(tmp_path)
    hostile = {
        "POSTGRES_NAME": "host_database",
        "POSTGRES_USER": "host_role",
        "POSTGRES_PASSWORD": "host-password",
        "APP_EMBED_MODEL": "host/embed",
        "APP_EMBED_MODEL_REVISION": "d" * 40,
        "APP_EMBED_TOKENIZER_REVISION": "2" * 40,
        "APP_EMBED_CODE_REVISION": "3" * 40,
        "APP_EMBED_VLLM_RUNNER": "generate",
        "APP_EMBED_VLLM_DTYPE": "float64",
        "APP_EMBED_VLLM_STRICT_PROTECTED_ARGS": "0",
        "APP_EMBED_VLLM_API_KEY": "must-not-win",
        "APP_EMBED_VLLM_DOWNLOAD_DIR": "/must-not-win",
        "APP_EMBED_VLLM_PYTHON_BIN": "must-not-win",
        "APP_RERANK_MODEL": "host/rerank-served",
        "APP_RERANK_VLLM_MODEL": "host/rerank-model",
        "APP_RERANK_TOKENIZER": "host/rerank-tokenizer",
        "APP_RERANK_MODEL_REVISION": "e" * 40,
        "APP_RERANK_TOKENIZER_REVISION": "f" * 40,
        "APP_RERANK_CODE_REVISION": "0" * 40,
        "APP_RERANK_VLLM_RUNNER": "generate",
        "APP_RERANK_VLLM_DTYPE": "float64",
        "APP_RERANK_VLLM_STRICT_PROTECTED_ARGS": "0",
        "APP_RERANK_VLLM_TASK": "embed",
        "APP_RERANK_VLLM_DOWNLOAD_DIR": "/must-not-win",
        "APP_RERANK_VLLM_PYTHON_BIN": "must-not-win",
        "KG_GLINER2_MODEL": "host/gliner",
        "KG_GLINER2_REVISION": "1" * 40,
        "KG_GLINER2_DEVICE": "cuda:9",
        "KG_GLINER2_BATCH_SIZE": "999",
        "KG_GLINER2_MAX_BATCH_CHARACTERS": "999999999",
    }
    for name, value in hostile.items():
        monkeypatch.setenv(name, value)

    config = render(env_file, COMPOSE_DIR / "development.yml", EVAL_OVERRIDE)
    worker = config["services"]["worker_knowledge_graph"]["environment"]
    database = config["services"]["db"]["environment"]
    embed = config["services"]["vllm_embed"]["environment"]
    rerank = config["services"]["vllm_rerank"]["environment"]
    assert database["POSTGRES_DB"] == worker["POSTGRES_NAME"] == "task21_database"
    assert database["POSTGRES_USER"] == worker["POSTGRES_USER"] == "task21_role"
    assert worker["APP_EMBED_MODEL"] == embed["VLLM_MODEL"] == EMBED_MODEL
    assert embed["VLLM_REVISION"] == EMBED_REVISION
    assert worker["APP_EMBED_TOKENIZER_REVISION"] == EMBED_REVISION
    assert worker["APP_EMBED_CODE_REVISION"] == EMBED_REVISION
    assert embed["VLLM_RUNNER"] == "pooling"
    assert embed["VLLM_DTYPE"] == "float16"
    assert embed["VLLM_API_KEY"] == "EMPTY"
    assert embed["VLLM_DOWNLOAD_DIR"] == "/root/.cache/huggingface/hub"
    assert embed["VLLM_PYTHON_BIN"] == "python3"
    assert worker["APP_EMBED_VLLM_STRICT_PROTECTED_ARGS"] == "1"
    assert worker["APP_EMBED_VLLM_API_KEY"] == "EMPTY"
    assert worker["APP_EMBED_VLLM_DOWNLOAD_DIR"] == "/root/.cache/huggingface/hub"
    assert worker["APP_EMBED_VLLM_PYTHON_BIN"] == "python3"
    assert embed["VLLM_STRICT_PROTECTED_ARGS"] == "1"
    assert (
        worker["APP_RERANK_MODEL"] == rerank["VLLM_SERVED_MODEL_NAME"] == RERANK_MODEL
    )
    assert rerank["VLLM_REVISION"] == RERANK_REVISION
    assert rerank["VLLM_RUNNER"] == "pooling"
    assert rerank["VLLM_DTYPE"] == "float16"
    assert rerank["VLLM_API_KEY"] == "EMPTY"
    assert rerank["VLLM_DOWNLOAD_DIR"] == "/root/.cache/huggingface/hub"
    assert rerank["VLLM_PYTHON_BIN"] == "python3"
    assert worker["APP_RERANK_VLLM_STRICT_PROTECTED_ARGS"] == "1"
    assert worker["APP_RERANK_VLLM_TASK"] == "score"
    assert worker["APP_RERANK_VLLM_DOWNLOAD_DIR"] == ("/root/.cache/huggingface/hub")
    assert worker["APP_RERANK_VLLM_PYTHON_BIN"] == "python3"
    assert rerank["VLLM_STRICT_PROTECTED_ARGS"] == "1"
    assert rerank["VLLM_TASK"] == "score"
    assert embed["VLLM_TRUST_REMOTE_CODE"] == "1"
    assert rerank["VLLM_TRUST_REMOTE_CODE"] == "1"
    assert worker["KG_GLINER2_MODEL"] == GLINER_MODEL
    assert worker["KG_GLINER2_REVISION"] == GLINER_REVISION
    assert worker["KG_GLINER2_DEVICE"] == "cpu"
    assert worker["KG_GLINER2_BATCH_SIZE"] == "8"
    assert worker["KG_GLINER2_MAX_BATCH_CHARACTERS"] == "64000"
