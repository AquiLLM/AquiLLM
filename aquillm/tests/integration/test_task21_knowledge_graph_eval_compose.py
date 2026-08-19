from __future__ import annotations

import os
from pathlib import Path

from tests.integration.task21_compose_test_support import (
    COMPOSE_DIR,
    EMBED_EXTRA_ARGS,
    EMBED_MODEL,
    EMBED_REVISION,
    ENV_FILE_SERVICES,
    EVAL_OVERRIDE,
    GLINER_MODEL,
    GLINER_REVISION,
    HOSTILE_SIDECAR_ENV,
    PROJECT,
    RERANK_EXTRA_ARGS,
    RERANK_MODEL,
    RERANK_REVISION,
    compose_command,
    env_file_path,
    render,
    render_without_env_resolution,
    reviewed_env,
    run_compose,
    volume_by_target,
)


def test_eval_override_renders_isolated_local_contract(tmp_path) -> None:
    env_file = reviewed_env(tmp_path, HOSTILE_SIDECAR_ENV)
    config = render(env_file, COMPOSE_DIR / "development.yml", EVAL_OVERRIDE)
    path_config = render_without_env_resolution(
        env_file, COMPOSE_DIR / "development.yml", EVAL_OVERRIDE
    )
    services = config["services"]
    path_services = path_config["services"]
    assert ENV_FILE_SERVICES <= services.keys()
    for name in ENV_FILE_SERVICES:
        assert "env_file" in path_services[name], name
    assert {name: env_file_path(path_services[name]) for name in ENV_FILE_SERVICES} == {
        name: str(env_file) for name in ENV_FILE_SERVICES
    }
    assert "../../.env" not in str(path_config)
    assert services["db"]["restart"] == "no"
    assert services["redis"]["restart"] == "no"
    assert services["worker_knowledge_graph"]["restart"] == "no"
    assert services["vllm_embed"]["restart"] == "no"
    assert services["vllm_rerank"]["restart"] == "no"

    database = services["db"]
    worker = services["worker_knowledge_graph"]
    assert database["environment"]["POSTGRES_DB"] == "task21_database"
    assert database["environment"]["POSTGRES_USER"] == "task21_role"
    assert database["environment"]["POSTGRES_PASSWORD"] == "task21-test-password"
    assert worker["environment"]["POSTGRES_DB"] == "task21_database"
    assert (
        worker["environment"]["POSTGRES_NAME"] == database["environment"]["POSTGRES_DB"]
    )
    assert (
        worker["environment"]["POSTGRES_USER"]
        == database["environment"]["POSTGRES_USER"]
    )
    assert (
        worker["environment"]["POSTGRES_PASSWORD"]
        == database["environment"]["POSTGRES_PASSWORD"]
    )
    assert worker["environment"]["POSTGRES_HOST"] == "db"
    assert worker["environment"]["POSTGRES_PORT"] == "5432"
    assert worker["environment"]["DJANGO_CACHE_REDIS_URL"] == "redis://redis:6379/1"
    assert worker["environment"]["RAG_CACHE_ENABLED"] == "1"
    assert worker["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
    health = " ".join(database["healthcheck"]["test"])
    assert "POSTGRES_USER" in health and "POSTGRES_DB" in health
    assert "aquillm" not in health and "postgres" not in health
    assert database["healthcheck"]["interval"] == "10s"
    assert database["healthcheck"]["timeout"] == "5s"
    assert database["healthcheck"]["retries"] == 12
    assert database["healthcheck"]["start_period"] == "20s"

    worker_env = worker["environment"]
    assert worker_env["COHERE_KEY"] == ""
    assert worker_env["APP_EMBED_BASE_URL"] == "http://vllm_embed:8000/v1"
    assert worker_env["APP_EMBED_API_KEY"] == "EMPTY"
    assert worker_env["APP_EMBED_MODEL"] == EMBED_MODEL
    assert worker_env["APP_EMBED_MODEL_REVISION"] == EMBED_REVISION
    assert worker_env["APP_EMBED_TOKENIZER_REVISION"] == EMBED_REVISION
    assert worker_env["APP_EMBED_CODE_REVISION"] == EMBED_REVISION
    assert worker_env["APP_EMBED_VLLM_RUNNER"] == "pooling"
    assert worker_env["APP_EMBED_VLLM_DTYPE"] == "float16"
    assert worker_env["APP_EMBED_VLLM_STRICT_PROTECTED_ARGS"] == "1"
    assert worker_env["APP_EMBED_VLLM_API_KEY"] == "EMPTY"
    assert worker_env["APP_EMBED_VLLM_DOWNLOAD_DIR"] == ("/root/.cache/huggingface/hub")
    assert worker_env["APP_EMBED_VLLM_PYTHON_BIN"] == "python3"
    assert worker_env["MEM0_EMBED_VLLM_EXTRA_ARGS"] == EMBED_EXTRA_ARGS
    assert worker_env["MEM0_EMBED_VLLM_TRUST_REMOTE_CODE"] == "1"
    assert worker_env["MEM0_EMBED_TENSOR_PARALLEL_SIZE"] == "1"
    assert worker_env["MEM0_EMBED_GPU_MEMORY_UTILIZATION"] == "0.20"
    assert worker_env["MEM0_EMBED_MAX_MODEL_LEN"] == "2048"
    assert worker_env["APP_EMBED_DIMS"] == "1024"
    assert worker_env["APP_EMBED_ALLOW_DIMENSIONS_OVERRIDE"] == "0"
    assert worker_env["APP_RERANK_PROVIDER"] == "local"
    assert worker_env["APP_RERANK_BASE_URL"] == "http://vllm_rerank:8000/v1"
    assert worker_env["APP_RERANK_API_KEY"] == "EMPTY"
    assert worker_env["APP_RERANK_MODEL"] == RERANK_MODEL
    assert worker_env["APP_RERANK_VLLM_MODEL"] == RERANK_MODEL
    assert worker_env["APP_RERANK_TOKENIZER"] == RERANK_MODEL
    assert worker_env["APP_RERANK_MODEL_REVISION"] == RERANK_REVISION
    assert worker_env["APP_RERANK_TOKENIZER_REVISION"] == RERANK_REVISION
    assert worker_env["APP_RERANK_CODE_REVISION"] == RERANK_REVISION
    assert worker_env["APP_RERANK_VLLM_RUNNER"] == "pooling"
    assert worker_env["APP_RERANK_VLLM_DTYPE"] == "float16"
    assert worker_env["APP_RERANK_VLLM_STRICT_PROTECTED_ARGS"] == "1"
    assert worker_env["APP_RERANK_VLLM_TASK"] == "score"
    assert worker_env["APP_RERANK_VLLM_DOWNLOAD_DIR"] == (
        "/root/.cache/huggingface/hub"
    )
    assert worker_env["APP_RERANK_VLLM_PYTHON_BIN"] == "python3"
    assert worker_env["APP_RERANK_VLLM_EXTRA_ARGS"] == RERANK_EXTRA_ARGS
    assert worker_env["APP_RERANK_VLLM_TRUST_REMOTE_CODE"] == "1"
    assert worker_env["APP_RERANK_TENSOR_PARALLEL_SIZE"] == "1"
    assert worker_env["APP_RERANK_GPU_MEMORY_UTILIZATION"] == "0.25"
    assert worker_env["APP_RERANK_MAX_MODEL_LEN"] == "1024"
    assert worker_env["KG_EXTRACTOR_PROVIDER"] == "gliner2_local"
    assert worker_env["KG_EXTRACTOR_FAIL_OPEN"] == "0"
    assert worker_env["KG_GLINER2_MODEL"] == GLINER_MODEL
    assert worker_env["KG_GLINER2_REVISION"] == GLINER_REVISION
    assert worker_env["KG_GLINER2_DEVICE"] == "cpu"
    assert worker_env["KG_GLINER2_BATCH_SIZE"] == "8"
    assert worker_env["KG_GLINER2_MAX_BATCH_CHARACTERS"] == "64000"
    assert worker_env["KG_GLINER2_LOCAL_FILES_ONLY"] == "1"
    assert worker_env["HF_HOME"] == "/opt/kg-eval-hf-cache"
    assert worker_env["KG_GLINER2_CACHE_DIR"] == "/opt/kg-eval-hf-cache"

    embed = services["vllm_embed"]
    rerank = services["vllm_rerank"]
    for isolated_service in (database, services["redis"], embed, rerank):
        assert not isolated_service.get("ports")
    assert embed["profiles"] == ["vllm"]
    assert rerank["profiles"] == ["vllm"]
    assert worker["profiles"] == ["knowledge-graph"]
    assert embed["environment"]["VLLM_MODEL"] == EMBED_MODEL
    assert embed["environment"]["VLLM_API_KEY"] == worker_env["APP_EMBED_VLLM_API_KEY"]
    assert embed["environment"]["VLLM_API_KEY"] == "EMPTY"
    assert embed["environment"]["VLLM_DOWNLOAD_DIR"] == ("/root/.cache/huggingface/hub")
    assert embed["environment"]["VLLM_PYTHON_BIN"] == "python3"
    assert embed["environment"]["VLLM_SERVED_MODEL_NAME"] == EMBED_MODEL
    assert embed["environment"]["VLLM_TOKENIZER"] == EMBED_MODEL
    assert embed["environment"]["VLLM_REVISION"] == EMBED_REVISION
    assert embed["environment"]["VLLM_TOKENIZER_REVISION"] == EMBED_REVISION
    assert embed["environment"]["VLLM_CODE_REVISION"] == EMBED_REVISION
    assert embed["environment"]["VLLM_RUNNER"] == "pooling"
    assert embed["environment"]["VLLM_DTYPE"] == "float16"
    assert rerank["environment"]["VLLM_MODEL"] == RERANK_MODEL
    assert rerank["environment"]["VLLM_API_KEY"] == worker_env["APP_RERANK_API_KEY"]
    assert (
        rerank["environment"]["VLLM_DOWNLOAD_DIR"]
        == worker_env["APP_RERANK_VLLM_DOWNLOAD_DIR"]
    )
    assert rerank["environment"]["VLLM_API_KEY"] == "EMPTY"
    assert rerank["environment"]["VLLM_DOWNLOAD_DIR"] == (
        "/root/.cache/huggingface/hub"
    )
    assert rerank["environment"]["VLLM_PYTHON_BIN"] == "python3"
    assert rerank["environment"]["VLLM_SERVED_MODEL_NAME"] == RERANK_MODEL
    assert rerank["environment"]["VLLM_TOKENIZER"] == RERANK_MODEL
    assert rerank["environment"]["VLLM_REVISION"] == RERANK_REVISION
    assert rerank["environment"]["VLLM_TOKENIZER_REVISION"] == RERANK_REVISION
    assert rerank["environment"]["VLLM_CODE_REVISION"] == RERANK_REVISION
    assert rerank["environment"]["VLLM_RUNNER"] == "pooling"
    assert rerank["environment"]["VLLM_DTYPE"] == "float16"
    assert rerank["environment"]["VLLM_TASK"] == "score"
    assert embed["environment"]["VLLM_TENSOR_PARALLEL_SIZE"] == "1"
    assert embed["environment"]["VLLM_GPU_MEMORY_UTILIZATION"] == "0.20"
    assert embed["environment"]["VLLM_MAX_MODEL_LEN"] == "2048"
    assert rerank["environment"]["VLLM_TENSOR_PARALLEL_SIZE"] == "1"
    assert rerank["environment"]["VLLM_GPU_MEMORY_UTILIZATION"] == "0.25"
    assert rerank["environment"]["VLLM_MAX_MODEL_LEN"] == "1024"
    assert embed["environment"]["VLLM_EXTRA_ARGS"] == EMBED_EXTRA_ARGS
    assert rerank["environment"]["VLLM_EXTRA_ARGS"] == RERANK_EXTRA_ARGS
    for service in (embed, rerank):
        assert service["environment"]["LMCACHE_ENABLED"] == "0"
        assert service["environment"]["LMCACHE_EXTRA_ARGS"] == ""
        assert service["environment"]["VLLM_STRICT_PROTECTED_ARGS"] == "1"
        assert service["environment"]["VLLM_TRUST_REMOTE_CODE"] == "1"
        for protected in (
            "--revision",
            "--tokenizer-revision",
            "--code-revision",
            "--runner",
            "--dtype",
            "--trust-remote-code",
            "--tensor-parallel-size",
            "--gpu-memory-utilization",
            "--max-model-len",
            "--api-key",
            "--download-dir",
        ):
            assert protected not in service["environment"]["VLLM_EXTRA_ARGS"]
        assert service["environment"]["HF_HOME"] == "/root/.cache/huggingface"
        assert service["environment"]["HF_HUB_CACHE"] == "/root/.cache/huggingface/hub"
        assert service["environment"]["TRANSFORMERS_CACHE"] == (
            "/root/.cache/huggingface/hub"
        )

    worker_volumes = volume_by_target(worker)
    embed_volumes = volume_by_target(embed)
    rerank_volumes = volume_by_target(rerank)
    redis_volumes = volume_by_target(services["redis"])
    db_volumes = volume_by_target(database)
    assert worker_volumes["/app/.venv"]["source"] == "kg_eval_app_venv_shadow"
    assert worker_volumes["/opt/kg-eval-hf-cache"]["source"] == "kg_eval_gliner_cache"
    assert "/root/.cache/huggingface" not in worker_volumes
    assert embed_volumes["/root/.cache/huggingface"]["source"] == "kg_eval_vllm_cache"
    assert rerank_volumes["/root/.cache/huggingface"]["source"] == "kg_eval_vllm_cache"
    assert redis_volumes["/data"]["source"] == "kg_eval_redis_data"
    assert db_volumes["/var/lib/postgresql/data"]["source"] == "postgres_data"

    expected_volumes = {
        "postgres_data",
        "kg_eval_redis_data",
        "kg_eval_app_venv_shadow",
        "kg_eval_gliner_cache",
        "kg_eval_vllm_cache",
    }
    for name in expected_volumes:
        assert config["volumes"][name]["name"] == f"{PROJECT}_{name}"
        assert config["volumes"][name]["labels"]["aquillm.task21.eval"] == "true"
    assert config["networks"]["default"]["name"] == f"{PROJECT}_default"


def test_eval_override_fails_without_reviewed_absolute_env(tmp_path) -> None:
    env_file = reviewed_env(tmp_path)
    command, environment = compose_command(
        env_file, COMPOSE_DIR / "development.yml", EVAL_OVERRIDE
    )
    environment.pop("TASK21_ENV_FILE")
    missing = __import__("subprocess").run(
        [*command, "config", "--quiet"],
        cwd=COMPOSE_DIR.parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert missing.returncode != 0
    assert "TASK21_ENV_FILE" in missing.stderr
    environment["TASK21_ENV_FILE"] = "unreviewed-relative.env"
    relative = __import__("subprocess").run(
        [*command, "config", "--quiet"],
        cwd=COMPOSE_DIR.parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert relative.returncode != 0


def test_eval_override_documents_no_deps_and_label_cleanup_contract() -> None:
    text = EVAL_OVERRIDE.read_text(encoding="utf-8")
    assert "up -d --no-deps vllm_embed" in text
    assert "up -d --no-deps vllm_rerank" in text
    assert "com.docker.compose.project" in text
    assert "--user 0:0" in text
    assert '--user "$(id -u):$(id -g)"' in text
    assert "test -r /opt/kg-eval-hf-cache" in text
    assert "test -w /opt/kg-eval-hf-cache" in text


def test_eval_config_quiet_succeeds_with_reviewed_env(tmp_path) -> None:
    env_file = reviewed_env(tmp_path)
    result = run_compose(
        env_file,
        COMPOSE_DIR / "development.yml",
        EVAL_OVERRIDE,
        args=("config", "--quiet"),
    )
    assert result.returncode == 0, result.stderr
    assert Path(env_file).is_absolute()
    assert os.path.isfile(env_file)
