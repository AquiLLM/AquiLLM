from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
COMPOSE_DIR = REPO / "deploy" / "compose"
EVAL_OVERRIDE = COMPOSE_DIR / "knowledge-graph-eval.yml"
PROJECT = "aquillm-kg-eval-static-test"
EMBED_MODEL = "Qwen/test-embedding"
EMBED_REVISION = "a" * 40
RERANK_MODEL = "Qwen/test-reranker"
RERANK_REVISION = "b" * 40
GLINER_MODEL = "fastino/test-gliner"
GLINER_REVISION = "c" * 40
EMBED_EXTRA_ARGS = (
    "--quantization bitsandbytes --load-format bitsandbytes "
    "--model-loader-extra-config "
    '\'{"load_in_4bit":true,"bnb_4bit_compute_dtype":"float16",'
    '"bnb_4bit_quant_type":"nf4","bnb_4bit_use_double_quant":true}\' '
    "--hf-overrides "
    "'{\"matryoshka_dimensions\":[1024]}'"
)
RERANK_EXTRA_ARGS = (
    "--chat-template /templates/qwen3_vl_reranker.jinja --hf-overrides "
    '\'{"architectures":["Qwen3VLForSequenceClassification"],'
    '"classifier_from_token":["no","yes"],'
    '"is_original_qwen3_reranker":true}\''
)
HOSTILE_SIDECAR_ENV = {
    "VLLM_EXTRA_ARGS": "--revision must-not-win",
    "MEM0_EMBED_VLLM_EXTRA_ARGS": "--revision must-not-win",
    "MEM0_EMBED_TENSOR_PARALLEL_SIZE": "99",
    "MEM0_EMBED_GPU_MEMORY_UTILIZATION": "0.99",
    "MEM0_EMBED_MAX_MODEL_LEN": "99999",
    "APP_RERANK_VLLM_EXTRA_ARGS": "--revision must-not-win",
    "APP_RERANK_TENSOR_PARALLEL_SIZE": "99",
    "APP_RERANK_GPU_MEMORY_UTILIZATION": "0.08",
    "APP_RERANK_MAX_MODEL_LEN": "99999",
    "LMCACHE_ENABLED": "1",
    "LMCACHE_EXTRA_ARGS": "--revision must-not-win",
    "VLLM_TRUST_REMOTE_CODE": "0",
    "MEM0_EMBED_VLLM_TRUST_REMOTE_CODE": "0",
    "APP_RERANK_VLLM_TRUST_REMOTE_CODE": "0",
    "APP_EMBED_VLLM_STRICT_PROTECTED_ARGS": "0",
    "APP_EMBED_VLLM_API_KEY": "must-not-win",
    "APP_EMBED_VLLM_DOWNLOAD_DIR": "/must-not-win",
    "APP_EMBED_VLLM_PYTHON_BIN": "must-not-win",
    "APP_RERANK_VLLM_STRICT_PROTECTED_ARGS": "0",
    "APP_RERANK_VLLM_TASK": "embed",
    "APP_RERANK_VLLM_DOWNLOAD_DIR": "/must-not-win",
    "APP_RERANK_VLLM_PYTHON_BIN": "must-not-win",
    "VLLM_STRICT_PROTECTED_ARGS": "0",
    "VLLM_API_KEY": "must-not-win",
    "VLLM_DOWNLOAD_DIR": "/must-not-win",
    "VLLM_PYTHON_BIN": "must-not-win",
    "PYTHONDONTWRITEBYTECODE": "0",
}
ENV_FILE_SERVICES = frozenset(
    {
        "web",
        "worker",
        "worker_knowledge_graph",
        "worker_memory_promotion",
        "nginx",
        "get_certs",
        "db",
        "storage",
        "vllm",
        "vllm_ocr",
        "vllm_transcribe",
        "vllm_embed",
        "vllm_rerank",
    }
)
DOCKER_CLIENT_ENV = frozenset(
    {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "TMPDIR",
        "HOME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "APPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "COMMONPROGRAMFILES",
        "COMMONPROGRAMFILES(X86)",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
        "DOCKER_CERT_PATH",
        "DOCKER_TLS_VERIFY",
        "DOCKER_API_VERSION",
        "DOCKER_DEFAULT_PLATFORM",
        "DOCKER_CLI_PLUGIN_EXTRA_DIRS",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
    }
)


def reviewed_env(tmp_path: Path, overrides: dict[str, str] | None = None) -> Path:
    path = (tmp_path / "task21-reviewed.env").resolve()
    values = {
        "POSTGRES_NAME": "task21_database",
        "POSTGRES_USER": "task21_role",
        "POSTGRES_PASSWORD": "task21-test-password",
        "STORAGE_ACCESS_KEY": "task21-storage",
        "STORAGE_SECRET_KEY": "task21-storage-secret",
        "APP_EMBED_MODEL": EMBED_MODEL,
        "APP_EMBED_MODEL_REVISION": EMBED_REVISION,
        "APP_EMBED_TOKENIZER_REVISION": EMBED_REVISION,
        "APP_EMBED_CODE_REVISION": EMBED_REVISION,
        "APP_EMBED_VLLM_RUNNER": "pooling",
        "APP_EMBED_VLLM_DTYPE": "float16",
        "APP_EMBED_VLLM_API_KEY": "EMPTY",
        "APP_EMBED_VLLM_DOWNLOAD_DIR": "/root/.cache/huggingface/hub",
        "APP_EMBED_VLLM_PYTHON_BIN": "python3",
        "APP_EMBED_BASE_URL": "http://vllm_embed:8000/v1",
        "APP_EMBED_API_KEY": "EMPTY",
        "APP_EMBED_DIMS": "1024",
        "APP_EMBED_ALLOW_DIMENSIONS_OVERRIDE": "0",
        "APP_RERANK_PROVIDER": "local",
        "APP_RERANK_BASE_URL": "http://vllm_rerank:8000/v1",
        "APP_RERANK_API_KEY": "EMPTY",
        "APP_RERANK_MODEL": RERANK_MODEL,
        "APP_RERANK_VLLM_MODEL": RERANK_MODEL,
        "APP_RERANK_TOKENIZER": RERANK_MODEL,
        "APP_RERANK_MODEL_REVISION": RERANK_REVISION,
        "APP_RERANK_TOKENIZER_REVISION": RERANK_REVISION,
        "APP_RERANK_CODE_REVISION": RERANK_REVISION,
        "APP_RERANK_VLLM_RUNNER": "pooling",
        "APP_RERANK_VLLM_DTYPE": "float16",
        "APP_RERANK_VLLM_DOWNLOAD_DIR": "/root/.cache/huggingface/hub",
        "APP_RERANK_VLLM_PYTHON_BIN": "python3",
        "KG_EXTRACTOR_PROVIDER": "gliner2_local",
        "KG_EXTRACTOR_FAIL_OPEN": "0",
        "KG_GLINER2_MODEL": GLINER_MODEL,
        "KG_GLINER2_REVISION": GLINER_REVISION,
        "KG_GLINER2_DEVICE": "cpu",
        "KG_GLINER2_BATCH_SIZE": "8",
        "KG_GLINER2_MAX_BATCH_CHARACTERS": "64000",
        "KG_GLINER2_LOCAL_FILES_ONLY": "1",
        "KG_EXTRACTION_QUEUE": "knowledge-graph-eval-static-test",
        "DJANGO_DEBUG": "1",
        "KG_EVAL_BYPASS_ALLOWED": "1",
        "KG_BUILD_ENABLED": "0",
        "KG_OVERLAY_ENABLED": "0",
        "DJANGO_CACHE_REDIS_URL": "redis://shared.invalid:6379/9",
        "RAG_CACHE_ENABLED": "0",
        "HF_HOME": "/must-not-win",
        "HF_HUB_CACHE": "/must-not-win",
        "TRANSFORMERS_CACHE": "/must-not-win",
        "KG_GLINER2_CACHE_DIR": "/must-not-win",
        "MEM0_EMBED_VLLM_EXTRA_ARGS": EMBED_EXTRA_ARGS,
        "MEM0_EMBED_TENSOR_PARALLEL_SIZE": "1",
        "MEM0_EMBED_GPU_MEMORY_UTILIZATION": "0.20",
        "MEM0_EMBED_MAX_MODEL_LEN": "2048",
        "APP_RERANK_VLLM_EXTRA_ARGS": RERANK_EXTRA_ARGS,
        "APP_RERANK_TENSOR_PARALLEL_SIZE": "1",
        "APP_RERANK_GPU_MEMORY_UTILIZATION": "0.25",
        "APP_RERANK_MAX_MODEL_LEN": "1024",
        "LMCACHE_ENABLED": "0",
        "LMCACHE_EXTRA_ARGS": "",
        "MEM0_EMBED_VLLM_TRUST_REMOTE_CODE": "1",
        "APP_RERANK_VLLM_TRUST_REMOTE_CODE": "1",
        "COHERE_KEY": "must-not-win",
    }
    values.update(overrides or {})
    path.write_text(
        "".join(f"{name}={value}\n" for name, value in values.items()),
        encoding="utf-8",
    )
    return path


def compose_command(env_file: Path, *files: Path) -> tuple[list[str], dict[str, str]]:
    assert shutil.which("docker"), "Docker CLI is required for rendered Compose tests"
    command = [
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "--project-name",
        PROJECT,
    ]
    for file in files:
        command.extend(("-f", str(file)))
    command.extend(("--profile", "*"))
    environment = {
        name: os.environ[name] for name in DOCKER_CLIENT_ENV if name in os.environ
    }
    environment["TASK21_ENV_FILE"] = str(env_file)
    return command, environment


def run_compose(env_file: Path, *files: Path, args=("config", "--format", "json")):
    command, environment = compose_command(env_file, *files)
    return subprocess.run(
        [*command, *args],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def render(env_file: Path, *files: Path) -> dict:
    result = run_compose(env_file, *files)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def render_without_env_resolution(env_file: Path, *files: Path) -> dict:
    result = run_compose(
        env_file,
        *files,
        args=("config", "--no-env-resolution", "--format", "json"),
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def env_file_shim(tmp_path: Path, env_file: Path, compose_file: Path) -> Path:
    services = []
    current = None
    in_services = False
    for line in compose_file.read_text(encoding="utf-8").splitlines():
        if line == "services:":
            in_services = True
        elif in_services and line and not line.startswith(" "):
            in_services = False
        elif in_services and line.startswith("  ") and not line.startswith("    "):
            current = line.strip().removesuffix(":")
        elif in_services and current and line.strip() == "env_file:":
            services.append(current)
    shim = tmp_path / f"{compose_file.stem}-env-shim.yml"
    body = ["services:"]
    for service in sorted(set(services)):
        body.extend(
            (
                f"  {service}:",
                "    env_file: !override",
                f"      - '{env_file.as_posix()}'",
            )
        )
    shim.write_text("\n".join(body) + "\n", encoding="utf-8")
    return shim


def compose_service_source(compose_file: Path, service: str) -> str:
    lines = compose_file.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"  {service}:") + 1
    end = len(lines)
    for index in range(start, len(lines)):
        line = lines[index]
        if line and (
            not line.startswith(" ")
            or (line.startswith("  ") and not line.startswith("    "))
        ):
            end = index
            break
    return "\n".join(lines[start:end])


def env_file_path(service: dict) -> str:
    entry = service["env_file"][0]
    return entry["path"] if isinstance(entry, dict) else entry


def volume_by_target(service: dict) -> dict[str, dict]:
    return {row["target"]: row for row in service.get("volumes", ())}
