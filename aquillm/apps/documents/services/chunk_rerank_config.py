"""Environment-driven rerank HTTP client settings."""

from __future__ import annotations

from os import getenv


def rerank_provider() -> str:
    return (getenv("APP_RERANK_PROVIDER") or "auto").strip().lower()


def rerank_base_url() -> str:
    base_url = (
        getenv("APP_RERANK_BASE_URL")
        or getenv("VLLM_RERANK_BASE_URL")
        or getenv("VLLM_BASE_URL")
        or "http://vllm_rerank:8000/v1"
    ).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def rerank_api_key() -> str:
    return (
        getenv("APP_RERANK_API_KEY")
        or getenv("VLLM_RERANK_API_KEY")
        or getenv("VLLM_API_KEY")
        or "EMPTY"
    )


def rerank_model() -> str:
    return (
        getenv("APP_RERANK_MODEL")
        or getenv("VLLM_RERANK_MODEL")
        or "Qwen/Qwen3-Reranker-4B"
    )


def rerank_model_revision() -> str:
    return (getenv("APP_RERANK_MODEL_REVISION") or "").strip()


def rerank_vllm_model() -> str:
    return (getenv("APP_RERANK_VLLM_MODEL") or "").strip()


def rerank_tokenizer() -> str:
    return (getenv("APP_RERANK_TOKENIZER") or "").strip()


def rerank_tokenizer_revision() -> str:
    return (getenv("APP_RERANK_TOKENIZER_REVISION") or "").strip()


def rerank_code_revision() -> str:
    return (getenv("APP_RERANK_CODE_REVISION") or "").strip()


def rerank_extra_args() -> str:
    return getenv("APP_RERANK_VLLM_EXTRA_ARGS") or ""


def rerank_trust_remote_code() -> str:
    return (getenv("APP_RERANK_VLLM_TRUST_REMOTE_CODE") or "").strip()


def rerank_runner() -> str:
    return (getenv("APP_RERANK_VLLM_RUNNER") or "").strip()


def rerank_task() -> str:
    return (getenv("APP_RERANK_VLLM_TASK") or "").strip()


def rerank_dtype() -> str:
    return (getenv("APP_RERANK_VLLM_DTYPE") or "").strip()


def rerank_tensor_parallel_size() -> str:
    return (getenv("APP_RERANK_TENSOR_PARALLEL_SIZE") or "").strip()


def rerank_gpu_memory_utilization() -> str:
    return (getenv("APP_RERANK_GPU_MEMORY_UTILIZATION") or "").strip()


def rerank_max_model_len() -> str:
    return (getenv("APP_RERANK_MAX_MODEL_LEN") or "").strip()


def rerank_timeout_seconds() -> int:
    try:
        timeout = int((getenv("APP_RERANK_TIMEOUT_SECONDS") or "3").strip())
    except Exception:
        timeout = 3
    return timeout if timeout > 0 else 3


def rerank_model_is_qwen3_vl() -> bool:
    model_name = (rerank_model() or "").lower()
    return "qwen3-vl-reranker" in model_name


def rerank_doc_char_limit() -> int:
    raw = (getenv("APP_RERANK_DOC_CHAR_LIMIT") or "").strip()
    if not raw:
        return 2000
    try:
        value = int(raw)
        return value if value > 0 else 2000
    except Exception:
        return 2000


__all__ = [
    "rerank_api_key",
    "rerank_base_url",
    "rerank_code_revision",
    "rerank_doc_char_limit",
    "rerank_dtype",
    "rerank_extra_args",
    "rerank_gpu_memory_utilization",
    "rerank_model",
    "rerank_model_is_qwen3_vl",
    "rerank_model_revision",
    "rerank_max_model_len",
    "rerank_provider",
    "rerank_runner",
    "rerank_task",
    "rerank_tokenizer",
    "rerank_tokenizer_revision",
    "rerank_timeout_seconds",
    "rerank_tensor_parallel_size",
    "rerank_trust_remote_code",
    "rerank_vllm_model",
]
