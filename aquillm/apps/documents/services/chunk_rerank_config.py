"""Environment-driven rerank HTTP client settings."""
from __future__ import annotations

from os import getenv


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
        return 900
    try:
        value = int(raw)
        return value if value > 0 else 900
    except Exception:
        return 900


def rerank_pair_token_limit() -> int:
    raw = (
        getenv("APP_RERANK_PAIR_TOKEN_LIMIT")
        or getenv("APP_RERANK_MAX_MODEL_LEN")
        or "1024"
    ).strip()
    try:
        return max(64, int(raw))
    except Exception:
        return 1024


def rerank_template_reserve_tokens() -> int:
    raw = (getenv("APP_RERANK_TEMPLATE_RESERVE_TOKENS") or "96").strip()
    try:
        return max(0, int(raw))
    except Exception:
        return 96


def rerank_score_concurrency() -> int:
    raw = (getenv("APP_RERANK_SCORE_CONCURRENCY") or "6").strip()
    try:
        return min(16, max(1, int(raw)))
    except Exception:
        return 6


__all__ = [
    "rerank_api_key",
    "rerank_base_url",
    "rerank_doc_char_limit",
    "rerank_model",
    "rerank_model_is_qwen3_vl",
    "rerank_pair_token_limit",
    "rerank_score_concurrency",
    "rerank_template_reserve_tokens",
    "rerank_timeout_seconds",
]
