"""Resolved request controls shared by dispatch and evaluation snapshots."""

from os import getenv


def request_limits():
    timeout = float(getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "120"))
    try:
        maximum = float(getenv("OPENAI_REQUEST_TIMEOUT_MAX_SECONDS", "360"))
    except Exception:
        maximum = 360.0
    if maximum < timeout:
        maximum = timeout
    try:
        overflow = int(getenv("OPENAI_CONTEXT_OVERFLOW_RETRIES", "3"))
    except Exception:
        overflow = 3
    try:
        retries = int(getenv("OPENAI_TIMEOUT_RETRIES", "2"))
    except Exception:
        retries = 2
    return timeout, maximum, max(3, overflow), max(0, retries)


def stream_enabled():
    return getenv("OPENAI_STREAM_RESPONSES", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def context_limit():
    raw = (getenv("OPENAI_CONTEXT_LIMIT", "") or "").strip() or (
        getenv("VLLM_MAX_MODEL_LEN", "") or ""
    ).strip()
    try:
        return int(raw)
    except Exception:
        return 0


def optional_float(name):
    raw = (getenv(name, "") or "").strip()
    try:
        return float(raw) if raw else None
    except Exception:
        return None
