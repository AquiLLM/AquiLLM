#!/usr/bin/env python3
"""Gate vLLM readiness on one-time thinking and non-thinking prewarms."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable


HEALTH_URL = "http://127.0.0.1:8000/health"
CHAT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MARKER_PATH = Path("/tmp/aquillm-vllm-ready")
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _prewarm_enabled() -> bool:
    return os.getenv("VLLM_READINESS_PREWARM", "1").strip().lower() in _TRUE_VALUES


def _served_model_name() -> str:
    model = os.getenv("VLLM_SERVED_MODEL_NAME", "").split(",", 1)[0].strip()
    if not model:
        raise RuntimeError("VLLM_SERVED_MODEL_NAME is required for readiness prewarm")
    return model


def _prewarm_request(*, model: str, enable_thinking: bool, max_tokens: int):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Respond with exactly: OK"}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    api_key = os.getenv("VLLM_API_KEY", "").strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return urllib.request.Request(
        CHAT_URL,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )


def _write_ready_marker(marker_path: Path) -> None:
    temporary = marker_path.with_name(f".{marker_path.name}.tmp")
    temporary.write_text("ready\n", encoding="utf-8")
    temporary.replace(marker_path)


def check_readiness(
    *,
    marker_path: Path = DEFAULT_MARKER_PATH,
    opener: Callable = urllib.request.urlopen,
) -> None:
    """Raise on unavailable/unwarmed vLLM; return only when it is ready."""
    with opener(HEALTH_URL, timeout=2) as response:
        response.read()

    if marker_path.exists():
        return
    if not _prewarm_enabled():
        _write_ready_marker(marker_path)
        return

    model = _served_model_name()
    for enable_thinking, max_tokens in ((True, 128), (False, 8)):
        request = _prewarm_request(
            model=model,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
        )
        with opener(request, timeout=170) as response:
            result = json.loads(response.read())
        if not result.get("choices"):
            mode = "thinking" if enable_thinking else "non-thinking"
            raise RuntimeError(f"vLLM {mode} readiness prewarm returned no choices")

    _write_ready_marker(marker_path)


def main() -> int:
    try:
        check_readiness()
    except Exception as error:
        print(f"vLLM readiness pending: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
