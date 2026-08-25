import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "deploy/scripts/vllm_readiness.py"


class _Response:
    def __init__(self, body: bytes = b"{}") -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _load_readiness_module():
    spec = importlib.util.spec_from_file_location("vllm_readiness", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readiness_prewarms_thinking_and_non_thinking_before_marking_ready(
    tmp_path, monkeypatch
):
    readiness = _load_readiness_module()
    marker = tmp_path / "ready"
    requests = []

    def opener(request, timeout):
        requests.append((request, timeout))
        if isinstance(request, str):
            return _Response()
        return _Response(
            json.dumps(
                {"choices": [{"finish_reason": "stop"}], "usage": {}}
            ).encode()
        )

    monkeypatch.setenv("VLLM_SERVED_MODEL_NAME", "qwen3.6:27b-mtp-awq")
    monkeypatch.setenv("VLLM_API_KEY", "test-key")

    readiness.check_readiness(marker_path=marker, opener=opener)

    assert marker.read_text(encoding="utf-8") == "ready\n"
    assert requests[0] == ("http://127.0.0.1:8000/health", 2)
    payloads = [json.loads(request.data) for request, _ in requests[1:]]
    assert [payload["chat_template_kwargs"]["enable_thinking"] for payload in payloads] == [
        True,
        False,
    ]
    assert [payload["max_tokens"] for payload in payloads] == [128, 8]
    assert all(payload["model"] == "qwen3.6:27b-mtp-awq" for payload in payloads)
    assert all(
        request.headers["Authorization"] == "Bearer test-key"
        for request, _ in requests[1:]
    )


def test_readiness_marker_skips_repeated_generation(tmp_path):
    readiness = _load_readiness_module()
    marker = tmp_path / "ready"
    marker.write_text("ready\n", encoding="utf-8")
    requests = []

    def opener(request, timeout):
        requests.append((request, timeout))
        return _Response()

    readiness.check_readiness(marker_path=marker, opener=opener)

    assert requests == [("http://127.0.0.1:8000/health", 2)]


def test_readiness_does_not_mark_ready_when_a_prewarm_fails(tmp_path, monkeypatch):
    readiness = _load_readiness_module()
    marker = tmp_path / "ready"
    calls = 0

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("second prewarm failed")
        return _Response(
            json.dumps({"choices": [{"finish_reason": "stop"}]}).encode()
        )

    monkeypatch.setenv("VLLM_SERVED_MODEL_NAME", "qwen3.6:27b-mtp-awq")

    with pytest.raises(TimeoutError, match="second prewarm failed"):
        readiness.check_readiness(marker_path=marker, opener=opener)

    assert not marker.exists()


def test_main_vllm_healthcheck_runs_the_readiness_gate():
    for compose_name in ("base.yml", "development.yml", "production.yml"):
        compose = (REPO_ROOT / "deploy/compose" / compose_name).read_text(
            encoding="utf-8"
        )
        assert compose.count(
            "../../deploy/scripts/vllm_readiness.py:"
            "/opt/aquillm/vllm_readiness.py:ro"
        ) == 1
        assert compose.count(
            'test: ["CMD", "python3", "/opt/aquillm/vllm_readiness.py"]'
        ) == 1
        assert compose.count("VLLM_READINESS_PREWARM=1") == 1
