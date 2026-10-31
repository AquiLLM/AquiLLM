"""Real loopback SSE tests: chunks/drafts must never substitute for usage."""
import importlib.util
import json
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/benchmark_kv_offloading.py"


def benchmark():
    if not SCRIPT.exists():
        pytest.fail("benchmark runner is missing")
    spec = importlib.util.spec_from_file_location("kv_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(n=2, t=16, output=8):
    identity = dict(model="fixture-model", revision="fixture-revision", runtime="fixture-runtime",
                    quantization="awq", mtp="depth-4", hardware="loopback-cpu",
                    genesis="fixture-genesis", weights="gpu-resident")
    profile = dict(active_sequences=n, context_tokens=t, retained_contexts=n,
                   kv_dtype="turboquant_k8v4", storage_mode="off", execution_mode="resident",
                   tier_budgets={}, identity=identity)
    fixture = dict(model=identity["model"], revision=identity["revision"],
                   prompts=[[i + 1] * (t - output) for i in range(n)])
    return profile, fixture


@contextmanager
def server(mode="normal", delay=0.03):
    captured = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured.append((payload, self.headers.get("X-Request-ID")))
            if mode == "slow_headers":
                try:
                    self.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nX-Slow: ")
                    self.wfile.flush()
                    for _ in range(10):
                        time.sleep(0.025)
                        self.wfile.write(b"x")
                        self.wfile.flush()
                    self.wfile.write(b"\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if mode == "redirect":
                self.send_response(307)
                self.send_header("Location", "http://127.0.0.1:1/stolen")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            def emit(data):
                raw = ("data: " + json.dumps(data) + "\r\n\r\n").encode()
                # Fragment in the middle of fields and CRLF boundaries.
                for start in range(0, len(raw), 7):
                    self.wfile.write(raw[start:start + 7])
                    self.wfile.flush()
            try:
                emit({"choices": [{"index": 0, "text": "accepted MTP chunk", "finish_reason": None}],
                      "draft_tokens": 1000})
                time.sleep(delay if mode != "slow" or payload["prompt"][0] == 1 else 0.3)
                emit({"choices": [{"index": 0, "text": "second", "finish_reason": "length"}]})
                if mode == "error":
                    emit({"error": {"message": "sensitive server response"}})
                if mode not in {"missing_usage", "truncated"}:
                    count = len(payload["prompt"]) + (1 if mode == "prompt_mismatch" else 0)
                    emit({"choices": [], "usage": {"completion_tokens": payload["max_tokens"],
                                                    "prompt_tokens": count}})
                if mode != "truncated":
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1/completions", captured
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def run(url, **kwargs):
    profile, fixture = inputs()
    return benchmark().run_benchmark(profile, fixture, url, "fixture-model", output_tokens=8,
                                     timeout=2, **kwargs)


def test_fragmented_interleaved_streams_count_usage_not_mtp_chunks_or_drafts():
    with server() as (url, captured):
        result = run(url)
    assert len(captured) == 2
    assert result["client_overlap"]["max_requests"] == 2
    assert result["evidence"]["active_decoders"]["status"] == "unverified"
    assert result["evidence"]["active_paging"]["status"] == "unverified"
    for row in result["requests"]:
        assert row["accepted_output_tokens"] == 8
        assert row["token_accounting_source"] == "server_usage.completion_tokens"
        assert row["nonempty_chunks"] == 2
        assert row["accepted_tokens_per_second"] == pytest.approx(8 / row["elapsed_seconds"])
        assert row["decode_only_tokens_per_second"] is None
        assert row["max_stream_gap_seconds"] >= 0.025
        assert row["exercised_context_tokens"] == 16
        assert row["error"] is None


@pytest.mark.parametrize("mode,error", [("error", "server_error"), ("truncated", "missing_done"),
                                       ("prompt_mismatch", "prompt_usage_mismatch"),
                                       ("redirect", "http_status_307")])
def test_errors_are_redacted_and_have_no_rate(mode, error):
    with server(mode) as (url, _):
        result = run(url)
    assert all(row["error"] == error for row in result["requests"])
    assert all(row["accepted_tokens_per_second"] is None for row in result["requests"])
    assert "sensitive" not in json.dumps(result)


def test_missing_usage_never_counts_sse_events():
    with server("missing_usage") as (url, _):
        result = run(url)
    assert result["rate_target"]["status"] == "unavailable"
    assert all(row["accepted_output_tokens"] is None for row in result["requests"])


def test_serial_requests_never_prove_concurrency():
    with server() as (url, _):
        result = run(url, concurrency=1)
    assert result["client_overlap"] == {"max_requests": 1, "configured_requests": 2}
    assert result["evidence"]["active_decoders"]["status"] == "unverified"


def test_aggregate_throughput_cannot_hide_slow_user():
    with server("slow", delay=0.01) as (url, _):
        result = run(url)
    assert result["requests"][0]["rate_target_status"] == "pass"
    assert result["requests"][1]["rate_target_status"] == "fail"
    assert result["rate_target"]["status"] == "fail"


@pytest.mark.parametrize("change", ["overflow", "duplicate", "bad_ids", "wrong_model", "zero"])
def test_invalid_workloads_are_rejected_before_http(change):
    profile, fixture = inputs()
    if change == "overflow":
        fixture["prompts"][1].append(1)
    elif change == "duplicate":
        fixture["prompts"][1] = fixture["prompts"][0]
    elif change == "bad_ids":
        fixture["prompts"][1][0] = True
    elif change == "wrong_model":
        fixture["model"] = "other"
    else:
        profile["active_sequences"] = 0
    with server() as (url, captured):
        with pytest.raises(ValueError):
            benchmark().run_benchmark(profile, fixture, url, "fixture-model", output_tokens=8)
        assert captured == []


def test_short_prompts_do_not_qualify_configured_long_context():
    profile, fixture = inputs(t=262144)
    fixture["prompts"] = [[1] * 8, [2] * 8]
    with server() as (url, _):
        result = benchmark().run_benchmark(profile, fixture, url, "fixture-model", output_tokens=8)
    assert result["capacity_target"]["status"] == "fail"
    assert [row["reserved_context_tokens"] for row in result["requests"]] == [16, 16]


def test_cli_dry_run_and_loopback_smoke(tmp_path):
    profile, fixture = inputs()
    profile_path, fixture_path, output = [tmp_path / name for name in ("profile.json", "prompts.json", "run.json")]
    profile_path.write_text(json.dumps(profile))
    fixture_path.write_text(json.dumps(fixture))
    with server() as (url, captured):
        command = [sys.executable, str(SCRIPT), "--profile", str(profile_path), "--prompts", str(fixture_path),
                   "--url", url, "--model", "fixture-model", "--output-tokens", "8", "--output", str(output)]
        dry = subprocess.run(command + ["--dry-run"], capture_output=True, text=True, timeout=10)
        assert dry.returncode == 0, dry.stderr
        assert captured == []
        assert json.loads(output.read_text())["status"] == "dry-run"
        live = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert live.returncode == 0, live.stderr
    assert len(captured) == 2
    assert len(json.loads(output.read_text())["requests"]) == 2
