"""Real local HTTP/SSE cancellation, survivor and recovery evidence."""
import importlib
import json
from pathlib import Path
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def implementation():
    return importlib.import_module("throughput_lifecycle")


@pytest.fixture
def server():
    state = dict(running=0, generated=0, cancelled=False, auth=[], wrong=False,
                 no_overlap=False, missing_done=False, wrong_recovery=False,
                 missing_metrics=False, paths=[])
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            state["auth"].append(self.headers.get("Authorization"))
            with lock:
                running = min(1, state["running"]) if state["no_overlap"] else state["running"]
                generated = state["generated"]
            names = dict(num_requests_running=running, num_requests_waiting=0,
                kv_cache_usage_perc=0.1, num_preemptions_total=0,
                spec_decode_num_drafts_total=generated,
                spec_decode_num_draft_tokens_total=generated * 4,
                spec_decode_num_accepted_tokens_total=generated * 3)
            if state["missing_metrics"]:
                names.pop("num_requests_waiting")
            body = "\n".join(f"vllm:{k}{{model_name=\"synthetic\"}} {v}" for k, v in names.items()).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            state["auth"].append(self.headers.get("Authorization"))
            state["paths"].append(self.path)
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not payload.get("stream"):
                content = payload["messages"][-1]["content"]
                import re
                answer = re.search(r"(?:is |value for this exercise is )(\d+|violet-\d+)", content)
                # Multiturn cases place the value in the first user message.
                if answer is None:
                    answer = re.search(r"is (violet-\d+)", payload["messages"][0]["content"])
                text = "wrong" if state["wrong"] or state["wrong_recovery"] else answer[1]
                body = json.dumps(dict(choices=[dict(message=dict(content=text), finish_reason="stop")],
                    usage=dict(prompt_tokens=512, completion_tokens=3))).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            with lock:
                state["running"] += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()

            def event(value):
                self.wfile.write(("data: " + json.dumps(value) + "\n\n").encode())
                self.wfile.flush()

            cancel = self.path == "/v1/completions"
            try:
                if cancel:
                    for index in range(500):
                        event(dict(id="cancel-request", choices=[dict(text=" synthetic", finish_reason=None)]))
                        with lock:
                            state["generated"] += 1
                        time.sleep(0.02)
                else:
                    import re
                    prompt = payload["messages"][0]["content"]
                    key = re.search(r"archive key.*?is (\d+)\.", prompt)[1]
                    repetitions = int(re.search(r"exactly (\d+) copies", prompt)[1])
                    answer = "wrong" if state["wrong"] else " ".join([key] * repetitions)
                    # Keep the survivor active after cancellation, then emit full SSE.
                    for piece in (answer[:len(answer)//2], answer[len(answer)//2:]):
                        time.sleep(0.3)
                        event(dict(id="survivor-request", choices=[dict(delta=dict(content=piece), finish_reason=None)]))
                    event(dict(id="survivor-request", choices=[dict(delta={}, finish_reason="stop")]))
                    event(dict(id="survivor-request", choices=[], usage=dict(prompt_tokens=41600, completion_tokens=64)))
                    if not state["missing_done"]:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
            except OSError:
                state["cancelled"] = cancel
            finally:
                with lock:
                    state["running"] -= 1
                self.close_connection = True

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    http.daemon_threads = True
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{http.server_port}", state
    http.shutdown()
    http.server_close()
    thread.join(2)


def run_cli(server, tmp_path, monkeypatch):
    base, _ = server
    monkeypatch.setenv("VLLM_API_KEY", "synthetic-private-key")
    output = tmp_path / "evidence.jsonl"
    code = implementation().main(["--base-url", base, "--label", "local", "--output", str(output),
        "--rounds", "1", "--timeout-seconds", "8", "--overlap-seconds", "1.5",
        "--drain-seconds", "4", "--metrics-settle-seconds", "1"])
    return code, json.loads(output.read_text()), output


def test_real_overlap_close_survivor_and_fresh_exact_answers(server, tmp_path, monkeypatch, capsys):
    code, row, output = run_cli(server, tmp_path, monkeypatch)
    assert code == 0 and row["passed"] and row["complete"]
    assert row["overlap_sample"]["values"]["running"] == 2
    assert row["cancelled"]["nonempty_chunks"] >= 3
    assert row["cancelled"]["cancel_requested"] and not row["cancelled"]["stream_done"]
    assert row["cancelled"]["cancelled_at"] < row["survivor"]["captured_at"]
    assert row["survivor"]["stream_done"] and row["survivor"]["passed"]
    assert all(item["passed"] for item in row["recovery"])
    assert len(row["recovery"]) == 6 and row["drain"]["values"]["running"] == 0
    assert row["physical_state_slot_reuse"] == "not_observed"
    assert row["server_abort_confirmation"] == "not_observed"
    assert "synthetic-private-key" not in output.read_text() + capsys.readouterr().out
    assert all(value == "Bearer synthetic-private-key" for value in server[1]["auth"])


def test_fresh_recovery_only_answer_corruption_fails(server, tmp_path, monkeypatch):
    server[1]["wrong_recovery"] = True
    code, row, _ = run_cli(server, tmp_path, monkeypatch)
    assert code == 1 and row["survivor"]["passed"] and not row["passed"]
    assert "recovery_oracle_failed" in row["errors"]
    assert not any(item["passed"] for item in row["recovery"])


def test_missing_instrumentation_stops_before_generation(server, tmp_path, monkeypatch):
    server[1]["missing_metrics"] = True
    code, row, _ = run_cli(server, tmp_path, monkeypatch)
    assert code == 1 and not row["passed"] and not server[1]["paths"]
    assert row["errors"] == ["instrumentation_or_transport:ValueError"]


@pytest.mark.parametrize("flag,error", [("no_overlap", "active_overlap_not_proved"),
    ("wrong", "survivor_oracle_failed"), ("missing_done", "survivor_oracle_failed")])
def test_ambiguous_overlap_wrong_survivor_and_partial_stream_fail(server, tmp_path, monkeypatch, flag, error):
    server[1][flag] = True
    code, row, _ = run_cli(server, tmp_path, monkeypatch)
    assert code == 1 and not row["passed"] and error in row["errors"]
    assert row["qualification"] == "failed_or_ambiguous_screen"


def test_payload_identities_and_distinct_oracles():
    module = implementation()
    first = module.payloads("model", 0, 32)
    second = module.payloads("model", 1, 32)
    assert first["survivor_case"]["expected"] != second["survivor_case"]["expected"]
    assert first["cancel"]["ignore_eos"] and first["cancel"]["max_tokens"] == 8192
    assert len(first["recovery_cases"]) == 6
    assert len({case["expected"] for case in first["recovery_cases"]}) == 6
    assert first["survivor"]["messages"] == module.payloads("model", 0, 32)["survivor"]["messages"]
    assert "Reply with the number only." not in first["survivor"]["messages"][0]["content"]


def test_existing_output_is_not_overwritten(server, tmp_path):
    output = tmp_path / "saved.jsonl"
    output.write_text("saved")
    with pytest.raises(FileExistsError):
        implementation().main(["--base-url", server[0], "--label", "local", "--output", str(output)])
    assert output.read_text() == "saved"
