"""Real loopback HTTP discovery and six outstanding synthetic outage pairs."""
from contextlib import contextmanager
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Lock, Thread
from time import monotonic
from types import SimpleNamespace

from apps.documents.services.rerank_transport_policy import (
    discover_rerank_protocol, revoke_rerank_protocol,
)


@contextmanager
def outage_transport(monkeypatch):
    state = SimpleNamespace(failing=False, requests=[], active_pairs=0, peak_pairs=0,
                            full=Event(), idle=Event(), release=Event(), lock=Lock(), errors=[])
    state.idle.set()
    model = "p1-outage-scorer"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            count, admitted = 0, False
            try:
                assert self.path == "/score"
                size = int(self.headers["Content-Length"])
                assert 0 < size <= 16384
                payload = json.loads(self.rfile.read(size))
                assert payload["model"] == model
                assert isinstance(payload["text_2"], list)
                count = len(payload["text_2"])
                assert 1 <= count <= 6
                with state.lock:
                    state.requests.append(payload)
                    admitted = True
                    state.idle.clear()
                    state.active_pairs += count
                    state.peak_pairs = max(state.peak_pairs, state.active_pairs)
                    if state.failing and state.active_pairs == 6:
                        state.full.set()
                if state.failing:
                    assert state.release.wait(3), "fixture did not release outstanding HTTP"
                code = 503 if state.failing else 200
                body = json.dumps({"model": model, "data": [
                    {"index": i, "score": 0.5} for i in range(count)]}).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                state.errors.append(type(exc).__name__)
            finally:
                if admitted:
                    with state.lock:
                        state.active_pairs -= count
                        if state.active_pairs == 0:
                            state.idle.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    binding = None
    try:
        monkeypatch.setenv("APP_RERANK_BASE_URL", base)
        monkeypatch.setenv("APP_RERANK_MODEL", model)
        binding = discover_rerank_protocol(base, model, deadline=monotonic() + 3)
        assert binding is not None and binding.payload_kind == "score_batch_text_pairs"
        assert len(state.requests) == 1 and not state.errors
        assert state.idle.wait(1)
        state.requests.clear()
        state.peak_pairs = 0
        state.failing = True
        yield state
    finally:
        state.release.set()
        if binding is not None:
            revoke_rerank_protocol(base, binding)
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()
