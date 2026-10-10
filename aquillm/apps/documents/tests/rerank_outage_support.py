"""rerank outage support: synthetic regression coverage."""

import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from types import SimpleNamespace
from uuid import UUID


@contextmanager
def server(mode="503", release=None):
    requests = []
    started = Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            started.set()
            if release:
                release.wait(2)
            if mode == "reset":
                self.connection.close()
                return
            if mode.isdigit():
                self.send_response(int(mode))
                body = b"{}"
            else:
                self.send_response(200)
                count = (
                    len(payload["text_2"]) if isinstance(payload["text_2"], list) else 1
                )
                body = (
                    b"bad-json"
                    if mode == "malformed"
                    else json.dumps(
                        {
                            "model": "wrong" if mode == "wrong_model" else "synthetic",
                            "data": [
                                {"index": i, "score": float(i)} for i in range(count)
                            ],
                        }
                    ).encode()
                )
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/score", requests, started
    finally:
        if release:
            release.set()
        httpd.shutdown()
        httpd.server_close()
        thread.join(2)


def chunks(count=24):
    return [
        SimpleNamespace(
            pk=i,
            doc_id=UUID(int=1),
            chunk_number=i - 1,
            content=f"synthetic source {i}",
        )
        for i in range(1, count + 1)
    ]


def configure(monkeypatch):
    monkeypatch.setenv("APP_RERANK_MODEL", "synthetic")
    monkeypatch.setenv("APP_RERANK_PROVIDER", "local")
    monkeypatch.setenv("RAG_RERANK_TEXT_MODE", "legacy")
    from apps.documents.services import rag_cache

    monkeypatch.setattr(rag_cache, "get_cached_rerank_capability", lambda *a, **k: None)
    monkeypatch.setattr(rag_cache, "get_cached_rerank_result", lambda *a, **k: None)
