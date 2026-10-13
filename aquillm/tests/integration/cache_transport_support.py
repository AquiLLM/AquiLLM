"""STAGED, UNEXECUTED: owned Redis plus finite loopback synthetic scorer HTTP."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from time import monotonic
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from django.core.cache import caches
from django.core.cache.backends.redis import RedisCache


@pytest.fixture
def cache_transport(settings, monkeypatch):
    from apps.documents.services import rag_cache
    from apps.documents.services.rerank_transport_policy import (
        discover_rerank_protocol, revoke_rerank_protocol,
    )
    configured = urlsplit(settings.CACHES["default"]["LOCATION"])
    assert (configured.scheme, configured.hostname, configured.port, configured.path) == (
        "redis", "redis", 6379, "/1")
    assert type(caches["default"]) is RedisCache
    settings.RAG_CACHE_ENABLED = True
    logical, physical, scoring = [], [], []
    lock = Lock()
    for name in ("get", "set"):
        original = getattr(RedisCache, name)
        def observe(self, *args, _original=original, _name=name, **kwargs):
            result = _original(self, *args, **kwargs)
            with lock:
                physical.append((_name, args[0], result))
            return result
        monkeypatch.setattr(RedisCache, name, observe)
    for name in ("cache_get", "cache_set"):
        original = getattr(rag_cache, name)
        def observe(*args, _original=original, _name=name, **kwargs):
            result = _original(*args, **kwargs)
            with lock:
                logical.append((_name, args[0], args[1:] if _name == "cache_set" else result))
            return result
        monkeypatch.setattr(rag_cache, name, observe)
    model = "p1-cache-" + uuid4().hex
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            size = int(self.headers["Content-Length"])
            assert 0 < size <= 1024 * 1024
            payload = json.loads(self.rfile.read(size))
            assert self.path == "/score" and payload["model"] == model
            texts = payload["text_2"]
            assert type(texts) is list and 1 <= len(texts) <= 6
            with lock:
                scoring.append(payload)
            body = json.dumps({"model": model, "data": [
                {"index": i, "score": 0.9 - i / 100} for i in range(len(texts))]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    for key, value in {"APP_RERANK_PROVIDER": "local", "APP_RERANK_BASE_URL": base,
                       "APP_RERANK_MODEL": model, "APP_RERANK_MODEL_REVISION": "synthetic-cache-v1"}.items():
        monkeypatch.setenv(key, value)
    binding = None
    try:
        binding = discover_rerank_protocol(base, model, deadline=monotonic() + 5)
        assert binding is not None and scoring
        assert any(row[0] == "set" for row in physical)
        yield SimpleNamespace(logical=logical, physical=physical, scoring=scoring, model=model)
    finally:
        if binding is not None:
            revoke_rerank_protocol(base, binding)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()
