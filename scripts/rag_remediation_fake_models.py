"""Synthetic deterministic HTTP models. Unknown APIs fail closed."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            self.reply(200, {"object": "list", "data": [{"id": "synthetic-rag-model", "object": "model"}]})
        else:
            self.reply(404, {"error": "unsupported synthetic API"})

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        if self.path == "/v1/embeddings":
            inputs = data.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            rows = [{"index": i, "object": "embedding", "embedding": [float(x) / 255 for x in hashlib.sha256(str(value).encode()).digest()] * 128} for i, value in enumerate(inputs)]
            self.reply(200, {"object": "list", "model": "synthetic-rag-model", "data": rows, "usage": {"prompt_tokens": 0, "total_tokens": 0}})
        elif self.path in ("/v1/rerank", "/rerank", "/score", "/v1/score"):
            query = set(str(data.get("query", "")).lower().split())
            documents = data.get("documents", data.get("text_2", []))
            if isinstance(documents, str):
                documents = [documents]
            rows = [{"index": i, "relevance_score": len(query & set(str(doc).lower().split())) / max(len(query), 1)} for i, doc in enumerate(documents)]
            self.reply(200, {"results": rows, "data": [{"index": r["index"], "score": r["relevance_score"]} for r in rows]})
        elif self.path == "/v1/chat/completions" and not data.get("stream"):
            self.reply(200, {"id": "synthetic-completion", "object": "chat.completion", "model": "synthetic-rag-model", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Synthetic test response."}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
        else:
            self.reply(501, {"error": "unsupported synthetic API"})


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
