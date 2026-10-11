"""Fresh-process ASGI host; substitute only external native backend work."""

import asyncio
import json
import os
from pathlib import Path
import socket
import sys
from threading import Lock
from time import monotonic, sleep

import uvicorn


ROOT = Path(sys.argv[1])
KIND = sys.argv[2]
LOCK = Lock()


def event(name, **values):
    with LOCK:
        path = ROOT / "events.jsonl"
        assert not path.exists() or path.stat().st_size < 32768
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(event=name, **values)) + "\n")


def native_work(name):
    event(name + "_entered")
    deadline = monotonic() + 15
    while (ROOT / (name + "_block")).exists():
        if monotonic() >= deadline:
            raise RuntimeError("synthetic backend barrier expired")
        sleep(0.005)
    if (ROOT / (name + "_fail")).exists():
        event(name + "_failed")
        raise RuntimeError("synthetic backend failure")
    event(name + "_finished")


def extractor_app():
    from lib.knowledge_graph.extractors import gliner2_local
    from lib.knowledge_graph.query_extractor import service
    from lib.knowledge_graph.types import EntityCandidate, ExtractionBatchResult

    class Backend:
        def __init__(self, *, settings):
            event("backend_constructed")

        def extract_entities_batch(self, texts, *, ontology):
            warmup = texts == ("Warmup.",)
            native_work("warmup" if warmup else "inference")
            entities = () if warmup else (EntityCandidate("model", "😀", 1, 2, 0.75),)
            return (ExtractionBatchResult(entities, (), ()),)

    gliner2_local.GLiNER2LocalBackend = Backend
    return service.app


def gateway_app():
    from apps.knowledge_graph.retrieval.topology import gateway_service
    from apps.knowledge_graph.projection.frontier_cypher_v2 import (
        INDEX_LOOKUP, MANIFEST_LOOKUP, RAW_LOOKUP,
    )

    class Driver:
        def __init__(self, *_args, **_kwargs):
            event("backend_constructed")
            path = ROOT / "graph.json"
            assert path.stat().st_size <= 1048576
            raw = path.read_bytes()
            self.records = json.loads(raw)
            self.raw = {(g, k): rows for g, k, rows in self.records["raw"]}

        def execute_read(self, query, parameters, *, timeout_seconds, max_records):
            readiness = query == "RETURN 1 AS ready"
            native_work("readiness" if readiness else "frontier")
            if readiness:
                return ({"ready": 1},)
            assert timeout_seconds > 0 and max_records > 0
            if query == MANIFEST_LOOKUP:
                return ({"record": self.records["markers"][parameters["generation_key"]]},)
            if query == INDEX_LOOKUP:
                row = self.records["index"].get(parameters["index_key"])
                return ({"record": row},) if row else ()
            if query == RAW_LOOKUP:
                return tuple(row for key in parameters["keys_csv"].split(",")
                             for row in self.raw.get((parameters["generation_key"], key), ())[:10])
            raise AssertionError("unexpected non-equality frontier query")

    gateway_service.Neo4jMemgraphDriver = Driver
    return gateway_service.app


async def main():
    app = extractor_app() if KIND == "extractor" else gateway_app()
    event("imported", pid=os.getpid(), ml_loaded=any(
        name == "gliner2" or name.startswith("gliner2.") for name in sys.modules
    ))

    async def observed(scope, receive, send):
        async def forwarded(message):
            if message["type"].startswith("lifespan."):
                event(message["type"])
            elif message["type"] == "http.response.body":
                event("http_body", path=scope["path"], size=len(message.get("body", b"")))
            await send(message)

        await app(scope, receive, forwarded)

    server = uvicorn.Server(uvicorn.Config(
        observed, lifespan="on" if KIND == "extractor" else "off",
        access_log=False, log_level="warning", timeout_graceful_shutdown=2,
    ))
    sock = socket.socket(fileno=int(sys.argv[3]))
    try:
        await server.serve(sockets=[sock])
    finally:
        sock.close()
        event("server_stopped")


if __name__ == "__main__":
    asyncio.run(main())
