"""Owned loopback children and bounded synthetic lifecycle controls."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from uuid import uuid4

from lib.knowledge_graph.query_extractor import service
from lib.knowledge_graph.query_extractor.config import load_query_extractor_settings
from lib.knowledge_graph.query_extractor.contracts import (
    QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM,
)


def eventually(predicate, seconds=10):
    end = monotonic() + seconds
    while monotonic() < end:
        result = predicate()
        if result:
            return result
        sleep(0.01)
    raise AssertionError("bounded cold-service observation expired")


class ColdService:
    def __init__(self, root, kind):
        assert os.environ.get("RAG_REMEDIATION_TEST") == "1"
        self.root, self.kind = root, kind
        root.mkdir()
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        self.process = None
        self.env = dict(os.environ)
        self.env.update(KG_GLINER2_LOCAL_FILES_ONLY="true")
        if kind == "extractor":
            self.configure_extractor()
        else:
            from apps.knowledge_graph.retrieval.topology.gateway_config import (
                load_topology_gateway_settings,
            )
            from apps.knowledge_graph.retrieval.topology.gateway_contracts import (
                MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES,
            )
            from apps.knowledge_graph.tests.frontier_fixture_support import FrontierFixture
            from apps.knowledge_graph.tests.frontier_index_fixture_driver import (
                FrontierIndexFixtureDriver,
            )

            driver = FrontierIndexFixtureDriver(FrontierFixture().bundles)
            records = json.dumps(dict(
                markers=driver.markers, index=driver.index,
                raw=[[generation, key, rows] for (generation, key), rows in driver.raw.items()],
            ))
            assert len(records.encode()) <= 1048576
            (root / "graph.json").write_text(records, encoding="utf-8")
            self.env.update(
                KG_TOPOLOGY_GATEWAY_BEARER_TOKEN="synthetic-token",
                KG_MEMGRAPH_URI="bolt://127.0.0.1:7687",
                KG_MEMGRAPH_DATABASE="memgraph",
                KG_MEMGRAPH_QUERY_USERNAME="synthetic",
                KG_MEMGRAPH_QUERY_PASSWORD="synthetic",
                KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES=str(MAX_REQUEST_BYTES),
                KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES=str(MAX_RESPONSE_BYTES),
                KG_TOPOLOGY_GATEWAY_TIMEOUT_MS="300",
            )
            load_topology_gateway_settings(self.env)

    def configure_extractor(self):
        document = dict(version="1.0.0", entity_types=[dict(
            name="model", description="Synthetic model", aliases=[],
            default_retrieval_weight=1.0, default_suppression_policy="none",
            default_suppression_threshold=0.0,
        )], relations=[dict(
            name="uses", description="Synthetic relation", direction="directed",
            allowed_head_types=["model"], allowed_tail_types=["model"],
        )])
        path = self.root / "ontology.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        self.ontology = service._load_activated_ontology(path)
        self.env.update(
            KG_QUERY_EXTRACTOR_URL=self.url + "/v1/extract",
            KG_QUERY_EXTRACTOR_BEARER_TOKEN="synthetic-token",
            KG_QUERY_EXTRACTOR_MODEL="fastino/gliner2-base-v1",
            KG_QUERY_EXTRACTOR_MODEL_REVISION="8437ba583a733d87f56ae902f3b197934eedd58e",
            KG_QUERY_EXTRACTOR_BUILD_HASH="b" * 64,
            KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_VERSION="query-entities-v1",
            KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_CHECKSUM=(
                QUERY_EXTRACTION_RESPONSE_SCHEMA_CHECKSUM
            ),
            KG_QUERY_EXTRACTOR_ONTOLOGY_PATH=str(path),
            KG_QUERY_EXTRACTOR_ONTOLOGY_CHECKSUM=self.ontology.checksum,
            KG_QUERY_EXTRACTOR_TIMEOUT_MS="300",
            KG_QUERY_MAX_BYTES="64", KG_QUERY_MAX_CODEPOINTS="32",
            KG_QUERY_MAX_SPANS="4",
        )
        self.settings = load_query_extractor_settings(self.env)

    def marker(self, name, enabled=True):
        path = self.root / name
        if enabled:
            path.touch()
        else:
            path.unlink(missing_ok=True)

    def events(self):
        path = self.root / "events.jsonl"
        if not path.exists():
            return []
        # Ignore an incomplete final append while a native worker owns the lock.
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        return [json.loads(line) for line in lines if line.endswith("\n")]

    def count(self, name):
        return sum(row["event"] == name for row in self.events())

    def status(self, path):
        try:
            with urlopen(self.url + path, timeout=1) as response:
                return response.status
        except HTTPError as exc:
            try:
                return exc.code
            finally:
                exc.close()
        except (URLError, TimeoutError, ConnectionError):
            return None

    def start(self):
        self.log = (self.root / "child.log").open("wb")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "tests.integration.cold_service_child",
             str(self.root), self.kind, str(self.sock.fileno())],
            env=self.env, pass_fds=(self.sock.fileno(),),
            stdout=self.log, stderr=subprocess.STDOUT,
        )
        self.sock.close()
        eventually(lambda: self.count("imported") or self.process.poll() is not None)
        assert self.count("imported"), (self.root / "child.log").read_text()
        assert self.events()[0] == dict(
            event="imported", pid=self.process.pid, ml_loaded=False,
        )
        return self

    def stop(self):
        for path in self.root.glob("*_block"):
            path.unlink(missing_ok=True)
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
                raise AssertionError("child required forced termination")
            finally:
                self.log.close()
            artifact = Path("/artifacts/cold-services") / (self.kind + "-" + uuid4().hex)
            artifact.mkdir(parents=True)
            for name in ("events.jsonl", "child.log"):
                path = self.root / name
                if path.exists():
                    (artifact / name).write_bytes(path.read_bytes())
            print(f"cold-service process evidence: {artifact}")
            self.process = None
        self.sock.close()


@contextmanager
def cold_service(tmp_path, kind, *, markers=()):
    child = ColdService(tmp_path, kind)
    try:
        for name in markers:
            child.marker(name)
        yield child.start()
    finally:
        child.stop()
