"""External SDK transport only; production driver and original deadline stay real."""
from types import SimpleNamespace

from apps.knowledge_graph.projection import database_deadline
from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver
from apps.knowledge_graph.projection.memgraph_edges import write_parameterized_batches
from apps.knowledge_graph.projection.memgraph_repository import MemgraphProjectionRepository


class Transport:
    def __init__(self, step):
        self.now, self.step = 100.0, step
        self.calls = []
        self.timeout = None
        self.session_delay = 0.0

    def session(self, *, database):
        assert database == "memgraph"
        self.now += self.session_delay
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute_read(self, callback):
        self.timeout = callback.timeout
        return callback(self)

    execute_write = execute_read

    def run(self, query, parameters=None):
        if parameters is None:
            text, timeout = query.text, query.timeout
        else:
            text, timeout = query, self.timeout
        self.calls.append((text, parameters, timeout, self.now))
        self.now += self.step
        return self

    def __iter__(self):
        return iter((SimpleNamespace(data=lambda: {"ok": 1}),))

    def consume(self):
        return SimpleNamespace(counters={"nodes_created": 1})


def setup_transport(monkeypatch, *, step):
    transport = Transport(step)
    monkeypatch.setattr(database_deadline, "monotonic", lambda: transport.now)
    driver = Neo4jMemgraphDriver(
        "bolt://synthetic:7687", "", "", database="memgraph", driver=transport,
        max_transaction_retry_time=0.0,
    )
    return transport, driver


def invoke(kind, driver, timeout):
    if kind == "read":
        for _ in range(2):
            assert driver.execute_read(
                "RETURN $value AS ok", {"value": 1},
                timeout_seconds=timeout, max_records=1,
            ) == ({"ok": 1},)
    elif kind == "write":
        write_parameterized_batches(
            driver, "CREATE (:Synthetic {key:row.key})",
            ({"key": "one"}, {"key": "two"}),
            generation_key="a" * 64, batch_size=1, timeout_seconds=timeout,
        )
    else:
        assert kind == "schema"
        MemgraphProjectionRepository(driver).ensure_schema(timeout_seconds=timeout)
