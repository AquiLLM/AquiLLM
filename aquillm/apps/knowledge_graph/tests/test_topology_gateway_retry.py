import pytest

from apps.knowledge_graph.projection.memgraph_driver import MemgraphDriverError
from apps.knowledge_graph.retrieval.topology import gateway_service as service
from apps.knowledge_graph.tests.test_topology_gateway_service import _settings


def test_gateway_factory_does_not_retry_failed_managed_reads(monkeypatch):
    # The fake clock models the driver's default retry window without sleeping.
    # Keep the real gateway factory and projection read/exception boundary.
    from neo4j import GraphDatabase

    attempts = []
    elapsed = [0.0]

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def run(self, *_args):
            attempts.append(1)
            elapsed[0] += 0.1
            raise ConnectionError("private backend message")

        def execute_read(self, callback):
            while True:
                try:
                    return callback(self)
                except ConnectionError:
                    if elapsed[0] >= self.retry_budget:
                        raise

    def connect(_uri, **options):
        session = Session()
        session.retry_budget = options.get("max_transaction_retry_time", 30.0)
        return type("Connection", (), {"session": lambda self, **kw: session})()

    monkeypatch.setattr(GraphDatabase, "driver", connect)
    monkeypatch.setattr(service, "_runtime", None)
    runtime = service._get_runtime(_settings())
    with pytest.raises(MemgraphDriverError, match="memgraph_unavailable"):
        runtime.driver.execute_read("RETURN 1", {}, timeout_seconds=0.1, max_records=1)
    assert len(attempts) == 1
    assert elapsed[0] == 0.1
