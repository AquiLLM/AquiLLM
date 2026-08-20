from __future__ import annotations

import pytest

from apps.knowledge_graph.projection.memgraph_driver import (
    MemgraphDriverError,
    Neo4jMemgraphDriver,
)


class _Record(dict):
    def data(self):
        return dict(self)


class _Result:
    def __iter__(self):
        return iter((_Record(ok=1),))

    def consume(self):
        return type("Summary", (), {"counters": {"nodes_created": 1}})()


class _Transaction:
    def __init__(self):
        self.calls = []

    def run(self, cypher, parameters):
        self.calls.append((cypher, parameters))
        return _Result()


class _Session:
    def __init__(self, transaction):
        self.transaction = transaction
        self.callbacks = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute_read(self, callback):
        self.callbacks.append(callback)
        return callback(self.transaction)

    def execute_write(self, callback):
        self.callbacks.append(callback)
        return callback(self.transaction)


class _Neo4jClient:
    def __init__(self):
        self.transaction = _Transaction()
        self.databases = []
        self.sessions = []

    def session(self, *, database):
        self.databases.append(database)
        session = _Session(self.transaction)
        self.sessions.append(session)
        return session


def test_driver_uses_transaction_function_timeout_not_a_cypher_parameter() -> None:
    client = _Neo4jClient()
    driver = Neo4jMemgraphDriver(
        "bolt://memgraph:7687", "reader", "secret", database="memgraph", driver=client
    )

    rows = driver.execute_read(
        "RETURN $value AS ok", {"value": 1}, timeout_seconds=0.5, max_records=1
    )

    assert rows == ({"ok": 1},)
    assert client.databases == ["memgraph"]
    assert client.transaction.calls == [("RETURN $value AS ok", {"value": 1})]
    assert client.sessions[0].callbacks[0].timeout == 0.5


def test_driver_errors_are_fixed_and_do_not_expose_credentials_or_cypher() -> None:
    class Broken:
        def session(self, **_kwargs):
            raise RuntimeError("secret RETURN private")

    driver = Neo4jMemgraphDriver(
        "bolt://memgraph:7687", "reader", "secret", database="memgraph", driver=Broken()
    )
    with pytest.raises(MemgraphDriverError) as captured:
        driver.execute_read("RETURN private", {}, timeout_seconds=1.0, max_records=1)
    assert str(captured.value) == "memgraph_read_failed"
    assert "secret" not in repr(captured.value)
