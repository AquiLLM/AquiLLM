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


@pytest.mark.parametrize(
    ("backend_error", "expected"),
    [
        (
            type(
                "AuthError",
                (RuntimeError,),
                {"code": "Neo.ClientError.Security.Unauthorized"},
            )("secret"),
            "memgraph_authentication_failed",
        ),
        (TimeoutError("private query"), "memgraph_timeout"),
        (
            type(
                "TransactionTimedOut",
                (RuntimeError,),
                {
                    "code": (
                        "Neo.ClientError.Transaction."
                        "TransactionTimedOutClientConfiguration"
                    )
                },
            )("private query"),
            "memgraph_timeout",
        ),
        (
            type(
                "Unavailable",
                (RuntimeError,),
                {"code": "Neo.TransientError.General.DatabaseUnavailable"},
            )("bolt://private"),
            "memgraph_unavailable",
        ),
    ],
)
def test_driver_classifies_closed_read_failures(backend_error, expected) -> None:
    class Broken:
        def session(self, **_kwargs):
            raise backend_error

    driver = Neo4jMemgraphDriver(
        "bolt://memgraph:7687", "reader", "secret", database="memgraph", driver=Broken()
    )
    with pytest.raises(MemgraphDriverError) as captured:
        driver.execute_read("RETURN private", {}, timeout_seconds=1.0, max_records=1)
    assert captured.value.code == expected
    assert str(captured.value) == expected


def test_schema_bootstrap_uses_only_fixed_indexes_in_bounded_implicit_transactions():
    calls = []

    class SchemaSession(_Session):
        def run(self, query):
            calls.append((query.text, query.timeout))
            return _Result()

    class SchemaClient(_Neo4jClient):
        def session(self, *, database):
            assert database == "memgraph"
            session = SchemaSession(self.transaction)
            self.sessions.append(session)
            return session

    client = SchemaClient()
    driver = Neo4jMemgraphDriver(
        "bolt://memgraph:7687", "", "", database="memgraph", driver=client
    )
    driver.ensure_projection_schema(timeout_seconds=0.5)

    assert [query for query, _ in calls] == [
        "CREATE INDEX ON :CollectionGeneration(generation_key)",
        "CREATE INDEX ON :ProjectedRecord(generation_key)",
        "CREATE INDEX ON :ProjectedEntity(entity_key)",
        "CREATE INDEX ON :ProjectedChunk(chunk_key)",
        "CREATE INDEX ON :ProjectedRelation(relation_key)",
        "CREATE INDEX ON :ProjectedRecord(generation_key, opaque_key)",
        "CREATE INDEX ON :ProjectedEntity(generation_key)",
        "CREATE INDEX ON :AutomaticMembership(generation_key)",
        "CREATE INDEX ON :ProjectedDocument(generation_key)",
        "CREATE INDEX ON :ProjectedChunk(generation_key)",
        "CREATE INDEX ON :ProjectedRelationSemantics(generation_key)",
        "CREATE INDEX ON :ProjectedRelation(generation_key)",
        "CREATE INDEX ON :ProjectedEvidence(generation_key)",
        "CREATE INDEX ON :ProjectedEntityMention(generation_key)",
        "CREATE INDEX ON :ArtifactProvenance(generation_key)",
        "CREATE EDGE INDEX ON :ENTITY_MEMBERSHIP",
        "CREATE EDGE INDEX ON :DOCUMENT_CHUNK",
        "CREATE EDGE INDEX ON :PROJECTED_RELATION",
        "CREATE EDGE INDEX ON :RELATION_EVIDENCE",
        "CREATE EDGE INDEX ON :ENTITY_MENTION",
    ]
    assert all(timeout == 0.5 for _, timeout in calls)
    assert client.transaction.calls == []
    assert all(session.callbacks == [] for session in client.sessions)


def test_schema_bootstrap_redacts_backend_failure():
    class Broken:
        def session(self, **kwargs):
            raise RuntimeError("secret schema details")

    driver = Neo4jMemgraphDriver(
        "bolt://memgraph:7687", "", "", database="memgraph", driver=Broken()
    )
    with pytest.raises(MemgraphDriverError, match="^memgraph_write_failed$"):
        driver.ensure_projection_schema(timeout_seconds=0.5)
