from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

type ProjectionScalar = str | int | float | bool | None

_COUNTERS = (
    "constraints_added",
    "constraints_removed",
    "indexes_added",
    "indexes_removed",
    "labels_added",
    "labels_removed",
    "nodes_created",
    "nodes_deleted",
    "properties_set",
    "relationships_created",
    "relationships_deleted",
)


class MemgraphDriverError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class MemgraphWriteSummaryV1:
    counters: Mapping[str, int]

    def __post_init__(self) -> None:
        if type(self.counters) is not dict or any(
            type(key) is not str or type(value) is not int or value < 0
            for key, value in self.counters.items()
        ):
            raise TypeError("counters must be a nonnegative exact mapping")


def _timeout(value: object) -> float:
    if type(value) is not float or not isfinite(value) or value <= 0.0:
        raise ValueError("timeout_seconds must be a finite positive float")
    return value


def _parameters(value: object) -> dict[str, ProjectionScalar]:
    if not isinstance(value, Mapping):
        raise TypeError("parameters must be a mapping")
    result: dict[str, ProjectionScalar] = {}
    for key, item in value.items():
        if (
            type(key) is not str
            or not key
            or type(item)
            not in {
                str,
                int,
                float,
                bool,
                type(None),
            }
        ):
            raise TypeError("parameters contain an unsupported scalar")
        if type(item) is float and not isfinite(item):
            raise ValueError("parameters contain a nonfinite scalar")
        result[key] = item
    return result


def _cypher(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("cypher must be a nonempty exact string")
    return value


def _read_failure_code(error: Exception) -> str:
    code = getattr(error, "code", None)
    if type(code) is str:
        if code.startswith("Neo.ClientError.Security."):
            return "memgraph_authentication_failed"
        if code in {
            "Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration",
            "Neo.ClientError.Transaction.TransactionTimedOut",
            "Neo.TransientError.Transaction.TransactionTimedOut",
            "Neo.TransientError.Transaction.TransactionTimedOutClientConfiguration",
        }:
            return "memgraph_timeout"
        if code in {
            "Neo.TransientError.General.DatabaseUnavailable",
            "Neo.TransientError.General.ServiceUnavailable",
        }:
            return "memgraph_unavailable"
    if isinstance(error, TimeoutError):
        return "memgraph_timeout"
    if isinstance(error, ConnectionError):
        return "memgraph_unavailable"
    if error.__class__.__module__.startswith("neo4j.") and error.__class__.__name__ in {
        "ServiceUnavailable",
        "SessionExpired",
    }:
        return "memgraph_unavailable"
    return "memgraph_read_failed"


class Neo4jMemgraphDriver:
    # The shipping adapter may opt into graph-seeded reads only for this driver.
    supports_bounded_topology = True

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        *,
        database: str,
        driver: Any | None = None,
    ) -> None:
        for name, value in (
            ("uri", uri),
            ("username", username),
            ("password", password),
            ("database", database),
        ):
            if type(value) is not str:
                raise TypeError(f"{name} must be an exact string")
        if not uri or not database or database != database.strip():
            raise ValueError("uri and database must be nonempty canonical strings")
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database
        self._client = driver

    def _connection(self):
        if self._client is None:
            from neo4j import GraphDatabase

            auth = (
                None
                if not self._username and not self._password
                else (self._username, self._password)
            )
            self._client = GraphDatabase.driver(self._uri, auth=auth)
        return self._client

    def _transaction_function(self, callback, *, timeout: float):
        try:
            from neo4j import unit_of_work
        except ModuleNotFoundError:
            if self._client is None:
                raise
            callback.timeout = timeout
            return callback
        return unit_of_work(timeout=timeout)(callback)

    def execute_read(
        self,
        cypher: str,
        parameters: Mapping[str, ProjectionScalar],
        *,
        timeout_seconds: float,
        max_records: int,
    ) -> tuple[Mapping[str, ProjectionScalar], ...]:
        query = _cypher(cypher)
        values = _parameters(parameters)
        timeout = _timeout(timeout_seconds)
        if type(max_records) is not int or not 1 <= max_records <= 5_000:
            raise ValueError("max_records must be an integer in 1..5000")

        def read(transaction):
            result = transaction.run(query, values)
            rows = []
            for record in result:
                if len(rows) == max_records:
                    raise MemgraphDriverError("memgraph_result_limit")
                data = record.data()
                if type(data) is not dict:
                    raise MemgraphDriverError("memgraph_result_invalid")
                rows.append(data)
            return tuple(rows)

        try:
            with self._connection().session(database=self._database) as session:
                return session.execute_read(
                    self._transaction_function(read, timeout=timeout)
                )
        except MemgraphDriverError:
            raise
        except Exception as error:
            raise MemgraphDriverError(_read_failure_code(error)) from None

    def execute_write(
        self,
        cypher: str,
        parameters: Mapping[str, ProjectionScalar],
        *,
        timeout_seconds: float,
    ) -> MemgraphWriteSummaryV1:
        query = _cypher(cypher)
        values = _parameters(parameters)
        timeout = _timeout(timeout_seconds)

        def write(transaction):
            result = transaction.run(query, values)
            summary = result.consume()
            counters = getattr(summary, "counters", {})
            if type(counters) is dict:
                encoded = {
                    key: value
                    for key, value in counters.items()
                    if type(key) is str and type(value) is int and value >= 0
                }
            else:
                encoded = {
                    name: value
                    for name in _COUNTERS
                    if type(value := getattr(counters, name, None)) is int
                    and value >= 0
                }
            return MemgraphWriteSummaryV1(encoded)

        try:
            with self._connection().session(database=self._database) as session:
                return session.execute_write(
                    self._transaction_function(write, timeout=timeout)
                )
        except MemgraphDriverError:
            raise
        except Exception:
            raise MemgraphDriverError("memgraph_write_failed") from None
