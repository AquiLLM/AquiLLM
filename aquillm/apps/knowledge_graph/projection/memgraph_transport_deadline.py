"""Retain one operation across session acquisition, callbacks and schema dispatch."""
from functools import wraps

from .operation_context import current_operation


class TransportDeadline:
    def __init__(self, timeout):
        self.timeout = timeout
        self.operation = current_operation()

    def remaining(self):
        if self.operation is None:
            return self.timeout
        return min(self.timeout, self.operation.remaining_ms() / 1000)

    def check(self):
        if self.operation is not None:
            self.operation.remaining_ms()


def transaction_function(self, callback, *, timeout: float):
    try:
        from neo4j import unit_of_work
    except ModuleNotFoundError:
        if self._client is None:
            raise
        callback.timeout = timeout
        return callback
    return unit_of_work(timeout=timeout)(callback)


def execute_managed(driver, callback, *, timeout, write):
    deadline = TransportDeadline(timeout)

    @wraps(callback)
    def bounded(transaction):
        # Neo4j may re-enter this callback; it must retain the original operation.
        deadline.check()
        result = callback(transaction)
        deadline.check()
        return result

    deadline.check()
    with driver._connection().session(database=driver._database) as session:
        execute = session.execute_write if write else session.execute_read
        result = execute(driver._transaction_function(bounded, timeout=deadline.remaining()))
        deadline.check()
        return result


def ensure_schema(driver, statements, *, timeout):
    from neo4j import Query

    deadline = TransportDeadline(timeout)
    deadline.check()
    with driver._connection().session(database=driver._database) as session:
        for statement in statements:
            session.run(Query(statement, timeout=deadline.remaining())).consume()
            deadline.check()
