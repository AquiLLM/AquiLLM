"""Narrow graph work inside the original caller's acquisition allowance."""

from lib.retrieval.operation import _OPERATION, capture_operation_bound


def captured_graph_bound():
    from .chunk_search_scope import current_scope

    scope = current_scope()
    runtime = scope.runtime if scope is not None else None
    if runtime is None:
        operation = _OPERATION.get()
        if operation is not None and operation.deadline is not None:
            raise ValueError("bounded graph caller lost original source runtime")
        return None
    if runtime.authorization is not scope.authorization:
        raise ValueError("graph source authorization identity changed")
    return capture_operation_bound(runtime.budget)


def graph_rerank_reserve_ms():
    # This is the existing acquisition transport cap, not a new turn allowance
    # or a guarantee that every fresh source/materialization fence fits.
    from .chunk_rerank_config import rerank_timeout_seconds

    return rerank_timeout_seconds() * 1000
