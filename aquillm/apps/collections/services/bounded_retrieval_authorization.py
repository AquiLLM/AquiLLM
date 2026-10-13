"""Selected-row authority and optional exact-context V2 seed read allowance."""

from contextlib import contextmanager
from contextvars import ContextVar

_BOUND = ContextVar("bounded_retrieval_reauthorization", default=None)


class BoundedAuthorizationUnavailable(ValueError):
    """The original policy has no supported bounded selected-scope read."""


def resolve_read_budget(context, explicit=None):
    if explicit is not None:
        from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
            FrontierBudgetV2,
        )

        if type(explicit) is not FrontierBudgetV2:
            raise TypeError("bounded reauthorization requires an exact frontier budget")
    captured = _BOUND.get()
    if captured is not None:
        if context is not captured[0] or (
            explicit is not None and explicit is not captured[1]
        ):
            raise ValueError("bounded reauthorization scope was replaced")
        return captured[1]
    return explicit


@contextmanager
def bounded_reauthorization_reads(original_context, read_budget):
    from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
        FrontierBudgetV2,
    )

    from .retrieval_authorization_contracts import RetrievalAuthorizationContext

    if (
        type(original_context) is not RetrievalAuthorizationContext
        or type(read_budget) is not FrontierBudgetV2
    ):
        raise TypeError("bounded reauthorization requires original exact inputs")
    resolve_read_budget(original_context, read_budget)
    token = _BOUND.set((original_context, read_budget))
    try:
        yield
    finally:
        _BOUND.reset(token)


def current_selected_rows(
    *,
    principal,
    policy,
    database_alias,
    selected_collection_ids,
    selected_document_ids,
    read_budget=None,
):
    from .retrieval_authorization_contracts import (
        _current_rows,
        validate_current_rows,
    )

    method = getattr(policy, "current_authorized_selected_document_scope", None)
    if not callable(method):
        if read_budget is not None:
            raise BoundedAuthorizationUnavailable("bounded authorization unavailable")
        return _current_rows(
            principal=principal,
            policy=policy,
            database_alias=database_alias,
            selected_collections=selected_collection_ids,
        )
    return validate_current_rows(
        method(
            principal=principal,
            database_alias=database_alias,
            selected_collection_ids=selected_collection_ids,
            selected_document_ids=selected_document_ids,
            read_budget=read_budget,
        )
    )
