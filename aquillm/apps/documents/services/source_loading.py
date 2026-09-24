"""Graph-independent, metadata-first authorized source body materialization."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass, field
from threading import RLock

from django.db import connections, transaction
from django.db.models.functions import MD5, Length

from apps.collections.services.retrieval_authorization import (
    revalidate_retrieval_authorization_context,
)
from lib.retrieval.turn_budget import TurnBudget

from .source_deadline import check_source_deadline, source_remaining_ms


class SourcePreparationLimited(ValueError):
    """Source work cannot safely proceed within this turn."""


@dataclass
class SourceRuntime:
    budget: TurnBudget
    authorization: object
    cache: dict = field(default_factory=dict, repr=False)
    windows: dict = field(default_factory=dict, repr=False)
    lock: RLock = field(default_factory=RLock, repr=False)
    observation: dict = field(default_factory=dict, repr=False)


_RUNTIME = ContextVar("source_runtime", default=None)


def current_source_runtime():
    return _RUNTIME.get()


@contextmanager
def source_runtime_scope(runtime):
    if not isinstance(runtime, SourceRuntime) or not isinstance(
        runtime.budget, TurnBudget
    ):
        raise TypeError("source runtime requires the caller's shared ledger")
    token = _RUNTIME.set(runtime)
    try:
        yield runtime
    finally:
        _RUNTIME.reset(token)


def source_mode_enabled():
    from apps.chat.services.rag_config import rag_preservation_config

    return rag_preservation_config().evidence_text_mode == "source"


def bounded_source_enabled():
    """Accounting is independent of whether the public evidence text is full."""
    return current_source_runtime() is not None or source_mode_enabled()


@contextmanager
def bounded_source_database(runtime, alias):
    """Finite DB statement timeout, never larger than the remaining turn."""
    milliseconds = source_remaining_ms(runtime.budget.remaining_ms())
    if milliseconds <= 0:
        raise SourcePreparationLimited("source deadline")
    with transaction.atomic(using=alias):
        connection = connections[alias]
        if connection.vendor == "postgresql":
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_setting('statement_timeout')")
                previous = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    [f"{milliseconds}ms"],
                )
        try:
            yield
        finally:
            if connection.vendor == "postgresql" and not connection.needs_rollback:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT set_config('statement_timeout', %s, true)", [previous]
                    )


def source_query_rows(queryset, *, runtime=None, refresh=False):
    """Read at most 45 ranked bodies after length/revision/current-scope checks.

    Database digest + length predicates close both same-length and growing-body
    races. Repeated reads are charged; a cached body can be reused only after
    fresh metadata and authorization checks. No deferred content access escapes.
    """
    runtime = runtime or current_source_runtime()
    if runtime is None:
        if source_mode_enabled():
            raise SourcePreparationLimited("source mode requires shared turn budget")
        return tuple(queryset)
    with runtime.lock, bounded_source_database(runtime, queryset.db):
        scope = revalidate_retrieval_authorization_context(
            context=runtime.authorization
        )
        allowed = frozenset(scope.document_ids)
        if not allowed or not runtime.budget.can_publish():
            return ()
        metadata = tuple(
            queryset.annotate(
                _source_length=Length("content"), _source_revision=MD5("content")
            ).values(
                "pk", "doc_id", "chunk_number", "_source_length", "_source_revision"
            )[:45]
        )
        rows = []
        for meta in metadata:
            check_source_deadline()
            if meta["doc_id"] not in allowed or not runtime.budget.can_publish():
                continue
            identity = (meta["pk"], meta["_source_revision"])
            cache_key = (queryset.db, *identity, meta["doc_id"], meta["chunk_number"])
            cached = runtime.cache.get(cache_key)
            if cached is not None and not refresh:
                rows.append(copy(cached))
                continue
            size = meta["_source_length"]
            if not runtime.budget.admit_source(identity):
                continue
            if size and not runtime.budget.reserve_text(size, kind="materialized"):
                continue
            if not runtime.budget.can_publish():
                break
            loaded = (
                queryset.model.objects.using(queryset.db)
                .annotate(
                    _source_length=Length("content"), _source_revision=MD5("content")
                )
                .filter(
                    pk=meta["pk"],
                    doc_id=meta["doc_id"],
                    chunk_number=meta["chunk_number"],
                    _source_length=size,
                    _source_revision=meta["_source_revision"],
                )
                .defer("embedding", "metadata")
                .first()
            )
            if loaded is None:
                continue
            # Keep a private frozen copy; callers never mutate the cached object.
            if runtime.budget.publish(
                lambda: runtime.cache.__setitem__(cache_key, copy(loaded))
            ):
                rows.append(loaded)
        return tuple(rows)
