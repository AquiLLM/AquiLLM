"""Graph-independent, metadata-first authorized source body materialization."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass, field
from threading import RLock
from uuid import UUID

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
                    "SELECT set_config('statement_timeout', LEAST(%s, COALESCE(NULLIF("
                    "(SELECT setting::bigint FROM pg_settings "
                    "WHERE name='statement_timeout'),"
                    "0), %s))::text || 'ms', true)",
                    [milliseconds, milliseconds],
                )
        completed = False
        try:
            yield
            completed = True
        finally:
            # An escaped error must reach atomic rollback without another query
            # against a potentially aborted transaction masking its exception.
            if (
                completed
                and connection.vendor == "postgresql"
                and not connection.needs_rollback
            ):
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT set_config('statement_timeout', %s, true)", [previous]
                    )


def source_query_rows(queryset, *, runtime=None, refresh=False, require_existing=False):
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
        from .representation_tool_scope import tool_scope

        frozen = tool_scope(
            runtime, using=queryset.db, require_existing=require_existing
        )
        # This eager locked operation owns fresh entry and final scope checks.
        # Keep the lazy predicates; no query or validation permit leaves it.
        queryset = _prepare_source_queryset(queryset, frozen)
        scope = revalidate_retrieval_authorization_context(
            context=runtime.authorization
        )
        allowed = frozenset(scope.document_ids)
        if not allowed or not runtime.budget.can_publish():
            return ()
        rows = _source_body_rows(
            queryset, frozen, runtime.budget, allowed, runtime.cache, refresh=refresh
        )
        frozen.validate()
        return rows


def _prepare_source_queryset(queryset, frozen):
    """Private SQL construction, consumed eagerly by a fresh-boundary caller."""
    from .representation_contracts import SourceScopeUnavailable
    from .representation_store import _active_chunk_predicates

    if queryset.query.combinator:
        raise SourceScopeUnavailable("source_scope_unavailable")
    low, high = queryset.query.low_mark, queryset.query.high_mark
    queryset = queryset.all()
    queryset.query.clear_limits()
    queryset = _active_chunk_predicates(
        queryset.filter(doc_id__in=frozen.requested_document_ids),
        snapshot=frozen.representation,
    )
    queryset.query.set_limits(low, high)
    return queryset


def _source_body_rows(queryset, frozen, budget, allowed, cache, *, refresh=False):
    """Eager body I/O; only closed callers own validation and cache publication."""
    from .representation_contracts import SourceScopeUnavailable
    from .representation_store import _active_chunk_predicates

    metadata = tuple(
        queryset.annotate(
            _source_length=Length("content"), _source_revision=MD5("content")
        ).values(
            "pk",
            "doc_id",
            "chunk_number",
            "representation_generation",
            "_source_length",
            "_source_revision",
        )[:45]
    )
    rows = []
    for meta in metadata:
        check_source_deadline()
        if meta["doc_id"] not in allowed or not budget.can_publish():
            continue
        identity = (meta["pk"], meta["_source_revision"])
        cache_key = (
            queryset.db,
            *identity,
            meta["doc_id"],
            meta["chunk_number"],
            meta["representation_generation"],
        )
        cached = cache.get(cache_key)
        if cached is not None and not refresh:
            rows.append(copy(cached))
            continue
        size = meta["_source_length"]
        if not budget.admit_source(identity):
            continue
        if size and not budget.reserve_text(size, kind="materialized"):
            continue
        if not budget.can_publish():
            break
        if (
            type(meta["doc_id"]) is not UUID
            or meta["doc_id"] not in frozen.requested_document_ids
        ):
            raise SourceScopeUnavailable("source_scope_unavailable")
        loaded = (
            _active_chunk_predicates(
                queryset.model.objects.using(queryset.db).filter(doc_id=meta["doc_id"]),
                snapshot=frozen.representation,
                document_id=meta["doc_id"],
            )
            .annotate(_source_length=Length("content"), _source_revision=MD5("content"))
            .filter(
                pk=meta["pk"],
                doc_id=meta["doc_id"],
                chunk_number=meta["chunk_number"],
                representation_generation=meta["representation_generation"],
                _source_length=size,
                _source_revision=meta["_source_revision"],
            )
            .defer("embedding", "metadata")
            .first()
        )
        if loaded is None:
            continue
        # A staged caller supplies a private overlay; no positive state escapes.
        if budget.publish(lambda: cache.__setitem__(cache_key, copy(loaded))):
            rows.append(loaded)
    return tuple(rows)
