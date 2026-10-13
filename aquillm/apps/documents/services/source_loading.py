"""Graph-independent, metadata-first authorized source body materialization."""

from collections import ChainMap
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock

from django.db import connections, transaction
from django.db.models.functions import MD5, Length

from apps.collections.services.retrieval_authorization import (
    revalidate_retrieval_authorization_context,
)
from lib.retrieval.turn_budget import TurnBudget

from .source_deadline import source_remaining_ms


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
    return _public_source_rows(queryset, runtime, refresh, require_existing)


def _public_source_rows(queryset, runtime, refresh, require_existing):
    from lib.retrieval.operation import _OPERATION

    from .representation_contracts import SourceScopeUnavailable
    from .representation_tool_scope import tool_scope
    from .source_deadline import _DEADLINE, check_source_deadline

    budget, cache, lock, authority = (
        runtime.budget,
        runtime.cache,
        runtime.lock,
        runtime.authorization,
    )
    context, operation, deadline = (
        current_source_runtime(),
        _OPERATION.get(),
        _DEADLINE.get(),
    )
    using = queryset.db
    original = getattr(runtime, "_frozen_retrieval_scope", None)
    frozen = None
    frozen_header = None

    def scope_header(scope):
        return (
            scope.runtime,
            scope.authorization,
            scope.representation,
            scope.dense,
            scope.space,
            scope.proposal_binding,
            scope.using,
            scope.requested_document_ids,
            scope.query_digest,
            scope.query_compatibility,
            scope.vocabulary_digest,
            scope.graph_enabled,
        )

    original_header = scope_header(original) if original is not None else None

    def same_scope(scope, captured):
        current = scope_header(scope)
        return (
            all(a is b for a, b in zip(current[:6], captured[:6], strict=True))
            and current[6:] == captured[6:]
        )

    def check():
        check_source_deadline()
        if operation is not None and not operation.active():
            raise SourceScopeUnavailable("source_scope_unavailable")
        if (
            not budget.can_publish()
            or runtime.budget is not budget
            or runtime.cache is not cache
            or runtime.lock is not lock
            or runtime.authorization is not authority
            or current_source_runtime() is not context
            or _OPERATION.get() is not operation
            or _DEADLINE.get() is not deadline
            or queryset.db != using
            or getattr(runtime, "_frozen_retrieval_scope", None) is not original
            or (original is not None and not same_scope(original, original_header))
            or (frozen is not None and not same_scope(frozen, frozen_header))
        ):
            raise SourceScopeUnavailable("source_scope_unavailable")

    if budget.remaining_ms() <= 0:
        raise SourcePreparationLimited("source deadline")
    check()
    pending = {}
    with lock:
        check()
        with bounded_source_database(runtime, using):
            check()
            frozen = tool_scope(runtime, using=using, require_existing=require_existing)
            if original is None:
                # Only the authorized initial scope capture may create a holder.
                original = getattr(runtime, "_frozen_retrieval_scope", None)
                original_header = scope_header(original)
            frozen_header = scope_header(frozen)
            check()
            if frozen.runtime is not runtime or frozen.authorization is not authority:
                raise SourceScopeUnavailable("source_scope_unavailable")
            prepared = _prepare_source_queryset(queryset, frozen)
            allowed = frozenset(
                revalidate_retrieval_authorization_context(
                    context=authority
                ).document_ids
            )
            rows = (
                _source_body_rows(
                    prepared,
                    frozen,
                    budget,
                    allowed,
                    ChainMap(pending, cache),
                    refresh=refresh,
                )
                if allowed
                else ()
            )
            frozen.validate()
            check()
        # SQL cleanup and every original lifetime gate precede real cache mutation.
        check()

        def publish():
            check()
            cache.update(pending)

        if not budget.publish(publish):
            raise SourceScopeUnavailable("source_scope_unavailable")
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

    if queryset.db != frozen.using:
        raise SourceScopeUnavailable("source_scope_unavailable")
    queryset = queryset.using(frozen.using)
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
    from .source_body_batch import _materialize_body_batch

    return _materialize_body_batch(
        queryset, frozen, budget, allowed, cache, metadata, refresh=refresh
    )
