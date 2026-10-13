"""One finite allowance; stricter enclosing database and turn limits win."""
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic

from django.db import connections, transaction

_DEADLINE = ContextVar("vocabulary_read_deadline", default=None)
_CLEANUP = ContextVar("vocabulary_savepoint_cleanup", default=None)


def check_read_deadline():
    deadline = _DEADLINE.get()
    if deadline is not None and monotonic() >= deadline:
        raise ValueError("vocabulary read deadline")


@contextmanager
def vocabulary_database(using):
    from apps.documents.services.source_loading import current_source_runtime

    check_read_deadline()
    runtime = current_source_runtime()
    deadline = min(_DEADLINE.get() or float("inf"), monotonic() + 3)
    if runtime:
        deadline = min(deadline, monotonic() + runtime.budget.remaining_ms() / 1000)
    with _bounded_vocabulary_database(using, deadline):
        yield


@contextmanager
def _vocabulary_atomic(using, connection):
    """Admit only this atomic scope's exact savepoint cleanup during its exit."""
    token = None
    try:
        with transaction.atomic(using=using):
            sid = connection.savepoint_ids[-1] if connection.savepoint_ids else None
            cleanup = (connection.ops.savepoint_rollback_sql(sid),
                       connection.ops.savepoint_commit_sql(sid)) if sid else ()
            try:
                yield
            finally:
                # Outer vocabulary guards must not SET against an aborted
                # transaction or reject the rollback after its deadline expires.
                token = _CLEANUP.set((connection, cleanup))
    finally:
        if token is not None:
            _CLEANUP.reset(token)


@contextmanager
def _bounded_vocabulary_database(using, deadline):
    """Private mechanics; only validated acquisition/publication entries call this."""
    check_read_deadline()
    deadline = min(deadline, _DEADLINE.get() or float("inf"), monotonic() + 3)
    connection = connections[using]
    with _vocabulary_atomic(using, connection):
        with connection.cursor() as cursor:
            cursor.execute("SELECT name, setting::bigint FROM pg_settings "
                           "WHERE name IN ('lock_timeout', 'statement_timeout')")
            operator = dict(cursor.fetchall())
        # A shorter operator statement limit also bounds total operation work.
        if operator["statement_timeout"]:
            deadline = min(deadline, monotonic() + operator["statement_timeout"] / 1000)
        token = _DEADLINE.set(deadline)

        def bounded_execute(execute, sql, params, many, context):
            cleanup = _CLEANUP.get()
            if (cleanup is not None and cleanup[0] is connection and not many
                and params is None and sql in cleanup[1]):
                return execute(sql, params, many, context)
            check_read_deadline()
            remaining = max(1, int((deadline - monotonic()) * 1000))
            limits = [min(remaining, operator[name]) if operator[name] else remaining
                      for name in ("lock_timeout", "statement_timeout")]
            with connection.connection.cursor() as control:
                control.execute(
                    "SELECT set_config('lock_timeout', LEAST(%s, COALESCE(NULLIF("
                    "(SELECT setting::bigint FROM pg_settings WHERE name='lock_timeout'), 0), %s))::text, true), "
                    "set_config('statement_timeout', LEAST(%s, COALESCE(NULLIF("
                    "(SELECT setting::bigint FROM pg_settings WHERE name='statement_timeout'), 0), %s))::text, true)",
                    [limits[0], limits[0], limits[1], limits[1]])
            result = execute(sql, params, many, context)
            check_read_deadline()
            return result

        completed = False
        try:
            with connection.execute_wrapper(bounded_execute):
                yield
                check_read_deadline()
                completed = True
        finally:
            _DEADLINE.reset(token)
            if completed and not connection.needs_rollback:
                with connection.connection.cursor() as control:
                    control.execute("SELECT set_config('lock_timeout', %s, true), "
                                    "set_config('statement_timeout', %s, true)",
                                    [f"{operator['lock_timeout']}ms",
                                     f"{operator['statement_timeout']}ms"])
