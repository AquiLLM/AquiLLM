"""One finite allowance; stricter enclosing database and turn limits win."""
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic

from django.db import connections, transaction

_DEADLINE = ContextVar("vocabulary_read_deadline", default=None)


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
def _bounded_vocabulary_database(using, deadline):
    """Private mechanics; only validated acquisition/publication entries call this."""
    check_read_deadline()
    deadline = min(deadline, _DEADLINE.get() or float("inf"), monotonic() + 3)
    connection = connections[using]
    with transaction.atomic(using=using):
        with connection.cursor() as cursor:
            cursor.execute("SELECT name, setting::bigint FROM pg_settings "
                           "WHERE name IN ('lock_timeout', 'statement_timeout')")
            operator = dict(cursor.fetchall())
        # A shorter operator statement limit also bounds total operation work.
        if operator["statement_timeout"]:
            deadline = min(deadline, monotonic() + operator["statement_timeout"] / 1000)
        token = _DEADLINE.set(deadline)

        def bounded_execute(execute, sql, params, many, context):
            check_read_deadline()
            remaining = max(1, int((deadline - monotonic()) * 1000))
            limits = [min(remaining, operator[name]) if operator[name] else remaining
                      for name in ("lock_timeout", "statement_timeout")]
            with connection.connection.cursor() as control:
                control.execute("SELECT set_config('lock_timeout', %s, true), "
                                "set_config('statement_timeout', %s, true)",
                                [f"{limits[0]}ms", f"{limits[1]}ms"])
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
